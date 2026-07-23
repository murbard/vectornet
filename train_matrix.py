"""
Meta-train the matrix-neuron optimizer (vectornet_matrix.LearnedMatrixOptimizer) on
structured MLP tasks, evaluating head-to-head against tuned Muon — the hypothesis being
that a learnable O(a)xO(b)-equivariant rule containing Newton-Schulz in its span can
beat a fixed orthogonalization.

Run: .venv/bin/python train_matrix.py [--meta-steps 3000] [--save learned_matrix.pt]
"""
import argparse
import os
import random

import torch

from transformer_problem import TransformerLMProblem
from vectornet_matrix import LearnedMatrixOptimizer
from vectornet_torch import (MLPProblem, ReparamWrapper, TeacherStudentProblem,
                             load_fashion, load_mnist, meta_loss, run_baseline)


def train_pes_matrix(model, sample_problem, args, device, evaluate_hook=None,
                     probe_score=None):
    """PES for the matrix optimizer — BPTT meta-gradients through the matrix unit are
    pathologically large at init (three runs: gnorm 1e6-1e13, storm-skip freezes it);
    PES sidesteps them entirely: forward unrolls only, unbiased across truncations.
    Episode lengths are sampled per episode (short ones weight the warmup steps that a
    fixed long episode under-trains). Checkpoint selection via probe_score if given."""
    import random as _random
    from torch.nn.utils import parameters_to_vector, vector_to_parameters
    _rng = _random.Random(1)

    theta = torch.nn.Parameter(parameters_to_vector(model.parameters()).detach().clone())
    meta_opt = torch.optim.Adam([theta], lr=args.meta_lr)
    n_part = 2 * args.pes_pairs

    particles = None
    best_ema, ema, best_probe = float("inf"), None, float("inf")
    for step in range(args.meta_steps):
        if particles is None:
            problem = sample_problem()
            x0 = problem.init_point()
            # WARM-START CURRICULUM: for a fraction of episodes, pre-run the CURRENT model
            # (detached, no grad) for a random number of steps so the episode starts from
            # a near-converged state. This trains the rule on LATE-PHASE states inside
            # SHORT stable rollouts -- the only route to late-phase experience that avoids
            # both BPTT explosion and PES-long-episode variance.
            # skip warm-start on the largest problems (the pre-run's live forward can OOM
            # on d256 transformers); curriculum matters most on the medium/large ones anyway
            if args.curriculum and _rng.random() < 0.5:
                # WARM-START via a CHEAP baseline (Adam), not the learned optimizer:
                # a few Adam steps push x0 to a partially-converged state at negligible
                # memory (one backward, no recurrent state), so the episode trains the
                # rule on LATE-PHASE states inside a short stable rollout -- the route to
                # late-phase experience that avoids BPTT explosion, PES-episode variance,
                # AND the memory blow-up of a learned-optimizer warm-start.
                warm = _rng.choice([25, 50, 100, 200, 400])
                xw = x0.detach().clone().requires_grad_(True)
                wopt = torch.optim.Adam([xw], lr=_rng.choice([3e-3, 1e-2, 3e-2]))
                for _ in range(warm):
                    wopt.zero_grad()
                    problem.meta_objective(xw).sum().backward()
                    wopt.step()
                x0 = xw.detach()
            f0 = problem.meta_objective(x0).detach() + 1e-9
            particles = [{"x": x0.clone(), "H": None, "s": None,
                          "xi": torch.zeros_like(theta)} for _ in range(n_part)]
            steps_done = 0
            # v11 (long episodes on the UN-damped/UN-orthogonalized v10 arch) destabilized.
            # v13's momentum+spectral is stable on long horizons, so --long-episodes
            # RETRIES the long-horizon idea to teach late-phase descent (the ~0.15-nat
            # deficit vs Muon is late-phase quality, and time_inputs give the schedule
            # mechanism that only long episodes train).
            if args.long_episodes:
                # MODERATE (v15 lesson: 5000-step episodes destabilize PES; ES variance
                # grows with rollout). Cap at ~1600 steps, paired with more PES particles
                # + low meta-lr to control variance while adding late-phase experience.
                episode_len = _rng.choice([args.episode // 5, args.episode,
                                           int(args.episode * 2.5), args.episode * 4])
            else:
                episode_len = _rng.choice([args.episode // 10, args.episode // 5,
                                           args.episode, int(args.episode * 2.5)])

        half = [args.sigma * torch.randn_like(theta) for _ in range(args.pes_pairs)]
        eps = [e for pair in zip(half, [-e for e in half]) for e in pair]

        losses = torch.zeros(n_part, device=device)
        for k in range(n_part):
            vector_to_parameters(theta.detach() + eps[k], model.parameters())
            p = particles[k]
            if getattr(problem, "resample", False):
                # common random numbers: every particle replays the same batch sequence
                problem.draw_seed_base = 7919 * step
                problem.draw_counter = steps_done
            x = p["x"].detach().requires_grad_(True)
            if p["H"] is None:
                H, s = model.init_state(x.shape[0], problem.shapes, device)
            else:
                H, s = p["H"], p["s"]
            terms = []
            for j in range(args.segment):
                x, H, s, y = model.step(problem, x, H, s, create_graph=False,
                                        t=steps_done + j, budget=episode_len)
                x = x.detach().requires_grad_(True)
                H = [h.detach() for h in H]
                s = [t.detach() for t in s]
                terms.append(torch.log(y.detach() / f0 + 1e-9).mean())
            losses[k] = torch.stack(terms).mean()
            p["x"], p["H"], p["s"] = x.detach(), H, s
            p["xi"] = p["xi"] + eps[k]

        centered = losses - losses.mean()
        grad = torch.stack([particles[k]["xi"] * centered[k] for k in range(n_part)]
                           ).mean(dim=0) / (args.sigma ** 2)
        meta_opt.zero_grad()
        theta.grad = grad
        torch.nn.utils.clip_grad_norm_([theta], 1.0)
        meta_opt.step()

        if args.curriculum and device.type == "cuda" and step % 20 == 0:
            torch.cuda.empty_cache()  # periodic defrag; curriculum churns alloc sizes
        steps_done += args.segment
        if steps_done >= episode_len:
            particles = None

        loss_now = losses.mean().item()
        ema = loss_now if ema is None else 0.98 * ema + 0.02 * loss_now
        if probe_score is not None:
            if step % 100 == 0 and step > 50:
                vector_to_parameters(theta.detach(), model.parameters())
                s = probe_score()
                if s < best_probe:
                    best_probe = s
                    torch.save({"state_dict": model.state_dict(), "unit_kwargs": model.unit_kwargs}, args.save)
                    print(f"  probe-score {s:.4f} (new best, saved)", flush=True)
        elif ema < best_ema and step > 50:
            best_ema = ema
            vector_to_parameters(theta.detach(), model.parameters())
            torch.save({"state_dict": model.state_dict(), "unit_kwargs": model.unit_kwargs}, args.save)
        if step % 25 == 0:
            print(f"pes {step:5d}  seg-loss {loss_now:.4f}  ema {ema:.4f}", flush=True)
        if evaluate_hook and (step % args.eval_every == 0 or step == args.meta_steps - 1):
            vector_to_parameters(theta.detach(), model.parameters())
            evaluate_hook()

    vector_to_parameters(theta.detach(), model.parameters())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta-steps", type=int, default=3000)
    ap.add_argument("--unroll", type=int, default=20)
    ap.add_argument("--problems", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--meta-lr", type=float, default=1e-3)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--save", default="learned_matrix.pt")
    ap.add_argument("--trainer", choices=["bptt", "pes"], default="bptt")
    ap.add_argument("--time-inputs", action="store_true",
                    help="feed log(1+t) and t/T to the scalar net (horizon-aware schedules)")
    ap.add_argument("--cross-layer", action="store_true",
                    help="adjacent-layer gradient-covariance couplings (Shampoo/K-FAC span)")
    ap.add_argument("--fanin-gauge", action="store_true",
                    help="scale updates by sqrt(48/fan_in): muP-style width extrapolation")
    ap.add_argument("--momentum", action="store_true",
                    help="learned-decay output momentum buffer (damps the under-damped "
                         "oscillation that plateaus the rule at scale)")
    ap.add_argument("--spectral", action="store_true",
                    help="orthogonalize the momentum buffer before applying (Muon's "
                         "structure; requires --momentum). Controls update magnitude.")
    ap.add_argument("--blend", action="store_true",
                    help="learned per-layer mix of orthogonalized and raw update "
                         "(requires --momentum): orthogonalize big matrices, not small)")
    ap.add_argument("--long-episodes", action="store_true",
                    help="PES episodes up to ~5000 steps to teach late-phase descent "
                         "(viable now that momentum+spectral is stable on long horizons)")
    ap.add_argument("--ns-iters", type=int, default=5,
                    help="Newton-Schulz orthogonalization iterations (higher = cleaner "
                         "orthogonalization = better late-phase; eval test showed 5->8 helps)")
    ap.add_argument("--curriculum", action="store_true",
                    help="warm-start half the PES episodes from a near-converged state "
                         "(pre-run the model N steps) to train late-phase in short rollouts")
    ap.add_argument("--openml", action="store_true",
                    help="add the OpenML-CC18 train split (56 datasets) to the zoo")
    ap.add_argument("--init-from", default=None,
                    help="warm-start meta-parameters from a checkpoint (must match arch)")
    ap.add_argument("--big", action="store_true",
                    help="larger unit (k_hidden 6, k_mid 12, n_triples 6, n_dot 24, "
                         "scalar width 96) — for zoo-scale task diversity")
    ap.add_argument("--multitask", action="store_true",
                    help="meta-train across datasets (MNIST+Covertype) with reparam "
                         "augmentation; Fashion-MNIST and shapes h64/(32,32) held out")
    ap.add_argument("--pes-pairs", type=int, default=4)
    ap.add_argument("--sigma", type=float, default=0.01)
    ap.add_argument("--segment", type=int, default=20)
    ap.add_argument("--episode", type=int, default=400)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)
    torch.manual_seed(0)
    rng = random.Random(0)

    x_np, y_np, tx_np, ty_np = load_mnist(os.path.expanduser("~/.cache/mnist"))
    train_x = torch.from_numpy(x_np).to(device)
    train_y = torch.from_numpy(y_np).to(device)
    test_x = torch.from_numpy(tx_np).to(device)
    test_y = torch.from_numpy(ty_np).to(device)

    size_kwargs = ({"k_hidden": 6, "k_mid": 12, "n_triples": 6, "n_dot": 24,
                    "n_scal_hidden": 96} if args.big else {})
    model = LearnedMatrixOptimizer(time_inputs=args.time_inputs,
                                   cross_layer=args.cross_layer,
                                   fanin_gauge=args.fanin_gauge,
                                   momentum=args.momentum, spectral=args.spectral,
                                   blend=args.blend, ns_iters=args.ns_iters,
                                   **size_kwargs).to(device)
    if args.init_from:
        src = torch.load(args.init_from, map_location=device)["state_dict"]
        own = model.state_dict()
        loaded = {k: v for k, v in src.items()
                  if k in own and v.shape == own[k].shape}  # skip reshaped layers
        own.update(loaded)
        model.load_state_dict(own)
        skipped = [k for k in own if k not in loaded]
        print(f"warm-started from {args.init_from} "
              f"({len(loaded)}/{len(own)} tensors; skipped {skipped})", flush=True)
    print(f"matrix optimizer: {sum(p.numel() for p in model.parameters())} meta-params")
    meta_opt = torch.optim.Adam(model.parameters(), lr=args.meta_lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(meta_opt, T_max=args.meta_steps)

    if args.multitask:
        from datasets import load_text, openml_registry, registry
        datasets = registry(device)
        lm_ids, lm_vocab = load_text("shakespeare", device)  # text8 HELD OUT
        if args.openml:
            cpu = torch.device("cpu")
            oml_train, _ = openml_registry(cpu, verbose=False)
            datasets.update(oml_train)  # CPU-resident; MLPProblem moves minibatches
            print(f"openml: +{len(oml_train)} train datasets (13 held out by dataset)",
                  flush=True)

    def jitter_width():
        # log-uniform widths in [8, 96]: h64 becomes interpolation (v7 showed the
        # fashion-h64 loss was width extrapolation beyond the [8,48] support);
        # the width-transfer holdout moves to h128
        w = int(round(8 * (96 / 8) ** rng.random()))
        w = w if abs(w - 64) > 4 else 56  # keep a small exclusion window at h64
        depth = rng.choices([1, 2], weights=[3, 1])[0]
        return w if depth == 1 else (w, int(round(8 * (48 / 8) ** rng.random())))

    def sample_problem():
        # held out from meta-training: Fashion-MNIST + pendigits + text8 (data),
        # h64 and (32,32) (MLP shapes)
        hidden = jitter_width()
        act = rng.choice(["relu", "tanh"])
        # batch size varies the gradient-noise scale; deliberately NOT fed as an input —
        # the rule must infer noise from successive-gradient trace features
        B = rng.choice([16, 32, 64, 128, 256])
        if not args.multitask:
            return MLPProblem(train_x, train_y, hidden, args.problems, B, device,
                              activation=act)
        kind = rng.choices(["real", "projected", "teacher", "lm"],
                           weights=[7, 3, 2, 3])[0]
        # resampled (real-SGD) episodes half the time: stochastic objectives are the
        # regime of actual model training (the scale benchmark showed the fixed-batch
        # rule regresses under per-step resampling)
        res = rng.random() < 0.5
        if kind == "lm":
            # d up to 256: shrink the geometry gap to the scale benchmark (d=384).
            # memory: PES particles each hold k_hidden x params of state, so the
            # biggest configs run tiny (P=1, ctx<=64, L<=2, B<=8)
            d_lm = rng.choice([16, 32, 48, 96, 128, 256])
            big = d_lm > 128
            n_prob = 1 if big else (4 if d_lm > 48 else min(args.problems, 8))
            problem = TransformerLMProblem(
                lm_ids, lm_vocab, n_prob,
                rng.choice([4, 8] if big else [4, 8, 16, 32]), device,
                d_model=d_lm,
                n_layers=rng.randint(1, 2 if big else (4 if d_lm > 48 else 3)),
                n_heads=rng.choice([2, 4]),
                ctx=rng.choice([32, 64] if big else [32, 64, 128]),
                resample=res)
        elif kind == "teacher":
            problem = TeacherStudentProblem(
                rng.choice([16, 64, 256]), rng.randint(2, 10), hidden,
                args.problems, B, device, activation=act)
        else:
            dx, dy, in_dim, n_cls = datasets[rng.choice(list(datasets))]
            proj = rng.choice([32, 64, 128, 256]) if kind == "projected" else None
            problem = MLPProblem(dx, dy, hidden, args.problems, B,
                                 device, in_dim=in_dim, n_classes=n_cls,
                                 activation=act, project_to=proj, resample=res)
        if kind != "lm" and rng.random() < 0.5:
            problem = ReparamWrapper(problem, param_scale=10.0 ** rng.uniform(-1, 1),
                                     loss_scale=10.0 ** rng.uniform(-2, 2))
        return problem

    fash_x, fash_y = load_fashion(device)

    def run_evals():
        model.eval()
        for tag, data, hid, steps in [
                ("h16", (test_x, test_y), 16, args.unroll),
                ("h64-wider", (test_x, test_y), 64, args.unroll),
                ("h16-long", (test_x, test_y), 16, 5 * args.unroll),
                ("fashion-h16 HELD-OUT", (fash_x, fash_y), 16, 5 * args.unroll),
                ("fashion-h64 HELD-OUT", (fash_x, fash_y), 64, 5 * args.unroll)]:
            gen = torch.Generator(device=device).manual_seed(1234)
            problem = MLPProblem(*data, hid, 8, args.batch_size, device, gen)
            x0 = problem.init_point(gen)
            x, traj = model(x0.clone(), problem, steps, meta=False)
            best = None
            for lr in [0.003, 0.01, 0.03, 0.1]:
                xb, tb = run_baseline(problem, x0, "muon", lr, steps)
                if best is None or tb[:, -1].mean() < best[1][:, -1].mean():
                    best = (lr, tb, xb)
            print(f"  eval[{tag}] learned: loss {traj[:, -1].mean():.4f} "
                  f"acc {problem.accuracy(x).mean():.3f}  vs  "
                  f"muon(lr={best[0]}): loss {best[1][:, -1].mean():.4f} "
                  f"acc {problem.accuracy(best[2]).mean():.3f}", flush=True)
        model.train()

    if args.trainer == "pes":
        # selection probes: train-domain only (Fashion stays untouched/held out)
        gen = torch.Generator(device=device).manual_seed(999)
        probes = [MLPProblem(train_x, train_y, 16, 4, 64, device, gen)]
        if args.multitask:
            for name in ("covtype", "cifar10", "kmnist", "letter", "svhn"):
                if name in datasets:
                    dx, dy, din, dcls = datasets[name]
                    probes.append(MLPProblem(dx, dy, 16, 4, 64, device, gen,
                                             in_dim=din, n_classes=dcls))
            probes.append(TransformerLMProblem(lm_ids, lm_vocab, 4, 8, device, gen))
        probes = [(p, p.init_point(torch.Generator(device=device).manual_seed(999)))
                  for p in probes]

        def probe_score():
            model.eval()
            score = 0.0
            for p, x0 in probes:
                _, traj = model(x0.clone(), p, 20, meta=False)
                score += torch.log(traj[:, -1] / traj[:, 0] + 1e-9).mean().item()
            model.train()
            return score / len(probes)

        train_pes_matrix(model, sample_problem, args, device, run_evals, probe_score)
        print("done")
        return

    best_ema, ema = float("inf"), None
    for step in range(args.meta_steps):
        problem = sample_problem()
        x0 = problem.init_point()
        _, traj = model(x0, problem, args.unroll, meta=True)
        loss = meta_loss(traj)
        meta_opt.zero_grad()
        if torch.isfinite(loss):
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if torch.isfinite(gnorm) and gnorm < 1e4:  # storm-skip
                meta_opt.step()
        else:
            gnorm = float("nan")
        sched.step()

        ema = loss.item() if ema is None else 0.98 * ema + 0.02 * loss.item()
        if ema < best_ema and step > 50:
            best_ema = ema
            torch.save({"state_dict": model.state_dict(), "unit_kwargs": model.unit_kwargs}, args.save)
        if step % 25 == 0:
            print(f"meta {step:5d}  loss {loss.item():.4f}  ema {ema:.4f}  "
                  f"gnorm {gnorm:.2f}", flush=True)
        if step % args.eval_every == 0 or step == args.meta_steps - 1:
            run_evals()

    print("done")


if __name__ == "__main__":
    main()

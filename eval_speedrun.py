"""
Speedrun-style optimizer benchmark (the modded-nanogpt metric, at 3060 scale).

modded-nanogpt (Keller Jordan) measures STEPS/TIME to reach a target validation loss on
FineWeb GPT-2 124M with 8xH100 -- Muon is the record holder, and the target (3.28) is a
tight-convergence point. We can't run 124M on one 3060, and that tight target is exactly
the late-phase regime where the learned rule has a ceiling. So here we run the SAME
METRIC -- steps to reach val loss <= T -- at a size the 3060 handles (10.7M char-GPT on
text8 with a held-out val split), across a RANGE of targets T, for each optimizer.

The honest question this answers: for a given target perplexity, which optimizer gets
there in the fewest steps? Expectation: the learned rule wins LOOSE targets (fast early
descent) and tuned Muon wins TIGHT targets (better late-phase). We report the crossover.
"""
import argparse
import math

import torch

from datasets import load_text
from transformer_problem import TransformerLMProblem, rmsnorm  # noqa: F401
from vectornet_matrix import LearnedMatrixOptimizer
from vectornet_torch import muon_step


def build(device, d_model=384, n_layers=6, n_heads=6, ctx=128, batch=24, seed=0):
    ids, vocab = load_text("text8", device)
    n = len(ids)
    train_ids, val_ids = ids[: int(n * 0.9)], ids[int(n * 0.9):]  # held-out val split
    gen = torch.Generator(device=device).manual_seed(seed)
    train = TransformerLMProblem(train_ids, vocab, 1, batch, device, gen, d_model=d_model,
                                 n_layers=n_layers, n_heads=n_heads, ctx=ctx, resample=True)
    # fixed val problem on the held-out split (larger batch, deterministic)
    val = TransformerLMProblem(val_ids, vocab, 1, 128, device,
                               torch.Generator(device=device).manual_seed(12345),
                               d_model=d_model, n_layers=n_layers, n_heads=n_heads, ctx=ctx)
    return train, val


@torch.no_grad()
def val_loss(problem, x):
    return problem.objective(x).item()


def steps_to_targets(trace, targets):
    """trace: list of (step, val_loss). Returns {T: first step with loss<=T or None}."""
    out = {}
    for T in targets:
        hit = next((s for s, v in trace if v <= T), None)
        out[T] = hit
    return out


def run(opt_name, train, val, x0, model, lr_scale, lr, n_steps, eval_every=25):
    x = x0.clone().requires_grad_(True)
    trace = []
    if opt_name == "learned":
        H, s = model.init_state(1, train.shapes, x.device)
        for t in range(n_steps):
            x, H, s, _ = model.step(train, x, H, s, create_graph=False,
                                    lr_scale=lr_scale, t=t, budget=n_steps)
            x = x.detach().requires_grad_(True)
            H = [h.detach() for h in H]
            s = [q.detach() for q in s]
            if t % eval_every == 0:
                trace.append((t, val_loss(val, x)))
    elif opt_name == "muon":
        momenta = {}
        for t in range(n_steps):
            y = train.objective(x)
            (g,) = torch.autograd.grad(y.sum(), x)
            with torch.no_grad():
                x = muon_step(train, x, momenta, g, lr)
            x.requires_grad_(True)
            if t % eval_every == 0:
                trace.append((t, val_loss(val, x)))
    else:  # adamw + warmup-cosine
        opt = torch.optim.AdamW([x], lr=lr, weight_decay=0.01)
        warm = max(1, int(n_steps * 0.05))
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda t: min((t + 1) / warm, 1.0)
            * 0.5 * (1 + math.cos(math.pi * min(1.0, max(0.0, (t - warm) / max(1, n_steps - warm))))))
        for t in range(n_steps):
            opt.zero_grad()
            train.objective(x).sum().backward()
            opt.step()
            sched.step()
            if t % eval_every == 0:
                trace.append((t, val_loss(val, x)))
    trace.append((n_steps, val_loss(val, x)))
    return trace


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="learned_matrix13.pt")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--pilot", type=int, default=300)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)
    torch.manual_seed(0)

    ck = torch.load(args.checkpoint, map_location=device)
    model = LearnedMatrixOptimizer(**ck.get("unit_kwargs", {})).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()

    train, val = build(device)
    x0 = train.init_point(torch.Generator(device=device).manual_seed(1))
    print(f"optimizee: {train.n_params/1e6:.1f}M params  |  val = held-out 10% of text8")
    print(f"init val loss {val_loss(val, x0):.3f}\n")

    # pilot each baseline's lr / the learned rule's lambda on a short run (by final val)
    def pilot(name, grid, is_learned=False):
        best = None
        for v in grid:
            tr = run(name, train, val, x0, model,
                     lr_scale=(v if is_learned else 1.0),
                     lr=(v if not is_learned else 0), n_steps=args.pilot)
            f = tr[-1][1]
            if best is None or f < best[1]:
                best = (v, f)
        return best[0]

    lam = pilot("learned", [0.3, 0.5, 1.0], is_learned=True)
    mu = pilot("muon", [0.003, 0.01, 0.03])
    aw = pilot("adamw", [1e-3, 3e-3, 1e-2])
    print(f"pilots: learned lambda={lam}  muon lr={mu}  adamw lr={aw}\n")

    traces = {
        f"learned (lambda={lam})": run("learned", train, val, x0, model, lam, 0, args.steps),
        f"muon (lr={mu})": run("muon", train, val, x0, model, 1.0, mu, args.steps),
        f"adamw (lr={aw})": run("adamw", train, val, x0, model, 1.0, aw, args.steps),
    }

    finals = {k: t[-1][1] for k, t in traces.items()}
    best_final = min(finals.values())
    # targets spanning loose -> tight (relative to the best achieved)
    targets = [round(best_final + d, 2) for d in (0.8, 0.5, 0.3, 0.15, 0.05)]
    print(f"final val loss: " + "  ".join(f"{k.split()[0]} {v:.3f}" for k, v in finals.items()))
    print(f"\nSTEPS TO REACH TARGET VAL LOSS (fewer = faster; '--' = never in {args.steps}):")
    hdr = "target  " + "  ".join(f"{k:>22}" for k in traces)
    print(hdr)
    for T in targets:
        row = f"{T:5.2f}   "
        s2t = {k: steps_to_targets(tr, [T])[T] for k, tr in traces.items()}
        winner = min((s for s in s2t.values() if s is not None), default=None)
        for k in traces:
            s = s2t[k]
            mark = "*" if s is not None and s == winner else " "
            row += f"{(str(s) if s is not None else '--'):>21}{mark} "
        print(row)
    print("\n* = fastest to that target. Loose targets favor fast early descent; "
          "tight targets favor late-phase quality.")


if __name__ == "__main__":
    main()

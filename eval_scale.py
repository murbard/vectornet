"""
Scale-transfer benchmark: does the learned optimizer's ADVANTAGE survive a ~100x
parameter jump beyond its meta-training distribution?

Optimizee: char-GPT on text8 (held-out corpus), d_model 384, 6 layers, 6 heads,
ctx 128 (~10.7M params — meta-training saw <=48-dim, <=0.2M-param transformers).
Stochastic minibatches (resample=True), real training semantics.

Contenders: learned (zero-shot, and lr_scale grid on a short pilot), Muon (lr grid on
pilot), AdamW with linear-warmup + cosine decay (lr grid on pilot). Pilot = 300 steps;
winners rerun for the full budget. Reports eval loss on a fixed held-out batch every
100 steps, plus wall-clock per step.

Run: .venv/bin/python eval_scale.py [--steps 3000] [--checkpoint learned_matrixX.pt]
"""
import argparse
import math
import time

import torch

from datasets import load_text
from transformer_problem import TransformerLMProblem
from vectornet_matrix import LearnedMatrixOptimizer
from vectornet_torch import muon_step


def make_problem(device, steps_seed=0, batch=24, eval_batch=64):
    ids, vocab = load_text("text8", device)
    gen = torch.Generator(device=device).manual_seed(steps_seed)
    prob = TransformerLMProblem(ids, vocab, 1, batch, device, gen, d_model=384,
                                n_layers=6, n_heads=6, ctx=128, resample=True)
    eval_prob = TransformerLMProblem(ids, vocab, 1, eval_batch, device,
                                     torch.Generator(device=device).manual_seed(999),
                                     d_model=384, n_layers=6, n_heads=6, ctx=128)
    return prob, eval_prob


def run_learned(model, prob, eval_prob, x0, n_steps, lr_scale, eval_every=100):
    x = x0.clone().requires_grad_(True)
    H, s, so = model.init_state(1, prob.shapes, x0.device)
    trace, t0 = [], time.time()
    for t in range(n_steps):
        x, H, s, so, _ = model.step(prob, x, H, s, so, create_graph=False,
                                    lr_scale=lr_scale, t=t, budget=n_steps)
        x = x.detach().requires_grad_(True)
        H = [h.detach() for h in H]
        s = [q.detach() for q in s]
        if t % eval_every == 0 or t == n_steps - 1:
            with torch.no_grad():
                trace.append((t, eval_prob.objective(x).item()))
    return trace, (time.time() - t0) / n_steps


def run_muon(prob, eval_prob, x0, n_steps, lr, eval_every=100):
    x = x0.clone().requires_grad_(True)
    momenta, trace, t0 = {}, [], time.time()
    for t in range(n_steps):
        y = prob.objective(x)
        (g,) = torch.autograd.grad(y.sum(), x)
        with torch.no_grad():
            x = muon_step(prob, x, momenta, g, lr)
        x.requires_grad_(True)
        if t % eval_every == 0 or t == n_steps - 1:
            with torch.no_grad():
                trace.append((t, eval_prob.objective(x).item()))
    return trace, (time.time() - t0) / n_steps


def run_adamw(prob, eval_prob, x0, n_steps, lr, eval_every=100, warmup_frac=0.05):
    x = x0.clone().requires_grad_(True)
    opt = torch.optim.AdamW([x], lr=lr, weight_decay=0.01)
    warm = max(1, int(n_steps * warmup_frac))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda t: min((t + 1) / warm, 1.0)
        * 0.5 * (1 + math.cos(math.pi * min(1.0, max(0.0, (t - warm) / max(1, n_steps - warm))))))
    trace, t0 = [], time.time()
    for t in range(n_steps):
        opt.zero_grad()
        prob.objective(x).sum().backward()
        opt.step()
        sched.step()
        if t % eval_every == 0 or t == n_steps - 1:
            with torch.no_grad():
                trace.append((t, eval_prob.objective(x).item()))
    return trace, (time.time() - t0) / n_steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="learned_matrix7.pt")
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

    prob, eval_prob = make_problem(device)
    x0 = prob.init_point(torch.Generator(device=device).manual_seed(1))
    n_par = prob.n_params
    print(f"optimizee: {n_par / 1e6:.1f}M params (meta-training max was ~0.2M)")

    def pilot(fn, grid, tag):
        best = None
        for v in grid:
            trace, _ = fn(v, args.pilot)
            final = trace[-1][1]
            print(f"  pilot {tag}={v}: {final:.4f}", flush=True)
            if best is None or final < best[1]:
                best = (v, final)
            if device.type == "cuda":
                torch.cuda.empty_cache()  # each pilot allocates fresh optimizer state
        return best[0]

    lam = pilot(lambda v, n: run_learned(model, prob, eval_prob, x0, n, v),
                [0.5, 1.0, 2.0], "lambda")
    mu_lr = pilot(lambda v, n: run_muon(prob, eval_prob, x0, n, v),
                  [0.003, 0.01, 0.03], "muon-lr")
    aw_lr = pilot(lambda v, n: run_adamw(prob, eval_prob, x0, n, v),
                  [1e-3, 3e-3, 1e-2], "adamw-lr")

    print(f"\nfull runs ({args.steps} steps):", flush=True)
    for tag, fn in [("learned zero-shot", lambda n: run_learned(model, prob, eval_prob, x0, n, 1.0)),
                    (f"learned lambda={lam}", lambda n: run_learned(model, prob, eval_prob, x0, n, lam)),
                    (f"muon lr={mu_lr}", lambda n: run_muon(prob, eval_prob, x0, n, mu_lr)),
                    (f"adamw+cosine lr={aw_lr}", lambda n: run_adamw(prob, eval_prob, x0, n, aw_lr))]:
        trace, sps = fn(args.steps)
        pts = "  ".join(f"{t}:{v:.3f}" for t, v in trace[:: max(1, len(trace) // 8)])
        print(f"{tag:24s} final {trace[-1][1]:.4f}  ({sps * 1000:.0f} ms/step)  [{pts}]",
              flush=True)


if __name__ == "__main__":
    main()

"""
Decisive probe for the NEW architectural direction: does second-order (Shampoo)
preconditioning beat Muon on OUR exact 10.7M text8 GPT, especially late-phase?

If hand-coded Shampoo descends below tuned Muon (particularly in the late phase where the
learned rule has its ceiling), then a LEARNED optimizer that CONTAINS Shampoo -- the
two-sided preconditioner L^{-1/4} G R^{-1/4} with L=GG^T, R=G^T G (the natural
O(a)xO(b)-equivariant second moments) -- has real headroom to beat Muon. If Shampoo does
NOT beat Muon here, the second-order direction won't help and we need a different path.

This is the cheap go/no-go before building the full learned Shampoo unit.
"""
import argparse

import torch

from eval_scale import make_problem
from vectornet_torch import muon_step


def inv_root(M, power=-0.25):
    """M^power for symmetric PSD M via eigh in float64 (robust), with relative damping
    lambda = 1e-6 * mean-eigenvalue added before the root (classic Shampoo damping)."""
    Md = (0.5 * (M + M.transpose(-1, -2))).double()
    tr = torch.diagonal(Md).mean().clamp_min(1e-12)
    Md = Md + (1e-6 * tr) * torch.eye(Md.shape[-1], device=Md.device, dtype=Md.dtype)
    evals, evecs = torch.linalg.eigh(Md)
    evals = evals.clamp_min(1e-12 * tr)
    out = (evecs * evals.pow(power).unsqueeze(-2)) @ evecs.transpose(-1, -2)
    return out.to(M.dtype)


def run_shampoo(problem, val, x0, n_steps, lr, beta=0.95, precond_every=25,
                warmup=25, eval_every=25):
    """Shampoo with running-sum preconditioner statistics, damping, a Muon-style warmup
    (orthogonalized momentum until L,R accumulate), and GRAFTING (the Shampoo direction is
    rescaled to the RMS norm of the orthogonalized update, so lr transfers)."""
    from vectornet_torch import newton_schulz
    x = x0.clone().requires_grad_(True)
    shapes = problem.shapes
    L = [None] * len(shapes)
    R = [None] * len(shapes)
    Mom = [None] * len(shapes)
    Lr = [None] * len(shapes)
    Rr = [None] * len(shapes)
    trace = []
    has_b = getattr(problem, "has_biases", True)
    for t in range(n_steps):
        y = problem.objective(x)
        (g,) = torch.autograd.grad(y.sum(), x)
        with torch.no_grad():
            parts, i = [], 0
            for l, (a, b) in enumerate(shapes):
                G = g[:, i:i + a * b].view(a, b)
                Mom[l] = G if Mom[l] is None else beta * Mom[l] + (1 - beta) * G
                L[l] = G @ G.T if L[l] is None else L[l] + G @ G.T      # running SUM
                R[l] = G.T @ G if R[l] is None else R[l] + G.T @ G
                M = Mom[l]
                graft = newton_schulz(M.unsqueeze(0))[0]  # orthogonalized momentum (Muon)
                if t < warmup:
                    upd = graft
                else:
                    if t % precond_every == 0 or Lr[l] is None:
                        Lr[l] = inv_root(L[l])
                        Rr[l] = inv_root(R[l])
                    sh = Lr[l] @ M @ Rr[l]
                    # graft: match the Shampoo direction to the orthogonalized-update RMS
                    sh = sh / (sh.pow(2).mean().sqrt() + 1e-12) * graft.pow(2).mean().sqrt()
                    upd = sh
                parts.append((x[:, i:i + a * b].view(a, b) - lr * upd).reshape(1, a * b))
                i += a * b
                if has_b:
                    gb = g[:, i:i + b]
                    parts.append((x[:, i:i + b] - lr * gb).reshape(1, b))
                    i += b
            x = torch.cat(parts, dim=1).requires_grad_(True)
        if t % eval_every == 0:
            with torch.no_grad():
                trace.append((t, val.objective(x).item()))
    trace.append((n_steps, val.objective(x).item()))
    return trace


def run_muon(problem, val, x0, n_steps, lr, eval_every=25):
    x = x0.clone().requires_grad_(True)
    momenta = {}
    trace = []
    for t in range(n_steps):
        y = problem.objective(x)
        (g,) = torch.autograd.grad(y.sum(), x)
        with torch.no_grad():
            x = muon_step(problem, x, momenta, g, lr)
        x.requires_grad_(True)
        if t % eval_every == 0:
            with torch.no_grad():
                trace.append((t, val.objective(x).item()))
    trace.append((n_steps, val.objective(x).item()))
    return trace


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)
    torch.manual_seed(0)

    train, val = make_problem(device, batch=24)
    x0 = train.init_point(torch.Generator(device=device).manual_seed(1))
    print(f"{train.n_params/1e6:.1f}M GPT, {args.steps} steps. init {val.objective(x0).item():.3f}\n")

    results = {}
    for lr in [0.003, 0.01, 0.03]:
        tr = run_muon(train, val, x0, min(args.steps, 400), lr)
        results[("muon", lr)] = tr[-1][1]
    mu = min([lr for (n, lr) in results if n == "muon"], key=lambda lr: results[("muon", lr)])
    for lr in [0.03, 0.1, 0.3]:
        tr = run_shampoo(train, val, x0, min(args.steps, 400), lr)
        results[("shampoo", lr)] = tr[-1][1]
    sh = min([lr for (n, lr) in results if n == "shampoo"], key=lambda lr: results[("shampoo", lr)])
    print(f"pilot(400): muon lr={mu} ({results[('muon',mu)]:.3f})  "
          f"shampoo lr={sh} ({results[('shampoo',sh)]:.3f})\n")

    tm = run_muon(train, val, x0, args.steps, mu)
    ts = run_shampoo(train, val, x0, args.steps, sh)
    print(f"{'step':>6} {'muon':>8} {'shampoo':>8}")
    for (s, vm), (_, vs) in zip(tm[::2], ts[::2]):
        lead = "shampoo" if vs < vm else "muon"
        print(f"{s:6d} {vm:8.3f} {vs:8.3f}   <- {lead}")
    print(f"\nFINAL: muon {tm[-1][1]:.3f}  shampoo {ts[-1][1]:.3f}")
    print("VERDICT:", "Shampoo BEATS Muon -> build learned 2nd-order unit"
          if ts[-1][1] < tm[-1][1] - 0.02 else
          "Shampoo does NOT clearly beat Muon here -> reconsider direction")


if __name__ == "__main__":
    main()

"""
Zero-shot transfer eval for the MATRIX optimizer on held-out datasets and shapes.

Rows per task:
  learned zero-shot      — frozen weights, NO hyperparameters (lr_scale = 1)
  learned tuned-lambda   — ONE tuned scalar multiplying the update (grid), the
                           apples-to-apples counterpart of lr-tuning the baselines
  muon / adam / sgd      — per-task lr-tuned on a grid (the tuned regime)
  prodigy (lr=1)         — self-tuning baseline (the untuned regime)

Held out of meta-training: fashion-mnist, pendigits, text8 (nanoGPT transfer),
h64 and (32,32) MLP shapes.
"""
import argparse
import os

import torch

from datasets import load_pendigits, load_text
from transformer_problem import TransformerLMProblem
from vectornet_matrix import LearnedMatrixOptimizer
from vectornet_torch import (CACHE, MLPProblem, load_covtype, load_fashion, load_mnist,
                             run_baseline)

LAMBDAS = [0.25, 0.5, 1.0, 2.0, 4.0]


def eval_task(model, problem, x0, n_steps, baselines):
    x, traj = model(x0.clone(), problem, n_steps, meta=False)
    print(f"    {'learned zero-shot':32s} loss {traj[:, -1].mean():8.4f}  "
          f"acc {problem.accuracy(x).mean():.3f}")
    best = None
    for lam in LAMBDAS:
        xl, tl = model(x0.clone(), problem, n_steps, meta=False, lr_scale=lam)
        if best is None or tl[:, -1].mean() < best[2][:, -1].mean():
            best = (lam, xl, tl)
    print(f"    {'learned tuned-lambda(%.2f)' % best[0]:32s} loss "
          f"{best[2][:, -1].mean():8.4f}  acc {problem.accuracy(best[1]).mean():.3f}")
    for name, lrs in baselines:
        b = None
        for lr in lrs:
            xb, tb = run_baseline(problem, x0, name, lr, n_steps)
            if b is None or tb[:, -1].mean() < b[2][:, -1].mean():
                b = (f"{name} (lr={lr})", xb, tb)
        print(f"    {b[0]:32s} loss {b[2][:, -1].mean():8.4f}  "
              f"acc {problem.accuracy(b[1]).mean():.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="learned_matrix.pt")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)
    torch.manual_seed(0)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model = LearnedMatrixOptimizer(**ckpt.get("unit_kwargs", {})).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    mx, my, mtx, mty = load_mnist(os.path.join(CACHE, "mnist"))
    mnist = (torch.from_numpy(mtx).to(device), torch.from_numpy(mty).to(device), 784, 10)
    fashion = (*load_fashion(device), 784, 10)
    cov = (*load_covtype(device), 54, 7)
    pen = load_pendigits(device)
    t8_ids, t8_vocab = load_text("text8", device)

    baselines = [("muon", [0.003, 0.01, 0.03, 0.1]),
                 ("adam", [1e-3, 3e-3, 1e-2, 3e-2, 1e-1]),
                 ("sgd", [0.03, 0.1, 0.3, 1.0, 3.0]),
                 ("prodigy", [1.0])]

    tasks = [
        ("mnist h16 (train domain)", mnist, 16),
        ("mnist h64 (unseen width)", mnist, 64),
        ("mnist (32,32) (unseen shape)", mnist, (32, 32)),
        ("fashion h16 (HELD-OUT dataset)", fashion, 16),
        ("fashion h64 (held-out data+width)", fashion, 64),
        ("covertype h16 (train domain)", cov, 16),
        ("pendigits h16 (HELD-OUT dataset)", pen, 16),
        ("pendigits h64 (held-out data+width)", pen, 64),
    ]
    for n_steps in [20, 100]:
        print(f"\n=== budget: {n_steps} steps ===")
        for name, (dx, dy, in_dim, n_cls), hidden in tasks:
            gen = torch.Generator(device=device).manual_seed(1234)
            problem = MLPProblem(dx, dy, hidden, 8, 128, device, gen,
                                 in_dim=in_dim, n_classes=n_cls)
            x0 = problem.init_point(gen)
            print(f"  {name}")
            eval_task(model, problem, x0, n_steps, baselines)
        print("  nanogpt d32 L2 on text8 (HELD-OUT corpus; d48 unseen at train? see cfg)")
        gen = torch.Generator(device=device).manual_seed(1234)
        problem = TransformerLMProblem(t8_ids, t8_vocab, 4, 16, device, gen,
                                       d_model=32, n_layers=2, n_heads=2, ctx=64)
        x0 = problem.init_point(gen)
        eval_task(model, problem, x0, n_steps, baselines)


if __name__ == "__main__":
    main()

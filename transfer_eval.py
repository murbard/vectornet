"""
Zero-shot transfer evaluation of the MNIST-meta-trained optimizer.

The learned optimizer's weights are independent of problem dimension AND were
meta-trained on exactly one task family (20 steps of training a 784-16-10 MLP
on MNIST minibatches). Here it is applied, frozen, to:

  fashion   : 784-16-10 MLP on Fashion-MNIST     (near transfer)
  covertype : 54-16-7 MLP on UCI Covertype        (tabular, n=1,000 vs 12,906)
  quartic   : the non-convex quartic from the 2015 vectornet.py, n=1,000
              (not a neural network; negative losses; different curvature)

Baselines: SGD and Adam with learning rate tuned per task/budget on the grid.
"""
import argparse
import os

import torch

from vectornet_torch import (CACHE, LearnedOptimizer, MLPProblem, QuarticProblem,
                             load_covtype, load_fashion, load_mnist, run_baseline)


def eval_task(model, make_problem, n_steps, device, seed, baselines, is_gap=False):
    gen = torch.Generator(device=device).manual_seed(seed)
    problem = make_problem(gen)
    x0 = problem.init_point(gen)

    rows = []
    x, traj = model(x0.clone(), problem.objective, n_steps, meta=False)
    rows.append(("learned (frozen, MNIST-trained)", problem, x, traj))
    for name, lrs in baselines:
        best = None
        for lr in lrs:
            xb, tb = run_baseline(problem, x0, name, lr, n_steps)
            if best is None or tb[:, -1].mean() < best[3][:, -1].mean():
                best = (f"{name} (best lr={lr})", problem, xb, tb)
        rows.append(best)

    for label, prob, xf, tr in rows:
        if is_gap:
            print(f"    {label:35s} final gap/coord {prob.gap(xf):10.4f}")
        else:
            acc = prob.accuracy(xf).mean().item()
            print(f"    {label:35s} final loss {tr[:, -1].mean().item():8.4f}  acc {acc:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="learned_opt.pt")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    device = torch.device(args.device)
    torch.manual_seed(0)
    ckpt = torch.load(args.checkpoint, map_location=device)
    if "state_dict" in ckpt:
        model = LearnedOptimizer(**ckpt["unit_kwargs"]).to(device)
        model.load_state_dict(ckpt["state_dict"])
    else:  # legacy raw state_dict checkpoint
        model = LearnedOptimizer().to(device)
        model.load_state_dict(ckpt)
    model.eval()

    print("loading data...")
    mnist_x, mnist_y, _, _ = load_mnist(os.path.join(CACHE, "mnist"))
    mnist_x, mnist_y = torch.from_numpy(mnist_x).to(device), torch.from_numpy(mnist_y).to(device)
    fash_x, fash_y = load_fashion(device)
    cov_x, cov_y = load_covtype(device)

    mlp_baselines = [("sgd", [0.03, 0.1, 0.3, 1.0, 3.0]),
                     ("adam", [1e-3, 3e-3, 1e-2, 3e-2, 1e-1]),
                     ("muon", [0.003, 0.01, 0.03, 0.1]),
                     ("lbfgs", [0.3, 1.0])]
    # no muon on the quartic: it has no matrix structure to orthogonalize
    quartic_baselines = [("sgd", [1e-4, 1e-3, 3e-3, 1e-2]),
                         ("adam", [3e-3, 1e-2, 3e-2, 1e-1, 3e-1]),
                         ("lbfgs", [0.3, 1.0])]

    tasks = {
        "mnist 784-16-10 (meta-training domain)": lambda g: MLPProblem(
            mnist_x, mnist_y, 16, 8, 128, device, g),
        "fashion-mnist 784-16-10": lambda g: MLPProblem(
            fash_x, fash_y, 16, 8, 128, device, g),
        "covertype 54-16-7 (n=1,000 params)": lambda g: MLPProblem(
            cov_x, cov_y, 16, 8, 128, device, g, in_dim=54, n_classes=7),
    }
    for n_steps in [20, 100]:
        print(f"\n=== budget: {n_steps} optimization steps ===")
        for name, make in tasks.items():
            print(f"  {name}")
            eval_task(model, make, n_steps, device, 1234, mlp_baselines)
        print("  quartic n=1,000 (2015 vectornet objective, gap above global min per coord)")
        eval_task(model, lambda g: QuarticProblem(1000, 8, device, g), n_steps, device,
                  1234, quartic_baselines, is_gap=True)


if __name__ == "__main__":
    main()

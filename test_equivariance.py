"""
Property tests for the framework's mathematical invariants (CPU-only, random weights).

An optimizer built from {linear combinations of vectors, dot products, scalar nets}
should commute with symmetries of the problem:

  1. Permutation equivariance:  pi(opt(f)) == opt(f o pi^-1)   — must hold always.
  2. Orthogonal equivariance:   Q(opt(f)) == opt(f o Q^T)      — must hold with
     vector_activation='linear'; the elementwise tanh deliberately breaks it.
  3. Replication consistency (intensive-units convention): duplicating every
     coordinate of a separable problem should duplicate the trajectory. Measures
     how dimension-invariant the algorithm really is.

Run: .venv/bin/python test_equivariance.py
"""
import math

import torch

from vectornet_torch import LearnedOptimizer, QuarticProblem

torch.manual_seed(7)
torch.set_default_dtype(torch.float64)  # property tests need headroom over accumulation error
DEV = torch.device("cpu")
N, P, STEPS = 12, 3, 6


def trajectory(model, x0, objective, steps=STEPS):
    xs = []
    x = x0.clone().requires_grad_(True)
    V, S = model.init_state(x0)
    for _ in range(steps):
        x, V, S, _ = model.unit(x, V, S, objective, create_graph=False)
        x = x.detach().requires_grad_(True)
        V, S = V.detach(), S.detach()
        xs.append(x.detach().clone())
    return xs


def max_dev(traj_a, traj_b, transform):
    return max((transform(a) - b).abs().max().item() for a, b in zip(traj_a, traj_b))


def make_quartic(offset):
    def f(x):
        z = x - offset
        return (z.pow(4) - 16.0 * z.pow(2) + 5.0 * z).sum(dim=1) / 2.0
    return f


def report(name, dev, should_hold):
    ok = (dev < 1e-4) == should_hold
    expect = "invariant" if should_hold else "broken (by design)"
    print(f"  {'PASS' if ok else 'FAIL'}  {name:55s} max deviation {dev:.3e}  [{expect}]")
    return ok


def main():
    offset = 2.0 * torch.randn(P, N)
    x0 = 2.0 * torch.randn(P, N)
    perm = torch.randperm(N)
    Q, _ = torch.linalg.qr(torch.randn(N, N))

    all_ok = True
    for rms, intensive in [(False, False), (True, False), (True, True)]:
        for act in ["tanh", "linear"]:
            torch.manual_seed(123)  # same random meta-weights for every variant
            model = LearnedOptimizer(vector_activation=act, rms_convention=rms,
                                     intensive_inputs=intensive).to(DEV)
            tag = f"rms={rms} intensive={intensive} act={act}"

            base = trajectory(model, x0, make_quartic(offset))

            t_perm = trajectory(model, x0[:, perm], make_quartic(offset[:, perm]))
            all_ok &= report(f"permutation equivariance   ({tag})",
                             max_dev(base, t_perm, lambda a: a[:, perm]), True)

            t_rot = trajectory(model, x0 @ Q, lambda x: make_quartic(offset)(x @ Q.T))
            all_ok &= report(f"orthogonal equivariance    ({tag})",
                             max_dev(base, t_rot, lambda a: a @ Q), act == "linear")

            # Exact invariant only for strict-intensive inputs: otherwise signed_log(y)
            # (extensive for this objective) and log n change under duplication.
            t_dup = trajectory(model, torch.cat([x0, x0], 1),
                               make_quartic(torch.cat([offset, offset], 1)))
            dev = max_dev(base, t_dup, lambda a: torch.cat([a, a], 1))
            if intensive:
                all_ok &= report(f"replication invariance     ({tag})", dev, True)
            else:
                print(f"  INFO  replication deviation      ({tag}) {dev:.3e}"
                      f"  [smaller = more dimension-invariant]")

    print("\nall properties as expected" if all_ok else "\nUNEXPECTED RESULTS — investigate")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

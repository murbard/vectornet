"""
Diagnose WHY the learned optimizer plateaus at scale (v10: descends to ~2.42 by step
1000 then stalls, while Muon keeps going). Hypothesis: state collapse -- the recurrent
hidden matrices converge to a fixed point, so the update direction stops evolving.

Runs the learned optimizer on the 10.7M GPT for N steps and logs, per step:
  - loss
  - state_change:  mean ||H_t - H_{t-1}|| / ||H_t||   (0 => hidden state frozen)
  - scal_change:   mean |s_t - s_{t-1}|                 (scalar state frozen?)
  - log_step:      mean of the emitted per-layer log step size (collapsing to -inf?)
  - upd_norm:      rms of the actual applied update       (updates -> 0?)
  - upd_cos_prev:  cosine(update_t, update_{t-1})         (direction stuck?)

If state_change and upd_norm collapse toward 0 around the plateau step, the cause is
state collapse (architectural). If they stay healthy but loss stalls, the cause is
that the update direction, though changing, stops being a descent direction (a
capacity/preconditioning gap vs Muon).
"""
import argparse

import torch

from eval_scale import make_problem
from vectornet_matrix import LearnedMatrixOptimizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="learned_matrix10.pt")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lr-scale", type=float, default=0.5)
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

    prob, eval_prob = make_problem(device, batch=24)
    x = prob.init_point(torch.Generator(device=device).manual_seed(1)).requires_grad_(True)
    H, s, so = model.init_state(1, prob.shapes, device)

    def flat(state):
        return torch.cat([t.reshape(-1) for t in state])

    prev_H, prev_s, prev_upd, prev_x = None, None, None, x.detach().clone()
    print(f"{'step':>5} {'loss':>8} {'dH/H':>8} {'ds':>8} {'logstep':>8} "
          f"{'updrms':>9} {'cos_prev':>8}", flush=True)
    for t in range(args.steps):
        x, H, s, so, y = model.step(prob, x, H, s, so, create_graph=False,
                                    lr_scale=args.lr_scale, t=t, budget=args.steps)
        x = x.detach().requires_grad_(True)
        H = [h.detach() for h in H]
        s = [q.detach() for q in s]
        upd = (x.detach() - prev_x)
        Hf, sf = flat(H), flat(s)
        if prev_H is not None:
            dH = (Hf - prev_H).norm().item() / (Hf.norm().item() + 1e-9)
            ds = (sf - prev_s).abs().mean().item()
            updrms = upd.pow(2).mean().sqrt().item()
            cos = (torch.dot(upd.reshape(-1), prev_upd.reshape(-1))
                   / (upd.norm() * prev_upd.norm() + 1e-12)).item()
            if t % 50 == 0 or t < 10:
                with torch.no_grad():
                    ev = eval_prob.objective(x).item()
                print(f"{t:5d} {ev:8.4f} {dH:8.4f} {ds:8.4f} "
                      f"{0.0:8.4f} {updrms:9.5f} {cos:8.4f}", flush=True)
        prev_H, prev_s, prev_upd, prev_x = Hf.clone(), sf.clone(), upd.clone(), x.detach().clone()


if __name__ == "__main__":
    main()

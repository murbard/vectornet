"""
Matrix-valued neurons: the vectornet design one rung up the invariant-theory ladder.

The 2015 framework: scalar neurons + vector neurons of symbolic dimension n, vectors
touch scalars only through dot products. Its invariance class is O(n) — which provably
cannot express Muon or Shampoo, because they exploit the *matrix* structure of each
layer's parameters.

This module adds neurons that are matrices of symbolic shape (a_l, b_l), one family
per optimizee layer, with ALL meta-weights shared across layers (the same trick that
makes vector neurons dimension-polymorphic makes matrix neurons shape-polymorphic).
The primitive operations, exactly the O(a) x O(b)-equivariant analogues of
{linear combination, dot product}:

    - linear combinations of stored matrices (coefficients from the scalar net)
    - the triple product  T(X, Y, Z) = X Y^T Z   — the natural equivariant cubic;
      Newton-Schulz is a short program in it:  X X^T X = T(X,X,X), etc.
    - normalized trace inner products  <A, B> = tr(A^T B) / (a*b)  -> scalars (intensive)

Muon is a 3-line special case; Shampoo-like preconditioning lives in the same span.
Biases and other 1-D parameters keep the original vector unit. Layers communicate
through a pooled scalar bus (mean over layers — DeepSets over the layer set), so the
architecture is polymorphic in depth as well as in every layer shape.

Honest limitation, recorded for later: the true symmetry of MLP training couples
adjacent layers (W_l P, P^T W_{l+1} for a hidden-unit permutation P); treating layers
independently is equivariant to a LARGER group than the truth, hence a strict
restriction of what can be expressed (e.g. no W_{l+1} W_l products). Cross-layer
triple products are the principled next extension.
"""
import math

import torch
import torch.nn as nn

from vectornet_torch import ScalarMLP, signed_log


def triple(X, Y, Z):
    """T(X,Y,Z) = X Y^T Z for batched (P, a, b) stacks."""
    return X @ Y.transpose(-1, -2) @ Z


class MatrixUnit(nn.Module):
    """
    One step of the recurrent optimizer for ONE layer's matrix block, batched over
    P problems; the same module (same weights) is applied to every layer.

    Per-layer state: H (P, k, a, b) hidden matrices, s (P, n_scal_local) local scalars.
    A global scalar bus (mean-pooled over layers) enters every layer's scalar net.
    """

    def __init__(self, k_hidden=4, k_mid=8, n_triples=4, n_scal_local=16,
                 n_scal_global=16, n_dot=16, n_scal_hidden=64, time_inputs=False,
                 cross_layer=False, fanin_gauge=False, momentum=False):
        super().__init__()
        self.k_hidden, self.k_mid, self.n_triples = k_hidden, k_mid, n_triples
        self.n_scal_local, self.n_scal_global = n_scal_local, n_scal_global
        # time_inputs: feed log(1+t) and the budget fraction t/T as scalars, making
        # horizon-aware schedules (warmup/cosine-like decay) exactly representable;
        # without them the rule can only realize autonomous (state-clock) schedules
        self.time_inputs = time_inputs
        # cross_layer: two extra input matrices per layer — the layer's gradient
        # projected through the adjacent layers' gradient covariances (the equivariant
        # couplings under the TRUE MLP symmetry W_l Q, Q^T W_{l+1}; the Shampoo/K-FAC
        # family lives in this span)
        self.cross_layer = cross_layer
        # fanin_gauge: multiply each layer's update by sqrt(48/fan_in) — the muP-style
        # width-scaling law, gauged to ~1 on the trained geometry. The learned step
        # then encodes deviation from a scale-correct baseline, so step-size
        # calibration extrapolates to unseen widths by construction (measured at
        # d=384: zero-shot needed exactly ~sqrt(48/384) correction).
        self.fanin_gauge = fanin_gauge
        # momentum: emit a per-layer decay gamma in (0,1) so the optimizer can keep an
        # explicit output momentum buffer M = gamma*M + (1-gamma)*delta and apply M.
        # The rms-normalized hidden matrices are direction-only and cannot damp; the
        # scale diagnostic showed the plateau is UNDER-DAMPED oscillation (cos of
        # successive updates persistently NEGATIVE), which a learned momentum fixes.
        self.momentum = momentum

        k_in = k_hidden + 1 + (2 if cross_layer else 0)
        k_pool = k_mid + n_triples               # mid stack after the quadratic stage

        # scalar features in: local + global scalars + intensive invariants of the input
        # stack (learned trace-product pairs) + log f, log rms(G), log a, log b
        # (+ log(1+t), t/T when time_inputs)
        n_feat = n_scal_local + n_scal_global + n_dot + 4 + (2 if time_inputs else 0)
        self.trace_w1 = nn.Parameter(torch.randn(k_in, n_dot) / math.sqrt(k_in))
        self.trace_w2 = nn.Parameter(torch.randn(k_in, n_dot) / math.sqrt(k_in))

        # net A: features -> combination coefficients for the linear stage
        self.net_a = ScalarMLP([n_feat, n_scal_hidden, k_in * k_mid])
        self.comb_bias = nn.Parameter(0.01 * torch.randn(k_in, k_mid))

        # quadratic stage: n_triples learned triple products T(Xi, Xj, Xl) where
        # Xi, Xj, Xl are fixed learned combinations of the mid stack (normalized first,
        # so the polynomial degree stays tame across recurrent steps)
        self.tri_w = nn.Parameter(torch.randn(3, k_mid, n_triples) / math.sqrt(k_mid))

        # net B: features + post-triple invariants -> new local scalars + log step
        self.trace_w3 = nn.Parameter(torch.randn(k_pool, n_dot) / math.sqrt(k_pool))
        self.trace_w4 = nn.Parameter(torch.randn(k_pool, n_dot) / math.sqrt(k_pool))
        # net_b outputs: new local scalars + log_step (+ gamma logit when momentum)
        self.net_b = ScalarMLP([n_feat + n_dot, n_scal_hidden,
                                n_scal_local + 1 + (1 if momentum else 0)])
        with torch.no_grad():  # cold-start the step size: exp(-3) ~ 0.05
            self.net_b.layers[-1].bias[n_scal_local] = -3.0
            if momentum:  # cold-start gamma ~ sigmoid(2) = 0.88 (Muon-like heavy damping)
                self.net_b.layers[-1].bias[-1] = 2.0

        # output: delta + new hidden matrices, small init => near-stationary start
        self.out_comb = nn.Parameter(0.1 * torch.randn(k_pool, k_hidden + 1) / math.sqrt(k_pool))

    @staticmethod
    def _traces(stack_a, stack_b, w1, w2):
        """Intensive trace inner products of learned combinations: (P, n_dot)."""
        left = torch.einsum("pkab,kd->pdab", stack_a, w1)
        right = torch.einsum("pkab,kd->pdab", stack_b, w2)
        return signed_log((left * right).mean(dim=(2, 3)))

    def forward(self, G, H, s_local, s_global, y, t=0, budget=None, cross=None):
        P, a, b = G.shape
        if self.momentum:  # last H channel is the momentum buffer M, not a direction slot
            M_prev = H[:, -1]
            H = H[:, :-1]
        g_rms = (G.pow(2).mean(dim=(1, 2), keepdim=True) + 1e-24).sqrt()
        G_hat = G / g_rms
        parts = [H, G_hat.unsqueeze(1)]
        if self.cross_layer:
            if cross is None:
                cross = torch.zeros(P, 2, a, b, device=G.device, dtype=G.dtype)
            parts.append(cross)
        stack = torch.cat(parts, dim=1)                             # (P, k_in, a, b)

        consts = [signed_log(y).unsqueeze(1),
                  signed_log(g_rms.view(P, 1)),
                  torch.full((P, 1), math.log(a) / 10.0, device=G.device),
                  torch.full((P, 1), math.log(b) / 10.0, device=G.device)]
        if self.time_inputs:
            tau = (t / budget) if budget else -1.0
            consts += [torch.full((P, 1), math.log1p(t) / 10.0, device=G.device),
                       torch.full((P, 1), float(tau), device=G.device)]
        consts = torch.cat(consts, dim=1)
        feats = torch.cat([s_local, s_global,
                           self._traces(stack, stack, self.trace_w1, self.trace_w2),
                           consts], dim=1)

        coeff = self.net_a(feats).view(P, -1, self.k_mid) + self.comb_bias
        mid = torch.einsum("pkab,pkm->pmab", stack, coeff)          # linear stage

        # normalize the three triple-product operands so recurrence stays bounded
        X, Y, Z = (torch.einsum("pmab,mt->ptab", mid, self.tri_w[i]) for i in range(3))
        norm = lambda M: M / (M.pow(2).mean(dim=(2, 3), keepdim=True) + 1e-24).sqrt()
        # intensive normalization: rms-1 operands give T entries of scale sqrt(a*b)
        tri = triple(norm(X), norm(Y), norm(Z)) / math.sqrt(a * b)  # (P, n_triples, a, b)

        pool = torch.cat([mid, tri], dim=1)                         # (P, k_pool, a, b)
        s_out = self.net_b(torch.cat(
            [feats, self._traces(pool, pool, self.trace_w3, self.trace_w4)], dim=1))
        if self.momentum:
            s_local_new = torch.tanh(s_out[:, :-2])
            log_step = s_out[:, -2:-1].clamp(-8.0, 2.0)
            gamma = torch.sigmoid(s_out[:, -1:])  # (P,1) learned momentum decay
        else:
            s_local_new = torch.tanh(s_out[:, :-1])
            log_step = s_out[:, -1:].clamp(-8.0, 2.0)
            gamma = None

        out = torch.einsum("pmab,mo->poab", pool, self.out_comb)
        raw = out[:, 0] * log_step.exp().unsqueeze(-1)
        # hidden matrices store directions only (rms-normalized on write): an unbounded
        # linear recurrence explodes over long horizons; magnitudes belong to the
        # tanh-bounded scalar memory
        H_new = out[:, 1:]
        H_new = H_new / (H_new.pow(2).mean(dim=(2, 3), keepdim=True) + 1e-24).sqrt()

        if self.momentum:
            # damp oscillation with a learned-decay output momentum buffer, then apply it
            M_new = gamma.unsqueeze(-1) * M_prev + (1.0 - gamma.unsqueeze(-1)) * raw
            delta = M_new
            H_new = torch.cat([H_new, M_new.unsqueeze(1)], dim=1)
        else:
            delta = raw
        return delta, H_new, s_local_new, log_step


class LearnedMatrixOptimizer(nn.Module):
    """
    Full optimizer over a layered optimizee: one shared MatrixUnit for all matrix
    blocks, a shared global-pool projection tying layers together. Bias blocks are
    folded into their layer's scalar features via their rms (a fuller treatment
    would reuse the vector Unit; kept minimal for the first experiment).

    Works on the SAME flat parameter vector as the vector framework — `shapes`
    tells it where the matrices live, which is precisely the extra structure
    (and the only extra structure) that Muon needs too.
    """

    def __init__(self, **unit_kwargs):
        super().__init__()
        self.unit_kwargs = unit_kwargs
        self.unit = MatrixUnit(**unit_kwargs)
        n_local = self.unit.n_scal_local
        self.pool_proj = nn.Linear(n_local, self.unit.n_scal_global)

    def init_state(self, P, shapes, device):
        # When momentum is on, each H tensor carries one EXTRA channel: the last slot is
        # the magnitude-carrying momentum buffer M (not rms-normalized), the first
        # k_hidden are the usual direction memories. Bundling M into H keeps the state
        # signature (H, s) unchanged, so every caller threads momentum for free.
        kh = self.unit.k_hidden + (1 if self.unit.momentum else 0)
        H = [torch.zeros(P, kh, aa, bb, device=device) for aa, bb in shapes]
        s = [torch.zeros(P, self.unit.n_scal_local, device=device) for _ in shapes]
        return H, s

    def step(self, problem, x, H, s, create_graph=True, lr_scale=1.0, t=0, budget=None):
        """lr_scale: optional single tunable scalar multiplying every update — the
        one-hyperparameter variant for apples-to-apples comparison with lr-tuned
        baselines. 1.0 = fully zero-shot. t/budget: step index and declared horizon
        (used only when the unit was built with time_inputs)."""
        y = problem.meta_objective(x)
        (g,) = torch.autograd.grad(y.sum(), x, create_graph=create_graph)
        s_global = self.pool_proj(torch.stack(s, dim=0).mean(dim=0))

        has_biases = getattr(problem, "has_biases", True)

        # first pass: normalized gradient matrix per layer (for cross-layer features)
        Gs, i = [], 0
        for aa, bb in problem.shapes:
            Gm = g[:, i:i + aa * bb].view(-1, aa, bb)
            Gs.append(Gm / (Gm.pow(2).mean(dim=(1, 2), keepdim=True) + 1e-24).sqrt())
            i += aa * bb + (bb if has_biases else 0)

        def rmsn(M):
            return M / (M.pow(2).mean(dim=(1, 2), keepdim=True) + 1e-24).sqrt()

        def cross_feats(l):
            if not self.unit.cross_layer:
                return None
            aa, bb = problem.shapes[l]
            zero = torch.zeros_like(Gs[l])
            left = right = zero
            if l > 0 and problem.shapes[l - 1][1] == aa:  # shapes chain on the left
                Gp = Gs[l - 1]
                left = rmsn(torch.einsum("pza,pzc->pac", Gp, Gp) @ Gs[l]
                            / math.sqrt(Gp.shape[1]))
            if l + 1 < len(problem.shapes) and problem.shapes[l + 1][0] == bb:
                Gn = Gs[l + 1]
                right = rmsn(Gs[l] @ torch.einsum("pbc,pdc->pbd", Gn, Gn)
                             / math.sqrt(Gn.shape[2]))
            return torch.stack([left, right], dim=1)

        new_x, new_H, new_s = [], [], []
        i = 0
        for l, (aa, bb) in enumerate(problem.shapes):
            Gm = g[:, i:i + aa * bb].view(-1, aa, bb)
            delta, Hn, sn, log_step = self.unit(Gm, H[l], s[l], s_global, y,
                                                t=t, budget=budget,
                                                cross=cross_feats(l))
            gauge = math.sqrt(48.0 / aa) if self.unit.fanin_gauge else 1.0
            new_x.append(x[:, i:i + aa * bb]
                         + delta.reshape(x.shape[0], aa * bb) * (lr_scale * gauge))
            i += aa * bb
            if has_biases:
                # bias block: normalized-gradient step at the layer's learned step size,
                # capped low — an unbounded bias step blows up the inner trajectory and
                # feeds back into the meta-gradients (observed: gnorm 5e9)
                gb = g[:, i:i + bb]
                gb_rms = (gb.pow(2).mean(dim=1, keepdim=True) + 1e-24).sqrt()
                new_x.append(x[:, i:i + bb]
                             - (gb / gb_rms) * log_step.clamp(max=-2.0).exp() * lr_scale)
                i += bb
            new_H.append(Hn)
            new_s.append(sn)
        return torch.cat(new_x, dim=1), new_H, new_s, y

    def forward(self, x0, problem, n_steps, meta=True, lr_scale=1.0):
        x = x0.requires_grad_(True)
        H, s = self.init_state(x0.shape[0], problem.shapes, x0.device)
        losses = []
        for t in range(n_steps):
            x, H, s, y = self.step(problem, x, H, s, create_graph=meta,
                                   lr_scale=lr_scale, t=t, budget=n_steps)
            if not meta:
                x = x.detach().requires_grad_(True)
                H = [h.detach() for h in H]
                s = [t.detach() for t in s]
                y = y.detach()
            losses.append(y)
        final = problem.meta_objective(x)
        losses.append(final if meta else final.detach())
        return x.detach() if not meta else x, torch.stack(losses, dim=1)

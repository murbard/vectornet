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

from vectornet_torch import ScalarMLP, newton_schulz, signed_log


def triple(X, Y, Z):
    """T(X,Y,Z) = X Y^T Z for batched (P, a, b) stacks."""
    return X @ Y.transpose(-1, -2) @ Z


def _ns_sqrt_pair(A, iters):
    """Coupled Newton-Schulz: Y->A^{1/2}, Z->A^{-1/2} for symmetric A, spectrum in (0,1]."""
    n = A.shape[-1]
    I = torch.eye(n, device=A.device, dtype=A.dtype)
    Y, Z = A, I.expand_as(A).clone()
    for _ in range(iters):
        T = 0.5 * (3 * I - Z @ Y)
        Y = Y @ T
        Z = T @ Z
    return Y, Z


def inv_fourth_root(M, iters=12, eps=1e-8):
    """M^{-1/4} for batched symmetric PSD M via composed Newton-Schulz (matmul-only, in
    the triple-product span, differentiable). Two nested sqrt-pairs: A^{-1/2} then its
    sqrt. Trace-normalized + damped for conditioning. Validated vs eigh to ~1e-3 rel."""
    M = 0.5 * (M + M.transpose(-1, -2))
    n = M.shape[-1]
    I = torch.eye(n, device=M.device, dtype=M.dtype)
    s = torch.diagonal(M, dim1=-2, dim2=-1).sum(-1).clamp_min(eps)  # ~ scale (trace)
    s = s.view(*s.shape, 1, 1)
    A = M / s + eps * I
    _, Zi = _ns_sqrt_pair(A, iters)                     # Zi ~ A^{-1/2}
    s2 = torch.diagonal(Zi, dim1=-2, dim2=-1).sum(-1).clamp_min(eps).view(*s.shape)
    Yh, _ = _ns_sqrt_pair(Zi / s2 + eps * I, iters)     # Yh ~ (Zi/s2)^{1/2}
    return Yh * s2.pow(0.5) * s.pow(-0.25)              # M^{-1/4}


class MatrixUnit(nn.Module):
    """
    One step of the recurrent optimizer for ONE layer's matrix block, batched over
    P problems; the same module (same weights) is applied to every layer.

    Per-layer state: H (P, k, a, b) hidden matrices, s (P, n_scal_local) local scalars.
    A global scalar bus (mean-pooled over layers) enters every layer's scalar net.
    """

    def __init__(self, k_hidden=4, k_mid=8, n_triples=4, n_scal_local=16,
                 n_scal_global=16, n_dot=16, n_scal_hidden=64, time_inputs=False,
                 cross_layer=False, fanin_gauge=False, momentum=False, spectral=False,
                 blend=False, ns_iters=5, second_order=False, so_max_dim=768):
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
        # spectral: orthogonalize the momentum buffer (Newton-Schulz) before applying,
        # bounding update singular values ~1 like Muon -- controls magnitude regardless
        # of momentum, curing the overshoot that momentum alone causes. Requires momentum.
        self.spectral = spectral
        # blend: emit a per-layer mix rho=sigmoid(.) and apply
        # (rho*orthogonalized + (1-rho)*raw)*step, so the rule ORTHOGONALIZES WHERE IT
        # HELPS (large square transformer matrices) and leaves small/tall MLP layers raw.
        # Generalizes v12 (rho=0) and v13 (rho=1); keyed on shape via log a, log b inputs.
        self.blend = blend
        # ns_iters: Newton-Schulz orthogonalization iterations. Higher = cleaner
        # orthogonalization = better LATE-PHASE descent (eval-time test: 5->8 improved
        # the scale trajectory, gap widening with steps). The rule is retrained at the
        # chosen count so it adapts to the higher-quality update.
        self.ns_iters = ns_iters
        # second_order: maintain per-layer curvature accumulators L=EMA(GG^T), R=EMA(G^T G)
        # and expose the Shampoo/K-FAC preconditioned direction L^{-1/4} M R^{-1/4} as an
        # additional learned update candidate (blended with the orthogonalized one). This
        # is the LATE-PHASE lever the diagnosis pointed to; v13 (orthogonalization) is the
        # L=R=I special case. so_max_dim caps which matrices get 2nd-order (memory); larger
        # ones fall back to orthogonalization. The learned rho2 mixes 2nd-order vs spectral.
        self.second_order = second_order
        self.so_max_dim = so_max_dim
        assert not ((spectral or blend) and not momentum), "spectral/blend require momentum"
        assert not (second_order and not momentum), "second_order requires momentum"

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
        # net_b output slots (explicit, extensible): [local(nl)] [log_step] [gamma?]
        # [rho?] [rho2?]. Indices computed in forward from the same flags.
        self._i_step = n_scal_local
        self._i_gamma = n_scal_local + 1
        self._i_rho = self._i_gamma + (1 if momentum else 0)
        self._i_rho2 = self._i_rho + (1 if blend else 0)
        n_out = self._i_rho2 + (1 if second_order else 0)
        self.net_b = ScalarMLP([n_feat + n_dot, n_scal_hidden, n_out])
        with torch.no_grad():
            self.net_b.layers[-1].bias[self._i_step] = -3.0   # step exp(-3)~0.05
            if momentum:  # gamma ~ sigmoid(2)=0.88 (Muon-like heavy damping)
                self.net_b.layers[-1].bias[self._i_gamma] = 2.0
            if blend:  # rho ~ sigmoid(1)=0.73 (lean toward orthogonalizing)
                self.net_b.layers[-1].bias[self._i_rho] = 1.0
            if second_order:  # rho2 ~ sigmoid(-1)=0.27 (start mostly spectral, ease in 2nd-order)
                self.net_b.layers[-1].bias[self._i_rho2] = -1.0

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
        nl = self.n_scal_local
        s_local_new = torch.tanh(s_out[:, :nl])
        log_step = s_out[:, self._i_step:self._i_step + 1].clamp(-8.0, 2.0)
        gamma = torch.sigmoid(s_out[:, self._i_gamma:self._i_gamma + 1]) if self.momentum else None
        rho = torch.sigmoid(s_out[:, self._i_rho:self._i_rho + 1]) if self.blend else None
        rho2 = (torch.sigmoid(s_out[:, self._i_rho2:self._i_rho2 + 1])
                if self.second_order else None)

        out = torch.einsum("pmab,mo->poab", pool, self.out_comb)
        raw_dir = out[:, 0]                                    # unscaled update direction
        step = log_step.exp().unsqueeze(-1)
        # hidden matrices store directions only (rms-normalized on write): an unbounded
        # linear recurrence explodes over long horizons; magnitudes belong to the
        # tanh-bounded scalar memory
        H_new = out[:, 1:]
        H_new = H_new / (H_new.pow(2).mean(dim=(2, 3), keepdim=True) + 1e-24).sqrt()

        if self.momentum:
            g_ = gamma.unsqueeze(-1)
            if self.blend:
                # learned mix of orthogonalized and raw-but-magnitude-controlled update:
                # rho->1 orthogonalizes (good on big square matrices), rho->0 leaves the
                # raw momentum direction (good on small/tall MLP layers).
                M_new = g_ * M_prev + (1.0 - g_) * raw_dir
                ortho = newton_schulz(M_new, iters=self.ns_iters)
                rawn = M_new / (M_new.pow(2).mean(dim=(1, 2), keepdim=True) + 1e-24).sqrt()
                r_ = rho.unsqueeze(-1)
                delta = (r_ * ortho + (1.0 - r_) * rawn) * step
            elif self.spectral:
                # Muon's exact structure: momentum buffer on the raw direction,
                # orthogonalize (bound every singular value ~1 -> magnitude controlled
                # regardless of momentum, curing the overshoot momentum alone causes),
                # then scale by the learned step. Newton-Schulz is a polynomial in our
                # triple product, so this stays in-span.
                M_new = g_ * M_prev + (1.0 - g_) * raw_dir
                delta = newton_schulz(M_new, iters=self.ns_iters) * step
            else:
                # momentum on the already-scaled raw delta, applied directly
                M_new = g_ * M_prev + (1.0 - g_) * (raw_dir * step)
                delta = M_new
            H_new = torch.cat([H_new, M_new.unsqueeze(1)], dim=1)
        else:
            delta = raw_dir * step
        # second-order preconditioning is applied in step() (it holds L,R and the
        # gradient); the unit exposes rho2 (mix) and the step scale for it.
        return delta, H_new, s_local_new, log_step, rho2, step


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

    def _so_ok(self, aa, bb):
        return self.unit.second_order and max(aa, bb) <= self.unit.so_max_dim

    def init_state(self, P, shapes, device):
        # When momentum is on, each H tensor carries one EXTRA channel: the last slot is
        # the magnitude-carrying momentum buffer M (not rms-normalized), the first
        # k_hidden are the usual direction memories.
        kh = self.unit.k_hidden + (1 if self.unit.momentum else 0)
        H = [torch.zeros(P, kh, aa, bb, device=device) for aa, bb in shapes]
        s = [torch.zeros(P, self.unit.n_scal_local, device=device) for _ in shapes]
        # second-order curvature accumulators L=(a,a), R=(b,b) per eligible layer
        so = None
        if self.unit.second_order:
            so = []
            for aa, bb in shapes:
                if self._so_ok(aa, bb):
                    so.append([torch.zeros(P, aa, aa, device=device),
                               torch.zeros(P, bb, bb, device=device)])
                else:
                    so.append(None)
        return H, s, so

    def step(self, problem, x, H, s, so=None, create_graph=True, lr_scale=1.0,
             t=0, budget=None):
        """lr_scale: single tunable scalar multiplying every update (1.0 = zero-shot).
        so: per-layer [L, R] curvature accumulators (second-order), threaded like H,s."""
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
        new_so = [] if so is not None else None
        i = 0
        for l, (aa, bb) in enumerate(problem.shapes):
            Gm = g[:, i:i + aa * bb].view(-1, aa, bb)
            delta, Hn, sn, log_step, rho2, step_scale = self.unit(
                Gm, H[l], s[l], s_global, y, t=t, budget=budget, cross=cross_feats(l))

            # SECOND-ORDER: blend the orthogonalized update with the Shampoo-preconditioned
            # momentum L^{-1/4} M R^{-1/4} (rms-matched), by the learned mix rho2.
            if so is not None and so[l] is not None:
                L, R = so[l]
                beta = 0.95
                Ln = beta * L + (1 - beta) * (Gm @ Gm.transpose(-1, -2))
                Rn = beta * R + (1 - beta) * (Gm.transpose(-1, -2) @ Gm)
                new_so.append([Ln.detach(), Rn.detach()])  # curvature stats: no meta-graph

                def _damp(A):  # relative damping so early rank-deficient A is conditioned
                    n = A.shape[-1]
                    I = torch.eye(n, device=A.device, dtype=A.dtype)
                    tr = torch.diagonal(A, dim1=-2, dim2=-1).mean(-1).clamp_min(1e-12)
                    return A + 0.1 * tr.view(-1, 1, 1) * I

                M = Hn[:, -1]                               # momentum buffer
                so_dir = (inv_fourth_root(_damp(Ln.detach())) @ M
                          @ inv_fourth_root(_damp(Rn.detach())))
                so_dir = (so_dir / (so_dir.pow(2).mean(dim=(1, 2), keepdim=True) + 1e-24).sqrt()
                          * (delta.pow(2).mean(dim=(1, 2), keepdim=True) + 1e-24).sqrt())
                # ramp 2nd-order in over the first ~50 steps while L,R accumulate (they are
                # near-rank-1 early -> the preconditioned direction is unreliable then)
                ramp = min(1.0, t / 50.0)
                r2 = rho2.unsqueeze(-1) * ramp
                delta = (1.0 - r2) * delta + r2 * so_dir
            elif new_so is not None:
                new_so.append(None)

            gauge = math.sqrt(48.0 / aa) if self.unit.fanin_gauge else 1.0
            new_x.append(x[:, i:i + aa * bb]
                         + delta.reshape(x.shape[0], aa * bb) * (lr_scale * gauge))
            i += aa * bb
            if has_biases:
                gb = g[:, i:i + bb]
                gb_rms = (gb.pow(2).mean(dim=1, keepdim=True) + 1e-24).sqrt()
                new_x.append(x[:, i:i + bb]
                             - (gb / gb_rms) * log_step.clamp(max=-2.0).exp() * lr_scale)
                i += bb
            new_H.append(Hn)
            new_s.append(sn)
        return torch.cat(new_x, dim=1), new_H, new_s, new_so, y

    def forward(self, x0, problem, n_steps, meta=True, lr_scale=1.0):
        x = x0.requires_grad_(True)
        H, s, so = self.init_state(x0.shape[0], problem.shapes, x0.device)
        losses = []
        for t in range(n_steps):
            x, H, s, so, y = self.step(problem, x, H, s, so, create_graph=meta,
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

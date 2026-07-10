"""
Batched tiny-GPT optimizee (nanoGPT-style, char-level), built so that EVERY trainable
parameter is a matrix: no biases, parameter-free RMSNorm, learned position embeddings.
That makes it a pure workload for the matrix optimizer — and for Muon, which is exactly
the apples-to-apples opponent on transformers.

Forward is vectorized over P independent problem instances (einsum with a leading P
axis), matching the MLPProblem interface: shapes, init_point, objective/meta_objective,
accuracy (next-char top-1).
"""
import math

import torch
import torch.nn.functional as F


def rmsnorm(x):
    return x / (x.pow(2).mean(dim=-1, keepdim=True) + 1e-8).sqrt()


class TransformerLMProblem:
    has_biases = False  # every parameter is a matrix

    def __init__(self, text_ids, vocab, P, batch_size, device, generator=None,
                 d_model=32, n_layers=2, n_heads=2, ctx=64):
        assert d_model % n_heads == 0
        self.V, self.d, self.L, self.h, self.T = vocab, d_model, n_layers, n_heads, ctx
        self.P, self.B, self.device = P, batch_size, device

        # sample (P, B) random windows of T+1 chars
        starts = torch.randint(len(text_ids) - ctx - 1, (P, batch_size),
                               device=device, generator=generator)
        offs = torch.arange(ctx + 1, device=device)
        chunks = text_ids[starts.unsqueeze(-1) + offs]  # (P, B, T+1)
        self.x_tok = chunks[:, :, :-1]
        self.y_tok = chunks[:, :, 1:]

        d, V, T = d_model, vocab, ctx
        self.shapes = [(V, d), (T, d)]
        for _ in range(n_layers):
            self.shapes += [(d, d), (d, d), (d, d), (d, d),  # Wq Wk Wv Wo
                            (d, 4 * d), (4 * d, d)]          # MLP
        self.shapes += [(d, V)]
        self.n_params = sum(a * b for a, b in self.shapes)  # no biases anywhere
        self.mask = torch.full((T, T), float("-inf"), device=device).triu(1)

    def init_point(self, generator=None):
        parts = []
        for a, b in self.shapes:
            parts.append(torch.randn(self.P, a * b, device=self.device,
                                     generator=generator) / math.sqrt(a))
        return torch.cat(parts, dim=1)

    def _unpack(self, x):
        mats, i = [], 0
        for a, b in self.shapes:
            mats.append(x[:, i:i + a * b].view(self.P, a, b))
            i += a * b
        return mats

    def logits(self, x):
        mats = self._unpack(x)
        E, Pos = mats[0], mats[1]
        U = mats[-1]
        blocks = mats[2:-1]
        d, h, T = self.d, self.h, self.T
        dh = d // h

        onehot = F.one_hot(self.x_tok, self.V).to(x.dtype)          # (P,B,T,V)
        z = torch.einsum("pbtv,pvd->pbtd", onehot, E) + Pos.unsqueeze(1)
        for l in range(self.L):
            Wq, Wk, Wv, Wo, W1, W2 = blocks[6 * l:6 * l + 6]
            zn = rmsnorm(z)
            q = torch.einsum("pbtd,pde->pbte", zn, Wq).view(*z.shape[:3], h, dh)
            k = torch.einsum("pbtd,pde->pbte", zn, Wk).view(*z.shape[:3], h, dh)
            v = torch.einsum("pbtd,pde->pbte", zn, Wv).view(*z.shape[:3], h, dh)
            att = torch.einsum("pbqhe,pbkhe->pbhqk", q, k) / math.sqrt(dh)
            att = torch.softmax(att + self.mask, dim=-1)
            o = torch.einsum("pbhqk,pbkhe->pbqhe", att, v).reshape(*z.shape[:3], d)
            z = z + torch.einsum("pbtd,pde->pbte", o, Wo)
            zn = rmsnorm(z)
            m = torch.relu(torch.einsum("pbtd,pdf->pbtf", zn, W1))
            z = z + torch.einsum("pbtf,pfd->pbtd", m, W2)
        return torch.einsum("pbtd,pdv->pbtv", rmsnorm(z), U)

    def objective(self, x):
        logp = F.log_softmax(self.logits(x), dim=-1)
        nll = -logp.gather(-1, self.y_tok.unsqueeze(-1)).squeeze(-1)  # (P,B,T)
        return nll.mean(dim=(1, 2))  # (P,)

    meta_objective = objective

    def accuracy(self, x):
        with torch.no_grad():
            return (self.logits(x).argmax(-1) == self.y_tok).float().mean(dim=(1, 2))

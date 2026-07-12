"""
Modern PyTorch port of vectornet: meta-learning a gradient-descent algorithm.

Design philosophy (unchanged from the 2015 Theano original):
  - Two neuron types: scalars, and vectors of *symbolic* dimension n.
  - The only vector operations are (a) linear combinations of vectors, with
    coefficients produced by a scalar network, and (b) dot products between
    linear combinations of vectors, which feed back into the scalar network.
  - Because none of the learned weights depend on n, an optimizer meta-trained
    on a small problem applies verbatim to a problem of any dimension.

Trainability changes vs. the original (all dimension-polymorphic):
  - The gradient is fed in normalized, with log ||g|| as a scalar input
    (||g|| is itself a dot product, so this stays inside the philosophy).
  - Dot products pass through a signed-log squashing before the scalar net.
  - The unit outputs a *delta* added to x rather than x itself.
  - Meta-loss is the trajectory average of f(x_t) / f(x_0).
"""
import argparse
import gzip
import math
import os
import struct
import urllib.request

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def signed_log(x, eps=1e-8):
    """Squash values spanning many orders of magnitude: sign(x) * log1p(|x|/eps scaled)."""
    return torch.sign(x) * torch.log1p(torch.abs(x) / eps) / 20.0


class ScalarMLP(nn.Module):
    """Fully connected scalar network (the only place nonlinear scalar computation happens)."""

    def __init__(self, sizes, activation=torch.tanh, final_linear=True):
        super().__init__()
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(sizes, sizes[1:]))
        self.activation = activation
        self.final_linear = final_linear

    def forward(self, s):
        for i, layer in enumerate(self.layers):
            s = layer(s)
            if not (self.final_linear and i == len(self.layers) - 1):
                s = self.activation(s)
        return s


class Unit(nn.Module):
    """
    One step of the recurrent optimizer, batched over P problem instances.

    State:  x (P, n) candidate point, V (P, n_hidden_vec, n) vector memory,
            S (P, n_hidden_scal) scalar memory.
    """

    def __init__(self, n_hidden_vec=8, n_mid_vec=16, n_hidden_scal=32, n_dot=32,
                 n_scal_hidden=64, vector_activation="tanh", rms_convention=False,
                 intensive_inputs=False):
        super().__init__()
        self.n_hidden_vec = n_hidden_vec
        self.n_mid_vec = n_mid_vec
        self.n_hidden_scal = n_hidden_scal
        self.n_dot = n_dot
        self.vector_activation = vector_activation
        # Intensive-units convention: vectors carry O(1) coordinates (g normalized by
        # rms rather than norm), dot products average over coordinates rather than sum,
        # and log(n) is fed to the scalar net. Makes every scalar magnitude — and the
        # regime the coordinatewise nonlinearity operates in — independent of n.
        self.rms_convention = rms_convention
        # Strict-intensive inputs (requires rms_convention): feed signed_log(y/n) and no
        # explicit log n, so every scalar input is per-coordinate. Buys exact replication
        # invariance ("thermodynamic limit" optimizer) at the cost of assuming f is
        # extensive — wrong for e.g. NLL, which is already intensive in n. A/B this.
        self.intensive_inputs = intensive_inputs
        assert not (intensive_inputs and not rms_convention)

        n_vec_in = n_hidden_vec + 1  # hidden vectors + normalized gradient
        n_vec_out = n_hidden_vec + 1  # delta + new hidden vectors

        # scalar net A: state scalars + f value + gradient scale (+ log n)
        n_scal_in = n_hidden_scal + 2 + (1 if (rms_convention and not intensive_inputs) else 0)
        self.scalar_net_a = ScalarMLP([n_scal_in, n_scal_hidden, n_vec_in * n_mid_vec])
        # data-independent bias combination (so e.g. plain momentum needs no scalar path)
        self.comb_bias = nn.Parameter(0.01 * torch.randn(n_vec_in, n_mid_vec))

        # dot products of learned linear combinations of the mid vectors
        self.dot_w1 = nn.Parameter(torch.randn(n_mid_vec, n_dot) / math.sqrt(n_mid_vec))
        self.dot_w2 = nn.Parameter(torch.randn(n_mid_vec, n_dot) / math.sqrt(n_mid_vec))

        # scalar net B: previous scalars + dot products -> new scalar state + log step size
        self.scalar_net_b = ScalarMLP([n_scal_in + n_dot, n_scal_hidden, n_hidden_scal + 1])

        # fixed linear combination producing the output vectors; small init so the
        # optimizer starts near-stationary
        self.out_comb = nn.Parameter(0.1 * torch.randn(n_mid_vec, n_vec_out) / math.sqrt(n_mid_vec))

    def forward(self, x, V, S, f, create_graph=True):
        P = x.shape[0]
        y = f(x)
        (g,) = torch.autograd.grad(y.sum(), x, create_graph=create_graph)

        n = x.shape[1]
        if self.rms_convention:
            g_scale = g.norm(dim=1, keepdim=True) / math.sqrt(n)  # rms: intensive
            g_hat = g / (g_scale + 1e-12)  # O(1) coordinates at any n
            if self.intensive_inputs:
                scalars_in = torch.cat(
                    [S, signed_log(y / n).unsqueeze(1), signed_log(g_scale)], dim=1)
            else:
                log_n = torch.full((P, 1), math.log(n) / 10.0, device=x.device)
                scalars_in = torch.cat(
                    [S, signed_log(y).unsqueeze(1), signed_log(g_scale), log_n], dim=1)
        else:
            g_norm = g.norm(dim=1, keepdim=True)
            g_hat = g / (g_norm + 1e-12)
            scalars_in = torch.cat([S, signed_log(y).unsqueeze(1), signed_log(g_norm)], dim=1)

        # (a) linear combinations of vectors, coefficients from the scalar net
        Vin = torch.cat([V, g_hat.unsqueeze(1)], dim=1)  # (P, n_vec_in, n)
        coeff = self.scalar_net_a(scalars_in).view(P, -1, self.n_mid_vec) + self.comb_bias
        mid = torch.einsum("pin,pim->pmn", Vin, coeff)
        if self.vector_activation == "tanh":
            mid = torch.tanh(mid)

        # (b) dot products between linear combinations of the mid vectors
        left = torch.einsum("pmn,md->pdn", mid, self.dot_w1)
        right = torch.einsum("pmn,md->pdn", mid, self.dot_w2)
        raw = (left * right).mean(dim=2) if self.rms_convention else (left * right).sum(dim=2)
        dots = signed_log(raw)  # (P, n_dot)

        s_out = self.scalar_net_b(torch.cat([scalars_in, dots], dim=1))
        S_new = torch.tanh(s_out[:, :-1])
        log_step = s_out[:, -1:].clamp(-8.0, 2.0)

        Vout = torch.einsum("pmn,mo->pon", mid, self.out_comb)
        delta = Vout[:, 0] * torch.exp(log_step)
        V_new = Vout[:, 1:]

        return x + delta, V_new, S_new, y


class LearnedOptimizer(nn.Module):
    """Runs the Unit for K steps; returns the trajectory of objective values."""

    def __init__(self, **unit_kwargs):
        super().__init__()
        self.unit_kwargs = unit_kwargs
        self.unit = Unit(**unit_kwargs)

    def init_state(self, x0):
        P, n = x0.shape
        V = torch.zeros(P, self.unit.n_hidden_vec, n, device=x0.device)
        S = torch.zeros(P, self.unit.n_hidden_scal, device=x0.device)
        return V, S

    def forward(self, x0, f, n_steps, meta=True):
        """meta=True keeps the graph across steps for meta-gradients; meta=False
        truncates it after every step (evaluation only)."""
        x = x0.requires_grad_(True)
        V, S = self.init_state(x0)
        losses = []
        for _ in range(n_steps):
            x, V, S, y = self.unit(x, V, S, f, create_graph=meta)
            if not meta:
                x = x.detach().requires_grad_(True)
                V, S, y = V.detach(), S.detach(), y.detach()
            losses.append(y)
        if meta:
            losses.append(f(x))
        else:
            with torch.no_grad():
                losses.append(f(x))
        return x.detach(), torch.stack(losses, dim=1)  # (P, n_steps + 1)


# ---------------------------------------------------------------- MNIST task

MNIST_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/"
CACHE = os.path.expanduser("~/.cache")


def load_idx(path):
    with gzip.open(path, "rb") as fh:
        magic = struct.unpack(">I", fh.read(4))[0]
        if magic == 2051:
            num, rows, cols = struct.unpack(">III", fh.read(12))
            return np.frombuffer(fh.read(), np.uint8).reshape(num, rows * cols)
        num = struct.unpack(">I", fh.read(4))[0]
        return np.frombuffer(fh.read(), np.uint8)


def load_fashion(device):
    d = os.path.join(CACHE, "fashion_mnist")
    x = load_idx(os.path.join(d, "train-images-idx3-ubyte.gz")).astype(np.float32) / 255.0
    y = load_idx(os.path.join(d, "train-labels-idx1-ubyte.gz")).astype(np.int64)
    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


def load_kmnist(device):
    d = os.path.join(CACHE, "kmnist")
    x = load_idx(os.path.join(d, "train-images-idx3-ubyte.gz")).astype(np.float32) / 255.0
    y = load_idx(os.path.join(d, "train-labels-idx1-ubyte.gz")).astype(np.int64)
    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


def load_cifar10(device):
    npz = os.path.join(CACHE, "cifar10", "cifar10.npz")
    if not os.path.exists(npz):
        import pickle
        import tarfile
        xs, ys = [], []
        with tarfile.open(os.path.join(CACHE, "cifar10", "cifar-10-python.tar.gz")) as tf:
            for i in range(1, 6):
                d = pickle.load(tf.extractfile(f"cifar-10-batches-py/data_batch_{i}"),
                                encoding="bytes")
                xs.append(d[b"data"])
                ys.append(d[b"labels"])
        np.savez(npz, x=np.concatenate(xs), y=np.concatenate(ys))
    d = np.load(npz)
    x = d["x"].astype(np.float32) / 255.0
    y = d["y"].astype(np.int64)
    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


def load_covtype(device):
    npy = os.path.join(CACHE, "covtype", "covtype.npy")
    if os.path.exists(npy):
        data = np.load(npy)
    else:
        data = np.loadtxt(os.path.join(CACHE, "covtype", "covtype.data.gz"), delimiter=",",
                          dtype=np.float32)
        np.save(npy, data)
    x, y = data[:, :54], data[:, 54].astype(np.int64) - 1
    mu, sd = x[:, :10].mean(0), x[:, :10].std(0) + 1e-8
    x = x.copy()
    x[:, :10] = (x[:, :10] - mu) / sd
    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


def load_mnist(cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    arrays = {}
    for name, kind in [("train-images-idx3-ubyte.gz", "img"), ("train-labels-idx1-ubyte.gz", "lbl"),
                       ("t10k-images-idx3-ubyte.gz", "img"), ("t10k-labels-idx1-ubyte.gz", "lbl")]:
        path = os.path.join(cache_dir, name)
        if not os.path.exists(path):
            print("downloading", name)
            urllib.request.urlretrieve(MNIST_URL + name, path)
        with gzip.open(path, "rb") as fh:
            if kind == "img":
                _, num, rows, cols = struct.unpack(">IIII", fh.read(16))
                arrays[name] = np.frombuffer(fh.read(), np.uint8).reshape(num, rows * cols)
            else:
                _, num = struct.unpack(">II", fh.read(8))
                arrays[name] = np.frombuffer(fh.read(), np.uint8)
    train_x = arrays["train-images-idx3-ubyte.gz"].astype(np.float32) / 255.0
    train_y = arrays["train-labels-idx1-ubyte.gz"].astype(np.int64)
    test_x = arrays["t10k-images-idx3-ubyte.gz"].astype(np.float32) / 255.0
    test_y = arrays["t10k-labels-idx1-ubyte.gz"].astype(np.int64)
    return train_x, train_y, test_x, test_y


class MLPProblem:
    """
    A batch of P independent 'train this MLP on this minibatch' problems.
    The MLP parameters live flattened in a single vector of dimension n:
    the learned optimizer never sees the layer structure.
    """

    def __init__(self, images, labels, hidden, P, batch_size, device, generator=None,
                 in_dim=784, n_classes=10, activation="relu", project_to=None):
        # project_to: pass the minibatch through a random projection to that input
        # dimension — turns one dataset into a family of tasks of arbitrary in_dim
        eff_in = project_to or in_dim
        hiddens = [hidden] if isinstance(hidden, int) else list(hidden)
        dims = [eff_in] + hiddens + [n_classes]
        self.shapes = list(zip(dims, dims[1:]))
        self.activation = activation
        self.n_params = sum(a * b + b for a, b in self.shapes)
        if images.device != device:  # dataset resident on CPU: sample there, move batch
            idx = torch.randint(len(images), (P, batch_size))
            self.x_data = images[idx].to(device)
            self.y_data_pre = labels[idx].to(device)
        else:
            idx = torch.randint(len(images), (P, batch_size), device=device,
                                generator=generator)
            self.x_data = images[idx]  # (P, B, in_dim)
            self.y_data_pre = labels[idx]
        if project_to:
            R = torch.randn(in_dim, project_to, device=device,
                            generator=generator) / math.sqrt(in_dim)
            self.x_data = self.x_data @ R
        self.y_data = self.y_data_pre  # (P, B)
        self.P, self.B = P, batch_size
        self.device = device

    def init_point(self, generator=None):
        # per-layer fan-in scaling, applied at init only; the optimizer itself
        # still treats the parameters as one unstructured vector
        parts = []
        for a, b in self.shapes:
            w = torch.randn(self.P, a * b, device=self.device, generator=generator) / math.sqrt(a)
            parts += [w, torch.zeros(self.P, b, device=self.device)]
        return torch.cat(parts, dim=1)

    def logits(self, x, sel=slice(None)):
        z = self.x_data[sel]
        P = z.shape[0]
        i = 0
        for j, (a, b) in enumerate(self.shapes):
            W = x[:, i:i + a * b].view(P, a, b)
            i += a * b
            bias = x[:, i:i + b].view(P, 1, b)
            i += b
            z = torch.bmm(z, W) + bias
            if j < len(self.shapes) - 1:
                z = torch.relu(z) if self.activation == "relu" else torch.tanh(z)
        return z

    def objective(self, x, sel=slice(None)):
        logp = F.log_softmax(self.logits(x, sel), dim=2)
        return -logp.gather(2, self.y_data[sel].unsqueeze(2)).squeeze(2).mean(dim=1)  # (P,)

    def objective_one(self, x, p):
        return self.objective(x, sel=slice(p, p + 1))

    meta_objective = objective  # already positive

    def accuracy(self, x):
        with torch.no_grad():
            return (self.logits(x).argmax(2) == self.y_data).float().mean(dim=1)


class ReparamWrapper:
    """Celo2-style global reparameterization augmentation: the wrapped problem is the
    same optimization task expressed in rescaled coordinates (x -> c*x) with a rescaled
    loss (f -> s*f). A scale-robust optimizer should be invariant to both; meta-training
    over random (c, s) enforces the invariance instead of assuming it."""

    def __init__(self, base, param_scale, loss_scale):
        self.base, self.c, self.s = base, param_scale, loss_scale
        self.shapes = getattr(base, "shapes", None)

    def init_point(self, generator=None):
        return self.base.init_point(generator) * self.c

    def objective(self, x):
        return self.s * self.base.objective(x / self.c)

    def meta_objective(self, x):
        return self.s * self.base.meta_objective(x / self.c)

    def accuracy(self, x):
        return self.base.accuracy(x / self.c)


class TeacherStudentProblem(MLPProblem):
    """Synthetic classification: gaussian inputs labeled by a random frozen teacher MLP.
    An unlimited task family at any input dimension / class count."""

    def __init__(self, d_in, n_classes, hidden, P, batch_size, device, generator=None,
                 teacher_hidden=32, activation="relu"):
        hiddens = [hidden] if isinstance(hidden, int) else list(hidden)
        dims = [d_in] + hiddens + [n_classes]
        self.shapes = list(zip(dims, dims[1:]))
        self.activation = activation
        self.n_params = sum(a * b + b for a, b in self.shapes)
        self.P, self.B, self.device = P, batch_size, device
        self.x_data = torch.randn(P, batch_size, d_in, device=device, generator=generator)
        W1 = torch.randn(P, d_in, teacher_hidden, device=device,
                         generator=generator) / math.sqrt(d_in)
        W2 = torch.randn(P, teacher_hidden, n_classes, device=device,
                         generator=generator) / math.sqrt(teacher_hidden)
        with torch.no_grad():
            self.y_data = torch.relu(self.x_data @ W1).bmm(W2).argmax(dim=2)


class QuarticProblem:
    """f(x) = sum_i (z_i^4 - 16 z_i^2 + 5 z_i) / 2, z = x - offset  (vectornet.py, 2015).
    Separable, non-convex (two basins per coordinate). Global min = n * F_MIN_PER_COORD."""

    F_MIN_PER_COORD = -39.166166  # of the half-scaled function, at z = -2.903535

    def __init__(self, n_dim, P, device, generator=None):
        self.offset = 2.0 * torch.randn(P, n_dim, device=device, generator=generator)
        self.n_dim, self.P, self.device = n_dim, P, device

    def init_point(self, generator=None):
        return 2.0 * torch.randn(self.P, self.n_dim, device=self.device, generator=generator)

    def objective(self, x):
        z = x - self.offset
        return (z.pow(4) - 16.0 * z.pow(2) + 5.0 * z).sum(dim=1) / 2.0

    def objective_one(self, x, p):
        z = x - self.offset[p:p + 1]
        return (z.pow(4) - 16.0 * z.pow(2) + 5.0 * z).sum(dim=1) / 2.0

    def meta_objective(self, x):
        return self.objective(x) - self.n_dim * self.F_MIN_PER_COORD  # positive gap

    def gap(self, x):
        with torch.no_grad():
            return (self.objective(x) / self.n_dim - self.F_MIN_PER_COORD).mean().item()


# ------------------------------------------------------------------ training

def train_pes(model, sample_problem, args, device, evaluate_hook=None):
    """
    Persistent Evolution Strategies (Vicol et al. 2021) meta-trainer.

    Why: backprop-through-unroll gave biased truncated gradients and the explosions in
    RESEARCH_LOG iteration 1; PES is unbiased across truncations, needs no graph through
    the unroll (each particle runs create_graph=False), and therefore scales to episodes
    of hundreds of inner steps on small GPU memory.

    2N antithetic particles share one problem per episode. Each particle k holds its own
    inner state, evolved under theta + eps_k with eps_k resampled every segment; xi_k
    accumulates eps_k over the episode. Estimator: g = mean_k [ xi_k * (L_k - L_bar) ] / sigma^2,
    where L_k is the particle's *segment* loss (log-ratio vs episode start).
    """
    from torch.nn.utils import parameters_to_vector, vector_to_parameters

    theta = nn.Parameter(parameters_to_vector(model.parameters()).detach().clone())
    meta_opt = torch.optim.Adam([theta], lr=args.meta_lr)
    n_part = 2 * args.pes_pairs
    seg, episode = args.segment, args.episode

    particles = None
    best_ema, emas = float("inf"), {}
    for step in range(args.meta_steps):
        if particles is None:  # new episode
            problem, _ = sample_problem()
            x0 = problem.init_point()
            with torch.enable_grad():
                f0 = problem.meta_objective(x0).detach() + 1e-9
            particles = [{
                "x": x0.clone(), "V": None, "S": None,
                "xi": torch.zeros_like(theta),
            } for _ in range(n_part)]
            steps_done = 0

        half = [args.sigma * torch.randn_like(theta) for _ in range(args.pes_pairs)]
        eps = [e for pair in zip(half, [-e for e in half]) for e in pair]

        losses = torch.zeros(n_part, device=device)
        for k in range(n_part):
            vector_to_parameters(theta.detach() + eps[k], model.parameters())
            p = particles[k]
            x = p["x"].detach().requires_grad_(True)
            V, S = (model.init_state(x) if p["V"] is None else (p["V"], p["S"]))
            seg_terms = []
            for _ in range(seg):
                x, V, S, y = model.unit(x, V, S, problem.meta_objective,
                                        create_graph=False)
                x = x.detach().requires_grad_(True)
                V, S = V.detach(), S.detach()
                seg_terms.append(torch.log(y.detach() / f0 + 1e-9).mean())
            losses[k] = torch.stack(seg_terms).mean()
            p["x"], p["V"], p["S"] = x.detach(), V, S
            p["xi"] = p["xi"] + eps[k]

        centered = losses - losses.mean()
        grad = torch.stack([particles[k]["xi"] * centered[k] for k in range(n_part)]
                           ).mean(dim=0) / (args.sigma ** 2)
        meta_opt.zero_grad()
        theta.grad = grad
        torch.nn.utils.clip_grad_norm_([theta], 1.0)
        meta_opt.step()

        steps_done += seg
        if steps_done >= episode:
            particles = None

        loss_now = losses.mean().item()
        kind = type(getattr(problem, "base", problem)).__name__
        emas[kind] = (loss_now if kind not in emas
                      else 0.98 * emas[kind] + 0.02 * loss_now)
        ema = sum(emas.values()) / len(emas)
        if ema < best_ema and step > 50:
            best_ema = ema
            vector_to_parameters(theta.detach(), model.parameters())
            torch.save({"state_dict": model.state_dict(),
                        "unit_kwargs": model.unit_kwargs}, args.save)
        if step % 25 == 0:
            print(f"pes {step:5d}  seg-loss {loss_now:.4f}  ema {ema:.4f}  "
                  f"episode-step {steps_done}", flush=True)
        if evaluate_hook and (step % args.eval_every == 0 or step == args.meta_steps - 1):
            vector_to_parameters(theta.detach(), model.parameters())
            evaluate_hook()

    vector_to_parameters(theta.detach(), model.parameters())


def meta_loss(loss_traj):
    """Mean over the trajectory of log f(x_t)/f(x_0): scale-free, dense signal,
    and self-normalizing gradients that stay bounded when the optimizer blows up."""
    return torch.log(loss_traj / loss_traj[:, :1].detach() + 1e-9).mean()


def newton_schulz(M, iters=5):
    """Odd-polynomial orthogonalization of a (P, a, b) batch (Muon; Jordan et al. 2024)."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = M / (M.norm(dim=(1, 2), keepdim=True) + 1e-7)
    transposed = X.shape[1] > X.shape[2]
    if transposed:
        X = X.transpose(1, 2)
    for _ in range(iters):
        A = X @ X.transpose(1, 2)
        X = a * X + (b * A + c * (A @ A)) @ X
    return X.transpose(1, 2) if transposed else X


def muon_step(problem, x, momenta, g, lr, beta=0.95):
    """Muon on the matrix blocks of the flattened parameter vector, momentum SGD on
    biases. Uses problem.shapes — unlike the learned optimizer, Muon NEEDS the layer
    structure; this is exactly what the flat-vector framework cannot express."""
    i = 0
    parts = []
    blocks = ([(True,), (True, False)][getattr(problem, "has_biases", True)])
    for k, (a, b) in enumerate(problem.shapes):
        for is_matrix in blocks:
            size = a * b if is_matrix else b
            gi = g[:, i:i + size]
            key = (k, is_matrix)
            momenta[key] = beta * momenta.get(key, torch.zeros_like(gi)) + gi
            if is_matrix:
                upd = newton_schulz(momenta[key].view(-1, a, b)).reshape(gi.shape)
                # Moonlight rescaling: match AdamW update RMS regardless of shape
                upd = upd * (0.2 * math.sqrt(max(a, b)))
            else:
                upd = momenta[key]
            parts.append(x[:, i:i + size] - lr * upd)
            i += size
    return torch.cat(parts, dim=1)


def run_baseline(problem, x0, opt_name, lr, n_steps):
    x = x0.clone().requires_grad_(True)
    losses = []
    if opt_name == "muon":
        momenta = {}
        for _ in range(n_steps):
            y = problem.objective(x)
            losses.append(y.detach())
            (g,) = torch.autograd.grad(y.sum(), x)
            with torch.no_grad():
                x = muon_step(problem, x, momenta, g, lr)
            x.requires_grad_(True)
    elif opt_name == "lbfgs":
        # per-problem L-BFGS (a joint one would share curvature across instances)
        cols, xs = [], []
        for p in range(x0.shape[0]):
            xp = x0[p:p + 1].clone().requires_grad_(True)
            fp = (lambda xx, pp=p: problem.objective_one(xx, pp))
            opt = torch.optim.LBFGS([xp], lr=lr, max_iter=1, history_size=10,
                                    line_search_fn="strong_wolfe")
            col = []
            for _ in range(n_steps):
                col.append(fp(xp).detach())
                def closure():
                    opt.zero_grad()
                    y = fp(xp)
                    y.backward()
                    return y
                opt.step(closure)
            col.append(fp(xp).detach())
            cols.append(torch.stack(col))
            xs.append(xp.detach())
        return torch.cat(xs, 0), torch.stack(cols, 0).squeeze(-1)
    else:
        if opt_name == "sgd":
            opt = torch.optim.SGD([x], lr=lr)
        elif opt_name == "prodigy":  # self-tuning (learning-rate-free) Adam family
            from prodigyopt import Prodigy
            opt = Prodigy([x], lr=lr)  # lr=1.0 is the canonical untuned setting
        else:
            opt = torch.optim.Adam([x], lr=lr)
        for _ in range(n_steps):
            opt.zero_grad()
            y = problem.objective(x)
            losses.append(y.detach())
            y.sum().backward()
            opt.step()
    losses.append(problem.objective(x).detach())
    return x.detach(), torch.stack(losses, dim=1)


def evaluate(model, images, labels, hidden, P, batch_size, n_steps, device, seed):
    gen = torch.Generator(device=device).manual_seed(seed)
    problem = MLPProblem(images, labels, hidden, P, batch_size, device, gen)
    x0 = problem.init_point(gen)
    x, traj = model(x0.clone(), problem.objective, n_steps, meta=False)
    results = {"learned": (traj.mean(0), problem.accuracy(x).mean().item())}
    for name, lrs in [("sgd", [0.1, 0.3, 1.0, 3.0]), ("adam", [3e-3, 1e-2, 3e-2, 1e-1])]:
        best = None
        for lr in lrs:
            xb, tb = run_baseline(problem, x0, name, lr, n_steps)
            if best is None or tb[:, -1].mean() < best[1][-1]:
                best = (f"{name}(lr={lr})", tb.mean(0), problem.accuracy(xb).mean().item())
        results[best[0]] = (best[1], best[2])
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta-steps", type=int, default=3000)
    ap.add_argument("--unroll", type=int, default=20)
    ap.add_argument("--problems", type=int, default=16, help="problem batch P")
    ap.add_argument("--batch-size", type=int, default=128, help="images per problem")
    ap.add_argument("--hidden", type=int, default=16, help="hidden units of the optimizee MLP")
    ap.add_argument("--meta-lr", type=float, default=1e-3)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--save", default="learned_opt.pt")
    ap.add_argument("--rms", action="store_true",
                    help="intensive-units convention: rms-normalized inputs, mean dot products, log n input")
    ap.add_argument("--trainer", choices=["bptt", "pes"], default="bptt")
    ap.add_argument("--pes-pairs", type=int, default=4, help="antithetic pairs (PES)")
    ap.add_argument("--sigma", type=float, default=0.01, help="perturbation std (PES)")
    ap.add_argument("--segment", type=int, default=20, help="steps per truncation (PES)")
    ap.add_argument("--episode", type=int, default=400, help="inner steps per episode (PES)")
    ap.add_argument("--intensive", action="store_true",
                    help="strict-intensive scalar inputs (implies --rms semantics; exact "
                         "replication invariance, assumes extensive f)")
    ap.add_argument("--multitask", action="store_true",
                    help="meta-train over a task distribution (MNIST/Covertype MLPs of varying "
                         "shape and activation, quartic family, varying unroll length); "
                         "Fashion-MNIST stays held out")
    ap.add_argument("--device", default="auto",
                    help="'cpu' never touches the CUDA driver (safe when the GPU is wedged)")
    args = ap.parse_args()
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(0)
    device = torch.device(args.device)
    train_x, train_y, test_x, test_y = load_mnist(os.path.expanduser("~/.cache/mnist"))
    train_x = torch.from_numpy(train_x).to(device)
    train_y = torch.from_numpy(train_y).to(device)
    test_x = torch.from_numpy(test_x).to(device)
    test_y = torch.from_numpy(test_y).to(device)

    unit_kwargs = {"rms_convention": args.rms or args.intensive,
                   "intensive_inputs": args.intensive}
    model = LearnedOptimizer(**unit_kwargs).to(device)
    n_meta_params = sum(p.numel() for p in model.parameters())
    print(f"learned optimizer has {n_meta_params} meta-parameters (independent of problem dim)")
    meta_opt = torch.optim.Adam(model.parameters(), lr=args.meta_lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(meta_opt, T_max=args.meta_steps)

    if args.multitask:
        import random
        rng = random.Random(0)
        cov_x, cov_y = load_covtype(device)

        def sample_problem():
            kind = rng.choices(["mnist", "covtype", "quartic"], weights=[5, 3, 2])[0]
            unroll = rng.randint(10, 30)
            if kind == "quartic":
                problem = QuarticProblem(rng.choice([300, 1000, 3000]),
                                         args.problems, device)
            else:
                hidden = rng.choice([8, 16, 32, (16, 16)])
                act = rng.choice(["relu", "tanh"])
                data = (train_x, train_y) if kind == "mnist" else (cov_x, cov_y)
                kw = {} if kind == "mnist" else {"in_dim": 54, "n_classes": 7}
                problem = MLPProblem(*data, hidden, args.problems, args.batch_size,
                                     device, activation=act, **kw)
            if rng.random() < 0.5:  # Celo2-style reparam augmentation
                problem = ReparamWrapper(problem,
                                         param_scale=10.0 ** rng.uniform(-1, 1),
                                         loss_scale=10.0 ** rng.uniform(-2, 2))
            return problem, unroll
    else:
        def sample_problem():
            return MLPProblem(train_x, train_y, args.hidden, args.problems,
                              args.batch_size, device), args.unroll

    def run_evals():
        model.eval()
        for tag, hid, steps in [("same-dim", args.hidden, args.unroll),
                                ("same-dim-long", args.hidden, 5 * args.unroll),
                                ("4x-wider", 4 * args.hidden, args.unroll)]:
            res = evaluate(model, test_x, test_y, hid, 8, args.batch_size,
                           steps, device, seed=1234)
            msg = "  ".join(f"{k}: loss {v[0][-1]:.3f} acc {v[1]:.3f}"
                            for k, v in res.items())
            print(f"  eval[{tag}] {msg}", flush=True)
        model.train()

    def make_probes():
        """Fixed, seeded problems: checkpoint selection runs the optimizer on these and
        scores mean log(final/init) — the deployed metric, comparable across all of
        training (training-loss EMAs are not: task kinds differ in scale and join at
        different times)."""
        gen = torch.Generator(device=device).manual_seed(999)
        probes = [MLPProblem(train_x, train_y, 16, 4, 64, device, gen)]
        if args.multitask:
            probes.append(MLPProblem(cov_x, cov_y, 16, 4, 64, device, gen,
                                     in_dim=54, n_classes=7))
            probes.append(QuarticProblem(1000, 4, device, gen))
        return [(p, p.init_point(torch.Generator(device=device).manual_seed(999)))
                for p in probes]

    probes = make_probes()

    def probe_score():
        model.eval()
        score = 0.0
        for p, x0 in probes:
            _, traj = model(x0.clone(), p.meta_objective, 20, meta=False)
            score += torch.log(traj[:, -1] / traj[:, 0] + 1e-9).mean().item()
        model.train()
        return score / len(probes)

    if args.trainer == "pes":
        train_pes(model, sample_problem, args, device, evaluate_hook=run_evals)
        print("done")
        return

    best_score = float("inf")
    for step in range(args.meta_steps):
        problem, unroll = sample_problem()
        x0 = problem.init_point()
        _, traj = model(x0, problem.meta_objective, unroll)
        loss = meta_loss(traj)
        meta_opt.zero_grad()
        skipped = ""
        if torch.isfinite(loss):
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # storm-skip: a clipped step in a garbage direction is still a full step
            if torch.isfinite(gnorm) and gnorm < 1e4:
                meta_opt.step()
            else:
                skipped = " [skipped]"
        else:
            gnorm = float("nan")
            skipped = " [skipped]"
        sched.step()

        if step % 100 == 0 and step > 50:
            s = probe_score()
            if s < best_score:
                best_score = s
                torch.save({"state_dict": model.state_dict(),
                            "unit_kwargs": unit_kwargs}, args.save)
                print(f"  probe-score {s:.4f} (new best, saved)", flush=True)

        if step % 25 == 0:
            ratio = (traj[:, -1] / traj[:, 0]).mean().item()
            print(f"meta {step:5d}  meta-loss {loss.item():.4f}  "
                  f"final/init {ratio:.4f}  gnorm {gnorm:.2f}{skipped}", flush=True)

        if step % args.eval_every == 0 or step == args.meta_steps - 1:
            run_evals()

    print("done")


if __name__ == "__main__":
    main()

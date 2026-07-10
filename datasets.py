"""
Dataset zoo for meta-training the learned optimizers.

registry() returns {name: (x, y, in_dim, n_classes)} for every dataset that is
actually present on disk — loaders that fail (missing/partial download) are skipped
with a note, so meta-training uses whatever subset is available.

HELD OUT by convention (never returned by registry): fashion-mnist, pendigits.
"""
import gzip
import os

import numpy as np
import torch

from vectornet_torch import CACHE, load_covtype, load_idx, load_mnist

UCI = os.path.join(CACHE, "uci")


def _csv(path, label_col, label_map=None, skip=0, delim=","):
    rows = np.genfromtxt(path, delimiter=delim, skip_header=skip, dtype=str)
    y_raw = rows[:, label_col]
    x = np.delete(rows, label_col, axis=1).astype(np.float32)
    if label_map == "alpha":
        y = np.array([ord(c) - ord("A") for c in y_raw], dtype=np.int64)
    elif label_map == "unique":
        _, y = np.unique(y_raw, return_inverse=True)
        y = y.astype(np.int64)
    else:
        y = y_raw.astype(np.float32).astype(np.int64)
        y -= y.min()
    mu, sd = x.mean(0), x.std(0) + 1e-8
    return (x - mu) / sd, y


def _tens(device, x, y):
    return (torch.from_numpy(np.ascontiguousarray(x)).to(device),
            torch.from_numpy(np.ascontiguousarray(y)).to(device))


def load_letter(device):
    x, y = _csv(os.path.join(UCI, "letter.data"), 0, "alpha")
    return (*_tens(device, x, y), 16, 26)


def load_pendigits(device):  # HELD OUT — used by eval only
    x, y = _csv(os.path.join(UCI, "pendigits.tra"), -1)
    return (*_tens(device, x, y), 16, 10)


def load_optdigits(device):
    x, y = _csv(os.path.join(UCI, "optdigits.tra"), -1)
    return (*_tens(device, x, y), 64, 10)


def load_satimage(device):
    x, y = _csv(os.path.join(UCI, "satimage.trn"), -1, "unique", delim=" ")
    return (*_tens(device, x, y), 36, 6)


def load_spambase(device):
    x, y = _csv(os.path.join(UCI, "spambase.data"), -1)
    return (*_tens(device, x, y), 57, 2)


def load_magic(device):
    x, y = _csv(os.path.join(UCI, "magic04.data"), -1, "unique")
    return (*_tens(device, x, y), 10, 2)


def load_poker(device):
    x, y = _csv(os.path.join(UCI, "poker.data"), -1)
    return (*_tens(device, x, y), 10, 10)


def load_miniboone(device):
    path = os.path.join(UCI, "miniboone.txt")
    with open(path) as fh:
        n_sig, n_bkg = map(int, fh.readline().split())
        x = np.loadtxt(fh, dtype=np.float32)
    y = np.concatenate([np.zeros(n_sig), np.ones(n_bkg)]).astype(np.int64)
    keep = (np.abs(x) < 1e4).all(axis=1)  # dataset uses -999 sentinels
    x, y = x[keep], y[keep]
    mu, sd = x.mean(0), x.std(0) + 1e-8
    return (*_tens(device, (x - mu) / sd, y), 50, 2)


def load_svhn(device):
    from scipy.io import loadmat
    d = loadmat(os.path.join(CACHE, "svhn", "train_32x32.mat"))
    x = d["X"].transpose(3, 0, 1, 2).reshape(-1, 32 * 32 * 3).astype(np.float32) / 255.0
    y = d["y"].ravel().astype(np.int64) % 10
    return (*_tens(device, x, y), 3072, 10)


def load_kmnist_reg(device):
    from vectornet_torch import load_kmnist
    return (*load_kmnist(device), 784, 10)


def load_qmnist(device):
    d = os.path.join(CACHE, "qmnist")
    x = load_idx(os.path.join(d, "qmnist-train-images-idx3-ubyte.gz"))
    x = x.astype(np.float32) / 255.0
    with gzip.open(os.path.join(d, "qmnist-train-labels-idx2-int.gz"), "rb") as fh:
        import struct
        _, n, c = struct.unpack(">III", fh.read(12))
        y = np.frombuffer(fh.read(), ">i4").reshape(n, c)[:, 0].astype(np.int64)
    return (*_tens(device, x, y), 784, 10)


def load_cifar10_reg(device):
    from vectornet_torch import load_cifar10
    return (*load_cifar10(device), 3072, 10)


def load_cifar100(device):
    import pickle
    import tarfile
    npz = os.path.join(CACHE, "cifar100", "cifar100.npz")
    if not os.path.exists(npz):
        with tarfile.open(os.path.join(CACHE, "cifar100", "cifar-100-python.tar.gz")) as tf:
            d = pickle.load(tf.extractfile("cifar-100-python/train"), encoding="bytes")
        np.savez(npz, x=d[b"data"], y=np.array(d[b"fine_labels"]))
    d = np.load(npz)
    x = d["x"].astype(np.float32) / 255.0
    return (*_tens(device, x, d["y"].astype(np.int64)), 3072, 100)


def load_mnist_reg(device):
    x, y, _, _ = load_mnist(os.path.join(CACHE, "mnist"))
    return (torch.from_numpy(x).to(device), torch.from_numpy(y).to(device), 784, 10)


def load_covtype_reg(device):
    return (*load_covtype(device), 54, 7)


LOADERS = {
    "mnist": load_mnist_reg, "kmnist": load_kmnist_reg, "qmnist": load_qmnist,
    "cifar10": load_cifar10_reg, "cifar100": load_cifar100, "svhn": load_svhn,
    "covtype": load_covtype_reg, "letter": load_letter, "optdigits": load_optdigits,
    "satimage": load_satimage, "spambase": load_spambase, "magic": load_magic,
    "poker": load_poker, "miniboone": load_miniboone,
}


def registry(device, verbose=True):
    out = {}
    for name, fn in LOADERS.items():
        try:
            out[name] = fn(device)
        except Exception as e:
            if verbose:
                print(f"  dataset {name} unavailable: {type(e).__name__}: {e}", flush=True)
    if verbose:
        print(f"dataset registry: {sorted(out)} ({len(out)} available)", flush=True)
    return out


# ------------------------------------------------------------- char-level text

def load_text(name, device):
    """Returns a 1-D LongTensor of char ids and the vocab size."""
    if name == "shakespeare":
        path = os.path.join(CACHE, "lm", "shakespeare.txt")
        with open(path, "rb") as fh:
            raw = fh.read()
    elif name == "text8":
        import zipfile
        path = os.path.join(CACHE, "lm", "text8.zip")
        with zipfile.ZipFile(path) as z:
            raw = z.read("text8")[:5_000_000]
    else:
        raise ValueError(name)
    data = np.frombuffer(raw, dtype=np.uint8)
    vocab, ids = np.unique(data, return_inverse=True)
    return torch.from_numpy(ids.astype(np.int64)).to(device), len(vocab)

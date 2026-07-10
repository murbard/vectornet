# vectornet — dimension-polymorphic learned optimization

Many gradient descent algorithms (SGD, momentum, Nesterov, conjugate gradient, L-BFGS)
can be described succinctly as recurrent networks with two types of neurons — scalars,
and **vectors of symbolic dimension n** — where vectors interact with the scalar
decision-making machinery *only through dot products*, and are updated only by *linear
combinations*. Functions built from these primitives are exactly the O(n)-equivariant
ones (classical invariant theory; cf. Villar et al. 2021, "Scalars are universal"), the
same symmetry class as the classical algorithms above. Because no learned weight depends
on n, an optimizer meta-trained on small problems applies verbatim to problems of any
dimension.

The original `vectornet.py` (Theano, committed February 2016) predates Andrychowicz et
al.'s "Learning to learn by gradient descent by gradient descent" (June 2016), which
reached dimension-generality differently — per-coordinate weight sharing, which admits
Adam but abandons rotation equivariance.

## Modern port (2026)

- **`vectornet_torch.py`** — the framework in PyTorch, plus meta-training on MNIST and a
  multi-task distribution (MNIST / UCI Covertype MLPs of varying width, depth, and
  activation; the 2015 quartic family; Celo2-style reparameterization augmentation).
  Key conventions, all dimension-polymorphic: intensive units (gradients normalized by
  RMS, dot products averaged over coordinates, log n as a scalar input — `--rms`), a
  strictly-intensive variant with exact replication invariance (`--intensive`), log-space
  meta-loss, keep-best checkpointing.
- **`vectornet_matrix.py`** — the same design one rung up the invariant-theory ladder:
  matrix-valued neurons of symbolic shape (a_l, b_l) per optimizee layer, shared
  meta-weights across layers, primitives {linear combination, triple product X Yᵀ Z,
  intensive trace inner products}. O(a)×O(b)-equivariant; Newton–Schulz (hence Muon)
  is a short program in the primitive set.
- **`transfer_eval.py`** — zero-shot transfer benchmark: Fashion-MNIST (held out),
  Covertype, quartic; baselines SGD / Adam / Muon (Moonlight scaling) / L-BFGS, each
  with a tuned learning-rate grid.
- **`test_equivariance.py`** — property tests for the design invariants (permutation and
  orthogonal equivariance, replication consistency).
- **`RESEARCH_LOG.md`** — running experimental record.

### Headline results so far (RTX 3060, ~35 min of meta-training)

Meta-trained on 20-step training runs of a 784-16-10 MLP on MNIST minibatches, the
19k-meta-parameter optimizer beats tuned Adam on its home task (0.0006 vs 0.037 final
minibatch NLL), transfers unchanged to a 4×-wider MLP (~0 loss vs tuned SGD 0.315), and
zero-shot ties tuned Adam on Covertype at 100 steps (0.0009 vs 0.0008) — a different
domain, input dimension, and parameter count than anything it was meta-trained on.

Run: `python vectornet_torch.py --rms --multitask` then `python transfer_eval.py`.

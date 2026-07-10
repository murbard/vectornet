# HANDOVER — vectornet session state (updated 2026-07-09, task-zoo era)

Read RESEARCH_LOG.md for the experimental record (iterations 1-20+). This file is the
operational quick-start. Everything runs from repo root, venv `.venv` (uv-managed).
NOTHING is committed to git (one 2016 commit); ask Arthur before committing.

## Goal
A mathematically principled learned optimizer (equivariant: scalar + vector + matrix
neurons of symbolic dimensions, interacting only through invariant contractions) that
beats every tuned optimizer (SGD/Adam/Muon/L-BFGS/Prodigy) across NN training tasks.
Meta-train small, transfer everywhere. Current focus: the MATRIX line (vector line is
mature; see scoreboard in RESEARCH_LOG iter 14).

## File map
- `vectornet_torch.py` — vector framework + trainers (BPTT/PES) + MLPProblem (now with
  project_to + activation), TeacherStudentProblem, ReparamWrapper, QuarticProblem,
  run_baseline (sgd/adam/muon-Moonlight/lbfgs/prodigy), dataset loaders (mnist/kmnist/
  fashion/cifar10/covtype).
- `datasets.py` — task-zoo registry: 14 loaders (11 live; cifar10/100+qmnist join when
  downloads land), graceful skip; char corpora (shakespeare, text8).
- `transformer_problem.py` — batched nanoGPT char-LM optimizee; bias-free + param-free
  RMSNorm => ALL parameters matrices. has_biases=False protocol.
- `vectornet_matrix.py` — matrix-neuron optimizer. Flags: time_inputs (log(1+t), t/T),
  cross_layer (adjacent-layer gradient-covariance couplings; Shampoo/K-FAC span).
  lr_scale threading for the tuned-lambda eval row. Checkpoints carry unit_kwargs.
- `train_matrix.py` — matrix trainer. PES ONLY (BPTT provably untrainable here: 3 failed
  runs, gnorm 1e6-1e13). --multitask = full zoo; probe-based keep-best; mixed episode
  lengths {80,400}. Flags: --time-inputs --cross-layer.
- `transfer_eval_matrix.py` — the deliverable table: held-out tasks x {learned zero-shot,
  learned tuned-lambda (ONE scalar, grid), muon/adam/sgd lr-tuned, prodigy lr=1}.
- `test_equivariance.py` — 18 property tests, all PASS (run after any Unit change).
- HELD OUT everywhere: fashion-mnist, pendigits, text8, MLP shapes h64 & (32,32).

## Current runs / state (as of writing)
- v4 RUNNING: `train_matrix.py --trainer pes --multitask --meta-steps 30000 --meta-lr
  3e-4 --eval-every 2500 --save learned_matrix4.pt > train_matrix4.log` (task zoo,
  NO time/cross flags — isolates the zoo variable vs v2).
- CIFAR resume loop RUNNING (Toronto server drops every ~15MB; loop re-curls with -C -).
- Checkpoints: learned_matrix2.pt (MNIST-only distro; MATCHES tuned Muon on 100-step
  MNIST-MLP probe 0.0004/100%; beats tuned SGD zero-shot on fashion-h64@100),
  learned_matrix4.pt (v4 probe-best, accumulating). Vector line: learned_opt_rms.pt
  (best single-task), learned_opt_pes.pt (interference-free multitask).

## Next queue
1. v4 finishes -> `transfer_eval_matrix.py --checkpoint learned_matrix4.pt` (full table
   incl. nanoGPT-on-text8 vs tuned Muon; fairness rows per Arthur).
2. v5: add `--time-inputs --cross-layer` (bundled; note the confound), 30k+ steps.
3. Scale meta-training (curves never flattened): 50k-100k steps / weekend run.
4. Schedule extraction plot per checkpoint (see RESEARCH_LOG iter 20 snippet) — shows
   the learned closed-loop schedule; nice artifact for writeups.
5. Ideas parked: strict-intensive matrix variant; Pareto checkpoint selection;
   deeper/conv optimizees (needs conv-shaped neurons or im2col); PES sigma anneal.

## Gotchas
- GPU wedge history + CUDA_VISIBLE_DEVICES="" trick: RESEARCH_LOG iters 1-9. If nvidia
  hangs: it's the driver, reboot; CPU path works with --device cpu (never queries CUDA).
- uv only for packages; venvs on /media/bigdisk are full copies (cache on /home).
- transformer d_model must be divisible by n_heads (sampler configs already are).
- Muon baseline = Moonlight scaling 0.2*sqrt(max(a,b)); quartic gets no Muon (no shapes).

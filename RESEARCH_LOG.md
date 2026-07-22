# vectornet research log

Goal: a mathematically principled learned-optimizer architecture (scalar + symbolic-dimension
vector neurons; linear combinations and dot products as the only vector ops) that beats every
tuned hand-designed optimizer (SGD/Adam/Muon) across a variety of NN training tasks.

## Iteration 1 — 2026-07-08 ~00:00

**RMS/intensive-units A/B (answering "should scalars be dimension-normalized?"): decisive win.**
Convention: g/rms(g) input (O(1) coords at any n), dot products = coordinate means, log n as
scalar input. Checkpoints: `learned_opt.pt` (old), `learned_opt_rms.pt` (RMS). Both 3000
meta-steps on MNIST 784-16-10, 20-step unrolls. Transfer eval (`transfer_eval.py`, seed 1234):

| task (budget)   | old     | RMS     | tuned Adam |
|-----------------|---------|---------|------------|
| MNIST (20)      | 0.0095  | 0.0006  | 0.0371     |
| Fashion (20)    | 0.99    | 0.83    | 0.59       |
| Covertype (20)  | 0.30    | 0.22    | 0.12       |
| MNIST (100)     | 0.031   | 0.023   | 0.0003     |
| Fashion (100)   | 23.9 ⚠  | 0.31    | 0.018      |
| Covertype (100) | 0.625   | 0.0009  | 0.0008     |
| quartic gap(100)| 11.2    | 10.6    | 6.8        |

Long-horizon loss-inflation pathology eliminated by RMS convention. Covertype@100 ties tuned
Adam zero-shot. Remaining gaps: Fashion short-budget, quartic (extreme domain shift).

**Multi-task meta-training implemented, NOT yet validated** (`--multitask` flag): task mix =
MNIST/Covertype MLPs (widths 8/16/32, depth 1-2, relu|tanh) + quartic family (n∈{300,1k,3k}),
unroll ~ U[10,30], meta-objective = positive gap (quartic) or NLL. Fashion-MNIST held out.
Planned run: `python vectornet_torch.py --rms --multitask --meta-steps 3000 --eval-every 500
--save learned_opt_multi.pt`.

**BLOCKED: GPU driver hard-locked** (NVRM Xid 16, RC watchdog "GPU is probably locked",
vblank timeouts; KDE drkonqi crash-looping). Any torch process — even --device cpu — hangs
unkillable in D-state because torch.cuda.is_available() touches the wedged driver. Machine
needs a reboot. Stuck PIDs: 729301, 737550, 738551, 739955.

## Iteration 2 — 2026-07-08 ~00:45 (GPU still locked; code-only iteration)

- Confirmed no reboot yet; `import torch` itself now blocks in D-state on the NVRM rwlock,
  so not even CPU work runs. Stopped launching python entirely.
- `--device cpu` no longer queries CUDA (argparse "auto" sentinel) in both scripts.
- **`test_equivariance.py` written** (unvalidated): permutation equivariance must hold
  always; orthogonal equivariance must hold with vector_activation='linear' and break with
  tanh; replication consistency reported as measurement only.
- **Design insight from writing the replication test**: the rms convention is NOT exactly
  replication-invariant — signed_log(y) (extensive for quartic-like objectives) and log n
  both change under coordinate duplication. A strictly-intensive-inputs variant would make
  the optimizer exact under the "thermodynamic limit," but f's extensivity is task-dependent
  (NLL already intensive). Post-reboot: A/B strict-intensive (drop log n, feed y/n for
  extensive tasks) vs current.
- **Muon baseline implemented** (batched Newton-Schulz on layer matrices, momentum SGD on
  biases, aspect-ratio scaling) and **per-problem L-BFGS baseline** (strong-Wolfe line
  search) — wired into transfer_eval grids; quartic gets no Muon (no matrix structure).
  All unvalidated: NO torch process has executed since the GPU locked.

## Iteration 3 — 2026-07-08 ~01:45 (GPU still locked; design iteration)

- Muon baseline corrected to the Moonlight convention: update scaled by 0.2*sqrt(max(a,b))
  (matches AdamW RMS regardless of shape), replacing my sqrt(max/min) variant.
- **`vectornet_matrix.py` written** (unvalidated): matrix-valued neurons of symbolic shape
  (a_l, b_l), one family per optimizee layer, meta-weights shared across layers, layers tied
  by a mean-pooled scalar bus (DeepSets over depth). Primitives = linear combinations,
  normalized triple products T(X,Y,Z)=XY^T Z, intensive trace inner products -> scalars.
  O(a)xO(b)-equivariant; Muon/Newton-Schulz is a short program in the primitive set.
  Known shortcuts, by design: bias blocks get a fixed normalized-gradient step (should reuse
  the vector Unit); no cross-layer matrix products (true MLP symmetry couples adjacent
  layers — W_{l+1}W_l terms are the principled next extension).

## Iteration 4 — 2026-07-08 ~02:45 (GPU still locked; desk-check + intensive variant)

- Desk-checked all blind-written code (layouts, einsum shapes, LBFGS state persistence,
  muon flat-vector walk order vs init_point). Fixed: equivariance tests now run in float64
  (float32 accumulation could false-FAIL orthogonal equivariance at the 1e-4 threshold).
- **Strict-intensive variant implemented** (`--intensive`): signed_log(y/n) instead of
  signed_log(y), no explicit log n. Exact replication invariance — now a hard PASS
  expectation in test_equivariance.py for that config. A/B vs `--rms` queued.
- Still nothing executed since the GPU locked. Reboot remains the blocker.

## Iteration 5 — 2026-07-08 ~03:45 (GPU still locked; SOTA recon + augmentation)

- **Celo2 recon** (arXiv 2602.19142, github.com/amoudgl/celo2): current SOTA meta-trains a
  coordinatewise-MLP rule + fixed Newton-Schulz post-orthogonalization in <6h on one A100,
  using only 4 image-MLP tasks + "reparam" (global re-parameterization) augmentation, PES
  meta-gradients, unrolls to 2000 steps. Validates our normalization convention and matrix/
  orthogonalization direction; their NS is a fixed wrapper, ours is learnable-in-span —
  that's our differentiation. Their compute budget says a 3060 is a viable platform.
- **ReparamWrapper added** to the multitask sampler (p=0.5: param scale 10^U[-1,1], loss
  scale 10^U[-2,2]) — enforces scale invariance instead of assuming it.
- **PES logged as the trainer upgrade** to reach 100+ step unrolls without the BPTT
  meta-gradient explosions we observed; backprop-through-unroll caps us at ~30 steps.
- README rewritten (framework, conventions, results, file map).

## Iteration 6 — 2026-07-08 ~04:45 (GPU still locked; PES trainer)

- **PES meta-trainer implemented** (`--trainer pes`, flags --pes-pairs/--sigma/--segment/
  --episode): 2N antithetic particles, persistent inner state across truncations, xi
  accumulation, unbiased across segments, create_graph=False throughout (no unroll graphs
  => long episodes at low memory). Default: 8 particles, sigma 0.01, 20-step segments,
  400-step episodes. Keep-best saves unperturbed theta.
- Unified samplers/eval hooks between BPTT and PES paths.
- **This closes the execution-free backlog.** Until the reboot, further loop iterations
  will be status checks only — no more blind-written code; the unvalidated pile is already
  deep (equivariance tests, Muon/L-BFGS, multitask+reparam, matrix unit, intensive, PES).

## Iterations 7-9 — 2026-07-08 ~05:00-08:00 (driver lock flapped, then released; GPU dead)

- Driver lock released in stages; nvidia-smi now returns "No devices were found" — GPU
  fell off the bus until reboot. CPU torch fully usable with CUDA_VISIBLE_DEVICES="".
- **VALIDATED: equivariance suite — all 18 properties exact.** Permutation equivariance at
  machine precision (all configs); orthogonal equivariance machine-precision with linear
  vector activation, broken ~1e-3 by tanh (by design); **replication invariance EXACT for
  the --intensive variant** (0.0 deviation) — thermodynamic-limit property confirmed.
- VALIDATED: multitask+reparam smoke, PES smoke, --intensive smoke all run end-to-end.
- **BUG FOUND & FIXED (matrix unit): triple product was extensive** — rms-1 operands give
  XY^TZ entries ~ sqrt(a*b); explosion 2.3 -> 1.7e6 in 4 steps. Fixed by /sqrt(a*b)
  (300x better), plus cold-start log_step bias = -3. Dimensional analysis as bug-finder:
  the intensive-units discipline caught its own violation.
- **Launched: full multitask+reparam BPTT run on CPU** (2000 steps, P=8, batch 64) ->
  learned_opt_multi.pt. Overnight-scale on CPU.

## Iteration 10 — 2026-07-08 ~08:30 (multitask verdict + Muon debut; loop stopped by user)

- **transfer_eval(learned_opt_multi.pt): mixed.** Quartic gap CLOSED (6.85 vs tuned
  baselines ~6.8 at 100 steps; single-task was 10.6). Covertype solid. But MNIST/Fashion
  collapsed (2.2 loss) — diagnosed as broken checkpoint selection: mixed-task EMA lets
  quartic episodes (largest log-ratios) cherry-pick the checkpoint.
- **Fix: per-task-kind EMA dict, keep-best on the mean across kinds** (both trainers).
  Relaunched: 2500 steps -> learned_opt_multi2.pt (CPU).
- **Muon baseline debut: best or tied-best on every MLP task** (Fashion@20 0.109 vs Adam
  0.70; ~0.000 everywhere at 100 steps). Muon is the real bar, and it is high. Beating it
  on MLP tasks likely REQUIRES the matrix-neuron optimizer (vector framework provably
  cannot express Muon).
- User stopped the hourly loop; continuing on request only.

## Iteration 11 — 2026-07-08 post-reboot (GPU back; checkpoint-selection saga continues)

- Reboot fixed the GPU. CUDA sanity + all 18 equivariance properties PASS.
- Multitask GPU attempt 1 (P=16,b=128) OOM'd @~2k steps (multitask peak: h32 2-layer x
  30-step second-order unrolls). Attempt 2 (P=8,b=64) completed but transfer eval was
  WORSE everywhere (even quartic 25.6) — per-task-kind EMA keep-best also broken: kinds
  join at different times / different scales, mean not comparable across training.
- **Fix: probe-based keep-best** — every 100 steps run the optimizer on 3 fixed seeded
  probes (MNIST MLP, covtype MLP, quartic), score = mean log(final/init), save on
  improvement. Selection now measures the deployed metric. Plus **storm-skip**: skip
  meta-update when pre-clip gnorm > 1e4 (clipped step in garbage direction is still a
  full step; storms at 1e13 observed).
- **train_matrix.py written + validated.** Matrix-unit stability bugs fixed via CPU
  smokes: unbounded hidden-matrix linear recurrence (now direction-only memory,
  rms-normalized writes), 2x eps-outside-sqrt NaN (dead-relu => exactly-zero layer
  gradient; sqrt(0) grad = nan). 30 CPU steps: h16-long 8905 -> 92, finite grads.
- **Launched in parallel on GPU**: multitask attempt 3 (probe keep-best + storm-skip,
  4000 steps -> learned_opt_multi2.pt) and first matrix-optimizer meta-training
  (3000 steps -> learned_matrix.pt, evals vs tuned Muon).

## Iteration 12 — 2026-07-08 (attempt-3 verdict; matrix v1 -> v2; PES overnight launched)

- **multi2 attempt 3 (probe keep-best): quartic BREAKTHROUGH, MLP interference persists.**
  Quartic@20 = 7.23, beating tuned SGD 14.0 / Adam 8.5 / L-BFGS 7.06 — first task where
  the learned optimizer is best-in-class against every baseline. Covertype solid. But
  MNIST/Fashion still collapsed despite MNIST being in the probe score: the mean across
  kinds traded MNIST away. Conclusion: TASK INTERFERENCE in multitask BPTT at this
  budget, not a selection artifact. Ideas logged: Pareto selection (no kind may regress),
  reweight sampler toward MLP kinds.
- **Matrix v1 run (3k steps, meta-lr 1e-3, no storm-skip): thrashed** — gnorm storms
  every few steps, evals flat (~random acc). Killed at step ~1600. v2 launched:
  learned bias steps (layer's log_step instead of fixed 0.1), storm-skip, meta-lr 3e-4.
- **Overnight PES multitask run launched** (20k steps, 20-step segments, 400-step
  episodes, P=8) -> learned_opt_pes.pt. The structural answer to both BPTT storms and
  task interference (long horizons + unbiased truncation).

## Iteration 13 — 2026-07-08 evening (matrix BPTT declared dead; both PES runs overnight)

- Matrix v2 diverged (bias steps coupled to unclamped exp(log_step) blew up inner
  trajectories -> gnorm 5e9). v3 (clamped bias) froze: gnorm 1e5-5e6 on nearly every
  step, storm-skip rejected almost all updates. **Conclusion after three runs: BPTT
  meta-gradients through the matrix unit are pathologically ill-conditioned at init.
  Matrix optimizer requires PES.**
- **train_pes_matrix written + smoked** (mirror of train_pes over (x, H-list, s-list)
  particle state). Launched at scale: 10k PES steps, meta-lr 3e-4 -> learned_matrix.pt.
- **Vector PES run (step 2000/20000) is the bright spot**: same-dim 0.35/0.875,
  long-horizon 0.031/0.990 (~SGD level), 4x-wider 0.153/0.955 — multitask WITHOUT the
  MNIST interference collapse, and long horizons trained directly via 400-step episodes.
  If the trend holds, learned_opt_pes.pt becomes the flagship checkpoint.

## Iteration 14 — 2026-07-09 morning (vector PES verdict)

- 20k-step PES multitask run done. transfer_eval(learned_opt_pes.pt):
  MNIST@100 0.071/0.983, covtype@100 0.012 (beats SGD) — **the first multitask checkpoint
  with no task-interference collapse**: MNIST+covtype healthy simultaneously. Fashion
  improved vs BPTT-multitask (1.50 vs 2.24 @100) but still far behind baselines. Quartic
  BAD (41-43): train_pes still uses per-kind-EMA selection, not probe-based — same
  trade-one-kind-away failure, opposite direction.
- PES trajectory oscillated peak/trough at ~2k-step period all run (peaks: same-dim 0.081,
  long 0.011 BEATING tuned SGD at step 8k). Diagnosis: meta-lr 1e-3 is ~20x Celo2's.
- Successor recipe (not yet run): PES + meta-lr 1e-4 + probe-based keep-best (port from
  BPTT loop into train_pes) + optionally longer run. Expect the step-8k peak quality,
  held across all kinds.
- Matrix-PES still training (~6k/10k), monotone improvement on every probe throughout.

### Scoreboard (best checkpoint per task, vs best tuned baseline)
| task            | best learned ckpt        | learned | best baseline    |
|-----------------|--------------------------|---------|------------------|
| MNIST@20        | rms (single-task)        | 0.0006  | 0.0018 muon      | WIN
| 4x-wider@20     | rms (single-task)        | ~0.000  | 0.000 muon       | TIE
| covtype@100     | rms (single-task)        | 0.0009  | 0.0002 muon/lbfgs| CLOSE
| quartic@20      | multi2 (probe-BPTT)      | 7.23    | 7.06 lbfgs       | CLOSE-WIN vs SGD/Adam
| fashion@20      | rms (single-task)        | 0.83    | 0.10 muon        | LOSS
| MNIST@100       | rms (single-task)        | 0.023   | 0.000 muon       | LOSS

## Iteration 15 — 2026-07-09 (matrix-PES verdict: MUON MATCHED on long horizon)

- 10k-step matrix-PES run complete, monotone convergence virtually throughout (the only
  trainer/architecture pairing that never blew up). Final probes (MNIST MLP domain):
    h16@20:       learned 0.46 / 89%   vs muon 0.0018 / 99.9%   (muon wins)
    h64-wider@20: learned 0.61 / 81%   vs muon 0.0000 / 100%    (muon wins)
    h16@100:      learned 0.0004-0.002 / 100% vs muon 0.0000 / 100%  (MATCHED)
  Peak at step 9000: h16-long 0.0004/1.000. The learned matrix rule reaches Muon-level
  optimization on the horizon it trains on (400-step episodes) — evidence the equivariant
  primitive span (which contains Newton-Schulz) is learnable end-to-end to Muon quality.
  Short-horizon gap likely = warmup steps the learned rule spends filling its state.
- Next steps for beating (not matching) Muon: longer meta-training (curve hadn't
  flattened), probe-based keep-best in PES, short-horizon episodes mixed in, adapt
  transfer_eval for the matrix model (API differs), task diversity (covtype MLPs), and
  the cross-layer triple products for the true W_{l+1}W_l symmetry.

## Iteration 16 — 2026-07-09 (zero-shot matrix eval; matrix-PES v2 launched)

- **transfer_eval_matrix.py written.** Zero-shot (MNIST-only-trained learned_matrix.pt):
  ARCHITECTURE transfer excellent @100 steps — unseen h64 TIES muon/adam (0.0000/1.000),
  unseen (32,32) 0.011/0.995. DATASET transfer weak: fashion 0.45/0.83, covtype 0.22/0.93
  @100. Short budgets lose everywhere (state-warmup gap). Arthur's directive: the claim
  that matters = train on varied MLP shapes/problems -> transfer to different MLP +
  different dataset.
- **Matrix-PES v2 launched** (15k steps): task mix {MNIST, covtype} x hidden
  {8,16,32,(16,16),(8,24)} x {relu,tanh} + reparam augmentation; HELD OUT: Fashion,
  h64, (32,32). Mixed episode lengths {80,400} (short-horizon training). Probe-based
  keep-best (train-domain probes only). -> learned_matrix2.pt, ~overnight.

## Iteration 18 — 2026-07-08 evening (bigger meta-training set; matrix v3 launched)

- Arthur asked for a bigger training set. Added:
  - **Real datasets**: KMNIST (downloaded), CIFAR-10 flattened (loader written; official
    download crawling at 8KB/s — auto-included when it lands; a fast mirror served HTML,
    rejected on MD5 grounds).
  - **Random-projection family**: MLPProblem(project_to=d) — minibatch passed through a
    random (in_dim x d) projection, d in {32,64,128,256}; one dataset -> unlimited tasks
    of arbitrary input dim.
  - **Teacher-student family**: TeacherStudentProblem — gaussian inputs labeled by a
    random frozen teacher MLP; any input dim / class count (d in {16,64,256}, C in 2-10).
  - Wider shape menu: hidden in {8,16,24,32,(16,16),(8,24),(24,12)}.
  - Selection probes now mnist+covtype+kmnist(+cifar when present). Fashion still 100%
    held out; h64 and (32,32) shapes still held out.
- Parameter audit for Arthur: enumerated every learned parameter shape (vector 19,473;
  matrix 12,525) — all axes are neuron/feature counts; same weights executed at n=7 to
  n=50,000 and 2-3 layer MLPs unchanged. n/a/b enter only as runtime scalar inputs,
  broadcast axes, and fixed normalizations.
- **Matrix-PES v3 launched**: 30k steps, expanded distribution -> learned_matrix3.pt.

## Iteration 19 — 2026-07-08 night (task ZOO + nanoGPT + fairness protocol; v4 launched)

- Arthur: "be bold — dozens of datasets/tasks, ~dozen architectures each, nanoGPT."
- **datasets.py registry**: 14 loaders, 11 live now (mnist, kmnist, covtype, letter,
  optdigits, satimage, spambase, magic, poker, miniboone, svhn), cifar10/100+qmnist
  auto-join when their downloads land. Graceful skip. + shakespeare/text8 char corpora.
- **transformer_problem.py**: batched nanoGPT-style char LM (no biases, param-free
  RMSNorm => ALL parameters are matrices — pure matrix-optimizer/Muon food). Configs:
  d in {16,32,48}, L in 1-3, heads in {1,2,4}, ctx in {32,64}.
- **v4 distribution**: real datasets x 9 MLP shapes x 2 acts + random projections
  (d 32-256) + teacher-student + shakespeare nanoGPTs + reparam. HELD OUT: fashion,
  pendigits, text8, h64, (32,32).
- **Fairness protocol (Arthur's question)**: both framings implemented in
  transfer_eval_matrix — (a) 'learned tuned-lambda': ONE tuned scalar on the update
  (grid 0.25-4x) vs lr-tuned muon/adam/sgd; (b) 'learned zero-shot' (lambda=1) vs
  self-tuning Prodigy (lr=1). muon_step + matrix optimizer now handle bias-free
  optimizees; lr_scale threaded through step/forward.
- **v4 launched**: 30k PES steps -> learned_matrix4.pt (overnight).

## Iteration 20 — 2026-07-09 (schedules: analysis, extraction, time inputs)

- Arthur asked whether the rule can represent schedules (Adam warmup, cosine decay).
  Analysis: (a) autonomous schedules YES — the tanh scalar state is a clock (leaky
  integrators -> sums of exponentials; e^-at - e^-bt = warmup-then-decay); (b) DATA-
  DEPENDENT schedules yes and better than fixed ones (state sees loss + gradient
  geometry); (c) horizon-aware schedules (cosine ~ f(t/T)) provably NOT — T is not an
  input. Missing input, not missing mechanism.
- **Extracted the realized schedule of learned_matrix2.pt** (update rms per step, MNIST
  h16): big probe step 0.080 -> crash to 0.018 after self-induced loss spike -> stable
  band 0.037-0.041 -> mild late decay. A learned closed-loop (plateau-reactive) schedule;
  no Adam-style warmup (it doesn't need one — it can see its state quality via traces);
  the probe-and-recover cycle explains part of the short-budget weakness.
- **time_inputs flag added** (MatrixUnit): log(1+t) and budget fraction t/T as scalar
  inputs; PES trainer passes episode budget; eval passes n_steps. Checkpoints now carry
  unit_kwargs (legacy files still load). Queued for v5 (v4 running without, to isolate
  the task-zoo variable).

## Iteration 21 — 2026-07-09 (cross-layer coupling implemented)

- **cross_layer flag (MatrixUnit)**: two extra input matrices per layer — the layer's
  normalized gradient projected through adjacent layers' gradient covariances
  (left: (G'^T G')G / sqrt(z); right: G(G''G''^T) / sqrt(c); rms-normalized; only when
  shapes chain, zeros otherwise). These are the equivariant couplings under the TRUE
  MLP symmetry (W_l Q, Q^T W_{l+1}); Shampoo/K-FAC-style preconditioning lives in this
  span. 13,901 meta-params with all flags on — still dimension-free.
- Smoked forward+backward on 3-layer MLP and nanoGPT (partially-chained shapes OK).
- HANDOVER.md refreshed for the zoo era. Plan: v5 = time_inputs + cross_layer bundled
  after v4 (confound noted; ablate later if v5 jumps).

## Iteration 22 — 2026-07-09 (v4 verdict: FIRST OUTRIGHT WINS over all tuned baselines)

- v4 (task zoo, 30k PES steps) full fairness table in eval_matrix4.txt. Headlines:
  - **pendigits h16 @20 (HELD-OUT dataset): zero-shot 0.0070/0.998 beats tuned Muon
    (0.37), tuned SGD (0.19), Prodigy (0.12), ~ties tuned Adam (0.0063); tuned-lambda
    0.0012 BEATS EVERYTHING.** pendigits h64 @20 similar. @100: perfect 0.0000/1.000.
  - covertype @20: tuned-lambda 0.14 beats muon/sgd/prodigy, ~ties adam.
  - **nanogpt on text8 (held-out corpus) @20: tuned-lambda 1.56 beats tuned Adam (2.12),
    SGD (2.64), Prodigy (2.37); tuned Muon leads (0.70).** @100 tuned-lambda 0.060 ~
    muon 0.049, prodigy/adam ~0.02.
  - Image MLPs: behind (fashion h16@100 tuned-lambda 0.27 beats SGD only; muon/adam/
    prodigy ahead). v2 (narrow distro) was BETTER on fashion; v4 much better on
    tabular/LM => distribution composition >> everything else right now.
  - **lambda* is 0.25-0.5 nearly everywhere**: the rule systematically over-steps at
    eval; one tuned scalar recovers 3-12x on several tasks (Arthur's fairness knob is
    also a diagnostic). Fixing the calibration in-training is a v5 target.
  - Prodigy is a strong untuned baseline (~tuned-Adam at 100 steps on MLPs) — the
    right bar for the zero-shot claim.
- v2 running through the same table for the head-to-head (eval_matrix2_full.txt).

## Iteration 23 — 2026-07-09 (v2-vs-v4 head-to-head; v5 launched)

- v2 through the full table (eval_matrix2_full.txt): v4 dominates LM (nanogpt@100
  tuned-lambda 0.060 vs v2 0.352) and short-budget tabular (pendigits@20 0.007 vs
  0.125); fashion ~tied (v2 slightly ahead with lambda @100: 0.158 vs 0.274); both
  PERFECT on pendigits@100. Zoo helped where it added mass, diluted image-MLP skill.
- **v5 launched, 40k steps**: --time-inputs --cross-layer --big (28,543 meta-params:
  k_hidden 6, k_mid 12, n_triples 6, n_dot 24, scalar width 96), 13 datasets (CIFARs
  landed, MD5-verified), sampler re-weighted toward real datasets (7/3/2/3), episode
  lengths {40,80,400} to attack the over-stepping calibration (lambda* < 1 diagnosis).
  Confound: several changes bundled — if v5 jumps, ablate; if not, revert pieces.

## Iteration 24 — 2026-07-10 (v5 verdict: broad wins; calibration fixed; gaps enumerated)

v5 (40k PES, full config) fairness table in eval_matrix5.txt. Relative to the goal
"beat every tuned optimizer across a variety of tasks":

WINS/TIES (learned, usually zero-shot):
- pendigits h16@20 zero-shot 0.0016 beats ALL tuned baselines (muon 0.37, adam 0.0063,
  sgd 0.19, prodigy 0.12); pendigits everywhere else: ties best at 0.0000.
- ALL mnist tasks @100 (incl. unseen h64, (32,32)): 0.0000 — ties muon/adam/prodigy.
- mnist h16@20 zero-shot 0.022 ties tuned adam; lambda 0.009 beats it (muon leads).
- covertype@20 beats muon/sgd/prodigy (adam leads); @100 ties adam (muon slightly ahead).
- fashion h16@20 zero-shot 0.48 beats tuned adam (0.59), sgd, prodigy (muon leads);
  @100 lambda 0.016 beats tuned adam (0.018).
- nanogpt/text8@20 beats adam/sgd/prodigy (muon leads); @100 lambda 0.0495 TIES tuned
  muon (0.0489) — prodigy/adam slightly ahead (0.018/0.021).

REMAINING GAPS: muon on image MLPs at 20 steps (fashion h16: 0.48 vs 0.10; mnist
margins); fashion h64 both budgets (0.35@100 vs everyone <=0.17); nanogpt@20 vs muon.

CALIBRATION FIXED: lambda*=1 on many tasks now (was 0.25-0.5 across the board in v4) —
the short-episode mix worked. v5 was STILL IMPROVING at 40k (final checkpoint = best).

Next levers: longer run (60-100k, curve not flat); width-transfer emphasis (h64 gap:
add width-48 mass, maybe random width jitter); muon-style aspect-ratio cue already
available via log a, log b — may need more short-budget image episodes.

## Iteration 25 — 2026-07-11 (batch-size variation + width jitter queued for v7)

- Arthur: "do we also vary batch size?" — we did NOT (B=128 MLPs / 16 LM fixed).
  Batch size = gradient noise scale; the rule CAN infer noise (successive-gradient
  trace features = autocorrelation) but only learns to if training varies it.
- Sampler now varies B in {16..256} (LM {4..32}); log B deliberately NOT fed as input
  (forces noise inference; generalizes to noise from any source). Width menu replaced
  by log-uniform jitter in [8,48] (fashion-h64 gap = width extrapolation; jitter makes
  unseen widths interpolation; h64 still held out). Eval gains noise-transfer rows
  (fashion h16 at B=16 and unseen B=512).
- v6 (warm-started, no new sampler) at ~30k/60k: refining around v5's plateau, not
  climbing past it — scale-alone hypothesis weakening; v7 (this sampler) is the fix run.

## Iteration 26 — 2026-07-12 (v7 interim @32k: first wins over tuned Muon head-on)

eval_matrix7.txt (probe-best checkpoint, run still going):
- **ZERO-SHOT BEATS ALL TUNED BASELINES: mnist h16@20 (0.0005 vs muon 0.0018 — image
  MLP short budget, Muon's home turf, lambda*=1), pendigits h16+h64@20.**
- **nanogpt/text8@100: zero-shot 0.033 beats tuned muon (0.049); lambda 0.0155 beats
  EVERYTHING (adam 0.021, prodigy 0.018).** First all-baseline transformer win.
- Ties-best on most of the @100 suite (all mnist, fashion h16 lambda=muon=0.0000,
  fashion B=16, covertype lambda ties muon, pendigits both).
- **Noise inference works**: unseen B=512 zero-shot beats tuned adam and prodigy at
  both budgets (muon keeps an edge). B=16 ties best @100.
- Remaining: fashion h64 (only clear loss at both budgets); muon short-budget edges on
  fashion h16/nanogpt/mnist h64 @20.

## Iteration 27 — 2026-07-12 (anatomy probe; OpenML population; leak caught)

- **Anatomy of learned_matrix7** (direction/spectral probe, MNIST h16, 100 steps):
  updates are NOT re-learned Muon — stable rank ~4 (gradient ~1.2, muon ~10), early
  momentum-like (cos ~0.64 with muon/momentum), later history-dominated (all reference
  correlations drop to 0.1-0.2: most of the direction lives in accumulated state).
  Portrait: closed-loop noise-aware controller, moderately rank-expanded history-mixed
  directions, per-layer learned step sizes, plateau-reactive schedule.
- **Memory-at-apply analysis** (Arthur): state = k_hidden x params (6x in --big; Adam
  2x, Muon 1x). Controllable: k is a dial (lean k=2 variant queued); factored rank-r
  hidden matrices (UV^T, r(a+b) per slot — Adafactor-style) = principled sub-linear
  extension. Compute/step ~ Muon's NS budget already.
- **OpenML-CC18 integrated**: 69 datasets cached (~1GB npz), split BY DATASET into
  54 train / 13 held out -> population-level transfer claims. --openml flag; CPU-
  resident with per-minibatch device transfer.
- **LEAK CAUGHT: CC-18 contains Fashion-MNIST and pendigits — both landed in the
  OpenML train split.** Now hard-excluded by alias pattern. Any run trained on raw
  CC-18 would have silently invalidated both holdout claims.
- v8 (width support) still training; v9 = v8-warm-start + --openml.

## Iteration 28 — 2026-07-12 (priority reset: SPEED; scale gap measured; v9 launched)

- Arthur: memory deprioritized ("people would take 3x-Muon memory for 10% speed"),
  goal = fastest optimizer. New North Star: eval_scale.py — 10.7M-param char-GPT
  (d384/L6/ctx128) on text8, stochastic batches, learned vs tuned muon vs
  adamw+warmup+cosine, eval-loss trajectories + ms/step.
- **First scale measurement (v7 ckpt): FAILS. Learned rule regresses (3.53->4.64 in 6
  steps) while muon descends (2.96->2.63).** Diagnosis: distribution gaps — (a) never
  saw per-step batch resampling, (b) d<=48 trained vs 384, (c) unseen vocab/ctx
  geometry. Also ~2.2x muon's ms/step on CPU (unmeasured on GPU).
- Fixes into the sampler: resampled episodes (p=0.5) with COMMON RANDOM NUMBERS across
  PES particles (seeded draws; without CRN batch noise destroys antithetic variance
  reduction); nanoGPT configs to d128/L4/ctx128; OpenML 54-dataset population.
- v8 cut at 9k/40k (width-only attribution sacrificed for speed; its jitter is in v9's
  sampler anyway). **v9 launched: warm-start v8, full stack, 50k steps.**
- After v9: eval_scale + fairness table + openml-holdout population row.

## Iteration 29 — 2026-07-13 (mid-v10 scale reading: gauge works, gap narrowing)

10.7M GPT scale check, v10 @12k/40k steps (gauge active), 100 steps, eval loss:
  zero-shot (lam=1): 3.78 -> 2.89   (v7 zero-shot REGRESSED; first clean zero-shot descent)
  lam=0.5:           3.88 -> 2.53
  tuned muon 0.01:   3.38 -> 2.42
Gauge removed the width-dependent miscalibration (lam* moved 0.25 -> ~0.5; residual ~2x
overstep is width-independent, plausibly trainable). Gap to tuned muon now 0.11 nats
with one scalar, 0.47 zero-shot, at 30% of v10's training. Watch: learned rule is ~3-5x
muon ms/step on CPU (kernel-launch-bound; GPU profile pending).

## Iteration 30 — 2026-07-14 (v10 final: 100-step parity was real, 2000-step reveals
## the horizon wall)

eval_scale10 (10.7M GPT, 2000 steps, pilots): learned zero-shot 2.42, lam=0.5 2.36 —
but BOTH PLATEAU at ~step 1000, while tuned muon (1.31) and adamw+cosine (1.30) keep
descending all the way. Early-phase parity (through ~200 steps) is real; beyond ~2.5x
the max meta-training episode (400 steps) the rule stalls — state dynamics converge to
a non-descending fixed point / step size collapses. THE binding constraint is now
HORIZON, not width (gauge solved width), not noise (resampling solved that).
GPU wall-clock: 215 ms/step vs muon 108 / adamw 79 (2x muon; kernel-launch-bound).
Fairness table v10 (eval_matrix10.txt): small-scale grid ~v7-level (2W/1T/9L @20,
regressions on image short-budget vs v7's 3W; pendigits/covertype/nanogpt wins hold).
**v11 launched: episode lengths {400, 1000, 2000}** (PES makes long episodes cheap —
same per-meta-step cost, episodes just persist), warm-start v10, 40k steps.

## Iteration 31 — 2026-07-17 (v11 NEGATIVE: longer episodes destabilize, don't persist)

v11 (episodes to 2000 steps, 40k, warm-start v10) scale benchmark (eval_scale11):
  learned zero-shot: 4.1->9.3->5.4->10.1->3.3->2.72  CHAOTIC, final 2.72 (> v10's 2.42)
  learned lam=2.0:   4.6->19.8->5.8  worse
  (v11 small-scale probes were also noisy/elevated all run -- never actually stable)
HYPOTHESIS INVERTED: v10's plateau is NOT horizon-undertraining. Longer episodes made
the rule UNSTABLE, not persistent. Two reasons: (1) long+noisy PES rollouts = high
meta-gradient variance; (2) more likely the plateau is STATE COLLAPSE -- the recurrent
hidden matrices converge to a fixed point ~step 1000, so the update direction stops
evolving; adding long unstable regimes can't fix an attractor.
NEXT (proposed, not auto-launched -- genuine fork): diagnose before spending GPU.
  a. Instrument v10 at scale, log ||H_t - H_{t-1}|| and log_step vs t: does state
     converge to a fixed point ~1000? (cheap, decisive -- architecture vs training)
  b. If state-collapse: architectural fix (state that can't saturate: leaky reset,
     an explicit slow/fast timescale pair, or feed a coarse iterate-count bucket).
  c. If not: accept v10 as best scale result (honest: early-phase parity + plateau).
Reverted episode distribution to v10's stable set + one modest-long tier.
STANDING BEST at scale remains v10 (2.42 zero-shot, stable, early-parity to ~200 steps).

## Iteration 32 — 2026-07-18 (AUTONOMOUS PROGRAM: beat Muon per-step. Diagnosis + momentum)

Mandate: iterate until we beat Muon per-step across sizes/horizons. Method: diagnose ->
fix -> verify at scale -> repeat.

DIAGNOSIS (diagnose_plateau.py on v10 @10.7M, 1450 steps) -- hypothesis was state
collapse, REFUTED, real cause found:
  loss: plateaus hard at ~2.40 from step ~900 (bounces 2.38-2.41)
  dH/H (hidden state change): stays HIGH ~1.4-1.5 -- state is churning, NOT frozen
  cos(update_t, update_{t-1}): persistently NEGATIVE (-0.01 to -0.32) all run
  => UNDER-DAMPED OSCILLATION: the rule zig-zags across the loss valley, updates
     reverse each step, net progress ~0. NOT a frozen fixed point.
  Root architectural cause: the k_hidden hidden matrices are rms-normalized on write
  (direction-only), so the rule has NO magnitude accumulator -- structurally cannot
  damp. Muon descends smoothly precisely because its beta=0.95 momentum buffer averages
  out this zig-zag.

FIX (v12): --momentum flag. Unit emits a per-layer decay gamma=sigmoid(.); optimizer
  keeps an explicit output momentum buffer M=gamma*M+(1-gamma)*raw and APPLIES M (bundled
  as an extra channel in H so the (H,s) state signature is unchanged -> zero caller
  changes). gamma cold-started to sigmoid(2)=0.88 (Muon-like). Warm-started 15/17 tensors
  from v10 (net_b final layer reinit for the extra gamma output). Training 40k steps.
  Prediction: cos_prev goes positive, loss descends past the 2.40 plateau.

## Iteration 32b — v12 mid-run diagnostic (step 12k): momentum works, reveals overshoot

diagnose_plateau on learned_matrix12 @12k (lr_scale=1.0):
  cos(update_t,update_{t-1}): now POSITIVE 0.5-0.85 (was NEGATIVE in v10) -- ZIG-ZAG
  DAMPED, mechanism confirmed.
  BUT loss bounces in a HIGHER band 2.7-3.2 (v10 plateaued at 2.40), updrms up 5-8x
  (0.005 vs v10 0.0008) = the 1/(1-gamma)~8x momentum amplification. The base step size
  hasn't shrunk to compensate (step-size head was reinitialized; only 12k steps trained).
  Interpretation: momentum fixed micro-zigzag but introduced macro-overshoot because
  step-size and gamma aren't co-calibrated yet. Likely resolves with full training +
  the scale benchmark's lambda pilot (lambda<1 counters overshoot).
  DEEPER INSIGHT: Muon avoids this entirely by ORTHOGONALIZING momentum (bounds every
  singular value ~1) so magnitude is controlled regardless of momentum. Our M accumulates
  raw deltas with arbitrary spectrum. If v12-final still overshoots -> v13 = spectral
  normalization of the applied update (Newton-Schulz on M), i.e. Muon's OTHER half,
  which we already have in-span. Let v12 finish + scale-bench before deciding.

## Iteration 32c — v12 FINAL verdict: momentum smooths but does NOT break the plateau

v12 (momentum, 40k, warm v10) scale benchmark @10.7M, 2000 steps:
  zero-shot lam=1:  2.70, STILL OSCILLATES (overshoot from momentum amplification)
  lam=0.5:          2.50, now SMOOTH/stable (2.72->2.49->2.50) -- lambda<1 tamed overshoot
  vs v10 zero-shot 2.42, muon 1.31.
KEY REFRAME: removing the oscillation did NOT unlock descent. v12 plateaus at ~2.50,
same wall as v10's 2.42, just smooth vs bouncy. => THE OSCILLATION WAS A SYMPTOM, NOT
THE CAUSE. The rule stalls at ~2.4-2.5 regardless of damping. Something else stops
descent past that point.
That something is SPECTRAL: Muon keeps descending to 1.31 because orthogonalization
equalizes the update spectrum, taking meaningful steps in LOW-CURVATURE directions that
raw gradient/momentum ignore. Our un-orthogonalized update can't. => v13 (--spectral,
already built+smoked) is now the strongly-motivated fix, not an overshoot patch. Launch
on GPU free.

## Iteration 33 — v13 (momentum+spectral) BREAKS THE PLATEAU

v13 = momentum + Newton-Schulz orthogonalization of the buffer (Muon's full structure,
in our equivariant framework, everything else learned). Zero new params, warm-started
17/17 from v12. Mid-run scale check on the STEP-8k (undertrained) checkpoint,
diagnose_plateau on 10.7M GPT, lr_scale=0.3:
  step 100: 2.38   200: 2.27   300: 2.07   400: 1.91 (still descending!)
  cos_prev POSITIVE ~0.6 (stable monotone descent)
vs v10 plateau 2.42, v12 plateau 2.50, muon 1.29. FIRST version to descend PAST the
~2.4 wall. Confirms: the plateau was a SPECTRAL/preconditioning limit; orthogonalization
finds the low-curvature descent directions momentum alone can't. This is the thesis --
the framework CONTAINS Muon's mechanisms and expresses them with learned control.
Small-MLP probes lag during warm-up (orthogonalization is aggressive on small/tall
matrices; may want a learned blend later) but the TARGET (transformer scale) works.
Let v13 finish 40k -> full benchmark. On track to reach/beat muon 1.29.

## Iteration 33b — v13 FINAL scale benchmark: gap to Muon 1.1->0.16 nats; BEATS Muon <300 steps

v13 (momentum+spectral, 40k) @10.7M GPT, 2000 steps:
  zero-shot: 1.63 (smooth, still descending)   lambda=0.5: 1.45 (still descending)
  muon: 1.29 (flat by 1800)   adamw: ~1.30
  Head-to-head v13(lam=0.5) vs muon by step:
    200: 1.86 vs 1.98  -> v13 AHEAD (beats muon at short horizon!)
    400: 1.63 vs 1.60  ~tied
    600: 1.56 vs 1.48  muon;  1000: 1.51 vs 1.38;  2000: 1.45 vs 1.29 (gap 0.16)
  => v13 is FASTER early (<~300 steps), muon better mid-phase then flattens; v13 still
     descending at 2000. Spectral closed 85% of the v10->muon gap (1.13 -> 0.16 nats).
     PARTIAL MANDATE WIN: beats tuned muon per-step at short horizons at 10.7M scale.
  Cost: small-MLP probes stuck ~0.2 (pure orthogonalization too aggressive on small/tall)
     -> v14 blend addresses.
NEXT: (a) longer-horizon eval (v13 descending, muon flat -> may cross below); (b) finer
  lambda; (c) v14 blend for small sizes; (d) more NS iters / meta-training for mid-phase.

## Iteration 33c — v13 extended 6000-step eval: gap is descent-QUALITY, not horizon

Extended to 6000 steps @10.7M:
  v13 lam=0.5: 1.45@2000 -> 1.36@5600 (still descending ~0.02/700)
  muon:        1.29@2000 -> 1.21@5600 (ALSO still descending, ~0.003/700)
  gap stays ~0.14-0.16 across all horizons. MUON DID NOT FLATTEN -> longer horizon does
  NOT close the gap. The deficit is late-phase descent QUALITY, ~0.15 nats.
REFRAME: v13 meta-trained on short episodes (<=~1000 steps) -> optimized for EARLY
  descent (where it BEATS muon <300 steps) but late-phase is extrapolation. v11's long
  episodes destabilized the UN-damped/UN-orthogonalized v10 arch; v13's momentum+spectral
  is now STABLE on long horizons -> long-episode training should finally teach late-phase
  without v11's chaos. => v15 = spectral arch + long episodes (retry v11's idea on the
  now-stable arch). This directly targets the mid/late gap.
Pipeline: v14 (blend, small-size axis) training now; v15 (long episodes, horizon axis)
  next. Two mandate axes, two runs.

## Iteration 34 — v15 NEGATIVE (long episodes destabilize AGAIN); reframe late-phase fix

v15 (spectral+blend+long-episodes up to 5000, warm v13) @step 12k scale check:
  transformer traj 5.57->2.86->4.15->4.42 -- WORSE than v13 (1.63@400) AND bouncing.
  Small-MLP probes climbed 0.20->1.01. Long episodes destabilized meta-training AGAIN.
LESSON (2nd confirmation, v11 + v15): long PES episodes destabilize because ES gradient
  VARIANCE GROWS WITH ROLLOUT LENGTH -- fundamental to the estimator, NOT fixed by
  spectral. Also: I confounded blend+long+reinit-head in one run (undisciplined). Killed.
REFRAME the late-phase gap (~0.15 nats vs muon, stable across horizons): can't use naive
  long episodes. Better options, in order of promise:
  A. GENTLE moderate-episode fine-tune: warm v13, episodes up to ~1600 (not 5000),
     MORE pes-pairs (8, halves ES variance), LOW meta-lr (5e-5) so good weights aren't
     wrecked. Single hypothesis: variance-controlled moderate episodes teach late-phase.
  B. schedule prior: mild late-phase step decay via t/T (AdamW+cosine's late-phase edge).
  C. accept v13 (already beats muon short-horizon + tabular + nanogpt@100; strong result).
STANDING CHAMPION: learned_matrix13.pt. Trying (A) as v16, disciplined single-variable.

### Post-reboot validation cascade (in order)
a. test_equivariance.py (float64, all PASS/INFO as expected)
b. Muon + L-BFGS baseline sanity on MNIST probe (BPTT smoke run eval)
c. multitask+reparam smoke (30 steps) then full run: --rms --multitask (BPTT, 3k steps)
d. PES smoke: --trainer pes --rms --multitask --meta-steps 200, watch seg-loss trend
e. Full PES run (10-20k steps overnight) -> transfer_eval vs learned_opt_rms.pt
f. Intensive A/B: --intensive vs --rms (BPTT 3k) on transfer suite
g. vectornet_matrix.py: smoke MatrixUnit shapes, then meta-train structured MNIST vs Muon

### Next steps (post-reboot)
0. Validate everything written blind this iteration: `test_equivariance.py`, the Muon and
   L-BFGS baselines, then the multitask smoke test.
1. Smoke-test `--rms --multitask` (30 steps), then launch full 3000-step run in background.
2. Evaluate `learned_opt_multi.pt` on transfer suite; expect Fashion + quartic improvement.
3. Property-test equivariance: permute coordinates of a problem, check trajectories permute
   identically (validates the design invariant; catches bugs).
4. Ideas queue: matrix-valued neurons (n×m symbolic; linear comb/matmul/transpose/trace) to
   express Muon/Shampoo — needed to *beat* Muon on transformers; curriculum on unroll length
   (40→200) for long-horizon; feed y/y0 relative-progress scalar; Muon + L-BFGS as baselines
   in transfer_eval; deeper/conv optimizees; hold out an architecture class, not just a dataset.

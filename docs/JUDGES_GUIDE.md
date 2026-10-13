# Reviewer guide

## The question

Can matching processor connectivity to a four-qubit quantum feature map reduce fidelity-kernel cost without changing the logical classifier?

The project implements PS-Q2 on the prescribed 100-point moons dataset. Quantum compute–uncompute circuits supply a fidelity kernel; a classical SVM learns the labels.

## Evidence to inspect first

1. **Correctness.** Open `main.ipynb`, an executed experiment with no error outputs. Exact quantum overlaps validate compiled physical measurement probabilities to better than 1e-9. The test suite also compares the complete optimized unitary across 72 random parameter assignments.
2. **Architecture.** `results/figures/architecture_cost.png` shows the same primary circuit on A and B. A uses depth 89, 62 CX gates, and 12 routing SWAPs; B uses depth 63, 30 CX gates, and no routing SWAPs. The guide's optimization level, seed, native gates, shots, and noise parameters are retained.
3. **Causal controls.** `routing_controls.png` shows linear interactions have equal cost on A and B. A five-qubit cycle control matches B's ring-circuit cost, separating connectivity from the larger qubit budget. The guide's reference B reproduces the ring benefit.
4. **Kernel quality.** `kernel_comparison.png` and `kernel_diagnostics.csv` show lower raw kernel distortion on B. Full measured outcome histograms, exact noise expectations, raw matrices, and actual PSD-corrected matrices are included.
5. **Classification.** Exact QSVM and RBF both reach 100% on the same held-out 20 points. The official single-seed noisy PSD run reaches A=90%, B=100%. Over 20 Monte Carlo shot realizations at 4096 shots, means are A=96.25%, B=96.75%. We do not treat the single-run gap as proof of a stable accuracy advantage.
6. **Reproducibility.** Run `python -m pytest -q` and `python scripts/verify_submission.py`. The pinned environment, raw dataset, split indices, compiled circuits, counts, probability oracles, model selection scores, predictions, and checksums support independent inspection.

## What makes this comparison useful

The main finding is the measured connectivity-to-resource relationship: B needs 51.6% fewer CX gates and 29.2% less depth for this ring circuit. Raw kernel RMSE falls from about 0.156 to 0.102. The same advantage is absent for a line interaction graph. A smaller processor with a suitable cycle can match B on the ring, so qubit count alone does not explain the result.

The report preserves results that do not improve: known-channel readout correction reduces kernel error but does not improve the official run's classification accuracy. Additional shots reduce sampling variance while the noise-bias floor remains.

## New architecture and compiler evidence

Start with `results/tables/compiler_and_topology_measurements.csv` and the
two figures `topology_grid.png` and `compiler_optimization.png`.

The expanded topology grid contains three A and three B graphs, each tested
with linear, ring, and full K4 interactions under BasicSwap and SABRE.
The five-node tree and nine-node tree also receive full noisy kernel
acquisitions. The nine-node tree costs more CX gates than the original
five-node chain on the frozen ring circuit. Extra qubits alone do not help.

Exact diagonal fusion and native parity synthesis reduce A from 62 to
30 CX gates and depth 89 to 52, with no SWAPs. B falls from 30 to 24 CX
gates and depth 63 to 52. Raw training-kernel RMSE falls from 0.156 to
0.104 on A and from 0.102 to 0.089 on B. Both optimized circuits preserve
the original ideal kernel. A fusion-only SABRE ablation isolates the
contribution of the two compiler changes. See `compiler_method.md` for
the phase equations and explicit nearest-neighbor construction.

Both native-parity classifiers achieve 100% in the official seeded
measurement run. Their conditional shot-resampling means remain close
to the baseline, so the gain claimed is cost and kernel distortion.
The report includes unfavorable controls and all sampled predictions.

The training-only Pareto table evaluates nine interaction/repetition
choices with fold-fitted scaling and C selection. The encoding control
compares repeated coordinates with mean/difference coordinates.
Ten predeclared outer split seeds repeat exact QSVM and classical model
selection independently. These diagnostics retain the original primary
model, and do not choose it from held-out scores.

The secondary cost-aware rule maximizes training CV first, then minimizes
the worst A/B baseline CX count. It selects linear interactions, two
layers, scale 0.25, C=100. Its measured circuits use 22 CX gates and depth
45 on both processors, with 100% seeded held-out accuracy. This changes
the kernel and is reported separately from exact compiler optimization.
Its error is measured against its own exact reference.
Its repeated-shot accuracy mean is 92% on both, below the optimized ring's
96.25%/96.5%. This is a cost-versus-shot-robustness tradeoff, not a uniformly
better classifier. The primary ring comparison is retained.

## Limits stated plainly

This is a synthetic-noise simulator experiment on one small dataset and split. It establishes no quantum speedup, hardware performance, or superiority over RBF. Placement uses a reproducible graph-distance heuristic, not exhaustive circuit-cost optimality. Readout correction assumes the specified channel is exactly known. Raw noisy SVM results are exploratory; the primary noisy classifier uses a consistent positive-eigenspace extension learned from the training kernel.

## Suggested three-minute explanation

- **0:00–0:40:** Show the moons and explain that a quantum circuit supplies pairwise similarities to a classical SVM.
- **0:40–1:10:** Show the feature circuit and all-zero overlap measurement.
- **1:10–2:00:** Show A/B graphs and CX/depth/SWAP results. Explain why a ring routes poorly on a chain.
- **2:00–2:35:** Show controls, kernel error, and shot sensitivity. Separate the strong resource result from uncertain accuracy differences.
- **2:35–3:00:** Show the executed notebook, preserved counts, tests, and limitations.

The complete scientific explanation and primary references are in `report.md`; setup and repository navigation are in `README.md`.

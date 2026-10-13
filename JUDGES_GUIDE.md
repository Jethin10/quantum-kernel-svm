# Reviewer guide

## The question

Can matching processor connectivity to a four-qubit quantum feature map reduce fidelity-kernel cost without changing the logical classifier?

The project implements PS-Q2 on the prescribed 100-point moons dataset. Quantum compute–uncompute circuits supply a fidelity kernel; a classical SVM learns the labels.

## Evidence to inspect first

1. **Correctness.** Open `main.ipynb`, an executed seven-code-cell experiment with no error outputs. Exact quantum overlaps validate compiled physical measurement probabilities to better than 1e-9. The test suite also verifies allowed CX edges and the four-qubit requirement.
2. **Architecture.** `results/figures/architecture_cost.png` shows the same primary circuit on A and B. A uses depth 89, 62 CX gates, and 12 routing SWAPs; B uses depth 63, 30 CX gates, and no routing SWAPs. The guide's optimization level, seed, native gates, shots, and noise parameters are retained.
3. **Causal controls.** `routing_controls.png` shows linear interactions have equal cost on A and B. A five-qubit cycle control matches B's ring-circuit cost, separating connectivity from the larger qubit budget. The guide's reference B reproduces the ring benefit.
4. **Kernel quality.** `kernel_comparison.png` and `kernel_diagnostics.csv` show lower raw kernel distortion on B. Full measured outcome histograms, exact noise expectations, raw matrices, and actual PSD-corrected matrices are included.
5. **Classification.** Exact QSVM and RBF both reach 100% on the same held-out 20 points. The official single-seed noisy PSD run reaches A=90%, B=100%. Over 20 Monte Carlo shot realizations at 4096 shots, means are A=96.25%, B=96.75%. We do not treat the single-run gap as proof of a stable accuracy advantage.
6. **Reproducibility.** Run `python -m pytest -q` and `python scripts/verify_submission.py`. The pinned environment, raw dataset, split indices, compiled circuits, counts, probability oracles, model selection scores, predictions, and checksums support independent inspection.

## What makes this comparison useful

The main finding is the measured connectivity-to-resource relationship: B needs 51.6% fewer CX gates and 29.2% less depth for this ring circuit. Raw kernel RMSE falls from about 0.156 to 0.102. The same advantage is absent for a line interaction graph. A smaller processor with a suitable cycle can match B on the ring, so qubit count alone does not explain the result.

The report preserves results that do not improve: known-channel readout correction reduces kernel error but does not improve the official run's classification accuracy. Additional shots reduce sampling variance while the noise-bias floor remains.

## Limits stated plainly

This is a synthetic-noise simulator experiment on one small dataset and split. It establishes no quantum speedup, hardware performance, or superiority over RBF. Placement uses a reproducible graph-distance heuristic, not exhaustive circuit-cost optimality. Readout correction assumes the specified channel is exactly known. Raw noisy SVM results are exploratory; the primary noisy classifier uses a consistent positive-eigenspace extension learned from the training kernel.

## Suggested three-minute explanation

- **0:00–0:40:** Show the moons and explain that a quantum circuit supplies pairwise similarities to a classical SVM.
- **0:40–1:10:** Show the feature circuit and all-zero overlap measurement.
- **1:10–2:00:** Show A/B graphs and CX/depth/SWAP results. Explain why a ring routes poorly on a chain.
- **2:00–2:35:** Show controls, kernel error, and shot sensitivity. Separate the strong resource result from uncertain accuracy differences.
- **2:35–3:00:** Show the executed notebook, preserved counts, tests, and limitations.

The complete scientific explanation and primary references are in `report.md`; setup and repository navigation are in `README.md`.

"""Generate result-linked documentation from measured artifacts."""


def write_documents(root, manifest, accuracy_rows, resource_rows, diagnostic_rows, sensitivity):
    lookup = {r["model"]: r for r in accuracy_rows}
    resources = {r["processor"]: r for r in resource_rows if r["experiment"] == "primary"}
    diag = {(r["processor"], r["variant"]): r for r in diagnostic_rows}
    A, B = resources["A"], resources["B"]
    cx_saving = 100*(A["cx_count"]-B["cx_count"])/max(A["cx_count"],1)
    depth_saving = 100*(A["depth"]-B["depth"])/max(A["depth"],1)
    spec = manifest["feature_map"]
    official = sensitivity[sensitivity.shots == manifest["execution"]["shots"]].groupby("processor").accuracy.agg(["mean", "std"])
    shot_interpretation = (
        f"Across 20 shot-resampling realizations at the official budget, mean accuracy was "
        f"A={official.loc['A','mean']:.2%} (SD {official.loc['A','std']:.2%}) and "
        f"B={official.loc['B','mean']:.2%} (SD {official.loc['B','std']:.2%}). "
        "The single-seed accuracy gap is therefore not strong evidence of a stable classification advantage. "
        "The gate-cost reduction and lower kernel distortion are the clearer architecture findings."
    )
    summary_table = "| Model | Test accuracy | F1 |\n|---|---:|---:|\n" + "\n".join(
        f"| {r['model']} | {r['accuracy']:.0%} | {r['f1']:.3f} |" for r in accuracy_rows)
    cost_table = "| Measure | A | B |\n|---|---:|---:|\n" + "\n".join(
        f"| {label} | {A[key]} | {B[key]} |" for key,label in [
            ("physical_qubits","Physical qubits"),("depth","Compiled depth"),
            ("cx_count","CX gates per pair"),("one_qubit_gates","1q gates per pair"),
            ("routing_swaps","Inserted routing SWAPs"),("evaluated_pairs","Measured pairs"),
            ("total_shots","Total shots"),("total_cx_executions","CX × shots × pairs")])
    kernel_table = "| Processor | Raw RMSE | Expected-noise RMSE | Raw minimum eigenvalue |\n|---|---:|---:|---:|\n" + "\n".join(
        f"| {n} | {diag[n,'noisy raw']['kernel_rmse']:.6f} | {diag[n,'noise expectation PSD']['raw_kernel_rmse']:.6f} | {diag[n,'noisy raw']['minimum_eigenvalue']:.6f} |"
        for n in ["A","B"])
    readme = f"""# Four-qubit QSVM: connectivity, kernel cost, and noise

**Qiskit Fall Fest 2026 · Phase 1 · PS-Q2**

We estimate a four-qubit fidelity kernel, train a classical SVM, and measure how two participant-defined processor graphs affect routing and kernel quality. Processor A is a five-node chain; Processor B is a nine-node square lattice. The quantum algorithm and synthetic noise rates are identical on both.

## Measured result

The training-selected exact QSVM achieved **{lookup['Exact QSVM']['accuracy']:.0%}** held-out accuracy. The RBF baseline achieved **{lookup['Classical RBF']['accuracy']:.0%}**. At the official {manifest['execution']['shots']} shots, noisy PSD-corrected QSVM accuracy was **{lookup['A noisy PSD']['accuracy']:.0%} on A** and **{lookup['B noisy PSD']['accuracy']:.0%} on B**.

For the same parameterized overlap circuit, B used **{cx_saving:.1f}% fewer CX gates** and **{depth_saving:.1f}% less depth** than A. These are simulator results for the documented synthetic architectures, not measurements of deployed hardware and not a quantum-advantage claim.

{shot_interpretation}

![Processor connectivity](results/figures/architectures.png)
![Architecture cost](results/figures/architecture_cost.png)

## Run from a clean environment

Python **3.12** was used. From the repository root:

```bash
python -m venv .venv
# Windows PowerShell
.venv\\Scripts\\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
python -m src.experiment
```

Open `main.ipynb` using this environment and run all cells. It calls the same implementation as the CLI. The default run measures 4,840 circuits per processor at 4,096 shots, including 80 noisy self-overlap diagnostics. Aer density-matrix simulation avoids a separate noisy statevector evolution for every shot. Batch checkpoints resume interrupted runs; clear `results/.cache/` for a fresh measurement run. Cache keys include circuit, processor, inputs, seed, shots, batch size, and Qiskit/Aer versions. Original simulation timings are preserved and cache reuse is disclosed.

Pinned dependencies and `requirements.lock.txt` capture the tested environment. `results/manifest.json` records versions, settings, selected model, predictions, compilation checks, and timings. `results/checksums.json` fingerprints source/configuration/data/result artifacts.

## Experiment contract

- Dataset: `make_moons(n_samples=100, noise=0.1, random_state=42)`.
- Stratified 80/20 split, seed 42. PS-Q2 does not prescribe this split.
- Two scaled coordinates `(a,b)` populate four qubits as `(a,b,a,b)`.
- Each feature-map layer applies H, RZ(2 theta), and data-dependent RZZ(2 theta_i theta_j), where theta = scale × encoded coordinates.
- Five-fold training-only cross-validation chooses repetitions, scale, and SVM C. Every fold fits its own scaler. Held-out labels never choose the primary model.
- Frozen feature map: ring interactions, **{spec['reps']} repetition(s), scale {spec['scale']}**, SVM **C={manifest['C']}**.
- Fidelity uses global compute–uncompute all-zero probability on four logical qubits.
- Native gates: rz/sx/x/cx. Optimization level 1, transpiler seed 42. Same graph-distance layout rule on both processors, followed by SABRE routing.
- Identical noise: 1q depolarizing parameter 0.001 on rz/sx/x, CX depolarizing parameter 0.005, symmetric independent readout flips 0.01.
- Exact statevectors validate the kernel and physical output mapping. Noiseless shot estimates use exact binomial sampling, explicitly separate from noisy Aer circuit measurements.
- Raw matrices and complete 16-outcome histograms are preserved. PSD correction uses a training-derived eigenspace and a consistent test extension.
- Known-channel readout correction is an optional model-assisted ablation; it does not remove gate errors and uses no claimed hardware calibration.

## Results

{summary_table}

Each test error changes accuracy by five percentage points. Interpret differences with the small test size in mind; confidence intervals and 20 shot-resampling trials are available in the exported tables.

![Decision boundaries](results/figures/decision_boundaries.png)

Boundary plots use exact simulation for illustration. They do not spend hardware shots and do not represent the noisy classifiers' boundaries.

## Repository map

```text
main.ipynb                  Executed end-to-end experiment and interpretation
requirements.txt           Pinned direct dependencies
requirements.lock.txt      Complete tested dependency freeze
data/raw/                  Generated points, labels, split membership
data/processed/            Exact/noisy kernels, states, histograms, probability oracles
processors/                Participant-defined A/B JSON configurations
src/problem.py             Dataset and four-qubit feature map, no cost Hamiltonian needed
src/processors.py          Coupling graphs and synthetic noise
src/routing.py             Layout, transpilation, mapping validation, SWAP accounting
src/kernels.py             Pair scheduling, batching, measurement caches
src/metrics.py             Diagnostics, PSD extension, classifier metrics
src/selection.py           Leakage-free training-only model selection
src/experiment.py          Complete experiment entry point
src/visualization.py       Scientific plots
src/documentation.py       Documentation generated from measured results
tests/                     Mathematical, routing, noise, and artifact checks
results/tables/            CSV and JSON comparison tables
results/figures/           Exported plots
assets/circuits/            Native compiled circuits in QPY and readable text
report.md                  Architecture analysis, controls, limitations, sources
SUBMISSION_CHECKLIST.md    Final submission checks
```

See [report.md](report.md) for the scientific analysis. Phase 1 uses A and B only. The master guide lists submission of the public GitHub URL at https://qffsrmap2026.com/learning/hackathon by **14 October 2026, 23:59 IST**. Publication and portal submission are separate from local experiment completion.
"""
    report = f"""# PS-Q2 scientific report

## Research question

How does physical connectivity affect the cost of estimating the same four-qubit fidelity kernel, and do extra routed gates change kernel quality and classification under matched synthetic noise?

The task is hybrid: quantum circuits supply similarity estimates; scikit-learn trains the SVM classically. With four qubits, classical simulation is inexpensive. We make no speedup, superiority over RBF, or asymptotic quantum-advantage claim.

## Data and model selection

The prescribed 100-point moons dataset is split into 80 training and 20 held-out samples with stratification and seed 42. Training-only five-fold CV evaluates repetitions 1/2, scales 0.25/0.5/1/1.5, and C 0.1/1/10/100. A fresh MinMaxScaler maps fold-training data to [0, pi]; validation points use those limits without clipping. Ties prefer lower CV variation, fewer repetitions, lower scale, then lower C. The selected model has {spec['reps']} repetitions, scale {spec['scale']}, C={manifest['C']}.

The two independent coordinates are repeated as `(a,b,a,b)` across four qubits. Derived coordinates or labels are not added. For each repetition, H on all four qubits precedes RZ(2 theta_i) and RZZ(2 theta_i theta_j) on a four-edge ring. theta_i equals the selected scale times the corresponding coordinate. Both rotations and entangling phases depend on data. A fixed final entangler would cancel from the overlap; this circuit does not rely on such a construction.

Linear and RBF baselines use fold-fitted preprocessing and the same five-fold split and C grid. RBF additionally evaluates gamma in scale/0.1/1/10. This is a reasonable baseline search, not an exhaustive classical optimization budget match. Quantum model selection uses exact simulation and is disclosed separately from the device kernel workload.

## Kernel definition and validation

Let phi(x) = U(x)|0000>. The kernel is K(x,x') = |<phi(x)|phi(x')>|². Preparing U(x) then applying U(x') inverse yields this value as the global all-zero measurement probability. No ancilla or second register is required.

The exact Gram matrix equals Tr(rho_x rho_x'), an inner product on density operators, so it is positive semidefinite. Exact diagonal entries equal one. Four-qubit states have 16 amplitudes; the fidelity kernel's operator feature representation has dimension at most 256, not merely 16 independent real features.

The exact reference's minimum eigenvalue is {manifest['exact_kernel_diagnostics']['minimum_eigenvalue']:.3e}. Compilation checks include self-overlap and distinct pairs on the actual physical output qubits. Maximum errors are A={manifest['compiled_overlap_max_errors']['A']:.3e}, B={manifest['compiled_overlap_max_errors']['B']:.3e}. These checks catch final routing permutations and measurement mapping errors.

![Feature map and overlap estimation](results/figures/feature_map.png)

## Architecture choices and routing controls

A is the five-node path 0–1–2–3–4. B is a participant-designed 3x3 square lattice with nine nodes and 12 undirected edges. All edges support CX in both directions. These choices comply with the guide's topology freedom. B is not described as an actual heavy-hex processor.

We enumerate four-logical-qubit placements using the same label-independent objective: minimize the sum of graph shortest-path distances over the feature-map interaction edges, then lexicographic tie-breaking. We compile the parameterized compute–uncompute template with SABRE routing, optimization level 1, and seed 42, and bind data afterward. Initial layouts are A={A['initial_layout']}, B={B['initial_layout']}. This graph-only placement is reproducible but does not prove globally minimum routed depth or CX count.

SWAP count is captured immediately after the routing pass, before basis translation decomposes SWAPs into CX gates. Final CX count includes all routing and compiler optimizations. Multiplying SWAP count by three is not assumed to equal net CX overhead because adjacent gates may cancel.

The exported architecture table also includes BasicSwap routing, linear/ring/all-to-all logical interactions, repetitions 1/2/3, the guide's reference B graph, and a five-qubit cycle-plus-tail A control. These are structural controls, not additional measured noisy kernel runs. Exact classification for feature-map controls is reported with frozen scale/C and is post-selection exploratory analysis; it does not redefine the primary winner.

{cost_table}

B reduces CX count by {cx_saving:.1f}% and depth by {depth_saving:.1f}% for this primary template. More physical qubits alone do not imply lower cost. A four-cycle is the specific graph property tested. Sparse linear logical interactions may need little or no routing on either architecture; the controls reveal whether the advantage persists under other interaction patterns.

The five-qubit cycle control tests the qubit-budget explanation directly: a five-qubit graph can also host the four-qubit ring without SWAPs. For linear interactions, the primary A and B both use 22 CX gates and depth 45. The guide's reference B also hosts the ring directly. The architecture benefit comes from matching connectivity to the interaction graph, and is not a general statement that nine qubits outperform five.

![Logical interaction and graph controls](results/figures/routing_controls.png)

## Workload and timing definitions

For 80 training samples, the upper triangle including diagonals contains 3,240 pairs. Prediction needs 20×80=1,600 additional pairs, for 4,840 circuits per processor. At {manifest['execution']['shots']} shots this is {manifest['execution']['shots_per_processor']:,} shots per processor. The minimal ideal workload excluding known diagonal values is 4,760 pairs. We deliberately measure 80 diagonal diagnostics rather than quietly forcing noisy self-overlap to one.

The template is compiled before parameter binding. The same structural gate counts therefore apply to all evaluated pairs, including diagonal pairs; we do not recompile diagonal pairs into an empty circuit. Gates that cancel symbolically in the template are optimized normally. This is a clearly specified execution policy, not a lower bound on all possible kernel implementations.

`CX × shots × pairs` is an estimated hardware gate-execution workload. It is not the number of classical density-matrix operations. Aer propagates each noisy circuit density matrix and then samples measurements efficiently. Its idle-qubit truncation is enabled and simulator metadata is recorded. No gate durations, decoherence times, queue model, or hardware calibration exist in this synthetic task; hardware execution time is therefore not inferred from simulator wall time. Original Aer timings and cache reuse are recorded separately.

## Noise, shot error, and readout correction

Both models use exactly the same channels. The guide does not specify a virtual-RZ exemption, so the 0.001 one-qubit depolarizing channel is applied to rz, sx, and x on both. CX uses depolarizing parameter 0.005; readout flips each measured bit independently with probability 0.01. Qiskit's depolarizing parameter describes E(rho)=(1-p)rho+p I/d, not directly an average gate infidelity.

Noisy kernels come from actual Aer measurement histograms at {manifest['execution']['shots']} shots, seeded by 42 plus each batch's start index. Full histograms and pre-readout distributions are saved. Applying the specified readout channel to those distributions yields an exact noisy expectation oracle. This separates gate/readout bias from finite-shot fluctuations.

For an all-zero probability p, the shot estimator standard error is sqrt(p(1-p)/S), at most 1/(2 sqrt(S)). At 4,096 shots the maximum is 0.0078125. More shots reduce variance, but cannot undo gate/readout bias. The sensitivity study uses 20 binomial realizations from the saved noisy expectation for each of 256/1,024/4,096/16,384 shots. These are Monte Carlo resampling trials, not 160 additional device kernel acquisitions. The same random seeds on A/B pair the trials, although probabilities differ.

Readout mitigation inverts the known independent-bit 16×16 channel. This is a model-assisted ablation, with exact synthetic calibration knowledge and no acquisition of calibration circuits. It corrects readout bias, not gate noise. Unclipped estimates are saved; values supplied to the corrected-kernel pipeline are clipped to [0,1]. Any variance increase or lack of accuracy improvement is retained rather than hidden.

## PSD correction and held-out consistency

Independent noisy pair measurements can produce an indefinite Gram matrix, even though the ideal kernel is PSD. We acquire one orientation per training pair and mirror the upper triangle; a reported symmetry error of zero is therefore enforced by assembly, not measured evidence of noisy-circuit symmetry. Noisy compute–uncompute circuits are operational estimators, not guaranteed mixed-state inner products. Noisy diagonal values are measured. The raw SVM is exploratory because an indefinite kernel need not give the usual convex SVM problem.

For the primary noisy classifier, diagonalize the symmetric training matrix and retain eigenvalues greater than 1e-8. Let V contain the retained eigenvectors and L their positive eigenvalues. Training features are V sqrt(L); held-out features are K_test,train V L^(-1/2). Consequently the repaired training kernel is V L V^T and the consistent cross-kernel is K_test,train V V^T. Only training-derived eigenvectors enter the correction; there is no test/test projection or test-label information. Raw matrices remain available for error diagnostics. The correction can change the classifier, so correction norms and retained ranks are exported.

{kernel_table}

## Classification results

{summary_table}

![Classification comparison](results/figures/classification.png)
![Kernel diagnostics](results/figures/kernel_comparison.png)
![Shot sensitivity](results/figures/shot_sensitivity.png)

The 20-sample holdout has five-percentage-point resolution. Wilson intervals are supplied for noisy variants, but this single synthetic split cannot establish general population superiority. Shot-resampling error bars quantify conditional shot variability, not uncertainty across newly generated datasets. The selected C is frozen across exact, sampled, noisy, and mitigated kernels, isolating execution effects; noise-aware retraining or tuning could produce different results.

{shot_interpretation}

## Reproducibility and scope

`python -m src.experiment` executes the complete pipeline. `main.ipynb` provides the same run with figures and explanations. The raw dataset, split indices, scaled points, exact statevectors, raw kernels, full counts, expectation oracles, compiled QPY circuits, model selection scores, predictions, resource tables, checksums, and versions are included. The unit/integration suite tests overlap identities, mapped compilation, valid couplings, PSD extension, shot assembly, readout inversion, and saved-artifact consistency.

Only A and B are implemented, as required for Phase 1. No access to a cloud quantum processor is claimed. Kernel evaluation grows quadratically in training size; four qubits and this toy dataset make no scalability demonstration. The encoding repeats features, the topology is selected for a known ring interaction pattern, and comparison results apply to those disclosed choices. Decision-boundary plots are exact simulations and spend no measurement shots.

## Primary references

- [Master guide](QFF_Phase_1_Master_Guide.pdf), pages 1–3, 5, and 7: task, architecture freedom, noise, repository, deadline, rubric.
- [Havlicek et al., quantum feature-space classifiers](https://arxiv.org/abs/1804.11326).
- [scikit-learn make_moons](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html).
- [scikit-learn precomputed SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html).
- [scikit-learn leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html).
- [Qiskit compute–uncompute fidelity](https://qiskit-community.github.io/qiskit-machine-learning/stubs/qiskit_machine_learning.state_fidelities.ComputeUncompute.html), mathematical method; the implementation here uses Aer directly rather than requiring Qiskit Machine Learning.
- [Qiskit transpiler](https://quantum.cloud.ibm.com/docs/en/api/qiskit/transpiler).
- [Aer density-matrix simulation](https://qiskit.github.io/qiskit-aer/stubs/qiskit_aer.AerSimulator.html).
- [Aer depolarizing channel](https://qiskit.github.io/qiskit-aer/stubs/qiskit_aer.noise.depolarizing_error.html).
"""
    (root / "README.md").write_text(readme, encoding="utf-8")
    (root / "report.md").write_text(report, encoding="utf-8")
    (root / "SUBMISSION_CHECKLIST.md").write_text("""# Phase 1 submission checklist

- [x] Exactly one problem selected: PS-Q2.
- [x] Prescribed dataset and reproducible train/test split.
- [x] Four logical qubits with explicit two-feature encoding.
- [x] Exact overlap reference and compiled measurement-mapping validation.
- [x] Classical precomputed SVM, linear and RBF baselines.
- [x] Five-fold training-only selection with preprocessing fitted per fold.
- [x] Five-qubit A and nine-qubit B defined in participant-authored JSON.
- [x] Same native gates, official noise rates, seed 42, optimization level 1.
- [x] Official 4,096-shot A/B measurement runs and preserved raw histograms.
- [x] Depth, CX, routing SWAP overhead, kernel error, and accuracy tables.
- [x] Architecture controls, shot sensitivity, and disclosed readout correction.
- [x] Required repository structure, notebook, pinned dependencies, report, plots.
- [ ] Verify `python -m pytest -q` and notebook execution on the final files.
- [ ] Publish the repository publicly and verify the notebook/figures render on GitHub.
- [ ] Submit the public GitHub URL at https://qffsrmap2026.com/learning/hackathon.

The guide lists 14 October 2026, 23:59 IST as the deadline. Tests and notebook execution have their own machine-readable verification artifacts; do not tick them based only on file existence. Publication and portal submission require separate confirmation.
""", encoding="utf-8")

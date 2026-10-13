# PS-Q2 technical research

Verified against primary documentation on 13 October 2026. These notes concern the algorithm; the master guide must define what platforms A and B mean.

## Verified building blocks

- `make_moons(n_samples=100, noise=0.1, random_state=42)` creates 100 two-dimensional points with binary class labels. Noise is the standard deviation of added Gaussian noise, not quantum hardware noise. [scikit-learn make_moons](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html)
- A quantum feature map prepares `|phi(x)> = U(x)|0000>`. A fidelity kernel uses `K(x,z)=|<phi(x)|phi(z)>|^2`; Qiskit provides this through `FidelityQuantumKernel`. This remains a classical SVM trained on a quantum-generated similarity matrix. [Qiskit kernel API](https://qiskit-community.github.io/qiskit-machine-learning/stubs/qiskit_machine_learning.kernels.FidelityQuantumKernel.html)
- Compute-uncompute combines the feature-map circuit with the other circuit's inverse. Use global fidelity (`local=False`), since local single-qubit projectors measure a different quantity. Current documentation uses `qiskit_machine_learning.state_fidelities.ComputeUncompute` with a V2 sampler; older examples import from obsolete namespaces. [ComputeUncompute API](https://qiskit-community.github.io/qiskit-machine-learning/stubs/qiskit_machine_learning.state_fidelities.ComputeUncompute.html)
- `SVC(kernel="precomputed")` expects a square training matrix, and prediction expects `(n_test, n_train)`, with columns in the original training order. A test-test matrix is unsuitable. [SVC API](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
- `zz_feature_map(feature_dimension=4, reps=1 or 2, entanglement="linear")` is an available four-qubit feature circuit. Its four input coordinates must be supplied explicitly; `make_moons` supplies only two. [IBM circuit API](https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.circuit.library.zz_feature_map)
- Qiskit's kernel can adjust its feature map when data feature count mismatches. Do not blindly pass two-column data to a four-parameter circuit: document a deterministic 2-to-4 encoding or use a custom four-qubit circuit with two independent data parameters. [Kernel source](https://github.com/qiskit-community/qiskit-machine-learning/blob/main/qiskit_machine_learning/kernels/fidelity_quantum_kernel.py)
- Split before fitting scaling; use training-fitted transformations for validation and test points. Tune SVM C and circuit/encoding choices using training-only validation. [scikit-learn pitfalls](https://scikit-learn.org/stable/common_pitfalls.html)
- Qiskit provides an exact statevector reference kernel and optional shot-noise simulation. This is useful as a correctness oracle before running sampled circuits. [FidelityStatevectorKernel](https://qiskit-community.github.io/qiskit-machine-learning/stubs/qiskit_machine_learning.kernels.FidelityStatevectorKernel.html)
- Havlicek et al. introduced quantum feature-space classifiers and a quantum kernel estimator with classical classifier optimization. This supports the algorithmic framing, not a claim that this four-qubit toy task proves quantum advantage. [Original paper](https://arxiv.org/abs/1804.11326)

## Mathematical deductions and practical recommendations

These calculations follow from the kernel definition; they are not quoted benchmark results.

- Apply `U(x)` and then `U(z).inverse()` to four zero qubits. The probability of measuring `0000` equals the desired overlap. This avoids the ancilla and two four-qubit registers needed for a textbook SWAP test.
- Four qubits have 16 complex amplitudes. The squared-overlap kernel is an inner product between density matrices: `K(x,z)=Tr(rho_x rho_z)`. Its feature space can be represented by 16-by-16 operators; ideal matrices are therefore positive semidefinite. Four qubits are easy to simulate, so experimental accuracy or speed differences here do not establish asymptotic quantum advantage.
- A defensible initial encoding is training-scaled coordinates `(a,b)` repeated as `(a,b,a,b)` for a four-input ZZ map. This reuses the data on all four qubits without inventing independent observations. Compare encoding variants only after the simplest pipeline works. Derived features such as `(a,b,a*b,a*a+b*b)` are another option, but introduce classical feature engineering that should also be disclosed to baselines.
- A fixed data-independent entangler appended only at the very end cancels in `U(z)^dagger U(x)` and cannot change the ideal kernel. Interleave entanglers with data-dependent rotations, or use data-dependent ZZ interactions.
- With S shots and ideal all-zero probability p, the count estimate has standard deviation `sqrt(p(1-p)/S)`, bounded by `1/(2*sqrt(S))`. For 1024 shots the maximum standard error is about 0.0156. Hardware noise adds bias that increasing shots does not remove.
- Full 100-by-100 output has 10,000 entries but only 4,950 distinct off-diagonal overlaps. Diagonals equal 1 ideally. For a proposed 80/20 split, training needs 3,160 off-diagonal evaluations, and test-to-training needs 1,600: 4,760 total. At 1,024 shots this is 4,874,240 circuit shots before validation, repeats, or visualizations.
- Cache repeated pair evaluations, mirror the training triangle, batch execution, and reuse kernels while tuning C. Record both the matrix dimensions and actual evaluated pairs/shots. Each distinct pair is not necessarily a separate submitted job.
- Sampled entries can produce an indefinite training matrix. Diagnose symmetry, diagonal, range, minimum eigenvalue, and difference from the ideal reference before correction. `FidelityQuantumKernel(enforce_psd=True)` supports PSD projection for a self-kernel; disclose it and save the raw matrix. Never separately PSD-project a rectangular test-to-training matrix.
- For A versus B, hold encoding, input order, split, shot count, transpilation settings and precision fixed. Compare logical/compiled depth, one-/two-qubit gates, inserted routing SWAPs, total shots, kernel error versus the exact oracle, and resulting accuracy. Separate submission/queue/wall time from quantum execution time and classical SVM fit time.
- A decision-boundary plot adds `n_grid*n_train` kernel evaluations, easily exceeding the training workload. Start with a coarse grid and report its extra cost separately.

## Suggested sequence

1. Freeze the dataset, split and training-only scaling; plot the original moons.
2. Define the explicit four-qubit encoding and verify one compute-uncompute probability against an exact statevector overlap.
3. Build and validate train/train and test/train kernels; fit the classical precomputed SVM.
4. Compare linear and RBF SVMs with the same held-out test set and training-only tuning.
5. Execute matched sampled/compiled circuits on guide-defined platforms A and B, then report resource/accuracy tradeoffs and shot sensitivity.
6. Preserve seeds, versions, raw/corrected matrices, compiled circuits, counts, timing definitions and readable plots.

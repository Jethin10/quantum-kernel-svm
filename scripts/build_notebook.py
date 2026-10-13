"""Build the narrated notebook, then optionally execute in this Python environment."""
import argparse
import json
from pathlib import Path
import sys
import time
import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]


def build():
    md,code = nbf.v4.new_markdown_cell,nbf.v4.new_code_cell
    cells = [
        md("""# Four-qubit QSVM: connectivity, cost, and noise

**Qiskit Fall Fest 2026 · Phase 1 · PS-Q2**

This notebook runs the prescribed moons classification task and compares the same fidelity kernel on a five-qubit chain and a participant-designed nine-qubit lattice. The quantum circuits estimate similarities; a classical support vector machine learns the classifier. All reported results come from saved measurements or explicitly labeled exact simulation.

Run all cells in a Python environment installed from `requirements.txt`. Execution is deterministic within the pinned stack and batch settings. The first full run measures 4,840 pairs per processor at 4,096 shots, including diagonal diagnostics. Later runs reuse matching batch checkpoints; delete `results/.cache/` to acquire fresh simulated counts. See `report.md` for assumptions, controls, and limitations.
"""),
        code("""from pathlib import Path
import sys
ROOT = Path.cwd().resolve()
assert (ROOT / 'src' / 'experiment.py').exists(), 'Run from the repository root'
print('Python:', sys.version.split()[0])
import numpy as np
import pandas as pd
from IPython.display import display, Image, Markdown
from src.problem import load_dataset, FeatureSpec, feature_map
from src.experiment import run_experiment
"""),
        md("""## 1. Prescribed data and four-qubit encoding

`make_moons(n_samples=100, noise=0.1, random_state=42)` generates two-dimensional points. Dataset noise is separate from processor noise. We use a stratified 80/20 split with seed 42. Training-only min/max scaling produces `(a,b)`; the circuit repeats these coordinates across the qubits as `(a,b,a,b)`.

Each feature-map layer applies Hadamards, data-dependent RZ rotations, and data-dependent ZZ interactions. A ring connects logical pairs 0–1, 1–2, 2–3, and 3–0. Repetitions, angle scale, and SVM C are selected inside the training set using five-fold CV. Preprocessing is fitted separately inside each fold.
"""),
        code("""X, y, train_idx, test_idx = load_dataset()
display(pd.DataFrame({'samples': [len(train_idx), len(test_idx)],
                      'class_0': [(y[train_idx] == 0).sum(), (y[test_idx] == 0).sum()],
                      'class_1': [(y[train_idx] == 1).sum(), (y[test_idx] == 1).sum()]},
                     index=['Training', 'Held-out test']))
"""),
        md(r"""## 2. Fidelity estimation and exact reference

We prepare $|\phi(x)\rangle=U(x)|0000\rangle$ and define

$$K(x,x')=|\langle\phi(x)|\phi(x')\rangle|^2.$$

Applying $U(x)$ followed by $U(x')^\dagger$ makes the global all-zero outcome probability equal to this kernel. The estimator uses exactly four logical qubits and no ancilla. An exact statevector reference verifies the probabilities, diagonal, symmetry, and positive semidefiniteness before noise is considered.

For scikit-learn, the training kernel has shape 80×80, and the prediction kernel has shape 20×80. Test columns must preserve the original training order.
"""),
        md("""## 3. Run the complete controlled experiment

Both processors use native `rz/sx/x/cx` gates, optimization level 1, seed 42, and identical noise rates. A label-independent graph-distance rule chooses placement, followed by SABRE routing. The code counts inserted SWAPs before decomposition, saves native circuits, and verifies the physical measurement mapping against exact overlaps.

The noise model applies depolarizing parameters 0.001 to every listed native single-qubit gate and 0.005 to CX, with symmetric independent readout error 0.01. The virtual-RZ choice is an explicit modeling assumption. Aer density-matrix execution saves the full outcome probabilities and acquires shot histograms.
"""),
        code("manifest = run_experiment(ROOT, shots=4096, repeats=20, batch_size=128)"),
        md("""## 4. Frozen training-only selection and architecture cost

Model selection sees training data only. Structural controls compare linear, ring, and all-to-all interactions; BasicSwap and SABRE; repetition counts; and the guide's reference B graph. They explain where the chosen topology helps without redefining the primary model using held-out accuracy.
"""),
        code("""selection = pd.read_csv(ROOT / 'results/tables/quantum_model_selection.csv')
display(selection.head(8))
display(Markdown(f"Selected feature map: **{manifest['feature_map']}**, **C={manifest['C']}**"))
display(Image(filename=str(ROOT / 'results/figures/feature_map.png')))
resources = pd.read_csv(ROOT / 'results/tables/architecture_resources.csv')
display(resources[resources.experiment == 'primary'][[
    'processor', 'physical_qubits', 'initial_layout', 'depth', 'cx_count',
    'routing_swaps', 'evaluated_pairs', 'total_shots', 'total_cx_executions']])
display(Image(filename=str(ROOT / 'results/figures/architectures.png')))
display(Image(filename=str(ROOT / 'results/figures/architecture_cost.png')))
display(Image(filename=str(ROOT / 'results/figures/routing_controls.png')))
"""),
        md("""## 5. Exact, sampled, noisy, and corrected classification

Linear and RBF SVMs use the same holdout and training-only selection. The exact QSVM establishes algorithmic behavior; ideal binomial samples isolate shot error. Noisy kernels use actual Aer counts.

Raw noisy kernels can be indefinite. The primary noisy classifier retains the positive training eigenspace and extends test similarities through that same eigenspace. Raw matrices are preserved and raw-SVM results are labeled exploratory. Known-channel readout correction is an optional synthetic-model-assisted ablation, not a hardware-calibration result. It cannot undo gate noise.
"""),
        code("""classification = pd.read_csv(ROOT / 'results/tables/classification.csv')
display(classification[['model', 'accuracy', 'balanced_accuracy', 'f1', 'confusion_matrix']])
display(Image(filename=str(ROOT / 'results/figures/classification.png')))
display(Image(filename=str(ROOT / 'results/figures/decision_boundaries.png')))
"""),
        md("""The boundary plot uses the exact reference only. Its grid is evaluated classically without physical shots; it is not a picture of the noisy classifiers. Every error on this 20-point holdout changes accuracy by five percentage points.

## 6. Kernel distortion, spectrum, and shot budgets

Noise expectations are saved separately from observed shot counts. Kernel errors use the raw matrices, not their repaired versions. The shot study draws 20 Monte Carlo realizations from the known noisy all-zero probabilities for each budget. It measures conditional shot variability, not new hardware acquisitions or uncertainty across datasets. More shots reduce sampling variance while noise bias remains.
"""),
        code("""display(pd.read_csv(ROOT / 'results/tables/kernel_diagnostics.csv'))
display(Image(filename=str(ROOT / 'results/figures/kernel_comparison.png')))
display(Image(filename=str(ROOT / 'results/figures/shot_sensitivity.png')))
display(resources[resources.experiment == 'routing_control'][[
    'processor', 'strategy', 'feature_map', 'depth', 'cx_count', 'routing_swaps']])
"""),
        md("""## 7. Verify the generated evidence

Tests check the overlap identity on mapped physical circuits, allowed couplings, PSD feature extension, readout inversion, shot scheduling, and saved-count reproduction of reported predictions. Artifact checksums catch stale results after code changes. Versions and settings are exported in `results/manifest.json`.
"""),
        code("""import subprocess
check = subprocess.run([sys.executable, '-m', 'pytest', '-q'], cwd=ROOT,
                       capture_output=True, text=True)
print(check.stdout)
if check.returncode:
    print(check.stderr)
assert check.returncode == 0, 'Scientific or artifact verification failed'
display(pd.Series(manifest['versions'], name='Installed version'))
"""),
        md("""## Interpretation and limits

Evaluate the architecture claim using CX, depth, routing SWAPs, and kernel distortion together. Better connectivity can reduce gate cost even when test accuracy is unchanged. The results concern these disclosed synthetic topologies, encoding, noise channels, and execution policy.

Four qubits are easy to simulate; this experiment establishes neither quantum speedup nor superiority over a strong classical classifier. The single small holdout limits accuracy claims. Readout mitigation assumes exact synthetic-channel knowledge. Simulator timing does not estimate hardware timing, and no physical gate durations or T1/T2 noise are supplied.

See [report.md](report.md), [README.md](README.md), and [SUBMISSION_CHECKLIST.md](SUBMISSION_CHECKLIST.md). Phase 1 requires only A and B. Submit a public GitHub URL through the official portal by the guide's stated deadline, 14 October 2026 at 23:59 IST.
"""),
    ]
    notebook = nbf.v4.new_notebook(cells=cells,metadata={
        'kernelspec': {'display_name':'Python 3','language':'python','name':'python3'},
        'language_info': {'name':'python','version':'3.12'},
    })
    path = ROOT / 'main.ipynb'
    nbf.write(notebook,path)
    return path


def execute(path):
    from nbclient import NotebookClient
    from jupyter_client import KernelManager
    notebook = nbf.read(path,as_version=4)
    manager = KernelManager(kernel_name='python3')
    manager.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}']
    client = NotebookClient(notebook,km=manager,timeout=1200,
                            resources={'metadata':{'path':str(ROOT)}})
    t0=time.perf_counter()
    client.execute()
    nbf.write(notebook,path)
    errors=[output for cell in notebook.cells if cell.cell_type=='code'
            for output in cell.get('outputs',[]) if output.output_type=='error']
    verification={
        'executed':True,'error_outputs':len(errors),'code_cells':sum(c.cell_type=='code' for c in notebook.cells),
        'execution_seconds':time.perf_counter()-t0,'python_executable':sys.executable,
        'notebook_sha256':__import__('hashlib').sha256(path.read_bytes()).hexdigest(),
    }
    (ROOT/'results/notebook_verification.json').write_text(json.dumps(verification,indent=2))
    print(json.dumps(verification,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    path=build()
    if args.execute:
        execute(path)
    else:
        print(path)

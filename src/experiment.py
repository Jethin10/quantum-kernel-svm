"""Single reproducible entry point for the complete PS-Q2 submission."""
import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time
import numpy as np
import pandas as pd
from qiskit import qpy
from .kernels import assemble, evaluation_pairs, ideal_pair_values, sample_values, simulate_pairs
from .metrics import diagnostics, fit_kernel, psd_extension, wilson_interval
from .problem import FeatureSpec, kernel, load_dataset, make_scaler, states
from .processors import load_processor, reference_b
from .routing import compile_overlap, validate_compilation
from .selection import classical_baselines, select_quantum
from . import visualization as viz


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=lambda v: v.item() if isinstance(v, np.generic) else str(v)), encoding="utf-8")


def export_table(root, name, rows):
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    frame.to_csv(root / "results" / "tables" / f"{name}.csv", index=False)
    write_json(root / "results" / "tables" / f"{name}.json", frame.to_dict(orient="records"))
    return frame


def run_experiment(root=None, shots=4096, repeats=20, batch_size=128):
    root = Path(root or Path(__file__).resolve().parents[1]).resolve()
    for folder in ["data/raw", "data/processed", "results/tables", "results/figures", "assets/circuits"]:
        (root / folder).mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    print("PS-Q2: exact reference, model selection, and A/B architecture benchmark", flush=True)
    X, y, train_idx, test_idx = load_dataset()
    Xtrain, Xtest = X[train_idx], X[test_idx]
    ytrain, ytest = y[train_idx], y[test_idx]
    pd.DataFrame({"x1": X[:,0], "x2": X[:,1], "label": y,
                  "split": ["train" if i in train_idx else "test" for i in range(100)]}).to_csv(
        root / "data" / "raw" / "moons.csv", index=False)
    spec, C, selection = select_quantum(Xtrain, ytrain)
    export_table(root, "quantum_model_selection", selection)
    print(f"Frozen training-selected feature map: {spec}, C={C}", flush=True)
    scaler = make_scaler().fit(Xtrain)
    train_states = states(scaler.transform(Xtrain), spec)
    test_states = states(scaler.transform(Xtest), spec)
    exact_train, exact_cross = kernel(train_states), kernel(test_states, train_states)
    exact_diagnostic = diagnostics(exact_train)
    assert exact_diagnostic["minimum_eigenvalue"] > -1e-9
    assert np.max(np.abs(np.diag(exact_train)-1)) < 1e-10
    baseline_rows, baseline_models = classical_baselines(Xtrain, ytrain, Xtest, ytest)
    accuracy_rows = list(baseline_rows)
    qmodel, stats, pred = fit_kernel(exact_train, exact_cross, ytrain, ytest, C)
    accuracy_rows.append({"model": "Exact QSVM", "cv_mean": float(selection.iloc[0].cv_mean), **stats})
    predictions = {"exact": pred.tolist()}
    pairs = evaluation_pairs(len(train_idx), len(test_idx))
    exact_values = ideal_pair_values(exact_train, exact_cross, pairs)
    ideal_sample = sample_values(exact_values, shots, 42)
    ideal_train, ideal_cross = assemble(ideal_sample, pairs)
    _, stats, pred = fit_kernel(ideal_train, ideal_cross, ytrain, ytest, C, repair=True)
    accuracy_rows.append({"model": "Ideal shot QSVM", **stats})
    predictions["ideal_shot"] = pred.tolist()
    np.savez_compressed(root / "data" / "processed" / "dataset_and_reference.npz",
                        X=X, y=y, train_idx=train_idx, test_idx=test_idx,
                        Xtrain_scaled=scaler.transform(Xtrain), Xtest_scaled=scaler.transform(Xtest),
                        train_states=train_states, test_states=test_states,
                        exact_train=exact_train, exact_cross=exact_cross,
                        ideal_shot_train=ideal_train, ideal_shot_cross=ideal_cross,
                        pair_indices=np.asarray(pairs), exact_pair_values=exact_values,
                        scaler_min=scaler.min_, scaler_scale=scaler.scale_)
    configs = [load_processor(root / "processors" / f"processor_{name}.json")[0] for name in ["A", "B"]]
    if configs[0]["noise"] != configs[1]["noise"]:
        raise ValueError("A/B noise rates must match")
    resource_rows, diagnostic_rows, raw_kernels, noise_outputs = [], [], {}, {}
    validations, timings, layouts = {}, {}, {}
    # Structural controls isolate topology/depth without spending extra shots.
    cycle_control = {**configs[0], "name": "A_cycle_control",
                     "edges": [[0,1],[1,2],[2,3],[3,0],[3,4]]}
    for config in configs + [reference_b(), cycle_control]:
        for topology in ["linear", "ring", "full"]:
            for strategy in ["sabre", "basic"]:
                ablation_spec = replace(spec, topology=topology)
                _, _, _, meta = compile_overlap(config, ablation_spec, strategy)
                resource_rows.append({"experiment": "routing_control", **meta})
        for reps in [1, 2, 3]:
            _, _, _, meta = compile_overlap(config, replace(spec, reps=reps))
            resource_rows.append({"experiment": "depth_control", **meta})
    for config in configs:
        name = config["name"]
        template, xp, yp, meta = compile_overlap(config, spec)
        meta.update(experiment="primary", evaluated_pairs=len(pairs), shots_per_pair=shots,
                    total_shots=len(pairs)*shots, total_cx_executions=len(pairs)*shots*meta["cx_count"])
        resource_rows.append(meta)
        layouts[name] = meta["initial_layout"]
        validations[name] = validate_compilation(template, xp, yp, scaler.transform(Xtrain), exact_train)
        with (root / "assets" / "circuits" / f"processor_{name}.qpy").open("wb") as handle:
            qpy.dump(template, handle)
        (root / "assets" / "circuits" / f"processor_{name}.txt").write_text(
            str(template.draw(output="text", fold=140)), encoding="utf-8")
        points = scaler.transform(np.vstack([Xtrain, Xtest]))
        simulation = simulate_pairs(template, xp, yp, points, pairs, config, root,
                                    shots=shots, batch_size=batch_size)
        noise_outputs[name] = simulation
        timings[name] = simulation["timing"]
        processed = {}
        for variant, values in [
            ("noisy raw", simulation["sampled_values"]),
            ("noisy PSD", simulation["sampled_values"]),
            ("noise expectation PSD", simulation["expected_values"]),
            ("readout-mitigated PSD", simulation["mitigated_values"]),
        ]:
            raw_train, raw_cross = assemble(values, pairs)
            repair = variant.endswith("PSD")
            correction = {}
            if repair:
                train, cross, correction = psd_extension(raw_train, raw_cross)
            else:
                train, cross = raw_train, raw_cross
            _, stats, pred = fit_kernel(train, cross, ytrain, ytest, C)
            stats.update(correction)
            stats["accuracy_wilson_95"] = wilson_interval(int(np.sum(pred == ytest)), len(ytest))
            accuracy_rows.append({"model": f"{name} {variant}", **stats})
            predictions[f"{name}_{variant}"] = pred.tolist()
            if variant == "noisy raw":
                raw_kernels[name] = train
            processed[f"{variant.replace(' ', '_')}_train"] = train
            processed[f"{variant.replace(' ', '_')}_cross"] = cross
            if repair:
                processed[f"{variant.replace(' ', '_')}_raw_train"] = raw_train
                processed[f"{variant.replace(' ', '_')}_raw_cross"] = raw_cross
            diagnostic_rows.append({"processor": name, "variant": variant,
                                    "matrix_kind": "corrected" if repair else "raw",
                                    **diagnostics(train, exact_train),
                                    "cross_kernel_rmse": float(np.sqrt(np.mean((cross-exact_cross)**2))),
                                    "raw_kernel_rmse": float(np.sqrt(np.mean((raw_train-exact_train)**2))),
                                    "raw_cross_kernel_rmse": float(np.sqrt(np.mean((raw_cross-exact_cross)**2)))})
        np.savez_compressed(root / "data" / "processed" / f"processor_{name}_kernels.npz",
                            **processed,
                            histograms=simulation["histograms"],
                            pre_readout_probabilities=simulation["pre_readout_probabilities"],
                            observed_probabilities=simulation["observed_probabilities"],
                            mitigated_values_unclipped=simulation["mitigated_values_unclipped"],
                            pair_indices=np.asarray(pairs))
    print("Shot-budget sensitivity and fixed-hyperparameter controls", flush=True)
    sensitivity_rows = []
    for name in ["A", "B"]:
        for budget in sorted(set([256, 1024, 4096, 16384, shots])):
            for repeat in range(repeats):
                sampled = sample_values(noise_outputs[name]["expected_values"], budget, 1000+repeat)
                train, cross = assemble(sampled, pairs)
                _, stats, _ = fit_kernel(train, cross, ytrain, ytest, C, repair=True)
                sensitivity_rows.append({"processor": name, "shots": budget, "repeat": repeat,
                                         "accuracy": stats["accuracy"],
                                         "kernel_rmse": float(np.sqrt(np.mean((train-exact_train)**2)))})
    accuracy_table = export_table(root, "classification", accuracy_rows)
    resource_table = export_table(root, "architecture_resources", resource_rows)
    export_table(root, "kernel_diagnostics", diagnostic_rows)
    sensitivity_table = export_table(root, "shot_sensitivity", sensitivity_rows)
    sensitivity_summary = sensitivity_table.groupby(["processor", "shots"]).agg(
        accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", "std"),
        kernel_rmse_mean=("kernel_rmse", "mean"), kernel_rmse_std=("kernel_rmse", "std"),
        realizations=("repeat", "count"),
    ).reset_index()
    export_table(root, "shot_sensitivity_summary", sensitivity_summary)
    # Data-independent depth ablation; no parameter retuning or winner selection.
    ablation_rows = []
    for topology in ["linear", "ring", "full"]:
        for reps in [1, 2, 3]:
            control_spec = replace(spec, topology=topology, reps=reps)
            trstates = states(scaler.transform(Xtrain), control_spec)
            teststates = states(scaler.transform(Xtest), control_spec)
            _, stats, _ = fit_kernel(kernel(trstates), kernel(teststates,trstates), ytrain,ytest,C)
            ablation_rows.append({**control_spec.to_dict(), **stats})
    export_table(root, "feature_map_controls", ablation_rows)
    viz.configure()
    viz.feature_circuit(spec,root / "results/figures/feature_map.png")
    viz.architectures(configs, layouts, root / "results/figures/architectures.png")
    viz.boundaries(X,y,train_idx,test_idx,scaler,spec,qmodel,train_states,baseline_models,
                   root / "results/figures/decision_boundaries.png")
    viz.kernel_comparison(exact_train, raw_kernels, root / "results/figures/kernel_comparison.png")
    viz.cost_comparison(resource_rows, root / "results/figures/architecture_cost.png")
    viz.routing_controls(resource_rows, root / "results/figures/routing_controls.png")
    viz.shot_sensitivity(sensitivity_table, root / "results/figures/shot_sensitivity.png")
    viz.accuracy_plot(accuracy_rows, root / "results/figures/classification.png")
    manifest = {
        "problem": "PS-Q2", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {"n_samples":100,"noise":0.1,"random_state":42},
        "split": {"train":80,"test":20,"stratified":True,"seed":42},
        "feature_map":spec.to_dict(), "C":C,
        "selection": {"folds":5,"seed":42,"scaler_fit_per_fold":True,
                      "candidate_feature_maps":8,"C_values":[.1,1,10,100],
                      "quantum_tuning_method":"exact statevector, no device shots"},
        "execution": {"shots":shots,"seed_simulator":42,"seed_transpiler":42,
                      "optimization_level":1,"batch_size":batch_size,
                      "pairs_including_diagonals":len(pairs),
                      "shots_per_processor":len(pairs)*shots,
                      "resampling_repeats":repeats,"simulation_method":"density_matrix"},
        "exact_kernel_diagnostics":exact_diagnostic,
        "compiled_overlap_max_errors":validations,"aer_timings":timings,
        "predictions":predictions,"test_labels":ytest.tolist(),
        "versions": {name:importlib.metadata.version(name) for name in [
            "qiskit","qiskit-aer","numpy","scipy","scikit-learn","matplotlib","networkx","pandas"]},
        "python":platform.python_version(),"platform":platform.platform(),
        "total_pipeline_seconds":time.perf_counter()-t0,
        "noise_assumption":"Depolarizing channel on rz/sx/x; same policy on A/B. No T1/T2 or durations.",
        "readout_mitigation":"Inverse of known synthetic independent-bit readout channel; no calibration cost claimed.",
        "psd_extension":"Positive training eigenspace above 1e-8; cross projected into same eigenspace.",
    }
    write_json(root / "results" / "manifest.json", manifest)
    from .documentation import write_documents
    write_documents(root, manifest, accuracy_rows, resource_rows, diagnostic_rows, sensitivity_table)
    checksums = {}
    for folder in ["src", "processors", "data", "results/tables", "results/figures"]:
        for path in sorted((root / folder).rglob("*")):
            if path.is_file() and "__pycache__" not in str(path):
                checksums[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    write_json(root / "results" / "checksums.json", checksums)
    print(accuracy_table[["model","accuracy","f1"]].to_string(index=False), flush=True)
    print("Completed. See README.md, report.md, main.ipynb, and results/.", flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shots",type=int,default=4096)
    parser.add_argument("--repeats",type=int,default=20)
    parser.add_argument("--batch-size",type=int,default=128)
    args = parser.parse_args()
    if args.shots < 1 or args.repeats < 2 or args.batch_size < 1:
        parser.error("shots/batch-size must be positive; repeats must be at least 2")
    run_experiment(shots=args.shots,repeats=args.repeats,batch_size=args.batch_size)


if __name__ == "__main__":
    main()

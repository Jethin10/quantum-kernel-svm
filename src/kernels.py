"""Batched Aer experiments with complete histograms and resumable pair caches."""
import hashlib
import json
import time
import numpy as np
from qiskit import ClassicalRegister
from qiskit_aer import AerSimulator
from .metrics import readout_channel
from .processors import noise_model
from .routing import bind_pair, measured_qubits


def evaluation_pairs(ntrain=80, ntest=20):
    # Include diagonals to reveal noisy self-overlap rather than forcing them to 1.
    return [(i, j) for i in range(ntrain) for j in range(i, ntrain)] + [
        (ntrain + i, j) for i in range(ntest) for j in range(ntrain)
    ]


def assemble(values, pairs, ntrain=80, ntest=20):
    train = np.zeros((ntrain, ntrain))
    cross = np.zeros((ntest, ntrain))
    for value, (i, j) in zip(values, pairs):
        if i < ntrain:
            train[i, j] = train[j, i] = value
        else:
            cross[i - ntrain, j] = value
    return train, cross


def ideal_pair_values(exact_train, exact_cross, pairs):
    ntrain = len(exact_train)
    return np.array([exact_train[i, j] if i < ntrain else exact_cross[i-ntrain, j]
                     for i, j in pairs])


def sample_values(probabilities, shots, seed):
    return np.random.default_rng(seed).binomial(shots, np.clip(probabilities, 0, 1)) / shots


def simulate_pairs(template, xp, yp, points, pairs, config, root,
                   shots=4096, seed=42, batch_size=128):
    identity = {
        "processor": config, "circuit": str(template),
        "points_sha256": hashlib.sha256(points.tobytes()).hexdigest(),
        "pair_schedule_sha256": hashlib.sha256(np.asarray(pairs,dtype=np.int64).tobytes()).hexdigest(),
        "shots": shots, "seed": seed, "batch_size": batch_size,
        "protocol_version": 2,
    }
    import qiskit, qiskit_aer
    identity.update(qiskit=qiskit.__version__, qiskit_aer=qiskit_aer.__version__)
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
    cache_dir = root / "results" / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    simulator = AerSimulator(
        method="density_matrix", noise_model=noise_model(config),
        max_parallel_threads=4, max_parallel_experiments=4,
        enable_truncation=True,
    )
    qargs = measured_qubits(template)
    base = template.remove_final_measurements(inplace=False)
    base.save_probabilities(qargs, label="probabilities")
    base.add_register(ClassicalRegister(4, "meas"))
    base.measure(qargs, range(4))
    histograms, quantum_distributions, elapsed = [], [], 0.0
    metadata = None
    cache_hits = 0
    for start in range(0, len(pairs), batch_size):
        batch_pairs = pairs[start:start+batch_size]
        path = cache_dir / f"{key}_{start:05d}.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as saved:
                counts = saved["histograms"]
                probabilities = saved["probabilities"]
                elapsed += float(saved["elapsed"])
                metadata = json.loads(str(saved["metadata"]))
            cache_hits += 1
        else:
            circuits = [bind_pair(base, xp, yp, points[i], points[j])
                        for i, j in batch_pairs]
            t0 = time.perf_counter()
            result = simulator.run(circuits, shots=shots, seed_simulator=seed+start).result()
            seconds = time.perf_counter() - t0
            if not result.success:
                raise RuntimeError(f"Aer failed: {result.status}")
            probabilities = np.asarray([result.data(i)["probabilities"]
                                        for i in range(len(circuits))])
            counts = np.zeros((len(circuits), 16), dtype=np.int64)
            for i in range(len(circuits)):
                for bitstring, count in result.get_counts(i).items():
                    counts[i, int(bitstring.replace(" ", ""), 2)] = count
            if not np.all(counts.sum(axis=1) == shots):
                raise AssertionError("Incomplete measurement counts")
            metadata = {
                k: v for k, v in result.results[0].metadata.items()
                if k in {"num_qubits", "method", "measure_sampling", "active_input_qubits"}
            }
            np.savez_compressed(path, histograms=counts, probabilities=probabilities,
                                elapsed=seconds, metadata=json.dumps(metadata))
            elapsed += seconds
        histograms.append(counts)
        quantum_distributions.append(probabilities)
        print(f"  {config['name']}: {min(start+batch_size,len(pairs))}/{len(pairs)} pairs"
              f" ({cache_hits} cached batches)", flush=True)
    histograms = np.vstack(histograms)
    pre_readout = np.vstack(quantum_distributions)
    channel = readout_channel(config["noise"]["readout"])
    observed = pre_readout @ channel.T
    mitigated = (histograms / shots) @ np.linalg.inv(channel).T
    return {
        "histograms": histograms, "pre_readout_probabilities": pre_readout,
        "observed_probabilities": observed,
        "sampled_values": histograms[:, 0] / shots,
        "expected_values": observed[:, 0],
        "mitigated_values_unclipped": mitigated[:, 0],
        "mitigated_values": np.clip(mitigated[:, 0], 0, 1),
        "timing": {"aer_original_seconds": elapsed, "cache_hit_batches": cache_hits,
                   "batches": len(histograms) // batch_size + int(len(histograms) % batch_size > 0),
                   "simulator_metadata": metadata, "cache_key": key},
    }

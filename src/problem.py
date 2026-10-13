"""Dataset, explicit two-coordinate encoding, and exact kernel reference."""
from dataclasses import asdict, dataclass
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


@dataclass(frozen=True)
class FeatureSpec:
    reps: int = 2
    scale: float = 0.5
    topology: str = "ring"

    def to_dict(self):
        return asdict(self)


def interaction_edges(topology):
    if topology == "ring":
        return [(0, 1), (1, 2), (2, 3), (3, 0)]
    if topology == "linear":
        return [(0, 1), (1, 2), (2, 3)]
    if topology == "full":
        return [(i, j) for i in range(4) for j in range(i + 1, 4)]
    raise ValueError(f"Unknown interaction topology: {topology}")


def feature_map(spec, prefix="x"):
    """Each layer: H on all qubits, RZ(2 theta_i), RZZ(2 theta_i theta_j).

    theta = scale * (a,b,a,b); only two independent input parameters.
    No barriers: compiler cancellation is part of the measured cost.
    """
    if spec.reps < 1 or spec.scale <= 0:
        raise ValueError("Feature-map repetitions and scale must be positive")
    params = ParameterVector(prefix, 2)
    theta = [spec.scale * params[i % 2] for i in range(4)]
    circuit = QuantumCircuit(4, name=f"phi_{spec.topology}_r{spec.reps}")
    for _ in range(spec.reps):
        circuit.h(range(4))
        for i in range(4):
            circuit.rz(2 * theta[i], i)
        for i, j in interaction_edges(spec.topology):
            circuit.rzz(2 * theta[i] * theta[j], i, j)
    return circuit, list(params)


def overlap_circuit(spec):
    left, xp = feature_map(spec, "x")
    right, yp = feature_map(spec, "y")
    circuit = left.compose(right.inverse())
    circuit.name = "fidelity_compute_uncompute"
    circuit.measure_all()
    return circuit, xp, yp


def states(points, spec):
    circuit, params = feature_map(spec)
    return np.asarray([
        Statevector.from_instruction(circuit.assign_parameters(dict(zip(params, x)))).data
        for x in points
    ])


def kernel(left_states, right_states=None):
    if right_states is None:
        right_states = left_states
    values = np.abs(left_states.conj() @ right_states.T) ** 2
    return np.clip(values.real, 0, 1)


def load_dataset():
    X, y = make_moons(n_samples=100, noise=0.1, random_state=42)
    train_idx, test_idx = train_test_split(
        np.arange(100), test_size=0.2, random_state=42, stratify=y
    )
    return X, y, train_idx, test_idx


def make_scaler():
    # Out-of-range validation/test values stay out of range; no test-fitted limits.
    return MinMaxScaler(feature_range=(0, np.pi), clip=False)

"""Validated processor configuration and identical synthetic noise channels."""
import json
import networkx as nx
from qiskit.transpiler import CouplingMap
from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error


def load_processor(path):
    config = json.loads(path.read_text(encoding="utf-8"))
    graph = nx.Graph()
    graph.add_nodes_from(range(config["num_qubits"]))
    for edge in config["edges"]:
        if len(edge) != 2 or edge[0] == edge[1] or any(
            q not in range(config["num_qubits"]) for q in edge
        ):
            raise ValueError(f"Invalid coupling edge: {edge}")
        graph.add_edge(*edge)
    if not nx.is_connected(graph):
        raise ValueError("Processor graph must be connected")
    if config["basis_gates"] != ["rz", "sx", "x", "cx"]:
        raise ValueError("Unexpected native gate set")
    return config, graph


def coupling_map(config):
    edges = [tuple(edge) for edge in config["edges"]]
    if not config["directed"]:
        edges += [(b, a) for a, b in edges]
    return CouplingMap(edges)


def noise_model(config):
    rates = config["noise"]
    model = NoiseModel()
    # The guide specifies a 1q rate, but not virtual-Z exemption. Apply it to
    # every listed 1q native gate, identically on A and B; disclose this choice.
    model.add_all_qubit_quantum_error(
        depolarizing_error(rates["one_qubit_depolarizing"], 1),
        config["one_qubit_noise_gates"],
    )
    model.add_all_qubit_quantum_error(
        depolarizing_error(rates["cx_depolarizing"], 2), ["cx"]
    )
    p = rates["readout"]
    model.add_all_qubit_readout_error(ReadoutError([[1 - p, p], [p, 1 - p]]))
    return model


def reference_b():
    return {
        "name": "B_reference", "num_qubits": 9, "directed": False,
        "edges": [[0,1],[1,2],[1,6],[2,3],[2,6],[2,5],
                  [3,4],[4,5],[5,6],[6,7],[6,8],[8,0]],
        "basis_gates": ["rz", "sx", "x", "cx"],
    }

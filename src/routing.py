"""Compile once before binding; account for routing before SWAP decomposition."""
from itertools import permutations
import networkx as nx
import numpy as np
from qiskit import transpile
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import (
    ApplyLayout, BasicSwap, FullAncillaAllocation, EnlargeWithAncilla,
    SetLayout, Unroll3qOrMore,
)
from qiskit.transpiler import Layout
from .processors import coupling_map
from .problem import interaction_edges, overlap_circuit


def distance_layout(config, spec):
    """Same graph-only objective on both processors; never use labels/results."""
    graph = nx.Graph()
    graph.add_nodes_from(range(config["num_qubits"]))
    graph.add_edges_from(config["edges"])
    distances = dict(nx.all_pairs_shortest_path_length(graph))
    edges = interaction_edges(spec.topology)
    return min(
        permutations(range(config["num_qubits"]), 4),
        key=lambda layout: (
            sum(distances[layout[i]][layout[j]] for i, j in edges), layout
        ),
    )


def measured_qubits(circuit):
    output = [None] * circuit.num_clbits
    for instruction in circuit.data:
        if instruction.operation.name == "measure":
            output[circuit.find_bit(instruction.clbits[0]).index] = circuit.find_bit(
                instruction.qubits[0]
            ).index
    if len(output) != 4 or any(q is None for q in output):
        raise ValueError("Expected four classical bits, each measuring a logical output")
    return output


def compile_overlap(config, spec, strategy="sabre"):
    if strategy == "fused_sabre":
        from .optimization import fused_overlap
        logical, xp, yp = fused_overlap(spec)
    else:
        logical, xp, yp = overlap_circuit(spec)
    cmap = coupling_map(config)
    initial = list(distance_layout(config, spec))
    if strategy == "basic":
        # Decompose RZZ etc. first, then expose BasicSwap inserted SWAPs.
        native = transpile(logical, basis_gates=config["basis_gates"],
                           optimization_level=1, seed_transpiler=42)
        layout = Layout({native.qubits[i]: initial[i] for i in range(4)})
        pm = PassManager([
            SetLayout(layout), FullAncillaAllocation(cmap), EnlargeWithAncilla(),
            ApplyLayout(), Unroll3qOrMore(), BasicSwap(cmap),
        ])
        routed = pm.run(native)
        swaps = int(routed.count_ops().get("swap", 0))
        compiled = transpile(routed, basis_gates=config["basis_gates"],
                             optimization_level=1, seed_transpiler=42)
    else:
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        pm = generate_preset_pass_manager(
            optimization_level=1, basis_gates=config["basis_gates"],
            coupling_map=cmap, initial_layout=initial,
            routing_method="sabre", seed_transpiler=42,
        )
        # The routing stage contains SWAPs before translation decomposes them.
        trace = []
        def capture(**kwargs):
            if kwargs["pass_"].__class__.__name__ == "SabreSwap":
                trace.append(int(kwargs["dag"].count_ops().get("swap", 0)))
        compiled = pm.run(logical, callback=capture)
        swaps = trace[-1] if trace else 0
    active = sorted({compiled.find_bit(q).index for item in compiled.data
                     if item.operation.name not in {"barrier", "measure"}
                     for q in item.qubits})
    metadata = {
        "processor": config["name"], "strategy": strategy,
        "initial_layout": initial, "measured_qubits": measured_qubits(compiled),
        "physical_qubits": config["num_qubits"], "active_qubits": active,
        "logical_depth": logical.depth(), "depth": compiled.depth(),
        "cx_count": int(compiled.count_ops().get("cx", 0)),
        "one_qubit_gates": sum(int(compiled.count_ops().get(g, 0))
                               for g in ("rz", "sx", "x")),
        "routing_swaps": swaps,
        "native_gate_counts": dict(compiled.count_ops()),
        "feature_map": spec.to_dict(),
    }
    return compiled, xp, yp, metadata


def bind_pair(template, xp, yp, left, right):
    return template.assign_parameters({**dict(zip(xp, left)), **dict(zip(yp, right))})


def validate_compilation(template, xp, yp, points, exact_kernel, tolerance=1e-9):
    """Check physical measurement mapping, including final routing permutation."""
    from qiskit.quantum_info import Statevector
    qubits = measured_qubits(template)
    errors = []
    for i, j in [(0, 0), (0, 1), (2, 7), (3, 9), (12, 15)]:
        bound = bind_pair(template, xp, yp, points[i], points[j])
        state = Statevector.from_instruction(bound.remove_final_measurements(inplace=False))
        p0 = state.probabilities(qargs=qubits)[0]
        errors.append(abs(float(p0) - exact_kernel[i, j]))
    if max(errors) > tolerance:
        raise AssertionError(f"Compiled kernel differs from exact reference: {max(errors)}")
    return max(errors)

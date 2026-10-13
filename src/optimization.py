"""Exact diagonal fusion and connectivity-aware parity synthesis.

Parity networks restore the wire basis after each diagonal block. Their
optimality is not claimed outside this restricted construction.
"""
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from .routing import distance_layout, measured_qubits


CHAIN = [(2,3),(1,2),(0,1),(1,2),(2,3)]
RING = [(0,1),(1,2),(2,3),(3,0)]


def fused_overlap(spec, parity=False, layout=None, physical_qubits=4):
    if spec.reps < 1 or spec.scale <= 0:
        raise ValueError('Feature-map repetitions and scale must be positive')
    if spec.topology != "ring":
        raise ValueError("Parity construction supports the ring feature map")
    xp, yp = ParameterVector("x",2), ParameterVector("y",2)
    x = [spec.scale*xp[i%2] for i in range(4)]
    y = [spec.scale*yp[i%2] for i in range(4)]
    layout = list(layout or range(4))
    circuit = QuantumCircuit(physical_qubits,4)
    def hadamards():
        for q in layout:
            circuit.h(q)
    def diagonal(a, b=None, sign=1):
        singles = [sign*2*(a[i]-(b[i] if b is not None else 0)) for i in range(4)]
        angles = {frozenset((i,j)): sign*2*(a[i]*a[j]-(b[i]*b[j] if b is not None else 0)) for i,j in RING}
        for i in range(4):
            circuit.rz(singles[i],layout[i])
        if not parity:
            for i,j in RING:
                circuit.rzz(angles[frozenset((i,j))],layout[i],layout[j])
        elif parity == "cycle":
            for i,j in RING:
                circuit.cx(layout[i],layout[j])
                circuit.rz(angles[frozenset((i,j))],layout[j])
                circuit.cx(layout[i],layout[j])
        else:
            # Expose 23,12,01,03 parities using nearest-neighbor CX only.
            for step,(i,j) in enumerate(CHAIN):
                circuit.cx(layout[i],layout[j])
                edge = {0:(2,3),1:(1,2),2:(0,1),4:(0,3)}.get(step)
                if edge is not None:
                    circuit.rz(angles[frozenset(edge)],layout[j])
            for i,j in reversed(CHAIN):
                circuit.cx(layout[i],layout[j])
    for _ in range(spec.reps-1):
        hadamards(); diagonal(x)
    hadamards(); diagonal(x,y)
    for _ in range(spec.reps-1):
        hadamards(); diagonal(y,sign=-1)
    hadamards()
    for i,q in enumerate(layout):
        circuit.measure(q,i)
    return circuit,xp,yp


def compile_optimized(config,spec):
    if config.get('directed',False):
        raise ValueError('Parity compiler currently requires bidirectional CX edges')
    layout = list(distance_layout(config,spec))
    edges = {frozenset(edge) for edge in config["edges"]}
    cycle = all(frozenset((layout[i],layout[j])) in edges for i,j in RING)
    if not cycle and not all(frozenset((layout[i],layout[i+1])) in edges for i in range(3)):
        raise ValueError("Chosen layout must contain a path or cycle")
    logical,xp,yp = fused_overlap(spec,"cycle" if cycle else "chain",layout,config["num_qubits"])
    compiled = transpile(logical,basis_gates=config["basis_gates"],optimization_level=1,seed_transpiler=42)
    for item in compiled.data:
        if item.operation.name == "cx":
            pair = frozenset(compiled.find_bit(q).index for q in item.qubits)
            assert pair in edges,"Non-native CX in parity circuit"
    meta = {"processor":config["name"],"strategy":"fused_parity",
            "physical_qubits":config["num_qubits"],"initial_layout":layout,
            "measured_qubits":measured_qubits(compiled),"depth":compiled.depth(),
            "cx_count":int(compiled.count_ops().get("cx",0)),"routing_swaps":0,
            "one_qubit_gates":sum(compiled.count_ops().get(g,0) for g in ("rz","sx","x")),
            "native_gate_counts":dict(compiled.count_ops()),"feature_map":spec.to_dict()}
    return compiled,xp,yp,meta

# Exact overlap compiler

The original feature map alternates Hadamard layers with diagonal RZ/RZZ
blocks. In compute–uncompute, the central forward and inverse diagonal blocks
touch. Diagonal gates commute, so they combine exactly. The single-wire angle
is `2*(theta_x[i]-theta_y[i])`; the pair angle is
`2*(theta_x[i]*theta_x[j]-theta_y[i]*theta_y[j])`.
The latter is a difference of products, not a product of differences.
For r repetitions, this removes one of the 2r diagonal blocks.

A CX network can expose a pair's XOR parity on a wire. Applying RZ there,
then restoring the original basis, implements its ZZ phase. The path compiler
uses the CX prefix `2→3, 1→2, 0→1, 1→2, 2→3`. It applies phases after
steps 1, 2, 3, and 5, exposing pairs 23, 12, 01, and 03 respectively.
Reversing the prefix restores the wire basis. Every CX connects neighboring
wires. This uses ten CX gates per diagonal block without an ancillary qubit
or a SWAP. A native four-cycle uses four CX–RZ–CX constructions, eight CX
gates per block. These are explicit constructions, not a global optimality claim.

The three compiler conditions isolate the effects:

- Baseline compute–uncompute and SABRE.
- Central diagonal fusion followed by SABRE.
- Central fusion and native parity synthesis.

All compile before input binding with optimization level 1 and seed 42.
Both native constructions support undirected processor edges only. Noise
remains attached to every executed native gate. A logically equivalent
compiler therefore changes physical noise exposure, not the ideal kernel.

`tests/test_optimization.py` compares the full operator up to global phase
for 72 parameter assignments across one, two, and three repetitions.
Physical mapped all-zero probabilities are also compared with the exact
training kernel before any noisy acquisition. This verification is independent
of the classifier labels and accuracy.

Sources for the construction:

- [Qiskit RZZ definition](https://eu-de.quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.circuit.library.RZZGate).
- [Amy, Azimzadeh, Mosca: CNOT-phase circuit optimization](https://arxiv.org/abs/1712.01859).
- [Connectivity-constrained phase-polynomial synthesis](https://arxiv.org/abs/1904.01972).

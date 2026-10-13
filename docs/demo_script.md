# Three-minute demonstration

## Opening, 30 seconds

“We use the prescribed 100-point moons dataset. A four-qubit quantum circuit
estimates pairwise fidelity similarities; a classical SVM trains on those
similarities. The question is how much the processor graph and compiler change
the cost of estimating the same kernel.”

Show the executed notebook's data and feature-map cells. Point to training-only
selection and the untouched 20-point test set.

## Baseline, 40 seconds

Show the five-node chain and nine-node square. The ring interaction fits a
four-cycle in B. With ordinary routing, A needs 62 CX gates and B needs 30.

“That result alone is partly built into the choice of interaction graph.
We therefore compare three A graphs and three B graphs, with linear, ring,
and full K4 interactions under two routing methods. A five-node cycle can
match the nine-node square. A nine-node tree can cost more than the five-node
chain. Connectivity matters more than the physical-qubit count.”

Show `topology_grid.png`, and briefly point to the full-interaction bars.

## Main contribution, 60 seconds

“We then ask whether the chain really needs routed SWAPs for this kernel.
The central diagonal blocks in compute–uncompute can be merged exactly.
Native nearest-neighbor CX gates expose the required XOR parities, receive
their phases, then restore the wire basis. No SWAP or extra logical qubit is
needed.”

Show `compiler_optimization.png` and its measured table. A falls to 30 CX gates
and depth 52. B falls to 24 CX and depth 52. The B-over-A CX advantage shrinks
from 51.6% to 20%. Full-unitary tests and mapped-probability checks establish
that both implement the original ideal kernel.

“The compiler and topology jointly determine cost. Comparing routing alone
can overstate the unavoidable penalty of a sparse graph.”

## Evidence and limits, 50 seconds

Show raw kernel RMSE, repeated-shot summaries, and the verification output.
The optimized seeded run is 100% on both, but repeated-shot mean accuracy
does not show a reliable gain. Claim lower native cost and kernel distortion.
Both exact QSVM and a tuned RBF baseline are strong; this is no quantum
advantage claim.

If time permits, show the secondary training-selected linear model. It
reaches the same training CV score with 22 CX gates on either graph.
Changing the model and compiling a fixed model are separate optimization
choices; the notebook preserves both comparisons.

End with the reproducible evidence: full counts, raw/repaired matrices,
training-only cost/quality and encoding controls, ten independently retuned
exact splits, executed notebook, pinned environment, and CI.

## Likely questions

- **Why duplicate the input coordinates?** The data has two independent
  features but the task requires four qubits. Repetition uses the register
  without claiming new information. A mean/difference encoding is compared
  inside training folds at the frozen angle scale.
- **Why not use the largest processor automatically?** More nodes do not
  guarantee an appropriate interaction subgraph. The tree and cycle controls
  test this directly.
- **Does compilation improve accuracy?** It reduces measured kernel error
  and resource cost here. Conditional shot-resampling averages remain close.
- **Is the compiler optimal?** No global optimality claim is made. It is an
  exact, verified construction for the disclosed ring and bidirectional edges.
- **Is the runtime a hardware estimate?** No. Aer density matrices evolve
  once per circuit and sample efficiently. Gate durations and queue times
  are unavailable in this synthetic setup.

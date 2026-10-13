"""Verify the complete optimized unitary, not just fitted scores."""
import numpy as np
from qiskit.quantum_info import Operator
from src.problem import FeatureSpec, overlap_circuit
from src.optimization import fused_overlap
from src.routing import bind_pair


def test_full_unitary_preserved():
    rng = np.random.default_rng(2026)
    for reps in [1,2,3]:
        spec = FeatureSpec(reps=reps,scale=.25)
        reference,xp,yp = overlap_circuit(spec)
        for strategy in [False,"chain","cycle"]:
            candidate,cx,cy = fused_overlap(spec,strategy)
            for _ in range(8):
                x,y = rng.uniform(-1,4,(2,2))
                a = Operator(bind_pair(reference,xp,yp,x,y).remove_final_measurements(inplace=False))
                b = Operator(bind_pair(candidate,cx,cy,x,y).remove_final_measurements(inplace=False))
                assert a.equiv(b,atol=1e-10)

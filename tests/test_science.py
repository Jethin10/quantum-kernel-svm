"""Checks at mathematical and physical boundaries, not implementation mirrors."""
from pathlib import Path
import numpy as np
import pytest
from qiskit.quantum_info import Statevector
from src.kernels import assemble, evaluation_pairs, sample_values
from src.metrics import psd_extension, readout_channel
from src.problem import FeatureSpec, feature_map, kernel, load_dataset, states
from src.processors import coupling_map, load_processor
from src.routing import bind_pair, compile_overlap, measured_qubits


ROOT = Path(__file__).resolve().parents[1]


def test_prescribed_dataset_and_disjoint_stratified_split():
    X, y, train, test = load_dataset()
    assert X.shape == (100, 2)
    assert len(train) == 80 and len(test) == 20
    assert set(train).isdisjoint(test)
    assert sorted(np.r_[train,test]) == list(range(100))
    assert np.bincount(y[train]).tolist() == [40,40]
    assert np.bincount(y[test]).tolist() == [10,10]


@pytest.mark.parametrize("topology",["linear","ring","full"])
def test_exact_kernel_is_psd_and_nontrivial(topology):
    points = np.array([[.1,.2],[.7,1.2],[1.1,.4],[.4,.9]])
    spec = FeatureSpec(2,.5,topology)
    circuit, params = feature_map(spec)
    assert circuit.num_qubits == 4 and len(params) == 2
    matrix = kernel(states(points,spec))
    np.testing.assert_allclose(matrix,matrix.T,atol=1e-12)
    np.testing.assert_allclose(np.diag(matrix),1,atol=1e-12)
    assert np.linalg.eigvalsh(matrix).min() > -1e-10
    assert matrix[0,1] < .99


@pytest.mark.parametrize("processor",["A","B"])
@pytest.mark.parametrize("strategy",["sabre","basic"])
def test_compiled_probability_matches_exact_and_respects_couplings(processor,strategy):
    config,_ = load_processor(ROOT / "processors" / f"processor_{processor}.json")
    spec = FeatureSpec(2,.5,"ring")
    template,xp,yp,meta = compile_overlap(config,spec,strategy)
    points = np.array([[.2,.7],[1.1,.4]])
    exact = kernel(states(points,spec))
    for i,j in [(0,0),(0,1),(1,0)]:
        circuit = bind_pair(template,xp,yp,points[i],points[j])
        state = Statevector.from_instruction(circuit.remove_final_measurements(inplace=False))
        probability = state.probabilities(qargs=measured_qubits(template))[0]
        assert abs(probability-exact[i,j]) < 1e-10
    edges = set(coupling_map(config).get_edges())
    for item in template.data:
        assert item.operation.name in {"rz","sx","x","cx","measure","barrier"}
        if item.operation.name == "cx":
            assert tuple(template.find_bit(q).index for q in item.qubits) in edges
    assert meta["physical_qubits"] == config["num_qubits"]
    if processor == "B":
        assert meta["routing_swaps"] == 0


def test_kernel_pair_workload_and_cross_orientation():
    pairs = evaluation_pairs()
    assert len(pairs) == 4840
    assert sum(i==j for i,j in pairs) == 80
    values = np.arange(len(pairs),dtype=float)
    train,cross = assemble(values,pairs)
    assert train.shape == (80,80) and cross.shape == (20,80)
    np.testing.assert_array_equal(train,train.T)
    assert cross[0,0] == 3240 and cross[-1,-1] == 4839


def test_psd_repair_has_consistent_out_of_sample_feature_representation():
    train = np.array([[1.,1.2,.1],[1.2,1.,.2],[.1,.2,1.]])
    cross = np.array([[.3,.4,.7],[.8,.9,.1]])
    repaired,extended,meta = psd_extension(train,cross)
    assert np.linalg.eigvalsh(train).min() < 0
    assert np.linalg.eigvalsh(repaired).min() > -1e-10
    eig,V = np.linalg.eigh(train)
    keep = eig>1e-8
    train_features = V[:,keep]*np.sqrt(eig[keep])
    test_features = cross@V[:,keep]/np.sqrt(eig[keep])
    np.testing.assert_allclose(repaired,train_features@train_features.T,atol=1e-12)
    np.testing.assert_allclose(extended,test_features@train_features.T,atol=1e-12)
    joint = np.vstack([train_features,test_features])
    assert np.linalg.eigvalsh(joint@joint.T).min() > -1e-10
    assert meta["retained_rank"] == 2


def test_known_readout_channel_preserves_mass_and_can_be_inverted():
    channel = readout_channel(.01)
    np.testing.assert_allclose(channel.sum(axis=0),1)
    true = np.random.default_rng(42).dirichlet(np.ones(16))
    observed = channel@true
    np.testing.assert_allclose(np.linalg.solve(channel,observed),true,atol=1e-12)
    zero = np.eye(16)[0]
    assert (channel@zero)[0] == pytest.approx(.99**4)


def test_seeded_shots_preserve_extremes_and_expected_statistics():
    estimates = sample_values(np.full(10000,.5),4096,42)
    assert abs(estimates.mean()-.5)<.0005
    assert estimates.std() == pytest.approx(1/(2*np.sqrt(4096)),rel=.05)
    np.testing.assert_array_equal(sample_values(np.array([0.,1.]),4096,42),[0.,1.])

"""Validate saved measurements and result claims when the experiment exists."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from src.kernels import assemble
from src.metrics import fit_kernel, psd_extension, readout_channel


ROOT = Path(__file__).resolve().parents[1]
READY = (ROOT / "results/manifest.json").exists()


@pytest.mark.skipif(not READY,reason="Run experiment first for artifact integration checks")
def test_exported_predictions_and_counts_reproduce_metrics():
    manifest = json.loads((ROOT / "results/manifest.json").read_text())
    reference = np.load(ROOT / "data/processed/dataset_and_reference.npz")
    table = pd.read_csv(ROOT / "results/tables/classification.csv").set_index("model")
    for name in ["A","B"]:
        data = np.load(ROOT / f"data/processed/processor_{name}_kernels.npz")
        hist = data["histograms"]
        assert hist.shape == (4840,16)
        assert np.all(hist.sum(axis=1)==manifest["execution"]["shots"])
        np.testing.assert_allclose(data["pre_readout_probabilities"].sum(axis=1),1,atol=1e-9)
        np.testing.assert_allclose(data["observed_probabilities"],
            data["pre_readout_probabilities"]@readout_channel(.01).T,atol=1e-12)
        train,cross = assemble(hist[:,0]/manifest["execution"]["shots"],data["pair_indices"])
        np.testing.assert_array_equal(train,data["noisy_raw_train"])
        np.testing.assert_array_equal(cross,data["noisy_raw_cross"])
        repaired,extended,_ = psd_extension(train,cross)
        np.testing.assert_allclose(repaired,data["noisy_PSD_train"],atol=1e-12)
        np.testing.assert_allclose(extended,data["noisy_PSD_cross"],atol=1e-12)
        assert np.linalg.eigvalsh(data["noisy_PSD_train"]).min() > -1e-10
        ytrain=reference["y"][reference["train_idx"]]
        ytest=reference["y"][reference["test_idx"]]
        _,stats,pred=fit_kernel(train,cross,ytrain,ytest,manifest["C"],repair=True)
        assert stats["accuracy"] == pytest.approx(table.loc[f"{name} noisy PSD","accuracy"])
        assert pred.tolist() == manifest["predictions"][f"{name}_noisy PSD"]


@pytest.mark.skipif(not READY,reason="Run experiment first for artifact integration checks")
def test_saved_artifact_checksums():
    checksums=json.loads((ROOT / "results/checksums.json").read_text())
    assert len(checksums)>20
    for name,expected in checksums.items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected,name


@pytest.mark.skipif(not (ROOT/'results/tables/compiler_and_topology_measurements.csv').exists(),reason='Run expanded experiment first')
def test_optimized_measurements_reproduce_predictions():
    reference = np.load(ROOT/'data/processed/dataset_and_reference.npz')
    ytrain = reference['y'][reference['train_idx']]
    for name in ['A_fused_parity','B_fused_parity','A_tree_baseline','B_tree_baseline','A_fused_sabre','B_fused_sabre','A_cost_aware','B_cost_aware']:
        saved = np.load(ROOT/f'data/processed/{name}_kernels.npz')
        assert saved['histograms'].shape == (4840,16)
        assert np.all(saved['histograms'].sum(axis=1)==4096)
        train,cross = assemble(saved['histograms'][:,0]/4096,saved['pair_indices'])
        np.testing.assert_array_equal(train,saved['raw_train'])
        repaired,extended,_ = psd_extension(train,cross)
        np.testing.assert_allclose(repaired,saved['psd_train'],atol=1e-12)
        np.testing.assert_allclose(extended,saved['psd_cross'],atol=1e-12)
        _,_,pred = fit_kernel(repaired,extended,ytrain,saved['test_labels'],float(saved['C']))
        np.testing.assert_array_equal(pred,saved['predictions'])

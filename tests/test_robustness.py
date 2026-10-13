"""Reproduce the paired noisy study from counts and training-only inputs."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split
from src.problem import make_scaler,FeatureSpec,states,kernel
from src.kernels import assemble
from src.metrics import fit_kernel,psd_extension

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(not (ROOT/'results/tables/noisy_split_robustness.csv').exists(),reason='Run noisy study first')
def test_noisy_split_evidence_reproduces_training_inputs_and_predictions():
    choices = pd.read_csv(ROOT/'results/tables/nested_split_robustness.csv')
    choices = choices[choices.model=='Exact QSVM'].set_index('split_seed')
    rows = pd.read_csv(ROOT/'results/tables/noisy_split_robustness.csv')
    assert len(rows)==20
    for row in rows.itertuples():
        saved = np.load(ROOT/f'data/processed/noisy_robustness/split_{row.split_seed}_{row.processor}_{row.compiler}.npz')
        choice = choices.loc[row.split_seed]
        assert (row.reps,row.scale,row.C)==(choice.reps,choice.scale,choice.C)
        tr,te = train_test_split(np.arange(100),test_size=.2,stratify=saved['y'],random_state=row.split_seed)
        np.testing.assert_array_equal(tr,saved['train_idx'])
        np.testing.assert_array_equal(te,saved['test_idx'])
        points = make_scaler().fit(saved['X'][tr]).transform(np.vstack([saved['X'][tr],saved['X'][te]]))
        np.testing.assert_allclose(points,saved['scaled_points'],atol=1e-12)
        s = states(points,FeatureSpec(int(row.reps),float(row.scale),'ring'))
        np.testing.assert_allclose(kernel(s[:80]),saved['exact_train'],atol=1e-12)
        np.testing.assert_allclose(kernel(s[80:],s[:80]),saved['exact_cross'],atol=1e-12)
        assert saved['histograms'].shape==(4840,16)
        assert np.all(saved['histograms'].sum(axis=1)==4096)
        a,b = assemble(saved['histograms'][:,0]/4096,saved['pair_indices'])
        np.testing.assert_array_equal(a,saved['raw_train'])
        np.testing.assert_array_equal(b,saved['raw_cross'])
        repaired,cross,_ = psd_extension(a,b)
        np.testing.assert_allclose(repaired,saved['psd_train'],atol=1e-12)
        np.testing.assert_allclose(cross,saved['psd_cross'],atol=1e-12)
        _,score,pred = fit_kernel(repaired,cross,saved['y'][tr],saved['y'][te],row.C)
        np.testing.assert_array_equal(pred,saved['predictions'])
        assert score['accuracy']==pytest.approx(row.accuracy)
        assert np.sqrt(np.mean((a-saved['exact_train'])**2))==pytest.approx(row.kernel_rmse)
    paired = pd.read_csv(ROOT/'results/tables/noisy_split_paired_differences.csv')
    for difference in paired.itertuples():
        group = rows[(rows.processor==difference.processor)&(rows.split_seed==difference.split_seed)].set_index('compiler')
        assert difference.accuracy_difference==pytest.approx(group.loc['fused_parity','accuracy']-group.loc['baseline','accuracy'])
        assert difference.kernel_rmse_reduction==pytest.approx(group.loc['baseline','kernel_rmse']-group.loc['fused_parity','kernel_rmse'])


@pytest.mark.skipif(not (ROOT/'frontend/data/experiment.json').exists(),reason='Export browser evidence first')
def test_browser_metrics_and_pair_counts_match_scientific_sources():
    data = json.loads((ROOT/'frontend/data/experiment.json').read_text())
    rows = pd.read_csv(ROOT/'results/tables/compiler_and_topology_measurements.csv')
    ref = np.load(ROOT/'data/processed/dataset_and_reference.npz')
    np.testing.assert_array_equal(data['dataset']['testIndices'],ref['test_idx'])
    assert len(data['conditions'])==6 and len(data['noisySplits'])==20
    for condition in data['conditions']:
        row = rows[(rows.processor==condition['processor'])&(rows.compiler==condition['compiler'])].iloc[0]
        assert condition['accuracy']==pytest.approx(row.accuracy)
        assert condition['cx']==row.cx_count
        np.testing.assert_allclose(condition['rawTrain'],assemble(np.asarray(condition['zeroCounts'])/4096,data['pairs'])[0],atol=5.1e-8)
        assert len(condition['predictions'])==20

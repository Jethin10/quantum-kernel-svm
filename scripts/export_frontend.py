"""Export the browser explorer from saved scientific evidence, never fabricated values."""
import ast
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.visualization import topology_positions


def export(root=ROOT):
    root = Path(root)
    sources = {}
    def track(path):
        path = root/path
        sources[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        return path
    def table(name):
        frame = pd.read_csv(track(f'results/tables/{name}.csv'))
        return json.loads(frame.to_json(orient='records'))
    ref = np.load(track('data/processed/dataset_and_reference.npz'))
    manifest = json.loads(track('results/manifest.json').read_text())
    measurements = table('compiler_and_topology_measurements')
    resampling = table('compiler_resampling_summary')
    conditions = []
    for row in measurements:
        if row['processor'] not in ['A','B']:
            continue
        name,compiler = row['processor'],row['compiler']
        path = f'data/processed/processor_{name}_kernels.npz' if compiler=='baseline' else f'data/processed/{name}_{compiler}_kernels.npz'
        saved = np.load(track(path))
        prefix = 'noisy_raw' if compiler=='baseline' else 'raw'
        mean = next(r for r in resampling if r['processor']==name and r['compiler']==compiler)
        conditions.append({
            'id':f'{name}_{compiler}','processor':name,'compiler':compiler,
            'depth':row['depth'],'cx':row['cx_count'],'swaps':row['routing_swaps'],
            'accuracy':row['accuracy'],'rmse':row['kernel_rmse'],'aerSeconds':row['aer_seconds'],
            'predictions':ast.literal_eval(row['test_predictions']),
            'layout':ast.literal_eval(row['initial_layout']),
            'rawTrain':saved[f'{prefix}_train'].round(7).tolist(),
            'rawCross':saved[f'{prefix}_cross'].round(7).tolist(),
            'zeroCounts':saved['histograms'][:,0].tolist(),
            'resampling':mean,
        })
    architectures = []
    for name in ['A','A_tree','A_cycle','B','B_reference','B_tree']:
        config = json.loads(track(f'processors/variants/{name}.json').read_text())
        config['positions'] = topology_positions(config)
        architectures.append(config)
    payload = {
        'schemaVersion':1,'generatedAt':manifest['generated_at_utc'],
        'featureMap':manifest['feature_map'],'C':manifest['C'],'shots':manifest['execution']['shots'],
        'dataset':{'X':ref['X'].round(7).tolist(),'y':ref['y'].tolist(),
            'trainIndices':ref['train_idx'].tolist(),'testIndices':ref['test_idx'].tolist()},
        'exactTrain':ref['exact_train'].round(7).tolist(),
        'pairs':ref['pair_indices'].tolist(),'conditions':conditions,
        'architectures':architectures,'topologyGrid':table('topology_grid'),
        'noisySplits':table('noisy_split_robustness'),
        'noisySummary':table('noisy_split_robustness_summary'),
        'pairedDifferences':table('noisy_split_paired_differences'),
        'exactSplits':table('nested_split_robustness'),
        'conditionalTrials':table('noisy_split_resampling'),
        'classifiers':table('classification'),
        'costAware':table('cost_aware_measurements'),
        'sources':sources,
        'displayPrecision':'Kernel matrices and point coordinates rounded to seven decimal places; full precision in source NPZ files.',
    }
    destination = root/'frontend/data/experiment.json'
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(payload,separators=(',',':'),allow_nan=False),encoding='utf-8')
    print(f'Exported {len(conditions)} circuit conditions, {len(payload["noisySplits"])} noisy split rows to {destination}')
    return destination


if __name__=='__main__':
    export()

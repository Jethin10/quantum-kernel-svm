"""Check the submission bundle, executed notebook, and data provenance."""
import hashlib
import json
from pathlib import Path
import sys
import nbformat
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def verify():
    required=[
        'README.md','main.ipynb','requirements.txt','requirements.lock.txt',
        'processors/processor_A.json','processors/processor_B.json','report.md',
        'src/__init__.py','src/problem.py','src/processors.py','src/routing.py',
        'src/metrics.py','src/visualization.py','results/manifest.json',
        'results/notebook_verification.json','data/raw/moons.csv',
        'data/processed/dataset_and_reference.npz',
        'data/processed/processor_A_kernels.npz','data/processed/processor_B_kernels.npz',
    ]
    for name in required:
        assert (ROOT/name).is_file(),f'Missing required artifact: {name}'
    notebook=nbformat.read(ROOT/'main.ipynb',as_version=4)
    nbformat.validate(notebook)
    code_cells=[cell for cell in notebook.cells if cell.cell_type=='code']
    assert all(cell.execution_count is not None for cell in code_cells),'Notebook contains unexecuted cells'
    assert not [out for cell in code_cells for out in cell.get('outputs',[])
                if out.output_type=='error'],'Notebook contains an error output'
    verification=json.loads((ROOT/'results/notebook_verification.json').read_text())
    assert verification['executed'] and verification['error_outputs']==0
    assert hashlib.sha256((ROOT/'main.ipynb').read_bytes()).hexdigest()==verification['notebook_sha256']
    manifest=json.loads((ROOT/'results/manifest.json').read_text())
    assert manifest['execution']['shots']==4096,'Official submission must use 4096 shots'
    assert manifest['execution']['pairs_including_diagonals']==4840
    assert max(manifest['compiled_overlap_max_errors'].values())<1e-9
    for name,n in [('A',5),('B',9)]:
        processor=json.loads((ROOT/f'processors/processor_{name}.json').read_text())
        assert processor['num_qubits']==n
        data=np.load(ROOT/f'data/processed/processor_{name}_kernels.npz')
        assert np.all(data['histograms'].sum(axis=1)==4096)
        assert np.linalg.eigvalsh(data['noisy_PSD_train']).min()>-1e-10
    checksums=json.loads((ROOT/'results/checksums.json').read_text())
    for path,expected in checksums.items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==expected,f'Stale artifact: {path}'
    return {'passed':True,'required_files':len(required),'executed_code_cells':len(code_cells),
            'artifact_checksums':len(checksums),'official_shots':4096}


if __name__=='__main__':
    try:
        print(json.dumps(verify(),indent=2))
    except Exception as error:
        print(f'Submission verification failed: {error}',file=sys.stderr)
        raise

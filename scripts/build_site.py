"""Build only the public website files; local tooling and credentials stay outside."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def build():
    source = ROOT/'frontend'
    target = source/'dist'
    target.mkdir(exist_ok=True)
    payload = json.loads((source/'data/experiment.json').read_text())
    for name,digest in payload['sources'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,f'Stale website evidence: {name}'
    assert len(payload['noisySplits'])==20 and len(payload['topologyGrid'])==36
    for name in ['index.html','styles.css','app.js','quantum.js','favicon.svg','data/experiment.json']+[g['path'] for g in payload.get('gallery',[])]:
        destination = target/name
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source/name,destination)
    (target/'.nojekyll').write_text('')
    print(f'Public website built: {target}')


if __name__=='__main__':
    build()

"""Package portable source and evidence, excluding local runtimes and caches."""
import hashlib
import json
from pathlib import Path
import zipfile
from verify_submission import verify


ROOT=Path(__file__).resolve().parents[1]


def package():
    verification=verify()
    target=ROOT/'submission'/'qff_ps_q2_submission.zip'
    target.parent.mkdir(exist_ok=True)
    directories=['src','processors','data','results','assets','tests','scripts','.github','research']
    files=[ROOT/name for name in [
        'README.md','report.md','main.ipynb','requirements.txt','requirements.lock.txt',
        'SUBMISSION_CHECKLIST.md','JUDGES_GUIDE.md','.gitignore','.gitattributes','pytest.ini',
        'QFF_Phase_1_Master_Guide.pdf',
    ] if (ROOT/name).is_file()]
    for directory in directories:
        files.extend(path for path in (ROOT/directory).rglob('*') if path.is_file()
                     and not any(part in {'__pycache__','.cache'} for part in path.parts))
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for path in sorted(files):
            archive.write(path,'qff_ps_q2/'+path.relative_to(ROOT).as_posix())
    result={'path':str(target),'files':len(files),'bytes':target.stat().st_size,
            'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'verification':verification}
    (target.parent/'package_manifest.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    package()

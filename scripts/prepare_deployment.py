"""Build an allowlisted deployment checkout without raw daily archives or user notes."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]


def copy_file(relative, target):
    source=ROOT/relative
    destination=target/relative
    destination.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(source,destination)


def copy_tree(relative, target):
    shutil.copytree(ROOT/relative,target/relative,
                    ignore=shutil.ignore_patterns('__pycache__','*.pyc','.DS_Store'))


def prepare(destination):
    destination=Path(destination).resolve()
    if destination.exists():
        raise ValueError('Deployment output already exists. Choose a new directory.')
    required=['outputs/live/regional/replay.json','outputs/cases/reference-regional/manifest.json',
              'outputs/evaluations/public-aircraft-screen.json','data/real/models/yamnet-onnx/manifest.json']
    for name in required:
        if not (ROOT/name).is_file():
            raise ValueError('Missing prepared deployment input: '+name)
    destination.mkdir(parents=True)
    files=['app.py','streamlit_app.py','requirements.txt','packages.txt','pyproject.toml','README.md','Dockerfile','railway.json','.dockerignore','.streamlit/config.toml',
           'deploy/requirements.txt','deploy/start.sh','deploy/healthcheck.py',
           'scripts/capture_wifi.py','scripts/capture_wifi.swift',
           'scripts/fetch_wifi3d.py','scripts/run_wifi3d.py','scripts/export_wifi3d.py',
           'scripts/fetch_wifi3d_suite.py','scripts/evaluate_wifi3d.py','scripts/export_wifi3d_evaluation.py',
           'scripts/prepare_deployment.py',
           'scripts/train_wifi3d.py', 'tests/test_reconstruction_train.py',
           'tests/test_reconstruction.py','tests/test_reconstruction_eval.py','tests/test_wifi3d_fetch.py',
           'requirements-reconstruction.txt']
    for name in files:copy_file(name,destination)
    shutil.copy2(ROOT/'deploy/source.gitignore', destination/'.gitignore')
    for name in ['soundleak','docs','outputs/cases/reference-regional','outputs/live/regional',
                 'outputs/wifi3d/mmfi-reference',
                 'outputs/wifi3d/evaluation',
                 'data/real/models/yamnet-onnx','data/real/sound-examples','data/real/regional/catalog-75km']:
        if (ROOT/name).exists():copy_tree(name,destination)
    for name in ['original.webm','provenance.json','trace_full_a1dfeb.json','trace_full_ace391.json']:
        copy_file('data/real/reference/'+name,destination)
    for name in ['Raw data_V_constant_H_constant.zip','Raw data_V_constant_H_varying.zip',
                 'Raw data_V_varying_H_constant.zip','Raw data_V_varying_H_varying.zip',
                 'screen-manifest.json','record.json','final_results_40_cases.csv']:
        copy_file('data/real/pathak/'+name,destination)
    for name in ['public-aircraft-screen.json','public-aircraft-screen.declaration.json','rejection-checks.json',
                 'public-dataset-audit.json','reference-regional.json','pilot-readiness.json']:
        if (ROOT/'outputs/evaluations'/name).exists():copy_file('outputs/evaluations/'+name,destination)
    report_file=ROOT/'outputs/evaluations/public-aircraft-screen.json'
    report_hash=sha256(report_file.read_bytes()).hexdigest()
    # Only immutable prepared display assets; never include local human annotations.
    assets=Path('outputs/reviews/public-aircraft')/report_hash/'assets'
    if not (ROOT/assets).is_dir():
        raise ValueError('Prepare public review assets before deployment.')
    copy_tree(assets,destination)
    manifest=[]
    for path in sorted(destination.rglob('*')):
        if path.is_file():
            manifest.append({'path':str(path.relative_to(destination)), 'bytes':path.stat().st_size,
                             'sha256':sha256(path.read_bytes()).hexdigest()})
    (destination/'deploy/bundle-manifest.json').write_text(json.dumps({'schema_version':1,'files':manifest,
        'total_bytes':sum(r['bytes'] for r in manifest),
        'excluded':'Virtualenv, raw daily tar parts, research scratch data, credentials, and private annotation/session files.'},indent=2))
    return destination,len(manifest),sum(r['bytes'] for r in manifest)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/deployment/source')
    args=parser.parse_args()
    folder,count,size=prepare(args.output)
    print(f'Prepared {count:,} files, {size/1e6:.1f} MB: {folder}')

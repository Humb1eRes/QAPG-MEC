"""Package the independent sensitivity experiment, then audit an isolated copy."""
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import zipfile

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
RESULTS = ROOT/'experiments/results/parameter_sensitivity_v1'
ARCHIVE = ROOT/'output/QAPG_Parameter_Sensitivity_Code_and_Results.zip'
PREFIX = 'QAPG_Parameter_Sensitivity_Code_and_Results'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def main():
    meta=json.loads((RESULTS/'metadata.json').read_text())
    validation=json.loads((RESULTS/'VALIDATION.json').read_text())
    assert meta['status']=='COMPLETED' and meta['passed_runs']==100
    assert validation['status']=='PASS' and validation['runs']==100
    assert validation['metadata_sha256']==sha(RESULTS/'metadata.json')
    if ARCHIVE.exists():raise FileExistsError('Preserve existing archive; do not overwrite')
    frozen=json.loads((BASE/'FREEZE.json').read_text())
    sources={}
    # Reconstruct all frozen dependencies under their existing relative paths.
    for name,digest in frozen['source_hashes'].items():
        path=ROOT/name
        assert sha(path)==digest
        sources[name]=path
    for path in BASE.rglob('*'):
        if path.is_file() and '__pycache__' not in path.parts and path.suffix not in ('.pyc','.so','.dylib') and path.name!='PACKAGE_VALIDATION.json':
            sources[str(path.relative_to(ROOT))]=path
    for path in RESULTS.rglob('*'):
        if path.is_file():sources[str(path.relative_to(ROOT))]=path
    matched=ROOT/'experiments/matched_backlog_v1'
    for name in ('FREEZE.json','CONTROLLER_DESIGN.md','MATH_REVIEW.md',
                 'QAPG_R_API_VALIDATION.json','QAPG_REVISED_MATH_VALIDATION.json',
                 'test_qapg_revised_api.py','test_qapg_revised_math.py'):
        path=matched/name
        if path.exists():sources[str(path.relative_to(ROOT))]=path
    root_readme=(BASE/'PACKAGE_README.md').read_bytes()
    manifest={'created_utc':datetime.now(timezone.utc).isoformat(),
              'purpose':meta['purpose'],'files':{}}
    for name,path in sorted(sources.items()):
        assert not path.is_symlink() and path.resolve().is_relative_to(ROOT.resolve())
        manifest['files'][name]={'sha256':sha(path),'size_bytes':path.stat().st_size}
    manifest['files']['README.md']={'sha256':hashlib.sha256(root_readme).hexdigest(),'size_bytes':len(root_readme)}
    manifest_bytes=(json.dumps(manifest,indent=2)+'\n').encode()
    ARCHIVE.parent.mkdir(exist_ok=True)
    temporary=ARCHIVE.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temporary,'w',allowZip64=True) as z:
        for name,path in sorted(sources.items()):
            # NPZ already uses ZIP compression; avoid redundant recompression.
            compression=zipfile.ZIP_STORED if path.suffix=='.npz' else zipfile.ZIP_DEFLATED
            z.write(path,PREFIX+'/'+name,compress_type=compression,compresslevel=None if compression==zipfile.ZIP_STORED else 6)
        z.writestr(PREFIX+'/README.md',root_readme,compress_type=zipfile.ZIP_DEFLATED)
        z.writestr(PREFIX+'/MANIFEST.json',manifest_bytes,compress_type=zipfile.ZIP_DEFLATED)
    with zipfile.ZipFile(temporary) as z:
        assert z.testzip() is None,'CRC failure'
        expected={PREFIX+'/'+name for name in manifest['files']}|{PREFIX+'/MANIFEST.json'}
        assert set(z.namelist())==expected
        for name,item in manifest['files'].items():
            data=z.read(PREFIX+'/'+name)
            assert len(data)==item['size_bytes'] and hashlib.sha256(data).hexdigest()==item['sha256']
        assert z.read(PREFIX+'/MANIFEST.json')==manifest_bytes
    os.replace(temporary,ARCHIVE)
    print(json.dumps({'archive':str(ARCHIVE),'members':len(manifest['files'])+1,
                      'bytes':ARCHIVE.stat().st_size,'crc_and_member_hashes':'PASS'}),flush=True)
    # Filesystem audit hook denies any access to the original project. Runtime
    # libraries outside that tree are allowed; inputs/code must resolve in copy.
    with tempfile.TemporaryDirectory(prefix='qapg-sensitivity-portability-') as temporary_dir:
        tmp=Path(temporary_dir)
        with zipfile.ZipFile(ARCHIVE) as z:z.extractall(tmp)
        extracted=tmp/PREFIX
        bootstrap=tmp/'isolated_validate.py'
        bootstrap.write_text('''import sys, os, runpy, json
from pathlib import Path
sys.dont_write_bytecode=True
original=Path(sys.argv[1]).resolve()
project=Path(sys.argv[2]).resolve()
attempts=[]
def audit(event,args):
    if event not in ("open","os.listdir","os.scandir") or not args:return
    value=args[0]
    if not isinstance(value,(str,bytes,os.PathLike)):return
    try:path=Path(os.fsdecode(value)).absolute()
    except (TypeError,ValueError):return
    if path==original or original in path.parents:
        attempts.append({"event":event,"path":str(path)})
        raise PermissionError("Original-project filesystem access forbidden: "+str(path))
sys.addaudithook(audit)
os.chdir(project)
sys.path.insert(0,str(project/"experiments/parameter_sensitivity_v1"))
runpy.run_path(str(project/"experiments/parameter_sensitivity_v1/validate_sensitivity.py"),run_name="__main__")
print("PORTABILITY_REPORT "+json.dumps({"status":"PASS","original_project_access_attempts":attempts}))
''')
        environment=dict(os.environ)
        environment.pop('PYTHONPATH',None)
        environment['PYTHONDONTWRITEBYTECODE']='1'
        executed=subprocess.run([sys.executable,str(bootstrap),str(ROOT.resolve()),str(extracted)],
                                cwd=extracted,env=environment,capture_output=True,text=True)
        if executed.returncode:
            raise RuntimeError('Isolated validation failed:\n'+executed.stdout+'\n'+executed.stderr)
        marker=next(line for line in executed.stdout.splitlines() if line.startswith('PORTABILITY_REPORT '))
        portable=json.loads(marker.split(' ',1)[1])
        assert portable['status']=='PASS' and portable['original_project_access_attempts']==[]
        copied_validation=json.loads((extracted/'experiments/results/parameter_sensitivity_v1/VALIDATION.json').read_text())
        assert copied_validation['status']=='PASS' and copied_validation['runs']==100
    # The archive is immutable; regeneration wrote only inside the temporary copy.
    assert sha(RESULTS/'VALIDATION.json')==manifest['files'][str((RESULTS/'VALIDATION.json').relative_to(ROOT))]['sha256']
    report={'status':'PASS','checked_utc':datetime.now(timezone.utc).isoformat(),
        'archive_path':str(ARCHIVE.relative_to(ROOT)),'archive_sha256':sha(ARCHIVE),
        'size_bytes':ARCHIVE.stat().st_size,'member_count':len(manifest['files'])+1,
        'zip_crc_check':'PASS','all_member_size_and_sha256_checks':'PASS',
        'isolated_extraction_validation':'PASS','isolated_recomputed_runs':100,
        'original_project_access_attempts':0,'all_inputs_included':True,
        'excluded':'Previous matched-backlog results, compiled shared libraries, Python caches, manuscript and figures',
        'scope':'Package integrity plus full numerical validator rerun in a separate extraction with original-project reads blocked; not a repeat of all100 controller simulations.'}
    (BASE/'PACKAGE_VALIDATION.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()

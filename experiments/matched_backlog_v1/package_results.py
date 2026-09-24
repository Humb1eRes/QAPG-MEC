"""Package the completed, verified matched-backlog experiment without old data.

Only this script is needed to verify an existing archive (--verify PATH).
Building requires every declared experiment/audit/figure gate to have passed.
Existing deliverables are never overwritten.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
import sys
import zipfile

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
RESULTS = ROOT / 'experiments/results/matched_backlog_v1'
ARCHIVE_NAME = 'QAPG_R_Matched_Backlog_Code_and_Results.zip'
DELIVERY_NAME = 'QAPG_R_Matched_Backlog_DELIVERY_MANIFEST.json'
EXCLUDED_PARTS = {'native_build', '__pycache__', '.git', '.DS_Store'}
EXCLUDED_SUFFIXES = {'.pyc', '.pyo', '.tmp', '.zip', '.dylib', '.so'}
CHUNK = 1024 * 1024


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(CHUNK), b''):
            digest.update(chunk)
    return digest.hexdigest()


def json_bytes(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode()


def read_json(path):
    require(path.is_file(), f'Missing required file: {path}')
    return json.loads(path.read_text())


def checked_relative(path):
    require(not path.is_symlink(), f'Symbolic links are not packaged: {path}')
    resolved = path.resolve(strict=True)
    require(resolved.is_relative_to(ROOT), f'File outside project: {path}')
    relative = resolved.relative_to(ROOT).as_posix()
    require('..' not in PurePosixPath(relative).parts and not relative.startswith('/'), f'Unsafe member: {relative}')
    return relative


def required_gates():
    import numpy as np

    gates = []
    for stage in ('development', 'selection', 'test'):
        gates.append(RESULTS / stage / 'INDEPENDENT_VALIDATION.json')
    for name in ('SELECTOR_AUDIT.json', 'QAPG_R_API_VALIDATION.json',
                 'QAPG_REVISED_MATH_VALIDATION.json', 'REPLAY_VALIDATION.json'):
        gates.append(BASE / name)
    gates.extend(RESULTS / 'analysis' / name for name in ('HOLDOUT_AUDIT.json', 'ANALYSIS_VALIDATION.json', 'FIGURE_QA.json'))
    records = {}
    for path in gates:
        report = read_json(path)
        require(report.get('status') == 'PASS', f'Required gate is not PASS: {path}')
        records[checked_relative(path)] = {'status': 'PASS', 'sha256': sha(path)}

    protocol_path = BASE / 'PROTOCOL.json'
    protocol = read_json(protocol_path)
    freeze = read_json(BASE / 'FREEZE.json')
    require(freeze['status'] == 'FROZEN_BEFORE_SELECTION', 'Missing source freeze')
    require(freeze['protocol_sha256'] == sha(protocol_path), 'Protocol changed after freeze')
    for relative, digest in freeze['source_hashes'].items():
        source = ROOT / relative
        require(checked_relative(source) == relative and sha(source) == digest, f'Frozen source changed: {relative}')
    selected_path = RESULTS / 'selection/SELECTED_SETTINGS.json'
    selected = read_json(selected_path)
    require(selected['status'] == 'FROZEN_BEFORE_TEST', 'Missing frozen selection')
    require(selected['protocol_sha256'] == sha(protocol_path), 'Selection uses another protocol')
    selector_audit = read_json(BASE / 'SELECTOR_AUDIT.json')
    require(selector_audit['selected_settings_sha256'] == sha(selected_path), 'Selector audit is stale')
    require(selector_audit['all_choices_match'] is True, 'Selector audit found a mismatch')
    require(selected['selector_source_sha256'] == sha(BASE / 'select_settings.py'), 'Selector source changed')
    require(selected['independent_validation_sha256'] == sha(RESULTS / 'selection/INDEPENDENT_VALIDATION.json'), 'Selection validation binding is stale')
    for name, key in (('QAPG_R_API_VALIDATION.json', 'controller_hashes'),
                      ('QAPG_REVISED_MATH_VALIDATION.json', 'controller_sha256')):
        report = read_json(BASE / name)
        for source, digest in report[key].items():
            require(sha(BASE / source) == digest, f'Controller validation is stale: {name}')
    replay = read_json(BASE / 'REPLAY_VALIDATION.json')
    require(replay['frozen_source_hashes'] == freeze['source_hashes'], 'Replay used different sources')
    require(replay['selection_metadata_sha256'] == sha(RESULTS / 'selection/metadata.json'), 'Replay selection binding is stale')
    holdout = read_json(RESULTS / 'analysis/HOLDOUT_AUDIT.json')
    for field, path in (('analysis_sha256', RESULTS / 'analysis/ANALYSIS.json'),
                        ('selected_settings_sha256', selected_path),
                        ('test_metadata_sha256', RESULTS / 'test/metadata.json'),
                        ('test_validation_sha256', RESULTS / 'test/INDEPENDENT_VALIDATION.json')):
        require(holdout[field] == sha(path), f'Hold-out audit binding is stale: {field}')

    counts = {}
    input_hash_cache = {}
    for stage in ('development', 'selection', 'test'):
        metadata_path = RESULTS / stage / 'metadata.json'
        metadata = read_json(metadata_path)
        validation = read_json(RESULTS / stage / 'INDEPENDENT_VALIDATION.json')
        require(metadata['status'] == 'COMPLETED', f'{stage} is incomplete')
        require(metadata['source_hashes'] == freeze['source_hashes'], f'{stage} source hashes differ')
        require(metadata['protocol_sha256'] == sha(protocol_path), f'{stage} protocol changed')
        require(validation['metadata_sha256'] == sha(metadata_path), f'{stage} metadata changed after validation')
        require(validation['runs'] == metadata['planned_runs'] == metadata['completed_runs'], f'{stage} run-count mismatch')
        require(validation['validator_sha256'] == sha(BASE / 'validate_results.py'), f'{stage} validator source changed')
        manifest = read_json(RESULTS / stage / 'input_manifest.json')
        input_ids = {(item['beta'], item['seed']) for item in manifest}
        expected_inputs = {(b, s) for b in protocol['betas'] for s in protocol[stage]['seeds']}
        require(input_ids == expected_inputs and len(manifest) == len(input_ids), f'{stage} input coverage differs')
        for item in manifest:
            for kind in ('arrivals', 'channels'):
                path = ROOT / item[kind + '_path']
                relative = checked_relative(path)
                require(relative.startswith(f'experiments/results/matched_backlog_v1/{stage}/inputs/'), 'Input is outside its stage')
                if relative not in input_hash_cache:
                    with np.load(path, allow_pickle=False) as data:
                        array = data[kind]
                        digest = hashlib.sha256()
                        digest.update(str(array.shape).encode())
                        digest.update(array.dtype.str.encode())
                        digest.update(np.ascontiguousarray(array).tobytes())
                        input_hash_cache[relative] = digest.hexdigest()
                require(input_hash_cache[relative] == item[kind + '_array_sha256'], f'Validated input changed: {relative}')
        records_in_stage = sorted((RESULTS / stage / 'records').glob('*.json'))
        require(len(records_in_stage) == metadata['completed_runs'], f'{stage} record-count mismatch')
        require({checked_relative(p): sha(p) for p in records_in_stage} == validation['record_hashes'], f'{stage} validated records changed')
        for record_path in records_in_stage:
            record = read_json(record_path)
            trace = ROOT / record['row']['trace_path']
            require(sha(trace) == record['row']['trace_sha256'], f'Trace changed after validation: {trace}')
        counts[stage] = metadata['completed_runs']
        if stage != 'development':
            require(freeze['frozen_utc'] < metadata['started_utc'], f'{stage} preceded freeze')
        if stage == 'test':
            require(metadata['selected_settings_sha256'] == sha(selected_path), 'Test used another selection')
            require(selected['frozen_utc'] < metadata['started_utc'], 'Test preceded selection freeze')
            require(counts[stage] == selected['planned_test_runs'], 'Test count differs from frozen plan')
    require(counts['development'] == 18 and counts['selection'] == protocol['budget']['selection_runs'] == 324,
            'Unexpected development or selection grid')
    require((RESULTS / 'analysis/RESULTS_REPORT.zh-CN.md').is_file(), 'Final Chinese report is missing')
    return records, freeze, selected, counts


def gather_files():
    paths = []
    for folder in (BASE, RESULTS):
        for path in folder.rglob('*'):
            parts = path.relative_to(folder).parts
            if any(part in EXCLUDED_PARTS or part.startswith('.') for part in parts):
                continue
            if path.is_symlink():
                raise RuntimeError(f'Symbolic links are not packaged: {path}')
            if path.is_file() and path.suffix not in EXCLUDED_SUFFIXES and not path.name.endswith('~'):
                paths.append(path)
    physical_sources = [
        'experiments/reference/edgesport_sim.py',
        'experiments/matched_recorder.py',
        'experiments/run_matched.py',
        'experiments/literature_baselines/common_simulator_v2.py',
        'experiments/literature_baselines/eedo.py',
        'experiments/literature_baselines/gupa_orthogonal.py',
    ]
    provenance_documents = [
        'EEDO_MAPPING.md', 'GUPA_MAPPING.md', 'GUPA_ADAPTATION_REVIEW.md',
        'GUPA_CONTROLLER_VALIDATION.json', 'GUPA_INTEGRATION_VALIDATION.json',
    ]
    paths.extend(ROOT / name for name in physical_sources)
    paths.extend(ROOT / 'experiments/literature_baselines' / name for name in provenance_documents)
    members = {}
    for path in paths:
        relative = checked_relative(path)
        require(path.is_file(), f'Missing selected source/document: {path}')
        require(relative not in members, f'Duplicate member: {relative}')
        members[relative] = path
    return dict(sorted(members.items()))


def archive_readme(counts, selected):
    return f'''# QAPG-R matched-backlog experiment

Start with [the Chinese results report](experiments/results/matched_backlog_v1/analysis/RESULTS_REPORT.zh-CN.md).
This archive contains the revised controller, all six comparison methods, the
complete fixed parameter-selection grid, frozen test choices, held-out results,
figures, and audit records. The comparison concerns energy under a shared
finite-window aggregate-backlog requirement; it is not a universal optimality
or indefinite queue-stability certificate.

There are {counts['development']} development runs, {counts['selection']} selection
runs, and {counts['test']} held-out runs. The frozen selector retained
{selected['decision_count']} method/load/target decisions, including
{selected['no_feasible_grid_decision_count']} with no feasible setting in the
declared grid. Failed targets and all parameter-grid results are preserved.

## Contents

- `experiments/matched_backlog_v1/`: protocol, derivation, source, independent
  audits and reproducibility instructions.
- `experiments/results/matched_backlog_v1/`: all development/selection/test
  inputs, raw traces, run records and analysis/figures.
- `experiments/reference/`, the two shared experiment helpers, and the selected
  files in `experiments/literature_baselines/`: unchanged required dependencies
  and source-to-adaptation explanations.
- `MANIFEST.json`: SHA-256 and byte count for every payload member, plus the
  creation time, frozen source hashes, and required audit-gate hashes. It excludes
  its own hash to avoid circular self-reference. The separate delivery manifest
  records the manifest hash, archive hash, and every archive member's hash.

Original literature PDFs, earlier manuscript archives/data, compiled shared
libraries and Python bytecode caches are not included. Required controller
binaries are compiled locally from the included C++17 source.

## Verification and reuse

Keep this delivered archive unchanged. Its ZIP CRC and every member SHA-256 were
checked after creation. To verify again with standard Python, extract the code
and run from the extracted project root:

```sh
python3 experiments/matched_backlog_v1/package_results.py --verify /path/to/{ARCHIVE_NAME}
```

See `experiments/matched_backlog_v1/REPRODUCE.md` for Python, NumPy, ReportLab,
Matplotlib and C++17 requirements. Keep the project-relative directory layout.
Use a separate working copy when running validators or replay scripts: they may
write new audit reports, whose timestamps/hashes differ from the original
committed audit evidence. Original traces must remain unchanged.

Historical source-adaptation audits describe their own earlier scope. The
matched-backlog `PROTOCOL.json`, `FREEZE.json`, and current stage/audit records
define this experiment. Review the final report's limits and all failed target
classifications alongside any energy comparisons.
'''.encode()


def verify_archive(path):
    with zipfile.ZipFile(path, 'r') as archive:
        entries = archive.infolist()
        names = [item.filename for item in entries]
        require(len(names) == len(set(names)), 'Duplicate ZIP members')
        for item in entries:
            member = PurePosixPath(item.filename)
            require(not member.is_absolute() and '..' not in member.parts and '\\' not in item.filename,
                    f'Unsafe ZIP path: {item.filename}')
            require(not stat.S_ISLNK(item.external_attr >> 16), f'ZIP symlink: {item.filename}')
        require(archive.testzip() is None, 'ZIP CRC check failed')
        manifest_bytes = archive.read('MANIFEST.json')
        manifest = json.loads(manifest_bytes)
        payload = manifest['members']
        require(set(names) == set(payload) | {'MANIFEST.json'}, 'Manifest/member coverage mismatch')
        member_hashes = {}
        for name in names:
            digest = hashlib.sha256()
            size = 0
            with archive.open(name) as handle:
                for chunk in iter(lambda: handle.read(CHUNK), b''):
                    digest.update(chunk)
                    size += len(chunk)
            member_hashes[name] = {'sha256': digest.hexdigest(), 'bytes': size}
            if name != 'MANIFEST.json':
                require(member_hashes[name] == payload[name], f'Member SHA/size mismatch: {name}')
        return {'status': 'PASS', 'zip_crc': 'PASS', 'member_sha256': 'PASS',
                'member_count': len(names), 'members': member_hashes,
                'manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest()}


def build_archive():
    gates, freeze, selected, counts = required_gates()
    files = gather_files()
    for relative in freeze['source_hashes']:
        require(relative in files, f'Frozen dependency missing from archive: {relative}')
    output = ROOT / 'output'
    output.mkdir(exist_ok=True)
    destination, delivery_path = output / ARCHIVE_NAME, output / DELIVERY_NAME
    temporary = output / (ARCHIVE_NAME + '.tmp')
    require(not destination.exists() and not delivery_path.exists() and not temporary.exists(),
            'Refusing to overwrite an existing delivery or temporary archive')
    generated = {'README.md': archive_readme(counts, selected)}
    members = {name: {'sha256': sha(path), 'bytes': path.stat().st_size} for name, path in files.items()}
    members.update({name: {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}
                    for name, data in generated.items()})
    created = datetime.now(timezone.utc).isoformat()
    manifest = {'format': 'QAPG-R matched-backlog payload manifest v1', 'created_utc': created,
                'member_hash_scope': 'Every payload member except this MANIFEST.json; delivery manifest records its hash separately.',
                'packager_sha256': sha(Path(__file__)), 'frozen_source_hashes': freeze['source_hashes'],
                'verification_gates': gates, 'stage_run_counts': counts,
                'selected_settings_sha256': sha(RESULTS / 'selection/SELECTED_SETTINGS.json'),
                'members': dict(sorted(members.items()))}
    generated['MANIFEST.json'] = json_bytes(manifest)
    try:
        with temporary.open('xb') as raw:
            with zipfile.ZipFile(raw, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6,
                                 allowZip64=True) as archive:
                for name, path in files.items():
                    compression = zipfile.ZIP_STORED if path.suffix.lower() in {'.npz', '.png', '.pdf', '.jpg', '.jpeg'} else zipfile.ZIP_DEFLATED
                    archive.write(path, arcname=name, compress_type=compression)
                for name, data in generated.items():
                    archive.writestr(name, data)
        verified = verify_archive(temporary)
        for name, path in files.items():
            require(sha(path) == members[name]['sha256'] and path.stat().st_size == members[name]['bytes'],
                    f'Source changed during packaging: {name}')
        gates_after, _, _, _ = required_gates()
        require(gates_after == gates, 'Required audit evidence changed during packaging')
        archive_hash, archive_size = sha(temporary), temporary.stat().st_size
        delivery = {'status': 'PASS', 'created_utc': created,
                    'verified_utc': datetime.now(timezone.utc).isoformat(),
                    'archive': ARCHIVE_NAME, 'archive_sha256': archive_hash, 'archive_bytes': archive_size,
                    'packager_sha256': sha(Path(__file__)), 'frozen_source_hashes': freeze['source_hashes'],
                    'verification_gates': gates, 'stage_run_counts': counts, **verified}
        temporary.rename(destination)
        with delivery_path.open('x') as handle:
            json.dump(delivery, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write('\n')
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    print(json.dumps({'status': 'PASS', 'archive': str(destination), 'archive_bytes': archive_size,
                      'archive_sha256': archive_hash, 'members': verified['member_count'],
                      'delivery_manifest': str(delivery_path)}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', type=Path, help='Verify an existing ZIP only; do not build or write files')
    args = parser.parse_args()
    if args.verify:
        report = verify_archive(args.verify)
        print(json.dumps({key: value for key, value in report.items() if key != 'members'}, indent=2))
    else:
        build_archive()


if __name__ == '__main__':
    main()

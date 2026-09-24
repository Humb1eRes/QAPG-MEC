"""Independent, source-frozen, one-factor QAPG sensitivity experiment."""
from __future__ import annotations
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
import argparse
import csv
import hashlib
import json
import multiprocessing
import platform
import sys
import time
import traceback

import numpy as np

sys.dont_write_bytecode = True
BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
MATCHED = BASE.parent / 'matched_backlog_v1'
OUT = BASE.parent / 'results/parameter_sensitivity_v1'
sys.path.insert(0, str(MATCHED))
import common_bridge as bridge

METRICS = [
    'mean_energy_j_per_slot', 'mean_backlog_mbit', 'mean_device_backlog_mbit',
    'mean_edge_backlog_mbit', 'mean_local_energy_j_per_slot',
    'mean_transmit_energy_j_per_slot', 'mean_edge_energy_j_per_slot',
    'mean_local_completed_mbit_per_slot', 'mean_edge_completed_mbit_per_slot',
    'mean_completed_mbit_per_slot', 'mean_arrival_mbit_per_slot',
    'window_local_completion_fraction', 'window_edge_completion_fraction',
    'completion_to_arrival_ratio', 'late_half_backlog_slope_mbit_per_slot',
    'mean_sweeps', 'convergence_fraction', 'mean_decision_runtime_ms',
    'maximum_best_deviation_gain', 'maximum_objective_rise', 'wall_seconds',
]


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def protocol():
    return json.loads((BASE / 'PROTOCOL.json').read_text())


def configurations(p):
    ref = dict(p['reference'])
    points = [dict(config_id='c00', parameters=ref, memberships=[])]
    by_key = {(ref['V_scale'], ref['beta'], ref['M'], ref['bandwidth_mhz']): points[0]}
    for axis, values in p['grids'].items():
        for value in values:
            params = dict(ref)
            params[axis] = value
            params['V'] = p['reference']['V'] * params['V_scale']
            params['bandwidth_hz'] = params['bandwidth_mhz'] * 1e6
            params['noise_w'] = params['bandwidth_hz'] * p['noise_psd_w_per_hz']
            key = (params['V_scale'], params['beta'], params['M'], params['bandwidth_mhz'])
            if key not in by_key:
                item = dict(config_id=f'c{len(points):02d}', parameters=params, memberships=[])
                points.append(item)
                by_key[key] = item
            by_key[key]['memberships'].append({'axis': axis, 'value': value})
    assert len(points) == p['distinct_configurations'] == 20
    assert len(points) * len(p['seeds']) == p['planned_runs'] == 100
    assert len(points[0]['memberships']) == 4
    return points


def old_source_hashes():
    expected = json.loads((MATCHED / 'FREEZE.json').read_text())['source_hashes']
    actual = {name: sha(ROOT / name) for name in expected}
    assert actual == expected, 'Existing matched-backlog source changed'
    return actual


def source_hashes():
    old = old_source_hashes()
    names = ['PROTOCOL.json', 'configurations.json', 'run_sensitivity.py', 'validate_sensitivity.py']
    return {**old, **{str((BASE / name).relative_to(ROOT)): sha(BASE / name) for name in names}}


def freeze():
    if (BASE / 'FREEZE.json').exists():
        raise FileExistsError('A freeze already exists; preserve this experiment version')
    p = protocol()
    dump(BASE / 'configurations.json', configurations(p))
    value = {'status': 'FROZEN_BEFORE_INPUTS_AND_RUNS', 'frozen_utc': now(),
             'protocol_sha256': sha(BASE / 'PROTOCOL.json'), 'source_hashes': source_hashes(),
             'original_freeze_sha256': sha(MATCHED / 'FREEZE.json'),
             'configuration_count': 20, 'planned_runs': 100}
    dump(BASE / 'FREEZE.json', value)
    print(json.dumps({'status': value['status'], 'configurations': 20, 'planned_runs': 100}), flush=True)


def configured(params, seed, p):
    return bridge.sim.rec.original.SimConfig(
        T=p['T'], seed=seed, N=params['N'], M=params['M'], beta=params['beta'],
        V=params['V'], bandwidth_hz=params['bandwidth_hz'], noise_w=params['noise_w'])


def make_inputs(points, p):
    folder = OUT / 'inputs'
    folder.mkdir(exist_ok=True)
    files = {}
    for seed in p['seeds']:
        for beta in p['grids']['beta']:
            cfg = bridge.sim.rec.original.SimConfig(T=p['T'], seed=seed, beta=beta)
            array, _ = bridge.sim.rec.original.make_arrivals(cfg)
            path = folder / f'arrivals_b{beta:g}_s{seed}.npz'
            np.savez_compressed(path, arrivals=array)
            files['arrivals', beta, seed] = {
                'path': str(path.relative_to(ROOT)), 'array_sha256': bridge.array_sha(array),
                'file_sha256': sha(path), 'shape': list(array.shape), 'dtype': array.dtype.str}
        for m in p['grids']['M']:
            cfg = bridge.sim.rec.original.SimConfig(T=p['T'], seed=seed, M=m)
            array = bridge.sim.rec.original.generate_channels(cfg)
            path = folder / f'channels_M{m}_s{seed}.npz'
            np.savez_compressed(path, channels=array)
            files['channels', m, seed] = {
                'path': str(path.relative_to(ROOT)), 'array_sha256': bridge.array_sha(array),
                'file_sha256': sha(path), 'shape': list(array.shape), 'dtype': array.dtype.str}
        print(json.dumps({'inputs_seed_completed': seed}), flush=True)
    manifest = []
    for point in points:
        params = point['parameters']
        for seed in p['seeds']:
            manifest.append({'config_id': point['config_id'], 'seed': seed,
                'arrivals': files['arrivals', params['beta'], seed],
                'channels': files['channels', params['M'], seed]})
    dump(OUT / 'input_manifest.json', manifest)
    return manifest


@lru_cache(maxsize=2)
def read_array(path, key):
    with np.load(ROOT / path, allow_pickle=False) as data:
        array = data[key]
    array.flags.writeable = False
    return array


def worker(job):
    identity = {'method': 'QAPG-R', 'config_id': job['point']['config_id'], 'seed': job['seed']}
    started = time.perf_counter()
    try:
        # Compilation/setup is outside the measured simulation runtime.
        from qapg_revised import _load_engine
        _load_engine()
        p, point, inp = job['protocol'], job['point'], job['input']
        cfg = configured(point['parameters'], job['seed'], p)
        arrays = {kind: read_array(inp[kind]['path'], kind) for kind in ('arrivals', 'channels')}
        for kind, array in arrays.items():
            assert bridge.array_sha(array) == inp[kind]['array_sha256']
        start_sim = time.perf_counter()
        trace, checks = bridge.simulate('QAPG-R', cfg, arrays['arrivals'], arrays['channels'])
        wall = time.perf_counter() - start_sim
        for kind, array in arrays.items():
            assert bridge.array_sha(array) == inp[kind]['array_sha256']
        sl = slice(p['warmup'], None)
        metrics = bridge.metrics(trace, p['warmup'])
        metrics.update(
            mean_local_energy_j_per_slot=float(np.mean(trace['local_energy_j'][sl])),
            mean_transmit_energy_j_per_slot=float(np.mean(trace['transmit_energy_j'][sl])),
            mean_edge_energy_j_per_slot=float(np.mean(trace['edge_energy'][sl])),
            mean_completed_mbit_per_slot=metrics['window_completed_mbit']/(cfg.T-p['warmup']),
            mean_sweeps=float(np.mean(trace['revised_sweeps'][sl])),
            convergence_fraction=float(np.mean(trace['revised_converged'][sl])),
            mean_decision_runtime_ms=float(np.mean(trace['runtime_ms'][sl])),
            maximum_best_deviation_gain=float(np.max(trace['best_deviation_gain'][sl])),
            maximum_objective_rise=float(np.max(trace['objective_max_rise'])),
            wall_seconds=wall)
        trace_path = OUT / 'traces' / f"{identity['config_id']}_s{cfg.seed}.npz"
        np.savez_compressed(trace_path, **trace)
        row = {**identity, **point['parameters'], 'T': cfg.T, 'warmup': p['warmup'],
               **metrics, 'trace_path': str(trace_path.relative_to(ROOT)),
               'trace_sha256': sha(trace_path)}
        record = {'status': 'PASS', 'identity': identity, 'config': asdict(cfg),
                  'memberships': point['memberships'], 'input': inp, 'row': row,
                  'checks': checks, 'recorded_utc': now(),
                  'source_hashes': job['source_hashes'],
                  'wall_seconds_with_io': time.perf_counter()-started}
    except BaseException as error:
        record = {'status': 'FAIL', 'identity': identity, 'recorded_utc': now(),
                  'error': repr(error), 'traceback': traceback.format_exc(),
                  'source_hashes': job['source_hashes'], 'input': job['input'],
                  'point': job['point'], 'wall_seconds_with_io': time.perf_counter()-started}
    dump(OUT / 'records' / f"{identity['config_id']}_s{identity['seed']}.json", record)
    return record


def summarize(records, points, p):
    passed = [r['row'] for r in records if r['status'] == 'PASS']
    write_csv(OUT / 'per_seed.csv', sorted(passed, key=lambda r: (r['config_id'], r['seed'])))
    grouped = defaultdict(list)
    for record in records:
        grouped[record['identity']['config_id']].append(record)
    config_rows, axis_rows = [], []
    for point in points:
        recs = grouped[point['config_id']]
        success = [r['row'] for r in recs if r['status'] == 'PASS']
        complete = len(success) == len(p['seeds'])
        row = {'config_id': point['config_id'], **point['parameters'],
               'seed_count': len(success), 'failed_seed_count': len(recs)-len(success),
               'status': 'PASS' if complete else 'INCOMPLETE_OR_FAILED'}
        for key in METRICS:
            vals = [r[key] for r in success]
            # Failed settings remain visible; no partial-seed performance estimate.
            row[key+'_mean'] = float(np.mean(vals)) if complete else ''
            row[key+'_sd'] = float(np.std(vals, ddof=1)) if complete else ''
        config_rows.append(row)
        for membership in point['memberships']:
            axis_rows.append({'axis': membership['axis'], 'axis_value': membership['value'], **row})
    write_csv(OUT / 'config_summary.csv', config_rows)
    order = {axis: i for i, axis in enumerate(p['grids'])}
    axis_rows.sort(key=lambda r: (order[r['axis']], r['axis_value']))
    write_csv(OUT / 'summary.csv', axis_rows)
    return len(passed)


def run(workers):
    p = protocol()
    if not 1 <= workers <= p['maximum_workers']:
        raise ValueError('Use one to four workers')
    frozen = json.loads((BASE / 'FREEZE.json').read_text())
    hashes = source_hashes()
    assert frozen['source_hashes'] == hashes
    assert frozen['protocol_sha256'] == sha(BASE / 'PROTOCOL.json')
    points = json.loads((BASE / 'configurations.json').read_text())
    assert points == configurations(p)
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError('Output must be new/empty; preserve every prior result')
    OUT.mkdir(parents=True, exist_ok=True)
    for name in ('records', 'traces'):
        (OUT / name).mkdir()
    dump(OUT / 'configurations.json', points)
    meta = {'status': 'RUNNING', 'started_utc': now(), 'planned_runs': p['planned_runs'],
            'configuration_count': len(points), 'workers': workers,
            'protocol_sha256': sha(BASE / 'PROTOCOL.json'), 'freeze_sha256': sha(BASE / 'FREEZE.json'),
            'source_hashes': hashes, 'python': sys.version, 'numpy': np.__version__,
            'platform': platform.platform(), 'purpose': p['purpose']}
    assert frozen['frozen_utc'] < meta['started_utc']
    dump(OUT / 'metadata.json', meta)
    began = time.perf_counter()
    try:
        manifest = make_inputs(points, p)
        meta['input_generation_seconds'] = time.perf_counter()-began
        jobs = [{'point': point, 'seed': seed, 'protocol': p, 'source_hashes': hashes,
                 'input': next(x for x in manifest if x['config_id']==point['config_id'] and x['seed']==seed)}
                for point in points for seed in p['seeds']]
        records = []
        with ProcessPoolExecutor(max_workers=workers,
                                 mp_context=multiprocessing.get_context('spawn')) as pool:
            futures = [pool.submit(worker, job) for job in jobs]
            for future in as_completed(futures):
                records.append(future.result())
                if len(records) % 10 == 0 or records[-1]['status'] == 'FAIL':
                    print(json.dumps({'completed': len(records), 'planned': len(jobs),
                        'failed': sum(r['status']=='FAIL' for r in records),
                        'elapsed_seconds': round(time.perf_counter()-began, 1)}), flush=True)
        assert source_hashes() == hashes
        passed = summarize(records, points, p)
        meta.update(status='COMPLETED' if passed==len(jobs) else 'COMPLETED_WITH_FAILURES',
                    completed_runs=len(records), passed_runs=passed,
                    failed_runs=len(records)-passed, finished_utc=now(),
                    duration_seconds=time.perf_counter()-began)
    except BaseException as error:
        meta.update(status='FAILED', error=repr(error), traceback=traceback.format_exc(),
                    finished_utc=now(), duration_seconds=time.perf_counter()-began)
        dump(OUT / 'metadata.json', meta)
        raise
    dump(OUT / 'metadata.json', meta)
    print(json.dumps({'status': meta['status'], 'runs': meta['completed_runs'],
                      'passed': meta['passed_runs'], 'duration_seconds': meta['duration_seconds']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', action='store_true')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    freeze() if args.freeze else run(args.workers)

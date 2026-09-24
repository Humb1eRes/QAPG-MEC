"""Recompute the sensitivity evidence from every saved trajectory and record."""
from __future__ import annotations
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import csv
import hashlib
import json
import sys
import numpy as np

sys.dont_write_bytecode = True
BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
import run_sensitivity as runner


def close(a, b, atol=1e-8, rtol=1e-10):
    np.testing.assert_allclose(a, b, atol=atol, rtol=rtol)


def rows(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def main():
    out, root = runner.OUT, runner.ROOT
    p = runner.protocol()
    frozen = json.loads((BASE/'FREEZE.json').read_text())
    meta = json.loads((out/'metadata.json').read_text())
    assert meta['status'] == 'COMPLETED' and meta['passed_runs'] == meta['completed_runs'] == 100
    assert meta['source_hashes'] == frozen['source_hashes'] == runner.source_hashes()
    assert meta['protocol_sha256'] == frozen['protocol_sha256'] == runner.sha(BASE/'PROTOCOL.json')
    assert frozen['frozen_utc'] < meta['started_utc']
    points = json.loads((out/'configurations.json').read_text())
    assert points == runner.configurations(p)
    configs = {x['config_id']: x for x in points}
    expected = {(config, seed) for config in configs for seed in p['seeds']}
    seen, checked_inputs, input_refs = set(), {}, {}
    recomputed, hashes = [], {}
    max_flow = max_energy = max_queue = max_unused = max_rise = 0.0
    for path in sorted((out/'records').glob('*.json')):
        record = json.loads(path.read_text())
        assert record['status'] == 'PASS'
        identity, row = record['identity'], record['row']
        key = (identity['config_id'], identity['seed'])
        assert key in expected and key not in seen
        seen.add(key)
        point = configs[key[0]]
        cfg = record['config']
        assert cfg == asdict(runner.configured(point['parameters'], key[1], p))
        assert record['memberships'] == point['memberships']
        assert record['source_hashes'] == frozen['source_hashes']
        assert record['checks']['conservation_and_resource_assertions_passed']
        assert row['T'] == p['T'] and row['warmup'] == p['warmup']
        assert cfg['noise_w'] == cfg['bandwidth_hz'] * p['noise_psd_w_per_hz']
        for kind in ('arrivals', 'channels'):
            inp = record['input'][kind]
            if inp['path'] not in checked_inputs:
                f = root/inp['path']
                assert runner.sha(f) == inp['file_sha256']
                with np.load(f, allow_pickle=False) as z: a = z[kind]
                assert runner.bridge.array_sha(a) == inp['array_sha256']
                assert list(a.shape) == inp['shape'] and a.dtype.str == inp['dtype']
                assert np.all(np.isfinite(a)) and np.all(a >= 0)
                checked_inputs[inp['path']] = dict(inp)
            assert checked_inputs[inp['path']] == inp
            sharing_key = (kind, cfg['beta'] if kind=='arrivals' else cfg['M'], key[1])
            if sharing_key in input_refs: assert input_refs[sharing_key] == inp
            input_refs[sharing_key] = inp
        tracepath = root/row['trace_path']
        assert runner.sha(tracepath) == row['trace_sha256']
        with np.load(tracepath, allow_pickle=False) as z: a = {k: z[k] for k in z.files}
        for name, x in a.items():
            assert x.shape[0] == p['T'] and np.all(np.isfinite(x)), name
        with np.load(root/record['input']['arrivals']['path']) as z: arrivals = z['arrivals']
        close(a['arrival_bits'], arrivals.sum(axis=1), atol=0, rtol=0)
        energy = a['local_energy_j'] + a['transmit_energy_j'] + a['edge_energy']
        complete = a['local_completed_bits'] + a['edge_completed_bits']
        close(a['total_energy'], energy, atol=1e-12)
        close(a['device_energy'], a['local_energy_j']+a['transmit_energy_j'], atol=1e-12)
        close(a['total_queue'], a['device_queue']+a['edge_queue'])
        oldq = np.r_[0, a['total_queue'][:-1]]
        flow = oldq+a['arrival_bits']-complete-a['total_queue']
        close(flow, 0, atol=1e-5, rtol=0)
        close(np.r_[0,a['device_queue'][:-1]]+a['arrival_bits']-a['local_completed_bits']-a['uploaded_bits'],a['device_queue'],atol=1e-5)
        close(np.r_[0,a['edge_queue'][:-1]]+a['uploaded_bits']-a['edge_completed_bits'],a['edge_queue'],atol=1e-5)
        close(a['edge_loads'].sum(axis=1),a['uploaded_bits'],atol=1e-5)
        close(np.cumsum(a['arrival_bits']-complete),a['total_queue'],atol=1e-4)
        assert a['edge_loads'].shape == (p['T'],cfg['M'])
        for name in ('total_energy','total_queue','local_completed_bits','edge_completed_bits','uploaded_bits'):
            assert np.all(a[name]>=-1e-7)
        unused=max(float(np.max(np.abs(a[name]))) for name in ('unused_local_capacity_bits','unused_edge_capacity_bits'))
        assert unused < 1e-4
        assert np.all((a['revised_sweeps']>=1)&(a['revised_sweeps']<=12))
        assert np.all(a['revised_converged'] == (a['best_deviation_gain']<=1e-10))
        assert float(a['objective_max_rise'].max()) <= 1e-9
        close(a['effective_V'],cfg['V'],atol=0,rtol=0)
        sl=slice(p['warmup'],None)
        late=a['total_queue'][p['T']//2:]/1e6
        x=np.arange(len(late),dtype=float);x-=x.mean()
        calc={
            'mean_energy_j_per_slot':float(energy[sl].mean()),
            'mean_backlog_mbit':float(a['total_queue'][sl].mean()/1e6),
            'mean_device_backlog_mbit':float(a['device_queue'][sl].mean()/1e6),
            'mean_edge_backlog_mbit':float(a['edge_queue'][sl].mean()/1e6),
            'mean_local_energy_j_per_slot':float(a['local_energy_j'][sl].mean()),
            'mean_transmit_energy_j_per_slot':float(a['transmit_energy_j'][sl].mean()),
            'mean_edge_energy_j_per_slot':float(a['edge_energy'][sl].mean()),
            'mean_local_completed_mbit_per_slot':float(a['local_completed_bits'][sl].mean()/1e6),
            'mean_edge_completed_mbit_per_slot':float(a['edge_completed_bits'][sl].mean()/1e6),
            'mean_completed_mbit_per_slot':float(complete[sl].mean()/1e6),
            'mean_arrival_mbit_per_slot':float(a['arrival_bits'][sl].mean()/1e6),
            'window_local_completion_fraction':float(a['local_completed_bits'][sl].sum()/complete[sl].sum()),
            'window_edge_completion_fraction':float(a['edge_completed_bits'][sl].sum()/complete[sl].sum()),
            'completion_to_arrival_ratio':float(complete[sl].sum()/a['arrival_bits'][sl].sum()),
            'late_half_backlog_slope_mbit_per_slot':float(np.dot(x,late-late.mean())/np.dot(x,x)),
            'mean_sweeps':float(a['revised_sweeps'][sl].mean()),
            'convergence_fraction':float(a['revised_converged'][sl].mean()),
            'mean_decision_runtime_ms':float(a['runtime_ms'][sl].mean()),
            'maximum_best_deviation_gain':float(a['best_deviation_gain'][sl].max()),
            'maximum_objective_rise':float(a['objective_max_rise'].max()),
            'wall_seconds':row['wall_seconds'],
        }
        for name,val in calc.items():close(row[name],val,atol=1e-9)
        recomputed.append({**identity,**point['parameters'],**calc})
        hashes[str(path.relative_to(root))]=runner.sha(path)
        max_flow=max(max_flow,float(np.abs(flow).max()))
        max_energy=max(max_energy,float(np.abs(a['total_energy']-energy).max()))
        max_queue=max(max_queue,float(np.abs(a['total_queue']-a['device_queue']-a['edge_queue']).max()))
        max_unused=max(max_unused,unused);max_rise=max(max_rise,float(a['objective_max_rise'].max()))
    assert seen == expected and len(seen)==100
    summary=rows(out/'summary.csv');configsummary=rows(out/'config_summary.csv')
    assert len(summary)==23 and len(configsummary)==20
    for row in summary+configsummary:
        vals=[r for r in recomputed if r['config_id']==row['config_id']]
        assert len(vals)==5 and row['status']=='PASS' and int(row['seed_count'])==5
        for metric in runner.METRICS:
            numbers=[v[metric] for v in vals]
            close(float(row[metric+'_mean']),float(np.mean(numbers)),atol=1e-9)
            close(float(row[metric+'_sd']),float(np.std(numbers,ddof=1)),atol=1e-9)
    expected_memberships={(m['axis'],float(m['value']),point['config_id']) for point in points for m in point['memberships']}
    assert {(r['axis'],float(r['axis_value']),r['config_id']) for r in summary}==expected_memberships
    assert runner.source_hashes()==frozen['source_hashes']
    runner.write_csv(out/'recomputed_per_seed.csv',recomputed)
    report={'status':'PASS','checked_utc':datetime.now(timezone.utc).isoformat(),
        'scope':'All100 saved trajectory/metadata/configuration/summary records recomputed and checked; not an independent reimplementation of QAPG.',
        'runs':100,'configurations':20,'grid_rows':23,'shared_input_files':len(checked_inputs),
        'protocol_sha256':runner.sha(BASE/'PROTOCOL.json'),'freeze_sha256':runner.sha(BASE/'FREEZE.json'),
        'metadata_sha256':runner.sha(out/'metadata.json'),'validator_sha256':runner.sha(__file__),
        'record_hashes':hashes,'original_frozen_sources_unchanged':True,
        'max_slot_flow_error_bits':max_flow,'max_energy_component_error_j':max_energy,
        'max_queue_component_error_bits':max_queue,'max_unused_cpu_capacity_bits':max_unused,
        'max_objective_rise':max_rise,
        'checks':['20one-factor configurations/100independent-seed runs','referenceconfiguration deduplicated','matched input arrays and hash provenance','unchanged old source freeze','bandwidth-scaled noise at fixed PSD','physical resource assertions from common simulator','allslot flow and energy/queue sums','seedmetrics from raw arrays','five-seed means and sampleSD','allgrid points retained']}
    runner.dump(out/'VALIDATION.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='record_hashes'},indent=2))


if __name__=='__main__':main()

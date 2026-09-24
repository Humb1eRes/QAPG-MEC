"""Recompute metrics and ledger identities directly from every saved raw trace."""
from __future__ import annotations
import argparse, hashlib, json, csv, sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
sys.dont_write_bytecode=True
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE))
import run_protocol as runner
ROOT=runner.ROOT

def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def array_digest(a):
    h=hashlib.sha256();h.update(str(a.shape).encode());h.update(a.dtype.str.encode());h.update(np.ascontiguousarray(a).tobytes());return h.hexdigest()
def close(a,b,atol=1e-7,rtol=1e-10): np.testing.assert_allclose(a,b,atol=atol,rtol=rtol)
def validate(stage):
    folder=runner.RESULTS/stage
    p=json.loads((BASE/'PROTOCOL.json').read_text());spec=p[stage]
    meta=json.loads((folder/'metadata.json').read_text())
    assert meta['status']=='COMPLETED' and meta['source_hashes']==runner.source_hashes()
    assert meta['protocol_sha256']==digest(BASE/'PROTOCOL.json')
    if stage!='development':
        freeze=json.loads((BASE/'FREEZE.json').read_text())
        assert freeze['source_hashes']==meta['source_hashes']
        assert freeze['frozen_utc']<meta['started_utc']
    if stage=='test':
        sf=runner.RESULTS/'selection/SELECTED_SETTINGS.json';sel=json.loads(sf.read_text())
        assert sel['status']=='FROZEN_BEFORE_TEST' and digest(sf)==meta['selected_settings_sha256']
        assert sel['frozen_utc']<meta['started_utc']
        settings=sel['unique_settings']
    else:
        settings=[dict(method=m,beta=b,scale_index=p['energy_scales'].index(v),scale=v) for m in p['methods'] for b in p['betas'] for v in spec.get('energy_scales',p['energy_scales'])]
    expected={(r['method'],r['beta'],r['scale_index'],s) for r in settings for s in spec['seeds']}
    manifest=json.loads((folder/'input_manifest.json').read_text());inputmap={}
    arrays={}
    for item in manifest:
        for key in ['arrivals','channels']:
            path=ROOT/item[key+'_path']
            if str(path) not in arrays:
                with np.load(path,allow_pickle=False) as z:arrays[str(path)]=z[key]
            assert array_digest(arrays[str(path)])==item[key+'_array_sha256']
        inputmap[(item['beta'],item['seed'])]=item
    seen=set();out=[];recordhash={};max_flow=max_energy=max_queue=max_unused=0.
    for f in sorted((folder/'records').glob('*.json')):
        r=json.loads(f.read_text());assert r['status']=='PASS';id=r['identity'];row=r['row'];cfg=r['config']
        key=(id['method'],id['beta'],id['scale_index'],id['seed']);assert key in expected and key not in seen;seen.add(key)
        assert id['stage']==stage and id['scale']==p['energy_scales'][id['scale_index']]
        wanted=asdict(runner.sim.rec.original.SimConfig(T=spec['T'],seed=id['seed'],beta=id['beta'],V=p['base_V']*id['scale']))
        if id['method'] in p['scale_adaptive_floor_for']:wanted['adaptive_v_min']=p['base_adaptive_v_min']*id['scale']
        assert cfg==wanted
        assert r['controller_override']==({'server_load_penalty':0} if id['method']=='NoQuad-capped' else {})
        assert r['input']==inputmap[(id['beta'],id['seed'])]
        assert r['checks']['conservation_and_resource_assertions_passed']
        for k in ['stage','method','beta','seed','scale_index']:assert row[k]==id[k]
        assert row['energy_scale']==id['scale'] and row['T']==spec['T'] and row['warmup']==spec['warmup']
        trace=ROOT/row['trace_path'];assert digest(trace)==row['trace_sha256']
        with np.load(trace,allow_pickle=False) as z: a={k:z[k] for k in z.files}
        for k,v in a.items(): assert v.shape[0]==spec['T'] and np.all(np.isfinite(v)),(f,k)
        arrivals=arrays[str(ROOT/r['input']['arrivals_path'])]
        close(a['arrival_bits'],arrivals.sum(axis=1),atol=0,rtol=0)
        completed=a['local_completed_bits']+a['edge_completed_bits']
        energy=a['local_energy_j']+a['transmit_energy_j']+a['edge_energy']
        close(a['total_energy'],energy,atol=1e-12)
        close(a['device_energy'],a['local_energy_j']+a['transmit_energy_j'],atol=1e-12)
        close(a['total_queue'],a['device_queue']+a['edge_queue'])
        oldq=np.r_[0,a['total_queue'][:-1]]
        flow=oldq+a['arrival_bits']-completed-a['total_queue'];close(flow,0,atol=1e-5,rtol=0)
        close(np.r_[0,a['device_queue'][:-1]]+a['arrival_bits']-a['local_completed_bits']-a['uploaded_bits'],a['device_queue'],atol=1e-5)
        close(np.r_[0,a['edge_queue'][:-1]]+a['uploaded_bits']-a['edge_completed_bits'],a['edge_queue'],atol=1e-5)
        close(a['edge_loads'].sum(axis=1),a['uploaded_bits'],atol=1e-5)
        for k in ['total_energy','total_queue','local_completed_bits','edge_completed_bits','uploaded_bits']:assert np.all(a[k]>=-1e-7)
        unused=max(float(np.max(np.abs(a[k]))) for k in ['unused_local_capacity_bits','unused_edge_capacity_bits'])
        assert unused<1e-4,unused
        sl=slice(spec['warmup'],None);count=spec['T']-spec['warmup']
        late=a['total_queue'][spec['T']//2:]/1e6;x=np.arange(len(late),dtype=float);x-=x.mean()
        slope=float(np.dot(x,late-late.mean())/np.dot(x,x))
        calc={
          'mean_energy_j_per_slot':float(energy[sl].mean()),
          'mean_backlog_mbit':float(a['total_queue'][sl].mean()/1e6),
          'mean_device_backlog_mbit':float(a['device_queue'][sl].mean()/1e6),
          'mean_edge_backlog_mbit':float(a['edge_queue'][sl].mean()/1e6),
          'mean_arrival_mbit_per_slot':float(a['arrival_bits'][sl].mean()/1e6),
          'mean_completed_mbit_per_slot':float(completed[sl].mean()/1e6),
          'mean_local_completed_mbit_per_slot':float(a['local_completed_bits'][sl].mean()/1e6),
          'mean_edge_completed_mbit_per_slot':float(a['edge_completed_bits'][sl].mean()/1e6),
          'completion_to_arrival_ratio':float(completed[sl].sum()/a['arrival_bits'][sl].sum()),
          'late_half_backlog_slope_mbit_per_slot':slope,
          'window_start_queue_bits':float(oldq[spec['warmup']]),
          'window_end_queue_bits':float(a['total_queue'][-1]),
          'window_arrival_bits':float(a['arrival_bits'][sl].sum()),
          'window_completed_bits':float(completed[sl].sum()),
          'window_energy_j':float(energy[sl].sum()),
        }
        for k,v in calc.items():close(row[k],v,atol=1e-6 if k.endswith('_bits') else 1e-9)
        calc.update(mean_local_energy_j_per_slot=float(a['local_energy_j'][sl].mean()),mean_transmit_energy_j_per_slot=float(a['transmit_energy_j'][sl].mean()),mean_edge_energy_j_per_slot=float(a['edge_energy'][sl].mean()))
        if id['method']=='QAPG-R':
            close(a['effective_V'],cfg['V'],atol=0,rtol=0)
            assert np.all((a['revised_sweeps']>=1)&(a['revised_sweeps']<=12))
            assert np.max(a['objective_max_rise'])<=1e-9
            close(row['revised_convergence_fraction'],float(a['revised_converged'].mean()))
            close(row['maximum_best_deviation_gain'],float(a['best_deviation_gain'].max()))
        for k in ['revised_convergence_fraction','maximum_best_deviation_gain','maximum_objective_rise','maximum_sweeps']:calc[k]=row[k]
        out.append({**id,**calc,'record_path':str(f.relative_to(ROOT)),'trace_path':row['trace_path']})
        recordhash[str(f.relative_to(ROOT))]=digest(f)
        max_flow=max(max_flow,float(np.abs(flow).max()));max_energy=max(max_energy,float(np.abs(a['total_energy']-energy).max()));max_queue=max(max_queue,float(np.abs(a['total_queue']-a['device_queue']-a['edge_queue']).max()));max_unused=max(max_unused,unused)
    assert seen==expected and len(seen)==meta['planned_runs']==meta['completed_runs']
    with (folder/'recomputed_per_seed.csv').open('w',newline='') as h:
        w=csv.DictWriter(h,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
    report={'status':'PASS','verification_status':'ANALYZED','stage':stage,'checked_utc':datetime.now(timezone.utc).isoformat(),'runs':len(seen),'protocol_sha256':digest(BASE/'PROTOCOL.json'),'metadata_sha256':digest(folder/'metadata.json'),'validator_sha256':digest(__file__),'record_hashes':recordhash,'max_slot_flow_error_bits':max_flow,'max_energy_sum_error_j':max_energy,'max_queue_sum_error_bits':max_queue,'max_unused_cpu_capacity_bits':max_unused,'checks':['exact expected run grid','frozen source and chronology','common hardware and matched input arrays','all saved trace checksums','all slot aggregate conservation and component sums','work-capped CPU on all methods','independently recomputed metrics','finite values and bounds','proposed objective monotonicity diagnostics'],'scope':'Recomputation and trace/metadata audit; simulator checked individual physical decisions during execution. This is not an independent reimplementation of every controller.'}
    (folder/'INDEPENDENT_VALIDATION.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='record_hashes'},indent=2))
    return report
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['development','selection','test'],required=True);validate(ap.parse_args().stage)

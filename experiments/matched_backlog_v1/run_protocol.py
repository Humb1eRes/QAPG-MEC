"""Frozen split-seed matched-backlog experiment runner; no test-set retuning."""
from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import asdict,replace
from datetime import datetime,timezone
from pathlib import Path
import argparse,csv,hashlib,json,multiprocessing,platform,sys,time
import numpy as np
sys.dont_write_bytecode=True
BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[1]
sys.path.insert(0,str(BASE))
import common_bridge as bridge
sim=bridge.sim
RESULTS=ROOT/'experiments/results/matched_backlog_v1'
CACHE={}


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2))
def now():return datetime.now(timezone.utc).isoformat()
def save_csv(p,rows):
 if not rows:return
 with p.open('w',newline='') as h:
  w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def source_hashes():
 files=[BASE/'PROTOCOL.json',BASE/'common_bridge.py',BASE/'run_protocol.py',BASE/'qapg_revised.py',BASE/'qapg_revised.cpp']
 files += [ROOT/'experiments'/x for x in ['reference/edgesport_sim.py','matched_recorder.py','run_matched.py','literature_baselines/common_simulator_v2.py','literature_baselines/eedo.py','literature_baselines/gupa_orthogonal.py']]
 return {str(p.relative_to(ROOT)):sha(p) for p in files}


def inputs_for(stage,protocol):
 spec=protocol[stage];folder=RESULTS/stage/'inputs';folder.mkdir(parents=True,exist_ok=True);manifest=[]
 for seed in spec['seeds']:
  cfg=sim.rec.original.SimConfig(T=spec['T'],seed=seed)
  cp=folder/f'channels_seed{seed}.npz'
  if cp.exists():
   with np.load(cp) as z:channels=z['channels']
  else:
   channels=sim.rec.original.generate_channels(cfg);np.savez_compressed(cp,channels=channels)
  for beta in protocol['betas']:
   ap=folder/f'arrivals_beta{beta}_seed{seed}.npz'
   if ap.exists():
    with np.load(ap) as z:arrivals=z['arrivals']
   else:
    arrivals,_=sim.rec.original.make_arrivals(replace(cfg,beta=beta));np.savez_compressed(ap,arrivals=arrivals)
   manifest.append({'beta':beta,'seed':seed,'T':spec['T'],'channels_path':str(cp.relative_to(ROOT)),'arrivals_path':str(ap.relative_to(ROOT)),'channels_array_sha256':bridge.array_sha(channels),'arrivals_array_sha256':bridge.array_sha(arrivals)})
 dump(RESULTS/stage/'input_manifest.json',manifest)
 return manifest


def array(path,key):
 if path not in CACHE:
  with np.load(ROOT/path,allow_pickle=False) as z:a=z[key]
  a.flags.writeable=False;CACHE[path]=a
 return CACHE[path]


def run_one(job):
 start=time.perf_counter();method=job['method'];scale=job['scale']
 cfg=sim.rec.original.SimConfig(T=job['T'],seed=job['seed'],beta=job['beta'],V=job['base_V']*scale)
 if method in {'QAPG-capped','NoQuad-capped'}:cfg=replace(cfg,adaptive_v_min=job['base_floor']*scale)
 item=job['input'];arrivals=array(item['arrivals_path'],'arrivals');channels=array(item['channels_path'],'channels')
 assert bridge.array_sha(arrivals)==item['arrivals_array_sha256'] and bridge.array_sha(channels)==item['channels_array_sha256']
 result,checks=bridge.simulate(method,cfg,arrivals,channels)
 row={'stage':job['stage'],'method':method,'beta':job['beta'],'seed':job['seed'],'scale_index':job['scale_index'],'energy_scale':scale,'V':cfg.V,'T':cfg.T,'warmup':job['warmup'],**bridge.metrics(result,job['warmup'])}
 row['mean_completed_mbit_per_slot']=row['window_completed_mbit']/(cfg.T-job['warmup'])
 row['revised_convergence_fraction']=float(np.mean(result['revised_converged'])) if method=='QAPG-R' else ''
 row['maximum_best_deviation_gain']=float(np.max(result['best_deviation_gain'])) if method=='QAPG-R' else ''
 row['maximum_objective_rise']=float(np.max(result['objective_max_rise'])) if method=='QAPG-R' else ''
 row['maximum_sweeps']=int(np.max(result['revised_sweeps'])) if method=='QAPG-R' else ''
 row['wall_seconds']=time.perf_counter()-start
 trace=Path(job['trace']);np.savez_compressed(trace,**result)
 row['trace_path']=str(trace.relative_to(ROOT));row['trace_sha256']=sha(trace)
 record={'status':'PASS','identity':{k:job[k] for k in ['stage','method','beta','seed','scale_index','scale']},'config':asdict(cfg),'controller_override':{'server_load_penalty':0} if method=='NoQuad-capped' else {},'row':row,'input':item,'checks':checks,'recorded_utc':now()}
 assert bridge.array_sha(arrivals)==item['arrivals_array_sha256'] and bridge.array_sha(channels)==item['channels_array_sha256']
 dump(Path(job['record']),record)
 return row


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['development','selection','test'],required=True);ap.add_argument('--workers',type=int,default=4);ap.add_argument('--resume',action='store_true');args=ap.parse_args()
 protocol=json.loads((BASE/'PROTOCOL.json').read_text());stage=args.stage;spec=protocol[stage]
 if stage!='development':
  freeze=json.loads((BASE/'FREEZE.json').read_text());assert freeze['source_hashes']==source_hashes(),'Frozen sources changed'
  assert freeze['protocol_sha256']==sha(BASE/'PROTOCOL.json')
 if stage=='test':
  selection=json.loads((RESULTS/'selection/metadata.json').read_text());assert selection['status']=='COMPLETED'
  chosen=json.loads((RESULTS/'selection/SELECTED_SETTINGS.json').read_text());assert chosen['status']=='FROZEN_BEFORE_TEST';assert chosen['protocol_sha256']==sha(BASE/'PROTOCOL.json')
 out=RESULTS/stage
 if (out/'metadata.json').exists() and not args.resume:raise FileExistsError(f'{out} already exists; explicit --resume required')
 out.mkdir(parents=True,exist_ok=True);(out/'traces').mkdir(exist_ok=True);(out/'records').mkdir(exist_ok=True)
 hashes=source_hashes()
 if args.resume and (out/'metadata.json').exists():
  prior=json.loads((out/'metadata.json').read_text());assert prior['source_hashes']==hashes
 metadata={'status':'RUNNING','stage':stage,'started_utc':now(),'protocol_sha256':sha(BASE/'PROTOCOL.json'),'source_hashes':hashes,'numpy':np.__version__,'python':sys.version,'platform':platform.platform(),'workers':args.workers}
 if stage=='test':metadata['selected_settings_sha256']=sha(RESULTS/'selection/SELECTED_SETTINGS.json')
 dump(out/'metadata.json',metadata)
 inputs=inputs_for(stage,protocol);jobs=[]
 if stage=='test':settings=chosen['unique_settings']
 else:
  scales=spec.get('energy_scales',protocol['energy_scales'])
  settings=[{'method':method,'beta':beta,'scale_index':protocol['energy_scales'].index(scale),'scale':scale} for method in protocol['methods'] for beta in protocol['betas'] for scale in scales]
 for item in settings:
  for seed in spec['seeds']:
   name=f"b{item['beta']}_s{seed}_{item['method']}_v{item['scale_index']}"
   jobs.append(dict(item,seed=seed,stage=stage,T=spec['T'],warmup=spec['warmup'],base_V=protocol['base_V'],base_floor=protocol['base_adaptive_v_min'],input=next(r for r in inputs if r['seed']==seed and r['beta']==item['beta']),trace=str(out/'traces'/f'{name}.npz'),record=str(out/'records'/f'{name}.json')))
 metadata['planned_runs']=len(jobs);dump(out/'metadata.json',metadata)
 rows=[];pending=[]
 for job in jobs:
  if Path(job['record']).exists():
   r=json.loads(Path(job['record']).read_text());assert r['status']=='PASS' and sha(Path(job['trace']))==r['row']['trace_sha256'];rows.append(r['row'])
  else:pending.append(job)
 started=time.perf_counter()
 try:
  with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
   futures={pool.submit(run_one,j):j for j in pending}
   for future in as_completed(futures):
    rows.append(future.result());save_csv(out/'per_seed.csv',sorted(rows,key=lambda r:(r['method'],r['beta'],r['scale_index'],r['seed'])))
    if len(rows)%12==0 or len(rows)==len(jobs):print(json.dumps({'stage':stage,'completed':len(rows),'planned':len(jobs),'elapsed_seconds':round(time.perf_counter()-started,1)}),flush=True)
  assert hashes==source_hashes()
  save_csv(out/'per_seed.csv',sorted(rows,key=lambda r:(r['method'],r['beta'],r['scale_index'],r['seed'])))
  metadata.update(status='COMPLETED',finished_utc=now(),completed_runs=len(rows),duration_seconds=time.perf_counter()-started)
 except BaseException as e:
  metadata.update(status='FAILED',error=repr(e),finished_utc=now());dump(out/'metadata.json',metadata);raise
 dump(out/'metadata.json',metadata)
 print(json.dumps({'stage':stage,'status':metadata['status'],'runs':len(rows)}),flush=True)

if __name__=='__main__':main()

"""Controller API/ledger checks; mathematical checks are a separate review."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

BASE=Path(__file__).resolve().parent
sys.path[:0]=[str(BASE),str(BASE.parent/'literature_baselines'),str(BASE.parent)]
from qapg_revised import qapg_revised_decision
from common_simulator_v2 import validate_decision
import matched_recorder as rec


def main():
    cfg=replace(rec.original.SimConfig(),N=8,M=3)
    rng=np.random.default_rng(20260922)
    checks=[]
    timings=[]
    worst_rise=0.0
    for i in range(30):
        c=replace(cfg,V=cfg.V*(0.03,0.1,1,10,30)[i%5],beta=(1,2,3)[i%3])
        Q=rng.uniform(0,2e6,c.N)
        B=rng.uniform(0,4e5,(c.N,c.M)); H=B.sum(axis=0)
        gains=np.exp(rng.uniform(np.log(1e-12),np.log(1e-7),B.shape))
        gains[rng.random(B.shape)<.1]=0
        if i==0: Q[:]=0;B[:]=0;H[:]=0
        if i==1: gains[:]=0
        if i==2: c=replace(c,p_max_w=0)
        if i==3: c=replace(c,local_f_max_hz=0,edge_f_max_hz=0)
        original=[x.copy() for x in (Q,H,B,gains)]
        for x in (Q,H,B,gains): x.flags.writeable=False
        start=time.perf_counter(); d=qapg_revised_decision(Q,H,B,gains,c)
        timings.append(1000*(time.perf_counter()-start))
        local,edge=validate_decision(d,Q,H,B,gains,c)
        # Tight work caps imply no positive CPU charge for unused capacity.
        np.testing.assert_allclose(local,c.tau*d.local_cpu_hz/c.cycles_per_bit,atol=1e-8,rtol=1e-12)
        np.testing.assert_allclose(edge,c.tau*d.edge_cpu_hz/c.edge_cycles_per_bit,atol=1e-8,rtol=1e-12)
        for old,now in zip(original,(Q,H,B,gains)): np.testing.assert_array_equal(old,now)
        # A seed is not an observation or a source of randomized control.
        repeated=qapg_revised_decision(Q,H,B,gains,replace(c,seed=99991))
        for key in ('local_cpu_hz','transmit_power_w','association','uploads_bits','edge_cpu_hz','edge_service_bits'):
            np.testing.assert_array_equal(getattr(d,key),getattr(repeated,key))
        if i in (0,1,2):
            assert np.all(d.association==-1) and not np.any(d.transmit_power_w) and not np.any(d.uploads_bits)
        if i==0: assert not np.any(d.local_cpu_hz) and not np.any(d.edge_cpu_hz)
        diag=d.diagnostics
        assert 1<=diag['revised_sweeps']<=12
        assert diag['revised_converged']==(diag['best_deviation_gain']<=1e-10)
        assert diag['objective_final']<=diag['objective_initial']+1e-10
        assert diag['objective_max_rise']<=1e-10
        worst_rise=max(worst_rise,diag['objective_max_rise'])
        checks.append({'case':i,'status':'PASS','sweeps':diag['revised_sweeps'],
                       'converged':diag['revised_converged'],'best_deviation_gain':diag['best_deviation_gain']})
    badQ=np.full(cfg.N,-1.)
    try: qapg_revised_decision(badQ,np.zeros(cfg.M),np.zeros((cfg.N,cfg.M)),np.ones((cfg.N,cfg.M)),cfg)
    except ValueError: checks.append({'case':'negative_queue_rejected','status':'PASS'})
    else: raise AssertionError('Negative queue was not rejected')
    hashes={name:hashlib.sha256((BASE/name).read_bytes()).hexdigest()
            for name in ('qapg_revised.py','qapg_revised.cpp')}
    result={'status':'PASS','test_count':len(checks),'controller_hashes':hashes,
            'max_objective_rise':worst_rise,'mean_call_ms_N8_M3':float(np.mean(timings)),
            'checks':checks,'scope':'API feasibility, work caps, ledger, deterministic control, immutability, edge cases. Independent mathematical review is separate.'}
    (BASE/'QAPG_R_API_VALIDATION.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('checks',)},indent=2))


if __name__=='__main__': main()

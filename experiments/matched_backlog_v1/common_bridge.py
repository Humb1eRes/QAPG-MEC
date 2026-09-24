"""Common CPU correction and source-frozen controllers for matched-backlog runs."""
from dataclasses import replace
from pathlib import Path
import sys
import numpy as np
sys.dont_write_bytecode=True
BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[1]
sys.path.insert(0,str(ROOT/'experiments/literature_baselines'))
import common_simulator_v2 as sim
from run_matched import metrics,array_sha
METHODS={'QAPG-capped':'QAPG','NoQuad-capped':'No-quadratic-load','GUPA-O-capped':'GUPA-O','EEDO-adapted':'EEDO-adapted','BP-Greedy-capped':'BP-Greedy'}


def capped(d,Q,cfg):
    local=np.minimum(Q,cfg.tau*d.local_cpu_hz/cfg.cycles_per_bit)
    edge=d.diagnostics.get('aggregate_edge_service_bits',d.edge_service_bits.sum(axis=0))
    f=np.minimum(d.local_cpu_hz,local*cfg.cycles_per_bit/cfg.tau*(1+1e-13))
    F=np.minimum(d.edge_cpu_hz,edge*cfg.edge_cycles_per_bit/cfg.tau*(1+1e-13))
    np.testing.assert_array_equal(np.minimum(Q,cfg.tau*f/cfg.cycles_per_bit),local)
    assert np.all(edge<=cfg.tau*F/cfg.edge_cycles_per_bit+1e-7)
    return replace(d,local_cpu_hz=f,edge_cpu_hz=F)


def simulate(method,cfg,arrivals,channels):
    original={name:getattr(sim,name) for name in ['legacy_decision','gupa_orthogonal_decision','eedo_decision']}
    extra={'sweeps':[],'converged':[],'best_gain':[],'objective_rise':[]}
    if method=='QAPG-R':
        from qapg_revised import qapg_revised_decision
        def legacy(key,Q,H,B,g,c,rng,disable):
            d=qapg_revised_decision(Q,H,B,g,c)
            dg=d.diagnostics
            extra['sweeps'].append(dg.get('revised_sweeps',0));extra['converged'].append(bool(dg.get('revised_converged',False)))
            extra['best_gain'].append(float(dg.get('best_deviation_gain',0)))
            extra['objective_rise'].append(float(dg.get('objective_max_rise',0)))
            return capped(d,Q,c)
        sim.legacy_decision=legacy
        policy='QAPG'
    else:
        policy=METHODS[method]
        def legacy(key,Q,H,B,g,c,rng,disable):return capped(original['legacy_decision'](key,Q,H,B,g,c,rng,disable),Q,c)
        def gupa(Q,H,B,g,c):return capped(original['gupa_orthogonal_decision'](Q,H,B,g,c),Q,c)
        def eedo(Q,B,g,c):return capped(original['eedo_decision'](Q,B,g,c),Q,c)
        sim.legacy_decision=legacy;sim.gupa_orthogonal_decision=gupa;sim.eedo_decision=eedo
    try:
        result,checks=sim.simulate(policy,cfg,arrivals,channels)
        if method=='QAPG-R':
            result['revised_sweeps']=np.asarray(extra['sweeps'])
            result['revised_converged']=np.asarray(extra['converged'])
            result['best_deviation_gain']=np.asarray(extra['best_gain'])
            result['objective_max_rise']=np.asarray(extra['objective_rise'])
        return result,checks
    finally:
        for name,value in original.items():setattr(sim,name,value)

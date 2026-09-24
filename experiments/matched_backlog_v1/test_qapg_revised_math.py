"""Independent direct-objective checks; does not copy production root formulas."""
from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
from dataclasses import replace

import numpy as np

sys.dont_write_bytecode = True
BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parent))
import matched_recorder as recorder
from qapg_revised import qapg_revised_decision
import qapg_revised as production_controller


def bounded_minimum(function, upper):
    """Derivative-free convex scalar search, including both endpoints."""
    if upper <= 0:
        return 0.0, function(0.0)
    lo, hi = 0.0, float(upper)
    ratio = (math.sqrt(5)-1)/2
    x1, x2 = hi-ratio*(hi-lo), lo+ratio*(hi-lo)
    f1, f2 = function(x1), function(x2)
    for _ in range(68):
        if f1 <= f2:
            hi, x2, f2 = x2, x1, f1
            x1 = hi-ratio*(hi-lo)
            f1 = function(x1)
        else:
            lo, x1, f1 = x1, x2, f2
            x2 = lo+ratio*(hi-lo)
            f2 = function(x2)
    candidates = [(0.0, function(0.0)), (upper, function(upper)),
                  (x1, f1), (x2, f2)]
    return min(candidates, key=lambda pair: pair[1])


def parameters(cfg):
    return dict(v=cfg.V/1e12, w=cfg.M/cfg.N,
                arrival=cfg.beta*cfg.arrival_lambda*cfg.task_packet_bits/1e6,
                kd=cfg.kappa_device*cfg.cycles_per_bit**3*1e18/cfg.tau**2,
                ke=cfg.kappa_edge*cfg.edge_cycles_per_bit**3*1e18/cfg.tau**2,
                k=math.log(2)*1e6/(cfg.tau*cfg.bandwidth_hz),
                lmax=cfg.tau*cfg.local_f_max_hz/cfg.cycles_per_bit/1e6,
                dmax=cfg.tau*cfg.edge_f_max_hz/cfg.edge_cycles_per_bit/1e6)


def local_oracle(q, upload, cfg):
    p = parameters(cfg)
    s = q+p['arrival']-upload
    return bounded_minimum(lambda local: .5*(s-local)**2+p['v']*p['kd']*local**3,
                           min(p['lmax'], max(0.,q-upload)))


def device_oracle(q, gain, others, cfg):
    p = parameters(cfg)
    if gain <= 0 or cfg.p_max_w <= 0:
        local, value = local_oracle(q, 0, cfg)
        return value, local, 0.
    umax = min(q, cfg.tau*cfg.bandwidth_hz*math.log2(1+gain*cfg.p_max_w/cfg.noise_w)/1e6)
    def cost(upload):
        _, local_value = local_oracle(q, upload, cfg)
        radio = cfg.tau*cfg.noise_w/gain*math.expm1(p['k']*upload)
        return local_value+p['v']*radio+p['w']*(others*upload+.5*upload**2)
    upload, value = bounded_minimum(cost, umax)
    local, _ = local_oracle(q, upload, cfg)
    return value, local, upload


def edge_oracle(h, load, cfg):
    p = parameters(cfg)
    return bounded_minimum(lambda done: .5*p['w']*(h+load-done)**2+p['v']*p['ke']*done**3,
                           min(h,p['dmax']))


def objective(q,h,l,u,d,a,gains,cfg):
    p=parameters(cfg)
    loads=np.zeros(cfg.M)
    energy=p['kd']*sum(l**3)+p['ke']*sum(d**3)
    for n,m in enumerate(a):
        if m>=0:
            loads[m]+=u[n]
            energy+=cfg.tau*cfg.noise_w/gains[n,m]*math.expm1(p['k']*u[n])
    return .5*sum((q+p['arrival']-l-u)**2)+.5*p['w']*sum((h-d+loads)**2)+p['v']*energy


def physical_decision(q,h,gains,cfg):
    B=np.tile(h*1e6/cfg.N,(cfg.N,1))
    decision=qapg_revised_decision(q*1e6,h*1e6,B,gains,cfg)
    l=decision.local_cpu_hz*cfg.tau/cfg.cycles_per_bit/1e6
    u=decision.uploads_bits/1e6
    d=decision.edge_service_bits.sum(axis=0)/1e6
    return decision,l,u,d,decision.association


def check_feasible(q,h,gains,cfg,decision,l,u,d,a):
    p=parameters(cfg)
    assert np.all(l>=0) and np.all(u>=0) and np.all(d>=0)
    assert np.all(l+u<=q+1e-12) and np.all(l<=p['lmax']+1e-12)
    assert np.all(d<=h+1e-12) and np.all(d<=p['dmax']+1e-12)
    assert np.all((a>=-1)&(a<cfg.M))
    assert np.all(u[a<0]==0) and np.all(decision.transmit_power_w[a<0]==0)
    assert np.all(decision.transmit_power_w<=cfg.p_max_w+1e-13)
    active=np.flatnonzero(a>=0)
    delivered=cfg.tau*cfg.bandwidth_hz*np.log1p(decision.transmit_power_w[active]*gains[active,a[active]]/cfg.noise_w)/np.log(2)/1e6
    np.testing.assert_allclose(u[active],delivered,rtol=1e-10,atol=1e-12)
    np.testing.assert_allclose(d*1e6,decision.diagnostics['aggregate_edge_service_bits'],rtol=1e-12,atol=1e-7)
    np.testing.assert_allclose(d,cfg.tau*decision.edge_cpu_hz/cfg.edge_cycles_per_bit/1e6,rtol=1e-12,atol=1e-12)


def run():
    cfg=recorder.original.SimConfig()
    rng=np.random.default_rng(929226)
    records=[]
    maximum_single_gap=maximum_objective_error=maximum_residual_error=0.

    # Conversion checks are independent of the native optimizer.
    for _ in range(40):
        p=parameters(cfg)
        l=rng.uniform(0,p['lmax']); d=rng.uniform(0,p['dmax'])
        gain=10**rng.uniform(-12,-8); power=rng.uniform(0,cfg.p_max_w)
        upload=cfg.tau*cfg.bandwidth_hz*math.log2(1+power*gain/cfg.noise_w)/1e6
        f=l*1e6*cfg.cycles_per_bit/cfg.tau; F=d*1e6*cfg.edge_cycles_per_bit/cfg.tau
        np.testing.assert_allclose(p['kd']*l**3,cfg.kappa_device*cfg.tau*f**3,rtol=1e-12)
        np.testing.assert_allclose(p['ke']*d**3,cfg.kappa_edge*cfg.tau*F**3,rtol=1e-12)
        np.testing.assert_allclose(cfg.tau*cfg.noise_w/gain*math.expm1(p['k']*upload),cfg.tau*power,rtol=1e-12)
    records.append({'check':'physical_energy_conversion','cases':40,'status':'PASS'})

    # One user with empty old edge queues has only one free device block;
    # exhaustive server enumeration plus two nested direct-cost minimizers
    # gives an independent global oracle for this bounded special case.
    for case in range(28):
        scale=(1/16,1/4,1,4,16)[case%5]
        current=replace(cfg,N=1,M=3,V=cfg.V*scale,beta=(1,2,3)[case%3],
                        local_f_max_hz=0. if case%7==0 else cfg.local_f_max_hz,
                        p_max_w=0. if case%11==0 else cfg.p_max_w)
        q=np.array([0. if case==0 else 10**rng.uniform(-3,-.1)])
        h=np.zeros(current.M); gains=10**rng.uniform(-12,-8,(1,current.M))
        if case%9==0: gains[:]=0
        result,l,u,d,a=physical_decision(q,h,gains,current)
        check_feasible(q,h,gains,current,result,l,u,d,a)
        _,null_cost=local_oracle(q[0],0,current)
        best=min([null_cost]+[device_oracle(q[0],gain,0,current)[0] for gain in gains[0]])
        observed=objective(q,h,l,u,d,a,gains,current)
        gap=observed-best
        maximum_single_gap=max(maximum_single_gap,abs(gap))
        assert -2e-9<=gap<=2e-9, (case,observed,best)
        assert result.diagnostics['revised_converged']
    records.append({'check':'one_device_all_servers_global_oracle','cases':28,'status':'PASS','maximum_absolute_cost_difference':maximum_single_gap})

    # Multi-user states: physical ledger, independent global objective,
    # initialization, edge optima, and every final unilateral deviation.
    nonconverged=0
    for case in range(12):
        current=replace(cfg,N=4,M=3,V=cfg.V*(1/16,1/4,1,4,16)[case%5],beta=(1,2,3)[case%3])
        q=rng.uniform(0,.65,current.N); h=rng.uniform(0,2.8,current.M)
        gains=10**rng.uniform(-12,-8,(current.N,current.M))
        if case==0: q[:]=0
        if case==1: h[:]=0
        if case==2: gains[:,0]=0
        result,l,u,d,a=physical_decision(q,h,gains,current)
        check_feasible(q,h,gains,current,result,l,u,d,a)
        observed=objective(q,h,l,u,d,a,gains,current)
        error=abs(observed-result.diagnostics['objective_final'])
        maximum_objective_error=max(maximum_objective_error,error)
        np.testing.assert_allclose(observed,result.diagnostics['objective_final'],rtol=1e-11,atol=1e-11)
        init_l=np.array([local_oracle(qn,0,current)[0] for qn in q])
        init_d=np.array([edge_oracle(hm,0,current)[0] for hm in h])
        initial=objective(q,h,init_l,np.zeros(current.N),init_d,np.full(current.N,-1),gains,current)
        np.testing.assert_allclose(initial,result.diagnostics['objective_initial'],rtol=1e-10,atol=1e-10)
        assert observed<=initial+1e-10
        assert result.diagnostics['objective_max_rise']<=1e-10
        loads=np.zeros(current.M)
        for n,m in enumerate(a):
            if m>=0: loads[m]+=u[n]
        for m in range(current.M):
            best_d,best_cost=edge_oracle(h[m],loads[m],current)
            p=parameters(current)
            actual=.5*p['w']*(h[m]+loads[m]-d[m])**2+p['v']*p['ke']*d[m]**3
            assert actual-best_cost<=1e-10
        largest=0.
        for n in range(current.N):
            others=loads.copy()
            if a[n]>=0: others[a[n]]-=u[n]
            for m in range(-1,current.M):
                if m<0:
                    new_l,_=local_oracle(q[n],0,current); new_u=0.
                else:
                    _,new_l,new_u=device_oracle(q[n],gains[n,m],h[m]-d[m]+others[m],current)
                l_new=l.copy(); u_new=u.copy(); a_new=a.copy()
                l_new[n]=new_l; u_new[n]=new_u; a_new[n]=m if new_u>0 else -1
                candidate=objective(q,h,l_new,u_new,d,a_new,gains,current)
                largest=max(largest,observed-candidate)
        residual_error=abs(largest-result.diagnostics['best_deviation_gain'])
        maximum_residual_error=max(maximum_residual_error,residual_error)
        assert residual_error<=2e-9, (case,largest,result.diagnostics['best_deviation_gain'])
        if result.diagnostics['revised_converged']:
            assert largest<=2e-9
        else:
            nonconverged+=1
        # Seed changes cannot reveal current/future arrivals or alter this
        # deterministic controller's decision at an identical observed state.
        again=qapg_revised_decision(q*1e6,h*1e6,np.tile(h*1e6/current.N,(current.N,1)),gains,replace(current,seed=123456789))
        for field in ('local_cpu_hz','transmit_power_w','association','uploads_bits','edge_cpu_hz','edge_service_bits'):
            np.testing.assert_array_equal(getattr(result,field),getattr(again,field))
    records.append({'check':'multi_user_objective_feasibility_edge_and_unilateral_oracles','cases':12,'status':'PASS','maximum_objective_error':maximum_objective_error,'maximum_residual_error':maximum_residual_error,'finite_cap_nonconverged_cases_retained':nonconverged})
    records.append({'check':'identical_observed_state_independent_of_seed','cases':12,'status':'PASS'})

    # Low old edge queues force simultaneous offloading interactions. Include
    # the actual N=100,M=10 normalization and deliberately stop after one
    # sweep to verify the reported positive residual and nonconvergence flag.
    interacting=[]
    for n,m in ((12,3),(100,10)):
        current=replace(cfg,N=n,M=m)
        q=rng.uniform(.04,.16,n); h=np.zeros(m)
        gains=10**rng.uniform(-10,-8,(n,m))
        for sweeps in (1,12):
            old_budget=production_controller.MAX_SWEEPS
            try:
                production_controller.MAX_SWEEPS=sweeps
                result,l,u,d,a=physical_decision(q,h,gains,current)
            finally:
                production_controller.MAX_SWEEPS=old_budget
            check_feasible(q,h,gains,current,result,l,u,d,a)
            observed=objective(q,h,l,u,d,a,gains,current)
            loads=np.zeros(m)
            for i,server in enumerate(a):
                if server>=0: loads[server]+=u[i]
            largest=0.
            p=parameters(current)
            for i in range(n):
                others=loads.copy()
                if a[i]>=0: others[a[i]]-=u[i]
                old_cost=.5*(q[i]+p['arrival']-l[i]-u[i])**2+p['v']*p['kd']*l[i]**3
                if a[i]>=0:
                    old_cost+=p['v']*current.tau*current.noise_w/gains[i,a[i]]*math.expm1(p['k']*u[i])
                    old_cost+=p['w']*((h[a[i]]-d[a[i]]+others[a[i]])*u[i]+.5*u[i]**2)
                _,best=local_oracle(q[i],0,current)
                for server in range(m):
                    best=min(best,device_oracle(q[i],gains[i,server],h[server]-d[server]+others[server],current)[0])
                largest=max(largest,old_cost-best)
            np.testing.assert_allclose(largest,result.diagnostics['best_deviation_gain'],rtol=1e-6,atol=2e-10)
            np.testing.assert_allclose(observed,result.diagnostics['objective_final'],rtol=1e-11,atol=1e-11)
            if sweeps==1:
                assert largest>1e-8 and not result.diagnostics['revised_converged']
            elif result.diagnostics['revised_converged']:
                assert largest<=2e-10
            interacting.append({'N':n,'M':m,'sweep_budget':sweeps,'active_uploads':int(sum(u>0)),
                                'independent_best_deviation_gain':largest,
                                'reported_converged':bool(result.diagnostics['revised_converged'])})
    records.append({'check':'interacting_uploads_nominal_hardware_and_nonconvergence_reporting','cases':interacting,'status':'PASS'})

    # Exact finite-distribution expectation, no Monte Carlo approximation.
    current=replace(cfg,N=2,M=2)
    q=np.array([.2,.35]); h=np.array([.4,.6]); gains=np.array([[1e-10,3e-10],[2e-10,1e-9]])
    result,l,u,d,a=physical_decision(q,h,gains,current)
    p=parameters(current); delta=.5*p['arrival']
    loads=np.zeros(current.M)
    for n,m in enumerate(a):
        if m>=0: loads[m]+=u[n]
    energy=(cfg.kappa_device*cfg.tau*sum(result.local_cpu_hz**3)+cfg.tau*sum(result.transmit_power_w)+cfg.kappa_edge*cfg.tau*sum(result.edge_cpu_hz**3))
    values=[]
    for signs in itertools.product((-1,1),repeat=current.N):
        arrivals=p['arrival']+delta*np.array(signs)
        values.append(.5*sum((q-l-u+arrivals)**2)+.5*p['w']*sum((h-d+loads)**2)+p['v']*energy)
    expected=objective(q,h,l,u,d,a,gains,current)+.5*current.N*delta**2
    np.testing.assert_allclose(np.mean(values),expected,rtol=1e-12,atol=1e-12)
    records.append({'check':'conditional_Lyapunov_expectation_with_variance_constant','finite_arrival_outcomes':4,'status':'PASS'})

    # Random feasible unilateral moves test the algebraic potential identity
    # away from optimized states, including server switches and null actions.
    max_identity_error=0.
    for _ in range(120):
        current=replace(cfg,N=3,M=2)
        p=parameters(current); q=rng.uniform(.1,.6,3); h=rng.uniform(0,2,2)
        gains=10**rng.uniform(-11,-8,(3,2)); d=np.minimum(h*.3,p['dmax'])
        a=rng.integers(-1,2,3); l=np.minimum(q*.2,p['lmax']); u=np.zeros(3)
        for n,m in enumerate(a):
            if m>=0:
                cap=current.tau*current.bandwidth_hz*math.log2(1+gains[n,m]*current.p_max_w/current.noise_w)/1e6
                u[n]=min((q[n]-l[n])*.2,cap*.2)
        n=int(rng.integers(0,3)); mnew=int(rng.integers(-1,2))
        ln=float(min(q[n]*rng.uniform(0,.8),p['lmax'])); un=0.
        if mnew>=0:
            cap=current.tau*current.bandwidth_hz*math.log2(1+gains[n,mnew]*current.p_max_w/current.noise_w)/1e6
            un=min((q[n]-ln)*rng.uniform(0,.8),cap*.7)
        loads=np.zeros(2)
        for j,m in enumerate(a):
            if j!=n and m>=0: loads[m]+=u[j]
        def conditional(local,upload,server):
            cost=.5*(q[n]+p['arrival']-local-upload)**2+p['v']*p['kd']*local**3
            if server>=0:
                cost+=p['v']*current.tau*current.noise_w/gains[n,server]*math.expm1(p['k']*upload)
                cost+=p['w']*((h[server]-d[server]+loads[server])*upload+.5*upload**2)
            return cost
        before=objective(q,h,l,u,d,a,gains,current)
        local_difference=conditional(ln,un,mnew)-conditional(l[n],u[n],a[n])
        l[n]=ln;u[n]=un;a[n]=mnew
        global_difference=objective(q,h,l,u,d,a,gains,current)-before
        max_identity_error=max(max_identity_error,abs(global_difference-local_difference))
        np.testing.assert_allclose(global_difference,local_difference,rtol=1e-10,atol=1e-11)
    records.append({'check':'unilateral_exact_potential_identity','cases':120,'status':'PASS','maximum_absolute_difference':max_identity_error})

    report={'status':'PASS','scope':'independent analytic/numerical per-slot correctness; not global stochastic optimality or comparative performance',
            'numpy_version':np.__version__,'controller_sha256':{path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in (BASE/'qapg_revised.py',BASE/'qapg_revised.cpp')},
            'checks':records}
    (BASE/'QAPG_REVISED_MATH_VALIDATION.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    run()

"""QAPG-R controller for the independently recorded matched-backlog experiment.

This controller minimizes an expected one-slot weighted quadratic queue objective
with energy, using fixed V and hardware-derived edge weight M/N. The finite cyclic block budget provides monotone descent,
not a claim of global stochastic optimality. All scientific source is in the
adjacent C++ file; the library is compiled locally with standard IEEE arithmetic.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

BASE = Path(__file__).resolve().parent
LITERATURE = BASE.parent / "literature_baselines"
if str(LITERATURE) not in sys.path:
    sys.path.insert(0, str(LITERATURE))
from eedo import EEDODecision

MAX_SWEEPS = 12
IMPROVEMENT_TOLERANCE = 1e-10
_ENGINE = None


def _load_engine():
    global _ENGINE
    if _ENGINE is not None:
        return _ENGINE
    source = BASE / "qapg_revised.cpp"
    flags = ["-O3", "-std=c++17", "-shared", "-fPIC"]
    fingerprint = hashlib.sha256(source.read_bytes()+" ".join(flags).encode()).hexdigest()[:20]
    extension = ".dylib" if sys.platform == "darwin" else ".so"
    build = BASE / "native_build"
    build.mkdir(exist_ok=True)
    library = build / f"qapg_revised_{fingerprint}{extension}"
    if not library.exists():
        compiler = shutil.which("clang++") or shutil.which("g++")
        if compiler is None:
            raise RuntimeError("QAPG-R requires clang++ or g++ for its portable C++17 controller")
        temporary = library.with_name(library.name+f".{os.getpid()}.tmp")
        result = subprocess.run([compiler,*flags,str(source),"-o",str(temporary)],
                                check=True,capture_output=True,text=True)
        os.replace(temporary,library)
        metadata = {"source_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
                    "compiler":compiler,"flags":flags,"stdout":result.stdout,"stderr":result.stderr}
        library.with_suffix(library.suffix+".json").write_text(json.dumps(metadata,indent=2)+"\n")
    handle = ctypes.CDLL(str(library))
    function = handle.qapg_optimize
    doubles = np.ctypeslib.ndpointer(dtype=np.float64,flags="C_CONTIGUOUS")
    integers = np.ctypeslib.ndpointer(dtype=np.int32,flags="C_CONTIGUOUS")
    function.argtypes = [ctypes.c_int,ctypes.c_int,doubles,doubles,doubles,doubles,
                        ctypes.c_int,ctypes.c_double,doubles,doubles,doubles,integers,doubles]
    function.restype = ctypes.c_int
    _ENGINE = (handle,function,library)
    return _ENGINE


def by_origin_service(B, aggregate):
    """Only an origin ledger: no per-origin service constraints beyond work present."""
    service = np.zeros_like(B)
    for m, amount in enumerate(aggregate):
        remaining = float(amount)
        for n in np.argsort(-B[:,m],kind="stable"):
            if remaining <= 0:
                break
            served = min(float(B[n,m]),remaining)
            service[n,m] = served
            remaining -= served
        if remaining > 1e-5:
            raise AssertionError("Aggregate service exceeds origin ledger")
    return service


def qapg_revised_decision(Q,H,B,gains,cfg):
    """Return controls from slot-start queues and current links only.

    Q/H/B and uploads are bits, frequencies Hz, powers W. The optimization
    internally uses Mbit so V/1e12 multiplies energy in the scaled objective.
    Arrival means come from cfg, never the realized current/future arrival trace.
    """
    Q,H,B,gains = (np.asarray(x,dtype=float) for x in (Q,H,B,gains))
    if cfg.N<=0 or cfg.M<=0:
        raise ValueError("Device and server counts must be positive")
    if Q.shape!=(cfg.N,) or H.shape!=(cfg.M,) or B.shape!=(cfg.N,cfg.M) or gains.shape!=B.shape:
        raise ValueError("Expected Q:(N,), H:(M,), B/gains:(N,M)")
    if any(not np.all(np.isfinite(x)) or np.any(x<0) for x in (Q,H,B,gains)):
        raise ValueError("Queues and gains must be finite and nonnegative")
    positive=(cfg.V,cfg.tau,cfg.kappa_device,cfg.kappa_edge,cfg.cycles_per_bit,
              cfg.edge_cycles_per_bit,cfg.noise_w,cfg.bandwidth_hz)
    nonnegative=(cfg.p_max_w,cfg.local_f_max_hz,cfg.edge_f_max_hz,cfg.beta,
                 cfg.arrival_lambda,cfg.task_packet_bits)
    if any(not np.isfinite(x) or x<=0 for x in positive) or any(not np.isfinite(x) or x<0 for x in nonnegative):
        raise ValueError("Invalid physical constants, V, capacity, or arrival mean")
    if not np.allclose(H,B.sum(axis=0),rtol=1e-10,atol=1e-4):
        raise ValueError("Aggregate and origin queues disagree")
    q=np.ascontiguousarray(Q/1e6); h=np.ascontiguousarray(H/1e6)
    channels=np.ascontiguousarray(gains)
    parameters=np.array([
        cfg.V/1e12,cfg.beta*cfg.arrival_lambda*cfg.task_packet_bits/1e6,
        cfg.kappa_device*cfg.cycles_per_bit**3*1e18/cfg.tau**2,
        cfg.kappa_edge*cfg.edge_cycles_per_bit**3*1e18/cfg.tau**2,
        cfg.tau,cfg.noise_w,np.log(2)*1e6/(cfg.tau*cfg.bandwidth_hz),cfg.p_max_w,
        cfg.tau*cfg.local_f_max_hz/cfg.cycles_per_bit/1e6,
        cfg.tau*cfg.edge_f_max_hz/cfg.edge_cycles_per_bit/1e6,cfg.M/cfg.N],dtype=float)
    local=np.zeros(cfg.N); upload=np.zeros(cfg.N); edge=np.zeros(cfg.M)
    association=np.full(cfg.N,-1,dtype=np.int32); stats=np.zeros(8)
    _,function,library=_load_engine()
    status=function(cfg.N,cfg.M,q,h,channels,parameters,MAX_SWEEPS,
                    IMPROVEMENT_TOLERANCE,local,upload,edge,association,stats)
    if status:
        raise RuntimeError(f"QAPG-R native controller failed with status {status}")
    power=np.zeros(cfg.N)
    active=np.flatnonzero(association>=0)
    power[active]=(cfg.noise_w/gains[active,association[active]])*np.expm1(parameters[6]*upload[active])
    # Inverting the Shannon cap can exceed pmax by a few ulps; this does not
    # change the upload decision, whose feasibility is checked by the ledger.
    power=np.minimum(power,cfg.p_max_w)
    aggregate=edge*1e6
    return EEDODecision(
        local_cpu_hz=local*1e6*cfg.cycles_per_bit/cfg.tau,
        transmit_power_w=power,association=association.astype(int),uploads_bits=upload*1e6,
        edge_cpu_hz=aggregate*cfg.edge_cycles_per_bit/cfg.tau,
        edge_service_bits=by_origin_service(B,aggregate),
        diagnostics={"effective_V":cfg.V,"iterations":int(stats[3]),
            "aggregate_edge_service_bits":aggregate,"revised_sweeps":int(stats[3]),
            "revised_converged":bool(stats[6]),"objective_initial":float(stats[0]),
            "objective_final":float(stats[1]),"objective_max_rise":float(stats[2]),
            "accepted_updates":int(stats[4]),"best_deviation_gain":float(stats[5]),
            "objective_checks":int(stats[7]),"edge_queue_weight":cfg.M/cfg.N,"native_library":library.name})

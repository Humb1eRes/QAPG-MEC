"""Replay two prespecified selection identities; never overwrite saved traces.

This is a deterministic rerun of the frozen implementation, not an independent
implementation and not evidence that the entire experiment has been replayed.
"""
from __future__ import annotations
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np

sys.dont_write_bytecode=True
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE))
import common_bridge as bridge
import run_protocol as runner

PRESPECIFIED_IDENTITIES=(
    {"stage":"selection","method":"QAPG-R","beta":2,"scale_index":4,"seed":320927},
    {"stage":"selection","method":"NoQuad-capped","beta":2,"scale_index":4,"seed":320927},
)
EXCLUDED_TRACE_FIELDS={"runtime_ms"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha(array):
    digest=hashlib.sha256()
    digest.update(str(array.shape).encode())
    digest.update(array.dtype.str.encode())
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def main():
    folder=runner.RESULTS/'selection'
    metadata_path=folder/'metadata.json'
    metadata=json.loads(metadata_path.read_text())
    if metadata['status']!='COMPLETED':
        raise RuntimeError('Complete selection before running the prespecified replays')
    freeze=json.loads((BASE/'FREEZE.json').read_text())
    hashes=runner.source_hashes()
    assert hashes==freeze['source_hashes']==metadata['source_hashes']
    assert freeze['protocol_sha256']==metadata['protocol_sha256']==sha(BASE/'PROTOCOL.json')
    report={
        'status':'RUNNING','verification_status':'PARTIAL_DETERMINISTIC_REPLAY',
        'scope':'Two prespecified selection trajectories rerun with frozen controllers and saved inputs. All numeric traces except wall-clock runtime are compared. This is not an independent controller reimplementation or a replay of every experiment.',
        'started_utc':datetime.now(timezone.utc).isoformat(),
        'replay_script_sha256':sha(__file__),'frozen_source_hashes':hashes,
        'selection_metadata_sha256':sha(metadata_path),
        'python':sys.version,'numpy':np.__version__,'platform':platform.platform(),
        'prespecified_identities':PRESPECIFIED_IDENTITIES,
        'excluded_fields':sorted(EXCLUDED_TRACE_FIELDS),
        'comparison':'Exact np.array_equal for all other numeric trace arrays, including diagnostic and edge-load arrays. No tolerance fallback.',
        'runs':[],
    }
    output=BASE/'REPLAY_VALIDATION.json'
    try:
        # Setup-only native compilation is outside replay timing.
        from qapg_revised import _load_engine
        _load_engine()
        for identity in PRESPECIFIED_IDENTITIES:
            name=f"b{identity['beta']}_s{identity['seed']}_{identity['method']}_v{identity['scale_index']}"
            record_path=folder/'records'/f'{name}.json'
            record=json.loads(record_path.read_text())
            assert record['status']=='PASS'
            assert all(record['identity'][key]==value for key,value in identity.items())
            trace_path=runner.ROOT/record['row']['trace_path']
            assert sha(trace_path)==record['row']['trace_sha256']
            arrays={}
            for key in ('arrivals','channels'):
                input_path=runner.ROOT/record['input'][key+'_path']
                with np.load(input_path,allow_pickle=False) as content: arrays[key]=content[key]
                assert array_sha(arrays[key])==record['input'][key+'_array_sha256']
                arrays[key].flags.writeable=False
            cfg=bridge.sim.rec.original.SimConfig(**record['config'])
            begin=time.perf_counter()
            actual,checks=bridge.simulate(identity['method'],cfg,arrays['arrivals'],arrays['channels'])
            duration=time.perf_counter()-begin
            with np.load(trace_path,allow_pickle=False) as content:
                expected={key:content[key] for key in content.files}
            assert set(actual)==set(expected)
            fields=[]
            for key in sorted(set(expected)-EXCLUDED_TRACE_FIELDS):
                first=np.asarray(expected[key]); second=np.asarray(actual[key])
                assert first.shape==second.shape and first.dtype==second.dtype,(key,first.dtype,second.dtype)
                assert np.issubdtype(first.dtype,np.number) or np.issubdtype(first.dtype,np.bool_),key
                assert np.all(np.isfinite(first)) and np.all(np.isfinite(second)),key
                exact=bool(np.array_equal(first,second))
                difference=np.abs(first.astype(np.float64)-second.astype(np.float64))
                fields.append({'field':key,'shape':list(first.shape),'dtype':str(first.dtype),
                               'exact':exact,'max_absolute_error':float(difference.max(initial=0)),
                               'saved_array_sha256':array_sha(first),'replay_array_sha256':array_sha(second)})
            run={'identity':identity,'record_sha256':sha(record_path),'trace_sha256':sha(trace_path),
                 'wall_seconds':duration,'compared_field_count':len(fields),
                 'exact_field_count':sum(item['exact'] for item in fields),
                 'maximum_absolute_error':max(item['max_absolute_error'] for item in fields),
                 'physical_checks_exact':checks==record['checks'],'fields':fields}
            report['runs'].append(run)
            assert all(item['exact'] for item in fields),f'Nonidentical replay arrays: {identity}'
            assert run['physical_checks_exact'],f'Physical summary checks differ: {identity}'
            for key in ('arrivals','channels'):
                assert array_sha(arrays[key])==record['input'][key+'_array_sha256']
            print(json.dumps({'identity':identity,'status':'EXACT','fields':len(fields),
                              'max_absolute_error':run['maximum_absolute_error']}),flush=True)
        assert runner.source_hashes()==hashes
        report['status']='PASS'
        report['replayed_run_count']=len(report['runs'])
    except BaseException as error:
        report['status']='FAIL';report['error']=repr(error)
        raise
    finally:
        report['finished_utc']=datetime.now(timezone.utc).isoformat()
        output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__': main()

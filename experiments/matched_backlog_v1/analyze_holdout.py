"""Descriptive seed-level comparison at all frozen aggregate-backlog budgets."""
from __future__ import annotations
import csv,hashlib,json,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
sys.dont_write_bytecode=True
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE))
import run_protocol as runner
ROOT=runner.ROOT;RESULTS=runner.RESULTS
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save_csv(p,rows):
    if not rows:
        p.write_text('beta,rho,backlog_target_mbit,comparator,paired_seeds\n')
        return
    with p.open('w',newline='') as h:
        w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def mean(x):return float(np.mean(x))
def sd(x):return float(np.std(x,ddof=1))
def main():
    p=json.loads((BASE/'PROTOCOL.json').read_text());selected=json.loads((RESULTS/'selection/SELECTED_SETTINGS.json').read_text())
    meta=json.loads((RESULTS/'test/metadata.json').read_text());valid=json.loads((RESULTS/'test/INDEPENDENT_VALIDATION.json').read_text())
    assert meta['status']=='COMPLETED' and valid['status']=='PASS'
    assert valid['metadata_sha256']==sha(RESULTS/'test/metadata.json')
    assert meta['selected_settings_sha256']==sha(RESULTS/'selection/SELECTED_SETTINGS.json')
    assert meta['source_hashes']==runner.source_hashes()
    groups={}
    for path in sorted((RESULTS/'test/records').glob('*.json')):
        assert valid['record_hashes'][str(path.relative_to(ROOT))]==sha(path)
        r=json.loads(path.read_text());i=r['identity'];row=r['row'];assert sha(ROOT/row['trace_path'])==row['trace_sha256']
        groups.setdefault((i['method'],i['beta'],i['scale_index']),[]).append(row)
    for rows in groups.values():
        rows.sort(key=lambda r:r['seed']);assert [r['seed'] for r in rows]==p['test']['seeds']
    decisions=selected['decisions'];assert len(decisions)==len(p['methods'])*len(p['betas'])*len(p['rho_targets'])
    output=[];pairrows=[];summaries=[]
    metrics=['energy','backlog','completion_ratio','late_slope','local_energy','transmit_energy','edge_energy','start_backlog','end_backlog']
    for d in decisions:
        row={k:d[k] for k in ['method','beta','rho','backlog_target_mbit']}
        row.update(scale_index=d['scale_index'],scale=d['scale'],status='NO_FEASIBLE_GRID_POINT',test_seeds=0,pass_seeds=0)
        for k in metrics:row[k+'_mean']='';row[k+'_sd']=''
        row.update(backlog_max='',backlog_min='',worst_seed_excess_mbit='',late_slope_max='',completion_ratio_min='',qapg_convergence_fraction_mean='',qapg_max_best_deviation_gain='')
        if d['scale_index'] is not None:
            rows=groups[(d['method'],d['beta'],d['scale_index'])];energy=[r['mean_energy_j_per_slot'] for r in rows];backlog=[r['mean_backlog_mbit'] for r in rows]
            passes=sum(q<=d['backlog_target_mbit'] for q in backlog)
            row.update(status='PASS' if passes==len(rows) else 'TEST_BACKLOG_FAIL',test_seeds=len(rows),pass_seeds=passes,backlog_max=max(backlog),backlog_min=min(backlog),worst_seed_excess_mbit=max(0.,max(backlog)-d['backlog_target_mbit']))
            value={'energy':energy,'backlog':backlog,'completion_ratio':[r['completion_to_arrival_ratio'] for r in rows],'late_slope':[r['late_half_backlog_slope_mbit_per_slot'] for r in rows],'start_backlog':[r['window_start_queue_bits']/1e6 for r in rows],'end_backlog':[r['window_end_queue_bits']/1e6 for r in rows]}
            for name,key in [('local_energy','local_energy_j'),('transmit_energy','transmit_energy_j'),('edge_energy','edge_energy')]:
                value[name]=[]
                for r in rows:
                    with np.load(ROOT/r['trace_path'],allow_pickle=False) as z:value[name].append(float(z[key][p['test']['warmup']:].mean()))
            for k,v in value.items():row[k+'_mean']=mean(v);row[k+'_sd']=sd(v)
            row['late_slope_max']=max(value['late_slope']);row['completion_ratio_min']=min(value['completion_ratio'])
            if d['method']=='QAPG-R':
                row['qapg_convergence_fraction_mean']=mean([r['revised_convergence_fraction'] for r in rows]);row['qapg_max_best_deviation_gain']=max(r['maximum_best_deviation_gain'] for r in rows)
        output.append(row)
    for beta in p['betas']:
        for rho in p['rho_targets']:
            rows=[r for r in output if r['beta']==beta and r['rho']==rho]
            proposed=next(r for r in rows if r['method']=='QAPG-R');competitors=[r for r in rows if r['method']!='QAPG-R' and r['status']=='PASS']
            best=min(competitors,key=lambda r:r['energy_mean']) if competitors else None
            eligible=proposed['status']=='PASS' and best is not None
            summary={'beta':beta,'rho':rho,'backlog_target_mbit':proposed['backlog_target_mbit'],'qapg_status':proposed['status'],'qapg_energy_mean':proposed['energy_mean'],'qapg_energy_sd':proposed['energy_sd'],'qapg_backlog_mean':proposed['backlog_mean'],'qapg_backlog_max':proposed['backlog_max'],'passing_comparator_count':len(competitors),'best_comparator':best['method'] if best else '', 'best_comparator_energy_mean':best['energy_mean'] if best else '', 'best_comparator_energy_sd':best['energy_sd'] if best else '', 'energy_saving_vs_best_percent':100*(best['energy_mean']-proposed['energy_mean'])/best['energy_mean'] if eligible else '', 'lowest_energy_among_passing':eligible and proposed['energy_mean']<best['energy_mean']}
            summaries.append(summary)
            if proposed['status']=='PASS':
                pp=groups[('QAPG-R',beta,proposed['scale_index'])]
                for c in competitors:
                    cc=groups[(c['method'],beta,c['scale_index'])]
                    delta=[r['mean_energy_j_per_slot']-q['mean_energy_j_per_slot'] for r,q in zip(cc,pp)]
                    pairrows.append({'beta':beta,'rho':rho,'backlog_target_mbit':proposed['backlog_target_mbit'],'comparator':c['method'],'qapg_scale':proposed['scale'],'comparator_scale':c['scale'],'paired_seeds':len(delta),'energy_difference_comparator_minus_qapg_mean_j':mean(delta),'energy_difference_sample_sd_j':sd(delta),'energy_difference_min_j':min(delta),'energy_difference_max_j':max(delta),'qapg_lower_energy_seed_count':sum(x>0 for x in delta),'ratio_of_means_energy_saving_percent':100*(c['energy_mean']-proposed['energy_mean'])/c['energy_mean']})
    out=RESULTS/'analysis';out.mkdir(exist_ok=True)
    save_csv(out/'HOLDOUT_COMPARISON.csv',output);save_csv(out/'HOLDOUT_PAIRED_DIFFERENCES.csv',pairrows);save_csv(out/'ALL_TARGET_SUMMARY.csv',summaries)
    counts={status:sum(r['status']==status for r in output) for status in ['PASS','NO_FEASIBLE_GRID_POINT','TEST_BACKLOG_FAIL']}
    summary={'status':'ANALYZED','created_utc':datetime.now(timezone.utc).isoformat(),'protocol_sha256':sha(BASE/'PROTOCOL.json'),'selected_settings_sha256':sha(RESULTS/'selection/SELECTED_SETTINGS.json'),'test_metadata_sha256':sha(RESULTS/'test/metadata.json'),'test_validation_sha256':sha(RESULTS/'test/INDEPENDENT_VALIDATION.json'),'analyzer_sha256':sha(__file__),'test_runs':meta['completed_runs'],'method_load_target_cells':len(output),'cell_status_counts':counts,'qapg_lowest_energy_cells':sum(r['lowest_energy_among_passing'] for r in summaries),'target_count':len(summaries),'all_target_summaries':summaries,'statistics':'Mean and sample SD across five seed-level metrics; relative reductions use ratios of means. Paired differences retain shared seed alignment. No significance, global-optimality or stochastic-stability assertion.'}
    (out/'ANALYSIS.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()

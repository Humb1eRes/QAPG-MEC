"""Independent raw-trace audit of held-out aggregation and paired comparisons.

Does not import the analyzer, runner, simulator, controller, or their metrics.
NumPy is used only to read arrays and check their basic shape/finite values;
seed metrics and sample summaries are recomputed with Python math.fsum.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
RESULTS = ROOT / 'experiments/results/matched_backlog_v1'
ANALYSIS = RESULTS / 'analysis'
MAX_ERROR = 0.0
CHECK_COUNT = 0


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def average(values):
    return math.fsum(float(value) for value in values) / len(values)


def sample_sd(values):
    center = average(values)
    return math.sqrt(math.fsum((float(value)-center)**2 for value in values)/(len(values)-1))


def compare(actual, expected, location):
    global MAX_ERROR, CHECK_COUNT
    CHECK_COUNT += 1
    if isinstance(expected, bool):
        require(actual is expected or str(actual) == str(expected), f'Boolean mismatch {location}')
    elif expected == '' or isinstance(expected, str):
        require(actual == expected, f'Text mismatch {location}: {actual!r} != {expected!r}')
    else:
        value = float(actual)
        require(math.isfinite(value), f'Nonfinite value at {location}')
        error = abs(value-expected)
        MAX_ERROR = max(MAX_ERROR, error)
        require(math.isclose(value, expected, rel_tol=1e-10, abs_tol=2e-9),
                f'Numeric mismatch {location}: {value} != {expected}')


def check_rows(path, expected_rows, key_fields):
    with path.open(newline='') as handle:
        observed = list(csv.DictReader(handle))
    def key(row):
        return tuple(str(row[field]) if field in ('method','comparator') else float(row[field])
                     for field in key_fields)
    expected = {key(row): row for row in expected_rows}
    actual = {key(row): row for row in observed}
    require(len(actual) == len(observed) and len(expected) == len(expected_rows), f'Duplicate keys {path.name}')
    require(set(actual) == set(expected), f'Missing/extra rows {path.name}')
    for identity, row in expected.items():
        for field, value in row.items():
            compare(actual[identity][field], value, f'{path.name}/{identity}/{field}')


def raw_seed_metrics(record, protocol):
    row = record['row']
    trace = ROOT / row['trace_path']
    require(sha(trace) == row['trace_sha256'], 'Trace hash mismatch')
    with np.load(trace, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    horizon, warmup = protocol['test']['T'], protocol['test']['warmup']
    for key, array in arrays.items():
        require(array.shape[0] == horizon and np.all(np.isfinite(array)), f'Bad raw array {key}')
    total = arrays['local_energy_j']+arrays['transmit_energy_j']+arrays['edge_energy']
    require(np.max(np.abs(total-arrays['total_energy'])) < 1e-10, 'Energy ledger mismatch')
    queue = arrays['device_queue']+arrays['edge_queue']
    require(np.max(np.abs(queue-arrays['total_queue'])) < 1e-5, 'Queue ledger mismatch')
    completed = arrays['local_completed_bits']+arrays['edge_completed_bits']
    late = [float(value)/1e6 for value in arrays['total_queue'][horizon//2:]]
    late_center, x_center = average(late), (len(late)-1)/2
    slope = (math.fsum((i-x_center)*(value-late_center) for i,value in enumerate(late))
             / (len(late)*(len(late)**2-1)/12))
    out = {'seed': record['identity']['seed'],
           'energy': average(total[warmup:]),
           'backlog': average(arrays['total_queue'][warmup:])/1e6,
           'completion_ratio': math.fsum(completed[warmup:])/math.fsum(arrays['arrival_bits'][warmup:]),
           'late_slope': slope,
           'local_energy': average(arrays['local_energy_j'][warmup:]),
           'transmit_energy': average(arrays['transmit_energy_j'][warmup:]),
           'edge_energy': average(arrays['edge_energy'][warmup:]),
           'start_backlog': float(arrays['total_queue'][warmup-1])/1e6 if warmup else 0.,
           'end_backlog': float(arrays['total_queue'][-1])/1e6}
    if record['identity']['method']=='QAPG-R':
        out['convergence_fraction']=average(arrays['revised_converged'])
        out['maximum_deviation']=float(max(arrays['best_deviation_gain']))
    for name, field in (('energy','mean_energy_j_per_slot'),('backlog','mean_backlog_mbit'),
                        ('completion_ratio','completion_to_arrival_ratio'),
                        ('late_slope','late_half_backlog_slope_mbit_per_slot')):
        compare(row[field],out[name], f'raw vs record/{record["identity"]}/{name}')
    # The finite-window completion ratio can exceed one while draining old work.
    flow = (out['start_backlog']+math.fsum(arrays['arrival_bits'][warmup:])/1e6
            -math.fsum(completed[warmup:])/1e6-out['end_backlog'])
    require(abs(flow) < 1e-7, f'Window flow imbalance {flow}')
    return out


def audit():
    protocol_path=BASE/'PROTOCOL.json'
    protocol=read_json(protocol_path)
    freeze=read_json(BASE/'FREEZE.json')
    selection_path=RESULTS/'selection/SELECTED_SETTINGS.json'
    selected=read_json(selection_path)
    selection_meta=read_json(RESULTS/'selection/metadata.json')
    metadata_path=RESULTS/'test/metadata.json'
    metadata=read_json(metadata_path)
    validation_path=RESULTS/'test/INDEPENDENT_VALIDATION.json'
    validation=read_json(validation_path)
    analysis_path=ANALYSIS/'ANALYSIS.json'
    analysis=read_json(analysis_path)
    require(metadata['status']=='COMPLETED' and validation['status']=='PASS' and analysis['status']=='ANALYZED',
            'Completed test, independent raw validation, and analysis are required')
    require(selected['status']=='FROZEN_BEFORE_TEST', 'Selection is not frozen')
    dates=[freeze['frozen_utc'],selection_meta['started_utc'],selection_meta['finished_utc'],
           selected['frozen_utc'],metadata['started_utc'],metadata['finished_utc'],analysis['created_utc']]
    instants=[datetime.fromisoformat(value) for value in dates]
    require(all(earlier<=later for earlier,later in zip(instants,instants[1:])), 'Execution chronology mismatch')
    require(metadata['selected_settings_sha256']==sha(selection_path), 'Frozen selection changed')
    require(validation['metadata_sha256']==sha(metadata_path), 'Validation metadata binding mismatch')
    require(validation['protocol_sha256']==sha(protocol_path), 'Validation protocol binding mismatch')
    require(freeze['source_hashes']==metadata['source_hashes'], 'Production source changed since freeze')
    for path,digest in freeze['source_hashes'].items():
        require(sha(ROOT/path)==digest, f'Current production source differs: {path}')
    for key,path in [('protocol_sha256',protocol_path),('selected_settings_sha256',selection_path),
                     ('test_metadata_sha256',metadata_path),('test_validation_sha256',validation_path),
                     ('analyzer_sha256',BASE/'analyze_holdout.py')]:
        require(analysis[key]==sha(path), f'Analysis provenance mismatch {key}')

    expected={(setting['method'],setting['beta'],setting['scale_index'],seed)
              for setting in selected['unique_settings'] for seed in protocol['test']['seeds']}
    require(len(expected)==selected['planned_test_runs']==metadata['planned_runs']==metadata['completed_runs']==225,
            'Expected complete 225-run frozen held-out grid')
    records=sorted((RESULTS/'test/records').glob('*.json'))
    require({str(path.relative_to(ROOT)):sha(path) for path in records}==validation['record_hashes'],
            'Validation did not bind exactly these raw records')
    seen=set();groups={}
    for path in records:
        record=read_json(path);identity=record['identity']
        key=tuple(identity[field] for field in ('method','beta','scale_index','seed'))
        require(key in expected and key not in seen, f'Unexpected/duplicate test identity {key}')
        seen.add(key)
        require(record['status']=='PASS' and identity['stage']=='test', 'Non-PASS/non-test record')
        require(identity['scale']==protocol['energy_scales'][identity['scale_index']], 'Wrong frozen scale')
        groups.setdefault(key[:3],{})[identity['seed']]=raw_seed_metrics(record,protocol)
    require(seen==expected, 'Some frozen test runs are missing')

    metrics=('energy','backlog','completion_ratio','late_slope','local_energy','transmit_energy','edge_energy','start_backlog','end_backlog')
    cells=[]
    for decision in selected['decisions']:
        cell={key:decision[key] for key in ('method','beta','rho','backlog_target_mbit')}
        cell.update(scale_index='' if decision['scale_index'] is None else decision['scale_index'],
                    scale='' if decision['scale'] is None else decision['scale'],
                    status='NO_FEASIBLE_GRID_POINT',test_seeds=0,pass_seeds=0)
        for metric in metrics:
            cell[metric+'_mean']=cell[metric+'_sd']=''
        for key in ('backlog_max','backlog_min','worst_seed_excess_mbit','late_slope_max',
                    'completion_ratio_min','qapg_convergence_fraction_mean','qapg_max_best_deviation_gain'):
            cell[key]=''
        if decision['scale_index'] is not None:
            data=groups[(decision['method'],decision['beta'],decision['scale_index'])]
            require(set(data)==set(protocol['test']['seeds']), 'Seed alignment mismatch')
            seed_rows=[data[seed] for seed in protocol['test']['seeds']]
            passes=sum(row['backlog']<=decision['backlog_target_mbit'] for row in seed_rows)
            cell.update(status='PASS' if passes==len(seed_rows) else 'TEST_BACKLOG_FAIL',
                        test_seeds=len(seed_rows),pass_seeds=passes)
            for metric in metrics:
                values=[row[metric] for row in seed_rows]
                cell[metric+'_mean']=average(values);cell[metric+'_sd']=sample_sd(values)
            cell.update(backlog_max=max(row['backlog'] for row in seed_rows),backlog_min=min(row['backlog'] for row in seed_rows),
                        worst_seed_excess_mbit=max(0.,max(row['backlog'] for row in seed_rows)-decision['backlog_target_mbit']),
                        late_slope_max=max(row['late_slope'] for row in seed_rows),
                        completion_ratio_min=min(row['completion_ratio'] for row in seed_rows))
            if decision['method']=='QAPG-R':
                cell['qapg_convergence_fraction_mean']=average([row['convergence_fraction'] for row in seed_rows])
                cell['qapg_max_best_deviation_gain']=max(row['maximum_deviation'] for row in seed_rows)
        cells.append(cell)
    require(len(cells)==72, 'Missing target cells')
    check_rows(ANALYSIS/'HOLDOUT_COMPARISON.csv',cells,('method','beta','rho'))

    summaries=[];pairs=[]
    for beta in protocol['betas']:
        for rho in protocol['rho_targets']:
            candidates=[cell for cell in cells if cell['beta']==beta and cell['rho']==rho]
            proposed=next(cell for cell in candidates if cell['method']=='QAPG-R')
            passing=[cell for cell in candidates if cell['method']!='QAPG-R' and cell['status']=='PASS']
            best=min(passing,key=lambda cell:cell['energy_mean']) if passing else None
            eligible=proposed['status']=='PASS' and best is not None
            summary={'beta':beta,'rho':rho,'backlog_target_mbit':proposed['backlog_target_mbit'],
                     'qapg_status':proposed['status'],'qapg_energy_mean':proposed['energy_mean'],
                     'qapg_energy_sd':proposed['energy_sd'],'qapg_backlog_mean':proposed['backlog_mean'],
                     'qapg_backlog_max':proposed['backlog_max'],'passing_comparator_count':len(passing),
                     'best_comparator':best['method'] if best else '',
                     'best_comparator_energy_mean':best['energy_mean'] if best else '',
                     'best_comparator_energy_sd':best['energy_sd'] if best else '',
                     'energy_saving_vs_best_percent':100*(best['energy_mean']-proposed['energy_mean'])/best['energy_mean'] if eligible else '',
                     'lowest_energy_among_passing':eligible and proposed['energy_mean']<best['energy_mean']}
            summaries.append(summary)
            if proposed['status']=='PASS':
                pgroup=groups[('QAPG-R',beta,proposed['scale_index'])]
                for competitor in passing:
                    cgroup=groups[(competitor['method'],beta,competitor['scale_index'])]
                    # Align using seed keys, not CSV or dictionary row position.
                    difference=[cgroup[seed]['energy']-pgroup[seed]['energy'] for seed in protocol['test']['seeds']]
                    pairs.append({'beta':beta,'rho':rho,'backlog_target_mbit':proposed['backlog_target_mbit'],
                                  'comparator':competitor['method'],'qapg_scale':proposed['scale'],
                                  'comparator_scale':competitor['scale'],'paired_seeds':len(difference),
                                  'energy_difference_comparator_minus_qapg_mean_j':average(difference),
                                  'energy_difference_sample_sd_j':sample_sd(difference),
                                  'energy_difference_min_j':min(difference),'energy_difference_max_j':max(difference),
                                  'qapg_lower_energy_seed_count':sum(value>0 for value in difference),
                                  'ratio_of_means_energy_saving_percent':100*(competitor['energy_mean']-proposed['energy_mean'])/competitor['energy_mean']})
    check_rows(ANALYSIS/'HOLDOUT_PAIRED_DIFFERENCES.csv',pairs,('beta','rho','comparator'))
    check_rows(ANALYSIS/'ALL_TARGET_SUMMARY.csv',summaries,('beta','rho'))
    require(len(summaries)==12, 'Missing target summaries')
    counts={status:sum(cell['status']==status for cell in cells) for status in ('PASS','NO_FEASIBLE_GRID_POINT','TEST_BACKLOG_FAIL')}
    require(analysis['cell_status_counts']==counts, 'Analysis status counts mismatch')
    require(analysis['qapg_lowest_energy_cells']==sum(row['lowest_energy_among_passing'] for row in summaries),
            'Descriptive lowest-energy count mismatch')
    for expected_summary,observed_summary in zip(summaries,analysis['all_target_summaries']):
        for key,value in expected_summary.items():
            compare(observed_summary[key],value,f'ANALYSIS.json/{expected_summary["beta"]}/{expected_summary["rho"]}/{key}')
    require(len(analysis['all_target_summaries'])==12, 'JSON omitted target summaries')
    report={'status':'PASS','checked_utc':datetime.now(timezone.utc).isoformat(),
            'audit_source_sha256':sha(__file__),'analysis_sha256':sha(analysis_path),
            'protocol_sha256':sha(protocol_path),'selected_settings_sha256':sha(selection_path),
            'test_metadata_sha256':sha(metadata_path),'test_validation_sha256':sha(validation_path),
            'test_runs':len(seen),'unique_test_settings':len(groups),'checked_cells':len(cells),
            'checked_paired_rows':len(pairs),'checked_target_summaries':len(summaries),
            'cell_status_counts':counts,'numeric_and_categorical_checks':CHECK_COUNT,
            'maximum_absolute_numeric_difference':MAX_ERROR,
            'descriptive_qapg_lowest_energy_cells':sum(row['lowest_energy_among_passing'] for row in summaries),
            'test_backlog_failures':[cell for cell in cells if cell['status']=='TEST_BACKLOG_FAIL'],
            'scope':'Independent raw-trace aggregation, seed-key pairing and chronology/source audit. No global-optimality, significance or infinite-horizon stability conclusion.',
            'production_source_hashes':freeze['source_hashes']}
    proposed_cells=[cell for cell in cells if cell['method']=='QAPG-R' and cell['test_seeds']]
    report['qapg_minimum_observed_backlog_margin_mbit']=min(cell['backlog_target_mbit']-cell['backlog_max'] for cell in proposed_cells)
    report['qapg_minimum_mean_convergence_fraction']=min(cell['qapg_convergence_fraction_mean'] for cell in proposed_cells)
    report['qapg_maximum_mean_convergence_fraction']=max(cell['qapg_convergence_fraction_mean'] for cell in proposed_cells)
    report['qapg_maximum_remaining_unilateral_gain']=max(cell['qapg_max_best_deviation_gain'] for cell in proposed_cells)
    (ANALYSIS/'HOLDOUT_AUDIT.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    text=['# Independent held-out analysis audit','',
          '## Material Passport','',
          '- Status: **PASS**. Source: saved raw test traces, frozen protocol and selection, immutable run records, and final analysis tables.',
          '- No analyzer, simulator, runner, controller, or shared metric helper was imported. Seed-level summaries were recomputed with Python compensated sums; standard deviations use the sample denominator.',
          '',f'All {len(seen)} held-out runs, {len(cells)} method/load/target cells, {len(pairs)} eligible paired comparisons and all 12 target summaries were verified.',
          '',f'Cell counts: {counts["PASS"]} satisfy the requirement in every held-out seed; {counts["TEST_BACKLOG_FAIL"]} fail it in at least one held-out seed; {counts["NO_FEASIBLE_GRID_POINT"]} had no feasible selection-grid point.',
          '',f'Largest numeric discrepancy from the independent computation: {MAX_ERROR:.3g}. Paired energy differences align explicitly by seed; relative reductions use ratios of means. The same operating point reused at multiple targets is not treated as an additional independent experiment.',
          '', 'The audit confirms algorithm/source freezing before selection, selected settings fixed before testing, unchanged production hashes, and complete preservation of infeasible and failed cells.',
          '',f'The smallest observed proposed-method backlog margin is {report["qapg_minimum_observed_backlog_margin_mbit"]:.12g} Mbit. This is empirical passage of the declared bound in these five seeds, not a confidence or robustness guarantee. Proposed-method mean slot-convergence fractions range from {report["qapg_minimum_mean_convergence_fraction"]:.6g} to {report["qapg_maximum_mean_convergence_fraction"]:.6g}; capped nonconverged slots remain in the data.',
          '', '“Lowest energy” is a descriptive ranking among the declared, empirically feasible frozen settings. It is not a proof of the globally best controller, a significance result, or indefinite queue stability.']
    (ANALYSIS/'HOLDOUT_AUDIT.md').write_text('\n'.join(text)+'\n')
    print(json.dumps({key:report[key] for key in ('status','test_runs','checked_cells','checked_paired_rows',
                                                 'checked_target_summaries','cell_status_counts','maximum_absolute_numeric_difference')},indent=2))


if __name__=='__main__':
    audit()

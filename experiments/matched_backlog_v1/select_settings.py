"""Freeze every target's operating point before held-out execution.

Reads only the completed, independently validated selection split. The complete
grid and all infeasible targets remain visible. No simulator/controller import
or test outcome is used. Existing outputs or any test execution prevent a run.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
RESULTS = ROOT / 'experiments/results/matched_backlog_v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite_nonnegative(value, label):
    require(isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value >= 0, f'Invalid {label}: {value!r}')
    return float(value)


def assert_test_not_started():
    test = RESULTS / 'test'
    require(not (test / 'metadata.json').exists(), 'Test metadata already exists; selection cannot be frozen retrospectively')
    for subfolder in ('records', 'traces'):
        folder = test / subfolder
        require(not folder.exists() or not any(folder.iterdir()),
                f'Test {subfolder} already exist; selection cannot be frozen retrospectively')


def summarize_and_choose(records, protocol):
    """Pure selection logic: one complete seed grid in, all decisions out."""
    methods, betas = protocol['methods'], protocol['betas']
    scales, seeds = protocol['energy_scales'], protocol['selection']['seeds']
    require(len(seeds) >= 2 and len(seeds) == len(set(seeds)), 'Need distinct selection seeds for sample SD')
    expected = {(method, beta, index, seed) for method in methods for beta in betas
                for index in range(len(scales)) for seed in seeds}
    indexed = {}
    arrival_bases = set()
    for record in records:
        identity, row, cfg = record['identity'], record['row'], record['config']
        key = (identity['method'], identity['beta'], identity['scale_index'], identity['seed'])
        require(key in expected and key not in indexed, f'Unexpected or duplicate identity {key}')
        require(record['status'] == 'PASS' and identity['stage'] == 'selection', f'Nonselection/non-PASS record {key}')
        require(row['stage'] == 'selection', f'Wrong row stage {key}')
        for field in ('method', 'beta', 'scale_index', 'seed'):
            require(row[field] == identity[field], f'Identity mismatch for {field}: {key}')
        scale = scales[identity['scale_index']]
        require(identity['scale'] == scale and row['energy_scale'] == scale, f'Wrong energy scale {key}')
        require(cfg['V'] == row['V'] == protocol['base_V'] * scale, f'Wrong V {key}')
        require(cfg['T'] == row['T'] == protocol['selection']['T'], f'Wrong horizon {key}')
        require(row['warmup'] == protocol['selection']['warmup'], f'Wrong warmup {key}')
        require(cfg['seed'] == identity['seed'] and cfg['beta'] == identity['beta'], f'Wrong config identity {key}')
        e = finite_nonnegative(row['mean_energy_j_per_slot'], 'mean energy')
        q = finite_nonnegative(row['mean_backlog_mbit'], 'mean backlog')
        require(record['checks']['conservation_and_resource_assertions_passed'] is True,
                f'Conservation/resource validation failed {key}')
        arrival_base = (cfg['N'], cfg['arrival_lambda'], cfg['task_packet_bits'])
        for value in arrival_base:
            finite_nonnegative(value, 'arrival target constant')
        arrival_bases.add(arrival_base)
        indexed[key] = {'seed': identity['seed'], 'mean_energy_j_per_slot': e,
                        'mean_backlog_mbit': q, 'trace_sha256': row['trace_sha256']}
    require(set(indexed) == expected, f'Incomplete selection grid: missing {len(expected-set(indexed))} records')
    require(len(arrival_bases) == 1, 'Target-defining arrival configuration differs across runs')
    n, arrival_lambda, packet_bits = next(iter(arrival_bases))
    require(n * arrival_lambda * packet_bits > 0, 'Positive nominal arrivals required for workload targets')

    aggregates = []
    lookup = {}
    for method in methods:
        for beta in betas:
            for index, scale in enumerate(scales):
                samples = [indexed[(method, beta, index, seed)] for seed in seeds]
                energy = [sample['mean_energy_j_per_slot'] for sample in samples]
                backlog = [sample['mean_backlog_mbit'] for sample in samples]
                summary = {'method': method, 'beta': beta, 'scale_index': index, 'scale': scale,
                           'V': protocol['base_V'] * scale, 'selection_seed_count': len(seeds),
                           'mean_energy_j_per_slot': statistics.mean(energy),
                           'sd_energy_j_per_slot': statistics.stdev(energy),
                           'mean_backlog_mbit': statistics.mean(backlog),
                           'sd_backlog_mbit': statistics.stdev(backlog),
                           'maximum_seed_mean_backlog_mbit': max(backlog),
                           'minimum_seed_mean_backlog_mbit': min(backlog)}
                for sample in samples:
                    seed = sample['seed']
                    summary[f'seed_{seed}_mean_energy_j_per_slot'] = sample['mean_energy_j_per_slot']
                    summary[f'seed_{seed}_mean_backlog_mbit'] = sample['mean_backlog_mbit']
                aggregates.append(summary)
                lookup[(method, beta, index)] = dict(summary, seed_results=samples)

    decisions, eligibility, unique = [], [], {}
    for method in methods:
        for beta in betas:
            expected_arrivals = n * beta * arrival_lambda * packet_bits / 1e6
            for rho in protocol['rho_targets']:
                # Deliberately uses configured expected arrivals, never an
                # observed seed arrival rate. No eligibility epsilon is added.
                target = rho * n * beta * arrival_lambda * packet_bits / 1e6
                candidates = []
                for index in range(len(scales)):
                    summary = lookup[(method, beta, index)]
                    passes = [sample['mean_backlog_mbit'] <= target for sample in summary['seed_results']]
                    feasible = all(passes)
                    eligibility.append({'method': method, 'beta': beta, 'rho': rho,
                                        'backlog_target_mbit': target, 'scale_index': index,
                                        'scale': scales[index], 'selection_feasible': feasible,
                                        'passing_selection_seeds': sum(passes),
                                        'selection_seed_count': len(seeds),
                                        'maximum_seed_mean_backlog_mbit': summary['maximum_seed_mean_backlog_mbit'],
                                        'worst_seed_excess_mbit': max(0., summary['maximum_seed_mean_backlog_mbit'] - target)})
                    if feasible:
                        candidates.append(summary)
                decision = {'method': method, 'beta': beta, 'rho': rho,
                            'expected_arrivals_mbit_per_slot': expected_arrivals,
                            'backlog_target_mbit': target, 'grid_candidate_count': len(scales),
                            'eligible_scale_indices': [candidate['scale_index'] for candidate in candidates]}
                if candidates:
                    chosen = min(candidates, key=lambda item: (item['mean_energy_j_per_slot'],
                                                              item['mean_backlog_mbit'], item['scale_index']))
                    setting = {field: chosen[field] for field in ('method', 'beta', 'scale_index', 'scale')}
                    unique[(method, beta, chosen['scale_index'])] = setting
                    decision.update(status='SELECTED', scale_index=chosen['scale_index'], scale=chosen['scale'],
                                    selected_setting=setting, selection_summary=chosen,
                                    selection_mean_energy=chosen['mean_energy_j_per_slot'],
                                    selection_mean_backlog=chosen['mean_backlog_mbit'],
                                    max_selection_backlog=chosen['maximum_seed_mean_backlog_mbit'])
                else:
                    decision.update(status='NO_FEASIBLE_GRID_POINT', scale_index=None, scale=None,
                                    selected_setting=None, selection_summary=None,
                                    selection_mean_energy=None, selection_mean_backlog=None,
                                    max_selection_backlog=None)
                decisions.append(decision)
    expected_decisions = len(methods)*len(betas)*len(protocol['rho_targets'])
    require(len(decisions) == expected_decisions, 'Missing target decisions')
    unique_settings = [unique[key] for method in methods for beta in betas
                       for index in range(len(scales)) if (key := (method, beta, index)) in unique]
    return aggregates, eligibility, decisions, unique_settings


def write_csv_exclusive(path, rows):
    with path.open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    assert_test_not_started()
    out = RESULTS / 'selection'
    destination = out / 'SELECTED_SETTINGS.json'
    grid_path, eligibility_path = out / 'selection_grid.csv', out / 'selection_eligibility.csv'
    for path in (destination, grid_path, eligibility_path):
        require(not path.exists(), f'Refusing to overwrite existing selection output: {path}')
    protocol_path = BASE / 'PROTOCOL.json'
    metadata_path = out / 'metadata.json'
    validation_path = out / 'INDEPENDENT_VALIDATION.json'
    protocol = json.loads(protocol_path.read_text())
    metadata = json.loads(metadata_path.read_text())
    validation = json.loads(validation_path.read_text())
    freeze = json.loads((BASE / 'FREEZE.json').read_text())
    protocol_hash, metadata_hash = sha(protocol_path), sha(metadata_path)
    require(metadata['status'] == 'COMPLETED' and metadata['stage'] == 'selection', 'Selection is not complete')
    require(validation['status'] == 'PASS', 'Root independent validation has not passed')
    require(validation['stage'] == 'selection', 'Independent validation is for a different stage')
    require(validation['metadata_sha256'] == metadata_hash, 'Independent validation metadata binding is stale')
    require(validation['protocol_sha256'] == protocol_hash, 'Independent validation protocol binding is stale')
    require(metadata['protocol_sha256'] == freeze['protocol_sha256'] == protocol_hash, 'Protocol changed')
    require(metadata['source_hashes'] == freeze['source_hashes'], 'Selection/freeze source hashes differ')
    for relative, expected in freeze['source_hashes'].items():
        require(sha(ROOT / relative) == expected, f'Frozen source changed: {relative}')
    paths = sorted((out / 'records').glob('*.json'))
    manifest = [{'path': str(path.relative_to(ROOT)), 'sha256': sha(path)} for path in paths]
    records_hash = canonical_sha(manifest)
    records = [json.loads(path.read_text()) for path in paths]
    require(validation['record_hashes'] == {item['path']: item['sha256'] for item in manifest},
            'Independent validation record bindings are stale/incomplete')
    expected_count = len(protocol['methods'])*len(protocol['betas'])*len(protocol['energy_scales'])*len(protocol['selection']['seeds'])
    require(metadata['planned_runs'] == metadata['completed_runs'] == len(records) == expected_count,
            'Metadata/record counts do not match the complete declared grid')
    require(validation['runs'] == expected_count, 'Independent validation did not cover the declared grid')
    # The validator may use these standard provenance bindings. Its complete
    # file is also hashed into the frozen result regardless of schema extras.
    if 'selection_metadata_sha256' in validation:
        require(validation['selection_metadata_sha256'] == metadata_hash, 'Validation metadata binding is stale')
    if 'selection_records_combined_sha256' in validation:
        require(validation['selection_records_combined_sha256'] == records_hash, 'Validation record binding is stale')
    grid, eligibility, decisions, unique = summarize_and_choose(records, protocol)
    source_hash = sha(Path(__file__))
    output = {'status': 'FROZEN_BEFORE_TEST', 'frozen_utc': datetime.now(timezone.utc).isoformat(),
              'protocol_sha256': protocol_hash, 'selection_metadata_sha256': metadata_hash,
              'selection_records_combined_sha256': records_hash,
              'selection_records_hash_encoding': "sha256(json.dumps(sorted manifest by path,sort_keys=True,separators=(',',':')).encode())",
              'selection_record_manifest': manifest,
              'independent_validation_sha256': sha(validation_path),
              'analyzer_source_sha256': source_hash, 'selector_source_sha256': source_hash,
              'selection_rule': protocol['selection_rule'], 'test_rule': protocol['test_rule'],
              'selection_seeds': protocol['selection']['seeds'], 'test_seeds': protocol['test']['seeds'],
              'selection_record_count': len(records), 'grid_setting_count': len(grid),
              'decision_count': len(decisions),
              'selected_decision_count': sum(item['status'] == 'SELECTED' for item in decisions),
              'no_feasible_grid_decision_count': sum(item['status'] != 'SELECTED' for item in decisions),
              'unique_setting_count': len(unique), 'planned_test_runs': len(unique)*len(protocol['test']['seeds']),
              'decisions': decisions, 'unique_settings': unique}
    # Recheck immediately before writing. SELECTED_SETTINGS is the last output,
    # so the test runner cannot consume a commitment before both audit tables exist.
    assert_test_not_started()
    require(sha(metadata_path) == metadata_hash and sha(protocol_path) == protocol_hash, 'Inputs changed during selection')
    require(canonical_sha([{'path': str(path.relative_to(ROOT)), 'sha256': sha(path)} for path in paths]) == records_hash,
            'Selection records changed during analysis')
    write_csv_exclusive(grid_path, grid)
    write_csv_exclusive(eligibility_path, eligibility)
    output['selection_grid_sha256'] = sha(grid_path)
    output['selection_eligibility_sha256'] = sha(eligibility_path)
    with destination.open('x') as handle:
        json.dump(output, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps({key: output[key] for key in ('status', 'decision_count', 'selected_decision_count',
                                                   'no_feasible_grid_decision_count', 'unique_setting_count', 'planned_test_runs')}))


if __name__ == '__main__':
    main()

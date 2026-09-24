"""Independent raw-trace audit of frozen choices; does not import the selector."""
from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
import statistics
import numpy as np

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
SELECTION = ROOT / 'experiments/results/matched_backlog_v1/selection'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def same(a, b):
    assert math.isclose(float(a), float(b), rel_tol=1e-12, abs_tol=1e-12), (a, b)


def main():
    protocol = json.loads((BASE / 'PROTOCOL.json').read_text())
    freeze = json.loads((BASE / 'FREEZE.json').read_text())
    metadata = json.loads((SELECTION / 'metadata.json').read_text())
    validation = json.loads((SELECTION / 'INDEPENDENT_VALIDATION.json').read_text())
    selected_path = SELECTION / 'SELECTED_SETTINGS.json'
    selected_hash = sha(selected_path)
    selected = json.loads(selected_path.read_text())
    assert metadata['status'] == 'COMPLETED'
    assert selected['status'] == 'FROZEN_BEFORE_TEST'
    assert validation['status'] == 'PASS'
    assert freeze['frozen_utc'] < metadata['started_utc'] < metadata['finished_utc'] < selected['frozen_utc']
    assert metadata['protocol_sha256'] == selected['protocol_sha256'] == sha(BASE / 'PROTOCOL.json')
    assert selected['selection_metadata_sha256'] == validation['metadata_sha256'] == sha(SELECTION / 'metadata.json')
    assert selected['independent_validation_sha256'] == sha(SELECTION / 'INDEPENDENT_VALIDATION.json')
    assert selected['selector_source_sha256'] == sha(BASE / 'select_settings.py')
    assert metadata['source_hashes'] == freeze['source_hashes']
    for path, digest in freeze['source_hashes'].items():
        assert sha(ROOT / path) == digest

    seeds = protocol['selection']['seeds']
    expected_runs = {(m, b, i, s) for m in protocol['methods'] for b in protocol['betas']
                     for i in range(len(protocol['energy_scales'])) for s in seeds}
    seen = set()
    groups = {}
    manifest = []
    arrival_bases = set()
    for path in sorted((SELECTION / 'records').glob('*.json')):
        record = json.loads(path.read_text())
        identity, row, cfg = record['identity'], record['row'], record['config']
        key = (identity['method'], identity['beta'], identity['scale_index'], identity['seed'])
        assert key in expected_runs and key not in seen
        seen.add(key)
        assert record['status'] == 'PASS' and identity['stage'] == 'selection'
        trace = ROOT / row['trace_path']
        assert sha(trace) == row['trace_sha256']
        with np.load(trace, allow_pickle=False) as data:
            assert len(data['total_energy']) == len(data['total_queue']) == protocol['selection']['T']
            warm = protocol['selection']['warmup']
            e = float(np.mean(data['total_energy'][warm:]))
            q = float(np.mean(data['total_queue'][warm:])) / 1e6
        assert math.isfinite(e) and math.isfinite(q) and e >= 0 and q >= 0
        same(e, row['mean_energy_j_per_slot'])
        same(q, row['mean_backlog_mbit'])
        groups.setdefault(key[:3], {})[key[3]] = {'e': e, 'q': q, 'trace_sha256': row['trace_sha256']}
        manifest.append({'path': str(path.relative_to(ROOT)), 'sha256': sha(path)})
        arrival_bases.add((cfg['N'], cfg['arrival_lambda'], cfg['task_packet_bits']))
    assert seen == expected_runs and len(seen) == metadata['planned_runs'] == metadata['completed_runs'] == 324
    assert len(groups) == 162 and all(set(g) == set(seeds) for g in groups.values())
    assert len(arrival_bases) == 1
    n, lam, packet = next(iter(arrival_bases))
    assert manifest == selected['selection_record_manifest']
    assert {r['path']: r['sha256'] for r in manifest} == validation['record_hashes']
    manifest_hash = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    assert manifest_hash == selected['selection_records_combined_sha256']

    summaries = {}
    for key, samples in groups.items():
        es = [samples[s]['e'] for s in seeds]
        qs = [samples[s]['q'] for s in seeds]
        summaries[key] = {'e': math.fsum(es) / len(es), 'q': math.fsum(qs) / len(qs),
                          'sd_e': statistics.stdev(es), 'sd_q': statistics.stdev(qs),
                          'max_q': max(qs), 'min_q': min(qs)}
    cells = {(m, b, rho) for m in protocol['methods'] for b in protocol['betas'] for rho in protocol['rho_targets']}
    seen_cells, selected_settings, checked = set(), set(), []
    for decision in selected['decisions']:
        key = (decision['method'], decision['beta'], decision['rho'])
        assert key in cells and key not in seen_cells
        seen_cells.add(key)
        m, b, rho = key
        target = rho * n * b * lam * packet / 1e6
        assert target == decision['backlog_target_mbit']
        eligible = [i for i in range(len(protocol['energy_scales']))
                    if all(groups[(m, b, i)][s]['q'] <= target for s in seeds)]
        assert eligible == decision['eligible_scale_indices']
        if eligible:
            index = min(eligible, key=lambda i: (summaries[(m, b, i)]['e'], summaries[(m, b, i)]['q'], i))
            assert decision['status'] == 'SELECTED' and decision['scale_index'] == index
            assert decision['scale'] == protocol['energy_scales'][index]
            same(decision['selection_mean_energy'], summaries[(m, b, index)]['e'])
            same(decision['selection_mean_backlog'], summaries[(m, b, index)]['q'])
            same(decision['max_selection_backlog'], summaries[(m, b, index)]['max_q'])
            setting = {'method': m, 'beta': b, 'scale_index': index, 'scale': protocol['energy_scales'][index]}
            assert decision['selected_setting'] == setting
            selected_settings.add((m, b, index))
        else:
            index = None
            assert decision['status'] == 'NO_FEASIBLE_GRID_POINT'
            assert all(decision[field] is None for field in ['scale_index', 'scale', 'selected_setting', 'selection_summary', 'selection_mean_energy', 'selection_mean_backlog', 'max_selection_backlog'])
        checked.append({'method': m, 'beta': b, 'rho': rho, 'status': 'MATCH', 'scale_index': index,
                        'eligible_count': len(eligible)})
    assert seen_cells == cells and len(checked) == selected['decision_count'] == 72
    expected_unique = [{'method': m, 'beta': b, 'scale_index': i, 'scale': scale}
                       for m in protocol['methods'] for b in protocol['betas']
                       for i, scale in enumerate(protocol['energy_scales']) if (m, b, i) in selected_settings]
    assert selected['unique_settings'] == expected_unique
    assert selected['unique_setting_count'] == len(expected_unique)
    assert selected['planned_test_runs'] == len(expected_unique) * len(protocol['test']['seeds'])
    assert selected['selection_seeds'] == seeds and selected['test_seeds'] == protocol['test']['seeds']

    with (SELECTION / 'selection_grid.csv').open(newline='') as handle:
        grid = list(csv.DictReader(handle))
    assert sha(SELECTION / 'selection_grid.csv') == selected['selection_grid_sha256']
    seen_grid = set()
    for row in grid:
        key = (row['method'], int(row['beta']), int(row['scale_index']))
        assert key in summaries and key not in seen_grid
        seen_grid.add(key)
        for field, metric in [('mean_energy_j_per_slot', 'e'), ('mean_backlog_mbit', 'q'),
                              ('sd_energy_j_per_slot', 'sd_e'), ('sd_backlog_mbit', 'sd_q'),
                              ('maximum_seed_mean_backlog_mbit', 'max_q'), ('minimum_seed_mean_backlog_mbit', 'min_q')]:
            same(row[field], summaries[key][metric])
        for seed in seeds:
            same(row[f'seed_{seed}_mean_energy_j_per_slot'], groups[key][seed]['e'])
            same(row[f'seed_{seed}_mean_backlog_mbit'], groups[key][seed]['q'])
    assert seen_grid == set(summaries) and len(grid) == 162
    with (SELECTION / 'selection_eligibility.csv').open(newline='') as handle:
        eligibility = list(csv.DictReader(handle))
    assert sha(SELECTION / 'selection_eligibility.csv') == selected['selection_eligibility_sha256']
    eligibility_keys = set()
    for row in eligibility:
        m, b, rho, i = row['method'], int(row['beta']), float(row['rho']), int(row['scale_index'])
        key = (m, b, rho, i)
        assert key not in eligibility_keys
        eligibility_keys.add(key)
        target = rho * n * b * lam * packet / 1e6
        passed = sum(groups[(m, b, i)][s]['q'] <= target for s in seeds)
        assert (row['selection_feasible'] == 'True') == (passed == len(seeds))
        assert int(row['passing_selection_seeds']) == passed
        same(row['worst_seed_excess_mbit'], max(0., summaries[(m, b, i)]['max_q'] - target))
    assert eligibility_keys == {(m, b, rho, i) for m, b, rho in cells for i in range(len(protocol['energy_scales']))}
    assert len(eligibility) == 648
    test_metadata = ROOT / 'experiments/results/matched_backlog_v1/test/metadata.json'
    if test_metadata.exists():
        test = json.loads(test_metadata.read_text())
        assert selected['frozen_utc'] < test['started_utc'] and test['selected_settings_sha256'] == selected_hash
    assert sha(selected_path) == selected_hash

    report = {'status': 'PASS', 'checked_utc': datetime.now(timezone.utc).isoformat(),
              'audit_source_sha256': sha(Path(__file__)), 'protocol_sha256': sha(BASE / 'PROTOCOL.json'),
              'selected_settings_sha256': selected_hash, 'selector_source_sha256': sha(BASE / 'select_settings.py'),
              'validator_source_sha256': sha(BASE / 'validate_results.py'), 'selection_raw_traces_recomputed': len(seen),
              'grid_rows_checked': len(grid), 'eligibility_rows_checked': len(eligibility), 'decisions_checked': len(checked),
              'selected_decisions': sum(c['scale_index'] is not None for c in checked),
              'no_feasible_decisions': sum(c['scale_index'] is None for c in checked),
              'unique_settings': len(expected_unique), 'planned_test_runs': selected['planned_test_runs'],
              'all_choices_match': True, 'decision_checks': checked,
              'scope': 'Direct independent raw-trace means and complete-grid all-seed constraint selection. The production selector was not imported or invoked. No empirical performance superiority is inferred.'}
    (BASE / 'SELECTOR_AUDIT.json').write_text(json.dumps(report, indent=2) + '\n')
    (BASE / 'SELECTOR_AUDIT.md').write_text(
        '# Independent selection audit\n\n'
        'Status: **PASS**. The reviewer independently recomputed energy and backlog directly from all 324 saved selection traces, without importing or calling the production selector.\n\n'
        'All 162 grid summaries, 648 target/candidate feasibility rows, and 72 method/load/target decisions agree. Eligibility uses every selection seed mean backlog ≤ the exact configured target; the tie order is mean energy, mean backlog, then scale index. No completion or slope filter was introduced.\n\n'
        f"There are {report['selected_decisions']} selected target cells and {report['no_feasible_decisions']} cells with no feasible point in the declared grid. Deduplication yields {len(expected_unique)} settings and {selected['planned_test_runs']} planned held-out runs. No-feasible cells remain in the frozen table.\n\n"
        'Source/protocol/validation/record/trace bindings, the complete seed split, and freeze chronology were checked. The selected artifact hash is bound in `SELECTOR_AUDIT.json`; if test metadata already exists, its frozen-choice hash and start time are checked too. Test feasibility and energy comparisons remain separate validation tasks.\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'decision_checks'}, indent=2))


if __name__ == '__main__':
    main()

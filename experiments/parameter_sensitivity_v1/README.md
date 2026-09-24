# Independent QAPG parameter sensitivity

This experiment evaluates the existing QAPG controller (`QAPG-R` in code) at
prespecified one-factor settings. It does not select backlog-budget operating
points, replace comparison data, tune the controller, or exclude unfavorable
settings. The original matched-backlog experiment and its frozen sources remain
unchanged.

The reference is N=100, M=10, beta=2, V=3e11, independent uplink bandwidth2MHz,
and receiver noise1e-10W. One factor varies at a time: nine V scales, five arrival
multipliers, four server counts, or five bandwidths. The shared reference point
is executed once and reused across panels:20distinct settings×five new seeds
(520927–520931)=100runs. Each run has3000slots; reported performance averages
use slots300–2999. Sample SD is computed across five seed-level values, not
across slot samples. Every setting and any failure remains recorded.

Bandwidth experiments hold receiver noise PSD at5e-17W/Hz, so noise power changes
with bandwidth. M experiments change aggregate processing capacity, original
server geometry, destination opportunities, and the existing controller's M/N
weight together. They therefore describe system-size sensitivity, not a pure
causal effect of capacity alone. All other `SimConfig` defaults are preserved.
Arrivals are shared across configurations with the same beta/seed; channels are
shared across configurations with the same M/seed. Both use the unchanged
original generators. No future arrivals are supplied to the controller.

`PROTOCOL.json`, `configurations.json`, and runner/validator hashes are frozen
before generating inputs or running trajectories. The freeze also captures and
checks the earlier controller/source hashes. The simulation imports the original
common bridge rather than modifying it. Per-decision resource feasibility,
queue conservation, and available-work CPU constraints remain checked there.

## Outputs

- `experiments/results/parameter_sensitivity_v1/metadata.json`: completion,
  versions, protocol/source hashes, and wall-clock duration.
- `configurations.json`:20unique settings and their memberships in four grids.
- `inputs/` and `input_manifest.json`: complete shared arrivals/channels and
  their array/file hashes. All paths are project-relative.
- `records/cXX_sSEED.json`: full config, source hashes, input references, metrics,
  physical checks, and any failure. `traces/` preserves every numeric trace.
- `per_seed.csv`:100unique successful-run rows if all runs pass; failures remain
  in records and are counted in metadata. `config_summary.csv`:20unique rows.
- `summary.csv`:23axis/point rows, with the identical reference setting reused
  in each grid. Columns include `axis`, `axis_value`, `config_id`, configuration
  values, seed counts, and metric `_mean`/`_sd` pairs. Partial-seed settings do
  not receive a performance summary.
- `recomputed_per_seed.csv` and `VALIDATION.json`: independent aggregation from
  raw arrays and checks of all saved evidence. This is not a second independent
  algorithm implementation.

Key metrics use J/slot for energy, Mbit for queue backlogs, Mbit/slot for completed
work, fractions for local/edge shares, milliseconds for decision runtime, and
sweep counts for `mean_sweeps`. Completion shares are means of seed-level ratios.
`maximum_objective_rise` covers the full run; other diagnostics use the specified
measurement window. Timing is diagnostic and depends on concurrent workloads.

## Reproduce

Use Python3.12 with NumPy2.3.5 and ReportLab4.4.9, plus clang++ or g++ with C++17
support. The unchanged native controller compiles locally without fast-math.
Use a separate working copy preserving the project-relative layout and an empty
`experiments/results/parameter_sensitivity_v1` output folder. Preserve the
supplied frozen sources; do not rerun over the original archive.

```sh
python3 experiments/parameter_sensitivity_v1/run_sensitivity.py --workers 4
python3 experiments/parameter_sensitivity_v1/validate_sensitivity.py
```

The initial run used `--freeze` first, before all inputs and trajectories. Do not
replace the original freeze after seeing results. Regenerated audit reports
have new timestamps/hashes and belong in the reproduction copy. No plot or
manuscript is written by these scripts.

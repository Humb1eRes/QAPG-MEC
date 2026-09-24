# Matched-backlog comparison: reproduce and audit

This experiment compares energy at a shared **finite-window mean total backlog
requirement**. A backlog requirement is not a measured task deadline or a QoE
score. The protocol retains every method, load, target, and failed operating
point; it does not choose comparators or delete loads after seeing the test set.
The saved experiment and the previous manuscript experiments remain separate.

## What changed in the proposed controller

The original QAPG separately selected local CPU speed, candidate upload actions,
server association, and edge CPU speed. It used an adaptive energy coefficient,
an association load penalty, and an edge pressure multiplier. QAPG-R instead
jointly optimizes each device's local service, upload quantity/power, and server
choice against a fixed-V one-slot objective; it also reoptimizes edge service.
No original heuristic pressure multipliers or adaptive-V coefficients are used
by QAPG-R. The edge quadratic weight is the fixed hardware ratio M/N. This
normalization was specified before trajectory experiments; it is a proxy for
balanced per-origin queues, not an exact identity for unbalanced origins.

See `CONTROLLER_DESIGN.md` for the full objective, scalar derivatives, resource
bounds, initialization, and 12-sweep budget. Every accepted update decreases
the checked one-slot objective, but the controller is not certified globally
optimal. The saved convergence flag distinguishes a small remaining unilateral
gain from termination at the sweep budget. A known configured arrival mean is
used; realized current or future arrivals are never passed to its controller.

All six methods receive the same CPU correction: the selected useful service
determines the maximum CPU frequency charged for that slot. This avoids charging
CPU cycles that cannot process queued work. The common full-slot radio energy,
cubic CPU energy, capacity limits, arrival traces, channel traces, and queue
ordering otherwise remain the same. New uploads reach the edge queue only after
that slot's edge service. Uploaded work is not counted as completed work.

## Comparison methods and provenance

| Method | Implemented rule and scope |
|---|---|
| QAPG-R | New fixed-V joint block controller described above. |
| QAPG-capped | Original QAPG adaptive controller with the common useful-work CPU cap. Its V and adaptive floor scale together. |
| NoQuad-capped | The same original QAPG controller with the association quadratic load penalty set to zero; this is an internal mechanism control, not a literature method. |
| GUPA-O-capped | Orthogonal-link adaptation of Yang et al., IEEE TVT 75(2), 3251–3264 (2026), DOI `10.1109/TVT.2025.3605977`. Retains negative-log transmission-delay utility, null initialization, best total-utility proposals, and single-winner strict-improvement updates. |
| EEDO-adapted | Adaptation of Chen and Jiang, Journal of Cloud Computing 13, 92 (2024), DOI `10.1186/s13677-024-00645-5`. Retains queue-capped local CPU control, fixed-power offloading thresholds, and per-origin queue-priority edge service. |
| BP-Greedy-capped | Existing queue-score/power heuristic with the common CPU correction; no SAC or other learning mechanism is claimed. |

The original Yang model is downlink NOMA. Under the common independent orthogonal
uplinks, its adapted utility increases with rate and power, so GUPA-O selects
the strongest positive channel at maximum power. This analytical equivalence
is documented and independently trajectory-checked in the earlier baseline
audit; the two labels do not represent two independent comparators. The current
GUPA-O comparison retains its utility/update mechanism, uses common CPU rules,
and receives the same work cap as other methods. It does not reproduce the
original NOMA experiment or import its equilibrium guarantees.

EEDO's source uses a single HAP and different processing/transmission costs. The
adaptation extends server choice, retains its threshold and priority structures,
and resolves the service amount under the shared cubic energy model. Full-slot
radio cost is used when residual work would otherwise truncate transmission.
See `../literature_baselines/GUPA_MAPPING.md` and `EEDO_MAPPING.md` for explicit
source-to-code differences. Their original experiment seed lists describe the
earlier baseline audit; **`PROTOCOL.json` governs this experiment**.

## Fixed protocol

There are six methods, three loads (beta 1, 2, and 3), and nine energy scales
(1/16, 1/8, 1/4, 1/2, 1, 2, 4, 8, and 16) multiplying base V=3e11. Adaptive
controllers scale their original floor together with V. Other baseline rules
and parameters are preserved. Four workload-normalized targets are
`rho * N * beta * arrival_lambda * task_packet_bits / 1e6` Mbit, with rho equal
to 1.5, 2, 3, and 5.

Development uses seed 220927, 600 slots, and a 200-slot warmup at scale 1. It is
separate from the selection/test evidence. Selection uses seeds 320927 and
320928, 1500 slots, and a 300-slot warmup: 324 runs across the full common grid.
For each method/load/target, a setting is eligible only if each selection seed
satisfies the backlog bound. The lowest mean energy wins; ties use lower mean
backlog and then scale order. The resulting choices are locked before test runs.

Test uses five new seeds, 420927 through 420931, for 3000 slots with a 300-slot
warmup. Each distinct selected method/load/scale is run once per seed, even if
multiple targets select it. A target passes only if each test seed satisfies
the mean-backlog requirement. A failed setting is not replaced after testing.
“No feasible point” means none in this grid, not inherent infeasibility of the
method. Energy is compared at a target only among passing methods. Seed-level
means and sample standard deviations are reported; slot samples are not treated
as independent replicates. A finite-window result is not a proof of stability
or universal optimality. Full details are frozen in `PROTOCOL.json` and
`FREEZE.json`.

## Portable runtime and directory layout

The recorded simulations use Python 3.12.14, NumPy 2.3.5, and macOS arm64.
The portable code needs:

- Python 3.12 and NumPy; NumPy 2.3.5 matches the recorded environment.
- ReportLab 4.4.9, required because the preserved reference module imports it.
- A C++17 compiler (`clang++` or `g++`) on the executable search path.
- Matplotlib 3.11.1 only when rebuilding the figures.

QAPG-R compiles its adjacent C++ source automatically into a local source-hashed
shared library using `-O3 -std=c++17 -shared -fPIC`, without `-ffast-math`.
Archived binaries and `__pycache__` are unnecessary. The first call can include
compilation overhead; controller timing is diagnostic, and no cross-method
speed superiority is claimed. Bit-for-bit replay is checked on the recorded
runtime; floating-point results on other architectures require their own audit.

Keep the supplied project-relative layout. Required files include the whole
`experiments/matched_backlog_v1` source folder, the frozen controllers under
`experiments/literature_baselines`, `experiments/reference/edgesport_sim.py`,
`experiments/matched_recorder.py`, and `experiments/run_matched.py`. Results are
located at `experiments/results/matched_backlog_v1`, with stage-specific inputs,
traces, records, metadata, and validation reports. Source and input references
are project-relative and do not require the original author's home directory.

## Inspect supplied results without rerunning the grid

Run these commands from the extracted project root using the configured Python:

```sh
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install numpy==2.3.5 reportlab==4.4.9 matplotlib==3.11.1
python3 experiments/matched_backlog_v1/validate_results.py --stage selection
python3 experiments/matched_backlog_v1/validate_results.py --stage test
python3 experiments/matched_backlog_v1/replay_check.py
```

The validators recompute metrics and ledger identities from every saved trace,
checking input/source hashes and the exact planned identity grid. They write
their validation reports; they do not rerun controllers or modify saved traces.
`replay_check.py` separately reruns exactly two identities specified before
selection completed: QAPG-R and NoQuad-capped at beta=2, scale index=4 (scale 1),
seed 320927. It compares every numeric trace except `runtime_ms` exactly, reports
maximum absolute differences, and leaves saved trajectories intact. These two
deterministic replays are not an independent controller implementation and do
not certify that every saved run was replayed. Controller numerical tests and
the complete trace audit supply different, explicitly stated evidence.

## Generate a fresh experiment without overwriting the archive

Create a **separate copy of the code tree** with the same relative layout and no
`experiments/results/matched_backlog_v1` directory. Retain the supplied frozen
source and protocol files. Work in that new project root; the runner resolves
its output path from its own location and has no arbitrary output-path option.
Keep the archived project unchanged. Run:

```sh
python3 experiments/matched_backlog_v1/test_qapg_revised_api.py
python3 experiments/matched_backlog_v1/test_qapg_revised_math.py
python3 experiments/matched_backlog_v1/run_protocol.py --stage development --workers 4
python3 experiments/matched_backlog_v1/validate_results.py --stage development
python3 experiments/matched_backlog_v1/run_protocol.py --stage selection --workers 4
python3 experiments/matched_backlog_v1/validate_results.py --stage selection
python3 experiments/matched_backlog_v1/select_settings.py
python3 experiments/matched_backlog_v1/run_protocol.py --stage test --workers 4
python3 experiments/matched_backlog_v1/validate_results.py --stage test
python3 experiments/matched_backlog_v1/analyze_holdout.py
python3 experiments/matched_backlog_v1/make_figures.py
python3 experiments/matched_backlog_v1/replay_check.py
```

The runner refuses an existing stage unless `--resume` is explicitly supplied.
Use `--resume` only for an interrupted run in the new reproduction tree with
unchanged frozen sources; it checks existing records and trace checksums. Never
use it to combine results from changed code. Compiler/input/code changes require
a separately versioned experiment. Keep all failed targets and all selection
grid points in any resulting report.

# QAPG parameter sensitivity: code and complete results

This archive contains an independent parameter study of the existing QAPG
algorithm. It includes all 20 configurations, 100 runs, shared random inputs,
raw trajectories, frozen sources, per-seed metrics, and five-seed summaries.
There is no backlog-budget parameter selection in this study.

The four sweeps vary energy weight, arrival multiplier, server count, and link
bandwidth individually. The default setting is shared across all four sweeps.
Bandwidth changes retain a fixed receiver-noise PSD. Changing server count
changes capacity, geometry, available associations, and the algorithm's
dimension-derived queue weight together. All settings and any failure are
retained; this package does not replace the earlier comparative experiment.

Start with `experiments/parameter_sensitivity_v1/README.md` and `PROTOCOL.json`.
The main numerical outputs are under
`experiments/results/parameter_sensitivity_v1/`:

- `summary.csv`: 23 grid rows, including the shared default in each sweep.
- `config_summary.csv`: 20 unique configuration summaries.
- `per_seed.csv`: 100 unique seed/configuration rows.
- `records/`, `traces/`, and `inputs/`: full provenance and numerical evidence.
- `VALIDATION.json`: complete raw-trace/configuration/summary checks.

Public **QAPG** uses the preserved internal name **QAPG-R**. The included earlier
controller freeze records identify its source version; earlier comparison
results are intentionally not copied into this package. Required common
simulator and controller files retain their original relative directory layout.
Compiled native libraries and caches are excluded; C++ source is included.

Use Python 3.12, NumPy 2.3.5, ReportLab 4.4.9, and a C++17 compiler (clang++ or
g++). A compiler is required to rerun simulations, but the saved-data validator
does not compile or run the controller. No plotting dependency is needed.

Preserve this extracted archive as immutable evidence. Create a separate working
copy before running any audit or simulation script: audit scripts regenerate
reports with new timestamps and hashes, so old report provenance hashes are not
expected to remain unchanged in that working copy.

To check the supplied numerical evidence in the working copy, run from its root:

```sh
python3 experiments/parameter_sensitivity_v1/validate_sensitivity.py
```

To reproduce simulations, use a second source copy without the
`experiments/results/parameter_sensitivity_v1` folder, preserve the supplied
frozen source/protocol files, and follow the source README. The runner refuses
to overwrite an existing result tree. Do not refreeze sources after inspecting
the results.

`MANIFEST.json` lists SHA-256 checksums and byte counts for every other packaged
file. The package build separately checks every ZIP member's CRC and SHA-256,
then reruns the full numerical validator after extraction into an isolated
directory with filesystem access to the original project explicitly blocked.
That portability check validates all saved evidence; it is not an independent
reimplementation or a rerun of all 100 simulations.

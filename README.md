# QAPG for multi-server mobile edge computing

Reproduction materials for queue-aware computation offloading and resource
allocation with monitoring-device and edge-server queues.

QAPG jointly updates local computation, offloaded data volume, edge-server
association, and edge computation. The experiments include comparisons at common
mean-backlog budgets and parameter sweeps for energy weight, task arrivals,
server count, and uplink bandwidth.

## Start here

- [Comparison protocol and reproduction instructions](experiments/matched_backlog_v1/REPRODUCE.md)
- [Parameter-study instructions](experiments/parameter_sensitivity_v1/README.md)
- [Twelve-condition comparison summary](experiments/results/matched_backlog_v1/analysis/ALL_TARGET_SUMMARY.csv)
- [Parameter-study results](experiments/results/parameter_sensitivity_v1/summary.csv)
- [Figure data](figure-data)

The source code and main numerical summaries can be browsed in this repository.
The **v1.0.0 release** contains two complete experiment archives, including saved
inputs, every trajectory, per-run records, frozen sources, and validation reports.
Download each archive and follow its own README in a separate working directory.
The smaller repository-upload archive also contains extended numerical summaries
and provenance records.

## Runtime

Use Python 3.12 and a C++17 compiler (clang++ or g++):

```sh
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

The native QAPG component is compiled automatically from the included C++ source.
Preserve the project-relative folder layout. Full-data validators require the
corresponding complete release archive. Existing results should be kept in an
immutable copy; validation scripts write new reports. Follow the protocol-specific
instructions to run fresh simulations in a separate working copy.

## Names used in code

| Manuscript name | Implementation name |
|---|---|
| QAPG | QAPG-R |
| Sequential-QAPG | QAPG-capped |
| Sequential-NoQuad | NoQuad-capped |
| GUPA-O | GUPA-O-capped |
| EEDO | EEDO-adapted |
| BP-Greedy | BP-Greedy-capped |

The literature comparisons are adaptations to a shared simulation model. Their
source mechanisms and modifications are documented in
[the GUPA mapping](experiments/literature_baselines/GUPA_MAPPING.md) and
[the EEDO mapping](experiments/literature_baselines/EEDO_MAPPING.md).

## Complete data

The release includes:

- `QAPG_R_Matched_Backlog_Code_and_Results.zip`: complete comparison study.
- `QAPG_Parameter_Sensitivity_Code_and_Results.zip`: complete parameter study.
- `QAPG_SR4_13_Repository_Upload.zip`: extended code and numerical-summary package.
- `SHA256SUMS.txt`: checksums for the three downloadable archives.

All experimental outcomes and the original numerical evidence are retained.

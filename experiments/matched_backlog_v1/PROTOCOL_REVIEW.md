# Matched-backlog experiment: independent protocol review

## Material Passport

- Date: 2026-09-22. Scope: a new algorithm revision and energy comparison under the same aggregate-backlog requirement.
- Inputs reviewed: the common simulation configuration, SR3 baseline protocol, and the complete 2026-09-22 performance diagnosis. No new selection or held-out outcome was consulted for this recommendation.
- Status: **ADOPTED DESIGN REVIEW**, including the six-method amendment agreed before selection. This is not completed experiments or a preregistration at an external registry. The runner's frozen manifest must record the settings actually adopted and the algorithm/source hashes before selection begins.
- Earlier SR3 seeds, the previous development seed 190922, and all earlier results are development evidence. They are preserved and cannot become new held-out evidence.

## 1. Question and fixed evaluation domain

For each offered load and a common finite-window mean aggregate-backlog upper bound, which candidate setting of each method has the smallest measured energy? This is a constrained operating-point comparison, not a proof of global optimality or equal per-task delay.

Keep all six methods: **QAPG-R**, **QAPG-capped**, **NoQuad-capped**, **GUPA-O-capped**, **EEDO-adapted**, and **BP-Greedy-capped**. NoQuad-capped is the previous no-quadratic-load component comparison with the common CPU correction; its strong earlier performance is a reason to retain it. Keep all three loads β = 1, 2, 3. Do not remove a method, load, or target based on its ranking. Any algorithm or protocol changes after opening selection results start a separately labelled version; changes after opening test results require new test seeds.

QAPG-R jointly minimizes the expected next-slot **weighted** quadratic queue objective plus V times energy. Its device-square coefficient is 1 and its server-square coefficient is **M/N**, fixed from pool sizes before the new trajectory experiments. If a server nominally pools N/M equal origin queues, this weighting equals their sum of squares; for unequal or differently populated origin queues it is a chosen proxy, not that identity or an identity for aggregate backlog. The arrival mean is obtained from the shared configuration (β, arrival_lambda and task_packet_bits); all methods have access to that same configuration. QAPG-R uses a fixed V at each candidate grid setting, without the old a/b/gamma terms or adaptive-V rule. It is an explicitly revised controller rather than a relabelled rerun of QAPG. Separate analytic and numerical review is recorded in `MATH_REVIEW.md` and `QAPG_REVISED_MATH_VALIDATION.json`; it does not imply a favorable performance ranking.

Use the same original hardware, orthogonal-link channel model, arrival process, empty initial queues, and service timing. Old edge work alone is executable in a slot; new uploads and arrivals enter their queues for the next slot. All methods receive the same arrivals and channels for a given load/seed, verified by input hashes. No controller can inspect future inputs.

For every method, cap device and server CPU frequencies by both hardware limits and the frequency needed for that slot's available service. Charge the same full-slot cubic CPU energy and full-slot transmit energy. Preserve GUPA's identifiable negative-log-delay utility, one-winner updates, and maximum-power orthogonal-link rule; its CPU energy scale can be tuned, but its radio power must not silently become an energy optimizer. Preserve the documented EEDO and BP mechanisms. A shared physical executor should validate actions, queue conservation, and all energy components.

## 2. Targets independent of algorithm outcomes

Define expected total arrivals per slot as λβ = N × β × arrival_lambda × task_packet_bits. With the current configuration these are 4.8, 9.6, and 14.4 Mbit/slot. Define target B* = ρ λβ for **ρ ∈ {1.5, 2, 3, 5}** before generating selection results.

| β | Expected arrivals (Mbit/slot) | Four backlog upper bounds (Mbit) |
|---|---:|---|
| 1 | 4.8 | 7.2, 9.6, 14.4, 24.0 |
| 2 | 9.6 | 14.4, 19.2, 28.8, 48.0 |
| 3 | 14.4 | 21.6, 28.8, 43.2, 72.0 |

These are prespecified evaluation requirements, not a claim that a real application supplied them. Describe ρ as backlog in arrival-slot units. Do not present it as directly measured task latency or invoke steady-state Little's law without its assumptions.

The measured backlog is the arithmetic mean of end-of-slot sum(Q) + sum(H) over the stated observation window. Device and edge queues use bits consistently. Use the expected arrival rate for the bound; do not make a seed's bound easier by substituting its observed arrival rate.

## 3. Equal, bounded parameter budgets

Adopted common energy-scale grid: V = 3 × 10^11 × {1/16, 1/8, 1/4, 1/2, 1, 2, 4, 8, 16}. This gives **nine candidate settings per method and load**. Freeze all other method parameters and numerical stopping criteria before selection. The development pilot checks the fixed design and implementation; selection/test results cannot choose its structure.

Both QAPG-capped and NoQuad-capped use the previous adaptive V with an absolute floor. Scale both their base V and adaptive_v_min by the grid multiplier, retaining the ratio 8/30 = 4/15 and all other original coefficients (including the existing zero quadratic penalty in NoQuad-capped). Otherwise several small V settings collapse to the same controller and do not constitute a comparable search range. Describe this as tuning the complete energy-penalty scale, not changing the old adaptive mechanism. For QAPG-R and the fixed-V baselines, change their fixed energy scale only.

The same numerical grid is appropriate only while the revised objective uses the original queue-energy units and normalization. Record any nondimensional conversion explicitly; apply corresponding physical energy weights rather than equating incompatible numbers. If the selected value lies at a grid boundary, report that boundary and describe the search as limited to the declared grid. Do not extend only a losing method or only the proposed method after test results are known.

## 4. Seed split and execution budget

| Stage | Seeds | Horizon / warm-up | Permitted use |
|---|---|---|---|
| Development pilot | 220927 only; 190922 is already-seen development evidence and 220928 remains unused | 600 / 200, β = 1, 2, 3 at V = 3 × 10^11 only | correctness and runtime checks of the fixed design; no performance-based tuning |
| Selection | 320927, 320928 | 1500 / 300 | run every declared grid point and freeze the selected setting for every method/load/target |
| Held-out evaluation | 420927, 420928, 420929, 420930, 420931 | 3000 / 300 | evaluate frozen selections; no tuning, substitution, or selective seed deletion |

These seed lists were searched in existing JSON and Python experiment records without a previous match; actual input/run manifests must still carry them. No selection seed can be promoted to held-out evaluation. A longer held-out horizon is intentional; failure of a selection-feasible point to satisfy the requirement at that horizon must be retained.

Freeze the protocol JSON before the development pilot. After correctness review and any documented bug fixes, freeze the implementation before selection. The development pilot uses only seed 220927, all three loads and the single base V; it does not authorize searching design choices for the most favorable pilot ranking. Save and lock the selected configuration table after selection and before any held-out execution.

The selection stage has 6 methods × 3 loads × 9 candidates × 2 seeds = **324 runs**, or 486,000 simulated slots. Two selection seeds are a bounded search budget, not a robust population guarantee. The held-out stage has at most 6 × 3 × 4 × 5 = **360 runs**; deduplicate identical method/load/V settings selected for different targets. Each distinct setting runs only once per held-out seed and its measurements are reused without treating that reuse as independent evidence. Save every selection point, including infeasible and dominated points.

## 5. Selection and held-out classification

For a fixed method, load and target, a candidate is selection-feasible only when **each of the two selection-seed mean backlogs is ≤ B***. Among feasible candidates choose the lowest arithmetic mean energy across the selection seeds. Resolve exact numerical ties by smaller mean backlog, then smaller V. Never use held-out energy or backlog to pick the setting.

If no grid candidate is selection-feasible, record **no feasible candidate in the declared grid**. This is not proof that the method cannot meet the bound. Do not replace the method, interpolate an unrun controller, or enlarge the target. Its full selection frontier remains visible. An optional minimum-backlog diagnostic is permissible only if declared and distinctly labelled; it cannot be reported as a selected feasible competitor.

Before opening held-out results, save a selection table containing method, load, target, selected V, selection energy/backlog for both seeds, and the selector/config/code hashes. Held-out classification uses the same bound: a selected setting is empirically test-feasible only if **all five held-out seed means are ≤ B***. Report the fraction passing and the worst-seed excess even when this all-seed criterion fails. Retain failed test points and their energy, explicitly marked as not meeting the requirement. Do not replace them with another V using the test set.

An energy comparison “under the same backlog requirement” is supported only where the compared selections both meet that same held-out requirement. Their actual backlogs need not be equal; report both. An operating point with a lower energy but excess backlog is not an energy win under that constraint. Do not claim an advantage over a competitor with no feasible selection as if it had a measured feasible energy.

## 6. Diagnostics, uncertainty, and complete reporting

For every run save mean energy and its device/radio/edge components; mean device/edge/total backlog; total completed work and arrivals over the observation window; completion-to-arrival ratio; final backlog; upload and local/edge completion amounts; and an OLS total-backlog slope over the last half of the horizon (750–1499 for selection, 1500–2999 for test). Also report that slope divided by λβ. Preserve conservation residuals and action-bound checks. CPU service caps must hold on individual devices/servers, not merely in aggregate.

Completion and late growth are diagnostics, not secretly added selection criteria. Flag completion/arrival below 0.99 or positive late growth above 0.005 λβ per slot for interpretation, without deleting the run. Such flags weaken a stable-operation interpretation even if a finite-window backlog mean passes. Ratios above one can occur when a window drains backlog accumulated before the window; reconcile them with its opening and closing queue values.

Report the entire **selection-grid energy–backlog frontier**, all grid points and infeasibility labels, plus all frozen held-out operating points across all loads and targets. Do not call the selected-only held-out points a complete test frontier. Do not interpolate an untested policy or present a convex combination of settings as an executable controller.

The experimental replicate is a seed, not a slot. Report held-out means and sample standard deviations across five seeds. For a fixed target/load where both methods pass, paired seed energy differences can be accompanied by their mean and a descriptive 95% Student-t interval (df = 4, t = 2.776); note the small-sample distributional assumption. Do not treat the 2700 correlated slots as independent samples, average rankings across unequal loads, or claim universal/statistical optimality from a limited grid. Avoid a collection of uncorrected significance claims across the 12 load/target cells. A percentage energy difference should identify its baseline and use the same paired held-out runs.

## 7. Acceptance checks before writing a new manuscript result

1. Frozen manifest predates selection outputs; selected configurations are saved before test execution; both include source hashes and all candidate/seed lists.
2. All methods use identical physical configuration and input hashes for each shared run. All CPU corrections apply to every method; no baseline is left charging avoidable idle CPU energy.
3. Unit checks cover objective derivatives/feasibility, null and zero-backlog actions, server association and service timing. Conservation and energy are independently recomputed on held-out traces.
4. Every scheduled run is accounted for. Failures are documented; deterministic implementation fixes trigger a new version and appropriately repeated affected comparisons, rather than silent row removal.
5. The report retains all loads, targets, methods and selection candidates, identifies finite-grid and finite-horizon scope, and explains any failure of held-out feasibility. New algorithm results do not silently overwrite SR3.

This protocol can establish a reproducible constrained comparison in the stated simulator. It cannot guarantee that the revised method is the best possible algorithm, nor that all finite-window feasible policies are stable indefinitely.

## 8. Research applicability and limits

- **Appropriate claim:** among the six retained controllers and their declared energy-scale grid, a fixed selection rule identifies operating points, and independent finite-horizon runs compare energy where both points satisfy the same aggregate-backlog requirement. If QAPG-R wins only some cells, name those cells or the supported load/requirement range.
- **No global optimum inference:** exact continuous resource substeps or monotone coordinate improvement of a one-slot objective do not prove global optimality of mixed associations, the best long-run constrained policy, or superiority over untested methods/settings. A separate proof would be needed for any such claim.
- **Information and model scope:** QAPG-R uses the correctly specified configured arrival mean. The experiment does not test robustness to unknown or changing arrival rates, estimation error, shared-bandwidth interference, NOMA scheduling, deadline violations, or real task traces. Avoid converting this known-model result into a general MEC claim.
- **Baseline scope:** GUPA-O-capped and EEDO-adapted are documented transfers into this orthogonal uplink/common CPU model; their result is not a reproduction of their source papers' original NOMA or other physical environments. NoQuad-capped is an informative strong component control, not an independently published algorithm.
- **Finite-sample scope:** two selection seeds and five test seeds provide a bounded reproducible comparison. All-seed observed feasibility is not a probabilistic service guarantee, and a 3000-slot trace cannot establish indefinite queue stability. Four prespecified backlog levels provide discrete coverage, not the full continuous energy-delay frontier.
- **Development provenance:** earlier adverse results legitimately motivated QAPG-R. Keep that history and label the controller revision explicitly; the new test split supplies independent evidence for the resulting frozen design rather than erasing the old result.

## 9. Pre-selection implementation adherence review

Reviewed `run_protocol.py`, `common_bridge.py`, the called metric definitions in `run_matched.py`/`matched_recorder.py`, and common executor policy mappings. The current runner expands all six methods across all three loads, nine scales and two selection seeds. It changes only V and the declared adaptive floor at this stage; NoQuad's original zero-load-penalty override is carried by the shared executor and recorded separately. GUPA/EEDO/BP route to the source-frozen controllers.

The bridge applies its CPU service-preserving cap to each controller, including QAPG-R, and checks local service equality. The physical executor validates edge service/capacity, timing, inputs and action bounds and computes the common energy formula. Inputs are loaded read-only and their content hashes checked before and after a run. Parallel execution uses separate processes, avoiding concurrent mutation of the bridge's temporary per-process controller routing.

The runner imports the original metric implementation: aggregate backlog is end-of-slot queue mean after the fixed warm-up, with opening/closing queue reconciliation, completed/arrival ratio, and OLS slope over the literal latter half of the horizon. Selection and test source hashes are checked against a freeze record. Test settings come only from the frozen selection artifact and are deduplicated upstream; the test metadata records that artifact's hash. The selector itself and eventual test results still require independent review when available.

At review time the development metadata records **18/18 runs completed**, with six methods, β = 1, 2, 3, seed 220927, T = 600, warm-up 200, and base V only. This is development evidence; no ranking was used in this protocol review. Current controller hashes agree with the independent mathematical/API validation records. No source changes were made by this reviewer.

No blocking deviation was identified for first execution into a fresh selection/test stage. The original reviewed resume path reused records after checking status and trace hash, without separately checking each stored identity/config/input against the new job; a test resume likewise needs to keep its selected-settings hash fixed. These are provenance-hardening considerations for resumed or externally modified output folders, not evidence of a defect in the fresh development runs. Final result validation must independently check each record's identity/config/input against its declared stage and confirm unchanged frozen selection settings. Any runner hardening before the formal freeze should be identified by its updated source hash; the physical controller need not be changed.

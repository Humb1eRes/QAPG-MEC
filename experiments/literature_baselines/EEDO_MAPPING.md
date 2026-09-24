# EEDO-adapted: source-to-implementation mapping

## Material Passport

- Scope: implement and verify a literature-based comparison controller; no performance ranking is assumed.
- Source: Sihan Chen and Wanchun Jiang, “Online dynamic multi-user computation offloading and resource allocation for HAP-assisted MEC: an energy efficient approach,” *Journal of Cloud Computing* **13**, 92 (2024), [doi:10.1186/s13677-024-00645-5](https://doi.org/10.1186/s13677-024-00645-5).
- Local source: `../../literature/baseline-audit/recent/pdfs/Chen_Jiang_2024_EEDO.pdf`.
- PDF SHA-256: `45f883e3ccf9133942329561477af983e335c4a52e979198b155d2c35a0595d1`.
- Structural check: PASS, 12 declared/enumerated/readable pages, no parser warnings.
- Read scope: model and energy equations (5)–(12), optimization equations (23)–(29), Algorithm 1, and simulation settings; equation page 6 and algorithm page 7 also visually inspected. This is an agent read record, not an author-attested full-paper reading claim.
- Implementation: `eedo.py`. The original reference simulator is unchanged.

## What the name identifies

**EEDO-adapted** retains the source's local CPU rule, fixed-power queue-difference offloading structure, and per-device edge-queue priority allocation. It is a multi-server adaptation evaluated with this study's workload, channels, hardware limits, and energy accounting. It is not a reproduction of the source's single-HAP experiment.

This controller must not be described as the old Fixed-power heuristic with a new label. It maintains one queue per device and server, explicitly allocates each server's service among those queues, and retains the source's available-work CPU cap.

## State and decisions

The controller receives slot-start device backlogs `Q[n]`, per-device edge backlogs `B[n,m]`, and channel power gains `gains[n,m]`. The aggregate edge backlog used in the manuscript is exactly `H[m] = sum_n B[n,m]`. A device may have unfinished work at multiple servers after changing association; that work remains at its original server.

`eedo_decision(Q, B, gains, cfg)` returns:

| Dataclass field | Shape | Meaning |
|---|---|---|
| `local_cpu_hz` | N | Device CPU frequencies |
| `transmit_power_w` | N | Fixed active transmit power, zero for no upload |
| `association` | N | Selected server, or −1 for no upload |
| `uploads_bits` | N | Work transferred from the device queue |
| `edge_cpu_hz` | M | Server CPU frequencies implementing the allocated service |
| `edge_service_bits` | N×M | Work completed from each pre-existing edge queue |
| `diagnostics` | dict | Candidate, cost-correction, and piecewise-service counts; edge objective values |

Inputs are not mutated. The caller executes the common queue update and charges all active CPU and radio resources over the whole slot.

## Rules retained from the source

1. **Local CPU, equation (25).** With the manuscript's symbols, the rule is

   `f_n = min(sqrt(Q_n/(3 V κ_d c_d)), f_max, Q_n c_d/τ)`.

   The last term is present in the source and prevents selecting more local computation than the available work. This is a property of the comparison controller, not a change to QAPG.

2. **Offloading, equation (27).** The source tests the sign of `V p/R_nm − Q_n + B_nm` at fixed exogenous transmit power. It uploads the largest feasible amount when this coefficient is nonpositive. The threshold uses **slot-start Q**, as in the source. For the common local-first execution order, the available amount is capped by the queue remaining after local service, preventing the same bit from being served twice. The adaptation keeps this source test as an initial filter, then uses the full-slot cost test below. The source has one destination; the adaptation evaluates all servers.

3. **Edge queue priority, equation (29) and Algorithm 1.** The source orders the per-device edge queues by increasing `V ℓ_1 − B_nm`. Because its per-bit energy coefficient is common, this is descending queue length. It then fills them in that order subject to the shared server capacity. The adaptation preserves this ordering and each queue's individual feasibility bound.

## Required changes for the common physical model

The source uses a single HAP and charges transmission energy as `p u/R`, with linear server processing energy `ℓ_1 d`. This study uses multiple servers and full-slot energy `τp + τκ_d f³ + τκ_e F³`. Both the upload acceptance test and the edge-service amount are therefore re-evaluated under the common cost model, without adding or tuning any parameter.

**Upload acceptance.** For every eligible fixed-power link, define the feasible amount `u_nm = min(Q_n − l_n, τR_nm)` and the full-slot benefit

`S_nm = (Q_n − B_nm) u_nm − Vτp_fixed`.

The source filter checks `V p_fixed/R_nm − Q_n + B_nm ≤ 0`. The common-cost adaptation additionally requires **S_nm > 0** and selects the server with the highest positive S_nm; ties against no upload retain no upload and zero power. Equal positive server scores retain the lowest server index. For `u_nm=τR_nm`, this score is exactly the negative source objective in equation (26), so the decisions agree whenever the source's gain is strictly positive. For a residual-capped amount, `u_nm<τR_nm`, using the source's activity-time cost alone can admit an upload that costs more than its backlog benefit under the full-slot ledger; the second test prevents that inconsistency. Transmit power is the existing common setting `cfg.fixed_power_w`, clipped to the common maximum (default 0.08 W).

`source_accepted_candidates` counts all evaluated device/server candidates with positive upload capacity and a nonpositive source coefficient **before choosing one server per device**. `source_accepted_common_rejected_candidates` counts the subset with S_nm≤0, including exact zero-score ties. The latter is a candidate count, not a count of transmitted users or errors. `link_candidates` includes all servers considered for a device with positive residual workload and positive fixed power, including any zero-rate links. These diagnostic definitions allow inspection of the cost correction without changing the baseline parameters.

**Edge service.** Copying the HAP's per-bit coefficient into the current cubic model would add an unrelated hardware parameter. Instead, its queue-priority mechanism is retained and its service-amount problem is re-solved as follows.

For server m and total service s, the allocation `d_n(s)` fills the queues in descending `B_nm` order. Let `c_e` denote the edge CPU cycles per bit. The one-dimensional objective is

`V κ_e c_e³ s³/τ² − sum_n B_nm d_n(s)`,

subject to `0 ≤ s ≤ min(sum_n B_nm, τ F_max/c_e)`.

Within a segment where queue i receives the next units of service, the stationary point is

`s* = sqrt(B_im τ²/(3 V κ_e c_e³))`.

The implementation projects this point onto that segment, compares the feasible segment minima including zero service, and returns the global minimum of this convex scalar problem. The server clock is `F_m = c_e s/τ`. Thus the **priority mechanism is retained**, while the original constant-cost threshold in equation (29) is replaced by its counterpart under the shared cubic energy model. No claim from the original HAP stability theorem is transferred to this adaptation.

## Queue timing and energy ledger

Local work is `l_n = τ f_n/c_d`. Given arrivals A at the slot end, the caller executes:

`Q'_n = Q_n − l_n − u_n + A_n`,

`B'_{nm} = B_{nm} − d_{nm} + 1{a_n=m} u_n`.

Both right-hand sides are nonnegative up to floating-point roundoff. The edge allocator only sees slot-start B, so new uploads cannot be completed immediately in the same slot. The update preserves

`sum Q' + sum B' = sum Q + sum B + sum A − sum l − sum d`.

Every comparison must include both device and edge energy, completed local work, completed edge work, and their associated backlogs. Transfer to an edge queue is not completion.

## Verification completed

`test_eedo.py` checks empty states and zero-rate links, local available-work caps, full-slot equivalence to the source threshold, zero-score inactivity, a partial-slot example requiring the corrected cost test, the single-queue closed-form edge solution, 100 independently generated multi-queue edge problems against a derivative-free scalar minimizer, and a 100-slot flow-conservation sequence. Inputs remain unchanged and every per-device/per-server service allocation respects the hardware and queue constraints. Eight checks passed on 2026-09-18. These are controller-correctness checks, not evidence that it outperforms any other method.

## Suggested manuscript description

**EEDO-adapted** follows Chen and Jiang in applying queue-capped local CPU control, fixed-power threshold offloading, and descending-backlog service priorities for individual edge queues. We extend its destination choice to multiple servers and evaluate its upload threshold and edge-service amount using the common full-slot radio and cubic CPU energy costs. All methods receive the same arrivals, channels, and hardware limits.

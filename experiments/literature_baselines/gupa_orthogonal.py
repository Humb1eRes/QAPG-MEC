"""GUPA-O: an orthogonal-link adaptation inspired by Yang et al.'s GUPA.

Source: Yang et al., IEEE TVT 75(2), 3251--3264 (2026),
doi:10.1109/TVT.2025.3605977; Eqs. (17)--(19), (24), Algorithm 1.

Retained mechanisms are negative-log transmission-delay utility, all-null
initialization, each user's best system-QoE proposal, and strict single-winner
updates. The common simulator has independent uplinks and per-device power
limits. NOMA interference, SIC, shared channel power, and server proximity
constraints are absent. Positive delay utility is explicitly adopted for every
feasible assignment, while null utility is zero. This admission convention is
an adaptation, not an affine transformation preserving the full original game.

In this independent-link model delay utility increases with transmit power.
Trial and committed actions both use the common p_max_w. This re-solves power
selection for the common model rather than reproducing the source's p_min
trial / p_max commit. No NOMA equilibrium or price-of-anarchy claim is made.
CPU choices come from the common fixed-V controller, not the source paper.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from eedo import EEDODecision


def _by_origin_service(B: np.ndarray, aggregate: np.ndarray) -> np.ndarray:
    """Bookkeeping only, matching the common aggregate-service ledger."""
    service = np.zeros_like(B)
    for m, amount in enumerate(aggregate):
        remaining = float(amount)
        for n in np.argsort(-B[:, m], kind="stable"):
            if remaining <= 0:
                break
            done = min(float(B[n, m]), remaining)
            service[n, m] = done
            remaining -= done
        if remaining > 1e-5:
            raise AssertionError("Aggregate service exceeds source ledger")
    return service


def gupa_qoe_updates(
    residual_bits: np.ndarray, gains: np.ndarray, cfg: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Execute the separable QoE game from a null strategy profile.

    D_n is the device's residual work after common local service. Within the
    slot, log(T_max) = log(max positive D) - log(min positive feasible R) is
    fixed, and assigned utility is 1 + log(T_max) - log(D_n/R_nm). Thus each
    feasible utility is at least one. The offset affects admission versus null
    and is disclosed as an explicit convention, not a source calibration.

    Every candidate is enumerated by utility; no channel-gain argmax or legacy
    heuristic is invoked. Candidate utilities are cached because another user's
    update changes no rate or feasibility constraint. Lowest user index wins
    each round (source arbitration is unspecified). Once committed, that user's
    cached best response cannot change, so the exact sequence can be executed
    without redundantly re-enumerating unchanged proposals in every round.
    """
    residual = np.asarray(residual_bits, dtype=float)
    gains = np.asarray(gains, dtype=float)
    if residual.shape != (cfg.N,) or gains.shape != (cfg.N, cfg.M):
        raise ValueError("Expected residual_bits:(N,) and gains:(N,M)")
    if any(not np.all(np.isfinite(x)) or np.any(x < 0) for x in (residual, gains)):
        raise ValueError("Residual work and gains must be finite and nonnegative")
    if min(cfg.noise_w, cfg.bandwidth_hz, cfg.tau) <= 0 or cfg.p_max_w < 0:
        raise ValueError("Noise, bandwidth, and slot length must be positive; power nonnegative")

    rates = np.zeros((cfg.N, cfg.M), dtype=float)
    active_users = np.flatnonzero(residual > 0)
    link_candidates = 0
    if cfg.p_max_w > 0:
        for n in active_users:
            for m in range(cfg.M):
                # Match the common physical uplink arithmetic exactly. The
                # logarithm for physical rate is base two, QoE uses natural log.
                snr = cfg.p_max_w * float(gains[n, m]) / cfg.noise_w
                rates[n, m] = cfg.bandwidth_hz * math.log2(1.0 + max(snr, 0.0))
                link_candidates += 1
    positive_rates = rates[rates > 0]
    log_t_max = None
    if len(active_users) and len(positive_rates):
        log_t_max = math.log(float(np.max(residual[active_users]))) - math.log(float(np.min(positive_rates)))

    best_server = np.full(cfg.N, -1, dtype=int)
    best_qoe = np.zeros(cfg.N, dtype=float)
    candidate_qoe = np.full((cfg.N, cfg.M), -np.inf, dtype=float)
    if log_t_max is not None:
        for n in active_users:
            log_d = math.log(float(residual[n]))
            for m in range(cfg.M):
                rate = float(rates[n, m])
                if rate <= 0:
                    continue
                utility = 1.0 + log_t_max - log_d + math.log(rate)
                candidate_qoe[n, m] = utility
                # Stable equal-utility ties retain the lowest server index.
                if utility > best_qoe[n]:
                    best_qoe[n] = utility
                    best_server[n] = m

    # Algorithm 1 initialization: no user is assigned and all QoE values are 0.
    association = np.full(cfg.N, -1, dtype=int)
    powers, uploads = np.zeros(cfg.N), np.zeros(cfg.N)
    current_qoe = np.zeros(cfg.N)
    objective_path = [0.0]
    accepted_users: list[int] = []
    accepted_servers: list[int] = []
    qoe_gains: list[float] = []
    # All users propose their best system-QoE change. In the separable model,
    # this is best_qoe[n] - current_qoe[n], independently of the other choices.
    # The source does not define how competitors are arbitrated; select the
    # lowest pending index deterministically and commit exactly one per round.
    pending = [int(n) for n in active_users if best_qoe[n] > 0]
    for n in pending:
        gain = float(best_qoe[n] - current_qoe[n])
        candidate_total = objective_path[-1] + gain
        if not gain > 0 or not candidate_total > objective_path[-1]:
            raise AssertionError("A GUPA-O update must strictly increase system QoE")
        m = int(best_server[n])
        association[n] = m
        powers[n] = cfg.p_max_w
        uploads[n] = min(float(residual[n]), cfg.tau * float(rates[n, m]))
        current_qoe[n] = best_qoe[n]
        accepted_users.append(n)
        accepted_servers.append(m)
        qoe_gains.append(gain)
        objective_path.append(candidate_total)

    terminal_gain = float(np.max(best_qoe - current_qoe, initial=0.0))
    if terminal_gain > 0:
        raise AssertionError("Terminated while a strictly improving update remains")
    diagnostics = {
        "source": "Yang et al., IEEE TVT 75(2):3251-3264 (2026)",
        "source_doi": "10.1109/TVT.2025.3605977",
        "adaptation": "GUPA-O",
        "admission_convention": "feasible assigned utility >= 1; null utility = 0",
        "power_adaptation": "trial and committed power use independent-device p_max_w",
        "winner_rule": "lowest index among strictly improving users",
        "candidate_cache": "independent-link utilities and feasibility remain unchanged by other users",
        "accepted_users": accepted_users,
        "accepted_servers": accepted_servers,
        "qoe_gains": qoe_gains,
        "objective_path": objective_path,
        "candidate_qoe": candidate_qoe,
        "assigned_qoe": current_qoe,
        "log_t_max": log_t_max,
        "rounds": len(accepted_users) + 1,
        "iterations": len(accepted_users),
        "link_candidates": link_candidates,
        "qoe_candidate_evaluations": int(np.sum(np.isfinite(candidate_qoe))),
        "uncached_link_evaluations": link_candidates * (len(accepted_users) + 1),
        "qoe_total": objective_path[-1],
        "qoe_accepted_updates": len(accepted_users),
        "qoe_terminal_gain": terminal_gain,
    }
    return association, powers, uploads, diagnostics


def gupa_orthogonal_decision(
    Q: np.ndarray, H: np.ndarray, B: np.ndarray, gains: np.ndarray, cfg: Any,
) -> EEDODecision:
    """Combine GUPA-O association with the common fixed-V CPU controls.

    Q:(N,) and H:(M,) are slot-start queues. B:(N,M) only attributes queued
    edge work to originating devices; it changes neither utility nor controls.
    The caller charges full-slot energy and updates all queues after decisions.
    """
    Q, H, B, gains = (np.asarray(x, dtype=float) for x in (Q, H, B, gains))
    if Q.shape != (cfg.N,) or H.shape != (cfg.M,) or B.shape != (cfg.N, cfg.M) or gains.shape != B.shape:
        raise ValueError("Expected Q:(N,), H:(M,), B:(N,M), and gains:(N,M)")
    if any(not np.all(np.isfinite(x)) or np.any(x < 0) for x in (Q, H, B, gains)):
        raise ValueError("Queues and gains must be finite and nonnegative")
    if min(cfg.V, cfg.tau, cfg.kappa_device, cfg.kappa_edge,
           cfg.cycles_per_bit, cfg.edge_cycles_per_bit) <= 0:
        raise ValueError("V, slot length, capacitances, and cycles per bit must be positive")
    if min(cfg.local_f_max_hz, cfg.edge_f_max_hz) < 0:
        raise ValueError("CPU limits must be nonnegative")
    np.testing.assert_allclose(B.sum(axis=0), H, rtol=1e-10, atol=1e-4)

    # Exact common fixed-V CPU arithmetic, including its unclipped clock and
    # separate available-work service cap. These are not source GUPA formulas.
    local_denom = 3.0 * cfg.V * cfg.kappa_device * cfg.cycles_per_bit
    local_cpu = np.minimum(np.sqrt(np.maximum(Q, 0.0) / max(local_denom, 1.0e-30)), cfg.local_f_max_hz)
    local_bits = np.minimum(Q, cfg.tau * local_cpu / cfg.cycles_per_bit)
    residual = np.maximum(Q - local_bits, 0.0)
    association, powers, uploads, diagnostics = gupa_qoe_updates(residual, gains, cfg)
    edge_denom = 3.0 * cfg.V * cfg.kappa_edge * cfg.edge_cycles_per_bit
    edge_cpu = np.minimum(np.sqrt(np.maximum(1.0 * H, 0.0) / max(edge_denom, 1.0e-30)), cfg.edge_f_max_hz)
    edge_done = np.minimum(H, cfg.tau * edge_cpu / cfg.edge_cycles_per_bit)
    diagnostics.update({"effective_V": cfg.V, "aggregate_edge_service_bits": edge_done})
    return EEDODecision(local_cpu, powers, association, uploads, edge_cpu,
                        _by_origin_service(B, edge_done), diagnostics)

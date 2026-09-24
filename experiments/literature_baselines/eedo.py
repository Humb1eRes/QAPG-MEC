"""EEDO adaptation for the common multi-server MEC simulator.

Source: Chen and Jiang (2024), doi:10.1186/s13677-024-00645-5,
equations (25), (27), (29), and Algorithm 1. See EEDO_MAPPING.md for
the physical-model changes; this is not the original HAP experiment.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class EEDODecision:
    local_cpu_hz: np.ndarray
    transmit_power_w: np.ndarray
    association: np.ndarray
    uploads_bits: np.ndarray
    edge_cpu_hz: np.ndarray
    edge_service_bits: np.ndarray
    diagnostics: dict[str, Any]


def priority_edge_service(
    queues_bits: np.ndarray,
    cfg: Any,
) -> tuple[np.ndarray, float, float, int]:
    """Solve one server's cubic-cost service problem exactly, piecewise.

    For each total service amount s, allocating it by descending queue
    length maximizes sum_i B_i d_i. On each resulting linear segment of
    this reward, the objective is a scalar convex cubic. We project its
    stationary point onto the segment and compare all feasible segments.
    """
    queues = np.asarray(queues_bits, dtype=float)
    if queues.ndim != 1 or not np.all(np.isfinite(queues)) or np.any(queues < 0):
        raise ValueError("queues_bits must be a finite nonnegative vector")
    if cfg.V <= 0 or cfg.kappa_edge <= 0 or cfg.tau <= 0 or cfg.edge_cycles_per_bit <= 0:
        raise ValueError("V, edge capacitance, slot length, and cycles per bit must be positive")
    capacity = min(float(np.sum(queues)), cfg.tau * cfg.edge_f_max_hz / cfg.edge_cycles_per_bit)
    if capacity <= 0:
        return np.zeros_like(queues), 0.0, 0.0, 0

    cubic_cost = cfg.V * cfg.kappa_edge * cfg.edge_cycles_per_bit**3 / cfg.tau**2
    order = np.argsort(-queues, kind="stable")
    best_s, best_objective = 0.0, 0.0
    lower, preceding_reward, segments = 0.0, 0.0, 0
    for n in order:
        backlog = float(queues[n])
        if backlog <= 0 or lower >= capacity:
            break
        upper = min(capacity, lower + backlog)
        stationary = np.sqrt(backlog / (3.0 * cubic_cost))
        candidate = float(np.clip(stationary, lower, upper))
        objective = cubic_cost * candidate**3 - preceding_reward - backlog * (candidate - lower)
        segments += 1
        if objective < best_objective:
            best_s, best_objective = candidate, objective
        preceding_reward += backlog * (upper - lower)
        lower = upper

    allocation = np.zeros_like(queues)
    remaining = best_s
    for n in order:
        if remaining <= 0:
            break
        served = min(float(queues[n]), remaining)
        allocation[n] = served
        remaining -= served
    clock_hz = float(np.sum(allocation)) * cfg.edge_cycles_per_bit / cfg.tau
    return allocation, clock_hz, best_objective, segments


def eedo_decision(
    Q: np.ndarray,
    B: np.ndarray,
    gains: np.ndarray,
    cfg: Any,
) -> EEDODecision:
    """Choose controls from slot-start queues without mutating state.

    Q has shape (N,), while B and gains have shape (N,M). B[n,m] is
    device n's outstanding work at server m. Uploads are processed no
    earlier than the next slot. The caller applies the common full-slot
    energy ledger and updates Q and B only after all decisions are made.
    """
    Q = np.asarray(Q, dtype=float)
    B = np.asarray(B, dtype=float)
    gains = np.asarray(gains, dtype=float)
    if Q.shape != (cfg.N,) or B.shape != (cfg.N, cfg.M) or gains.shape != B.shape:
        raise ValueError("Expected Q:(N,), B:(N,M), and gains:(N,M)")
    if any(not np.all(np.isfinite(x)) or np.any(x < 0) for x in (Q, B, gains)):
        raise ValueError("Queues and gains must be finite and nonnegative")
    if min(cfg.V, cfg.tau, cfg.kappa_device, cfg.cycles_per_bit, cfg.noise_w) <= 0:
        raise ValueError("V and physical model constants must be positive")

    # Source Eq. (25), including its available-work CPU cap.
    local_clock = np.sqrt(Q / (3.0 * cfg.V * cfg.kappa_device * cfg.cycles_per_bit))
    local_clock = np.minimum(local_clock, np.minimum(cfg.local_f_max_hz, Q * cfg.cycles_per_bit / cfg.tau))
    local_bits = cfg.tau * local_clock / cfg.cycles_per_bit
    residual = np.maximum(Q - local_bits, 0.0)

    association = np.full(cfg.N, -1, dtype=int)
    power = np.zeros(cfg.N, dtype=float)
    uploads = np.zeros(cfg.N, dtype=float)
    fixed_power = float(np.clip(cfg.fixed_power_w, 0.0, cfg.p_max_w))
    link_candidates = 0
    source_accepted_candidates = 0
    source_accepted_common_rejected_candidates = 0
    if fixed_power > 0:
        rates = cfg.bandwidth_hz * np.log1p(fixed_power * gains / cfg.noise_w) / np.log(2.0)
        for n in range(cfg.N):
            if residual[n] <= 0:
                continue
            best_score = 0.0
            for m in range(cfg.M):
                rate = float(rates[n, m])
                link_candidates += 1
                if rate <= 0:
                    continue
                # Source Eq. (27) uses Q at slot start, not residual Q.
                coefficient = cfg.V * fixed_power / rate - Q[n] + B[n, m]
                if coefficient > 0:
                    continue
                amount = min(float(residual[n]), cfg.tau * rate)
                if amount <= 0:
                    continue
                source_accepted_candidates += 1
                # Eq. (27)'s p*u/R charge equals tau*p only for a full
                # slot of transmission. Correct its threshold when the
                # available-work cap truncates the uploaded amount.
                score = (Q[n] - B[n, m]) * amount - cfg.V * cfg.tau * fixed_power
                if score <= 0:
                    source_accepted_common_rejected_candidates += 1
                    continue
                if score > best_score:
                    best_score = score
                    association[n] = m
                    uploads[n] = amount
                    power[n] = fixed_power

    edge_service = np.zeros_like(B)
    edge_clock = np.zeros(cfg.M, dtype=float)
    edge_objective = np.zeros(cfg.M, dtype=float)
    edge_segments = 0
    for m in range(cfg.M):
        allocation, clock_hz, objective, segments = priority_edge_service(B[:, m], cfg)
        edge_service[:, m] = allocation
        edge_clock[m] = clock_hz
        edge_objective[m] = objective
        edge_segments += segments

    return EEDODecision(
        local_cpu_hz=local_clock,
        transmit_power_w=power,
        association=association,
        uploads_bits=uploads,
        edge_cpu_hz=edge_clock,
        edge_service_bits=edge_service,
        diagnostics={
            "source": "Chen and Jiang, Journal of Cloud Computing 13:92 (2024)",
            "adaptation": "EEDO-adapted",
            "link_candidates": link_candidates,
            "source_accepted_candidates": source_accepted_candidates,
            "source_accepted_common_rejected_candidates": source_accepted_common_rejected_candidates,
            "edge_segments": edge_segments,
            "edge_objective": edge_objective,
        },
    )

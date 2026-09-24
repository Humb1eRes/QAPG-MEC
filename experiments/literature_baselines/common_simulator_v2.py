"""Version 2: adds the source-verified orthogonal GUPA adaptation.

Common physical ledger for literature adaptations and unchanged controls.

The original aggregate H arithmetic is retained for legacy controls. A source
ledger B[n,m] is also maintained for every method; it adds no per-user quota.
"""
from __future__ import annotations

import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parent))
import matched_recorder as rec
from eedo import EEDODecision, eedo_decision
from gupa_orthogonal import gupa_orthogonal_decision

POLICIES = {
    "QAPG": ("EdgeSport-DPP", {}, False),
    "Adaptive-local-only": ("EdgeSport-DPP", {}, True),
    "Fixed-V-QAPG": ("EdgeSport-DPP", {"adaptive_v_gain": 0.0}, False),
    "No-quadratic-load": ("EdgeSport-DPP", {"server_load_penalty": 0.0}, False),
    "BP-Greedy": ("Least-loaded", {}, False),
    "Strongest-channel": ("LOC-SAC (2026)", {}, False),
}


def by_origin_service(B, aggregate):
    """Work-conserving accounting only: descending source backlog, stable ties."""
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


def legacy_decision(key, Q, H, B, gains, cfg, rng, disable):
    adaptive = key == "EdgeSport-DPP"
    V = rec.original.adaptive_control_v(Q, H, cfg) if adaptive else cfg.V
    f = rec.original.local_cpu(Q, cfg, V * cfg.local_v_multiplier if adaptive else None)
    local = np.minimum(Q, cfg.tau * f / cfg.cycles_per_bit)
    residual = np.maximum(Q - local, 0.0)
    if disable:
        association = np.full(cfg.N, -1, dtype=int)
        power, upload, iterations = np.zeros(cfg.N), np.zeros(cfg.N), 0
    else:
        association, power, upload, iterations = rec.original.choose_offloading(
            key, residual, H, gains, cfg, rng, V if adaptive else None
        )
    F = rec.original.edge_cpu(H, cfg, V if adaptive else None,
                             cfg.edge_cpu_pressure_gain if adaptive else 1.0)
    done = np.minimum(H, cfg.tau * F / cfg.edge_cycles_per_bit)
    return EEDODecision(f, power, association, upload, F, by_origin_service(B, done), {
        "effective_V": V, "iterations": iterations,
        "aggregate_edge_service_bits": done,
    })


def validate_decision(d, Q, H, B, gains, cfg):
    specs = {"local_cpu_hz": (cfg.N,), "transmit_power_w": (cfg.N,),
             "association": (cfg.N,), "uploads_bits": (cfg.N,),
             "edge_cpu_hz": (cfg.M,), "edge_service_bits": (cfg.N, cfg.M)}
    for name, shape in specs.items():
        x = np.asarray(getattr(d, name))
        if x.shape != shape or not np.all(np.isfinite(x)):
            raise AssertionError(f"Invalid {name}: expected finite {shape}")
        if name != "association" and np.any(x < -1e-8):
            raise AssertionError(f"Negative {name}")
    a, p, u = d.association, d.transmit_power_w, d.uploads_bits
    assert np.issubdtype(a.dtype, np.integer) and np.all((a >= -1) & (a < cfg.M))
    assert np.all(d.local_cpu_hz <= cfg.local_f_max_hz * (1 + 1e-12))
    assert np.all(d.edge_cpu_hz <= cfg.edge_f_max_hz * (1 + 1e-12))
    assert np.all(p <= cfg.p_max_w * (1 + 1e-12))
    assert np.all(p[a == -1] == 0) and np.all(u[a == -1] == 0)
    local = np.minimum(Q, cfg.tau * d.local_cpu_hz / cfg.cycles_per_bit)
    assert np.all(local + u <= Q + 1e-6)
    assert np.all(d.edge_service_bits <= B + 1e-6)
    edge = d.diagnostics.get("aggregate_edge_service_bits", d.edge_service_bits.sum(axis=0))
    np.testing.assert_allclose(d.edge_service_bits.sum(axis=0), edge, rtol=1e-10, atol=1e-5)
    assert np.all(edge <= H + 1e-5)
    assert np.all(edge <= cfg.tau * d.edge_cpu_hz / cfg.edge_cycles_per_bit + 1e-5)
    active = np.flatnonzero(a >= 0)
    rates = cfg.bandwidth_hz * np.log1p(p[active] * gains[active, a[active]] / cfg.noise_w) / np.log(2)
    assert np.all(u[active] <= cfg.tau * rates + 1e-6)
    return local, edge


def simulate(policy, cfg, arrivals, channels):
    if policy not in {"EEDO-adapted", "GUPA-O"} and policy not in POLICIES:
        raise ValueError(f"Unknown policy {policy}")
    if arrivals.shape != (cfg.T, cfg.N) or channels.shape != (cfg.T, cfg.N, cfg.M):
        raise ValueError("Input shapes do not match configuration")
    if not np.all(np.isfinite(arrivals)) or np.any(arrivals < 0):
        raise ValueError("Arrivals must be finite and nonnegative")
    if not np.all(np.isfinite(channels)) or np.any(channels < 0):
        raise ValueError("Channels must be finite and nonnegative")
    if policy in POLICIES:
        key, overrides, disable = POLICIES[policy]
        cfg = replace(cfg, **overrides)
        policy_seed = sum((i + 1) * ord(ch) for i, ch in enumerate(key))
        rng = np.random.default_rng(cfg.seed + policy_seed % 10000)
    Q, H, B = np.zeros(cfg.N), np.zeros(cfg.M), np.zeros((cfg.N, cfg.M))
    keys = ("total_energy", "device_energy", "edge_energy", "total_queue", "device_queue",
            "edge_queue", "fairness", "runtime_ms", "iterations") + rec.EXTRA_KEYS
    extras = ("effective_V", "local_cap_fraction", "unused_local_capacity_bits",
              "unused_edge_capacity_bits", "source_threshold_common_cost_rejections", "source_accepted_candidates",
              "origin_ledger_residual_bits", "edge_segments", "link_candidates",
              "qoe_total", "qoe_accepted_updates", "qoe_terminal_gain")
    result = {k: np.zeros(cfg.T) for k in keys + extras}
    result["edge_loads"] = np.zeros((cfg.T, cfg.M))
    arrived = completed = 0.0
    max_q_res = max_b_res = max_h_res = 0.0
    for t in range(cfg.T):
        start = time.perf_counter()
        Q.flags.writeable = H.flags.writeable = B.flags.writeable = False
        if policy == "GUPA-O":
            d = gupa_orthogonal_decision(Q, H, B, channels[t], cfg)
        elif policy == "EEDO-adapted":
            d = eedo_decision(Q, B, channels[t], cfg)
        else:
            d = legacy_decision(key, Q, H, B, channels[t], cfg, rng, disable)
        decision_ms = (time.perf_counter() - start) * 1000
        local, edge = validate_decision(d, Q, H, B, channels[t], cfg)
        upload_by_origin = np.zeros_like(B)
        server_uploads = np.zeros(cfg.M)
        for n, m in enumerate(d.association):
            if m >= 0:
                upload_by_origin[n, m] = d.uploads_bits[n]
                server_uploads[m] += d.uploads_bits[n]
        next_Q = np.maximum(np.maximum(Q - local, 0) - d.uploads_bits, 0) + arrivals[t]
        next_H = np.maximum(H - edge, 0) + server_uploads
        next_B = np.maximum(B - d.edge_service_bits, 0) + upload_by_origin
        q_res = float(np.max(np.abs(next_Q - (Q - local - d.uploads_bits + arrivals[t]))))
        b_res = float(np.max(np.abs(next_B - (B - d.edge_service_bits + upload_by_origin))))
        h_res = float(np.max(np.abs(next_H - next_B.sum(axis=0))))
        max_q_res, max_b_res, max_h_res = max(max_q_res, q_res), max(max_b_res, b_res), max(max_h_res, h_res)
        np.testing.assert_allclose(next_H, next_B.sum(axis=0), rtol=1e-10, atol=1e-4)
        np.testing.assert_allclose(q_res, 0, atol=1e-6, rtol=0)
        np.testing.assert_allclose(b_res, 0, atol=1e-6, rtol=0)
        Q, H, B = next_Q, next_H, next_B
        local_e = float(np.sum(cfg.kappa_device * cfg.tau * d.local_cpu_hz ** 3))
        tx_e = float(np.sum(d.transmit_power_w) * cfg.tau)
        edge_e = float(np.sum(cfg.kappa_edge * cfg.tau * d.edge_cpu_hz ** 3))
        values = {"total_energy": local_e + tx_e + edge_e, "device_energy": local_e + tx_e,
                  "edge_energy": edge_e, "device_queue": float(Q.sum()), "edge_queue": float(H.sum()),
                  "total_queue": float(Q.sum()) + float(H.sum()), "fairness": rec.original.jains_fairness(server_uploads),
                  "runtime_ms": decision_ms, "iterations": d.diagnostics.get("iterations", 0),
                  "arrival_bits": float(arrivals[t].sum()), "local_completed_bits": float(local.sum()),
                  "edge_completed_bits": float(edge.sum()), "uploaded_bits": float(d.uploads_bits.sum()),
                  "local_energy_j": local_e, "transmit_energy_j": tx_e,
                  "effective_V": d.diagnostics.get("effective_V", cfg.V),
                  "local_cap_fraction": float(np.mean(d.local_cpu_hz >= cfg.local_f_max_hz)),
                  "unused_local_capacity_bits": float((cfg.tau * d.local_cpu_hz / cfg.cycles_per_bit - local).sum()),
                  "unused_edge_capacity_bits": float((cfg.tau * d.edge_cpu_hz / cfg.edge_cycles_per_bit - edge).sum()),
                  "origin_ledger_residual_bits": h_res,
                  "source_threshold_common_cost_rejections": d.diagnostics.get("source_accepted_common_rejected_candidates", 0),
                  "source_accepted_candidates": d.diagnostics.get("source_accepted_candidates", 0),
                  "edge_segments": d.diagnostics.get("edge_segments", 0),
                  "link_candidates": d.diagnostics.get("link_candidates", 0),
                  "qoe_total": d.diagnostics.get("qoe_total", 0),
                  "qoe_accepted_updates": d.diagnostics.get("qoe_accepted_updates", 0),
                  "qoe_terminal_gain": d.diagnostics.get("qoe_terminal_gain", 0)}
        arrived += values["arrival_bits"]
        completed += values["local_completed_bits"] + values["edge_completed_bits"]
        values["flow_residual_bits"] = arrived - completed - values["total_queue"]
        np.testing.assert_allclose(arrived - completed, values["total_queue"], rtol=1e-10, atol=1e-5)
        for name, value in values.items():
            result[name][t] = value
        result["edge_loads"][t] = server_uploads
    checks = {"max_device_flow_residual_bits": max_q_res, "max_origin_flow_residual_bits": max_b_res,
              "max_origin_aggregate_discrepancy_bits": max_h_res,
              "max_cumulative_system_flow_residual_bits": float(np.abs(result["flow_residual_bits"]).max()),
              "conservation_and_resource_assertions_passed": True,
              "final_device_queues_bits": Q.tolist(), "final_origin_queues_bits": B.tolist()}
    return result, checks

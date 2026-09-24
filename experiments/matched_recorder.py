"""Matched-control recorder derived from the R11 audited synthetic simulator.

The original internal policy keys and policy-dependent RNG are preserved.
Public names describe implemented heuristics. No SportsMOT or SAC is claimed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import platform
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
BASE = Path(__file__).resolve().parent
ORIGINAL_PATH = BASE / "reference" / "edgesport_sim.py"
spec = importlib.util.spec_from_file_location("edgesport_original_recorded", ORIGINAL_PATH)
original = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = original
spec.loader.exec_module(original)

POLICY_NAMES = {
    "EdgeSport-DPP": "EdgeSport",
    "DPQ-EEDOA (2026)": "No-edge-pressure",
    "Fixed-power": "Fixed-power heuristic",
    "Least-loaded": "Queue-score heuristic",
    "LOC-SAC (2026)": "Strongest-channel",
    "Random-offload": "Random-offload",
    "Local-only": "Local-only",
}
POLICY_EVIDENCE = {
    "EdgeSport-DPP": "Original lines 200-270: adaptive-V potential-game heuristic.",
    "DPQ-EEDOA (2026)": "Original lines 297-317: enumerate servers; water-fill with device queue only; no fixed association.",
    "Fixed-power": "Original lines 303-317: queue-score selection with fixed power; no potential-game load term.",
    "Least-loaded": "Original lines 283-317: sort by H, then evaluate every server and maximize score.",
    "LOC-SAC (2026)": "Original lines 276-282: argmax channel gain and p_max; no learning, actor, critic or checkpoint.",
    "Random-offload": "Original lines 287-318: random server order; take first positive-power feasible candidate.",
    "Local-only": "Original lines 273-274: no offloading; original shared local CPU controller.",
}
EXTRA_KEYS = (
    "arrival_bits", "local_completed_bits", "edge_completed_bits", "uploaded_bits",
    "local_energy_j", "transmit_energy_j", "flow_residual_bits",
)


def simulate_recorded(policy, cfg, arrivals, channels, *, disable_offloading=False):
    """Same decision/update arithmetic as original.simulate, with extra records."""
    policy_seed = sum((idx + 1) * ord(ch) for idx, ch in enumerate(policy))
    rng = np.random.default_rng(cfg.seed + policy_seed % 10000)
    Q, H = np.zeros(cfg.N, dtype=float), np.zeros(cfg.M, dtype=float)
    keys = ("total_energy", "device_energy", "edge_energy", "total_queue", "device_queue", "edge_queue", "fairness", "runtime_ms", "iterations")
    result = {key: np.zeros(cfg.T) for key in keys + EXTRA_KEYS}
    result["edge_loads"] = np.zeros((cfg.T, cfg.M))
    result["effective_V"] = np.zeros(cfg.T)
    result["local_cap_fraction"] = np.zeros(cfg.T)
    cumulative_arrivals = cumulative_completed = 0.0
    maximum_local_residual = maximum_edge_residual = 0.0
    for t in range(cfg.T):
        Q_before, H_before = Q.copy(), H.copy()
        start = time.perf_counter()
        V_eff = original.adaptive_control_v(Q, H, cfg) if policy == "EdgeSport-DPP" else cfg.V
        local_v = V_eff * cfg.local_v_multiplier if policy == "EdgeSport-DPP" else None
        f = original.local_cpu(Q, cfg, local_v)
        local_bits = np.minimum(Q, cfg.tau * f / cfg.cycles_per_bit)
        Q_after_local = np.maximum(Q - local_bits, 0.0)
        if disable_offloading:
            assoc = np.full(cfg.N, -1, dtype=int)
            powers = np.zeros(cfg.N, dtype=float)
            uploads = np.zeros(cfg.N, dtype=float)
            iters = 0
        else:
            assoc, powers, uploads, iters = original.choose_offloading(
                policy, Q_after_local, H, channels[t], cfg, rng, V_eff if policy == "EdgeSport-DPP" else None
            )
        F = original.edge_cpu(H, cfg, V_eff if policy == "EdgeSport-DPP" else None, cfg.edge_cpu_pressure_gain if policy == "EdgeSport-DPP" else 1.0)
        edge_done = np.minimum(H, cfg.tau * F / cfg.edge_cycles_per_bit)
        decision_ms = (time.perf_counter() - start) * 1000.0
        server_uploads = np.zeros(cfg.M, dtype=float)
        for n, m in enumerate(assoc):
            if m >= 0:
                server_uploads[m] += uploads[n]
        Q = np.maximum(Q_after_local - uploads, 0.0) + arrivals[t]
        H = np.maximum(H - edge_done, 0.0) + server_uploads
        dev_e = float(np.sum(cfg.kappa_device * cfg.tau * f ** 3) + np.sum(powers) * cfg.tau)
        edge_e = float(np.sum(cfg.kappa_edge * cfg.tau * F ** 3))
        result["total_energy"][t] = dev_e + edge_e
        result["device_energy"][t] = dev_e
        result["edge_energy"][t] = edge_e
        result["device_queue"][t] = float(np.sum(Q))
        result["edge_queue"][t] = float(np.sum(H))
        result["total_queue"][t] = result["device_queue"][t] + result["edge_queue"][t]
        result["fairness"][t] = original.jains_fairness(server_uploads)
        result["runtime_ms"][t] = decision_ms
        result["iterations"][t] = iters
        result["effective_V"][t] = V_eff
        result["local_cap_fraction"][t] = float(np.mean(f >= cfg.local_f_max_hz))
        result["edge_loads"][t] = server_uploads
        result["arrival_bits"][t] = float(np.sum(arrivals[t]))
        result["local_completed_bits"][t] = float(np.sum(local_bits))
        result["edge_completed_bits"][t] = float(np.sum(edge_done))
        result["uploaded_bits"][t] = float(np.sum(uploads))
        result["local_energy_j"][t] = float(np.sum(cfg.kappa_device * cfg.tau * f ** 3))
        result["transmit_energy_j"][t] = float(np.sum(powers) * cfg.tau)
        cumulative_arrivals += result["arrival_bits"][t]
        cumulative_completed += result["local_completed_bits"][t] + result["edge_completed_bits"][t]
        result["flow_residual_bits"][t] = cumulative_arrivals - cumulative_completed - result["total_queue"][t]
        local_residual = Q - (Q_before - local_bits - uploads + arrivals[t])
        edge_residual = H - (H_before - edge_done + server_uploads)
        maximum_local_residual = max(maximum_local_residual, float(np.max(np.abs(local_residual))))
        maximum_edge_residual = max(maximum_edge_residual, float(np.max(np.abs(edge_residual))))
        assert np.all(Q >= 0) and np.all(H >= 0)
        np.testing.assert_allclose(local_residual, 0.0, rtol=0, atol=1e-6)
        np.testing.assert_allclose(edge_residual, 0.0, rtol=0, atol=1e-6)
        np.testing.assert_allclose(np.sum(uploads), np.sum(server_uploads), rtol=1e-12, atol=1e-6)
        np.testing.assert_allclose(cumulative_arrivals - cumulative_completed, result["total_queue"][t], rtol=1e-10, atol=1e-6)
        np.testing.assert_allclose(result["local_energy_j"][t] + result["transmit_energy_j"][t] + edge_e, result["total_energy"][t], rtol=1e-12, atol=1e-12)
    checks = {"max_device_flow_residual_bits": maximum_local_residual, "max_edge_flow_residual_bits": maximum_edge_residual, "max_cumulative_system_flow_residual_bits": float(np.max(np.abs(result["flow_residual_bits"]))), "conservation_assertions_passed": True}
    return result, checks


def summary(result, warmup):
    if not 0 <= warmup < len(result["total_energy"]) // 2:
        raise ValueError("Warm-up must be fixed in advance and smaller than half the run")
    cut = warmup
    halfway = len(result["total_energy"]) // 2
    sl = slice(cut, None)
    row = original.summarize(result, warmup=warmup)
    for name in EXTRA_KEYS[:-1]:
        row["avg_" + name] = float(np.mean(result[name][sl]))
    complete = float(np.sum(result["local_completed_bits"][sl] + result["edge_completed_bits"][sl]))
    arrived = float(np.sum(result["arrival_bits"][sl]))
    energy = float(np.sum(result["total_energy"][sl]))
    row.update({"effective_warmup_slots": cut, "window_arrival_bits": arrived, "window_completed_bits": complete, "window_energy_j": energy, "energy_per_completed_mbit_j": energy / complete * 1e6 if complete else float("nan"), "window_start_queue_bits": float(result["total_queue"][cut-1]) if cut else 0.0, "window_end_queue_bits": float(result["total_queue"][-1])})
    row["window_flow_residual_bits"] = row["window_start_queue_bits"] + arrived - complete - row["window_end_queue_bits"]
    window_local = float(np.sum(result["local_completed_bits"][sl]))
    window_edge = float(np.sum(result["edge_completed_bits"][sl]))
    late_backlog = result["total_queue"][halfway:]
    late_x = np.arange(halfway, len(result["total_queue"]), dtype=float)
    centered_x = late_x - float(np.mean(late_x))
    denominator = float(np.sum(centered_x ** 2))
    late_slope = float(np.sum(centered_x * (late_backlog-float(np.mean(late_backlog)))) / denominator) if denominator else float("nan")
    row.update({
        "full_run_energy_j": float(np.sum(result["total_energy"])),
        "full_run_completed_mbit": float(np.sum(result["local_completed_bits"]+result["edge_completed_bits"])) / 1e6,
        "window_completed_mbit": complete / 1e6,
        "window_local_completed_bits": window_local,
        "window_edge_completed_bits": window_edge,
        "window_local_completion_fraction": window_local/complete if complete else float("nan"),
        "window_edge_completion_fraction": window_edge/complete if complete else float("nan"),
        "early_half_avg_total_queue_bits": float(np.mean(result["total_queue"][cut:halfway])) if cut < halfway else float("nan"),
        "late_half_avg_total_queue_bits": float(np.mean(late_backlog)),
        "late_half_backlog_slope_bits_per_slot": late_slope,
        "tail_10pct_avg_total_queue_bits": float(np.mean(result["total_queue"][int(len(result["total_queue"])*0.9):])),
    })
    np.testing.assert_allclose(row["window_start_queue_bits"] + arrived - complete, row["window_end_queue_bits"], rtol=1e-10, atol=1e-6)
    return row


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


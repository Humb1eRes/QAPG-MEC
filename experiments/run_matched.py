"""Reproducible E1/E4 mechanism ablations for the Scientific Reports revision.

The controller implementation is inherited from R11; this experiment adds
matched controls and a prespecified load protocol. No literature baseline is
claimed. Outputs are written to a new directory and never overwrite a run.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
import matched_recorder as rec

BASE = Path(__file__).resolve().parent
INTERNAL_POLICY = "EdgeSport-DPP"
VARIANTS = {
    "QAPG": ({}, False),
    "Adaptive-local-only": ({}, True),
    "Fixed-V-QAPG": ({"adaptive_v_gain": 0.0}, False),
    "No-quadratic-load": ({"server_load_penalty": 0.0}, False),
    "No-association-edge-pressure": ({"offload_edge_backlog_weight": 0.0}, False),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha(array):
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode())
    digest.update(array.dtype.str.encode())
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def dump(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def metrics(result, warmup):
    row = rec.summary(result, warmup)
    sl = slice(warmup, None)
    row["mean_energy_j_per_slot"] = float(np.mean(result["total_energy"][sl]))
    row["mean_backlog_mbit"] = float(np.mean(result["total_queue"][sl])) / 1e6
    row["mean_device_backlog_mbit"] = float(np.mean(result["device_queue"][sl])) / 1e6
    row["mean_edge_backlog_mbit"] = float(np.mean(result["edge_queue"][sl])) / 1e6
    row["mean_local_completed_mbit_per_slot"] = float(np.mean(result["local_completed_bits"][sl])) / 1e6
    row["mean_edge_completed_mbit_per_slot"] = float(np.mean(result["edge_completed_bits"][sl])) / 1e6
    row["mean_arrival_mbit_per_slot"] = float(np.mean(result["arrival_bits"][sl])) / 1e6
    row["completion_to_arrival_ratio"] = row["window_completed_bits"] / row["window_arrival_bits"]
    row["late_half_backlog_slope_mbit_per_slot"] = row["late_half_backlog_slope_bits_per_slot"] / 1e6
    row["local_cap_fraction"] = float(np.mean(result["local_cap_fraction"][sl]))
    row["mean_effective_V"] = float(np.mean(result["effective_V"][sl]))
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--T", type=int, default=3000)
    p.add_argument("--warmup", type=int, default=300)
    p.add_argument("--seeds", type=int, nargs="+", default=[7, 108, 209, 310, 411])
    p.add_argument("--betas", type=float, nargs="+", default=[1, 2, 3])
    p.add_argument("--output", type=Path, default=BASE / "results" / "matched_e1_e4")
    args = p.parse_args()
    if not 0 <= args.warmup < args.T // 2:
        raise ValueError("Prespecified warm-up must be below half the horizon")
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.betas)) != len(args.betas):
        raise ValueError("Repeated seeds or loads are forbidden")
    if args.T <= 0 or min(args.betas) < 0:
        raise ValueError("Invalid horizon or load")
    out = args.output.resolve()
    if not out.is_relative_to(BASE / "results"):
        raise ValueError("Output must stay below experiments/results")
    out.mkdir(parents=True, exist_ok=False)
    (out / "inputs").mkdir()
    (out / "traces").mkdir()
    started = time.perf_counter()
    cfg = rec.original.SimConfig(T=args.T)
    metadata = {
        "status": "RUNNING",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "New E1/E4 matched mechanism and load experiment; distinct from prior R11 experiments",
        "planned_T": args.T, "warmup_slots": args.warmup,
        "seeds": args.seeds, "loads_beta": args.betas,
        "default_config": asdict(cfg),
        "variant_overrides_and_disable_offloading": VARIANTS,
        "internal_policy_key_for_every_variant": INTERNAL_POLICY,
        "rng_rule": "seed + sum((i+1)*ord(c) for i,c in enumerate('EdgeSport-DPP')) % 10000; unchanged by public label",
        "arrival_rng_rule": "seed + 13", "channel_rng_rule": "seed + 29",
        "reference_sha256": sha(rec.ORIGINAL_PATH),
        "recorder_sha256": sha(BASE / "matched_recorder.py"),
        "runner_sha256": sha(__file__),
        "python": sys.version, "numpy": np.__version__, "platform": platform.platform(),
        "load_scale_rule": "N * arrival_lambda * task_packet_bits; independent of beta",
        "execution": "Sequential in one process; decision timings are local diagnostics only",
        "statistics": "Independent replicate is a seed. Means/SD and paired differences across seeds; slots are not independent replicates.",
        "finite_horizon_note": "Backlog slope is a finite-window diagnostic, not a proof of stochastic stability.",
        "energy_boundary": "Inherited dynamic CPU plus transmit energy; excludes idle power, network infrastructure, and controller execution energy.",
        "wireless_boundary": "Inherited independent orthogonal links, not a finite shared system bandwidth budget.",
    }
    dump(out / "metadata.json", metadata)
    rows, checks, input_manifest = [], [], []
    for seed in args.seeds:
        seed_cfg = replace(cfg, seed=seed)
        channels = rec.original.generate_channels(seed_cfg)
        channel_path = out / "inputs" / f"channels_seed{seed}.npz"
        np.savez_compressed(channel_path, channels=channels)
        channels.flags.writeable = False
        for beta in args.betas:
            base_cfg = replace(seed_cfg, beta=beta)
            arrivals, _ = rec.original.make_arrivals(base_cfg)
            arrival_path = out / "inputs" / f"arrivals_beta{beta:g}_seed{seed}.npz"
            np.savez_compressed(arrival_path, arrivals=arrivals)
            arrivals.flags.writeable = False
            input_row = {
                "seed": seed, "beta": beta,
                "arrivals_array_sha256": array_sha(arrivals),
                "channels_array_sha256": array_sha(channels),
                "arrivals_file": str(arrival_path.relative_to(out)),
                "channels_file": str(channel_path.relative_to(out)),
            }
            input_manifest.append(input_row)
            for variant, (overrides, disable) in VARIANTS.items():
                variant_cfg = replace(base_cfg, **overrides)
                wall = time.perf_counter()
                result, flow = rec.simulate_recorded(INTERNAL_POLICY, variant_cfg, arrivals, channels, disable_offloading=disable)
                duration = time.perf_counter() - wall
                if disable:
                    for key in ["uploaded_bits", "edge_completed_bits", "edge_queue", "transmit_energy_j", "edge_energy"]:
                        assert np.all(result[key] == 0), key
                if variant == "Fixed-V-QAPG":
                    assert np.all(result["effective_V"] == variant_cfg.V)
                trace_path = out / "traces" / f"beta{beta:g}_seed{seed}_{variant}.npz"
                np.savez_compressed(trace_path, **result)
                identity = {"beta": beta, "seed": seed, "policy": variant}
                row = dict(identity, **metrics(result, args.warmup))
                row["wall_seconds"] = duration
                row["trace_file"] = str(trace_path.relative_to(out))
                rows.append(row)
                checks.append(dict(identity, **flow, config=asdict(variant_cfg), disable_offloading=disable,
                                   input_hashes=input_row, trace_sha256=sha(trace_path)))
                rec.write_csv(out / "per_seed.csv", rows)
                dump(out / "flow_and_config.json", checks)
                dump(out / "input_manifest.json", input_manifest)
                print(json.dumps({"completed": len(rows), "total": len(args.seeds)*len(args.betas)*len(VARIANTS),
                                  **identity, "seconds": round(duration, 3),
                                  "energy": row["mean_energy_j_per_slot"], "backlog_mbit": row["mean_backlog_mbit"]}), flush=True)
    metadata.update(status="COMPLETED", finished_utc=datetime.now(timezone.utc).isoformat(),
                    duration_seconds=time.perf_counter()-started, policy_seed_load_runs=len(rows))
    dump(out / "metadata.json", metadata)
    print(json.dumps({"status": "COMPLETED", "duration_seconds": metadata["duration_seconds"]}), flush=True)


if __name__ == "__main__":
    main()

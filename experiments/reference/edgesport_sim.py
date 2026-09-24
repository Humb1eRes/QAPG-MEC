from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figs"
RAW_DIR = ROOT / "results" / "raw"
SUMMARY_DIR = ROOT / "results" / "summary"

CHART_FONT = "Helvetica"
CHART_FONT_BOLD = "Helvetica-Bold"

def chart_text(text: str) -> str:
    return text


@dataclass(frozen=True)
class SimConfig:
    T: int = 300
    N: int = 100
    M: int = 10
    V: float = 3.0e11
    beta: float = 1.0
    seed: int = 7
    tau: float = 0.1
    bandwidth_hz: float = 2.0e6
    noise_w: float = 1.0e-10
    p_max_w: float = 0.2
    fixed_power_w: float = 0.08
    local_f_max_hz: float = 1.0e9
    edge_f_max_hz: float = 18.0e9
    cycles_per_bit: float = 1000.0
    edge_cycles_per_bit: float = 1000.0
    kappa_device: float = 1.0e-27
    kappa_edge: float = 5.0e-30
    arrival_lambda: float = 6.0
    task_packet_bits: float = 8.0e3
    game_max_sweeps: int = 12
    game_tolerance: float = 1.0e-6
    plot_smoothing_window: int = 9
    adaptive_v_min: float = 8.0e10
    adaptive_v_gain: float = 2.0
    adaptive_edge_queue_weight: float = 0.25
    offload_edge_backlog_weight: float = 0.05
    offload_min_pressure_fraction: float = 0.20
    server_load_penalty: float = 1.0
    edge_cpu_pressure_gain: float = 1.25
    local_v_multiplier: float = 1.0


POLICIES = [
    "EdgeSport-DPP",
    "DPQ-EEDOA (2026)",
    "Fixed-power",
    "Least-loaded",
    "LOC-SAC (2026)",
    "Random-offload",
    "Local-only",
]

PALETTE = [
    colors.HexColor("#1f77b4"),
    colors.HexColor("#d62728"),
    colors.HexColor("#2ca02c"),
    colors.HexColor("#9467bd"),
    colors.HexColor("#ff7f0e"),
    colors.HexColor("#4d4d4d"),
    colors.HexColor("#8c564b"),
]

EVAL_SEED_OFFSETS = (0, 101, 202)


def ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)


def jains_fairness(values: np.ndarray) -> float:
    total = float(np.sum(values))
    if total <= 0:
        return 1.0
    denom = float(len(values) * np.sum(values * values))
    return (total * total) / denom if denom > 0 else 1.0


def generate_poisson_workload(cfg: SimConfig) -> np.ndarray:
    """Generate independent Poisson task-packet arrivals.

    The default trace is deliberately simple: each monitoring device receives
    Z_n(t) task packets in each slot, where Z_n(t) follows a Poisson
    distribution with mean beta * arrival_lambda. Each packet contains
    task_packet_bits bits. This makes the arrival-rate sweeps comparable and
    avoids mixing traffic-model changes with algorithmic effects.
    """
    rng = np.random.default_rng(cfg.seed + 13)
    lam = max(0.0, cfg.beta * cfg.arrival_lambda)
    packets = rng.poisson(lam=lam, size=(cfg.T, cfg.N))
    return cfg.task_packet_bits * packets.astype(float)


def make_arrivals(cfg: SimConfig) -> Tuple[np.ndarray, Dict[str, str]]:
    return generate_poisson_workload(cfg), {
        "arrival_source": "simulated_poisson",
        "mapping": "A_n(t)=task_packet_bits*Z_n(t), Z_n(t)~Poisson(beta*arrival_lambda)",
    }


def generate_channels(cfg: SimConfig) -> np.ndarray:
    rng = np.random.default_rng(cfg.seed + 29)
    width, height = 100.0, 60.0
    base_pos = np.column_stack([rng.uniform(0, width, cfg.N), rng.uniform(0, height, cfg.N)])
    server_angles = np.linspace(0, 2 * np.pi, cfg.M, endpoint=False)
    server_pos = np.column_stack(
        [
            width / 2 + 58.0 * np.cos(server_angles),
            height / 2 + 38.0 * np.sin(server_angles),
        ]
    )
    channels = np.empty((cfg.T, cfg.N, cfg.M), dtype=float)
    velocity = rng.normal(0.0, 1.2, size=(cfg.N, 2))
    for t in range(cfg.T):
        drift = 3.0 * np.column_stack(
            [np.sin(2 * np.pi * (t / 65.0 + np.arange(cfg.N) / max(1, cfg.N))),
             np.cos(2 * np.pi * (t / 83.0 + np.arange(cfg.N) / max(1, cfg.N)))]
        )
        pos = base_pos + cfg.tau * t * velocity + drift
        pos[:, 0] = np.mod(pos[:, 0], width)
        pos[:, 1] = np.mod(pos[:, 1], height)
        dist = np.linalg.norm(pos[:, None, :] - server_pos[None, :, :], axis=2)
        path_loss = 1.8e-6 / np.power(dist + 6.0, 2.15)
        fading = rng.exponential(scale=1.0, size=(cfg.N, cfg.M))
        channels[t] = np.clip(path_loss * fading, 1.0e-12, None)
    return channels


def local_cpu(Q: np.ndarray, cfg: SimConfig, V_eff: Optional[float] = None) -> np.ndarray:
    control_v = cfg.V if V_eff is None else V_eff
    denom = 3.0 * control_v * cfg.kappa_device * cfg.cycles_per_bit
    f = np.sqrt(np.maximum(Q, 0.0) / max(denom, 1.0e-30))
    return np.minimum(f, cfg.local_f_max_hz)


def edge_cpu(H: np.ndarray, cfg: SimConfig, V_eff: Optional[float] = None, pressure_gain: float = 1.0) -> np.ndarray:
    control_v = cfg.V if V_eff is None else V_eff
    denom = 3.0 * control_v * cfg.kappa_edge * cfg.edge_cycles_per_bit
    F = np.sqrt(np.maximum(pressure_gain * H, 0.0) / max(denom, 1.0e-30))
    return np.minimum(F, cfg.edge_f_max_hz)


def waterfill_power(weight: float, gain: float, cfg: SimConfig, V_eff: Optional[float] = None) -> float:
    if weight <= 0.0 or gain <= 0.0:
        return 0.0
    control_v = cfg.V if V_eff is None else V_eff
    p = weight * cfg.bandwidth_hz / (control_v * math.log(2.0)) - cfg.noise_w / gain
    return float(min(cfg.p_max_w, max(0.0, p)))


def uplink_rate(power: float, gain: float, cfg: SimConfig) -> float:
    if power <= 0.0:
        return 0.0
    snr = power * gain / cfg.noise_w
    return cfg.bandwidth_hz * math.log2(1.0 + max(snr, 0.0))


def adaptive_control_v(Q: np.ndarray, H: np.ndarray, cfg: SimConfig) -> float:
    """Reduce the energy penalty when queues approach the offered-load scale."""
    load_scale = max(cfg.N * cfg.arrival_lambda * cfg.task_packet_bits, 1.0)
    pressure = (float(np.sum(Q)) + cfg.adaptive_edge_queue_weight * float(np.sum(H))) / load_scale
    return max(cfg.adaptive_v_min, cfg.V / (1.0 + cfg.adaptive_v_gain * pressure))


def choose_offloading(
    policy: str,
    Q_after_local: np.ndarray,
    H: np.ndarray,
    gains: np.ndarray,
    cfg: SimConfig,
    rng: np.random.Generator,
    V_eff: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    N, M = Q_after_local.shape[0], H.shape[0]
    assoc = np.full(N, -1, dtype=int)
    powers = np.zeros(N, dtype=float)
    uploads = np.zeros(N, dtype=float)
    iterations = 0

    if policy == "EdgeSport-DPP":
        control_v = cfg.V if V_eff is None else V_eff
        edge_pressure = cfg.offload_edge_backlog_weight * H
        cand_power = np.zeros((N, M), dtype=float)
        cand_upload = np.zeros((N, M), dtype=float)
        for n in range(N):
            remaining = float(Q_after_local[n])
            if remaining <= 0.0:
                continue
            for m in range(M):
                queue_pressure = float(Q_after_local[n])
                min_pressure = cfg.offload_min_pressure_fraction * queue_pressure
                weight = float(max(queue_pressure - edge_pressure[m], min_pressure))
                p = waterfill_power(weight, float(gains[n, m]), cfg, control_v)
                if p <= 0.0:
                    continue
                r = uplink_rate(p, float(gains[n, m]), cfg)
                cand_power[n, m] = p
                cand_upload[n, m] = min(remaining, cfg.tau * r)

        server_loads = np.zeros(M, dtype=float)

        def utility(n: int, m: int, loads_without_n: np.ndarray) -> float:
            if m < 0:
                return 0.0
            u = float(cand_upload[n, m])
            p = float(cand_power[n, m])
            if u <= 0.0 or p <= 0.0:
                return -math.inf
            queue_gain = float(Q_after_local[n] * u)
            energy_cost = float(control_v * cfg.tau * p)
            edge_linear_cost = float(edge_pressure[m] * u)
            edge_quadratic_cost = 0.5 * cfg.server_load_penalty * float((loads_without_n[m] + u) ** 2 - loads_without_n[m] ** 2)
            return queue_gain - energy_cost - edge_linear_cost - edge_quadratic_cost

        for _ in range(cfg.game_max_sweeps):
            changed = False
            for n in range(N):
                old_m = int(assoc[n])
                loads_without_n = server_loads.copy()
                if old_m >= 0:
                    loads_without_n[old_m] -= uploads[n]
                current_utility = utility(n, old_m, loads_without_n)
                best_m = -1
                best_utility = 0.0
                for m in range(M):
                    iterations += 1
                    value = utility(n, m, loads_without_n)
                    if value > best_utility:
                        best_utility = value
                        best_m = m
                if best_utility > current_utility + cfg.game_tolerance and best_m != old_m:
                    if old_m >= 0:
                        server_loads[old_m] -= uploads[n]
                    assoc[n] = best_m
                    if best_m >= 0:
                        powers[n] = cand_power[n, best_m]
                        uploads[n] = cand_upload[n, best_m]
                        server_loads[best_m] += uploads[n]
                    else:
                        powers[n] = 0.0
                        uploads[n] = 0.0
                    changed = True
                else:
                    if old_m >= 0:
                        server_loads[old_m] = loads_without_n[old_m] + uploads[n]
            if not changed:
                break
        return assoc, powers, uploads, iterations

    for n in range(N):
        remaining = float(Q_after_local[n])
        if remaining <= 0.0 or policy == "Local-only":
            continue

        if policy == "LOC-SAC (2026)":
            m = int(np.argmax(gains[n]))
            p = cfg.p_max_w
            r = uplink_rate(p, float(gains[n, m]), cfg)
            assoc[n], powers[n], uploads[n] = m, p, min(remaining, cfg.tau * r)
            iterations += M
            continue

        if policy == "Least-loaded":
            server_order = np.argsort(H)
        elif policy == "Random-offload":
            server_order = rng.permutation(M)
        else:
            server_order = np.arange(M)

        best_score = 0.0
        best_m = -1
        best_p = 0.0
        best_u = 0.0
        for m in server_order:
            iterations += 1
            if policy == "DPQ-EEDOA (2026)":
                weight = float(max(Q_after_local[n], 0.0))
                p = waterfill_power(weight, float(gains[n, m]), cfg)
            elif policy == "Fixed-power":
                weight = float(max(Q_after_local[n] - H[m], 0.0))
                p = cfg.fixed_power_w if weight > 0.0 else 0.0
            elif policy == "Random-offload":
                weight = float(max(Q_after_local[n] - H[m], 0.0))
                p = waterfill_power(weight, float(gains[n, m]), cfg)
            else:
                weight = float(max(Q_after_local[n] - H[m], 0.0))
                p = waterfill_power(weight, float(gains[n, m]), cfg)
            if p <= 0.0:
                continue
            r = uplink_rate(p, float(gains[n, m]), cfg)
            u = min(remaining, cfg.tau * r)
            score = weight * u - cfg.V * cfg.tau * p
            if score > best_score:
                best_score, best_m, best_p, best_u = score, int(m), p, u
            if policy == "Random-offload":
                break

        if best_m >= 0:
            assoc[n], powers[n], uploads[n] = best_m, best_p, best_u

    return assoc, powers, uploads, iterations


def simulate(
    policy: str,
    cfg: SimConfig,
    arrivals: np.ndarray,
    channels: np.ndarray,
) -> Dict[str, np.ndarray]:
    policy_seed = sum((idx + 1) * ord(ch) for idx, ch in enumerate(policy))
    rng = np.random.default_rng(cfg.seed + policy_seed % 10000)
    Q = np.zeros(cfg.N, dtype=float)
    H = np.zeros(cfg.M, dtype=float)

    total_energy = np.zeros(cfg.T)
    device_energy = np.zeros(cfg.T)
    edge_energy = np.zeros(cfg.T)
    total_queue = np.zeros(cfg.T)
    device_queue = np.zeros(cfg.T)
    edge_queue = np.zeros(cfg.T)
    fairness = np.zeros(cfg.T)
    runtime_ms = np.zeros(cfg.T)
    iterations = np.zeros(cfg.T)
    edge_loads = np.zeros((cfg.T, cfg.M))

    for t in range(cfg.T):
        start = time.perf_counter()
        V_eff = adaptive_control_v(Q, H, cfg) if policy == "EdgeSport-DPP" else cfg.V
        local_v = V_eff * cfg.local_v_multiplier if policy == "EdgeSport-DPP" else None
        f = local_cpu(Q, cfg, local_v)
        local_bits = np.minimum(Q, cfg.tau * f / cfg.cycles_per_bit)
        Q_after_local = np.maximum(Q - local_bits, 0.0)
        assoc, powers, uploads, iters = choose_offloading(
            policy, Q_after_local, H, channels[t], cfg, rng, V_eff if policy == "EdgeSport-DPP" else None
        )
        F = edge_cpu(H, cfg, V_eff if policy == "EdgeSport-DPP" else None, cfg.edge_cpu_pressure_gain if policy == "EdgeSport-DPP" else 1.0)
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
        total_energy[t] = dev_e + edge_e
        device_energy[t] = dev_e
        edge_energy[t] = edge_e
        device_queue[t] = float(np.sum(Q))
        edge_queue[t] = float(np.sum(H))
        total_queue[t] = device_queue[t] + edge_queue[t]
        fairness[t] = jains_fairness(server_uploads)
        runtime_ms[t] = decision_ms
        iterations[t] = iters
        edge_loads[t] = server_uploads

    return {
        "total_energy": total_energy,
        "device_energy": device_energy,
        "edge_energy": edge_energy,
        "total_queue": total_queue,
        "device_queue": device_queue,
        "edge_queue": edge_queue,
        "fairness": fairness,
        "runtime_ms": runtime_ms,
        "iterations": iterations,
        "edge_loads": edge_loads,
    }


def summarize(result: Dict[str, np.ndarray], warmup: int = 50) -> Dict[str, float]:
    sl = slice(min(warmup, len(result["total_energy"]) // 2), None)
    return {
        "avg_energy_j": float(np.mean(result["total_energy"][sl])),
        "avg_device_energy_j": float(np.mean(result["device_energy"][sl])),
        "avg_edge_energy_j": float(np.mean(result["edge_energy"][sl])),
        "avg_total_queue_bits": float(np.mean(result["total_queue"][sl])),
        "avg_device_queue_bits": float(np.mean(result["device_queue"][sl])),
        "avg_edge_queue_bits": float(np.mean(result["edge_queue"][sl])),
        "avg_fairness": float(np.mean(result["fairness"][sl])),
        "avg_runtime_ms": float(np.mean(result["runtime_ms"][sl])),
        "avg_iterations": float(np.mean(result["iterations"][sl])),
    }


def evaluation_seeds(cfg: SimConfig) -> List[int]:
    return [cfg.seed + offset for offset in EVAL_SEED_OFFSETS]


def mean_summary(rows: Sequence[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        return {}
    keys = rows[0].keys()
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def evaluate_policy_mean(
    policy: str,
    cfg: SimConfig,
    seeds: Sequence[int],
) -> Dict[str, float]:
    summaries = []
    for seed in seeds:
        scfg = replace(cfg, seed=seed)
        arrivals, _ = make_arrivals(scfg)
        channels = generate_channels(scfg)
        summaries.append(summarize(simulate(policy, scfg, arrivals, channels)))
    return mean_summary(summaries)


def moving_average(values: Sequence[float], window: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if window <= 1 or arr.size <= 2:
        return arr
    window = min(window, arr.size)
    if window % 2 == 0:
        window -= 1
    if window <= 1:
        return arr
    pad = window // 2
    padded = np.pad(arr, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(padded, kernel, mode="valid")


def write_timeseries(path: Path, results: Dict[str, Dict[str, np.ndarray]]) -> None:
    fields = [
        "slot",
        "policy",
        "total_energy_j",
        "device_energy_j",
        "edge_energy_j",
        "total_queue_bits",
        "device_queue_bits",
        "edge_queue_bits",
        "fairness",
        "runtime_ms",
        "iterations",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for policy, res in results.items():
            for t in range(len(res["total_energy"])):
                writer.writerow(
                    {
                        "slot": t,
                        "policy": policy,
                        "total_energy_j": f"{res['total_energy'][t]:.8g}",
                        "device_energy_j": f"{res['device_energy'][t]:.8g}",
                        "edge_energy_j": f"{res['edge_energy'][t]:.8g}",
                        "total_queue_bits": f"{res['total_queue'][t]:.8g}",
                        "device_queue_bits": f"{res['device_queue'][t]:.8g}",
                        "edge_queue_bits": f"{res['edge_queue'][t]:.8g}",
                        "fairness": f"{res['fairness'][t]:.8g}",
                        "runtime_ms": f"{res['runtime_ms'][t]:.8g}",
                        "iterations": int(res["iterations"][t]),
                    }
                )


def write_summary(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def fmt_value(value: float) -> str:
    av = abs(value)
    if av >= 1.0e6:
        return f"{value / 1.0e6:.1f}M"
    if av >= 1.0e3:
        return f"{value / 1.0e3:.1f}k"
    if av >= 100:
        return f"{value:.0f}"
    if av >= 10:
        return f"{value:.1f}"
    if av >= 1:
        return f"{value:.2f}"
    return f"{value:.3f}"


def fmt_axis_value(value: float, step: float, axis_label: str) -> str:
    """Format axis ticks according to the axis meaning and visible precision."""
    if axis_label in {"Slot", "Devices", "Servers"}:
        return str(int(round(value)))
    if axis_label == "V":
        return f"{value:.1e}".replace("e+", "e")

    av = abs(value)
    astep = abs(step)
    suffix = ""
    scale = 1.0
    if av >= 1.0e9 or astep >= 1.0e9:
        suffix, scale = "G", 1.0e9
    elif av >= 1.0e6 or astep >= 1.0e6:
        suffix, scale = "M", 1.0e6
    elif av >= 1.0e3 or astep >= 1.0e3:
        suffix, scale = "k", 1.0e3

    scaled = value / scale
    scaled_step = astep / scale
    if scaled_step >= 10:
        decimals = 0
    elif scaled_step >= 1:
        decimals = 1
    elif scaled_step >= 0.1:
        decimals = 2
    elif scaled_step >= 0.01:
        decimals = 3
    else:
        decimals = 4
    return f"{scaled:.{decimals}f}{suffix}"


def chart_canvas(path: Path, title: str) -> Tuple[canvas.Canvas, float, float]:
    title = chart_text(title)
    w, h = landscape((520, 360))
    c = canvas.Canvas(str(path), pagesize=(w, h))
    c.setTitle(title)
    c.setFont(CHART_FONT_BOLD, 15)
    c.drawCentredString(w / 2, h - 25, title)
    return c, w, h


def draw_axes(
    c: canvas.Canvas,
    left: float,
    bottom: float,
    width: float,
    height: float,
    x_label: str,
    y_label: str,
    y_min: float,
    y_max: float,
) -> None:
    c.setStrokeColor(colors.black)
    c.setLineWidth(0.8)
    c.line(left, bottom, left, bottom + height)
    c.line(left, bottom, left + width, bottom)
    c.setFont(CHART_FONT, 11)
    y_step = (y_max - y_min) / 4
    for i in range(5):
        y = bottom + height * i / 4
        value = y_min + (y_max - y_min) * i / 4
        c.setStrokeColor(colors.HexColor("#dddddd"))
        c.line(left, y, left + width, y)
        c.setFillColor(colors.black)
        c.drawRightString(left - 5, y - 3, fmt_axis_value(value, y_step, y_label))
    c.setFillColor(colors.black)
    c.setFont(CHART_FONT_BOLD, 12)
    c.drawCentredString(left + width / 2, 20, chart_text(x_label))
    c.saveState()
    c.translate(15, bottom + height / 2)
    c.rotate(90)
    c.drawCentredString(0, 0, chart_text(y_label))
    c.restoreState()


def draw_line_chart(
    path: Path,
    title: str,
    x_label: str,
    y_label: str,
    series: Sequence[Tuple[str, Sequence[float], Sequence[float]]],
) -> None:
    c, w, h = chart_canvas(path, title)
    left, bottom = 62, 58
    plot_w, plot_h = w - 210, h - 105
    all_x = np.array([x for _, xs, _ in series for x in xs], dtype=float)
    all_y = np.array([y for _, _, ys in series for y in ys], dtype=float)
    x_min, x_max = float(np.min(all_x)), float(np.max(all_x))
    y_min, y_max = float(np.min(all_y)), float(np.max(all_y))
    if y_max <= y_min:
        y_max = y_min + 1.0
    y_pad = 0.05 * (y_max - y_min)
    y_min = max(0.0, y_min - y_pad)
    y_max += y_pad
    if x_max <= x_min:
        x_max = x_min + 1.0
    draw_axes(c, left, bottom, plot_w, plot_h, x_label, y_label, y_min, y_max)
    c.setFont(CHART_FONT, 11)
    x_step = (x_max - x_min) / 4
    for i in range(5):
        x = left + plot_w * i / 4
        value = x_min + (x_max - x_min) * i / 4
        c.drawCentredString(x, bottom - 15, fmt_axis_value(value, x_step, x_label))
    c.setFont(CHART_FONT, 10)
    for idx, (label, xs, ys) in enumerate(series):
        col = PALETTE[idx % len(PALETTE)]
        c.setStrokeColor(col)
        c.setFillColor(col)
        c.setLineWidth(1.4)
        pts = []
        for x, y in zip(xs, ys):
            px = left + plot_w * (float(x) - x_min) / (x_max - x_min)
            py = bottom + plot_h * (float(y) - y_min) / (y_max - y_min)
            pts.append((px, py))
        for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
            c.line(x0, y0, x1, y1)
        for px, py in pts[:: max(1, len(pts) // 12)]:
            c.circle(px, py, 1.6, fill=1, stroke=0)
        leg_x = left + plot_w + 16
        leg_y = bottom + plot_h - 10 - 15 * idx
        c.rect(leg_x, leg_y - 4, 9, 7, stroke=0, fill=1)
        c.setFillColor(colors.black)
        c.drawString(leg_x + 13, leg_y - 3, chart_text(label))
    c.showPage()
    c.save()


def draw_grouped_bar_chart(
    path: Path,
    title: str,
    x_label: str,
    y_label: str,
    groups: Sequence[str],
    series: Sequence[Tuple[str, Sequence[float]]],
) -> None:
    c, w, h = chart_canvas(path, title)
    left, bottom = 62, 70
    plot_w, plot_h = w - 165, h - 117
    all_y = np.array([v for _, vals in series for v in vals], dtype=float)
    y_min, y_max = 0.0, float(np.max(all_y) if all_y.size else 1.0)
    if y_label == "Jain index":
        y_max = 1.0
    else:
        y_max = y_max * 1.12 if y_max > 0 else 1.0
    draw_axes(c, left, bottom, plot_w, plot_h, x_label, y_label, y_min, y_max)
    n_groups = len(groups)
    n_series = len(series)
    group_w = plot_w / max(1, n_groups)
    bar_w = group_w * 0.72 / max(1, n_series)
    c.setFont(CHART_FONT, 9)
    for gi, group in enumerate(groups):
        cx = left + group_w * (gi + 0.5)
        c.setFillColor(colors.black)
        for line_idx, line in enumerate(chart_text(group).split("\n")):
            c.drawCentredString(cx, bottom - 15 - 9 * line_idx, line)
        for si, (_, vals) in enumerate(series):
            val = float(vals[gi])
            x = left + gi * group_w + group_w * 0.14 + si * bar_w
            y = bottom
            bh = plot_h * (val - y_min) / (y_max - y_min)
            c.setFillColor(PALETTE[si % len(PALETTE)])
            c.rect(x, y, bar_w * 0.85, bh, stroke=0, fill=1)
    for si, (label, _) in enumerate(series):
        leg_x = left + plot_w + 16
        leg_y = bottom + plot_h - 10 - 15 * si
        c.setFillColor(PALETTE[si % len(PALETTE)])
        c.rect(leg_x, leg_y - 4, 9, 7, stroke=0, fill=1)
        c.setFillColor(colors.black)
        c.drawString(leg_x + 13, leg_y - 3, chart_text(label))
    c.showPage()
    c.save()


def downsample(xs: Sequence[float], ys: Sequence[float], max_points: int = 75) -> Tuple[np.ndarray, np.ndarray]:
    xs_arr = np.asarray(xs, dtype=float)
    ys_arr = np.asarray(ys, dtype=float)
    if len(xs_arr) <= max_points:
        return xs_arr, ys_arr
    idx = np.linspace(0, len(xs_arr) - 1, max_points).astype(int)
    return xs_arr[idx], ys_arr[idx]


def grouped_policy_label(policy: str) -> str:
    """Compact, non-overlapping labels for grouped policy charts."""
    labels = {
        "EdgeSport-DPP": "EdgeSport",
        "DPQ-EEDOA (2026)": "DPQ-\nEEDOA\n(2026)",
        "Fixed-power": "Fixed-power",
        "Least-loaded": "Least-loaded",
        "LOC-SAC (2026)": "LOC-SAC\n(2026)",
        "Random-offload": "Random",
        "Local-only": "Local-only",
    }
    return labels.get(policy, policy.replace("-DPP", ""))


def run_policy_set(cfg: SimConfig, arrivals: np.ndarray, channels: np.ndarray, policies: Sequence[str]) -> Dict[str, Dict[str, np.ndarray]]:
    results = {}
    for policy in policies:
        results[policy] = simulate(policy, cfg, arrivals, channels)
    return results


def run_all(cfg: SimConfig) -> Dict[str, object]:
    ensure_dirs()
    seeds = evaluation_seeds(cfg)
    arrivals, arrival_meta = make_arrivals(cfg)
    channels = generate_channels(cfg)
    metadata: Dict[str, object] = {
        "config": asdict(cfg),
        **arrival_meta,
        "evaluation_seeds": seeds,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    baseline = run_policy_set(cfg, arrivals, channels, POLICIES)
    write_timeseries(RAW_DIR / "baseline_timeseries.csv", baseline)
    baseline_rows = []
    for policy in POLICIES:
        sm = evaluate_policy_mean(policy, cfg, seeds)
        baseline_rows.append({"scenario": "baseline_mean", "policy": policy, "num_seeds": len(seeds), **sm})
    write_summary(RAW_DIR / "baseline_summary.csv", baseline_rows)
    baseline_summary_by_policy = {row["policy"]: row for row in baseline_rows}

    slots = np.arange(cfg.T)
    energy_series = []
    queue_series = []
    for policy in POLICIES:
        xs, ys = downsample(slots, moving_average(baseline[policy]["total_energy"], cfg.plot_smoothing_window))
        energy_series.append((policy, xs, ys))
        xs, ys = downsample(slots, moving_average(baseline[policy]["total_queue"], cfg.plot_smoothing_window))
        queue_series.append((policy, xs, ys))
    draw_line_chart(FIG_DIR / "energy_vs_time_cmp.pdf", "Energy over time", "Slot", "Energy (J/slot)", energy_series)
    draw_line_chart(FIG_DIR / "queue_vs_time_cmp.pdf", "Queue backlog over time", "Slot", "Backlog (bits)", queue_series)

    sweep_rows: List[Dict[str, object]] = []

    v_values = [1.0e11, 3.0e11, 1.0e12, 3.0e12, 1.0e13]
    v_energy, v_queue = [], []
    for V in v_values:
        scfg = replace(cfg, V=V)
        sm = evaluate_policy_mean("EdgeSport-DPP", scfg, seeds)
        v_energy.append(sm["avg_energy_j"])
        v_queue.append(sm["avg_total_queue_bits"])
        sweep_rows.append({"scenario": "V", "x": V, "policy": "EdgeSport-DPP", "num_seeds": len(seeds), **sm})
    draw_line_chart(FIG_DIR / "energy_vs_V.pdf", "Energy versus V", "V", "Avg energy (J/slot)", [("EdgeSport-DPP", v_values, v_energy)])
    draw_line_chart(FIG_DIR / "queue_vs_V.pdf", "Backlog versus V", "V", "Avg backlog (bits)", [("EdgeSport-DPP", v_values, v_queue)])

    n_values = [50, 100, 150, 200, 250]
    n_energy, n_queue = [], []
    for N in n_values:
        scfg = replace(cfg, N=N)
        sm = evaluate_policy_mean("EdgeSport-DPP", scfg, seeds)
        n_energy.append(sm["avg_energy_j"])
        n_queue.append(sm["avg_total_queue_bits"])
        sweep_rows.append({"scenario": "N", "x": N, "policy": "EdgeSport-DPP", "num_seeds": len(seeds), **sm})
    draw_line_chart(FIG_DIR / "energy_vs_N.pdf", "Energy versus number of devices", "Devices", "Avg energy (J/slot)", [("EdgeSport-DPP", n_values, n_energy)])
    draw_line_chart(FIG_DIR / "queue_vs_N.pdf", "Backlog versus number of devices", "Devices", "Avg backlog (bits)", [("EdgeSport-DPP", n_values, n_queue)])

    m_values = [4, 6, 8, 10, 12, 16]
    m_energy, m_queue = [], []
    for M in m_values:
        scfg = replace(cfg, M=M)
        sm = evaluate_policy_mean("EdgeSport-DPP", scfg, seeds)
        m_energy.append(sm["avg_energy_j"])
        m_queue.append(sm["avg_total_queue_bits"])
        sweep_rows.append({"scenario": "M", "x": M, "policy": "EdgeSport-DPP", "num_seeds": len(seeds), **sm})
    draw_line_chart(FIG_DIR / "energy_vs_M.pdf", "Energy versus number of servers", "Servers", "Avg energy (J/slot)", [("EdgeSport-DPP", m_values, m_energy)])
    draw_line_chart(FIG_DIR / "queue_vs_M.pdf", "Backlog versus number of servers", "Servers", "Avg backlog (bits)", [("EdgeSport-DPP", m_values, m_queue)])

    beta_values = [0.8, 1.0, 1.2]
    rate_energy_series = []
    rate_queue_series = []
    for beta in beta_values:
        scfg = replace(cfg, beta=beta)
        sm = evaluate_policy_mean("EdgeSport-DPP", scfg, seeds)
        sweep_rows.append({"scenario": "arrival_rate", "x": beta, "policy": "EdgeSport-DPP", "num_seeds": len(seeds), **sm})
        arr, _ = make_arrivals(scfg)
        res = simulate("EdgeSport-DPP", scfg, arr, channels)
        xs, ys = downsample(slots, moving_average(res["total_energy"], cfg.plot_smoothing_window))
        rate_label = f"beta={beta:g}"
        rate_energy_series.append((rate_label, xs, ys))
        xs, ys = downsample(slots, moving_average(res["total_queue"], cfg.plot_smoothing_window))
        rate_queue_series.append((rate_label, xs, ys))
    draw_line_chart(FIG_DIR / "energy_vs_time_rate.pdf", "Energy under different arrival rates", "Slot", "Energy (J/slot)", rate_energy_series)
    draw_line_chart(FIG_DIR / "queue_vs_time_rate.pdf", "Backlog under different arrival rates", "Slot", "Backlog (bits)", rate_queue_series)

    write_summary(RAW_DIR / "sweep_summary.csv", sweep_rows)

    ablation_policies = ["EdgeSport-DPP", "DPQ-EEDOA (2026)", "Fixed-power", "Local-only"]
    ablation_groups = [grouped_policy_label(p) for p in ablation_policies]
    ab_energy = [float(baseline_summary_by_policy[p]["avg_energy_j"]) for p in ablation_policies]
    ab_queue = [float(baseline_summary_by_policy[p]["avg_total_queue_bits"]) / 1.0e6 for p in ablation_policies]
    draw_grouped_bar_chart(
        FIG_DIR / "ablation_energy_queue.pdf",
        "Ablation study",
        "Policy",
        "Value",
        ablation_groups,
        [("Energy (J/slot)", ab_energy), ("Backlog (Mbits)", ab_queue)],
    )

    fairness_values = [float(baseline_summary_by_policy[p]["avg_fairness"]) for p in POLICIES]
    draw_grouped_bar_chart(
        FIG_DIR / "fairness1.pdf",
        "Edge load fairness",
        "Policy",
        "Jain index",
        [grouped_policy_label(p) for p in POLICIES],
        [("Fairness", fairness_values)],
    )

    runtime_values = [float(baseline_summary_by_policy[p]["avg_runtime_ms"]) for p in POLICIES]
    iter_values = [float(baseline_summary_by_policy[p]["avg_iterations"]) / 100.0 for p in POLICIES]
    draw_grouped_bar_chart(
        FIG_DIR / "runtime_iterations.pdf",
        "Runtime and enumeration effort",
        "Policy",
        "Value",
        [grouped_policy_label(p) for p in POLICIES],
        [("Runtime (ms/slot)", runtime_values), ("Iterations / 100", iter_values)],
    )

    metadata["baseline_summary"] = baseline_rows
    metadata["sweep_summary_rows"] = len(sweep_rows)
    with (SUMMARY_DIR / "run_metadata.json").open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run reproducible EdgeSport MEC simulations.")
    parser.add_argument("--T", type=int, default=300, help="Number of slots.")
    parser.add_argument("--N", type=int, default=100, help="Number of monitoring devices.")
    parser.add_argument("--M", type=int, default=10, help="Number of edge servers.")
    parser.add_argument("--V", type=float, default=3.0e11, help="Nominal DPP control parameter.")
    parser.add_argument("--beta", type=float, default=1.0, help="Arrival-rate multiplier.")
    parser.add_argument("--arrival-lambda", type=float, default=6.0, help="Mean Poisson task packets per device per slot before beta scaling.")
    parser.add_argument("--task-packet-bits", type=float, default=8.0e3, help="Bits per Poisson task packet.")
    parser.add_argument("--plot-smoothing-window", type=int, default=9, help="Odd moving-average window used only for plotted time-series curves.")
    parser.add_argument("--adaptive-v-min", type=float, default=8.0e10, help="Lower bound for the adaptive EdgeSport control parameter.")
    parser.add_argument("--adaptive-v-gain", type=float, default=2.0, help="Queue-pressure gain for adaptive EdgeSport V.")
    parser.add_argument("--offload-edge-backlog-weight", type=float, default=0.05, help="Edge backlog weight in the EdgeSport offloading game.")
    parser.add_argument("--offload-min-pressure-fraction", type=float, default=0.20, help="Minimum fraction of device backlog retained as offloading pressure.")
    parser.add_argument("--server-load-penalty", type=float, default=1.0, help="Same-slot server-load penalty in the EdgeSport potential game.")
    parser.add_argument("--edge-cpu-pressure-gain", type=float, default=1.25, help="Edge CPU pressure multiplier for EdgeSport.")
    parser.add_argument("--local-v-multiplier", type=float, default=1.0, help="Multiplier applied to adaptive V for EdgeSport local CPU control.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = SimConfig(
        T=args.T,
        N=args.N,
        M=args.M,
        V=args.V,
        beta=args.beta,
        seed=args.seed,
        arrival_lambda=args.arrival_lambda,
        task_packet_bits=args.task_packet_bits,
        plot_smoothing_window=args.plot_smoothing_window,
        adaptive_v_min=args.adaptive_v_min,
        adaptive_v_gain=args.adaptive_v_gain,
        offload_edge_backlog_weight=args.offload_edge_backlog_weight,
        offload_min_pressure_fraction=args.offload_min_pressure_fraction,
        server_load_penalty=args.server_load_penalty,
        edge_cpu_pressure_gain=args.edge_cpu_pressure_gain,
        local_v_multiplier=args.local_v_multiplier,
    )
    start = time.perf_counter()
    metadata = run_all(cfg)
    elapsed = time.perf_counter() - start
    print(f"EdgeSport simulations completed in {elapsed:.2f} s")
    print(f"Arrival source: {metadata['arrival_source']}")
    print(f"Raw results: {RAW_DIR}")
    print(f"Figures: {FIG_DIR}")
    print(f"Metadata: {SUMMARY_DIR / 'run_metadata.json'}")


if __name__ == "__main__":
    main()

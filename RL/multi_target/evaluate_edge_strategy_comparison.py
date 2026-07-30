#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from RL.multi_target.common import SOFTMAX_TEMPERATURE, load_coords, load_model_module, normalize_label_key, set_seed
    from RL.multi_target.evaluate_multi_target_ppo import load_episode_list, summarize
    from RL.multi_target.multi_target_ppo_agent import MultiTargetPPOActorCritic
    from RL.multi_target.multi_target_ppo_env import MultiTargetAGVPathEnv, load_node_coords
    from RL.multi_target.train_multi_target_ppo import obs_to_tensors
else:
    from .common import SOFTMAX_TEMPERATURE, load_coords, load_model_module, normalize_label_key, set_seed
    from .evaluate_multi_target_ppo import load_episode_list, summarize
    from .multi_target_ppo_agent import MultiTargetPPOActorCritic
    from .multi_target_ppo_env import MultiTargetAGVPathEnv, load_node_coords
    from .train_multi_target_ppo import obs_to_tensors


REPO_ROOT = Path(__file__).resolve().parents[2]
FIG_WIDTH = 760
FIG_HEIGHT = 540
PLOT_LEFT = 118
PLOT_RIGHT = 42
PLOT_TOP = 36
PLOT_BOTTOM = 104
TICK_FONT = 32
LABEL_FONT = 40
LEGEND_FONT = 32
FONT_FAMILY = "Times New Roman, Times, serif"
COLORS = {
    "ppo": "#1f77b4",
    "random": "#ff7f0e",
    "a2c": "#2ca02c",
    "msloc": "#d62728",
}
DISPLAY_NAMES = {
    "ppo": "SRSense",
    "random": "Random",
    "a2c": "A2C",
    "msloc": "MS-Loc",
}


def plot_display_name(strategy: str, target_count: int) -> str:
    return f"{DISPLAY_NAMES.get(strategy, strategy)} ({target_count} targets)"


def plot_display_lines(strategy: str, target_count: int) -> tuple[str, str]:
    return DISPLAY_NAMES.get(strategy, strategy), f"({target_count} targets)"


def plot_display_axis_lines(strategy: str, target_count: int) -> tuple[str, str, str]:
    return DISPLAY_NAMES.get(strategy, strategy), "", f"({target_count} targets)"


def save_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def ticks(min_value: float, max_value: float, count: int = 6) -> list[float]:
    if max_value <= min_value:
        max_value = min_value + 1.0
    return [min_value + (max_value - min_value) * idx / (count - 1) for idx in range(count)]


def smooth_empirical_cdf(errors: np.ndarray, x_min: float, x_max: float, samples: int = 360, window: int = 9) -> tuple[np.ndarray, np.ndarray]:
    values = np.sort(errors[np.isfinite(errors)])
    if values.size == 0:
        return np.asarray([x_min, x_max]), np.asarray([0.0, 1.0])
    x_grid = np.linspace(x_min, x_max, samples)
    y = np.searchsorted(values, x_grid, side="right").astype(np.float64) / float(values.size)
    if window > 1:
        pad = window // 2
        kernel = np.ones(window, dtype=np.float64) / float(window)
        y = np.convolve(np.pad(y, (pad, pad), mode="edge"), kernel, mode="valid")
        y = np.maximum.accumulate(np.clip(y, 0.0, 1.0))
    return x_grid, y


def plot_strategy_cdf(frames: dict[str, pd.DataFrame], output_path: Path, target_count: int) -> None:
    width, height = FIG_WIDTH, FIG_HEIGHT
    left, right, top, bottom = PLOT_LEFT, PLOT_RIGHT, PLOT_TOP, PLOT_BOTTOM
    plot_w, plot_h = width - left - right, height - top - bottom
    x_min, x_max = 0.0, 8.0

    def sx(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * plot_w

    def sy(value: float) -> float:
        return top + (1.0 - value) * plot_h

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
    ]
    for y_tick in ticks(0.0, 1.0):
        y = sy(y_tick)
        lines.append(f'<line x1="{left}" y1="{y:.2f}" x2="{width - right}" y2="{y:.2f}" stroke="#edf2f7" stroke-width="1.3"/>')
        lines.append(f'<text x="{left - 16}" y="{y + 8:.2f}" text-anchor="end" font-size="{TICK_FONT}" font-family="{FONT_FAMILY}">{y_tick:.1f}</text>')
    for x_tick in range(0, 9):
        x = sx(float(x_tick))
        lines.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{height - bottom}" stroke="#f8fafc" stroke-width="1.3"/>')
        lines.append(f'<text x="{x:.2f}" y="{height - bottom + 38}" text-anchor="middle" font-size="{TICK_FONT}" font-family="{FONT_FAMILY}">{x_tick}</text>')

    for idx, (strategy, df) in enumerate(frames.items()):
        cdf_x, cdf_y = smooth_empirical_cdf(df["error"].to_numpy(dtype=np.float64), x_min, x_max)
        color = COLORS.get(strategy, "#111827")
        points = " ".join(f"{sx(float(x)):.2f},{sy(float(y)):.2f}" for x, y in zip(cdf_x, cdf_y))
        lines.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="4.0" stroke-linejoin="round" stroke-linecap="round"/>')
        legend_x = width - right - 270
        legend_y = top + 28 + idx * 34
        lines.append(f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 42}" y2="{legend_y}" stroke="{color}" stroke-width="4.0" stroke-linecap="round"/>')
        lines.append(f'<text x="{legend_x + 54}" y="{legend_y + 8}" font-size="{LEGEND_FONT}" font-family="{FONT_FAMILY}">{plot_display_name(strategy, target_count)}</text>')

    lines.extend(
        [
            f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#111827" stroke-width="2.6"/>',
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#111827" stroke-width="2.6"/>',
            f'<text x="{left + plot_w / 2:.1f}" y="{height - 30}" text-anchor="middle" font-size="{LABEL_FONT}" font-family="{FONT_FAMILY}">Localization Error (m)</text>',
            f'<text x="40" y="{top + plot_h / 2:.1f}" text-anchor="middle" font-size="{LABEL_FONT}" font-family="{FONT_FAMILY}" transform="rotate(-90 40 {top + plot_h / 2:.1f})">Cumulative Probability</text>',
            "</svg>",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def plot_strategy_boxplot(frames: dict[str, pd.DataFrame], output_path: Path, target_count: int) -> None:
    width, height = FIG_WIDTH, FIG_HEIGHT
    left, right, top, bottom = PLOT_LEFT, PLOT_RIGHT, PLOT_TOP, 132
    plot_w, plot_h = width - left - right, height - top - bottom
    strategies = list(frames.keys())
    values_by_strategy = {name: df["error"].to_numpy(dtype=np.float64) for name, df in frames.items()}
    percentile_95 = max(float(np.quantile(values, 0.95)) for values in values_by_strategy.values())
    y_min, y_max = 0.0, max(1.2, percentile_95 * 1.18)

    def sx(index: int) -> float:
        return left + (index + 0.5) / len(strategies) * plot_w

    def sy(value: float) -> float:
        clipped = max(y_min, min(y_max, value))
        return top + (1.0 - (clipped - y_min) / (y_max - y_min)) * plot_h

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
    ]
    for tick in ticks(y_min, y_max):
        y = sy(tick)
        lines.append(f'<line x1="{left}" y1="{y:.2f}" x2="{width - right}" y2="{y:.2f}" stroke="#edf2f7" stroke-width="1.3"/>')
        lines.append(f'<text x="{left - 16}" y="{y + 8:.2f}" text-anchor="end" font-size="{TICK_FONT}" font-family="{FONT_FAMILY}">{tick:.1f}</text>')

    box_w = plot_w / len(strategies) * 0.38
    for idx, strategy in enumerate(strategies):
        values = values_by_strategy[strategy]
        q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
        low, high = np.quantile(values, [0.05, 0.95])
        mean = float(np.mean(values))
        x = sx(idx)
        color = COLORS.get(strategy, "#111827")
        y_q1, y_q3 = sy(float(q1)), sy(float(q3))
        y_med, y_low, y_high = sy(float(median)), sy(float(low)), sy(float(high))
        y_mean = sy(mean)
        lines.append(f'<line x1="{x:.2f}" y1="{y_high:.2f}" x2="{x:.2f}" y2="{y_low:.2f}" stroke="#111827" stroke-width="2.4"/>')
        lines.append(f'<line x1="{x - box_w * 0.35:.2f}" y1="{y_high:.2f}" x2="{x + box_w * 0.35:.2f}" y2="{y_high:.2f}" stroke="#111827" stroke-width="2.4"/>')
        lines.append(f'<line x1="{x - box_w * 0.35:.2f}" y1="{y_low:.2f}" x2="{x + box_w * 0.35:.2f}" y2="{y_low:.2f}" stroke="#111827" stroke-width="2.4"/>')
        lines.append(f'<rect x="{x - box_w / 2:.2f}" y="{y_q3:.2f}" width="{box_w:.2f}" height="{y_q1 - y_q3:.2f}" fill="{color}" fill-opacity="0.55" stroke="#111827" stroke-width="1.8"/>')
        lines.append(f'<line x1="{x - box_w / 2:.2f}" y1="{y_med:.2f}" x2="{x + box_w / 2:.2f}" y2="{y_med:.2f}" stroke="#111827" stroke-width="2.7"/>')
        lines.append(f'<circle cx="{x:.2f}" cy="{y_mean:.2f}" r="5.0" fill="#111827"/>')
        label, _, target_suffix = plot_display_axis_lines(strategy, target_count)
        lines.append(f'<text x="{x:.2f}" y="{height - bottom + 36}" text-anchor="middle" font-size="{TICK_FONT}" font-family="{FONT_FAMILY}">{label}</text>')
        lines.append(f'<text x="{x:.2f}" y="{height - bottom + 64}" text-anchor="middle" font-size="{TICK_FONT}" font-family="{FONT_FAMILY}">{target_suffix}</text>')

    lines.extend(
        [
            f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#111827" stroke-width="2.6"/>',
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#111827" stroke-width="2.6"/>',
            f'<text x="{left + plot_w / 2:.1f}" y="{height - 30}" text-anchor="middle" font-size="{LABEL_FONT}" font-family="{FONT_FAMILY}">Path Selection Strategy</text>',
            f'<text x="40" y="{top + plot_h / 2:.1f}" text-anchor="middle" font-size="{LABEL_FONT}" font-family="{FONT_FAMILY}" transform="rotate(-90 40 {top + plot_h / 2:.1f})">Localization Error (m)</text>',
            "</svg>",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


MOVEMENT_DISTANCE_PER_STEP_M = 1.7


def plot_strategy_movement_distance(summary_rows: list[dict], output_path: Path, target_count: int) -> None:
    width, height = FIG_WIDTH, FIG_HEIGHT
    left, right, top, bottom = PLOT_LEFT, PLOT_RIGHT, PLOT_TOP, 132
    plot_w, plot_h = width - left - right, height - top - bottom
    strategies = [str(row["strategy"]) for row in summary_rows]
    means = np.asarray([row["mean_edge_count"] for row in summary_rows], dtype=np.float64) * MOVEMENT_DISTANCE_PER_STEP_M
    stds = np.asarray([row["std_edge_count"] for row in summary_rows], dtype=np.float64) * MOVEMENT_DISTANCE_PER_STEP_M
    y_min = 0.0
    y_max = max(4.0, float(np.max(means + stds)) + 0.3)

    def sx(index: int) -> float:
        return left + (index + 0.5) / len(strategies) * plot_w

    def sy(value: float) -> float:
        return top + (1.0 - (value - y_min) / (y_max - y_min)) * plot_h

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
    ]
    for tick in ticks(y_min, y_max):
        y = sy(tick)
        lines.append(f'<line x1="{left}" y1="{y:.2f}" x2="{width - right}" y2="{y:.2f}" stroke="#edf2f7" stroke-width="1.3"/>')
        lines.append(f'<text x="{left - 16}" y="{y + 8:.2f}" text-anchor="end" font-size="{TICK_FONT}" font-family="{FONT_FAMILY}">{tick:.1f}</text>')

    bar_w = plot_w / len(strategies) * 0.58
    for idx, (strategy, mean, std) in enumerate(zip(strategies, means, stds)):
        x = sx(idx)
        y = sy(float(mean))
        baseline = sy(0.0)
        color = COLORS.get(strategy, "#111827")
        lines.append(f'<rect x="{x - bar_w / 2:.2f}" y="{y:.2f}" width="{bar_w:.2f}" height="{baseline - y:.2f}" fill="{color}" fill-opacity="0.78" stroke="#1f2937" stroke-width="1.5"/>')
        y_low = sy(float(max(0.0, mean - std)))
        y_high = sy(float(mean + std))
        cap = bar_w * 0.24
        lines.append(f'<line x1="{x:.2f}" y1="{y_high:.2f}" x2="{x:.2f}" y2="{y_low:.2f}" stroke="#111827" stroke-width="2.4"/>')
        lines.append(f'<line x1="{x - cap:.2f}" y1="{y_high:.2f}" x2="{x + cap:.2f}" y2="{y_high:.2f}" stroke="#111827" stroke-width="2.4"/>')
        lines.append(f'<line x1="{x - cap:.2f}" y1="{y_low:.2f}" x2="{x + cap:.2f}" y2="{y_low:.2f}" stroke="#111827" stroke-width="2.4"/>')
        label, _, target_suffix = plot_display_axis_lines(strategy, target_count)
        lines.append(f'<text x="{x:.2f}" y="{height - bottom + 36}" text-anchor="middle" font-size="{TICK_FONT}" font-family="{FONT_FAMILY}">{label}</text>')
        lines.append(f'<text x="{x:.2f}" y="{height - bottom + 64}" text-anchor="middle" font-size="{TICK_FONT}" font-family="{FONT_FAMILY}">{target_suffix}</text>')

    lines.extend(
        [
            f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#111827" stroke-width="2.6"/>',
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#111827" stroke-width="2.6"/>',
            f'<text x="{left + plot_w / 2:.1f}" y="{height - 30}" text-anchor="middle" font-size="{LABEL_FONT}" font-family="{FONT_FAMILY}">Path Selection Strategy</text>',
            f'<text x="40" y="{top + plot_h / 2 + 30:.1f}" text-anchor="middle" font-size="{LABEL_FONT}" font-family="{FONT_FAMILY}" transform="rotate(-90 40 {top + plot_h / 2 + 30:.1f})">Mean Movement Distance (m)</text>',
            "</svg>",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def load_policy(path: Path, env: MultiTargetAGVPathEnv, max_edges: int, max_targets: int, device: torch.device) -> MultiTargetPPOActorCritic:
    policy = MultiTargetPPOActorCritic(
        node_num=env.node_num,
        edge_num=env.edge_num,
        action_size=env.action_size,
        max_edges=max_edges,
        max_targets=max_targets,
    ).to(device)
    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    policy.load_state_dict(state_dict)
    policy.eval()
    return policy


def generate_episode_specs(
    env: MultiTargetAGVPathEnv,
    episodes: int,
    target_count: int,
    target_labels: list[str] | None,
    random_start: bool,
    start_node: int,
    seed: int,
) -> list[dict]:
    rng = random.Random(seed)
    specs: list[dict] = []
    attempts = 0
    while len(specs) < episodes:
        attempts += 1
        if attempts > episodes * 300:
            raise RuntimeError("failed to generate enough evaluation episodes")
        labels = list(target_labels) if target_labels is not None else rng.sample(env.available_labels, target_count)
        graphs = [rng.choice(env.graphs_by_label[label]) for label in labels]
        common_nodes = set(graphs[0].nodes)
        for graph in graphs[1:]:
            common_nodes &= set(graph.nodes)
        if random_start:
            candidates = list(common_nodes)
            rng.shuffle(candidates)
        else:
            candidates = [int(start_node)] if int(start_node) in common_nodes else []
        chosen_start = None
        for node in candidates:
            env.used_edge_keys = set()
            if env._candidate_direction_edges(graphs, int(node)):
                chosen_start = int(node)
                break
        if chosen_start is None:
            continue
        specs.append(
            {
                "episode": len(specs),
                "target_count": target_count,
                "start_node": chosen_start,
                "target_groups": [graph.group for graph in graphs],
            }
        )
    return specs


def write_episode_specs(path: Path, specs: list[dict]) -> None:
    rows = []
    for spec in specs:
        rows.append(
            {
                "episode": spec["episode"],
                "target_count": spec["target_count"],
                "start_node": spec["start_node"],
                "target_groups": "|".join(spec["target_groups"]),
            }
        )
    save_csv(path, rows)


def predict_probabilities(env: MultiTargetAGVPathEnv) -> np.ndarray | None:
    if not env.edge_history_keys:
        return None
    use_path_id = bool(
        getattr(
            env.classifier,
            "uses_path_id",
            env.model_module.input_mode_uses_path_id(str(getattr(env.classifier, "input_mode", "srs_rsrp_path"))),
        )
    )
    signal_batches = []
    rsrp_batches = []
    path_id_batches = []
    edge_mask_batches = []
    for history in env.target_edge_histories:
        sample = env._build_sample(history, use_path_id)
        signal_batches.append(sample["signal"])
        rsrp_batches.append(sample["rsrp"])
        path_id_batches.append(sample["path_id"])
        edge_mask_batches.append(sample["edge_mask"])
    signal = torch.stack(signal_batches, dim=0).to(env.device)
    rsrp = torch.stack(rsrp_batches, dim=0).to(env.device)
    path_id = torch.stack(path_id_batches, dim=0).to(env.device)
    edge_mask = torch.stack(edge_mask_batches, dim=0).to(env.device)
    env.classifier.eval()
    with torch.no_grad():
        logits = env.classifier(signal, rsrp, path_id, edge_mask) / SOFTMAX_TEMPERATURE
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    return probs


def topk_confidence(probs: np.ndarray | None, k: int) -> float:
    if probs is None or probs.size == 0:
        return 0.0
    k = max(1, min(int(k), probs.shape[1]))
    values = np.sort(probs, axis=1)[:, -k:]
    return float(np.mean(np.sum(values, axis=1)))


def edge_variation(model_module: Any, edge_path: str, stats: Any) -> float:
    signal, rsrp = model_module.load_csv_cached(edge_path)
    signal = model_module.normalize_signal(signal, stats)
    rsrp = ((rsrp - stats.rsrp_mean) / stats.rsrp_std).astype(np.float32)
    signal_feature = signal.mean(axis=1)
    feature = np.concatenate([signal_feature, rsrp], axis=1)
    if feature.shape[0] <= 1:
        return float(np.mean(np.abs(feature)))
    return float(np.mean(np.abs(np.diff(feature, axis=0))))


def build_rssd_table(model_module: Any, rssd_dir: Path, path_to_index: dict[str, int], stats: Any) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    per_label_values: dict[str, dict[str, list[float]]] = {}
    edge_values: dict[str, list[float]] = {}
    for label_dir in model_module.iter_label_dirs(str(rssd_dir)):
        label = normalize_label_key(label_dir.name)
        for turn_dir in model_module.iter_turn_dirs(label_dir):
            for edge in model_module.collect_edges(turn_dir):
                if edge.key not in path_to_index:
                    continue
                value = edge_variation(model_module, edge.path, stats)
                per_label_values.setdefault(label, {}).setdefault(edge.key, []).append(value)
                edge_values.setdefault(edge.key, []).append(value)

    if not edge_values:
        raise RuntimeError(f"no edge variation statistics could be built from {rssd_dir}")

    rssd = {
        label: {edge_key: float(np.mean(values)) for edge_key, values in edge_map.items()}
        for label, edge_map in per_label_values.items()
    }
    edge_mean = {edge_key: float(np.mean(values)) for edge_key, values in edge_values.items()}
    return rssd, edge_mean


def confidence_stop_action(env: MultiTargetAGVPathEnv, threshold: float, topk: int) -> bool:
    mask = env.action_mask()
    if mask[0] <= 0:
        return False
    return topk_confidence(predict_probabilities(env), topk) >= threshold


def select_policy_action(
    policy: MultiTargetPPOActorCritic,
    env: MultiTargetAGVPathEnv,
    obs: dict[str, np.ndarray],
    device: torch.device,
) -> int:
    tensors = obs_to_tensors(obs, device)
    legal_action_mask = torch.as_tensor(env.action_mask(), dtype=torch.float32, device=device).view(1, -1)
    with torch.no_grad():
        logits, _ = policy(
            tensors["current_node"],
            tensors["edge_history"],
            tensors["signal_state"],
            tensors["rsrp_state"],
        )
        logits = policy.mask_logits(logits, legal_action_mask)
        return int(torch.argmax(logits, dim=1).item())


def select_random_action(
    env: MultiTargetAGVPathEnv,
    rng: random.Random,
    min_edges_before_stop: int,
    fixed_edges: int | None,
) -> int:
    mask = env.action_mask()
    edge_count = len(env.edge_history_keys)
    if fixed_edges is not None and edge_count >= int(fixed_edges) and mask[0] > 0:
        return 0
    # Random path baseline: no score is used; stop is only allowed after a minimum path budget.
    legal = [int(action) for action in np.flatnonzero(mask > 0).tolist()]
    if fixed_edges is not None:
        movement_actions = [action for action in legal if action != 0]
        if movement_actions:
            legal = movement_actions
    elif edge_count < int(min_edges_before_stop):
        movement_actions = [action for action in legal if action != 0]
        if movement_actions:
            legal = movement_actions
    if not legal:
        return 0
    return int(rng.choice(legal))


def select_msloc_action(
    env: MultiTargetAGVPathEnv,
    rssd: dict[str, dict[str, float]],
    edge_mean: dict[str, float],
    threshold: float,
    topk: int,
) -> int:
    mask = env.action_mask()
    if confidence_stop_action(env, threshold, topk):
        return 0
    direction_edges = env._candidate_direction_edges(env.target_graphs, int(env.current_node))
    legal_actions = [int(action) for action in np.flatnonzero(mask > 0).tolist() if int(action) != 0 and int(action) in direction_edges]
    if not legal_actions:
        return 0

    probs = predict_probabilities(env)
    best_action = legal_actions[0]
    best_score = -float("inf")
    for action in legal_actions:
        edge_key = direction_edges[action]
        if probs is None:
            score = edge_mean.get(edge_key, 0.0)
        else:
            target_scores = []
            for prob in probs:
                k = max(1, min(int(topk), prob.shape[0]))
                top_indices = np.argsort(-prob)[:k]
                value = 0.0
                for idx in top_indices:
                    label = normalize_label_key(env.labels[int(idx)])
                    value += float(prob[int(idx)]) * float(rssd.get(label, {}).get(edge_key, edge_mean.get(edge_key, 0.0)))
                target_scores.append(value)
            score = float(np.mean(target_scores)) if target_scores else edge_mean.get(edge_key, 0.0)
        if score > best_score:
            best_score = score
            best_action = action
    return int(best_action)


def evaluate_strategy(
    strategy: str,
    env: MultiTargetAGVPathEnv,
    episode_specs: list[dict],
    device: torch.device,
    rng: random.Random,
    policy: MultiTargetPPOActorCritic | None = None,
    rssd: dict[str, dict[str, float]] | None = None,
    edge_mean: dict[str, float] | None = None,
    confidence_threshold: float = 0.70,
    topk: int = 3,
    random_min_edges: int = 3,
    random_fixed_edges: int | None = None,
) -> list[dict]:
    rows: list[dict] = []
    for episode, spec in enumerate(episode_specs):
        obs = env.reset_to(spec["target_groups"], spec["start_node"])
        done = False
        total_reward = 0.0
        final_info: dict[str, Any] = {}
        while not done:
            if strategy in {"ppo", "a2c"}:
                if policy is None:
                    raise ValueError(f"{strategy} requires a policy")
                action = select_policy_action(policy, env, obs, device)
            elif strategy == "random":
                action = select_random_action(env, rng, random_min_edges, random_fixed_edges)
            elif strategy == "msloc":
                if rssd is None or edge_mean is None:
                    raise ValueError("msloc requires rssd statistics")
                action = select_msloc_action(env, rssd, edge_mean, confidence_threshold, topk)
            else:
                raise ValueError(f"unknown strategy: {strategy}")
            obs, reward, done, info = env.step(action)
            total_reward += float(reward)
            final_info = info

        rows.append(
            {
                "episode": episode,
                "strategy": strategy,
                "reward": total_reward,
                "error": float(final_info.get("error", 0.0)),
                "edge_count": int(final_info.get("edge_count", 0)),
                "target_count": int(final_info.get("target_count", 0)),
                "forced_terminal": bool(final_info.get("forced_terminal", False)),
                "group": final_info.get("group", ""),
                "label": final_info.get("label", ""),
                "path": final_info.get("path", ""),
                "directions": final_info.get("directions", ""),
                "start_node": final_info.get("start_node", ""),
                "est_x": final_info.get("est_x", ""),
                "est_y": final_info.get("est_y", ""),
                "true_x": final_info.get("true_x", ""),
                "true_y": final_info.get("true_y", ""),
                "top1_label": final_info.get("top1_label", ""),
                "target_labels": "|".join(map(str, final_info.get("target_labels", []))),
                "target_groups": "|".join(map(str, final_info.get("target_groups", []))),
                "target_errors": "|".join(f"{x:.4f}" for x in final_info.get("target_errors", [])),
                "max_error": float(final_info.get("max_error", 0.0)),
                "median_error": float(final_info.get("median_error", 0.0)),
            }
        )
    return rows


def summarize_strategy(strategy: str, rows: list[dict]) -> dict:
    base = summarize(rows)
    errors = np.asarray([row["error"] for row in rows], dtype=np.float64)
    edges = np.asarray([row["edge_count"] for row in rows], dtype=np.float64)
    return {
        "strategy": strategy,
        "display_name": DISPLAY_NAMES.get(strategy, strategy),
        **base,
        "p80_error": float(np.quantile(errors, 0.80)) if errors.size else 0.0,
        "std_error": float(np.std(errors, ddof=1)) if len(errors) > 1 else 0.0,
        "std_edge_count": float(np.std(edges, ddof=1)) if len(edges) > 1 else 0.0,
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare edge-selection strategies with the same ILC-MG classifier.")
    parser.add_argument("--eval-dir", default=str(REPO_ROOT / "data" / "PPO-test_target"))
    parser.add_argument("--classifier-dir", required=True)
    parser.add_argument("--ppo-policy-path", required=True)
    parser.add_argument("--a2c-policy-path", default=None)
    parser.add_argument("--rssd-dir", default=str(REPO_ROOT / "data" / "SRS-train"), help="Offline RP data used by the MS-Loc-inspired heuristic.")
    parser.add_argument("--rp-coords", default=str(REPO_ROOT / "data" / "ILC-MG_rp_coords.csv"))
    parser.add_argument("--true-coords", default=str(REPO_ROOT / "data" / "PPO_target_coords.csv"))
    parser.add_argument("--map-node-coords", default=str(REPO_ROOT / "data" / "map_node_coords.csv"))
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--episode-list", default=None)
    parser.add_argument("--episodes", type=int, default=4500)
    parser.add_argument("--target-count", type=int, default=3)
    parser.add_argument("--target-labels", nargs="+", default=None)
    parser.add_argument("--random-start", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--start-node", type=int, default=21)
    parser.add_argument("--max-targets", type=int, default=5)
    parser.add_argument("--max-edges", type=int, default=4)
    parser.add_argument("--wknn-k", type=int, default=3)
    parser.add_argument("--eta", type=float, default=0.04)
    parser.add_argument("--msloc-threshold", type=float, default=0.70)
    parser.add_argument("--random-threshold", type=float, default=0.70)
    parser.add_argument("--random-min-edges", type=int, default=3)
    parser.add_argument("--random-fixed-edges", type=int, default=None)
    parser.add_argument("--heuristic-topk", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", default=None)
    parser.add_argument("--disable-cudnn", action=argparse.BooleanOptionalAction, default=False)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    if args.target_count != 3:
        print(f"Warning: this comparison is intended for 3 targets, got target_count={args.target_count}", flush=True)
    if args.target_labels is not None and len(args.target_labels) != args.target_count:
        raise ValueError(f"--target-labels must contain exactly {args.target_count} labels")
    set_seed(args.seed)

    model_module = load_model_module()
    device = model_module.build_device(args.gpu)
    model_module.configure_torch_runtime(device, args.disable_cudnn)
    print(f"Using device: {device}", flush=True)

    classifier, labels, path_keys, path_to_index, stats, config = model_module.load_artifacts(args.classifier_dir, device)
    classifier.eval()
    for param in classifier.parameters():
        param.requires_grad = False

    max_edges = int(args.max_edges or config.get("max_edges", 4))
    env = MultiTargetAGVPathEnv(
        model_module=model_module,
        classifier=classifier,
        labels=labels,
        path_keys=path_keys,
        path_to_index=path_to_index,
        stats=stats,
        rp_coords=load_coords(args.rp_coords),
        true_coords=load_coords(args.true_coords),
        node_coords=load_node_coords(args.map_node_coords),
        train_dir=args.eval_dir,
        device=device,
        max_edges=max_edges,
        max_targets=args.max_targets,
        min_targets=args.target_count,
        fixed_target_count=args.target_count,
        eta=args.eta,
        wknn_k=args.wknn_k,
        seed=args.seed,
    )

    if args.episode_list:
        episode_specs = load_episode_list(args.episode_list, args.target_count)
    else:
        episode_specs = generate_episode_specs(
            env=env,
            episodes=args.episodes,
            target_count=args.target_count,
            target_labels=args.target_labels,
            random_start=args.random_start,
            start_node=args.start_node,
            seed=args.seed,
        )

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    write_episode_specs(out_dir / "edge_strategy_eval_episodes.csv", episode_specs)

    ppo_policy = load_policy(Path(args.ppo_policy_path), env, max_edges, args.max_targets, device)
    a2c_policy = None
    if args.a2c_policy_path:
        a2c_path = Path(args.a2c_policy_path)
        if a2c_path.exists():
            a2c_policy = load_policy(a2c_path, env, max_edges, args.max_targets, device)
        else:
            print(f"Skip A2C: checkpoint not found: {a2c_path}", flush=True)

    print(f"Building MS-Loc-inspired RSSD table from {args.rssd_dir}", flush=True)
    rssd, edge_mean = build_rssd_table(model_module, Path(args.rssd_dir), path_to_index, stats)

    strategy_specs: list[tuple[str, MultiTargetPPOActorCritic | None]] = [("ppo", ppo_policy), ("random", None)]
    if a2c_policy is not None:
        strategy_specs.append(("a2c", a2c_policy))
    strategy_specs.append(("msloc", None))

    frames: dict[str, pd.DataFrame] = {}
    all_rows: list[dict] = []
    summary_rows: list[dict] = []
    for strategy, policy in strategy_specs:
        print(f"Evaluating {DISPLAY_NAMES.get(strategy, strategy)}", flush=True)
        rows = evaluate_strategy(
            strategy=strategy,
            env=env,
            episode_specs=episode_specs,
            device=device,
            rng=random.Random(args.seed + 1009),
            policy=policy,
            rssd=rssd,
            edge_mean=edge_mean,
            confidence_threshold=args.random_threshold if strategy == "random" else args.msloc_threshold,
            topk=args.heuristic_topk,
            random_min_edges=args.random_min_edges,
            random_fixed_edges=args.random_fixed_edges,
        )
        strategy_dir = out_dir / "strategies" / strategy
        save_csv(strategy_dir / "eval_episodes.csv", rows)
        summary = summarize_strategy(strategy, rows)
        with (strategy_dir / "eval_summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        frames[strategy] = pd.DataFrame(rows)
        all_rows.extend(rows)
        summary_rows.append(summary)
        print(json.dumps(summary, indent=2), flush=True)

    save_csv(out_dir / "strategy_eval_episodes.csv", all_rows)
    save_csv(out_dir / "strategy_eval_summary.csv", summary_rows)
    plot_strategy_cdf(frames, out_dir / "strategy_error_cdf.svg", args.target_count)
    plot_strategy_boxplot(frames, out_dir / "strategy_error_boxplot.svg", args.target_count)
    plot_strategy_movement_distance(summary_rows, out_dir / "strategy_mean_movement_distance_bar.svg", args.target_count)
    print(f"Saved results under {out_dir}", flush=True)


if __name__ == "__main__":
    main()

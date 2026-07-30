#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from RL.multi_target.common import load_coords, load_model_module, set_seed
    from RL.multi_target.evaluate_edge_strategy_comparison import generate_episode_specs, write_episode_specs
    from RL.multi_target.evaluate_multi_target_ppo import load_episode_list, summarize
    from RL.multi_target.multi_target_ppo_agent import MultiTargetPPOActorCritic
    from RL.multi_target.multi_target_ppo_env import MultiTargetAGVPathEnv, load_node_coords
    from RL.multi_target.train_multi_target_ppo import obs_to_tensors
else:
    from .common import load_coords, load_model_module, set_seed
    from .evaluate_edge_strategy_comparison import generate_episode_specs, write_episode_specs
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
    "ilcmg": "#1f77b4",
    "aarescnn": "#ff7f0e",
    "hiloc": "#2ca02c",
    "aares": "#d62728",
}
DISPLAY_NAMES = {
    "ilcmg": "ILC-MG",
    "aarescnn": "AAResCNN",
    "hiloc": "Hi-Loc",
    "aares": "AARES",
}


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


def classifier_color(key: str, index: int) -> str:
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    return COLORS.get(key, palette[index % len(palette)])


def display_name(key: str, fallback: str) -> str:
    return DISPLAY_NAMES.get(key, fallback)


def plot_classifier_cdf(frames: dict[str, pd.DataFrame], names: dict[str, str], output_path: Path) -> None:
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

    for idx, (key, df) in enumerate(frames.items()):
        cdf_x, cdf_y = smooth_empirical_cdf(df["error"].to_numpy(dtype=np.float64), x_min, x_max)
        color = classifier_color(key, idx)
        points = " ".join(f"{sx(float(x)):.2f},{sy(float(y)):.2f}" for x, y in zip(cdf_x, cdf_y))
        lines.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="4.0" stroke-linejoin="round" stroke-linecap="round"/>')
        legend_x = width - right - 190
        legend_y = top + 28 + idx * 34
        lines.append(f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 42}" y2="{legend_y}" stroke="{color}" stroke-width="4.0" stroke-linecap="round"/>')
        lines.append(f'<text x="{legend_x + 54}" y="{legend_y + 8}" font-size="{LEGEND_FONT}" font-family="{FONT_FAMILY}">{names[key]}</text>')

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


def plot_classifier_boxplot(frames: dict[str, pd.DataFrame], names: dict[str, str], output_path: Path) -> None:
    width, height = FIG_WIDTH, FIG_HEIGHT
    left, right, top, bottom = PLOT_LEFT, PLOT_RIGHT, PLOT_TOP, 124
    plot_w, plot_h = width - left - right, height - top - bottom
    keys = list(frames.keys())
    values_by_key = {key: df["error"].to_numpy(dtype=np.float64) for key, df in frames.items()}
    percentile_95 = max(float(np.quantile(values, 0.95)) for values in values_by_key.values())
    y_min, y_max = 0.0, max(1.2, percentile_95 * 1.18)

    def sx(index: int) -> float:
        return left + (index + 0.5) / len(keys) * plot_w

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

    box_w = plot_w / len(keys) * 0.38
    for idx, key in enumerate(keys):
        values = values_by_key[key]
        q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
        low, high = np.quantile(values, [0.05, 0.95])
        mean = float(np.mean(values))
        x = sx(idx)
        color = classifier_color(key, idx)
        y_q1, y_q3 = sy(float(q1)), sy(float(q3))
        y_med, y_low, y_high = sy(float(median)), sy(float(low)), sy(float(high))
        y_mean = sy(mean)
        lines.append(f'<line x1="{x:.2f}" y1="{y_high:.2f}" x2="{x:.2f}" y2="{y_low:.2f}" stroke="#111827" stroke-width="2.4"/>')
        lines.append(f'<line x1="{x - box_w * 0.35:.2f}" y1="{y_high:.2f}" x2="{x + box_w * 0.35:.2f}" y2="{y_high:.2f}" stroke="#111827" stroke-width="2.4"/>')
        lines.append(f'<line x1="{x - box_w * 0.35:.2f}" y1="{y_low:.2f}" x2="{x + box_w * 0.35:.2f}" y2="{y_low:.2f}" stroke="#111827" stroke-width="2.4"/>')
        lines.append(f'<rect x="{x - box_w / 2:.2f}" y="{y_q3:.2f}" width="{box_w:.2f}" height="{y_q1 - y_q3:.2f}" fill="{color}" fill-opacity="0.55" stroke="#111827" stroke-width="1.8"/>')
        lines.append(f'<line x1="{x - box_w / 2:.2f}" y1="{y_med:.2f}" x2="{x + box_w / 2:.2f}" y2="{y_med:.2f}" stroke="#111827" stroke-width="2.7"/>')
        lines.append(f'<circle cx="{x:.2f}" cy="{y_mean:.2f}" r="5.0" fill="#111827"/>')
        lines.append(f'<text x="{x:.2f}" y="{height - bottom + 42}" text-anchor="middle" font-size="{TICK_FONT}" font-family="{FONT_FAMILY}">{names[key]}</text>')

    lines.extend(
        [
            f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#111827" stroke-width="2.6"/>',
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#111827" stroke-width="2.6"/>',
            f'<text x="{left + plot_w / 2:.1f}" y="{height - 30}" text-anchor="middle" font-size="{LABEL_FONT}" font-family="{FONT_FAMILY}">Classifier Network</text>',
            f'<text x="40" y="{top + plot_h / 2:.1f}" text-anchor="middle" font-size="{LABEL_FONT}" font-family="{FONT_FAMILY}" transform="rotate(-90 40 {top + plot_h / 2:.1f})">Localization Error (m)</text>',
            "</svg>",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def plot_classifier_mean_error(summary_rows: list[dict], output_path: Path) -> None:
    width, height = FIG_WIDTH, FIG_HEIGHT
    left, right, top, bottom = PLOT_LEFT, PLOT_RIGHT, PLOT_TOP, 124
    plot_w, plot_h = width - left - right, height - top - bottom
    keys = [str(row["classifier"]) for row in summary_rows]
    names = [str(row["display_name"]) for row in summary_rows]
    means = np.asarray([row["mean_error"] for row in summary_rows], dtype=np.float64)
    stds = np.asarray([row["std_error"] for row in summary_rows], dtype=np.float64)
    y_min = 0.0
    y_max = max(1.2, float(np.max(means + stds)) * 1.18)

    def sx(index: int) -> float:
        return left + (index + 0.5) / len(keys) * plot_w

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

    bar_w = plot_w / len(keys) * 0.58
    for idx, (key, name, mean, std) in enumerate(zip(keys, names, means, stds)):
        x = sx(idx)
        y = sy(float(mean))
        baseline = sy(0.0)
        color = classifier_color(key, idx)
        lines.append(f'<rect x="{x - bar_w / 2:.2f}" y="{y:.2f}" width="{bar_w:.2f}" height="{baseline - y:.2f}" fill="{color}" fill-opacity="0.78" stroke="#1f2937" stroke-width="1.5"/>')
        y_low = sy(float(max(0.0, mean - std)))
        y_high = sy(float(mean + std))
        cap = bar_w * 0.24
        lines.append(f'<line x1="{x:.2f}" y1="{y_high:.2f}" x2="{x:.2f}" y2="{y_low:.2f}" stroke="#111827" stroke-width="2.4"/>')
        lines.append(f'<line x1="{x - cap:.2f}" y1="{y_high:.2f}" x2="{x + cap:.2f}" y2="{y_high:.2f}" stroke="#111827" stroke-width="2.4"/>')
        lines.append(f'<line x1="{x - cap:.2f}" y1="{y_low:.2f}" x2="{x + cap:.2f}" y2="{y_low:.2f}" stroke="#111827" stroke-width="2.4"/>')
        lines.append(f'<text x="{x:.2f}" y="{height - bottom + 42}" text-anchor="middle" font-size="{TICK_FONT}" font-family="{FONT_FAMILY}">{name}</text>')

    lines.extend(
        [
            f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#111827" stroke-width="2.6"/>',
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#111827" stroke-width="2.6"/>',
            f'<text x="{left + plot_w / 2:.1f}" y="{height - 30}" text-anchor="middle" font-size="{LABEL_FONT}" font-family="{FONT_FAMILY}">Classifier Network</text>',
            f'<text x="40" y="{top + plot_h / 2:.1f}" text-anchor="middle" font-size="{LABEL_FONT}" font-family="{FONT_FAMILY}" transform="rotate(-90 40 {top + plot_h / 2:.1f})">Mean Error (m)</text>',
            "</svg>",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_classifier_specs(values: list[str]) -> list[tuple[str, str, Path]]:
    specs: list[tuple[str, str, Path]] = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"--classifier must be formatted as key=path, got {value!r}")
        key, raw_path = value.split("=", 1)
        key = key.strip().lower()
        if not key:
            raise ValueError(f"empty classifier key in {value!r}")
        fallback_name = key
        specs.append((key, display_name(key, fallback_name), Path(raw_path).expanduser().resolve()))
    return specs


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


def select_policy_action(policy: MultiTargetPPOActorCritic, env: MultiTargetAGVPathEnv, obs: dict[str, np.ndarray], device: torch.device) -> int:
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


def evaluate_classifier(
    classifier_key: str,
    display: str,
    env: MultiTargetAGVPathEnv,
    policy: MultiTargetPPOActorCritic,
    episode_specs: list[dict],
    device: torch.device,
) -> list[dict]:
    rows: list[dict] = []
    for episode, spec in enumerate(episode_specs):
        obs = env.reset_to(spec["target_groups"], spec["start_node"])
        done = False
        total_reward = 0.0
        final_info: dict = {}
        while not done:
            action = select_policy_action(policy, env, obs, device)
            obs, reward, done, info = env.step(action)
            total_reward += float(reward)
            final_info = info

        rows.append(
            {
                "episode": episode,
                "classifier": classifier_key,
                "display_name": display,
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


def summarize_classifier(classifier_key: str, display: str, rows: list[dict]) -> dict:
    base = summarize(rows)
    errors = np.asarray([row["error"] for row in rows], dtype=np.float64)
    edges = np.asarray([row["edge_count"] for row in rows], dtype=np.float64)
    return {
        "classifier": classifier_key,
        "display_name": display,
        **base,
        "p80_error": float(np.quantile(errors, 0.80)) if errors.size else 0.0,
        "std_error": float(np.std(errors, ddof=1)) if len(errors) > 1 else 0.0,
        "std_edge_count": float(np.std(edges, ddof=1)) if len(edges) > 1 else 0.0,
    }


def build_env(
    model_module,
    classifier_dir: Path,
    eval_dir: str,
    rp_coords: dict[str, tuple[float, float]],
    true_coords: dict[str, tuple[float, float]],
    node_coords: dict[int, tuple[float, float]],
    device: torch.device,
    max_edges: int,
    max_targets: int,
    target_count: int,
    eta: float,
    wknn_k: int,
    seed: int,
) -> tuple[MultiTargetAGVPathEnv, dict]:
    classifier, labels, path_keys, path_to_index, stats, config = model_module.load_artifacts(str(classifier_dir), device)
    classifier.eval()
    for param in classifier.parameters():
        param.requires_grad = False
    env = MultiTargetAGVPathEnv(
        model_module=model_module,
        classifier=classifier,
        labels=labels,
        path_keys=path_keys,
        path_to_index=path_to_index,
        stats=stats,
        rp_coords=rp_coords,
        true_coords=true_coords,
        node_coords=node_coords,
        train_dir=eval_dir,
        device=device,
        max_edges=max_edges,
        max_targets=max_targets,
        min_targets=target_count,
        fixed_target_count=target_count,
        eta=eta,
        wknn_k=wknn_k,
        seed=seed,
    )
    return env, config


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare classifier backbones with the same PPO edge-selection policy.")
    parser.add_argument("--eval-dir", default=str(REPO_ROOT / "data" / "PPO-test_target"))
    parser.add_argument("--classifier", action="append", required=True, help="Classifier spec formatted as key=/path/to/model_dir.")
    parser.add_argument("--policy-path", required=True)
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", default=None)
    parser.add_argument("--disable-cudnn", action=argparse.BooleanOptionalAction, default=False)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    if args.target_labels is not None and len(args.target_labels) != args.target_count:
        raise ValueError(f"--target-labels must contain exactly {args.target_count} labels")
    set_seed(args.seed)
    random.seed(args.seed)
    classifier_specs = parse_classifier_specs(args.classifier)

    model_module = load_model_module()
    device = model_module.build_device(args.gpu)
    model_module.configure_torch_runtime(device, args.disable_cudnn)
    print(f"Using device: {device}", flush=True)

    rp_coords = load_coords(args.rp_coords)
    true_coords = load_coords(args.true_coords)
    node_coords = load_node_coords(args.map_node_coords)

    first_key, _, first_dir = classifier_specs[0]
    first_env, first_config = build_env(
        model_module=model_module,
        classifier_dir=first_dir,
        eval_dir=args.eval_dir,
        rp_coords=rp_coords,
        true_coords=true_coords,
        node_coords=node_coords,
        device=device,
        max_edges=int(args.max_edges or first_config.get("max_edges", 4)),
        max_targets=args.max_targets,
        target_count=args.target_count,
        eta=args.eta,
        wknn_k=args.wknn_k,
        seed=args.seed,
    )
    max_edges = int(args.max_edges or first_config.get("max_edges", 4))

    if args.episode_list:
        episode_specs = load_episode_list(args.episode_list, args.target_count)
    else:
        episode_specs = generate_episode_specs(
            env=first_env,
            episodes=args.episodes,
            target_count=args.target_count,
            target_labels=args.target_labels,
            random_start=args.random_start,
            start_node=args.start_node,
            seed=args.seed,
        )

    policy = load_policy(Path(args.policy_path), first_env, max_edges, args.max_targets, device)
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    write_episode_specs(out_dir / "classifier_episode_list.csv", episode_specs)

    frames: dict[str, pd.DataFrame] = {}
    names: dict[str, str] = {}
    all_rows: list[dict] = []
    summary_rows: list[dict] = []
    for idx, (key, name, classifier_dir) in enumerate(classifier_specs):
        print(f"Evaluating {name}: {classifier_dir}", flush=True)
        if idx == 0 and key == first_key:
            env = first_env
        else:
            env, _ = build_env(
                model_module=model_module,
                classifier_dir=classifier_dir,
                eval_dir=args.eval_dir,
                rp_coords=rp_coords,
                true_coords=true_coords,
                node_coords=node_coords,
                device=device,
                max_edges=max_edges,
                max_targets=args.max_targets,
                target_count=args.target_count,
                eta=args.eta,
                wknn_k=args.wknn_k,
                seed=args.seed,
            )
        rows = evaluate_classifier(key, name, env, policy, episode_specs, device)
        classifier_dir_out = out_dir / "classifiers" / key
        save_csv(classifier_dir_out / "eval_episodes.csv", rows)
        summary = summarize_classifier(key, name, rows)
        with (classifier_dir_out / "eval_summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        frames[key] = pd.DataFrame(rows)
        names[key] = name
        all_rows.extend(rows)
        summary_rows.append(summary)
        print(json.dumps(summary, indent=2), flush=True)

    save_csv(out_dir / "classifier_eval_episodes.csv", all_rows)
    save_csv(out_dir / "classifier_eval_summary.csv", summary_rows)
    plot_classifier_cdf(frames, names, out_dir / "classifier_error_cdf.svg")
    plot_classifier_boxplot(frames, names, out_dir / "classifier_error_boxplot.svg")
    plot_classifier_mean_error(summary_rows, out_dir / "classifier_mean_error_bar.svg")
    print(f"Saved results under {out_dir}", flush=True)


if __name__ == "__main__":
    main()

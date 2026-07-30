from __future__ import annotations

import csv
import importlib.util
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


SOFTMAX_TEMPERATURE = 2.0
REWARD_DISTANCE_NORM = 4.0
REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_CANDIDATES = [
    REPO_ROOT / "Model" / "Model.py",
    REPO_ROOT / "ILC-MG-train" / "Model.py",
]


def normalize_label_key(value: Any) -> str:
    text = str(value).strip()
    try:
        number = float(text)
        if number.is_integer():
            return str(int(number))
    except ValueError:
        pass
    return text


def natural_label_sort_key(value: Any) -> tuple[str, int | str]:
    text = normalize_label_key(value)
    try:
        return ("", int(text))
    except ValueError:
        prefix = "".join(ch for ch in text if not ch.isdigit())
        suffix = "".join(ch for ch in text if ch.isdigit())
        return (prefix, int(suffix) if suffix else text)


def load_coords(csv_path: str | Path) -> dict[str, tuple[float, float]]:
    df = pd.read_csv(csv_path)
    required = {"label", "x", "y"}
    if not required.issubset(df.columns):
        raise ValueError(f"{csv_path} missing columns {required - set(df.columns)}")
    return {
        normalize_label_key(row["label"]): (float(row["x"]), float(row["y"]))
        for _, row in df.iterrows()
    }


def wknn_estimate(
    prob: np.ndarray,
    labels: np.ndarray,
    rp_coords: dict[str, tuple[float, float]],
    k: int,
) -> tuple[float, float, list[str], list[float]]:
    k = max(1, min(int(k), prob.shape[0]))
    order = np.argsort(-prob)[:k]
    top_labels = [normalize_label_key(labels[idx]) for idx in order]
    top_probs = prob[order].astype(np.float64)
    denom = float(np.sum(top_probs))
    weights = top_probs / denom if denom > 0 else np.full_like(top_probs, 1.0 / len(top_probs))

    est_x = 0.0
    est_y = 0.0
    for label, weight in zip(top_labels, weights):
        x, y = rp_coords[label]
        est_x += float(weight) * x
        est_y += float(weight) * y
    return est_x, est_y, top_labels, [float(w) for w in weights]


@dataclass(frozen=True)
class EpisodeGraph:
    label: str
    group: str
    edges: list[Any]
    edge_by_key: dict[str, Any]
    adjacency: dict[int, list[Any]]
    nodes: list[int]


def load_model_module():
    model_file = next((path for path in MODEL_CANDIDATES if path.exists()), None)
    if model_file is None:
        raise FileNotFoundError(f"no model module found in: {MODEL_CANDIDATES}")
    spec = importlib.util.spec_from_file_location("srs_model_module", model_file)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load model module from {model_file}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["srs_model_module"] = module
    spec.loader.exec_module(module)
    return module


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_gae(
    rewards: list[float],
    values: list[float],
    dones: list[bool],
    gamma: float,
    gae_lambda: float,
    last_value: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    advantages = np.zeros((len(rewards),), dtype=np.float32)
    last_gae = 0.0
    next_value = float(last_value)
    for t in reversed(range(len(rewards))):
        non_terminal = 0.0 if dones[t] else 1.0
        delta = rewards[t] + gamma * next_value * non_terminal - values[t]
        last_gae = delta + gamma * gae_lambda * non_terminal * last_gae
        advantages[t] = last_gae
        next_value = values[t]
    returns = advantages + np.asarray(values, dtype=np.float32)
    return advantages, returns.astype(np.float32)


def save_history(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_training_curves(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    movement_distance_per_step_m = 1.7
    metrics = [
        ("mean_reward", "Mean Reward", "#1f77b4", 1.0),
        ("mean_error", "Mean Error (m)", "#d62728", 1.0),
        ("mean_edge_count", "Mean Movement Distance (m)", "#2ca02c", movement_distance_per_step_m),
    ]
    width = 980
    height = 760
    left = 82
    right = 28
    top = 48
    bottom = 58
    panel_gap = 46
    panel_h = int((height - top - bottom - panel_gap * 2) / 3)
    plot_w = width - left - right
    updates = np.asarray([row["update"] for row in rows], dtype=np.float64)
    x_min = float(np.min(updates))
    x_max = float(np.max(updates))
    if x_max <= x_min:
        x_max = x_min + 1.0

    def sx(value: float) -> float:
        return left + (float(value) - x_min) / (x_max - x_min) * plot_w

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2:.1f}" y="28" text-anchor="middle" font-size="28" font-family="Arial">PPO Training Curves</text>',
    ]

    for panel_idx, (key, label, color, scale) in enumerate(metrics):
        y_top = top + panel_idx * (panel_h + panel_gap)
        y_bottom = y_top + panel_h
        values = np.asarray([row[key] for row in rows], dtype=np.float64) * scale
        v_min = float(np.min(values))
        v_max = float(np.max(values))
        if v_max <= v_min:
            v_max = v_min + 1.0
        margin = 0.06 * (v_max - v_min)
        v_min -= margin
        v_max += margin

        def sy(value: float) -> float:
            return y_bottom - (float(value) - v_min) / (v_max - v_min) * panel_h

        lines.append(f'<text x="{left}" y="{y_top - 12}" font-size="23" font-family="Arial" fill="#111">{label}</text>')
        for i in range(5):
            y_val = v_min + (v_max - v_min) * i / 4.0
            y_pos = sy(y_val)
            lines.append(f'<line x1="{left}" y1="{y_pos:.2f}" x2="{width - right}" y2="{y_pos:.2f}" stroke="#e6e6e6" stroke-dasharray="4 4"/>')
            lines.append(f'<text x="{left - 10}" y="{y_pos + 4:.2f}" text-anchor="end" font-size="18" font-family="Arial">{y_val:.3g}</text>')
        lines.append(f'<line x1="{left}" y1="{y_bottom}" x2="{width - right}" y2="{y_bottom}" stroke="#222" stroke-width="1"/>')
        lines.append(f'<line x1="{left}" y1="{y_top}" x2="{left}" y2="{y_bottom}" stroke="#222" stroke-width="1"/>')
        points = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in zip(updates, values))
        lines.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.6" stroke-linejoin="round" stroke-linecap="round"/>')
        if panel_idx == len(metrics) - 1:
            for i in range(6):
                x_val = x_min + (x_max - x_min) * i / 5.0
                x_pos = sx(x_val)
                lines.append(f'<text x="{x_pos:.2f}" y="{y_bottom + 24}" text-anchor="middle" font-size="18" font-family="Arial">{x_val:.0f}</text>')

    lines.append(f'<text x="{left + plot_w / 2:.1f}" y="{height - 14}" text-anchor="middle" font-size="21" font-family="Arial">Update</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")

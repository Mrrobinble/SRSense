#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd


COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
ERROR_X_TICKS = [float(value) for value in range(0, 9)]
FONT_FAMILY = "Times New Roman, Times, serif"
FIG_WIDTH = 760
FIG_HEIGHT = 540
PLOT_LEFT = 118
PLOT_RIGHT = 42
PLOT_TOP = 36
PLOT_BOTTOM = 104
TICK_FONT = 32
LABEL_FONT = 40
LEGEND_FONT = 32


def method_label(target_count: int) -> str:
    return f"SRSense ({target_count} targets)"


def save_summary(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_eval_rows(eval_root: Path, target_counts: list[int]) -> tuple[dict[int, pd.DataFrame], list[dict]]:
    frames: dict[int, pd.DataFrame] = {}
    summary_rows: list[dict] = []
    for target_count in target_counts:
        csv_path = eval_root / f"targets_{target_count}" / "ppo_eval_episodes.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"missing eval result: {csv_path}")
        df = pd.read_csv(csv_path)
        frames[target_count] = df
        errors = df["error"].to_numpy(dtype=np.float64)
        edges = df["edge_count"].to_numpy(dtype=np.float64)
        rewards = df["reward"].to_numpy(dtype=np.float64)
        summary_rows.append(
            {
                "target_count": target_count,
                "episodes": int(len(df)),
                "mean_reward": float(np.mean(rewards)),
                "mean_error": float(np.mean(errors)),
                "median_error": float(np.median(errors)),
                "p75_error": float(np.quantile(errors, 0.75)),
                "p80_error": float(np.quantile(errors, 0.80)),
                "p90_error": float(np.quantile(errors, 0.90)),
                "mean_edge_count": float(np.mean(edges)),
                "std_edge_count": float(np.std(edges, ddof=1)) if len(edges) > 1 else 0.0,
            }
        )
    return frames, summary_rows


def ticks(min_value: float, max_value: float, count: int = 6) -> list[float]:
    if max_value <= min_value:
        max_value = min_value + 1.0
    return [min_value + (max_value - min_value) * idx / (count - 1) for idx in range(count)]


def smooth_empirical_cdf(errors: np.ndarray, x_min: float, x_max: float, samples: int = 360, window: int = 9) -> tuple[np.ndarray, np.ndarray]:
    """Return a monotonic lightly-smoothed empirical CDF for cleaner paper figures."""
    sorted_errors = np.sort(errors[np.isfinite(errors)])
    if sorted_errors.size == 0:
        return np.asarray([x_min, x_max]), np.asarray([0.0, 1.0])
    x_grid = np.linspace(x_min, x_max, samples)
    y = np.searchsorted(sorted_errors, x_grid, side="right").astype(np.float64) / float(sorted_errors.size)
    if window > 1:
        pad = window // 2
        kernel = np.ones(window, dtype=np.float64) / float(window)
        y = np.convolve(np.pad(y, (pad, pad), mode="edge"), kernel, mode="valid")
        y = np.maximum.accumulate(y)
        y = np.clip(y, 0.0, 1.0)
    return x_grid, y


def target_error_values(df: pd.DataFrame) -> np.ndarray:
    values: list[float] = []
    if "target_errors" not in df.columns:
        return df["error"].to_numpy(dtype=np.float64)
    for raw_value in df["target_errors"].fillna("").astype(str):
        for item in raw_value.split("|"):
            item = item.strip()
            if not item:
                continue
            values.append(float(item))
    return np.asarray(values, dtype=np.float64)


def cdf_error_values(df: pd.DataFrame, source: str) -> np.ndarray:
    if source == "target_errors":
        return target_error_values(df)
    if source == "max_target_error":
        if "target_errors" not in df.columns:
            return df["error"].to_numpy(dtype=np.float64)
        values = []
        for raw_value in df["target_errors"].fillna("").astype(str):
            errors = [float(item) for item in raw_value.split("|") if item.strip()]
            if errors:
                values.append(max(errors))
        return np.asarray(values, dtype=np.float64)
    return df["error"].to_numpy(dtype=np.float64)


def plot_error_cdf(frames: dict[int, pd.DataFrame], output_path: Path, error_source: str = "episode_mean") -> None:
    width, height = FIG_WIDTH, FIG_HEIGHT
    left, right, top, bottom = PLOT_LEFT, PLOT_RIGHT, PLOT_TOP, PLOT_BOTTOM
    plot_w, plot_h = width - left - right, height - top - bottom

    x_min = 0.0
    x_max = 8.0
    y_min, y_max = 0.0, 1.0

    def sx(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * plot_w

    def sy(value: float) -> float:
        return top + (1.0 - (value - y_min) / (y_max - y_min)) * plot_h

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
    ]

    for y_tick in ticks(0.0, 1.0):
        y = sy(y_tick)
        lines.append(f'<line x1="{left}" y1="{y:.2f}" x2="{width - right}" y2="{y:.2f}" stroke="#edf2f7" stroke-width="1.3"/>')
        lines.append(f'<text x="{left - 16}" y="{y + 8:.2f}" text-anchor="end" font-size="{TICK_FONT}" font-family="{FONT_FAMILY}">{y_tick:.1f}</text>')
    for x_tick in ERROR_X_TICKS:
        x = sx(x_tick)
        lines.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{height - bottom}" stroke="#f8fafc" stroke-width="1.3"/>')
        lines.append(f'<text x="{x:.2f}" y="{height - bottom + 38}" text-anchor="middle" font-size="{TICK_FONT}" font-family="{FONT_FAMILY}">{x_tick:.0f}</text>')

    for idx, (target_count, df) in enumerate(frames.items()):
        errors = cdf_error_values(df, error_source)
        cdf_x, cdf_y = smooth_empirical_cdf(errors, x_min, x_max)
        points = " ".join(f"{sx(float(x)):.2f},{sy(float(y)):.2f}" for x, y in zip(cdf_x, cdf_y))
        color = COLORS[idx % len(COLORS)]
        lines.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="4.0" stroke-linejoin="round" stroke-linecap="round"/>')
        legend_x = width - right - 350
        legend_y = top + 28 + idx * 34
        lines.append(f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 42}" y2="{legend_y}" stroke="{color}" stroke-width="4.0" stroke-linecap="round"/>')
        lines.append(f'<text x="{legend_x + 54}" y="{legend_y + 8}" font-size="{LEGEND_FONT}" font-family="{FONT_FAMILY}">{method_label(target_count)}</text>')

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


MOVEMENT_DISTANCE_PER_STEP_M = 1.7


def plot_movement_distance_bar(summary_rows: list[dict], output_path: Path) -> None:
    width, height = FIG_WIDTH, FIG_HEIGHT
    left, right, top, bottom = PLOT_LEFT, PLOT_RIGHT, PLOT_TOP, 132
    plot_w, plot_h = width - left - right, height - top - bottom
    target_counts = [int(row["target_count"]) for row in summary_rows]
    means = np.asarray([row["mean_edge_count"] for row in summary_rows], dtype=np.float64) * MOVEMENT_DISTANCE_PER_STEP_M
    stds = np.asarray([row["std_edge_count"] for row in summary_rows], dtype=np.float64) * MOVEMENT_DISTANCE_PER_STEP_M
    y_min = 0.0
    y_max = max(4.0, float(np.max(means + stds)) + 0.3)

    def sx(index: int) -> float:
        return left + (index + 0.5) / len(target_counts) * plot_w

    def sy(value: float) -> float:
        return top + (1.0 - (value - y_min) / (y_max - y_min)) * plot_h

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
    ]
    for y_tick in ticks(y_min, y_max):
        y = sy(y_tick)
        lines.append(f'<line x1="{left}" y1="{y:.2f}" x2="{width - right}" y2="{y:.2f}" stroke="#edf2f7" stroke-width="1.3"/>')
        lines.append(f'<text x="{left - 16}" y="{y + 8:.2f}" text-anchor="end" font-size="{TICK_FONT}" font-family="{FONT_FAMILY}">{y_tick:.1f}</text>')

    bar_w = plot_w / len(target_counts) * 0.58
    for idx, (target_count, mean, std) in enumerate(zip(target_counts, means, stds)):
        x = sx(idx)
        y = sy(float(mean))
        baseline = sy(0.0)
        color = COLORS[idx % len(COLORS)]
        lines.append(f'<rect x="{x - bar_w / 2:.2f}" y="{y:.2f}" width="{bar_w:.2f}" height="{baseline - y:.2f}" fill="{color}" fill-opacity="0.78" stroke="#1f2937" stroke-width="1.5"/>')
        y_low = sy(float(max(0.0, mean - std)))
        y_high = sy(float(mean + std))
        cap = bar_w * 0.24
        lines.append(f'<line x1="{x:.2f}" y1="{y_high:.2f}" x2="{x:.2f}" y2="{y_low:.2f}" stroke="#111827" stroke-width="2.4"/>')
        lines.append(f'<line x1="{x - cap:.2f}" y1="{y_high:.2f}" x2="{x + cap:.2f}" y2="{y_high:.2f}" stroke="#111827" stroke-width="2.4"/>')
        lines.append(f'<line x1="{x - cap:.2f}" y1="{y_low:.2f}" x2="{x + cap:.2f}" y2="{y_low:.2f}" stroke="#111827" stroke-width="2.4"/>')
        lines.append(f'<text x="{x:.2f}" y="{height - bottom + 30}" text-anchor="middle" font-size="{TICK_FONT}" font-family="{FONT_FAMILY}">SRSense</text>')
        lines.append(f'<text x="{x:.2f}" y="{height - bottom + 58}" text-anchor="middle" font-size="{TICK_FONT}" font-family="{FONT_FAMILY}">({target_count} targets)</text>')

    lines.extend(
        [
            f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#111827" stroke-width="2.6"/>',
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#111827" stroke-width="2.6"/>',
            f'<text x="{left + plot_w / 2:.1f}" y="{height - 30}" text-anchor="middle" font-size="{LABEL_FONT}" font-family="{FONT_FAMILY}">Target Setting</text>',
            f'<text x="40" y="{top + plot_h / 2 + 30:.1f}" text-anchor="middle" font-size="{LABEL_FONT}" font-family="{FONT_FAMILY}" transform="rotate(-90 40 {top + plot_h / 2 + 30:.1f})">Mean Movement Distance (m)</text>',
            "</svg>",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot multi-target PPO test comparison figures.")
    parser.add_argument("--eval-root", required=True)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--target-counts", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument(
        "--cdf-error-source",
        choices=["episode_mean", "target_errors", "max_target_error"],
        default="episode_mean",
        help="Error values used in CDF. episode_mean uses one averaged error per episode; target_errors expands each target error.",
    )
    args = parser.parse_args()

    eval_root = Path(args.eval_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else eval_root
    frames, summary_rows = load_eval_rows(eval_root, args.target_counts)
    save_summary(out_dir / "multi_target_eval_summary.csv", summary_rows)
    cdf_name = {
        "episode_mean": "multi_target_error_cdf.svg",
        "target_errors": "multi_target_individual_error_cdf.svg",
        "max_target_error": "multi_target_max_error_cdf.svg",
    }[args.cdf_error_source]
    plot_error_cdf(frames, out_dir / cdf_name, args.cdf_error_source)
    plot_movement_distance_bar(summary_rows, out_dir / "multi_target_mean_movement_distance_bar.svg")
    print(f"Saved {out_dir / 'multi_target_eval_summary.csv'}")
    print(f"Saved {out_dir / cdf_name}")
    print(f"Saved {out_dir / 'multi_target_mean_movement_distance_bar.svg'}")


if __name__ == "__main__":
    main()

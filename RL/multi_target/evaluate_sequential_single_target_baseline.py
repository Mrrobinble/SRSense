#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from RL.multi_target.common import load_coords, load_model_module, set_seed
    from RL.multi_target.evaluate_multi_target_ppo import load_episode_list, save_csv
    from RL.multi_target.multi_target_ppo_agent import MultiTargetPPOActorCritic
    from RL.multi_target.multi_target_ppo_env import MultiTargetAGVPathEnv, load_node_coords
    from RL.multi_target.train_multi_target_ppo import obs_to_tensors
else:
    from .common import load_coords, load_model_module, set_seed
    from .evaluate_multi_target_ppo import load_episode_list, save_csv
    from .multi_target_ppo_agent import MultiTargetPPOActorCritic
    from .multi_target_ppo_env import MultiTargetAGVPathEnv, load_node_coords
    from .train_multi_target_ppo import obs_to_tensors


REPO_ROOT = Path(__file__).resolve().parents[2]
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


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate sequential single-target localization baseline.")
    parser.add_argument("--eval-dir", default=str(REPO_ROOT / "data" / "PPO-test_target"))
    parser.add_argument("--classifier-dir", required=True)
    parser.add_argument("--single-policy-path", required=True)
    parser.add_argument("--episode-root", required=True, help="Evaluation root containing group_XX/episode_lists.")
    parser.add_argument("--simultaneous-root", default=None, help="Existing simultaneous PPO evaluation root for edge comparison.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--rp-coords", default=str(REPO_ROOT / "data" / "ILC-MG_rp_coords.csv"))
    parser.add_argument("--true-coords", default=str(REPO_ROOT / "data" / "PPO_target_coords.csv"))
    parser.add_argument("--map-node-coords", default=str(REPO_ROOT / "data" / "map_node_coords.csv"))
    parser.add_argument("--target-counts", nargs="+", type=int, default=[1, 2, 3, 5])
    parser.add_argument("--max-targets", type=int, default=5)
    parser.add_argument("--max-edges", type=int, default=4)
    parser.add_argument("--wknn-k", type=int, default=3)
    parser.add_argument("--eta", type=float, default=0.04)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", default=None)
    parser.add_argument("--disable-cudnn", action=argparse.BooleanOptionalAction, default=False)
    return parser


def summarize(rows: list[dict]) -> dict:
    errors = np.asarray([row["error"] for row in rows], dtype=np.float64)
    edge_counts = np.asarray([row["edge_count"] for row in rows], dtype=np.float64)
    rewards = np.asarray([row["reward"] for row in rows], dtype=np.float64)
    target_counts = np.asarray([row["target_count"] for row in rows], dtype=np.float64)
    return {
        "episodes": len(rows),
        "mean_reward": float(np.mean(rewards)) if rewards.size else 0.0,
        "mean_error": float(np.mean(errors)) if errors.size else 0.0,
        "median_error": float(np.median(errors)) if errors.size else 0.0,
        "p75_error": float(np.quantile(errors, 0.75)) if errors.size else 0.0,
        "p90_error": float(np.quantile(errors, 0.90)) if errors.size else 0.0,
        "mean_edge_count": float(np.mean(edge_counts)) if edge_counts.size else 0.0,
        "std_edge_count": float(np.std(edge_counts, ddof=1)) if edge_counts.size > 1 else 0.0,
        "mean_target_count": float(np.mean(target_counts)) if target_counts.size else 0.0,
    }


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


def run_single_episode(
    env: MultiTargetAGVPathEnv,
    policy: MultiTargetPPOActorCritic,
    target_group: str,
    start_node: int,
    device: torch.device,
) -> tuple[float, dict]:
    obs = env.reset_to([target_group], start_node)
    done = False
    total_reward = 0.0
    final_info: dict = {}
    while not done:
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
            action = int(torch.argmax(logits, dim=1).item())
        obs, reward, done, info = env.step(action)
        total_reward += float(reward)
        final_info = info
    return total_reward, final_info


def group_dirs(episode_root: Path) -> list[Path]:
    groups = sorted(path for path in episode_root.glob("group_*") if path.is_dir())
    if not groups:
        raise RuntimeError(f"no group_* directories found under {episode_root}")
    return groups


def evaluate_target_count(
    target_count: int,
    groups: list[Path],
    env: MultiTargetAGVPathEnv,
    policy: MultiTargetPPOActorCritic,
    device: torch.device,
) -> list[dict]:
    rows: list[dict] = []
    global_episode = 0
    for group_dir in groups:
        list_path = group_dir / "episode_lists" / f"nested_eval_targets_{target_count}.csv"
        specs = load_episode_list(list_path, target_count)
        fixed5_group_id = int(group_dir.name.split("_")[-1])
        for spec in specs:
            target_groups = list(spec["target_groups"])
            start_node = int(spec["start_node"])
            rewards = []
            errors = []
            edge_counts = []
            paths = []
            directions = []
            target_labels = []
            top1_labels = []
            for target_group in target_groups:
                reward, info = run_single_episode(env, policy, target_group, start_node, device)
                rewards.append(float(reward))
                errors.append(float(info.get("error", 0.0)))
                edge_counts.append(int(info.get("edge_count", 0)))
                paths.append(str(info.get("path", "")))
                directions.append(str(info.get("directions", "")))
                labels = info.get("target_labels", [])
                target_labels.append(str(labels[0]) if labels else "")
                top1_labels.append(str(info.get("top1_label", "")))

            rows.append(
                {
                    "fixed5_group_id": fixed5_group_id,
                    "episode": global_episode,
                    "source_episode": int(spec["episode"]),
                    "reward": float(np.sum(rewards)),
                    "error": float(np.mean(errors)) if errors else 0.0,
                    "edge_count": int(np.sum(edge_counts)),
                    "target_count": target_count,
                    "mean_single_edge_count": float(np.mean(edge_counts)) if edge_counts else 0.0,
                    "start_node": start_node,
                    "target_groups": "|".join(target_groups),
                    "target_labels": "|".join(target_labels),
                    "target_errors": "|".join(f"{value:.4f}" for value in errors),
                    "single_edge_counts": "|".join(str(value) for value in edge_counts),
                    "single_paths": "||".join(paths),
                    "single_directions": "||".join(directions),
                    "single_top1_labels": "|".join(top1_labels),
                }
            )
            global_episode += 1
    return rows


def save_summary_csv(path: Path, summaries: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)


MOVEMENT_DISTANCE_PER_STEP_M = 1.7


def load_movement_distance_summary(root: Path, target_counts: list[int]) -> dict[int, tuple[float, float]]:
    result = {}
    for target_count in target_counts:
        csv_path = root / f"targets_{target_count}" / "ppo_eval_episodes.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"missing evaluation episodes: {csv_path}")
        df = pd.read_csv(csv_path)
        edges = df["edge_count"].to_numpy(dtype=np.float64)
        result[target_count] = (
            float(np.mean(edges)) * MOVEMENT_DISTANCE_PER_STEP_M,
            (float(np.std(edges, ddof=1)) if edges.size > 1 else 0.0) * MOVEMENT_DISTANCE_PER_STEP_M,
        )
    return result


def y_ticks(min_value: float, max_value: float, count: int = 6) -> list[float]:
    if max_value <= min_value:
        max_value = min_value + 1.0
    return [min_value + (max_value - min_value) * idx / (count - 1) for idx in range(count)]


def plot_movement_distance_comparison(
    simultaneous: dict[int, tuple[float, float]],
    sequential: dict[int, tuple[float, float]],
    target_counts: list[int],
    output_path: Path,
) -> None:
    width, height = FIG_WIDTH, FIG_HEIGHT
    left, right, top, bottom = PLOT_LEFT, PLOT_RIGHT, PLOT_TOP, 132
    plot_w, plot_h = width - left - right, height - top - bottom
    sim_means = np.asarray([simultaneous[k][0] for k in target_counts], dtype=np.float64)
    sim_stds = np.asarray([simultaneous[k][1] for k in target_counts], dtype=np.float64)
    seq_means = np.asarray([sequential[k][0] for k in target_counts], dtype=np.float64)
    seq_stds = np.asarray([sequential[k][1] for k in target_counts], dtype=np.float64)
    y_min = 0.0
    y_max = max(4.0, float(np.max(np.concatenate([sim_means + sim_stds, seq_means + seq_stds]))) + 0.6)

    def sx(index: int) -> float:
        return left + (index + 0.5) / len(target_counts) * plot_w

    def sy(value: float) -> float:
        return top + (1.0 - (value - y_min) / (y_max - y_min)) * plot_h

    colors = {"sim": "#2563eb", "seq": "#f97316"}
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
    ]
    for tick in y_ticks(y_min, y_max):
        y = sy(tick)
        lines.append(f'<line x1="{left}" y1="{y:.2f}" x2="{width - right}" y2="{y:.2f}" stroke="#edf2f7" stroke-width="1.3"/>')
        lines.append(f'<text x="{left - 16}" y="{y + 8:.2f}" text-anchor="end" font-size="{TICK_FONT}" font-family="{FONT_FAMILY}">{tick:.1f}</text>')

    group_w = plot_w / len(target_counts)
    bar_w = group_w * 0.34
    offsets = {"sim": -bar_w * 0.58, "seq": bar_w * 0.58}
    for idx, target_count in enumerate(target_counts):
        base_x = sx(idx)
        for key, means, stds in [("sim", sim_means, sim_stds), ("seq", seq_means, seq_stds)]:
            mean = float(means[idx])
            std = float(stds[idx])
            x = base_x + offsets[key]
            y = sy(mean)
            baseline = sy(0.0)
            lines.append(
                f'<rect x="{x - bar_w / 2:.2f}" y="{y:.2f}" width="{bar_w:.2f}" height="{baseline - y:.2f}" '
                f'fill="{colors[key]}" fill-opacity="0.78" stroke="#1f2937" stroke-width="1.5"/>'
            )
            y_low = sy(max(0.0, mean - std))
            y_high = sy(mean + std)
            cap = bar_w * 0.28
            lines.append(f'<line x1="{x:.2f}" y1="{y_high:.2f}" x2="{x:.2f}" y2="{y_low:.2f}" stroke="#111827" stroke-width="2.4"/>')
            lines.append(f'<line x1="{x - cap:.2f}" y1="{y_high:.2f}" x2="{x + cap:.2f}" y2="{y_high:.2f}" stroke="#111827" stroke-width="2.4"/>')
            lines.append(f'<line x1="{x - cap:.2f}" y1="{y_low:.2f}" x2="{x + cap:.2f}" y2="{y_low:.2f}" stroke="#111827" stroke-width="2.4"/>')
        lines.append(f'<text x="{base_x:.2f}" y="{height - bottom + 30}" text-anchor="middle" font-size="{TICK_FONT}" font-family="{FONT_FAMILY}">SRSense</text>')
        lines.append(f'<text x="{base_x:.2f}" y="{height - bottom + 58}" text-anchor="middle" font-size="{TICK_FONT}" font-family="{FONT_FAMILY}">({target_count} targets)</text>')

    legend_x = left + 22
    legend_y = top + 26
    lines.append(f'<rect x="{legend_x}" y="{legend_y - 16}" width="24" height="24" fill="{colors["sim"]}" fill-opacity="0.78" stroke="#1f2937" stroke-width="1.2"/>')
    lines.append(f'<text x="{legend_x + 36}" y="{legend_y + 5}" font-size="{LEGEND_FONT}" font-family="{FONT_FAMILY}">Simultaneous</text>')
    lines.append(f'<rect x="{legend_x}" y="{legend_y + 22}" width="24" height="24" fill="{colors["seq"]}" fill-opacity="0.78" stroke="#1f2937" stroke-width="1.2"/>')
    lines.append(f'<text x="{legend_x + 36}" y="{legend_y + 43}" font-size="{LEGEND_FONT}" font-family="{FONT_FAMILY}">Sequential</text>')

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
    args = build_argparser().parse_args()
    set_seed(args.seed)

    model_module = load_model_module()
    device = model_module.build_device(args.gpu)
    model_module.configure_torch_runtime(device, args.disable_cudnn)
    print(f"Using device: {device}")

    classifier, labels, path_keys, path_to_index, stats, config = model_module.load_artifacts(args.classifier_dir, device)
    classifier.eval()
    for param in classifier.parameters():
        param.requires_grad = False

    max_edges = int(args.max_edges or config.get("max_edges", 4))
    rp_coords = load_coords(args.rp_coords)
    true_coords = load_coords(args.true_coords)
    node_coords = load_node_coords(args.map_node_coords)
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
        train_dir=args.eval_dir,
        device=device,
        max_edges=max_edges,
        max_targets=args.max_targets,
        min_targets=1,
        fixed_target_count=1,
        eta=args.eta,
        wknn_k=args.wknn_k,
        seed=args.seed,
    )
    policy = load_policy(Path(args.single_policy_path), env, max_edges, args.max_targets, device)

    episode_root = Path(args.episode_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    groups = group_dirs(episode_root)

    summary_rows = []
    for target_count in args.target_counts:
        rows = evaluate_target_count(target_count, groups, env, policy, device)
        target_dir = out_dir / f"targets_{target_count}"
        save_csv(target_dir / "ppo_eval_episodes.csv", rows)
        summary = summarize(rows)
        summary["target_count"] = target_count
        with (target_dir / "ppo_eval_summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        summary_rows.append(summary)
        print(json.dumps(summary, indent=2), flush=True)

    save_summary_csv(out_dir / "sequential_eval_summary.csv", summary_rows)
    if args.simultaneous_root:
        simultaneous = load_movement_distance_summary(Path(args.simultaneous_root).expanduser().resolve(), args.target_counts)
        sequential = load_movement_distance_summary(out_dir, args.target_counts)
        plot_movement_distance_comparison(
            simultaneous,
            sequential,
            args.target_counts,
            out_dir / "sequential_vs_simultaneous_movement_distance.svg",
        )
        print(f"Saved {out_dir / 'sequential_vs_simultaneous_edges.svg'}", flush=True)


if __name__ == "__main__":
    main()

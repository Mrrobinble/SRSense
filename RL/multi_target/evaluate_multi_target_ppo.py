#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from RL.multi_target.multi_target_ppo_agent import MultiTargetPPOActorCritic
    from RL.multi_target.multi_target_ppo_env import MultiTargetAGVPathEnv, load_node_coords
    from RL.multi_target.common import load_coords
    from RL.multi_target.train_multi_target_ppo import obs_to_tensors, set_seed
else:
    from .multi_target_ppo_agent import MultiTargetPPOActorCritic
    from .multi_target_ppo_env import MultiTargetAGVPathEnv, load_node_coords
    from .common import load_coords
    from .train_multi_target_ppo import obs_to_tensors, set_seed


REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_CANDIDATES = [
    REPO_ROOT / "Model" / "Model.py",
    REPO_ROOT / "ILC-MG-train" / "Model.py",
]


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


def save_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


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
        "mean_target_count": float(np.mean(target_counts)) if target_counts.size else 0.0,
    }


def load_episode_list(path: str | Path, target_count: int | None = None) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if target_count is not None and "target_count" in row and int(row["target_count"]) != int(target_count):
                continue
            target_groups = [item for item in str(row["target_groups"]).split("|") if item]
            rows.append(
                {
                    "episode": int(row.get("episode", len(rows))),
                    "target_count": int(row.get("target_count", len(target_groups))),
                    "start_node": int(row["start_node"]),
                    "target_groups": target_groups,
                }
            )
    if not rows:
        raise RuntimeError(f"no evaluation episodes loaded from {path}")
    return rows


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a trained multi-target PPO policy.")
    parser.add_argument("--eval-dir", default=str(REPO_ROOT / "data" / "PPO-test_target"))
    parser.add_argument("--classifier-dir", required=True)
    parser.add_argument("--policy-path", required=True)
    parser.add_argument("--rp-coords", default=str(REPO_ROOT / "data" / "ILC-MG_rp_coords.csv"))
    parser.add_argument("--true-coords", default=str(REPO_ROOT / "data" / "PPO_target_coords.csv"))
    parser.add_argument("--map-node-coords", default=str(REPO_ROOT / "data" / "map_node_coords.csv"))
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--episode-list", default=None, help="CSV with fixed start_node and target_groups for deterministic evaluation.")
    parser.add_argument("--target-count", type=int, default=1)
    parser.add_argument("--start-node", type=int, default=None, help="Fix the initial map node for every evaluation episode.")
    parser.add_argument("--max-targets", type=int, default=5)
    parser.add_argument("--min-targets", type=int, default=1)
    parser.add_argument("--max-edges", type=int, default=4)
    parser.add_argument("--wknn-k", type=int, default=3)
    parser.add_argument("--eta", type=float, default=0.04)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--gpu", default=None)
    parser.add_argument("--disable-cudnn", action=argparse.BooleanOptionalAction, default=False)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    set_seed(args.seed)
    random.seed(args.seed)

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
        min_targets=args.min_targets,
        fixed_target_count=args.target_count,
        eta=args.eta,
        wknn_k=args.wknn_k,
        seed=args.seed,
        fixed_start_node=args.start_node,
    )

    policy = MultiTargetPPOActorCritic(
        node_num=env.node_num,
        edge_num=env.edge_num,
        action_size=env.action_size,
        max_edges=max_edges,
        max_targets=args.max_targets,
    ).to(device)
    checkpoint = torch.load(args.policy_path, map_location=device)
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    policy.load_state_dict(state_dict)
    policy.eval()

    episode_specs = load_episode_list(args.episode_list, args.target_count) if args.episode_list else None
    total_episodes = len(episode_specs) if episode_specs is not None else args.episodes

    rows = []
    for episode in range(total_episodes):
        if episode_specs is None:
            obs = env.reset()
        else:
            spec = episode_specs[episode]
            obs = env.reset_to(spec["target_groups"], spec["start_node"])
        done = False
        total_reward = 0.0
        final_info = {}
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

        rows.append(
            {
                "episode": episode,
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
            }
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_csv(out_dir / "ppo_eval_episodes.csv", rows)
    summary = summarize(rows)
    with (out_dir / "ppo_eval_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from RL.multi_target.multi_target_ppo_agent import MultiTargetPPOActorCritic
    from RL.multi_target.multi_target_ppo_env import MultiTargetAGVPathEnv, load_node_coords
    from RL.multi_target.common import compute_gae, load_coords, load_model_module, plot_training_curves, save_history, set_seed
else:
    from .multi_target_ppo_agent import MultiTargetPPOActorCritic
    from .multi_target_ppo_env import MultiTargetAGVPathEnv, load_node_coords
    from .common import compute_gae, load_coords, load_model_module, plot_training_curves, save_history, set_seed


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Rollout:
    current_node: list[np.ndarray]
    edge_history: list[np.ndarray]
    signal_state: list[np.ndarray]
    rsrp_state: list[np.ndarray]
    legal_action_mask: list[np.ndarray]
    actions: list[int]
    log_probs: list[float]
    values: list[float]
    rewards: list[float]
    dones: list[bool]

    @classmethod
    def empty(cls) -> "Rollout":
        return cls([], [], [], [], [], [], [], [], [], [])


def obs_to_tensors(obs: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "current_node": torch.as_tensor(obs["current_node"], dtype=torch.long, device=device).view(1),
        "edge_history": torch.as_tensor(obs["edge_history"], dtype=torch.long, device=device).view(1, -1),
        "signal_state": torch.as_tensor(obs["signal_state"], dtype=torch.float32, device=device).unsqueeze(0),
        "rsrp_state": torch.as_tensor(obs["rsrp_state"], dtype=torch.float32, device=device).unsqueeze(0),
    }


def append_obs(rollout: Rollout, obs: dict[str, np.ndarray], legal_action_mask: np.ndarray) -> None:
    rollout.current_node.append(np.asarray(obs["current_node"], dtype=np.int64))
    rollout.edge_history.append(np.asarray(obs["edge_history"], dtype=np.int64))
    rollout.signal_state.append(np.asarray(obs["signal_state"], dtype=np.float32))
    rollout.rsrp_state.append(np.asarray(obs["rsrp_state"], dtype=np.float32))
    rollout.legal_action_mask.append(np.asarray(legal_action_mask, dtype=np.float32))


def stack_rollout(rollout: Rollout, advantages: np.ndarray, returns: np.ndarray) -> dict[str, torch.Tensor]:
    data = {
        "current_node": torch.as_tensor(np.stack(rollout.current_node), dtype=torch.long).view(-1),
        "edge_history": torch.as_tensor(np.stack(rollout.edge_history), dtype=torch.long),
        "signal_state": torch.as_tensor(np.stack(rollout.signal_state), dtype=torch.float32),
        "rsrp_state": torch.as_tensor(np.stack(rollout.rsrp_state), dtype=torch.float32),
        "legal_action_mask": torch.as_tensor(np.stack(rollout.legal_action_mask), dtype=torch.float32),
        "actions": torch.as_tensor(np.asarray(rollout.actions), dtype=torch.long),
        "old_log_probs": torch.as_tensor(np.asarray(rollout.log_probs), dtype=torch.float32),
        "advantages": torch.as_tensor(advantages, dtype=torch.float32),
        "returns": torch.as_tensor(returns, dtype=torch.float32),
    }
    adv = data["advantages"]
    data["advantages"] = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
    return data


def _parse_history_value(value: str):
    text = str(value).strip()
    if text == "":
        return text
    try:
        number = float(text)
    except ValueError:
        return text
    if number.is_integer() and "." not in text and "e" not in text.lower():
        return int(number)
    return number


def load_history(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return [
            {key: _parse_history_value(value) for key, value in row.items()}
            for row in csv.DictReader(f)
        ]


def ppo_update(
    policy: MultiTargetPPOActorCritic,
    optimizer: torch.optim.Optimizer,
    data: dict[str, torch.Tensor],
    batch_size: int,
    update_epochs: int,
    clip_coef: float,
    value_coef: float,
    entropy_coef: float,
    max_grad_norm: float,
) -> dict[str, float]:
    total = data["actions"].shape[0]
    indices = np.arange(total)
    device = next(policy.parameters()).device
    metrics = {"policy_loss": [], "value_loss": [], "entropy": [], "loss": []}
    for _ in range(update_epochs):
        np.random.shuffle(indices)
        for start in range(0, total, batch_size):
            idx_cpu = torch.as_tensor(indices[start : start + batch_size], dtype=torch.long)
            batch = {key: value[idx_cpu].to(device, non_blocking=True) for key, value in data.items()}
            log_probs, entropy, values = policy.evaluate_actions(
                batch["current_node"],
                batch["edge_history"],
                batch["signal_state"],
                batch["rsrp_state"],
                batch["legal_action_mask"],
                batch["actions"],
            )
            ratio = torch.exp(log_probs - batch["old_log_probs"])
            unclipped = ratio * batch["advantages"]
            clipped = torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef) * batch["advantages"]
            policy_loss = -torch.min(unclipped, clipped).mean()
            value_loss = F.mse_loss(values, batch["returns"])
            entropy_loss = entropy.mean()
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()

            metrics["policy_loss"].append(float(policy_loss.item()))
            metrics["value_loss"].append(float(value_loss.item()))
            metrics["entropy"].append(float(entropy_loss.item()))
            metrics["loss"].append(float(loss.item()))
    return {key: float(np.mean(values)) if values else 0.0 for key, values in metrics.items()}


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train PPO for multi-target AGV SRS path selection.")
    parser.add_argument("--train-dir", default=str(REPO_ROOT / "data" / "PPO-train_target"))
    parser.add_argument("--classifier-dir", required=True)
    parser.add_argument("--rp-coords", default=str(REPO_ROOT / "data" / "ILC-MG_rp_coords.csv"))
    parser.add_argument("--true-coords", default=str(REPO_ROOT / "data" / "PPO_target_coords.csv"))
    parser.add_argument("--map-node-coords", default=str(REPO_ROOT / "data" / "map_node_coords.csv"))
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "runs_multi_target_ppo"))
    parser.add_argument("--max-targets", type=int, default=5)
    parser.add_argument("--min-targets", type=int, default=1)
    parser.add_argument("--target-count", type=int, default=None)
    parser.add_argument(
        "--random-target-count",
        action="store_true",
        help="Sample target count uniformly from [min_targets, max_targets]. By default target_count must be fixed.",
    )
    parser.add_argument("--max-edges", type=int, default=4)
    parser.add_argument("--wknn-k", type=int, default=3)
    parser.add_argument("--eta", type=float, default=0.04)
    parser.add_argument("--total-updates", type=int, default=500)
    parser.add_argument("--rollout-steps", type=int, default=2048)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.1)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.003)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", default=None)
    parser.add_argument("--disable-cudnn", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--resume", action="store_true", help="Resume from out-dir/last_policy.pt and append out-dir/ppo_history.csv.")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    if args.random_target_count:
        raise ValueError(
            "random target-count training is disabled because target count is no longer part of the observation. "
            "Train separate fixed-count agents with --target-count."
        )
    else:
        if args.target_count is None:
            raise ValueError("fixed-target training requires --target-count. Use --random-target-count for old random-count mode.")
        fixed_target_count = int(args.target_count)
        if fixed_target_count < 1 or fixed_target_count > args.max_targets:
            raise ValueError(f"--target-count must be in [1, {args.max_targets}], got {fixed_target_count}")
        args.min_targets = fixed_target_count
        args.max_targets = max(args.max_targets, fixed_target_count)
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
        train_dir=args.train_dir,
        device=device,
        max_edges=max_edges,
        max_targets=args.max_targets,
        min_targets=args.min_targets,
        fixed_target_count=fixed_target_count,
        eta=args.eta,
        wknn_k=args.wknn_k,
        seed=args.seed,
    )

    policy = MultiTargetPPOActorCritic(
        node_num=env.node_num,
        edge_num=env.edge_num,
        action_size=env.action_size,
        max_edges=max_edges,
        max_targets=args.max_targets,
    ).to(device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    history_path = out_dir / "ppo_history.csv"
    last_policy_path = out_dir / "last_policy.pt"
    best_policy_path = out_dir / "best_policy.pt"

    history_rows: list[dict] = []
    start_update = 1
    best_mean_reward = -float("inf")
    if args.resume:
        if not last_policy_path.exists():
            raise FileNotFoundError(f"--resume requested but missing {last_policy_path}")
        checkpoint = torch.load(last_policy_path, map_location=device)
        state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
        policy.load_state_dict(state_dict)
        history_rows = load_history(history_path)
        if history_rows:
            start_update = int(history_rows[-1]["update"]) + 1
            best_mean_reward = max(float(row.get("mean_reward", -float("inf"))) for row in history_rows)
        print(f"Resuming from update {start_update}/{args.total_updates}", flush=True)

    with (out_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(
            vars(args) | {"node_num": env.node_num, "edge_num": env.edge_num, "action_size": env.action_size},
            f,
            indent=2,
        )

    train_start_time = time.perf_counter()

    obs = env.reset()
    for update in range(start_update, args.total_updates + 1):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        update_start_time = time.perf_counter()
        rollout = Rollout.empty()
        episode_rewards = []
        episode_errors = []
        episode_edge_counts = []
        episode_lengths = []
        episode_target_counts = []
        current_episode_reward = 0.0
        current_episode_len = 0

        while len(rollout.rewards) < args.rollout_steps or current_episode_len > 0:
            legal_action_mask = env.action_mask()
            append_obs(rollout, obs, legal_action_mask)
            tensors = obs_to_tensors(obs, device)
            tensors["legal_action_mask"] = torch.as_tensor(legal_action_mask, dtype=torch.float32, device=device).view(1, -1)
            with torch.no_grad():
                action, log_prob, value = policy.act(**tensors)

            next_obs, reward, done, info = env.step(int(action.item()))
            rollout.actions.append(int(action.item()))
            rollout.log_probs.append(float(log_prob.item()))
            rollout.values.append(float(value.item()))
            rollout.rewards.append(float(reward))
            rollout.dones.append(bool(done))

            current_episode_reward += float(reward)
            current_episode_len += 1

            if done:
                episode_rewards.append(current_episode_reward)
                episode_lengths.append(current_episode_len)
                if "error" in info:
                    episode_errors.append(float(info["error"]))
                    episode_edge_counts.append(int(info["edge_count"]))
                    episode_target_counts.append(int(info.get("target_count", 0)))
                current_episode_reward = 0.0
                current_episode_len = 0
                obs = env.reset()
            else:
                obs = next_obs

        last_value = 0.0
        advantages, returns = compute_gae(
            rollout.rewards,
            rollout.values,
            rollout.dones,
            args.gamma,
            args.gae_lambda,
            last_value=last_value,
        )
        data = stack_rollout(rollout, advantages, returns)
        update_metrics = ppo_update(
            policy=policy,
            optimizer=optimizer,
            data=data,
            batch_size=args.batch_size,
            update_epochs=args.ppo_epochs,
            clip_coef=args.clip_coef,
            value_coef=args.value_coef,
            entropy_coef=args.entropy_coef,
            max_grad_norm=args.max_grad_norm,
        )

        row = {
            "update": update,
            "episodes": len(episode_rewards),
            "mean_reward": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
            "mean_error": float(np.mean(episode_errors)) if episode_errors else 0.0,
            "median_error": float(np.median(episode_errors)) if episode_errors else 0.0,
            "mean_edge_count": float(np.mean(episode_edge_counts)) if episode_edge_counts else 0.0,
            "mean_target_count": float(np.mean(episode_target_counts)) if episode_target_counts else 0.0,
            "mean_episode_length": float(np.mean(episode_lengths)) if episode_lengths else 0.0,
            "policy_loss": update_metrics["policy_loss"],
            "value_loss": update_metrics["value_loss"],
            "entropy": update_metrics["entropy"],
            "loss": update_metrics["loss"],
            "eta": args.eta,
        }
        update_time = time.perf_counter() - update_start_time
        elapsed_time = time.perf_counter() - train_start_time
        mean_update_time = elapsed_time / max(update, 1)
        eta_seconds = mean_update_time * max(args.total_updates - update, 0)
        row["update_time_sec"] = update_time
        row["elapsed_time_sec"] = elapsed_time
        row["eta_sec"] = eta_seconds
        if device.type == "cuda":
            row["gpu_peak_allocated_mb"] = torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
            row["gpu_peak_reserved_mb"] = torch.cuda.max_memory_reserved(device) / (1024.0 * 1024.0)
        else:
            row["gpu_peak_allocated_mb"] = 0.0
            row["gpu_peak_reserved_mb"] = 0.0
        history_rows.append(row)
        save_history(out_dir / "ppo_history.csv", history_rows)
        plot_training_curves(out_dir / "ppo_training_curves.svg", history_rows)

        torch.save({"state_dict": policy.state_dict(), "config": vars(args)}, last_policy_path)
        if row["mean_reward"] > best_mean_reward and episode_rewards:
            best_mean_reward = row["mean_reward"]
            torch.save({"state_dict": policy.state_dict(), "config": vars(args)}, best_policy_path)

        print(
            f"Update {update}/{args.total_updates} "
            f"reward={row['mean_reward']:.4f} "
            f"error={row['mean_error']:.4f} "
            f"edges={row['mean_edge_count']:.2f} "
            f"targets={row['mean_target_count']:.2f} "
            f"entropy={row['entropy']:.4f} "
            f"episodes={row['episodes']} "
            f"eta={args.eta:.4f} "
            f"time={update_time:.2f}s "
            f"remain={eta_seconds / 60.0:.1f}min"
        )


if __name__ == "__main__":
    main()

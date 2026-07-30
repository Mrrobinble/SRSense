#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from RL.multi_target.common import compute_gae, load_coords, load_model_module, plot_training_curves, save_history, set_seed
    from RL.multi_target.multi_target_ppo_agent import MultiTargetPPOActorCritic
    from RL.multi_target.multi_target_ppo_env import MultiTargetAGVPathEnv, load_node_coords
    from RL.multi_target.train_multi_target_ppo import Rollout, append_obs, obs_to_tensors, stack_rollout
else:
    from .common import compute_gae, load_coords, load_model_module, plot_training_curves, save_history, set_seed
    from .multi_target_ppo_agent import MultiTargetPPOActorCritic
    from .multi_target_ppo_env import MultiTargetAGVPathEnv, load_node_coords
    from .train_multi_target_ppo import Rollout, append_obs, obs_to_tensors, stack_rollout


REPO_ROOT = Path(__file__).resolve().parents[2]


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
        return [{key: _parse_history_value(value) for key, value in row.items()} for row in csv.DictReader(f)]


def a2c_update(
    policy: MultiTargetPPOActorCritic,
    optimizer: torch.optim.Optimizer,
    data: dict[str, torch.Tensor],
    value_coef: float,
    entropy_coef: float,
    max_grad_norm: float,
) -> dict[str, float]:
    device = next(policy.parameters()).device
    batch = {key: value.to(device, non_blocking=True) for key, value in data.items()}
    log_probs, entropy, values = policy.evaluate_actions(
        batch["current_node"],
        batch["edge_history"],
        batch["signal_state"],
        batch["rsrp_state"],
        batch["legal_action_mask"],
        batch["actions"],
    )
    policy_loss = -(log_probs * batch["advantages"]).mean()
    value_loss = F.mse_loss(values, batch["returns"])
    entropy_value = entropy.mean()
    loss = policy_loss + value_coef * value_loss - entropy_coef * entropy_value

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
    optimizer.step()
    return {
        "policy_loss": float(policy_loss.item()),
        "value_loss": float(value_loss.item()),
        "entropy": float(entropy_value.item()),
        "loss": float(loss.item()),
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train A2C for fixed-count multi-target AGV SRS path selection.")
    parser.add_argument("--train-dir", default=str(REPO_ROOT / "data" / "PPO-train_target"))
    parser.add_argument("--classifier-dir", required=True)
    parser.add_argument("--rp-coords", default=str(REPO_ROOT / "data" / "ILC-MG_rp_coords.csv"))
    parser.add_argument("--true-coords", default=str(REPO_ROOT / "data" / "PPO_target_coords.csv"))
    parser.add_argument("--map-node-coords", default=str(REPO_ROOT / "data" / "map_node_coords.csv"))
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--target-count", type=int, default=3)
    parser.add_argument("--max-targets", type=int, default=5)
    parser.add_argument("--max-edges", type=int, default=4)
    parser.add_argument("--wknn-k", type=int, default=3)
    parser.add_argument("--eta", type=float, default=0.04)
    parser.add_argument("--total-updates", type=int, default=500)
    parser.add_argument("--rollout-steps", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.003)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", default=None)
    parser.add_argument("--disable-cudnn", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    if args.target_count < 1 or args.target_count > args.max_targets:
        raise ValueError(f"--target-count must be in [1, {args.max_targets}], got {args.target_count}")
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
        train_dir=args.train_dir,
        device=device,
        max_edges=max_edges,
        max_targets=args.max_targets,
        min_targets=args.target_count,
        fixed_target_count=args.target_count,
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
    history_path = out_dir / "a2c_history.csv"
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
        json.dump(vars(args) | {"node_num": env.node_num, "edge_num": env.edge_num, "action_size": env.action_size}, f, indent=2)

    train_start_time = time.perf_counter()
    obs = env.reset()
    for update in range(start_update, args.total_updates + 1):
        update_start_time = time.perf_counter()
        rollout = Rollout.empty()
        episode_rewards: list[float] = []
        episode_errors: list[float] = []
        episode_edge_counts: list[int] = []
        episode_lengths: list[int] = []
        episode_target_counts: list[int] = []
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

        advantages, returns = compute_gae(rollout.rewards, rollout.values, rollout.dones, args.gamma, args.gae_lambda)
        data = stack_rollout(rollout, advantages, returns)
        update_metrics = a2c_update(policy, optimizer, data, args.value_coef, args.entropy_coef, args.max_grad_norm)

        update_time = time.perf_counter() - update_start_time
        elapsed_time = time.perf_counter() - train_start_time
        mean_update_time = elapsed_time / max(update - start_update + 1, 1)
        eta_seconds = mean_update_time * max(args.total_updates - update, 0)
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
            "update_time_sec": update_time,
            "elapsed_time_sec": elapsed_time,
            "eta_sec": eta_seconds,
        }
        history_rows.append(row)
        save_history(history_path, history_rows)
        plot_training_curves(out_dir / "a2c_training_curves.svg", history_rows)
        torch.save({"state_dict": policy.state_dict(), "config": vars(args)}, last_policy_path)
        if row["mean_reward"] > best_mean_reward and episode_rewards:
            best_mean_reward = row["mean_reward"]
            torch.save({"state_dict": policy.state_dict(), "config": vars(args)}, best_policy_path)

        print(
            f"Update {update}/{args.total_updates} "
            f"reward={row['mean_reward']:.4f} "
            f"error={row['mean_error']:.4f} "
            f"edges={row['mean_edge_count']:.2f} "
            f"entropy={row['entropy']:.4f} "
            f"episodes={row['episodes']} "
            f"time={update_time:.2f}s "
            f"remain={eta_seconds / 60.0:.1f}min",
            flush=True,
        )


if __name__ == "__main__":
    main()

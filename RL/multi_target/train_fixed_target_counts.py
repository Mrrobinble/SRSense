#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_SCRIPT = REPO_ROOT / "RL" / "multi_target" / "train_multi_target_ppo.py"


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train separate multi-target PPO agents for fixed target counts.")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--train-dir", default=str(REPO_ROOT / "data" / "PPO-train_target"))
    parser.add_argument("--classifier-dir", required=True)
    parser.add_argument("--rp-coords", default=str(REPO_ROOT / "data" / "ILC-MG_rp_coords.csv"))
    parser.add_argument("--true-coords", default=str(REPO_ROOT / "data" / "PPO_target_coords.csv"))
    parser.add_argument("--map-node-coords", default=str(REPO_ROOT / "data" / "map_node_coords.csv"))
    parser.add_argument("--root-out-dir", default=str(REPO_ROOT / "runs_multi_target_fixed"))
    parser.add_argument("--target-counts", nargs="+", type=int, default=[2, 3, 4, 5])
    parser.add_argument("--max-targets", type=int, default=5)
    parser.add_argument("--max-edges", type=int, default=4)
    parser.add_argument("--wknn-k", type=int, default=3)
    parser.add_argument("--eta", type=float, default=0.045)
    parser.add_argument("--total-updates", type=int, default=500)
    parser.add_argument("--rollout-steps", type=int, default=2048)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--clip-coef", type=float, default=0.1)
    parser.add_argument("--entropy-coef", type=float, default=0.003)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", default=None)
    parser.add_argument("--disable-cudnn", action=argparse.BooleanOptionalAction, default=False)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    root_out_dir = Path(args.root_out_dir).expanduser().resolve()
    root_out_dir.mkdir(parents=True, exist_ok=True)

    for target_count in args.target_counts:
        out_dir = root_out_dir / f"targets_{target_count}" / f"d4_eta{args.eta:g}_gpu{args.gpu or 'auto'}"
        command = [
            args.python,
            str(TRAIN_SCRIPT),
            "--train-dir",
            args.train_dir,
            "--classifier-dir",
            args.classifier_dir,
            "--rp-coords",
            args.rp_coords,
            "--true-coords",
            args.true_coords,
            "--map-node-coords",
            args.map_node_coords,
            "--out-dir",
            str(out_dir),
            "--target-count",
            str(target_count),
            "--max-targets",
            str(args.max_targets),
            "--max-edges",
            str(args.max_edges),
            "--wknn-k",
            str(args.wknn_k),
            "--eta",
            str(args.eta),
            "--total-updates",
            str(args.total_updates),
            "--rollout-steps",
            str(args.rollout_steps),
            "--ppo-epochs",
            str(args.ppo_epochs),
            "--batch-size",
            str(args.batch_size),
            "--lr",
            str(args.lr),
            "--clip-coef",
            str(args.clip_coef),
            "--entropy-coef",
            str(args.entropy_coef),
            "--seed",
            str(args.seed),
        ]
        if args.gpu is not None:
            command.extend(["--gpu", str(args.gpu)])
        if args.disable_cudnn:
            command.append("--disable-cudnn")

        print("Running:", " ".join(command), flush=True)
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()

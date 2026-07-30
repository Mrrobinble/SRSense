#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import random
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from RL.multi_target.multi_target_ppo_env import MultiTargetAGVPathEnv, load_node_coords
    from RL.multi_target.common import load_coords
else:
    from .multi_target_ppo_env import MultiTargetAGVPathEnv, load_node_coords
    from .common import load_coords


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


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate nested fixed multi-target PPO evaluation episode lists.")
    parser.add_argument("--eval-dir", default=str(REPO_ROOT / "data" / "PPO-test_target"))
    parser.add_argument("--classifier-dir", required=True)
    parser.add_argument("--rp-coords", default=str(REPO_ROOT / "data" / "ILC-MG_rp_coords.csv"))
    parser.add_argument("--true-coords", default=str(REPO_ROOT / "data" / "PPO_target_coords.csv"))
    parser.add_argument("--map-node-coords", default=str(REPO_ROOT / "data" / "map_node_coords.csv"))
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--episodes", type=int, default=4500)
    parser.add_argument("--start-node", type=int, default=21)
    parser.add_argument("--random-start", action="store_true", help="Randomly choose a valid shared start node for each episode.")
    parser.add_argument("--target-labels", nargs="+", default=None, help="Fix the max-target TP labels, e.g. T09 T10 T11 T12 T13.")
    parser.add_argument("--max-targets", type=int, default=5)
    parser.add_argument("--max-edges", type=int, default=4)
    parser.add_argument("--wknn-k", type=int, default=3)
    parser.add_argument("--eta", type=float, default=0.04)
    parser.add_argument("--seed", type=int, default=20260618)
    parser.add_argument("--gpu", default=None)
    parser.add_argument("--disable-cudnn", action=argparse.BooleanOptionalAction, default=False)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    rng = random.Random(args.seed)

    model_module = load_model_module()
    device = model_module.build_device(args.gpu)
    model_module.configure_torch_runtime(device, args.disable_cudnn)

    classifier, labels, path_keys, path_to_index, stats, config = model_module.load_artifacts(args.classifier_dir, device)
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
        min_targets=args.max_targets,
        fixed_target_count=args.max_targets,
        eta=args.eta,
        wknn_k=args.wknn_k,
        seed=args.seed,
        fixed_start_node=None if args.random_start else args.start_node,
    )

    fixed_labels = args.target_labels
    if fixed_labels is not None:
        if len(fixed_labels) != args.max_targets:
            raise ValueError(f"--target-labels must contain exactly {args.max_targets} labels")
        missing_labels = [label for label in fixed_labels if label not in env.graphs_by_label]
        if missing_labels:
            raise KeyError(f"target labels missing from evaluation data: {missing_labels}")

    rows_by_count = {target_count: [] for target_count in range(1, args.max_targets + 1)}
    attempts = 0
    while len(rows_by_count[args.max_targets]) < args.episodes:
        attempts += 1
        if attempts > args.episodes * 200:
            raise RuntimeError("failed to generate enough fixed evaluation episodes")

        labels_sample = list(fixed_labels) if fixed_labels is not None else rng.sample(env.available_labels, args.max_targets)
        graphs = [rng.choice(env.graphs_by_label[label]) for label in labels_sample]
        common_nodes = set(graphs[0].nodes)
        for graph in graphs[1:]:
            common_nodes &= set(graph.nodes)

        if args.random_start:
            start_candidates = list(common_nodes)
            rng.shuffle(start_candidates)
        else:
            start_candidates = [args.start_node] if args.start_node in common_nodes else []

        chosen_start = None
        for start_node in start_candidates:
            valid = True
            for target_count in range(1, args.max_targets + 1):
                env.used_edge_keys = set()
                if not env._candidate_direction_edges(graphs[:target_count], start_node):
                    valid = False
                    break
            if valid:
                chosen_start = int(start_node)
                break
        if chosen_start is None:
            continue

        episode_idx = len(rows_by_count[args.max_targets])
        for target_count in range(1, args.max_targets + 1):
            target_graphs = graphs[:target_count]
            rows_by_count[target_count].append(
                {
                    "episode": episode_idx,
                    "target_count": target_count,
                    "start_node": chosen_start,
                    "target_labels": "|".join(graph.label for graph in target_graphs),
                    "target_groups": "|".join(graph.group for graph in target_graphs),
                }
            )

    out_dir = Path(args.out_dir)
    for target_count, rows in rows_by_count.items():
        path = out_dir / f"nested_eval_targets_{target_count}.csv"
        write_rows(path, rows)
        print(f"Saved {path} ({len(rows)} episodes)")


if __name__ == "__main__":
    main()

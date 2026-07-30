from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .common import (
    EpisodeGraph,
    REWARD_DISTANCE_NORM,
    SOFTMAX_TEMPERATURE,
    natural_label_sort_key,
    normalize_label_key,
    wknn_estimate,
)


@dataclass(frozen=True)
class TargetPrediction:
    label: str
    group: str
    error: float
    est_x: float
    est_y: float
    true_x: float
    true_y: float
    top1_label: str
    top1_weight: float


DIRECTION_ACTIONS = {
    0: "stop",
    1: "left_up",
    2: "right_up",
    3: "left_down",
    4: "right_down",
}
DIRECTION_TO_ACTION = {name: action for action, name in DIRECTION_ACTIONS.items()}
DIRECTION_ACTION_SIZE = len(DIRECTION_ACTIONS)


def load_node_coords(csv_path: str | Path) -> dict[int, tuple[float, float]]:
    df = pd.read_csv(csv_path)
    node_column = "node" if "node" in df.columns else "label" if "label" in df.columns else None
    if node_column is None or not {"x", "y"}.issubset(df.columns):
        raise ValueError(f"{csv_path} must contain node,x,y columns")
    coords: dict[int, tuple[float, float]] = {}
    for _, row in df.iterrows():
        coords[int(row[node_column])] = (float(row["x"]), float(row["y"]))
    return coords


def default_map_node_coords() -> dict[int, tuple[float, float]]:
    coords = {
        1: (0.0, 0.0),
        2: (2.4, 0.0),
        3: (4.8, 0.0),
        4: (7.2, 0.0),
        5: (0.0, 2.4),
        6: (2.4, 2.4),
        7: (4.8, 2.4),
        8: (7.2, 2.4),
        9: (0.0, 4.8),
        10: (2.4, 4.8),
        11: (4.8, 4.8),
        12: (7.2, 4.8),
        13: (0.0, 7.2),
        14: (2.4, 7.2),
        15: (4.8, 7.2),
        16: (7.2, 7.2),
    }
    coords.update(
        {
            17: (1.2, 1.2),
            18: (3.6, 1.2),
            19: (6.0, 1.2),
            20: (1.2, 3.6),
            21: (3.6, 3.6),
            22: (6.0, 3.6),
            23: (1.2, 6.0),
            24: (3.6, 6.0),
            25: (6.0, 6.0),
        }
    )
    return coords


class MultiTargetAGVPathEnv:
    def __init__(
        self,
        model_module,
        classifier,
        labels: np.ndarray,
        path_keys: np.ndarray,
        path_to_index: dict[str, int],
        stats,
        rp_coords: dict[str, tuple[float, float]],
        true_coords: dict[str, tuple[float, float]],
        train_dir: str | Path,
        device: torch.device,
        node_coords: dict[int, tuple[float, float]] | None = None,
        max_edges: int = 4,
        max_targets: int = 5,
        min_targets: int = 1,
        fixed_target_count: int | None = None,
        eta: float = 0.04,
        wknn_k: int = 3,
        distance_norm: float | None = None,
        seed: int = 42,
        allow_edge_repeat: bool = False,
        max_reset_attempts: int = 256,
        fixed_start_node: int | None = None,
    ):
        self.model_module = model_module
        self.classifier = classifier
        self.labels = labels
        self.path_keys = [str(key) for key in path_keys]
        self.path_to_index = path_to_index
        self.stats = stats
        self.rp_coords = rp_coords
        self.true_coords = true_coords
        self.node_coords = node_coords if node_coords is not None else default_map_node_coords()
        self.train_dir = Path(train_dir)
        self.device = device
        self.max_edges = int(max_edges)
        self.max_targets = int(max_targets)
        self.min_targets = int(min_targets)
        self.fixed_target_count = int(fixed_target_count) if fixed_target_count is not None else None
        self.eta = float(eta)
        self.wknn_k = int(wknn_k)
        self.distance_norm = float(distance_norm) if distance_norm is not None else REWARD_DISTANCE_NORM
        self.rng = random.Random(seed)
        self.allow_edge_repeat = allow_edge_repeat
        self.max_reset_attempts = int(max_reset_attempts)
        self.fixed_start_node = int(fixed_start_node) if fixed_start_node is not None else None

        self.edge_num = len(self.path_keys)
        self.action_size = DIRECTION_ACTION_SIZE
        self.pad_edge_id = self.edge_num
        self.node_ids = sorted(self._collect_known_nodes())
        if not self.node_ids:
            raise RuntimeError("no graph nodes found from path keys")
        self.node_to_index = {node: idx for idx, node in enumerate(self.node_ids)}

        self.episode_graphs = self._load_episode_graphs()
        if not self.episode_graphs:
            raise RuntimeError(f"no valid PPO episode graphs found under {self.train_dir}")
        self.graphs_by_label: dict[str, list[EpisodeGraph]] = {}
        self.graphs_by_group: dict[str, EpisodeGraph] = {}
        for graph in self.episode_graphs:
            self.graphs_by_label.setdefault(graph.label, []).append(graph)
            self.graphs_by_group[graph.group] = graph
        self.available_labels = sorted(self.graphs_by_label.keys(), key=natural_label_sort_key)
        if not self.available_labels:
            raise RuntimeError(f"no valid target labels found under {self.train_dir}")

        self.target_graphs: list[EpisodeGraph] = []
        self.target_edge_histories: list[list[Any]] = []
        self.current_node: int | None = None
        self.edge_history_keys: list[str] = []
        self.direction_history: list[str] = []
        self.used_edge_keys: set[str] = set()
        self.done = False
        self.last_info: dict[str, Any] = {}
        self.start_node: int | None = None
        self.target_count = 0
        self._prediction_cache: dict[tuple[str, tuple[str, ...]], TargetPrediction] = {}

    @property
    def node_num(self) -> int:
        return len(self.node_ids)

    def _collect_known_nodes(self) -> set[int]:
        nodes: set[int] = set()
        for key in self.path_keys:
            try:
                a_str, b_str = key.split("-")
                nodes.add(int(a_str))
                nodes.add(int(b_str))
            except ValueError:
                continue
        return nodes

    def _load_episode_graphs(self) -> list[EpisodeGraph]:
        graphs: list[EpisodeGraph] = []
        for label_dir in self.model_module.iter_label_dirs(str(self.train_dir)):
            label = label_dir.name
            for turn_dir in self.model_module.iter_turn_dirs(label_dir):
                edges = [
                    edge for edge in self.model_module.collect_edges(turn_dir)
                    if edge.key in self.path_to_index
                ]
                if not edges:
                    continue
                adjacency: dict[int, list[Any]] = {}
                edge_by_key: dict[str, Any] = {}
                nodes: set[int] = set()
                for edge in edges:
                    edge_by_key[edge.key] = edge
                    adjacency.setdefault(edge.a, []).append(edge)
                    adjacency.setdefault(edge.b, []).append(edge)
                    nodes.add(edge.a)
                    nodes.add(edge.b)
                if nodes:
                    graphs.append(
                        EpisodeGraph(
                            label=label,
                            group=f"{label}/{turn_dir.name}",
                            edges=edges,
                            edge_by_key=edge_by_key,
                            adjacency=adjacency,
                            nodes=sorted(nodes),
                        )
                    )
        return graphs

    def set_eta(self, eta: float) -> None:
        self.eta = float(eta)

    def _edge_action_id(self, edge_key: str) -> int:
        return int(self.path_to_index[edge_key])

    def _candidate_edge_keys(self, graphs: list[EpisodeGraph], current_node: int) -> set[str]:
        candidate: set[str] | None = None
        for graph in graphs:
            keys = {
                edge.key
                for edge in graph.adjacency.get(current_node, [])
                if self.allow_edge_repeat or edge.key not in self.used_edge_keys
            }
            candidate = keys if candidate is None else candidate & keys
            if not candidate:
                return set()
        return candidate or set()

    def _direction_for_neighbor(self, current_node: int, next_node: int) -> str | None:
        if current_node not in self.node_coords or next_node not in self.node_coords:
            return None
        x0, y0 = self.node_coords[current_node]
        x1, y1 = self.node_coords[next_node]
        dx = x1 - x0
        dy = y1 - y0
        eps = 1e-9
        if abs(dx) <= eps or abs(dy) <= eps:
            return None
        if dx < 0 and dy > 0:
            return "left_up"
        if dx > 0 and dy > 0:
            return "right_up"
        if dx < 0 and dy < 0:
            return "left_down"
        if dx > 0 and dy < 0:
            return "right_down"
        return None

    def _edge_next_node(self, edge: Any, current_node: int) -> int:
        return edge.b if current_node == edge.a else edge.a

    def _candidate_direction_edges(self, graphs: list[EpisodeGraph], current_node: int) -> dict[int, str]:
        candidate_keys = self._candidate_edge_keys(graphs, current_node)
        action_to_edge: dict[int, str] = {}
        for edge_key in sorted(candidate_keys):
            edge = graphs[0].edge_by_key.get(edge_key)
            if edge is None or current_node not in {edge.a, edge.b}:
                continue
            next_node = self._edge_next_node(edge, current_node)
            direction = self._direction_for_neighbor(current_node, next_node)
            if direction is None:
                continue
            action = DIRECTION_TO_ACTION[direction]
            action_to_edge.setdefault(action, edge_key)
        return action_to_edge

    def _sample_episode(self) -> tuple[list[EpisodeGraph], int]:
        if self.fixed_target_count is None:
            target_count = self.rng.randint(self.min_targets, self.max_targets)
        else:
            target_count = self.fixed_target_count
        target_count = max(1, min(target_count, len(self.available_labels)))

        for _ in range(self.max_reset_attempts):
            labels = self.rng.sample(self.available_labels, target_count)
            graphs = [self.rng.choice(self.graphs_by_label[label]) for label in labels]
            common_nodes = set(graphs[0].nodes)
            for graph in graphs[1:]:
                common_nodes &= set(graph.nodes)
            if not common_nodes:
                continue
            if self.fixed_start_node is not None:
                start_node = self.fixed_start_node
                if start_node not in common_nodes:
                    continue
                self.used_edge_keys = set()
                if self._candidate_direction_edges(graphs, start_node):
                    return graphs, start_node
                continue
            candidates = list(common_nodes)
            self.rng.shuffle(candidates)
            for start_node in candidates:
                self.used_edge_keys = set()
                if self._candidate_direction_edges(graphs, start_node):
                    return graphs, start_node
        raise RuntimeError("failed to find a valid multi-target episode after many attempts")

    def reset(self) -> dict[str, np.ndarray]:
        self.target_graphs, self.start_node = self._sample_episode()
        self.target_count = len(self.target_graphs)
        self.current_node = self.start_node
        self.edge_history_keys = []
        self.direction_history = []
        self.used_edge_keys = set()
        self.target_edge_histories = [[] for _ in self.target_graphs]
        self.done = False
        self.last_info = {
            "target_count": self.target_count,
            "target_labels": [graph.label for graph in self.target_graphs],
            "target_groups": [graph.group for graph in self.target_graphs],
            "start_node": self.start_node,
        }
        return self._observation()

    def reset_to(self, target_groups: list[str], start_node: int) -> dict[str, np.ndarray]:
        graphs = []
        missing_groups = []
        for group in target_groups:
            graph = self.graphs_by_group.get(group)
            if graph is None:
                missing_groups.append(group)
            else:
                graphs.append(graph)
        if missing_groups:
            raise KeyError(f"episode groups missing from evaluation data: {missing_groups}")
        if not graphs:
            raise ValueError("target_groups must not be empty")

        start_node = int(start_node)
        common_nodes = set(graphs[0].nodes)
        for graph in graphs[1:]:
            common_nodes &= set(graph.nodes)
        if start_node not in common_nodes:
            raise ValueError(f"start_node {start_node} is not common to target groups {target_groups}")

        self.used_edge_keys = set()
        if not self._candidate_direction_edges(graphs, start_node):
            raise ValueError(f"start_node {start_node} has no valid shared direction for target groups {target_groups}")

        self.target_graphs = graphs
        self.start_node = start_node
        self.target_count = len(self.target_graphs)
        self.current_node = self.start_node
        self.edge_history_keys = []
        self.direction_history = []
        self.used_edge_keys = set()
        self.target_edge_histories = [[] for _ in self.target_graphs]
        self.done = False
        self.last_info = {
            "target_count": self.target_count,
            "target_labels": [graph.label for graph in self.target_graphs],
            "target_groups": [graph.group for graph in self.target_graphs],
            "start_node": self.start_node,
        }
        return self._observation()

    def action_mask(self) -> np.ndarray:
        if self.done:
            return self._terminal_action_mask()
        if self.current_node is None or not self.target_graphs:
            raise RuntimeError("environment must be reset before requesting action mask")

        mask = np.zeros((self.action_size,), dtype=np.float32)
        step_count = len(self.edge_history_keys)
        if step_count >= 1:
            mask[0] = 1.0
        if step_count >= self.max_edges:
            mask[:] = 0.0
            mask[0] = 1.0
            return mask

        direction_edges = self._candidate_direction_edges(self.target_graphs, self.current_node)
        if not direction_edges and step_count >= 1:
            mask[0] = 1.0
            return mask
        for action_id in direction_edges:
            mask[int(action_id)] = 1.0
        return mask

    def _step_reward(self, step_count: int) -> float:
        return -self.eta * float(step_count) / float(self.max_edges)

    def step(self, action: int):
        if self.done:
            raise RuntimeError("step called after episode is done")
        mask = self.action_mask()
        action = int(action)
        if action < 0 or action >= self.action_size or mask[action] <= 0:
            valid = np.flatnonzero(mask > 0).tolist()
            raise ValueError(f"invalid action {action}; valid actions are {valid}")

        if action == 0:
            reward, info = self._terminal_reward(forced=False)
            self.done = True
            self.last_info = info
            return self._observation(), reward, True, info

        if self.current_node is None:
            raise RuntimeError("environment must be reset before step")
        direction_edges = self._candidate_direction_edges(self.target_graphs, self.current_node)
        edge_key = direction_edges.get(action)
        if edge_key is None:
            valid = np.flatnonzero(mask > 0).tolist()
            raise ValueError(f"direction action {action} has no edge at node {self.current_node}; valid actions are {valid}")
        action_direction = DIRECTION_ACTIONS[action]
        target_edges = []
        for graph in self.target_graphs:
            edge = graph.edge_by_key.get(edge_key)
            if edge is None:
                raise ValueError(f"selected edge {edge_key} is not present in target graph {graph.group}")
            target_edges.append(edge)

        base_edge = target_edges[0]
        if self.current_node not in {base_edge.a, base_edge.b}:
            raise ValueError(f"edge {edge_key} is not adjacent to current node {self.current_node}")
        next_node = base_edge.b if self.current_node == base_edge.a else base_edge.a

        for idx, edge in enumerate(target_edges):
            if self.current_node not in {edge.a, edge.b}:
                raise ValueError(
                    f"edge {edge_key} is not adjacent to current node {self.current_node} in target {self.target_graphs[idx].group}"
                )
            self.target_edge_histories[idx].append(edge)
        self.edge_history_keys.append(edge_key)
        self.direction_history.append(action_direction)
        self.used_edge_keys.add(edge_key)
        self.current_node = next_node

        if len(self.edge_history_keys) >= self.max_edges:
            terminal_reward, info = self._terminal_reward(forced=True)
            step_reward = self._step_reward(len(self.edge_history_keys))
            reward = terminal_reward + step_reward
            info["reward"] = reward
            info["step_reward"] = step_reward
            info["localization_reward"] = terminal_reward
            self.done = True
            self.last_info = info
            return self._observation(), reward, True, info

        step_count = len(self.edge_history_keys)
        reward = self._step_reward(step_count)
        info = {
            "terminal": False,
            "forced_terminal": False,
            "edge_count": step_count,
            "reward_type": "step",
            "action_direction": action_direction,
            "edge_key": edge_key,
            "target_count": self.target_count,
            "target_labels": [graph.label for graph in self.target_graphs],
            "target_groups": [graph.group for graph in self.target_graphs],
        }
        self.last_info = info
        return self._observation(), reward, False, info

    def _terminal_reward(self, forced: bool) -> tuple[float, dict[str, Any]]:
        if not self.target_graphs:
            raise RuntimeError("environment must be reset before reward computation")
        if not self.edge_history_keys:
            raise RuntimeError("terminal reward requested with empty edge history")

        target_stats = self._predict_target_stats()
        mean_error = float(np.mean([item.error for item in target_stats])) if target_stats else 0.0
        reward_error = min(mean_error, self.distance_norm)
        reward = -reward_error / self.distance_norm
        info = {
            "terminal": True,
            "forced_terminal": bool(forced),
            "edge_count": len(self.edge_history_keys),
            "reward_type": "localization",
            "reward": reward,
            "error": mean_error,
            "mean_error": mean_error,
            "reward_error": float(reward_error),
            "distance_norm": float(self.distance_norm),
            "target_count": self.target_count,
            "target_labels": [item.label for item in target_stats],
            "target_groups": [item.group for item in target_stats],
            "target_errors": [item.error for item in target_stats],
            "target_top1_labels": [item.top1_label for item in target_stats],
            "target_top1_weights": [item.top1_weight for item in target_stats],
            "est_x": float(np.mean([item.est_x for item in target_stats])) if target_stats else 0.0,
            "est_y": float(np.mean([item.est_y for item in target_stats])) if target_stats else 0.0,
            "true_x": float(np.mean([item.true_x for item in target_stats])) if target_stats else 0.0,
            "true_y": float(np.mean([item.true_y for item in target_stats])) if target_stats else 0.0,
            "group": "|".join(item.group for item in target_stats),
            "label": "|".join(item.label for item in target_stats),
            "path": "|".join(self.edge_history_keys),
            "directions": "|".join(self.direction_history),
            "start_node": self.start_node,
            "max_error": float(np.max([item.error for item in target_stats])) if target_stats else 0.0,
            "median_error": float(np.median([item.error for item in target_stats])) if target_stats else 0.0,
            "top1_label": target_stats[0].top1_label if target_stats else "",
            "top1_weight": target_stats[0].top1_weight if target_stats else 0.0,
        }
        return reward, info

    def _predict_target_stats(self) -> list[TargetPrediction]:
        use_path_id = bool(
            getattr(
                self.classifier,
                "uses_path_id",
                self.model_module.input_mode_uses_path_id(str(getattr(self.classifier, "input_mode", "srs_rsrp_path"))),
            )
        )
        predictions: list[TargetPrediction | None] = [None] * len(self.target_graphs)
        signal_batches = []
        rsrp_batches = []
        path_id_batches = []
        edge_mask_batches = []
        misses: list[tuple[int, str, str, tuple[str, tuple[str, ...]]]] = []
        for idx, (graph, history) in enumerate(zip(self.target_graphs, self.target_edge_histories)):
            cache_key = (graph.group, tuple(edge.key for edge in history[: self.max_edges]))
            cached = self._prediction_cache.get(cache_key)
            if cached is not None:
                predictions[idx] = cached
                continue
            sample = self._build_sample(history, use_path_id)
            signal_batches.append(sample["signal"])
            rsrp_batches.append(sample["rsrp"])
            path_id_batches.append(sample["path_id"])
            edge_mask_batches.append(sample["edge_mask"])
            misses.append((idx, graph.label, graph.group, cache_key))

        if misses:
            signal = torch.stack(signal_batches, dim=0).to(self.device)
            rsrp = torch.stack(rsrp_batches, dim=0).to(self.device)
            path_id = torch.stack(path_id_batches, dim=0).to(self.device)
            edge_mask = torch.stack(edge_mask_batches, dim=0).to(self.device)

            self.classifier.eval()
            with torch.no_grad():
                logits = self.classifier(signal, rsrp, path_id, edge_mask) / SOFTMAX_TEMPERATURE
                probs = torch.softmax(logits, dim=1).cpu().numpy()

            for (idx, label, group, cache_key), prob in zip(misses, probs):
                est_x, est_y, top_labels, top_weights = wknn_estimate(prob, self.labels, self.rp_coords, self.wknn_k)
                true_xy = self.true_coords.get(
                    normalize_label_key(label),
                    self.rp_coords.get(normalize_label_key(label)),
                )
                if true_xy is None:
                    raise KeyError(f"label {label} missing from true/rp coordinate CSV")
                error = math.dist((float(true_xy[0]), float(true_xy[1])), (est_x, est_y))
                prediction = TargetPrediction(
                    label=label,
                    group=group,
                    error=float(error),
                    est_x=float(est_x),
                    est_y=float(est_y),
                    true_x=float(true_xy[0]),
                    true_y=float(true_xy[1]),
                    top1_label=top_labels[0] if top_labels else "",
                    top1_weight=top_weights[0] if top_weights else 0.0,
                )
                self._prediction_cache[cache_key] = prediction
                predictions[idx] = prediction
        return [prediction for prediction in predictions if prediction is not None]

    def _build_sample(self, history: list[Any], use_path_id: bool) -> dict[str, torch.Tensor]:
        signal_batch = np.zeros((self.max_edges, self.model_module.SEGMENT_LENGTH, 3, self.model_module.SEQUENCE_LENGTH), dtype=np.float32)
        rsrp_batch = np.zeros((self.max_edges, self.model_module.SEGMENT_LENGTH, 1), dtype=np.float32)
        path_ids = np.zeros((self.max_edges,), dtype=np.int64)
        edge_mask = np.zeros((self.max_edges,), dtype=np.float32)
        for edge_idx, edge in enumerate(history[: self.max_edges]):
            signal, rsrp = self.model_module.load_csv_cached(edge.path)
            signal = self.model_module.normalize_signal(signal, self.stats)
            rsrp = ((rsrp - self.stats.rsrp_mean) / self.stats.rsrp_std).astype(np.float32)
            signal_batch[edge_idx] = np.transpose(signal, (0, 2, 1))
            rsrp_batch[edge_idx] = rsrp
            if use_path_id:
                path_ids[edge_idx] = np.int64(self.path_to_index[edge.key])
            edge_mask[edge_idx] = 1.0

        return {
            "signal": torch.from_numpy(signal_batch),
            "rsrp": torch.from_numpy(rsrp_batch),
            "path_id": torch.from_numpy(path_ids),
            "edge_mask": torch.from_numpy(edge_mask),
        }

    def _observation(self) -> dict[str, np.ndarray]:
        if self.current_node is None:
            current_node_idx = 0
        else:
            current_node_idx = self.node_to_index[self.current_node]

        edge_history_ids = np.full((self.max_edges,), self.pad_edge_id, dtype=np.int64)
        for idx, edge_key in enumerate(self.edge_history_keys[: self.max_edges]):
            edge_history_ids[idx] = self.path_to_index[edge_key] - 1

        target_count = len(self.target_edge_histories)
        signal_state = np.zeros(
            (
                target_count,
                self.max_edges,
                self.model_module.SEGMENT_LENGTH,
                3,
                self.model_module.SEQUENCE_LENGTH,
            ),
            dtype=np.float32,
        )
        rsrp_state = np.zeros(
            (target_count, self.max_edges, self.model_module.SEGMENT_LENGTH, 1),
            dtype=np.float32,
        )

        for target_idx, history in enumerate(self.target_edge_histories):
            for edge_idx, edge in enumerate(history[: self.max_edges]):
                signal, rsrp = self.model_module.load_csv_cached(edge.path)
                signal = self.model_module.normalize_signal(signal, self.stats)
                rsrp = ((rsrp - self.stats.rsrp_mean) / self.stats.rsrp_std).astype(np.float32)
                signal_state[target_idx, edge_idx] = np.transpose(signal, (0, 2, 1))
                rsrp_state[target_idx, edge_idx] = rsrp

        return {
            "current_node": np.asarray(current_node_idx, dtype=np.int64),
            "edge_history": edge_history_ids,
            "signal_state": signal_state,
            "rsrp_state": rsrp_state,
        }

    def _terminal_action_mask(self) -> np.ndarray:
        mask = np.zeros((self.action_size,), dtype=np.float32)
        mask[0] = 1.0
        return mask

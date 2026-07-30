import argparse
import csv
import json
import os
import random
import re
import sys
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset


SEQUENCE_LENGTH = 624
SEGMENT_LENGTH = 20
REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = str(REPO_ROOT / "data" / "SRS-train")
VAL_DIR = str(REPO_ROOT / "data" / "SRS-check")
TEST_DIR = str(REPO_ROOT / "data" / "SRS-test")
OUT_DIR = str(REPO_ROOT / "runs" / "ilcmg")

BATCH_SIZE = 16
EPOCHS = 150
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.02
RANDOM_SEED = 42
VAL_TURN_RATIO = 0.2
INPUT_MODE_SRS_RSRP_PATH = "srs_rsrp_path"
INPUT_MODE_SRS_RSRP = "srs_rsrp"
INPUT_MODE_PATH_ONLY = "path_only"
INPUT_MODES = (INPUT_MODE_SRS_RSRP_PATH, INPUT_MODE_SRS_RSRP, INPUT_MODE_PATH_ONLY)
PATH_ID_INPUT_MODES = {INPUT_MODE_SRS_RSRP_PATH, INPUT_MODE_PATH_ONLY}
ARCH_ILCMG = "ilcmg"
ARCH_AARESCNN = "aarescnn"
ARCH_HILOC = "hiloc"
ARCH_AARES = "aares"
ARCHITECTURES = (ARCH_ILCMG, ARCH_AARESCNN, ARCH_HILOC, ARCH_AARES)


def input_mode_uses_path_id(input_mode: str) -> bool:
    return input_mode in PATH_ID_INPUT_MODES


@dataclass(frozen=True)
class NormStats:
    rsrp_mean: float
    rsrp_std: float
    signal_mean: list[float]
    signal_std: list[float]


@dataclass(frozen=True)
class EdgeRecord:
    key: str
    a: int
    b: int
    path: str


@dataclass(frozen=True)
class SequenceExample:
    label: str | None
    group: str
    edge_keys: tuple[str, ...]
    edge_paths: tuple[str, ...]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_array_string(value, expected_len: int = SEQUENCE_LENGTH) -> np.ndarray:
    text = str(value).strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    arr = np.fromstring(text, sep=",", dtype=np.float32)
    if arr.size != expected_len:
        arr = np.fromstring(text.replace(",", " "), sep=" ", dtype=np.float32)
    if arr.size != expected_len:
        raise ValueError(f"array length {arr.size}, expected {expected_len}")
    return arr


def parse_path_name(csv_path: Path) -> tuple[int | None, int | None]:
    match = re.match(r"^(\d+)-(\d+)$", csv_path.stem)
    if not match:
        return None, None
    a, b = int(match.group(1)), int(match.group(2))
    return (a, b) if a <= b else (b, a)


def path_key(a: int, b: int) -> str:
    return f"{min(a, b)}-{max(a, b)}"


def natural_turn_key(path: Path) -> tuple[int, str]:
    match = re.search(r"turn_?(\d+)", path.name.lower())
    if match:
        return int(match.group(1)), path.name.lower()
    return 10**9, path.name.lower()


def numeric_label_key(path: Path) -> tuple[int, str]:
    try:
        return int(path.name), path.name
    except ValueError:
        return 10**9, path.name


def natural_label_key(path: Path) -> tuple[str, int | str]:
    name = path.name
    if name.isdigit():
        return "", int(name)
    prefix = "".join(ch for ch in name if not ch.isdigit())
    suffix = "".join(ch for ch in name if ch.isdigit())
    return prefix, int(suffix) if suffix else name


@lru_cache(maxsize=8192)
def load_csv_cached(csv_path_str: str) -> tuple[np.ndarray, np.ndarray]:
    csv_path = Path(csv_path_str)
    df = pd.read_csv(csv_path)
    required = {"ls_real", "ls_imag", "rsrp_value"}
    if not required.issubset(df.columns):
        raise ValueError(f"{csv_path} missing columns {required - set(df.columns)}")
    if len(df) != SEGMENT_LENGTH:
        raise ValueError(f"{csv_path} segment length {len(df)}, expected {SEGMENT_LENGTH}")

    real = np.stack([parse_array_string(v) for v in df["ls_real"].to_numpy()], axis=0)
    imag = np.stack([parse_array_string(v) for v in df["ls_imag"].to_numpy()], axis=0)
    rsrp = df["rsrp_value"].to_numpy(dtype=np.float32).reshape(SEGMENT_LENGTH, 1)
    amp = np.sqrt(real * real + imag * imag)
    signal = np.stack([real, imag, amp], axis=-1).astype(np.float32)
    return signal, rsrp


def validate_csv_shape(csv_path: str) -> tuple[bool, str]:
    try:
        load_csv_cached(csv_path)
        return True, ""
    except Exception as exc:
        return False, str(exc)


def filter_bad_examples(
    examples: list[SequenceExample],
    report_path: Path | None = None,
    strict: bool = False,
) -> list[SequenceExample]:
    bad_files: dict[str, str] = {}
    good_files: set[str] = set()
    kept: list[SequenceExample] = []

    def is_good(path: str) -> bool:
        if path in good_files:
            return True
        if path in bad_files:
            return False
        ok, reason = validate_csv_shape(path)
        if ok:
            good_files.add(path)
            return True
        bad_files[path] = reason
        return False

    for example in examples:
        if all(is_good(path) for path in example.edge_paths):
            kept.append(example)

    if bad_files:
        message = f"Skipped {len(examples) - len(kept)} examples because {len(bad_files)} CSV files are invalid."
        if strict:
            first_path, first_reason = next(iter(bad_files.items()))
            raise ValueError(f"{message} First bad file: {first_path} | {first_reason}")
        print(message)
        if report_path is not None:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            with report_path.open("w", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=["file", "reason"])
                writer.writeheader()
                for path, reason in sorted(bad_files.items()):
                    writer.writerow({"file": path, "reason": reason})
            print(f"Bad CSV report: {report_path}")

    if not kept:
        raise RuntimeError("all examples were filtered out")
    return kept


def iter_label_dirs(root_dir: str) -> list[Path]:
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"directory not found: {root}")
    return sorted([p for p in root.iterdir() if p.is_dir() and any(ch.isdigit() for ch in p.name)], key=natural_label_key)


def iter_turn_dirs(label_dir: Path) -> list[Path]:
    return sorted(
        [p for p in label_dir.iterdir() if p.is_dir() and p.name.lower().startswith("turn")],
        key=natural_turn_key,
    )


def collect_edges(turn_dir: Path) -> list[EdgeRecord]:
    records_by_key: dict[str, EdgeRecord] = {}
    for csv_path in sorted(turn_dir.glob("*.csv"), key=lambda p: p.name.lower()):
        a, b = parse_path_name(csv_path)
        if a is None or b is None:
            continue
        key = path_key(a, b)
        records_by_key.setdefault(key, EdgeRecord(key=key, a=min(a, b), b=max(a, b), path=str(csv_path)))
    return sorted(records_by_key.values(), key=lambda e: e.key)


def enumerate_valid_sequences(
    edges: list[EdgeRecord],
    max_edges: int,
    min_edges: int = 1,
    allow_edge_repeat: bool = False,
) -> list[tuple[EdgeRecord, ...]]:
    adjacency: dict[int, list[int]] = {}
    for idx, edge in enumerate(edges):
        adjacency.setdefault(edge.a, []).append(idx)
        adjacency.setdefault(edge.b, []).append(idx)

    seen: set[tuple[str, ...]] = set()
    sequences: list[tuple[EdgeRecord, ...]] = []

    def add_sequence(indices: list[int]) -> None:
        if len(indices) < min_edges:
            return
        key_tuple = tuple(edges[i].key for i in indices)
        if key_tuple in seen:
            return
        seen.add(key_tuple)
        sequences.append(tuple(edges[i] for i in indices))

    def dfs(current_node: int, indices: list[int], used: set[int]) -> None:
        add_sequence(indices)
        if len(indices) >= max_edges:
            return

        for next_idx in adjacency.get(current_node, []):
            if not allow_edge_repeat and next_idx in used:
                continue
            edge = edges[next_idx]
            next_node = edge.b if current_node == edge.a else edge.a
            dfs(next_node, indices + [next_idx], used | {next_idx})

    for idx, edge in enumerate(edges):
        dfs(edge.b, [idx], {idx})
        dfs(edge.a, [idx], {idx})

    return sequences


def limit_sequences(
    sequences: list[tuple[EdgeRecord, ...]],
    max_sequences: int,
    rng: random.Random,
) -> list[tuple[EdgeRecord, ...]]:
    if max_sequences <= 0 or len(sequences) <= max_sequences:
        return sequences

    singles = [seq for seq in sequences if len(seq) == 1]
    longer = [seq for seq in sequences if len(seq) > 1]
    rng.shuffle(longer)
    keep_count = max(0, max_sequences - len(singles))
    return singles + longer[:keep_count]


def build_examples(
    root_dir: str,
    max_edges: int,
    min_edges: int,
    max_sequences_per_turn: int,
    seed: int,
    allow_edge_repeat: bool,
    require_labels: bool = True,
) -> list[SequenceExample]:
    root = Path(root_dir)
    rng = random.Random(seed)
    examples: list[SequenceExample] = []

    label_dirs = iter_label_dirs(root_dir)
    if not label_dirs and not require_labels:
        label_dirs = [root]

    for label_dir in label_dirs:
        label = label_dir.name if label_dir.name.isdigit() else None
        if require_labels and label is None:
            continue
        for turn_dir in iter_turn_dirs(label_dir):
            edges = collect_edges(turn_dir)
            if not edges:
                continue

            sequences = enumerate_valid_sequences(
                edges=edges,
                max_edges=max_edges,
                min_edges=min_edges,
                allow_edge_repeat=allow_edge_repeat,
            )
            sequences = limit_sequences(sequences, max_sequences_per_turn, rng)

            for seq in sequences:
                examples.append(
                    SequenceExample(
                        label=label,
                        group=f"{label_dir.name}/{turn_dir.name}",
                        edge_keys=tuple(edge.key for edge in seq),
                        edge_paths=tuple(edge.path for edge in seq),
                    )
                )

    if not examples:
        raise RuntimeError(f"no valid multi-edge examples found under {root_dir}")
    return examples


def fit_normalizer(examples: list[SequenceExample]) -> NormStats:
    signal_sum = np.zeros(3, dtype=np.float64)
    signal_sq_sum = np.zeros(3, dtype=np.float64)
    signal_count = 0
    rsrp_sum = 0.0
    rsrp_sq_sum = 0.0
    rsrp_count = 0

    seen_files: set[str] = set()
    for example in examples:
        for path in example.edge_paths:
            if path in seen_files:
                continue
            seen_files.add(path)
            signal, rsrp = load_csv_cached(path)
            arr = signal.astype(np.float64)
            signal_sum += np.sum(arr, axis=(0, 1))
            signal_sq_sum += np.sum(arr * arr, axis=(0, 1))
            signal_count += arr.shape[0] * arr.shape[1]

            r = rsrp.astype(np.float64)
            rsrp_sum += float(np.sum(r))
            rsrp_sq_sum += float(np.sum(r * r))
            rsrp_count += r.size

    signal_mean = signal_sum / max(signal_count, 1)
    signal_var = signal_sq_sum / max(signal_count, 1) - signal_mean * signal_mean
    signal_std = np.sqrt(np.maximum(signal_var, 1e-12)) + 1e-6

    rsrp_mean = rsrp_sum / max(rsrp_count, 1)
    rsrp_var = rsrp_sq_sum / max(rsrp_count, 1) - rsrp_mean * rsrp_mean
    rsrp_std = float(np.sqrt(max(rsrp_var, 1e-12)) + 1e-6)

    return NormStats(
        rsrp_mean=float(rsrp_mean),
        rsrp_std=rsrp_std,
        signal_mean=[float(v) for v in signal_mean],
        signal_std=[float(v) for v in signal_std],
    )


def normalize_signal(signal: np.ndarray, stats: NormStats) -> np.ndarray:
    mean = np.asarray(stats.signal_mean, dtype=np.float32).reshape(1, 1, 3)
    std = np.asarray(stats.signal_std, dtype=np.float32).reshape(1, 1, 3)
    return ((signal - mean) / std).astype(np.float32)


def sort_label(label: str):
    return int(label) if str(label).isdigit() else str(label)


def encode_labels(train_examples: list[SequenceExample], other_examples: list[SequenceExample] | None = None):
    labels = sorted({ex.label for ex in train_examples if ex.label is not None}, key=sort_label)
    label_to_index = {label: idx for idx, label in enumerate(labels)}

    if other_examples is not None:
        missing = sorted({ex.label for ex in other_examples if ex.label is not None} - set(labels), key=sort_label)
        if missing:
            raise ValueError(f"validation/test contains labels not present in training: {missing}")

    return np.asarray(labels, dtype=object), label_to_index


def path_sort_key(value: str):
    try:
        a, b = value.split("-")
        return int(a), int(b)
    except ValueError:
        return (9999, 9999)


def encode_path_keys(train_examples: list[SequenceExample], other_examples: list[SequenceExample] | None = None):
    keys = sorted({key for ex in train_examples for key in ex.edge_keys}, key=path_sort_key)
    key_to_index = {key: idx + 1 for idx, key in enumerate(keys)}

    if other_examples is not None:
        missing = sorted({key for ex in other_examples for key in ex.edge_keys} - set(keys), key=path_sort_key)
        if missing:
            raise ValueError(f"validation/test contains path IDs not present in training: {missing}")

    return np.asarray(keys, dtype=object), key_to_index


class MultiEdgeDataset(Dataset):
    def __init__(
        self,
        examples: list[SequenceExample],
        labels: np.ndarray | None,
        path_to_index: dict[str, int],
        stats: NormStats,
        max_edges: int,
        training: bool,
        use_path_id: bool = True,
    ):
        self.examples = examples
        self.labels = labels
        self.path_to_index = path_to_index
        self.stats = stats
        self.max_edges = max_edges
        self.training = training
        self.use_path_id = use_path_id

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int):
        example = self.examples[index]
        signal_batch = np.zeros((self.max_edges, SEGMENT_LENGTH, 3, SEQUENCE_LENGTH), dtype=np.float32)
        rsrp_batch = np.zeros((self.max_edges, SEGMENT_LENGTH, 1), dtype=np.float32)
        path_ids = np.zeros((self.max_edges,), dtype=np.int64)
        edge_mask = np.zeros((self.max_edges,), dtype=np.float32)

        for edge_idx, (edge_key, edge_path) in enumerate(zip(example.edge_keys, example.edge_paths)):
            signal, rsrp = load_csv_cached(edge_path)
            signal = normalize_signal(signal, self.stats)
            rsrp = ((rsrp - self.stats.rsrp_mean) / self.stats.rsrp_std).astype(np.float32)

            if self.training:
                signal_scale = np.float32(np.random.uniform(0.99, 1.01))
                signal = signal * signal_scale
                signal = signal + np.random.normal(0.0, 0.01, size=signal.shape).astype(np.float32)
                rsrp = rsrp + np.random.normal(0.0, 0.01, size=rsrp.shape).astype(np.float32)

            signal_batch[edge_idx] = np.transpose(signal, (0, 2, 1))
            rsrp_batch[edge_idx] = rsrp
            if self.use_path_id:
                path_ids[edge_idx] = np.int64(self.path_to_index[edge_key])
            edge_mask[edge_idx] = 1.0

        sample = {
            "signal": torch.from_numpy(signal_batch),
            "rsrp": torch.from_numpy(rsrp_batch),
            "path_id": torch.from_numpy(path_ids),
            "edge_mask": torch.from_numpy(edge_mask),
            "edge_count": torch.tensor(len(example.edge_keys), dtype=torch.long),
        }

        if self.labels is None:
            return sample
        return sample, torch.tensor(int(self.labels[index]), dtype=torch.long)


class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, dropout: float = 0.04):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.shortcut = None
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.silu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.shortcut is not None:
            identity = self.shortcut(identity)
        x = x + identity
        return F.silu(x)


class PacketEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_dropout = nn.Dropout(0.02)
        self.stem = nn.Sequential(
            nn.Conv1d(3, 32, 9, stride=2, padding=4, bias=False),
            nn.BatchNorm1d(32),
            nn.SiLU(),
            nn.Dropout(0.04),
        )
        self.block1 = ResidualConvBlock(32, 64, 7, stride=2)
        self.block2 = ResidualConvBlock(64, 96, 5, stride=2)
        self.block3 = ResidualConvBlock(96, 128, 3, stride=2)
        self.fc = nn.Linear(256, 160)
        self.out_dropout = nn.Dropout(0.15)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_dropout(x)
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        avg = x.mean(dim=-1)
        mx = x.amax(dim=-1)
        x = torch.cat([avg, mx], dim=-1)
        x = F.silu(self.fc(x))
        return self.out_dropout(x)


class MultiEdgeRPClassifier(nn.Module):
    def __init__(self, num_classes: int, num_paths: int, input_mode: str = INPUT_MODE_SRS_RSRP_PATH):
        super().__init__()
        if input_mode not in INPUT_MODES:
            raise ValueError(f"unknown input_mode {input_mode!r}; expected one of {INPUT_MODES}")
        self.input_mode = input_mode
        self.uses_signal = input_mode in {INPUT_MODE_SRS_RSRP_PATH, INPUT_MODE_SRS_RSRP}
        self.uses_rsrp = self.uses_signal
        self.uses_path_id = input_mode_uses_path_id(input_mode)

        if self.uses_signal:
            self.packet_encoder = PacketEncoder()
        if self.uses_rsrp:
            self.rsrp_norm = nn.LayerNorm(1)
            self.rsrp_fc = nn.Linear(1, 16)
        if self.uses_path_id:
            self.path_embedding = nn.Embedding(num_paths + 1, 32, padding_idx=0)
            self.path_fc = nn.Linear(32, 32)

        if input_mode == INPUT_MODE_PATH_ONLY:
            self.path_only_fc = nn.Linear(32, 256)
        else:
            edge_input_dim = 160 + 16 + (32 if self.uses_path_id else 0)
            self.edge_input_norm = nn.LayerNorm(edge_input_dim)
            self.packet_gru1 = nn.GRU(edge_input_dim, 96, batch_first=True, bidirectional=True)
            self.packet_gru2 = nn.GRU(192, 64, batch_first=True, bidirectional=True)
            self.packet_dropout1 = nn.Dropout(0.2)
            self.packet_dropout2 = nn.Dropout(0.2)
            self.packet_attn_fc1 = nn.Linear(128, 32)
            self.packet_attn_fc2 = nn.Linear(32, 1)

        self.edge_gru = nn.GRU(256, 96, batch_first=True, bidirectional=True)
        self.edge_dropout = nn.Dropout(0.2)
        self.edge_attn_fc1 = nn.Linear(192, 64)
        self.edge_attn_fc2 = nn.Linear(64, 1)

        self.head1 = nn.Linear(384, 192)
        self.head2 = nn.Linear(192, 96)
        self.out = nn.Linear(96, num_classes)
        self.head_dropout1 = nn.Dropout(0.3)
        self.head_dropout2 = nn.Dropout(0.2)

    def forward(
        self,
        signal: torch.Tensor,
        rsrp: torch.Tensor,
        path_id: torch.Tensor,
        edge_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, max_edges, segment_length, channels, sequence_length = signal.shape

        if self.input_mode == INPUT_MODE_PATH_ONLY:
            p = self.path_embedding(path_id)
            p = F.silu(self.path_fc(p))
            edge_repr = F.silu(self.path_only_fc(p))
        else:
            x = signal.reshape(batch_size * max_edges * segment_length, channels, sequence_length)
            x = self.packet_encoder(x)
            x = x.reshape(batch_size, max_edges, segment_length, 160)

            r = self.rsrp_norm(rsrp)
            r = F.silu(self.rsrp_fc(r))
            edge_features = [x, r]

            if self.uses_path_id:
                p = self.path_embedding(path_id)
                p = F.silu(self.path_fc(p))
                p = p.unsqueeze(2).expand(-1, -1, segment_length, -1)
                edge_features.append(p)

            edge_seq = torch.cat(edge_features, dim=-1)
            edge_seq = self.edge_input_norm(edge_seq)
            edge_seq = edge_seq.reshape(batch_size * max_edges, segment_length, -1)

            packet_out, _ = self.packet_gru1(edge_seq)
            packet_out = self.packet_dropout1(packet_out)
            packet_out, _ = self.packet_gru2(packet_out)
            packet_out = self.packet_dropout2(packet_out)

            packet_score = torch.tanh(self.packet_attn_fc1(packet_out))
            packet_score = self.packet_attn_fc2(packet_score)
            packet_weight = torch.softmax(packet_score, dim=1)
            packet_attended = (packet_out * packet_weight).sum(dim=1)
            packet_pooled = packet_out.mean(dim=1)
            edge_repr = torch.cat([packet_attended, packet_pooled], dim=-1).reshape(batch_size, max_edges, 256)

        edge_repr = edge_repr * edge_mask.unsqueeze(-1)
        edge_lengths = edge_mask.sum(dim=1).long().clamp(min=1)
        packed = pack_padded_sequence(edge_repr, edge_lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.edge_gru(packed)
        edge_out, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=max_edges)
        edge_out = self.edge_dropout(edge_out)
        edge_out = edge_out * edge_mask.unsqueeze(-1)

        edge_score = torch.tanh(self.edge_attn_fc1(edge_out))
        edge_score = self.edge_attn_fc2(edge_score).squeeze(-1)
        edge_score = edge_score.masked_fill(edge_mask <= 0, -1e9)
        edge_weight = torch.softmax(edge_score, dim=1)
        edge_attended = (edge_out * edge_weight.unsqueeze(-1)).sum(dim=1)

        denom = edge_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        edge_avg = edge_out.sum(dim=1) / denom

        z = torch.cat([edge_attended, edge_avg], dim=-1)
        z = F.silu(self.head1(z))
        z = self.head_dropout1(z)
        z = F.silu(self.head2(z))
        z = self.head_dropout2(z)
        logits = self.out(z)
        return logits


def make_activation(name: str) -> nn.Module:
    if name == "leaky_relu":
        return nn.LeakyReLU(0.1, inplace=True)
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "silu":
        return nn.SiLU(inplace=True)
    raise ValueError(f"unknown activation: {name}")


class Conv2dResidualBlock(nn.Module):
    def __init__(self, channels: int = 32, kernel_size: int = 5, activation: str = "leaky_relu", dropout: float = 0.03):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.act1 = make_activation(activation)
        self.dropout = nn.Dropout2d(dropout)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.act2 = make_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.bn2(x)
        return self.act2(x + identity)


class PoolingBlock2d(nn.Module):
    def __init__(self, channels: int = 32, activation: str = "leaky_relu"):
        super().__init__()
        self.expand = nn.Conv2d(channels, channels * 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels * 2)
        self.pool = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.project = nn.Conv2d(channels * 2, channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.act = make_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.expand(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.pool(x)
        x = self.project(x)
        x = self.bn2(x)
        return self.act(x)


class GlobalContextGate2d(nn.Module):
    """Lightweight global-context attention used to adapt AAConv-style blocks to long SRS tensors."""

    def __init__(self, channels: int = 32, reduction: int = 4):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channel = self.channel_gate(x)
        avg = torch.mean(x, dim=1, keepdim=True)
        mx = torch.amax(x, dim=1, keepdim=True)
        spatial = self.spatial_gate(torch.cat([avg, mx], dim=1))
        return x * channel * spatial


class AttentionAugmentedResidual2d(nn.Module):
    def __init__(self, channels: int = 32, kernel_size: int = 5, activation: str = "leaky_relu", dropout: float = 0.03):
        super().__init__()
        padding = kernel_size // 2
        self.local_conv = nn.Conv2d(channels, channels, kernel_size, padding=padding, bias=False)
        self.context = GlobalContextGate2d(channels)
        self.fuse = nn.Conv2d(channels * 2, channels, kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.dropout = nn.Dropout2d(dropout)
        self.out = nn.Conv2d(channels, channels, kernel_size, padding=padding, bias=False)
        self.out_bn = nn.BatchNorm2d(channels)
        self.act1 = make_activation(activation)
        self.act2 = make_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        local = self.local_conv(x)
        global_context = self.context(x)
        x = torch.cat([local, global_context], dim=1)
        x = self.fuse(x)
        x = self.bn(x)
        x = self.act1(x)
        x = self.dropout(x)
        x = self.out(x)
        x = self.out_bn(x)
        return self.act2(x + identity)


class ResAttention2dEdgeEncoder(nn.Module):
    def __init__(
        self,
        cascades: int,
        blocks_after_pool: int,
        attention_interval: int,
        activation: str,
    ):
        super().__init__()
        channels = 32
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(channels),
            make_activation(activation),
        )
        body: list[nn.Module] = []
        for _ in range(cascades):
            body.append(Conv2dResidualBlock(channels, kernel_size=5, activation=activation))
            body.append(PoolingBlock2d(channels, activation=activation))
        for idx in range(blocks_after_pool):
            if idx % max(1, attention_interval) == 0:
                body.append(AttentionAugmentedResidual2d(channels, kernel_size=5, activation=activation))
            else:
                body.append(Conv2dResidualBlock(channels, kernel_size=5, activation=activation))
        self.body = nn.Sequential(*body)
        self.out = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(channels, 160),
            make_activation(activation),
            nn.Dropout(0.15),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.body(x)
        return self.out(x)


class HiLocEdgeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=2, padding=1, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.feature_norm = nn.LayerNorm(32)
        self.feature_attn = nn.Linear(32, 32)
        self.rsrp_fc = nn.Linear(1, 8)
        self.bilstm = nn.LSTM(40, 64, batch_first=True, bidirectional=True)
        self.sample_attn = nn.Linear(128, 1)
        self.out = nn.Sequential(
            nn.Linear(256, 160),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),
        )

    def forward(self, signal: torch.Tensor, rsrp: torch.Tensor) -> torch.Tensor:
        batch_edges, segment_length, channels, sequence_length = signal.shape
        x = signal.reshape(batch_edges * segment_length, channels, sequence_length)
        x = self.cnn(x).squeeze(-1)
        weights = torch.softmax(self.feature_attn(self.feature_norm(x)), dim=-1)
        x = x * weights
        x = x.reshape(batch_edges, segment_length, 32)
        r = F.relu(self.rsrp_fc(rsrp.reshape(batch_edges, segment_length, 1)))
        x = torch.cat([x, r], dim=-1)
        seq, _ = self.bilstm(x)
        scores = self.sample_attn(seq)
        attn = torch.softmax(scores, dim=1)
        attended = torch.sum(seq * attn, dim=1)
        pooled = torch.mean(seq, dim=1)
        return self.out(torch.cat([attended, pooled], dim=-1))


class FairHiLocClassifier(nn.Module):
    def __init__(self, num_classes: int, num_paths: int, use_path_id: bool):
        super().__init__()
        self.input_mode = INPUT_MODE_SRS_RSRP
        self.architecture = ARCH_HILOC
        self.uses_path_id = bool(use_path_id)
        self.cnn = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=2, padding=1, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.feature_norm = nn.LayerNorm(32)
        self.feature_attn = nn.Linear(32, 32)
        self.rsrp_fc = nn.Linear(4, 16)
        if self.uses_path_id:
            self.path_embedding = nn.Embedding(num_paths + 1, 32, padding_idx=0)
            self.path_fc = nn.Linear(32, 16)
        self.bilstm = nn.LSTM(32, 64, batch_first=True, bidirectional=True)
        self.sample_attn = nn.Linear(128, 1)
        head_in = 256 + 16 + (16 if self.uses_path_id else 0)
        self.head = nn.Sequential(
            nn.Linear(head_in, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(32, num_classes),
        )

    def forward(
        self,
        signal: torch.Tensor,
        rsrp: torch.Tensor,
        path_id: torch.Tensor,
        edge_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, max_edges, segment_length, channels, sequence_length = signal.shape
        x = signal.reshape(batch_size * max_edges * segment_length, channels, sequence_length)
        x = self.cnn(x).squeeze(-1)
        weights = torch.softmax(self.feature_attn(self.feature_norm(x)), dim=-1)
        x = (x * weights).reshape(batch_size * max_edges, segment_length, 32)
        seq, _ = self.bilstm(x)

        scores = self.sample_attn(seq).squeeze(-1)
        attn = torch.softmax(scores, dim=1).unsqueeze(-1)
        attended = torch.sum(seq * attn, dim=1)
        pooled = torch.mean(seq, dim=1)

        rsrp_mean = rsrp.mean(dim=2).squeeze(-1)
        rsrp_std = rsrp.std(dim=2, unbiased=False).squeeze(-1)
        rsrp_min = rsrp.amin(dim=2).squeeze(-1)
        rsrp_max = rsrp.amax(dim=2).squeeze(-1)
        rsrp_stats = torch.stack([rsrp_mean, rsrp_std, rsrp_min, rsrp_max], dim=-1)
        rsrp_features = F.relu(self.rsrp_fc(rsrp_stats.reshape(batch_size * max_edges, 4)))
        features = [attended, pooled, rsrp_features]
        if self.uses_path_id:
            path_features = F.relu(self.path_fc(self.path_embedding(path_id)))
            features.append(path_features.reshape(batch_size * max_edges, -1))
        edge_logits = self.head(torch.cat(features, dim=-1)).reshape(batch_size, max_edges, -1)
        denom = edge_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        return torch.sum(edge_logits * edge_mask.unsqueeze(-1), dim=1) / denom


class FairResAttentionClassifier(nn.Module):
    def __init__(self, num_classes: int, num_paths: int, architecture: str, use_path_id: bool):
        super().__init__()
        if architecture == ARCH_AARESCNN:
            self.encoder = ResAttention2dEdgeEncoder(cascades=2, blocks_after_pool=7, attention_interval=1, activation="leaky_relu")
        elif architecture == ARCH_AARES:
            self.encoder = ResAttention2dEdgeEncoder(cascades=2, blocks_after_pool=4, attention_interval=1, activation="relu")
        else:
            raise ValueError(f"unsupported strict residual architecture: {architecture}")
        self.input_mode = INPUT_MODE_SRS_RSRP
        self.architecture = architecture
        self.uses_path_id = bool(use_path_id)
        self.rsrp_fc = nn.Linear(4, 16)
        if self.uses_path_id:
            self.path_embedding = nn.Embedding(num_paths + 1, 32, padding_idx=0)
            self.path_fc = nn.Linear(32, 16)
        head_in = 160 + 16 + (16 if self.uses_path_id else 0)
        self.head = nn.Sequential(
            nn.Linear(head_in, 64),
            make_activation("leaky_relu" if architecture == ARCH_AARESCNN else "relu"),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            make_activation("leaky_relu" if architecture == ARCH_AARESCNN else "relu"),
            nn.Dropout(0.1),
            nn.Linear(32, num_classes),
        )

    def forward(
        self,
        signal: torch.Tensor,
        rsrp: torch.Tensor,
        path_id: torch.Tensor,
        edge_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, max_edges, segment_length, channels, sequence_length = signal.shape
        x = signal.reshape(batch_size * max_edges, segment_length, channels, sequence_length).permute(0, 2, 1, 3)
        rsrp_mean = rsrp.mean(dim=2).squeeze(-1)
        rsrp_std = rsrp.std(dim=2, unbiased=False).squeeze(-1)
        rsrp_min = rsrp.amin(dim=2).squeeze(-1)
        rsrp_max = rsrp.amax(dim=2).squeeze(-1)
        rsrp_stats = torch.stack([rsrp_mean, rsrp_std, rsrp_min, rsrp_max], dim=-1)
        rsrp_features = F.relu(self.rsrp_fc(rsrp_stats.reshape(batch_size * max_edges, 4)))
        features = [self.encoder(x), rsrp_features]
        if self.uses_path_id:
            path_features = F.relu(self.path_fc(self.path_embedding(path_id)))
            features.append(path_features.reshape(batch_size * max_edges, -1))
        edge_logits = self.head(torch.cat(features, dim=-1)).reshape(batch_size, max_edges, -1)
        denom = edge_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        return torch.sum(edge_logits * edge_mask.unsqueeze(-1), dim=1) / denom


class PaperBackboneRPClassifier(nn.Module):
    def __init__(self, num_classes: int, num_paths: int, architecture: str, input_mode: str = INPUT_MODE_SRS_RSRP_PATH):
        super().__init__()
        if input_mode not in INPUT_MODES:
            raise ValueError(f"unknown input_mode {input_mode!r}; expected one of {INPUT_MODES}")
        if architecture not in {ARCH_AARESCNN, ARCH_HILOC, ARCH_AARES}:
            raise ValueError(f"unsupported paper backbone: {architecture}")
        self.input_mode = input_mode
        self.architecture = architecture
        self.uses_signal = input_mode in {INPUT_MODE_SRS_RSRP_PATH, INPUT_MODE_SRS_RSRP}
        self.uses_rsrp = self.uses_signal
        self.uses_path_id = input_mode_uses_path_id(input_mode)

        if self.uses_signal:
            if architecture == ARCH_AARESCNN:
                self.signal_encoder = ResAttention2dEdgeEncoder(cascades=2, blocks_after_pool=7, attention_interval=1, activation="leaky_relu")
            elif architecture == ARCH_AARES:
                self.signal_encoder = ResAttention2dEdgeEncoder(cascades=2, blocks_after_pool=4, attention_interval=1, activation="relu")
            else:
                self.signal_encoder = HiLocEdgeEncoder()

        if self.uses_rsrp and architecture != ARCH_HILOC:
            self.rsrp_stats_norm = nn.LayerNorm(4)
            self.rsrp_fc = nn.Linear(4, 32)
        if self.uses_path_id:
            self.path_embedding = nn.Embedding(num_paths + 1, 32, padding_idx=0)
            self.path_fc = nn.Linear(32, 32)

        if input_mode == INPUT_MODE_PATH_ONLY:
            edge_input_dim = 32
        else:
            edge_input_dim = 160 + (0 if architecture == ARCH_HILOC else 32) + (32 if self.uses_path_id else 0)
        self.edge_input_norm = nn.LayerNorm(edge_input_dim)
        self.edge_fc = nn.Linear(edge_input_dim, 256)
        self.edge_gru = nn.GRU(256, 96, batch_first=True, bidirectional=True)
        self.edge_dropout = nn.Dropout(0.2)
        self.edge_attn_fc1 = nn.Linear(192, 64)
        self.edge_attn_fc2 = nn.Linear(64, 1)
        self.head1 = nn.Linear(384, 192)
        self.head2 = nn.Linear(192, 96)
        self.out = nn.Linear(96, num_classes)
        self.head_dropout1 = nn.Dropout(0.3)
        self.head_dropout2 = nn.Dropout(0.2)

    def _encode_rsrp_stats(self, rsrp: torch.Tensor) -> torch.Tensor:
        mean = rsrp.mean(dim=2).squeeze(-1)
        std = rsrp.std(dim=2, unbiased=False).squeeze(-1)
        mn = rsrp.amin(dim=2).squeeze(-1)
        mx = rsrp.amax(dim=2).squeeze(-1)
        stats = torch.stack([mean, std, mn, mx], dim=-1)
        return F.relu(self.rsrp_fc(self.rsrp_stats_norm(stats)))

    def forward(
        self,
        signal: torch.Tensor,
        rsrp: torch.Tensor,
        path_id: torch.Tensor,
        edge_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, max_edges, segment_length, channels, sequence_length = signal.shape

        if self.input_mode == INPUT_MODE_PATH_ONLY:
            p = self.path_embedding(path_id)
            edge_features = F.relu(self.path_fc(p))
        else:
            flat_signal = signal.reshape(batch_size * max_edges, segment_length, channels, sequence_length)
            flat_rsrp = rsrp.reshape(batch_size * max_edges, segment_length, 1)
            if self.architecture == ARCH_HILOC:
                encoded = self.signal_encoder(flat_signal, flat_rsrp)
            else:
                encoded = self.signal_encoder(flat_signal.permute(0, 2, 1, 3))
            encoded = encoded.reshape(batch_size, max_edges, -1)
            features = [encoded]
            if self.architecture != ARCH_HILOC:
                features.append(self._encode_rsrp_stats(rsrp))
            if self.uses_path_id:
                p = self.path_embedding(path_id)
                features.append(F.relu(self.path_fc(p)))
            edge_features = torch.cat(features, dim=-1)

        edge_features = self.edge_input_norm(edge_features)
        edge_repr = F.relu(self.edge_fc(edge_features)) * edge_mask.unsqueeze(-1)
        edge_lengths = edge_mask.sum(dim=1).long().clamp(min=1)
        packed = pack_padded_sequence(edge_repr, edge_lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.edge_gru(packed)
        edge_out, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=max_edges)
        edge_out = self.edge_dropout(edge_out) * edge_mask.unsqueeze(-1)

        edge_score = torch.tanh(self.edge_attn_fc1(edge_out))
        edge_score = self.edge_attn_fc2(edge_score).squeeze(-1)
        edge_score = edge_score.masked_fill(edge_mask <= 0, -1e9)
        edge_weight = torch.softmax(edge_score, dim=1)
        edge_attended = (edge_out * edge_weight.unsqueeze(-1)).sum(dim=1)
        denom = edge_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        edge_avg = edge_out.sum(dim=1) / denom

        z = torch.cat([edge_attended, edge_avg], dim=-1)
        z = F.relu(self.head1(z))
        z = self.head_dropout1(z)
        z = F.relu(self.head2(z))
        z = self.head_dropout2(z)
        return self.out(z)


def build_classifier(
    num_classes: int,
    num_paths: int,
    architecture: str = ARCH_ILCMG,
    input_mode: str = INPUT_MODE_SRS_RSRP_PATH,
) -> nn.Module:
    if architecture == ARCH_ILCMG:
        return MultiEdgeRPClassifier(num_classes, num_paths, input_mode=input_mode)
    if architecture in {ARCH_AARESCNN, ARCH_AARES}:
        if input_mode == INPUT_MODE_PATH_ONLY:
            raise ValueError("strict paper baselines require SRS input; use --input-mode srs_rsrp or srs_rsrp_path")
        return FairResAttentionClassifier(
            num_classes,
            num_paths,
            architecture=architecture,
            use_path_id=input_mode_uses_path_id(input_mode),
        )
    if architecture == ARCH_HILOC:
        if input_mode == INPUT_MODE_PATH_ONLY:
            raise ValueError("strict paper baselines require SRS input; use --input-mode srs_rsrp or srs_rsrp_path")
        return FairHiLocClassifier(
            num_classes,
            num_paths,
            use_path_id=input_mode_uses_path_id(input_mode),
        )
    raise ValueError(f"unknown architecture {architecture!r}; expected one of {ARCHITECTURES}")


def build_device(gpu_arg: str | None) -> torch.device:
    if gpu_arg is None or gpu_arg == "":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("CUDA is not available. Falling back to CPU.")
        return torch.device("cpu")
    gpu_index = int(gpu_arg)
    if gpu_index < 0 or gpu_index >= torch.cuda.device_count():
        raise ValueError(f"Requested GPU {gpu_index}, but only {torch.cuda.device_count()} GPU(s) are visible.")
    return torch.device(f"cuda:{gpu_index}")


def configure_torch_runtime(device: torch.device, disable_cudnn: bool) -> None:
    if device.type != "cuda":
        return

    torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = True

    if disable_cudnn:
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        print("cuDNN disabled for compatibility; using PyTorch CUDA kernels without cuDNN.")
    else:
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    return float((pred == labels).float().mean().item())


def topk_accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor, k: int) -> float:
    topk = logits.topk(k=min(k, logits.shape[1]), dim=1).indices
    correct = topk.eq(labels.unsqueeze(1)).any(dim=1)
    return float(correct.float().mean().item())


def edge_count_accuracy_rows(edge_counts: torch.Tensor, labels: torch.Tensor, logits: torch.Tensor) -> list[dict]:
    if labels.numel() == 0:
        return []
    pred = logits.argmax(dim=1)
    rows = []
    for edge_count in sorted(torch.unique(edge_counts).tolist()):
        mask = edge_counts == int(edge_count)
        total = int(mask.sum().item())
        if total <= 0:
            continue
        correct = int((pred[mask] == labels[mask]).sum().item())
        rows.append(
            {
                "edge_count": int(edge_count),
                "correct": correct,
                "total": total,
                "accuracy": correct / total,
            }
        )
    return rows


def edge_accuracy_value(edge_rows: list[dict], edge_count: int, default: float = 0.0) -> float:
    for row in edge_rows:
        if int(row["edge_count"]) == int(edge_count):
            return float(row["accuracy"])
    return default


def classification_report_text(y_true: np.ndarray, y_pred: np.ndarray, classes: np.ndarray, digits: int = 4) -> str:
    lines = []
    header = f"{'label':>12} {'precision':>10} {'recall':>10} {'f1-score':>10} {'support':>10}"
    lines.append(header)
    lines.append("")
    precisions, recalls, f1s, supports = [], [], [], []

    for idx, label in enumerate(classes):
        support = int(np.sum(y_true == idx))
        pred_support = int(np.sum(y_pred == idx))
        true_positive = int(np.sum((y_true == idx) & (y_pred == idx)))
        precision = true_positive / pred_support if pred_support else 0.0
        recall = true_positive / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        supports.append(support)
        lines.append(f"{str(label):>12} {precision:>10.{digits}f} {recall:>10.{digits}f} {f1:>10.{digits}f} {support:>10d}")

    total = int(np.sum(supports))
    accuracy = float(np.mean(y_true == y_pred)) if total else 0.0
    macro_precision = float(np.mean(precisions)) if precisions else 0.0
    macro_recall = float(np.mean(recalls)) if recalls else 0.0
    macro_f1 = float(np.mean(f1s)) if f1s else 0.0
    weighted_precision = float(np.average(precisions, weights=supports)) if total else 0.0
    weighted_recall = float(np.average(recalls, weights=supports)) if total else 0.0
    weighted_f1 = float(np.average(f1s, weights=supports)) if total else 0.0

    lines.append("")
    lines.append(f"{'accuracy':>12} {'':>10} {'':>10} {accuracy:>10.{digits}f} {total:>10d}")
    lines.append(f"{'macro avg':>12} {macro_precision:>10.{digits}f} {macro_recall:>10.{digits}f} {macro_f1:>10.{digits}f} {total:>10d}")
    lines.append(f"{'weighted avg':>12} {weighted_precision:>10.{digits}f} {weighted_recall:>10.{digits}f} {weighted_f1:>10.{digits}f} {total:>10d}")
    return "\n".join(lines)


def confusion_matrix_np(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_idx, pred_idx in zip(y_true, y_pred):
        cm[int(true_idx), int(pred_idx)] += 1
    return cm


def save_history_csv(path: Path, history_rows: list[dict]) -> None:
    if not history_rows:
        return
    fieldnames = list(history_rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history_rows)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    logits_list = []
    labels_list = []
    edge_count_list = []

    with torch.no_grad():
        for batch in loader:
            inputs, labels = batch
            signal = inputs["signal"].to(device, non_blocking=True)
            rsrp = inputs["rsrp"].to(device, non_blocking=True)
            path_id = inputs["path_id"].to(device, non_blocking=True)
            edge_mask = inputs["edge_mask"].to(device, non_blocking=True)
            edge_count = inputs["edge_count"].to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(signal, rsrp, path_id, edge_mask)
            loss = criterion(logits, labels).mean()
            batch_size = labels.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size
            logits_list.append(logits.detach().cpu())
            labels_list.append(labels.detach().cpu())
            edge_count_list.append(edge_count.detach().cpu())

    logits_all = torch.cat(logits_list, dim=0) if logits_list else torch.empty((0, 0))
    labels_all = torch.cat(labels_list, dim=0) if labels_list else torch.empty((0,), dtype=torch.long)
    edge_counts_all = torch.cat(edge_count_list, dim=0) if edge_count_list else torch.empty((0,), dtype=torch.long)
    edge_rows = edge_count_accuracy_rows(edge_counts_all, labels_all, logits_all)
    return {
        "loss": total_loss / max(total_samples, 1),
        "accuracy": accuracy_from_logits(logits_all, labels_all) if total_samples else 0.0,
        "top3": topk_accuracy_from_logits(logits_all, labels_all, 3) if total_samples else 0.0,
        "top5": topk_accuracy_from_logits(logits_all, labels_all, 5) if total_samples else 0.0,
        "edge_rows": edge_rows,
        "edge1_accuracy": edge_accuracy_value(edge_rows, 1),
        "edge_macro_accuracy": float(np.mean([row["accuracy"] for row in edge_rows])) if edge_rows else 0.0,
        "edge_min_accuracy": float(min(row["accuracy"] for row in edge_rows)) if edge_rows else 0.0,
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer,
    criterion: nn.Module,
):
    model.train()
    total_loss = 0.0
    total_samples = 0
    logits_list = []
    labels_list = []
    edge_count_list = []

    for batch in loader:
        inputs, labels = batch
        signal = inputs["signal"].to(device, non_blocking=True)
        rsrp = inputs["rsrp"].to(device, non_blocking=True)
        path_id = inputs["path_id"].to(device, non_blocking=True)
        edge_mask = inputs["edge_mask"].to(device, non_blocking=True)
        edge_count = inputs["edge_count"].to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(signal, rsrp, path_id, edge_mask)
        loss = criterion(logits, labels).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        batch_size = labels.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size
        logits_list.append(logits.detach().cpu())
        labels_list.append(labels.detach().cpu())
        edge_count_list.append(edge_count.detach().cpu())

    logits_all = torch.cat(logits_list, dim=0) if logits_list else torch.empty((0, 0))
    labels_all = torch.cat(labels_list, dim=0) if labels_list else torch.empty((0,), dtype=torch.long)
    edge_counts_all = torch.cat(edge_count_list, dim=0) if edge_count_list else torch.empty((0,), dtype=torch.long)
    edge_rows = edge_count_accuracy_rows(edge_counts_all, labels_all, logits_all)
    return {
        "loss": total_loss / max(total_samples, 1),
        "accuracy": accuracy_from_logits(logits_all, labels_all) if total_samples else 0.0,
        "top3": topk_accuracy_from_logits(logits_all, labels_all, 3) if total_samples else 0.0,
        "top5": topk_accuracy_from_logits(logits_all, labels_all, 5) if total_samples else 0.0,
        "edge_rows": edge_rows,
        "edge1_accuracy": edge_accuracy_value(edge_rows, 1),
        "edge_macro_accuracy": float(np.mean([row["accuracy"] for row in edge_rows])) if edge_rows else 0.0,
        "edge_min_accuracy": float(min(row["accuracy"] for row in edge_rows)) if edge_rows else 0.0,
    }


def predict_probs(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    probs = []
    with torch.no_grad():
        for batch in loader:
            inputs = batch[0] if isinstance(batch, (tuple, list)) else batch
            signal = inputs["signal"].to(device, non_blocking=True)
            rsrp = inputs["rsrp"].to(device, non_blocking=True)
            path_id = inputs["path_id"].to(device, non_blocking=True)
            edge_mask = inputs["edge_mask"].to(device, non_blocking=True)
            logits = model(signal, rsrp, path_id, edge_mask)
            probs.append(torch.softmax(logits, dim=1).cpu().numpy())
    if not probs:
        return np.empty((0, 0), dtype=np.float32)
    return np.concatenate(probs, axis=0)


def write_probabilities(
    path: Path,
    examples: list[SequenceExample],
    labels: np.ndarray,
    probs: np.ndarray,
    y_true: np.ndarray | None = None,
    y_pred: np.ndarray | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    max_edges_in_examples = max(len(ex.edge_keys) for ex in examples)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        fieldnames = ["sample_id", "group", "edge_count"]
        fieldnames += [f"edge_{i + 1}" for i in range(max_edges_in_examples)]
        if y_true is not None and y_pred is not None:
            fieldnames += ["true_label", "pred_label", "correct"]
        elif y_pred is not None:
            fieldnames += ["pred_label"]
        fieldnames += ["top1_prob", "top2_prob", "prob_gap", "entropy"]
        fieldnames += [f"prob_{label}" for label in labels]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, (example, prob) in enumerate(zip(examples, probs)):
            row = {
                "sample_id": i,
                "group": example.group,
                "edge_count": len(example.edge_keys),
            }
            for edge_idx in range(max_edges_in_examples):
                row[f"edge_{edge_idx + 1}"] = example.edge_keys[edge_idx] if edge_idx < len(example.edge_keys) else ""
            if y_true is not None and y_pred is not None:
                row["true_label"] = labels[int(y_true[i])]
                row["pred_label"] = labels[int(y_pred[i])]
                row["correct"] = int(y_true[i] == y_pred[i])
            elif y_pred is not None:
                row["pred_label"] = labels[int(y_pred[i])]
            sorted_prob = np.sort(prob)[::-1]
            top1_prob = float(sorted_prob[0]) if sorted_prob.size >= 1 else 0.0
            top2_prob = float(sorted_prob[1]) if sorted_prob.size >= 2 else 0.0
            row["top1_prob"] = top1_prob
            row["top2_prob"] = top2_prob
            row["prob_gap"] = top1_prob - top2_prob
            row["entropy"] = float(-np.sum(prob * np.log(prob + 1e-12)))
            for label, value in zip(labels, prob):
                row[f"prob_{label}"] = float(value)
            writer.writerow(row)


def save_model(model_path: Path, model: nn.Module) -> None:
    torch.save({"state_dict": model.state_dict()}, model_path)


def load_artifacts(model_dir: str, device: torch.device):
    model_dir_path = Path(model_dir)
    model_path = model_dir_path / "best.pt"
    if not model_path.exists():
        model_path = model_dir_path / "last.pt"
    labels = np.load(model_dir_path / "labels.npy", allow_pickle=True)
    path_keys = np.load(model_dir_path / "path_keys.npy", allow_pickle=True)
    with open(model_dir_path / "config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
    with open(model_dir_path / "norm_stats.json", "r", encoding="utf-8") as f:
        stats = NormStats(**json.load(f))

    input_mode = str(config.get("input_mode", INPUT_MODE_SRS_RSRP_PATH))
    architecture = str(config.get("architecture", ARCH_ILCMG))
    model = build_classifier(len(labels), len(path_keys), architecture=architecture, input_mode=input_mode)
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    path_to_index = {str(key): idx + 1 for idx, key in enumerate(path_keys)}
    return model, labels, path_keys, path_to_index, stats, config


def monitor_value(metrics: dict, monitor: str) -> float:
    mapping = {
        "val_loss": metrics["val_loss"],
        "val_accuracy": metrics["val_accuracy"],
        "val_top3": metrics["val_top3"],
        "val_top5": metrics["val_top5"],
        "val_edge1_accuracy": metrics["val_edge1_accuracy"],
        "val_edge_macro_accuracy": metrics["val_edge_macro_accuracy"],
        "val_edge_min_accuracy": metrics["val_edge_min_accuracy"],
    }
    return float(mapping[monitor])


def os_cpu_count_safe() -> int:
    count = os.cpu_count()
    return count if isinstance(count, int) and count > 0 else 1


def train(args):
    set_seed(args.seed)
    device = build_device(args.gpu)
    configure_torch_runtime(device, args.disable_cudnn)
    print(f"Using device: {device}")
    fair_paper_baseline = args.architecture != ARCH_ILCMG
    use_path_id = input_mode_uses_path_id(args.input_mode)
    if args.architecture != ARCH_ILCMG and args.input_mode == INPUT_MODE_PATH_ONLY:
        raise ValueError("paper backbone architectures require signal input; use --input-mode srs_rsrp_path or srs_rsrp")

    train_examples = build_examples(
        args.train_dir,
        max_edges=args.max_edges,
        min_edges=args.min_edges,
        max_sequences_per_turn=args.max_sequences_per_turn,
        seed=args.seed,
        allow_edge_repeat=args.allow_edge_repeat,
        require_labels=True,
    )
    val_examples = build_examples(
        args.val_dir,
        max_edges=args.max_edges,
        min_edges=args.min_edges,
        max_sequences_per_turn=args.max_sequences_per_turn,
        seed=args.seed + 1,
        allow_edge_repeat=args.allow_edge_repeat,
        require_labels=True,
    ) if args.val_dir else None

    if args.skip_bad_csv:
        train_examples = filter_bad_examples(train_examples, report_path=Path(args.out_dir) / "bad_train_csv.csv", strict=False)
        if val_examples is not None:
            val_examples = filter_bad_examples(val_examples, report_path=Path(args.out_dir) / "bad_val_csv.csv", strict=False)
    else:
        train_examples = filter_bad_examples(train_examples, strict=True)
        if val_examples is not None:
            val_examples = filter_bad_examples(val_examples, strict=True)

    if val_examples is None:
        raise ValueError("validation examples are required for this script. Provide --val-dir with SRS-check.")

    labels, label_to_index = encode_labels(train_examples, val_examples)
    path_keys, path_to_index = encode_path_keys(train_examples, val_examples)
    stats = fit_normalizer(train_examples)

    y_train = np.asarray([label_to_index[ex.label] for ex in train_examples], dtype=np.int64)
    y_val = np.asarray([label_to_index[ex.label] for ex in val_examples], dtype=np.int64)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "labels.npy", labels)
    np.save(out_dir / "path_keys.npy", path_keys)
    with open(out_dir / "norm_stats.json", "w", encoding="utf-8") as f:
        json.dump(asdict(stats), f, indent=2)
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "max_edges": args.max_edges,
                "min_edges": args.min_edges,
                "num_classes": len(labels),
                "num_paths": len(path_keys),
                "segment_length": SEGMENT_LENGTH,
                "sequence_length": SEQUENCE_LENGTH,
                "max_sequences_per_turn": args.max_sequences_per_turn,
                "allow_edge_repeat": args.allow_edge_repeat,
                "input_mode": args.input_mode,
                "architecture": args.architecture,
                "fair_paper_baseline": fair_paper_baseline,
                "uses_path_id": use_path_id,
                "monitor": args.monitor,
            },
            f,
            indent=2,
        )

    print(f"Training examples: {len(train_examples)}")
    print(f"Validation examples: {len(val_examples)}")
    print(f"Classes: {labels.tolist()}")
    print(f"Known path keys: {len(path_keys)}")
    print(f"Input mode: {args.input_mode}")
    print(f"Architecture: {args.architecture}")

    num_workers = min(args.num_workers, os_cpu_count_safe())
    pin_memory = device.type == "cuda"
    train_ds = MultiEdgeDataset(train_examples, y_train, path_to_index, stats, args.max_edges, training=True, use_path_id=use_path_id)
    val_ds = MultiEdgeDataset(val_examples, y_val, path_to_index, stats, args.max_edges, training=False, use_path_id=use_path_id)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    model = build_classifier(len(labels), len(path_keys), architecture=args.architecture, input_mode=args.input_mode).to(device)
    print(model)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing, reduction="none")
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    if not trainable_params:
        raise RuntimeError("no trainable parameters")
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min" if args.monitor == "val_loss" else "max",
        factor=0.5,
        patience=args.lr_patience,
        threshold=args.min_delta,
        min_lr=1e-6,
    )

    best_value = None
    patience_count = 0
    history_rows = []
    best_state = None

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            device,
            optimizer,
            criterion,
        )
        val_metrics = evaluate(model, val_loader, device, criterion)

        row = {
            "epoch": epoch,
            "loss": train_metrics["loss"],
            "accuracy": train_metrics["accuracy"],
            "top3": train_metrics["top3"],
            "top5": train_metrics["top5"],
            "edge1_accuracy": train_metrics["edge1_accuracy"],
            "edge_macro_accuracy": train_metrics["edge_macro_accuracy"],
            "edge_min_accuracy": train_metrics["edge_min_accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_top3": val_metrics["top3"],
            "val_top5": val_metrics["top5"],
            "val_edge1_accuracy": val_metrics["edge1_accuracy"],
            "val_edge_macro_accuracy": val_metrics["edge_macro_accuracy"],
            "val_edge_min_accuracy": val_metrics["edge_min_accuracy"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        history_rows.append(row)
        print(
            f"Epoch {epoch}/{args.epochs} - "
            f"loss: {row['loss']:.4f} acc: {row['accuracy']:.4f} edge1: {row['edge1_accuracy']:.4f} top3: {row['top3']:.4f} top5: {row['top5']:.4f} - "
            f"val_loss: {row['val_loss']:.4f} val_acc: {row['val_accuracy']:.4f} val_edge1: {row['val_edge1_accuracy']:.4f} "
            f"val_top3: {row['val_top3']:.4f} val_top5: {row['val_top5']:.4f}"
        )

        current = monitor_value(row, args.monitor)
        if best_value is None:
            improved = True
        elif args.monitor == "val_loss":
            improved = current < (best_value - args.min_delta)
        else:
            improved = current > (best_value + args.min_delta)

        if improved:
            best_value = current
            patience_count = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            save_model(out_dir / "best.pt", model)
            print(f"Saved best model to {out_dir / 'best.pt'}")
        else:
            patience_count += 1

        scheduler.step(row["val_loss"] if args.monitor == "val_loss" else current)
        save_history_csv(out_dir / "history.csv", history_rows)

        if patience_count >= args.patience:
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    save_model(out_dir / "last.pt", model)

    val_probs = predict_probs(model, val_loader, device)
    y_pred = np.argmax(val_probs, axis=1)
    val_edge_counts = np.asarray([len(ex.edge_keys) for ex in val_examples], dtype=np.int64)
    val_edge_rows = []
    for edge_count in sorted(np.unique(val_edge_counts).tolist()):
        mask = val_edge_counts == int(edge_count)
        total = int(np.sum(mask))
        correct = int(np.sum(y_pred[mask] == y_val[mask]))
        val_edge_rows.append({"edge_count": int(edge_count), "correct": correct, "total": total, "accuracy": correct / max(total, 1)})
    print("\nValidation report:")
    print(classification_report_text(y_val, y_pred, labels, digits=4))
    print("Confusion matrix:")
    cm = confusion_matrix_np(y_val, y_pred, len(labels))
    print(cm)
    np.savetxt(out_dir / "validation_confusion_matrix.csv", cm, delimiter=",", fmt="%d")
    by_edge_path = out_dir / "validation_accuracy_by_edge_count.csv"
    print("Validation accuracy by edge count:")
    for row in val_edge_rows:
        print(f"  edges={row['edge_count']}: acc={row['accuracy']:.4f} ({row['correct']}/{row['total']})")
    pd.DataFrame(val_edge_rows).to_csv(by_edge_path, index=False, encoding="utf-8-sig")
    write_probabilities(out_dir / "validation_probabilities.csv", val_examples, labels, val_probs, y_true=y_val, y_pred=y_pred)


def predict(args):
    device = build_device(args.gpu)
    configure_torch_runtime(device, args.disable_cudnn)
    print(f"Using device: {device}")

    model, labels, path_keys, path_to_index, stats, config = load_artifacts(args.model_dir, device)
    max_edges = int(config["max_edges"])
    input_mode = str(config.get("input_mode", INPUT_MODE_SRS_RSRP_PATH))
    use_path_id = input_mode_uses_path_id(input_mode)
    print(f"Input mode: {input_mode}")

    examples = build_examples(
        args.test_dir,
        max_edges=max_edges,
        min_edges=args.min_edges,
        max_sequences_per_turn=args.max_sequences_per_turn,
        seed=args.seed,
        allow_edge_repeat=args.allow_edge_repeat,
        require_labels=False,
    )

    missing = sorted({key for ex in examples for key in ex.edge_keys} - set(path_to_index), key=path_sort_key)
    if use_path_id and missing and not args.skip_unknown_paths:
        raise ValueError(f"test contains path IDs not present in training: {missing}")
    if use_path_id and missing:
        examples = [ex for ex in examples if not any(key in missing for key in ex.edge_keys)]

    ds = MultiEdgeDataset(
        examples,
        labels=None,
        path_to_index=path_to_index,
        stats=stats,
        max_edges=max_edges,
        training=False,
        use_path_id=use_path_id,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=min(args.num_workers, os_cpu_count_safe()),
        pin_memory=device.type == "cuda",
    )
    probs = predict_probs(model, loader, device)
    y_pred = np.argmax(probs, axis=1)
    output = Path(args.output)
    write_probabilities(output, examples, labels, probs, y_pred=y_pred)
    print(f"Saved probabilities to: {output}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train or run a multi-edge RP probability classifier for AGV SRS path data.")
    sub = parser.add_subparsers(dest="command")

    train_parser = sub.add_parser("train")
    train_parser.add_argument("--train-dir", default=TRAIN_DIR)
    train_parser.add_argument("--val-dir", default=VAL_DIR)
    train_parser.add_argument("--out-dir", default=OUT_DIR)
    train_parser.add_argument("--max-edges", type=int, default=3, help="Maximum AGV moving edge count k.")
    train_parser.add_argument("--min-edges", type=int, default=1)
    train_parser.add_argument("--max-sequences-per-turn", type=int, default=512, help="0 means keep all valid edge sequences.")
    train_parser.add_argument("--allow-edge-repeat", action="store_true", help="Allow a sequence to reuse the same edge more than once.")
    train_parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    train_parser.add_argument("--epochs", type=int, default=EPOCHS)
    train_parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    train_parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    train_parser.add_argument("--label-smoothing", type=float, default=LABEL_SMOOTHING)
    train_parser.add_argument(
        "--input-mode",
        choices=INPUT_MODES,
        default=INPUT_MODE_SRS_RSRP_PATH,
        help=(
            "Input ablation mode: srs_rsrp_path uses SRS, RSRP, and path ID; "
            "srs_rsrp removes path ID; path_only uses only path ID as leakage check."
        ),
    )
    train_parser.add_argument(
        "--architecture",
        choices=ARCHITECTURES,
        default=ARCH_ILCMG,
        help=(
            "Classifier backbone. ilcmg is the original model; aarescnn, hiloc, and aares "
            "adapt the three paper network backbones to the same RP classification output."
        ),
    )
    train_parser.add_argument("--patience", type=int, default=20)
    train_parser.add_argument("--lr-patience", type=int, default=6)
    train_parser.add_argument("--min-delta", type=float, default=0.001)
    train_parser.add_argument(
        "--monitor",
        default="val_accuracy",
        choices=[
            "val_top3",
            "val_top5",
            "val_accuracy",
            "val_loss",
            "val_edge1_accuracy",
            "val_edge_macro_accuracy",
            "val_edge_min_accuracy",
        ],
    )
    train_parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    train_parser.add_argument("--strict", action="store_true")
    train_parser.add_argument("--skip-bad-csv", action=argparse.BooleanOptionalAction, default=True)
    train_parser.add_argument("--gpu", default=None, help="CUDA device index after any CUDA_VISIBLE_DEVICES filtering.")
    train_parser.add_argument("--num-workers", type=int, default=4)
    train_parser.add_argument(
        "--disable-cudnn",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Disable cuDNN and use non-cuDNN CUDA kernels. Use only if PyTorch reports cuDNN runtime errors.",
    )

    pred_parser = sub.add_parser("predict")
    pred_parser.add_argument("--test-dir", default=TEST_DIR)
    pred_parser.add_argument("--model-dir", default=OUT_DIR)
    pred_parser.add_argument("--output", default=str(Path(OUT_DIR) / "test_probabilities.csv"))
    pred_parser.add_argument("--min-edges", type=int, default=1)
    pred_parser.add_argument("--max-sequences-per-turn", type=int, default=512)
    pred_parser.add_argument("--allow-edge-repeat", action="store_true")
    pred_parser.add_argument("--skip-unknown-paths", action="store_true")
    pred_parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    pred_parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    pred_parser.add_argument("--strict", action="store_true")
    pred_parser.add_argument("--gpu", default=None, help="CUDA device index after any CUDA_VISIBLE_DEVICES filtering.")
    pred_parser.add_argument("--num-workers", type=int, default=4)
    pred_parser.add_argument(
        "--disable-cudnn",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Disable cuDNN and use non-cuDNN CUDA kernels. Use only if PyTorch reports cuDNN runtime errors.",
    )

    args = parser.parse_args()
    if args.command is None:
        args = parser.parse_args(["train", *sys.argv[1:]])
    return args


if __name__ == "__main__":
    cli_args = parse_args()
    if cli_args.command == "train":
        train(cli_args)
    elif cli_args.command == "predict":
        predict(cli_args)
    else:
        raise ValueError(f"unknown command: {cli_args.command}")

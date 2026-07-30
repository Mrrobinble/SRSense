# SRSense

SRSense is a multi-target indoor localization and autonomous path-selection
framework based on 5G SRS measurements. It combines a localization
network with target-count-specific reinforcement-learning policies that choose
the signal-collection path for an AGV.

This public repository contains the model definitions, training and evaluation
programs, coordinate metadata, and paper experiment figures.

## Repository structure

```text
SRSense/
|-- ILC-MG-train/          # Multi-edge localization model and training entry
|-- RL/multi_target/       # PPO/A2C agents, environments, training and evaluation
|-- data/                  # Coordinate metadata and dataset layout
`-- figures/               # Paper experiment figures
```

## Environment

Python 3.10 or later is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

## Dataset

Download the complete dataset and place the four dataset directories under
`data/` as described in [`data/README.md`](data/README.md). The checked-in CSV
files define the reference-point, target, and map-node coordinates.

## Train the localization model

```bash
python ILC-MG-train/Model.py train \
  --train-dir data/SRS-train \
  --val-dir data/SRS-check \
  --out-dir runs/localization
```

## Train target-count-specific PPO policies

```bash
python -m RL.multi_target.train_fixed_target_counts \
  --train-dir data/PPO-train_target \
  --classifier-dir runs/localization \
  --rp-coords data/ILC-MG_rp_coords.csv \
  --true-coords data/PPO_target_coords.csv \
  --map-node-coords data/map_node_coords.csv \
  --root-out-dir runs/policies \
  --target-counts 1 2 3 4 5
```

## Evaluation programs

- `evaluate_multi_target_ppo.py`: simultaneous multi-target evaluation.
- `evaluate_sequential_single_target_baseline.py`: sequential versus joint localization.
- `evaluate_edge_strategy_comparison.py`: routing-strategy comparison.
- `evaluate_classifier_comparison.py`: localization networks under one shared policy.
- `evaluate_classifier_policy_comparison.py`: each localization network with its own trained policy.

Run each program with `--help` for its complete command-line interface.

## Reproducibility scope

The repository provides source code for model construction, training, episode
generation, evaluation, and figure generation. Generated checkpoints, runtime
logs, intermediate caches, and per-episode output files are intentionally
excluded. The selected SVG figures report the paper experiments produced from
the corresponding evaluation programs.

## License

This project is released under the [MIT License](LICENSE).

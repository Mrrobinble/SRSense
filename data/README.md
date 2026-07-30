# Dataset

The complete SRSense dataset is distributed separately because the CSV files
are approximately 2 GB in total. After downloading the dataset, arrange it as
follows:

```text
data/
├── SRS-train/
├── SRS-check/
├── PPO-train_target/
├── PPO-test_target/
├── ILC-MG_rp_coords.csv
├── PPO_target_coords.csv
└── map_node_coords.csv
```

Dataset roles:

- `SRS-train`: training data for the multi-edge localization network.
- `SRS-check`: validation data for the localization network.
- `PPO-train_target`: training data for target-count-specific routing policies.
- `PPO-test_target`: held-out data used by the evaluation scripts.

The repository contains the coordinate metadata required by the training and
evaluation programs. The public dataset download URL will be added here after
the dataset archive is published.
[Zenodo](https://zenodo.org/records/21483156)

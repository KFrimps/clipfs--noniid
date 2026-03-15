# clipfs-noniid

# Mitigating Non-IID Effects in Federated Learning through CLIP-based Linear Probing.

This repository provides a unified and modular framework for experimenting with **Federated Learning (FL)** using **frozen CLIP visual embeddings** and **lightweight linear classifier heads**.  
It supports multiple heterogeneity scenarios, including:

- **CIFAR-10 General Non-IID**  
  (Quantity skew + Label skew via Dirichlet distributions)
- **CIFAR-10 Extreme Non-IID**  
  (Strict one-class-per-client setups)
- **PACS Domain Shift Federated Learning**  
  (Clients = domains: Photo, Cartoon, Sketch, Art Painting)

The goal is to analyze how **pretrained CLIP representations** behave under severe data heterogeneity, how distribution shift affects convergence, and how simple models can be trained efficiently on top of strong frozen feature encoders.

---

## Key Features

- **Frozen CLIP (ViT-B/32) embeddings** for fast, compute-efficient FL.
- **Flexible non-IID splitting**:
  - quantity skew
  - label skew
  - extreme one-class splits
  - domain-based splits (PACS)
- **Lightweight classifier head** trained locally on each client.
- **Custom FedAvg strategy** with:
  - global metric logging  
  - per-client evaluation  
  - CSV export for reproducible analysis  
- **Hyperparameter tuning module** included.
- **Modular repository design** for easy extension.

---

## Project Structure

```
clipfs-noniid/
├── src/
│   ├── config.py                  # Global config dataclass (CFG)
│   ├── data/
│   │   ├── features_cifar10.py    # CIFAR-10 CLIP feature extraction + dataset
│   │   ├── features_pacs.py       # PACS CLIP feature extraction + dataset
│   │   ├── partitions_cifar10.py  # Non-IID data splitting for CIFAR-10
│   │   └── partitions_pacs.py     # Domain-based data splitting for PACS
│   ├── fl/
│   │   ├── client.py              # Flower NumPyClient wrapper
│   │   └── server.py              # Custom FedAvg strategy + metric logging
│   ├── models/
│   │   └── clip_head.py           # Lightweight linear classifier head
│   ├── scripts/
│   │   ├── extract_features.py    # One-time CLIP feature extraction script
│   │   ├── run_cifar10_training.py          # CIFAR-10 non-IID FL training
│   │   ├── run_cifar10_extreme_training.py  # CIFAR-10 extreme non-IID FL training
│   │   ├── run_pacs_domain_shift_training.py  # PACS domain-shift FL training
│   │   └── tune_hparams.py        # Hyperparameter tuning via K-fold CV
│   └── utils/
│       ├── paths.py               # Project path constants
│       └── seed.py                # Reproducibility seed utilities
├── requirements.txt               # Python dependencies
└── README.md
```

---

## Prerequisites

- **Python 3.9+**
- **CUDA-capable GPU** (recommended for feature extraction; CPU works but is slower)
- **Git** (for installing the OpenAI CLIP package)

---

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/KFrimps/clipfs--noniid.git
   cd clipfs--noniid
   ```

2. **Create and activate a virtual environment** (recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/macOS
   # venv\Scripts\activate    # Windows
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

All scripts are run as Python modules from the **project root** directory.

### Step 1: Extract CLIP Features (one-time)

Before training, you must extract frozen CLIP features from the raw images and save them to disk. This only needs to be done **once**.

**For CIFAR-10:**

```bash
python -m src.scripts.extract_features --dataset cifar10
```

**For PACS:**

```bash
python -m src.scripts.extract_features --dataset pacs
```

**For both:**

```bash
python -m src.scripts.extract_features --dataset all
```

Extracted features are saved under `data/features/`.

### Step 2: Run Federated Training

Choose one of the three FL experiment scenarios:

**CIFAR-10 General Non-IID** (quantity + label skew via Dirichlet):

```bash
python -m src.scripts.run_cifar10_training
```

**CIFAR-10 Extreme Non-IID** (one class per client):

```bash
python -m src.scripts.run_cifar10_extreme_training
```

**PACS Domain Shift** (each client = one visual domain):

```bash
python -m src.scripts.run_pacs_domain_shift_training
```

Training logs (global metrics + per-client metrics) are saved as CSV files in the `runs/` directory.

---

## Configuration

All hyperparameters are centralized in `src/config.py` via the `CFG` dataclass:

| Parameter        | Default | Description                                      |
|------------------|---------|--------------------------------------------------|
| `mode`           | `"non-iid"` | Experiment mode label                        |
| `clients`        | `5`     | Number of federated clients                      |
| `rounds`         | `200`   | Number of FL communication rounds                |
| `client_fraction`| `1.0`   | Fraction of clients selected per round           |
| `local_epochs`   | `4`     | Local training epochs per client per round       |
| `batch_size`     | `64`    | Batch size for local training                    |
| `lr`             | `0.01`  | Learning rate (may be overridden by tuner)       |
| `momentum`       | `0.9`   | SGD momentum                                     |
| `weight_decay`   | `5e-4`  | SGD weight decay                                 |
| `alpha_qty`      | `1.0`   | Dirichlet alpha for quantity skew                |
| `alpha_label`    | `0.1`   | Dirichlet alpha for label skew                   |
| `min_per_client` | `100`   | Minimum samples per client (soft constraint)     |
| `seed`           | `0`     | Random seed for reproducibility                  |
| `num_workers`    | `2`     | DataLoader workers                               |
| `device`         | auto    | `cuda` if available, else `cpu`                  |

You can modify these defaults directly in `src/config.py` or extend `CFG` to accept command-line arguments.

---

## Experiment Workflow

1. **Feature Extraction** — Run CLIP ViT-B/32 once to produce 512-dim embeddings.
2. **Data Partitioning** — Split features across clients using non-IID strategies.
3. **Hyperparameter Tuning** — Global K-fold CV selects optimal LR and epoch count.
4. **Federated Training** — Flower simulation runs FedAvg with lightweight linear heads.
5. **Logging** — Global and per-client accuracy/loss are saved to CSV each round.

---

## License

This project is for research and educational purposes.

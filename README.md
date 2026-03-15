# clipfs-noniid

# Mitigating Non-IID Effects in Federated Learning through CLIP-based Linear Probing

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

- **Frozen CLIP (ViT-B/32) embeddings** (via the [OpenAI CLIP](https://github.com/openai/CLIP) package) for fast, compute-efficient FL.
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
- **Hyperparameter tuning module** via global K-fold CV.
- **CLI arguments** for quick experimentation (`--clients`, `--rounds`, `--alpha-label`, etc.).
- **Modular repository design** for easy extension.

---

## Requirements

| Requirement        | Details                                                    |
|--------------------|------------------------------------------------------------|
| **Python**         | 3.10 or 3.11 (tested)                                     |
| **GPU**            | CUDA-capable GPU recommended (CPU works but is much slower)|
| **CUDA**           | CUDA 11.8+ (or 12.x) for GPU-accelerated PyTorch          |
| **Git**            | Required for installing the OpenAI CLIP package            |

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

3. **Install PyTorch with CUDA support:**

   > **Important:** `pip install torch` may install a CPU-only build.  
   > Use the official [PyTorch install command](https://pytorch.org/get-started/locally/) to get the correct CUDA wheels for your system. For example:
   >
   > ```bash
   > # Example for CUDA 11.8
   > pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   >
   > # Example for CUDA 12.1
   > pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   > ```

4. **Install remaining dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

   This installs: `flwr[simulation]`, `ray`, OpenAI CLIP (`clip`), `datasets` (Hugging Face), `scikit-learn`, `pandas`, `numpy`, `Pillow`, and `tqdm`.

---

## Quickstart (CIFAR-10)

A new user can reproduce the CIFAR-10 non-IID experiment in **3 commands**:

```bash
# 1. Install dependencies (see Installation above)
pip install -r requirements.txt

# 2. Extract CLIP features (one-time, ~2 min on GPU)
python -m src.scripts.extract_features --dataset cifar10

# 3. Run federated training (5 clients, 200 rounds by default)
python -m src.scripts.run_cifar10_training --clients 5 --rounds 20
```

### Smoke Test (quick validation, finishes in minutes)

```bash
python -m src.scripts.run_cifar10_training --clients 2 --rounds 2 --batch-size 32
```

---

## Usage

All scripts are run as **Python modules** from the **project root** directory.

### Step 1: Extract CLIP Features (one-time)

Before training, extract frozen CLIP features from raw images and save them to disk. This only needs to be done **once** per dataset.

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

Features are saved to `data/features/` as `.pt` files.

### Step 2: Run Federated Training

Choose one of three experiment scenarios:

**CIFAR-10 General Non-IID** (quantity + label skew via Dirichlet):

```bash
python -m src.scripts.run_cifar10_training
# With CLI overrides:
python -m src.scripts.run_cifar10_training --clients 5 --rounds 200 --alpha-label 0.1
```

**CIFAR-10 Extreme Non-IID** (one class per client):

```bash
python -m src.scripts.run_cifar10_extreme_training
# With CLI overrides:
python -m src.scripts.run_cifar10_extreme_training --clients 5 --rounds 200
```

**PACS Domain Shift** (each client = one visual domain):

```bash
python -m src.scripts.run_pacs_domain_shift_training
# With CLI overrides:
python -m src.scripts.run_pacs_domain_shift_training --clients 4 --rounds 100
```

### CLI Arguments

All training scripts accept the following optional arguments:

| Argument          | Default   | Description                                |
|-------------------|-----------|--------------------------------------------|
| `--clients`       | `5`       | Number of federated clients                |
| `--rounds`        | `200`     | Number of FL communication rounds          |
| `--batch-size`    | `64`      | Batch size for local training              |
| `--seed`          | `0`       | Random seed for reproducibility            |
| `--alpha-label`   | `0.1`     | Dirichlet alpha for label skew (CIFAR-10 only) |
| `--alpha-qty`     | `1.0`     | Dirichlet alpha for quantity skew (CIFAR-10 only) |

---

## Outputs

### Feature Files

| File                                | Location                          | Description                            |
|-------------------------------------|-----------------------------------|----------------------------------------|
| `cifar10_clip_features.pt`          | `data/features/`                  | CIFAR-10 CLIP embeddings + labels      |
| `pacs_clip_features.pt`             | `data/features/`                  | PACS CLIP embeddings + labels + domain indices |

### Training Logs (CSV)

| File                     | Location  | Columns                                                |
|--------------------------|-----------|--------------------------------------------------------|
| `global_metrics.csv`     | `runs/`   | `round`, `accuracy`, `loss`                            |
| `per_client_metrics.csv` | `runs/`   | `round`, `cid`, `client_idx`, `num_examples`, `loss`, `accuracy` |

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
│       ├── paths.py               # Project path constants (PROJECT_ROOT, DATA_DIR, FEATURES_DIR, RUNS_DIR)
│       └── seed.py                # Reproducibility seed utilities
├── requirements.txt               # Python dependencies
└── README.md
```

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

You can modify these defaults directly in `src/config.py`, or override them via CLI arguments when running training scripts.

---

## Experiment Workflow

1. **Feature Extraction** — Run CLIP ViT-B/32 once to produce 512-dim embeddings.
2. **Data Partitioning** — Split features across clients using non-IID strategies.
3. **Hyperparameter Tuning** — Global K-fold CV selects optimal LR and epoch count.
4. **Federated Training** — Flower simulation (backed by Ray) runs FedAvg with lightweight linear heads.
5. **Logging** — Global and per-client accuracy/loss are saved to CSV each round in `runs/`.

---

## Results

| Dataset   | Split Type                  | #Clients | Rounds | Expected Accuracy |
|-----------|-----------------------------|----------|--------|-------------------|
| CIFAR-10  | Quantity + Label skew       | 5        | 200    | ~85–90%           |
| CIFAR-10  | Extreme (1 class/client)    | 5        | 200    | ~50–70%           |
| PACS      | Domain shift (4 domains)    | 4        | 200    | ~70–85%           |

Full results are logged as CSV files in `runs/`. See [Outputs](#outputs) for the exact format.

---

## Troubleshooting

| Issue                                   | Solution                                                                                       |
|-----------------------------------------|------------------------------------------------------------------------------------------------|
| **CUDA not found / CPU-only PyTorch**   | Install PyTorch using the [official command](https://pytorch.org/get-started/locally/) for your CUDA version. |
| **`ModuleNotFoundError: clip`**         | Ensure `clip` is installed via `pip install git+https://github.com/openai/CLIP.git`.           |
| **Ray errors / resource issues**        | Reduce `num_gpus` in `backend_config` or set it to `0.0` for CPU-only runs.                    |
| **Dataset download fails (PACS)**       | Check your internet connection. PACS is downloaded from Hugging Face via the `datasets` library.|
| **Feature file not found**              | Run `python -m src.scripts.extract_features --dataset cifar10` before training.                |
| **Out of GPU memory**                   | Reduce `batch_size` via CLI (`--batch-size 32`) or use CPU for feature extraction.             |

---

## CLIP Implementation

This project uses the **[OpenAI CLIP](https://github.com/openai/CLIP)** package (`clip`) with the **ViT-B/32** model for feature extraction. The CLIP model is used only during the one-time feature extraction step and is not part of the federated training loop.

---

## License

This project is for research and educational purposes.

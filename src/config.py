from dataclasses import dataclass
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@dataclass
class CFG:
    mode: str = "non-iid"
    clients: int = 5
    rounds: int = 200
    client_fraction: float = 1.0
    local_epochs: int = 4
    batch_size: int = 64
    lr: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 5e-4
    alpha_qty: float = 1.0
    alpha_label: float = 0.1
    seed: int = 0
    num_workers: int = 2
    device: torch.device = DEVICE

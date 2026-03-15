"""
Extract CLIP features for CIFAR-10 and/or PACS and save them to disk.
Run this script ONCE before federated training.

Usage:
    python -m src.scripts.extract_features --dataset cifar10
    python -m src.scripts.extract_features --dataset pacs
    python -m src.scripts.extract_features --dataset all
"""

import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import clip

from src.utils.paths import FEATURES_DIR
from src.config import CFG


def extract_cifar10(cfg):
    """Extract CLIP features for CIFAR-10 and save to disk."""
    device = cfg.device

    print("-------------------------------------------------")
    print("Loading CLIP model for CIFAR-10 Feature Extraction...")
    print("-------------------------------------------------")
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()

    print("Downloading/Loading CIFAR-10 dataset...")
    train_data = datasets.CIFAR10(root="./data", train=True, download=True, transform=preprocess)
    test_data = datasets.CIFAR10(root="./data", train=False, download=True, transform=preprocess)

    full_dataset = torch.utils.data.ConcatDataset([train_data, test_data])
    loader = DataLoader(full_dataset, batch_size=128, shuffle=False, num_workers=2)

    print(f"Extracting features for {len(full_dataset)} images...")

    all_features = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            features = model.encode_image(images)
            all_features.append(features.cpu())
            all_labels.append(labels)

    features_tensor = torch.cat(all_features)
    targets_tensor = torch.cat(all_labels).long()

    print(f"Extraction Complete. Feature Shape: {features_tensor.shape}")

    del model
    torch.cuda.empty_cache()

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FEATURES_DIR / "cifar10_clip_features.pt"
    torch.save({"features": features_tensor, "labels": targets_tensor}, out_path)
    print(f"Saved CIFAR-10 CLIP features to: {out_path}")


def extract_pacs(cfg):
    """Extract CLIP features for PACS and save to disk."""
    from datasets import load_dataset as hf_load_dataset
    from src.data.features_pacs import HFImageDataset

    device = cfg.device

    print("-------------------------------------------------")
    print("Loading CLIP model for PACS Feature Extraction...")
    print("-------------------------------------------------")
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()

    print("Downloading PACS from Hugging Face...")
    full_hf_dataset = hf_load_dataset("flwrlabs/pacs", split="train")

    domains = ["art_painting", "cartoon", "photo", "sketch"]
    domain_indices = {}

    all_features = []
    all_labels = []
    current_idx = 0

    for domain_name in domains:
        print(f"--- Processing Domain: {domain_name} ---")
        domain_data = full_hf_dataset.filter(lambda x: x['domain'] == domain_name)

        if len(domain_data) == 0:
            print(f"Warning: No images found for domain '{domain_name}'.")
            continue

        dataset = HFImageDataset(domain_data, transform=preprocess)
        loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=2)

        print(f"Extracting features for {domain_name} ({len(dataset)} images)...")
        domain_start = current_idx

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(device)
                features = model.encode_image(images)
                all_features.append(features.cpu())
                all_labels.append(labels)
                current_idx += len(labels)

        domain_indices[domain_name] = list(range(domain_start, current_idx))

    if not all_features:
        raise RuntimeError("No features extracted.")

    features_tensor = torch.cat(all_features)
    targets_tensor = torch.cat(all_labels).long()

    del model
    torch.cuda.empty_cache()

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FEATURES_DIR / "pacs_clip_features.pt"
    torch.save({
        "features": features_tensor,
        "labels": targets_tensor,
        "domain_indices": domain_indices,
    }, out_path)
    print(f"Saved PACS CLIP features to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract CLIP features for FL experiments.")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["cifar10", "pacs", "all"],
        default="cifar10",
        help="Which dataset to extract features for (default: cifar10).",
    )
    args = parser.parse_args()

    cfg = CFG()

    if args.dataset in ("cifar10", "all"):
        extract_cifar10(cfg)
    if args.dataset in ("pacs", "all"):
        extract_pacs(cfg)


if __name__ == "__main__":
    main()

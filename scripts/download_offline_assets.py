#!/usr/bin/env python3
"""
Scarica asset per uso offline sul cluster (macchina con internet).

- CIFAR-100 in --dataset-dir (layout compatibile con torchvision.datasets.CIFAR100, download=False).
- Pesi ImageNet di ResNet-50 in --weights-dir (state_dict salvato con torch.save).

Esempi:
  python scripts/download_offline_assets.py
  python scripts/download_offline_assets.py --dataset-dir ./dataset --weights-dir ./weights

Poi trasferisci le cartelle `dataset/` e `weights/` sul cluster (es. scp -r).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download CIFAR-100 e pesi ResNet-50 per bundle offline.")
    root = Path(__file__).resolve().parent.parent
    p.add_argument(
        "--dataset-dir",
        type=Path,
        default=root / "dataset",
        help="Directory dove salvare CIFAR-100 (default: <repo>/dataset).",
    )
    p.add_argument(
        "--weights-dir",
        type=Path,
        default=root / "weights",
        help="Directory dove salvare il file .pth di ResNet-50 (default: <repo>/weights).",
    )
    p.add_argument(
        "--resnet-filename",
        type=str,
        default="resnet50_imagenet1k_v1.pth",
        help="Nome file per lo state_dict di ResNet-50.",
    )
    p.add_argument("--skip-dataset", action="store_true", help="Non scaricare CIFAR-100.")
    p.add_argument("--skip-weights", action="store_true", help="Non scaricare pesi ResNet-50.")
    return p.parse_args()


def download_cifar100(dataset_dir: Path) -> None:
    from torchvision import transforms
    from torchvision.datasets import CIFAR100

    dataset_dir.mkdir(parents=True, exist_ok=True)
    # Trasform dummy: serve solo a istanziare il dataset e innescare il download.
    t = transforms.ToTensor()
    CIFAR100(root=dataset_dir, train=True, download=True, transform=t)
    CIFAR100(root=dataset_dir, train=False, download=True, transform=t)
    print(f"CIFAR-100 pronto in: {dataset_dir.resolve()}")
    print("  (usa lo stesso path come root= in CIFAR100(..., download=False) sul cluster.)")


def download_resnet50_weights(weights_dir: Path, filename: str) -> None:
    weights_dir.mkdir(parents=True, exist_ok=True)
    out_path = weights_dir / filename

    try:
        from torchvision.models import ResNet50_Weights, resnet50

        w = ResNet50_Weights.IMAGENET1K_V1
        model = resnet50(weights=w)
        meta = str(w)
    except ImportError:
        from torchvision.models import resnet50

        model = resnet50(pretrained=True)
        meta = "pretrained=True (API legacy)"

    torch.save(model.state_dict(), out_path)
    print(f"Pesi ResNet-50 salvati: {out_path.resolve()} ({out_path.stat().st_size / 2**20:.1f} MiB)")
    print(f"  Origine pesi torchvision: {meta}")
    print("  Carica sul cluster con: model.load_state_dict(torch.load(path, map_location=...)) su resnet50(weights=None).")


def main() -> None:
    args = parse_args()
    if not args.skip_dataset:
        download_cifar100(args.dataset_dir)
    if not args.skip_weights:
        download_resnet50_weights(args.weights_dir, args.resnet_filename)
    if args.skip_dataset and args.skip_weights:
        print("Niente da fare: entrambi --skip-dataset e --skip-weights.")


if __name__ == "__main__":
    main()

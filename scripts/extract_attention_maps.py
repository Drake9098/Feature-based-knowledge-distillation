"""Estrae e salva mappe di attenzione F²_sum (teacher vs student) per Attention Transfer.

Usa la stessa logica del training AT:
- hook sui layer ResNet indicati nel config (es. layer1..layer4)
- mappa = somma dei quadrati lungo i canali (Zagoruyko & Komodakis, 2017)
- se le risoluzioni student/teacher differiscono, la mappa student viene
  interpolata a quella del teacher (come in src.training.attention_utils.at_loss)

Esempi:
  python scripts/extract_attention_maps.py \\
    --config configs/at_kd.yaml \\
    --student-checkpoint experiments/checkpoints/at_kd_t20/2026-05-23_14-54-54/at_kd_t20_at_best.pt

  python scripts/extract_attention_maps.py \\
    --config configs/at_kd.yaml \\
    --student-checkpoint PATH --layers layer3 --num-images 3 --indices 0,42,100

  python scripts/extract_attention_maps.py \\
    --config configs/at_kd.yaml \\
    --compare-students \\
      "AT+KD=experiments/checkpoints/at_kd/.../at_kd_at_best.pt" \\
      "Vanilla KD=experiments/checkpoints/phase2_kd/.../phase2_kd_kd_best.pt" \\
      "FitNet Deep=experiments/checkpoints/fitnet_deep_s2_t20/.../fitnet_deep_s2_t20_kd_best.pt" \\
    --layers layer3 layer4 --indices 0,42,100,500,999 --upsample-to-input
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.cifar100 import build_cifar100_loaders
from src.models.baseline import build_baseline
from src.models.distillation_utils import FeatureExtractor
from src.models.teacher import build_teacher
from src.utils.checkpoint import load_checkpoint
from src.utils.config import load_yaml_config
from src.utils.seed import set_seed

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Visualizza mappe di attenzione teacher/student per Attention Transfer."
    )
    p.add_argument(
        "--config",
        type=Path,
        default=_ROOT / "configs" / "at_kd.yaml",
        help="YAML AT (teacher checkpoint, layer, data).",
    )
    p.add_argument(
        "--student-checkpoint",
        type=Path,
        default=None,
        help="Checkpoint best/final di un singolo student (modalità teacher vs student).",
    )
    p.add_argument(
        "--compare-students",
        nargs="+",
        default=None,
        metavar="LABEL=PATH",
        help=(
            "Confronto multi-metodo: una o più coppie LABEL=PATH "
            "(es. 'AT+KD=experiments/.../at_kd_at_best.pt'). "
            "Genera una griglia originale | teacher | student₁ | student₂ | …"
        ),
    )
    p.add_argument(
        "--teacher-checkpoint",
        type=Path,
        default=None,
        help="Override di model.teacher_checkpoint nel YAML.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=_ROOT / "figures" / "attention_maps",
        help="Cartella radice per le immagini generate (default: figures/attention_maps).",
    )
    p.add_argument(
        "--layers",
        nargs="+",
        default=None,
        help="Layer da visualizzare (default: attention_transfer.layers nel YAML).",
    )
    p.add_argument(
        "--num-images",
        type=int,
        default=5,
        help="Quante immagini del test set processare (ignorato se --indices è impostato).",
    )
    p.add_argument(
        "--indices",
        type=str,
        default=None,
        help="Indici espliciti nel dataset di test, separati da virgola (es. 0,42,100).",
    )
    p.add_argument(
        "--split",
        choices=("test", "train"),
        default="test",
        help="Split CIFAR-100 da cui campionare le immagini.",
    )
    p.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Device per l'inferenza.",
    )
    p.add_argument(
        "--upsample-to-input",
        action="store_true",
        help="Upsample le mappe di attenzione a 32x32 per una visualizzazione più leggibile.",
    )
    return p.parse_args()


def resolve_device(name: str) -> torch.device:
    n = name.lower().strip()
    if n == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if n == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Richiesto device=cuda ma CUDA non è disponibile.")
    return torch.device(n)


def denormalize_image(tensor_chw: torch.Tensor) -> np.ndarray:
    img = tensor_chw.detach().cpu().clone()
    for c, (mean, std) in enumerate(zip(IMAGENET_MEAN, IMAGENET_STD)):
        img[c] = img[c] * std + mean
    img = img.clamp(0.0, 1.0).numpy()
    return np.transpose(img, (1, 2, 0))


def attention_map_2d(feat: torch.Tensor) -> torch.Tensor:
    """F²_sum → (B, H, W), coerente con src.training.attention_utils.attention_map."""
    return feat.pow(2).sum(dim=1)


def align_student_attention(
    student_feat: torch.Tensor,
    teacher_feat: torch.Tensor,
) -> torch.Tensor:
    """Allinea la mappa student a H×W del teacher (interpolazione bilineare sui quadrati)."""
    s_map = attention_map_2d(student_feat)
    if student_feat.shape[-2:] == teacher_feat.shape[-2:]:
        return s_map

    s_energy = student_feat.pow(2).sum(dim=1, keepdim=True)
    s_energy = F.interpolate(
        s_energy,
        size=teacher_feat.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
    return s_energy.squeeze(1)


def maybe_upsample(map_2d: np.ndarray, target_size: tuple[int, int] | None) -> np.ndarray:
    if target_size is None:
        return map_2d
    t = torch.from_numpy(map_2d).float().unsqueeze(0).unsqueeze(0)
    t = F.interpolate(t, size=target_size, mode="bilinear", align_corners=False)
    return t.squeeze().numpy()


def parse_compare_students(raw: list[str] | None) -> list[tuple[str, Path]]:
    if not raw:
        return []
    pairs: list[tuple[str, Path]] = []
    for item in raw:
        if "=" not in item:
            raise ValueError(
                f"Formato compare-students non valido: {item!r}. Usa LABEL=PATH."
            )
        label, path_str = item.split("=", 1)
        label = label.strip()
        path = Path(path_str.strip())
        if not label:
            raise ValueError(f"Etichetta vuota in compare-students: {item!r}")
        pairs.append((label, path))
    return pairs


def load_student(
    cfg: dict,
    student_ckpt: Path,
    device: torch.device,
) -> torch.nn.Module:
    student = build_baseline(cfg["model"])
    s_state = load_checkpoint(student_ckpt, map_location="cpu")
    student.load_state_dict(s_state["model_state_dict"], strict=True)
    return student.to(device).eval()


def load_teacher(
    cfg: dict,
    teacher_ckpt: Path,
    device: torch.device,
) -> torch.nn.Module:
    teacher = build_teacher(cfg["model"])
    t_state = load_checkpoint(teacher_ckpt, map_location="cpu")
    teacher.load_state_dict(t_state["model_state_dict"], strict=True)
    return teacher.to(device).eval()


def load_models(
    cfg: dict,
    teacher_ckpt: Path,
    student_ckpt: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, torch.nn.Module]:
    teacher = load_teacher(cfg, teacher_ckpt, device)
    student = load_student(cfg, student_ckpt, device)
    return teacher, student


def parse_indices(raw: str | None, num_images: int) -> list[int]:
    if raw:
        indices = [int(x.strip()) for x in raw.split(",") if x.strip()]
        if not indices:
            raise ValueError("--indices non contiene indici validi.")
        return indices
    if num_images <= 0:
        raise ValueError("--num-images deve essere > 0.")
    return list(range(num_images))


def save_attention_figure(
    *,
    out_path: Path,
    image_rgb: np.ndarray,
    teacher_map: np.ndarray,
    student_map: np.ndarray,
    layer: str,
    label: int,
    class_names: list[str],
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    class_name = class_names[label] if 0 <= label < len(class_names) else str(label)
    fig.suptitle(f"{layer} — label: {class_name} ({label})", fontsize=12)

    axes[0].imshow(image_rgb)
    axes[0].set_title("Immagine originale")
    axes[0].axis("off")

    im1 = axes[1].imshow(teacher_map, cmap="jet")
    axes[1].set_title("Teacher (ResNet-50)")
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(student_map, cmap="jet")
    axes[2].set_title("Student (ResNet-18)")
    axes[2].axis("off")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_comparison_figure(
    *,
    out_path: Path,
    image_rgb: np.ndarray,
    teacher_map: np.ndarray,
    student_maps: list[tuple[str, np.ndarray]],
    layer: str,
    label: int,
    class_names: list[str],
) -> None:
    n_cols = 2 + len(student_maps)
    fig, axes = plt.subplots(1, n_cols, figsize=(3.5 * n_cols, 4))
    if n_cols == 1:
        axes = [axes]

    class_name = class_names[label] if 0 <= label < len(class_names) else str(label)
    fig.suptitle(f"{layer} — {class_name} ({label})", fontsize=12)

    axes[0].imshow(image_rgb)
    axes[0].set_title("Immagine")
    axes[0].axis("off")

    im_t = axes[1].imshow(teacher_map, cmap="jet")
    axes[1].set_title("Teacher")
    axes[1].axis("off")
    fig.colorbar(im_t, ax=axes[1], fraction=0.046, pad=0.04)

    for ax, (student_label, student_map) in zip(axes[2:], student_maps):
        im_s = ax.imshow(student_map, cmap="jet")
        ax.set_title(student_label)
        ax.axis("off")
        fig.colorbar(im_s, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_single_student(args: argparse.Namespace, cfg: dict, device: torch.device) -> None:
    teacher_ckpt_raw = args.teacher_checkpoint or cfg["model"].get("teacher_checkpoint")
    if not teacher_ckpt_raw:
        raise ValueError("Specifica --teacher-checkpoint o model.teacher_checkpoint nel YAML.")
    teacher_ckpt = Path(teacher_ckpt_raw)
    if args.student_checkpoint is None:
        raise ValueError("Specifica --student-checkpoint oppure --compare-students.")
    student_ckpt = Path(args.student_checkpoint)
    for path, label in ((teacher_ckpt, "teacher"), (student_ckpt, "student")):
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint {label} non trovato: {path.resolve()}")

    at_layers = list(args.layers or cfg["attention_transfer"]["layers"])
    if not at_layers:
        raise ValueError("Nessun layer AT da visualizzare.")

    teacher, student = load_models(cfg, teacher_ckpt, student_ckpt, device)
    t_extractor = FeatureExtractor(teacher, at_layers)
    s_extractor = FeatureExtractor(student, at_layers)

    dataset, class_names = _build_dataset(args, cfg)
    run_name = f"{student_ckpt.stem}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    out_dir = args.output_dir / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir.resolve()}")

    indices = parse_indices(args.indices, args.num_images)
    upsample_size = (32, 32) if args.upsample_to_input else None

    with torch.no_grad():
        for idx in indices:
            image, label = _get_sample(dataset, idx)
            input_batch = image.unsqueeze(0).to(device)

            _, t_feats = t_extractor(input_batch)
            _, s_feats = s_extractor(input_batch)
            image_rgb = denormalize_image(image)

            for layer in at_layers:
                t_map = attention_map_2d(t_feats[layer])[0].cpu().numpy()
                s_map = align_student_attention(s_feats[layer], t_feats[layer])[0].cpu().numpy()
                t_map = maybe_upsample(t_map, upsample_size)
                s_map = maybe_upsample(s_map, upsample_size)

                out_path = out_dir / f"idx{idx:05d}_layer-{layer}.png"
                save_attention_figure(
                    out_path=out_path,
                    image_rgb=image_rgb,
                    teacher_map=t_map,
                    student_map=s_map,
                    layer=layer,
                    label=int(label),
                    class_names=list(class_names),
                )
                print(f"Salvata: {out_path.name}")

    t_extractor.remove_hooks()
    s_extractor.remove_hooks()

    _write_run_info(
        out_dir,
        args,
        teacher_ckpt,
        at_layers,
        indices,
        student_checkpoints=[("student", student_ckpt)],
    )


def run_compare_students(args: argparse.Namespace, cfg: dict, device: torch.device) -> None:
    teacher_ckpt_raw = args.teacher_checkpoint or cfg["model"].get("teacher_checkpoint")
    if not teacher_ckpt_raw:
        raise ValueError("Specifica --teacher-checkpoint o model.teacher_checkpoint nel YAML.")
    teacher_ckpt = Path(teacher_ckpt_raw)
    if not teacher_ckpt.is_file():
        raise FileNotFoundError(f"Checkpoint teacher non trovato: {teacher_ckpt.resolve()}")

    compare_pairs = parse_compare_students(args.compare_students)
    for label, path in compare_pairs:
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint student ({label}) non trovato: {path.resolve()}")

    at_layers = list(args.layers or cfg["attention_transfer"]["layers"])
    if not at_layers:
        raise ValueError("Nessun layer AT da visualizzare.")

    teacher = load_teacher(cfg, teacher_ckpt, device)
    t_extractor = FeatureExtractor(teacher, at_layers)

    student_entries: list[tuple[str, torch.nn.Module, FeatureExtractor]] = []
    for label, ckpt in compare_pairs:
        model = load_student(cfg, ckpt, device)
        student_entries.append((label, model, FeatureExtractor(model, at_layers)))

    dataset, class_names = _build_dataset(args, cfg)
    run_name = f"compare_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    out_dir = args.output_dir / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir.resolve()}")

    indices = parse_indices(args.indices, args.num_images)
    upsample_size = (32, 32) if args.upsample_to_input else None

    with torch.no_grad():
        for idx in indices:
            image, label = _get_sample(dataset, idx)
            input_batch = image.unsqueeze(0).to(device)

            _, t_feats = t_extractor(input_batch)
            image_rgb = denormalize_image(image)

            for layer in at_layers:
                t_map = attention_map_2d(t_feats[layer])[0].cpu().numpy()
                t_map = maybe_upsample(t_map, upsample_size)

                student_maps: list[tuple[str, np.ndarray]] = []
                for student_label, _model, s_extractor in student_entries:
                    _, s_feats = s_extractor(input_batch)
                    s_map = align_student_attention(s_feats[layer], t_feats[layer])[0].cpu().numpy()
                    s_map = maybe_upsample(s_map, upsample_size)
                    student_maps.append((student_label, s_map))

                out_path = out_dir / f"idx{idx:05d}_layer-{layer}.png"
                save_comparison_figure(
                    out_path=out_path,
                    image_rgb=image_rgb,
                    teacher_map=t_map,
                    student_maps=student_maps,
                    layer=layer,
                    label=int(label),
                    class_names=list(class_names),
                )
                print(f"Salvata: {out_path.name}")

    t_extractor.remove_hooks()
    for _label, _model, s_extractor in student_entries:
        s_extractor.remove_hooks()

    _write_run_info(
        out_dir,
        args,
        teacher_ckpt,
        at_layers,
        indices,
        student_checkpoints=compare_pairs,
    )


def _build_dataset(args: argparse.Namespace, cfg: dict):
    data_cfg = cfg["data"]
    train_loader, test_loader = build_cifar100_loaders(
        root=data_cfg["root"],
        batch_size=1,
        num_workers=int(data_cfg.get("num_workers", 0)),
        data_config={**data_cfg, "pin_memory": False},
    )
    loader = test_loader if args.split == "test" else train_loader
    dataset = loader.dataset
    class_names = getattr(dataset, "classes", [str(i) for i in range(100)])
    return dataset, class_names


def _get_sample(dataset, idx: int):
    if idx < 0 or idx >= len(dataset):
        raise IndexError(f"Indice {idx} fuori range [0, {len(dataset) - 1}]")
    return dataset[idx]


def _write_run_info(
    out_dir: Path,
    args: argparse.Namespace,
    teacher_ckpt: Path,
    at_layers: list[str],
    indices: list[int],
    student_checkpoints: list[tuple[str, Path]],
) -> None:
    lines = [
        f"config: {Path(args.config).resolve()}",
        f"teacher_checkpoint: {teacher_ckpt.resolve()}",
        f"layers: {', '.join(at_layers)}",
        f"split: {args.split}",
        f"indices: {', '.join(str(i) for i in indices)}",
        f"upsample_to_input: {args.upsample_to_input}",
        "student_checkpoints:",
    ]
    for label, path in student_checkpoints:
        lines.append(f"  - {label}: {path.resolve()}")

    meta_path = out_dir / "run_info.txt"
    meta_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Metadati: {meta_path.name}")
    print("Fatto.")


def main() -> None:
    args = parse_args()
    cfg = load_yaml_config(args.config)

    device = resolve_device(args.device)
    print(f"Device: {device}")

    set_seed(int(cfg["experiment"]["seed"]))

    if args.compare_students:
        run_compare_students(args, cfg, device)
    else:
        run_single_student(args, cfg, device)


if __name__ == "__main__":
    main()

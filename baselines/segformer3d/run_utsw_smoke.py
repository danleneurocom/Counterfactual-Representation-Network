from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from baselines.segformer3d import SegFormer3D
from baselines.segformer3d.data import UTSWGliomaDataset


def _resolve_device(spec: str) -> torch.device:
    if spec != "auto":
        return torch.device(spec)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    dims = tuple(range(2, probs.ndim))
    intersection = (probs * target).sum(dim=dims)
    denominator = probs.sum(dim=dims) + target.sum(dim=dims)
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice.mean()


def _build_tiny_model(num_classes: int) -> SegFormer3D:
    return SegFormer3D(
        in_channels=4,
        sr_ratios=[4, 2, 1, 1],
        embed_dims=[4, 8, 16, 32],
        patch_kernel_size=[7, 3, 3, 3],
        patch_stride=[4, 2, 2, 2],
        patch_padding=[3, 1, 1, 1],
        mlp_ratios=[2, 2, 2, 2],
        num_heads=[1, 1, 2, 4],
        depths=[1, 1, 1, 1],
        decoder_head_embedding_dim=8,
        num_classes=num_classes,
        decoder_dropout=0.0,
    )


def _build_base_model(num_classes: int) -> SegFormer3D:
    return SegFormer3D(in_channels=4, num_classes=num_classes)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one UTSW case through the SegFormer3D baseline.")
    parser.add_argument(
        "--data-root",
        default="data/brats/PKG - UTSW-Glioma/UTSW-Glioma",
        help="Directory containing one UTSW case folder per subject.",
    )
    parser.add_argument("--case-id", help="Optional case id, e.g. BT0001.")
    parser.add_argument("--volume-size", type=int, default=32, help="Cubic network input size for this smoke run.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--model-size", choices=["tiny", "base"], default="tiny")
    parser.add_argument("--checkpoint", help="Optional SegFormer3D state_dict/checkpoint to load before inference.")
    parser.add_argument("--prefer-manual-seg", action="store_true", help="Use manual segmentation when shape-compatible.")
    parser.add_argument("--use-ants-modalities", action="store_true", help="Use *_ants.nii.gz modality files when available.")
    args = parser.parse_args()

    dataset = UTSWGliomaDataset(
        root=Path(args.data_root),
        volume_size=args.volume_size,
        case_ids=[args.case_id] if args.case_id else None,
        limit=1 if args.case_id is None else None,
        prefer_manual_seg=args.prefer_manual_seg,
        use_ants_modalities=args.use_ants_modalities,
    )
    sample = dataset[0]
    image = sample["image"].unsqueeze(0)
    mask = sample["mask"].unsqueeze(0)

    device = _resolve_device(args.device)
    model = _build_tiny_model(num_classes=3) if args.model_size == "tiny" else _build_base_model(num_classes=3)
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        state_dict = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
        model.load_state_dict(state_dict)
    model.to(device).eval()

    image = image.to(device)
    mask = mask.to(device)
    with torch.no_grad():
        logits = model(image)
        bce = F.binary_cross_entropy_with_logits(logits, mask)
        dice = _dice_loss(logits, mask)

    print(
        {
            "case_id": sample["case_id"],
            "source_shape": sample["source_shape"].tolist(),
            "network_image_shape": list(image.shape),
            "mask_shape": list(mask.shape),
            "logits_shape": list(logits.shape),
            "bce_loss": float(bce.detach().cpu()),
            "dice_loss": float(dice.detach().cpu()),
            "device": str(device),
            "model_size": args.model_size,
        }
    )


if __name__ == "__main__":
    main()

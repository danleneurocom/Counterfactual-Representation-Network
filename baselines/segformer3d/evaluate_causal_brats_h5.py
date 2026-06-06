from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from baselines.segformer3d.causal import build_causal_segformer3d, default_utsw_scm
from baselines.segformer3d.data.utsw import _crop_to_foreground, _normalize_mri
from baselines.segformer3d.evaluate_causal_utsw import _context_overlap, _segmentation_loss
from baselines.segformer3d.train_causal_utsw import build_context_bank, _prefix_metrics
from baselines.segformer3d.train_utsw import _average_metric_dicts, _mean, _resolve_device
from crn.metrics import brats_region_metrics, brats_volume_metrics_from_probs


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def _path_candidates(raw_path: str | Path, data_root: Path | None) -> list[Path]:
    raw = Path(raw_path)
    candidates = [raw]
    if data_root is None:
        return candidates

    candidates.extend(
        [
            data_root / raw,
            data_root / raw.name,
            data_root / "content" / "data" / raw.name,
            data_root / "data" / raw.name,
        ]
    )
    parts = raw.parts
    if "content" in parts:
        candidates.append(data_root / Path(*parts[parts.index("content") :]))
    if "data" in parts:
        candidates.append(data_root / Path(*parts[parts.index("data") :]))
    return candidates


def _resolve_path(raw_path: str | Path, data_root: Path | None) -> Path:
    for candidate in _path_candidates(raw_path, data_root):
        if candidate.exists():
            return candidate
    return _path_candidates(raw_path, data_root)[0]


def _read_h5(path: Path, image_key: str, mask_key: str) -> tuple[np.ndarray, np.ndarray]:
    try:
        import h5py
    except ImportError as exc:
        raise ImportError(
            "BraTS HDF5 evaluation requires h5py. Install it with: "
            "/Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12 -m pip install h5py"
        ) from exc

    with h5py.File(path, "r") as handle:
        if image_key not in handle or mask_key not in handle:
            raise KeyError(f"{path} must contain keys {image_key!r} and {mask_key!r}; found {list(handle.keys())}")
        return np.asarray(handle[image_key]), np.asarray(handle[mask_key])


def _image_slice_to_channels(array: np.ndarray) -> np.ndarray:
    array = array.astype(np.float32, copy=False)
    if array.ndim == 2:
        return array[None, ...]
    if array.ndim != 3:
        raise ValueError(f"Expected 2D or 3D image slice, got shape {array.shape}")
    if array.shape[0] == 4:
        return array
    if array.shape[-1] == 4:
        return np.moveaxis(array, -1, 0)
    raise ValueError(f"Expected a 4-modality BraTS image slice, got shape {array.shape}")


def _mask_slice_to_channels(array: np.ndarray) -> np.ndarray:
    array = array.astype(np.float32, copy=False)
    if array.ndim == 2:
        ncr_net = array == 1
        edema = array == 2
        enhancing = (array == 4) | (array == 3)
        return np.stack([ncr_net, edema, enhancing], axis=0).astype(np.float32)
    if array.ndim != 3:
        raise ValueError(f"Expected 2D or 3D mask slice, got shape {array.shape}")
    if array.shape[0] == 3:
        return (array > 0).astype(np.float32)
    if array.shape[-1] == 3:
        return np.moveaxis(array, -1, 0).astype(np.float32)
    raise ValueError(f"Expected a 3-channel BraTS mask slice, got shape {array.shape}")


def _resize_volume(image: np.ndarray, mask: np.ndarray, volume_size: int) -> tuple[Tensor, Tensor]:
    image_tensor = torch.from_numpy(np.ascontiguousarray(image)).float().unsqueeze(0)
    mask_tensor = torch.from_numpy(np.ascontiguousarray(mask)).float().unsqueeze(0)
    size = (int(volume_size), int(volume_size), int(volume_size))
    image_tensor = F.interpolate(image_tensor, size=size, mode="trilinear", align_corners=False)
    mask_tensor = F.interpolate(mask_tensor, size=size, mode="nearest")
    return image_tensor.squeeze(0), mask_tensor.squeeze(0).clamp(0.0, 1.0)


class BraTSH5VolumeDataset(Dataset):
    """Reconstruct BraTS2020 HDF5 slices into 3D volumes for SegFormer3D."""

    metadata_encoder = None

    def __init__(
        self,
        csv_path: str | Path,
        data_root: str | Path | None = None,
        volume_size: int = 128,
        volume_ids: list[int] | None = None,
        limit_volumes: int | None = None,
        path_col: str = "path",
        volume_col: str = "volume",
        slice_col: str = "slice",
        image_key: str = "image",
        mask_key: str = "mask",
        crop_margin: int = 8,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.data_root = Path(data_root).expanduser() if data_root else None
        self.volume_size = int(volume_size)
        self.path_col = path_col
        self.volume_col = volume_col
        self.slice_col = slice_col
        self.image_key = image_key
        self.mask_key = mask_key
        self.crop_margin = int(crop_margin)

        frame = pd.read_csv(self.csv_path)
        missing = {path_col, volume_col, slice_col}.difference(frame.columns)
        if missing:
            raise ValueError(f"{self.csv_path} is missing required columns: {sorted(missing)}")
        if volume_ids is not None:
            selected = {int(volume_id) for volume_id in volume_ids}
            frame = frame.loc[frame[volume_col].astype(int).isin(selected)]
        if frame.empty:
            raise ValueError(f"No BraTS rows selected from {self.csv_path}")

        groups = list(frame.sort_values([volume_col, slice_col]).groupby(volume_col, sort=True))
        if limit_volumes is not None:
            groups = groups[: int(limit_volumes)]
        self.volumes = [(int(volume_id), group.reset_index(drop=True)) for volume_id, group in groups]
        self._validate_paths()

    def _validate_paths(self) -> None:
        missing: list[str] = []
        for _, group in self.volumes:
            for raw_path in group[self.path_col].head(3).tolist():
                path = _resolve_path(raw_path, self.data_root)
                if not path.exists():
                    missing.append(str(path))
                    break
            if len(missing) >= 5:
                break
        if missing:
            raise FileNotFoundError(
                "BraTS H5 files are missing. Pass --data-root pointing to the folder containing "
                f"volume_*_slice_*.h5. Examples not found: {missing}"
            )

    def __len__(self) -> int:
        return len(self.volumes)

    def __getitem__(self, index: int) -> dict[str, Any]:
        volume_id, group = self.volumes[index]
        images: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        paths: list[str] = []
        for _, row in group.sort_values(self.slice_col).iterrows():
            path = _resolve_path(row[self.path_col], self.data_root)
            image_slice, mask_slice = _read_h5(path, self.image_key, self.mask_key)
            images.append(_image_slice_to_channels(image_slice))
            masks.append(_mask_slice_to_channels(mask_slice))
            paths.append(str(path))

        image = np.stack(images, axis=1)
        mask = np.stack(masks, axis=1)
        for modality in range(image.shape[0]):
            image[modality] = _normalize_mri(image[modality])
        original_shape = image.shape[1:]
        image, mask = _crop_to_foreground(image, mask, self.crop_margin)
        image_tensor, mask_tensor = _resize_volume(image, mask, self.volume_size)
        return {
            "case_id": f"volume_{volume_id}",
            "volume": volume_id,
            "image": image_tensor,
            "mask": mask_tensor,
            "source_shape": torch.tensor(original_shape, dtype=torch.long),
            "path": paths[0],
        }


def _make_loader(dataset: Dataset, batch_size: int, num_workers: int) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def _build_model(checkpoint: dict[str, Any], args: argparse.Namespace):
    config = dict(checkpoint.get("config", {}))
    proxy_dims = checkpoint.get("proxy_dims") or {}
    model = build_causal_segformer3d(
        model_size=str(args.model_size or config.get("model_size", "base")),
        latent_dim=int(args.latent_dim or config.get("latent_dim", 128)),
        num_classes=3,
        context_proxy_dim=int(proxy_dims.get("context_proxy_dim", 0)),
        disease_proxy_dim=int(proxy_dims.get("disease_proxy_dim", 0)),
        annotation_proxy_dim=int(proxy_dims.get("annotation_proxy_dim", 0)),
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    return model


def _volume_metrics(logits: Tensor, target: Tensor, threshold: float) -> list[dict[str, float]]:
    probs_cpu = torch.sigmoid(logits.detach().cpu())
    target_cpu = target.detach().cpu()
    metrics: list[dict[str, float]] = []
    for row in range(probs_cpu.shape[0]):
        probs_volume = probs_cpu[row].permute(1, 0, 2, 3).contiguous()
        target_volume = target_cpu[row].permute(1, 0, 2, 3).contiguous()
        metrics.append(
            brats_volume_metrics_from_probs(
                probs_volume,
                target_volume,
                threshold=threshold,
                channel_names=["ncr_net", "edema", "enhancing_tumor"],
                region_channels={"WT": [0, 1, 2], "TC": [0, 2], "ET": [2]},
            )
        )
    return metrics


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = dict(checkpoint.get("config", {}))
    volume_size = int(args.volume_size or config.get("volume_size", 128))
    eval_dataset = BraTSH5VolumeDataset(
        csv_path=args.brats_csv,
        data_root=args.data_root,
        volume_size=volume_size,
        limit_volumes=args.max_volumes,
        path_col=args.path_col,
        volume_col=args.volume_col,
        slice_col=args.slice_col,
        image_key=args.h5_image_key,
        mask_key=args.h5_mask_key,
        crop_margin=args.crop_margin,
    )
    bank_dataset = BraTSH5VolumeDataset(
        csv_path=args.context_csv or args.brats_csv,
        data_root=args.data_root,
        volume_size=volume_size,
        limit_volumes=args.max_context_volumes,
        path_col=args.path_col,
        volume_col=args.volume_col,
        slice_col=args.slice_col,
        image_key=args.h5_image_key,
        mask_key=args.h5_mask_key,
        crop_margin=args.crop_margin,
    )

    device = _resolve_device(args.device)
    model = _build_model(checkpoint, args).to(device)
    model.eval()
    eval_loader = _make_loader(eval_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    bank_loader = _make_loader(bank_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    context_bank = build_context_bank(
        model,
        bank_loader,
        device,
        max_contexts=args.context_bank_size,
        max_batches=args.max_context_bank_batches,
        sampling=args.context_bank_sampling,
        seed=args.seed,
    )
    bank_device = context_bank.to(device) if context_bank is not None else None

    factual_losses: list[float] = []
    adjusted_losses: list[float] = []
    factual_batch_metrics: list[dict[str, float]] = []
    adjusted_batch_metrics: list[dict[str, float]] = []
    factual_volume_metrics: list[dict[str, float]] = []
    adjusted_volume_metrics: list[dict[str, float]] = []
    context_shifts: list[float] = []
    nearest_context_distances: list[float] = []

    for batch_idx, batch in enumerate(tqdm(eval_loader, desc=f"brats-h5-eval:{args.split_name}", leave=False), start=1):
        image = batch["image"].to(device)
        target = batch["mask"].to(device)
        outputs = model(image, context_bank=bank_device, max_adjustment_contexts=args.adjustment_contexts)
        logits = outputs["logits"]
        if not isinstance(logits, Tensor):
            raise TypeError("CausalSegFormer3D output 'logits' must be a tensor.")
        factual_losses.append(float(_segmentation_loss(logits, target).detach().cpu()))
        factual_batch_metrics.append(brats_region_metrics(logits.detach().cpu(), target.detach().cpu(), threshold=args.threshold))
        factual_volume_metrics.extend(_volume_metrics(logits, target, args.threshold))

        z_c = outputs.get("z_c")
        if isinstance(z_c, Tensor):
            nearest_context_distances.extend(_context_overlap(z_c, bank_device).get("overlap/nearest_context_l2", []))

        adjusted = outputs.get("adjusted_logits")
        if isinstance(adjusted, Tensor):
            adjusted_losses.append(float(_segmentation_loss(adjusted, target).detach().cpu()))
            adjusted_batch_metrics.append(brats_region_metrics(adjusted.detach().cpu(), target.detach().cpu(), threshold=args.threshold))
            adjusted_volume_metrics.extend(_volume_metrics(adjusted, target, args.threshold))
            context_shifts.append(float((torch.sigmoid(logits) - torch.sigmoid(adjusted)).abs().mean().detach().cpu()))

        if args.max_batches is not None and batch_idx >= args.max_batches:
            break

    scm = default_utsw_scm()
    metrics: dict[str, Any] = {
        "method": "Causal SegFormer3D zero-shot on BraTS H5",
        "causal_question": scm.question.query,
        "causal_estimand": scm.question.estimand,
        "causal_warning": "Cross-dataset BraTS H5 evaluation has no BraTS metadata proxies; proxy losses are intentionally omitted.",
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "split": args.split_name,
        "brats_csv": str(args.brats_csv),
        "context_csv": str(args.context_csv or args.brats_csv),
        "threshold": float(args.threshold),
        "num_cases": float(len(factual_volume_metrics)),
        "context_bank_size": float(0 if context_bank is None else context_bank.shape[0]),
        "factual/loss": _mean(factual_losses),
    }
    metrics.update(_average_metric_dicts(factual_batch_metrics))
    metrics.update({f"volume/{key}": value for key, value in _average_metric_dicts(factual_volume_metrics).items()})
    if adjusted_losses:
        metrics["adjusted/loss"] = _mean(adjusted_losses)
        metrics.update(_prefix_metrics(_average_metric_dicts(adjusted_batch_metrics), "adjusted"))
        metrics.update({f"adjusted/volume/{key}": value for key, value in _average_metric_dicts(adjusted_volume_metrics).items()})
        metrics["intervention/context_adjustment_mean_abs_prob_shift"] = _mean(context_shifts)
        metrics["intervention/adjusted_minus_factual_mean_dice"] = (
            float(metrics.get("adjusted/brats/mean_dice", float("nan"))) - float(metrics.get("brats/mean_dice", float("nan")))
        )
    if nearest_context_distances:
        distances = torch.tensor(nearest_context_distances, dtype=torch.float32)
        metrics["overlap/nearest_context_l2_mean"] = float(distances.mean())
        metrics["overlap/nearest_context_l2_max"] = float(distances.max())
        metrics["overlap/nearest_context_l2_p90"] = float(torch.quantile(distances, 0.9))
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the best Causal SegFormer3D checkpoint on BraTS2020 HDF5 volumes.")
    parser.add_argument("--checkpoint", default="runs/segformer3d_utsw_causal/best.pt")
    parser.add_argument("--brats-csv", default="data/brats/brats_val.csv")
    parser.add_argument("--context-csv", default="data/brats/brats_train.csv")
    parser.add_argument("--data-root", help="Folder containing volume_*_slice_*.h5 files.")
    parser.add_argument("--output-json")
    parser.add_argument("--split-name", default="brats_val")
    parser.add_argument("--model-size", choices=["tiny", "base"])
    parser.add_argument("--volume-size", type=int)
    parser.add_argument("--latent-dim", type=int)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--context-bank-size", type=int, default=64)
    parser.add_argument("--adjustment-contexts", type=int, default=4)
    parser.add_argument("--context-bank-sampling", choices=["uniform", "random", "farthest"], default="uniform")
    parser.add_argument("--max-context-bank-batches", type=int)
    parser.add_argument("--max-context-volumes", type=int)
    parser.add_argument("--max-volumes", type=int)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--crop-margin", type=int, default=8)
    parser.add_argument("--path-col", default="path")
    parser.add_argument("--volume-col", default="volume")
    parser.add_argument("--slice-col", default="slice")
    parser.add_argument("--h5-image-key", default="image")
    parser.add_argument("--h5-mask-key", default="mask")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = evaluate(args)
    output_json = (
        Path(args.output_json)
        if args.output_json
        else Path(args.checkpoint).with_name(f"{args.split_name}_causal_metrics.json")
    )
    _save_json(metrics, output_json)
    print(metrics)
    print({"metrics_json": str(output_json)})


if __name__ == "__main__":
    main()

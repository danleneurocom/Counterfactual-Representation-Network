from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from baselines.mednext.causal import CausalMedNeXt, build_causal_mednext
from baselines.mednext.common import _parse_channel_weights
from baselines.mednext.dataset_cache import maybe_disk_cache_dataset, warm_disk_cache_dataset
from baselines.mednext.common import checkpoint_monitor
from baselines.mednext.train_causal_utsw import (
    LesionInterventionBank,
    _add_causal_args,
    _build_teacher_model,
    _hard_case_sampler,
    _load_causal_init_checkpoint,
    _run_eval_epoch,
    _run_train_epoch,
    _set_et_volume_veto_scale_for_epoch,
    build_category_confounder_dictionary,
    build_sdd_cite_bank,
    prefill_lesion_intervention_bank,
)
from baselines.segformer3d.causal import default_utsw_scm
from baselines.segformer3d.evaluate_causal_brats_h5 import _mask_slice_to_channels, _resolve_path
from baselines.segformer3d.train_brats_h5 import _make_dataset, _make_loader as _make_plain_loader, _volume_ids
from baselines.segformer3d.train_causal_utsw import _build_optimizer, _set_backbone_trainable
from baselines.segformer3d.train_utsw import _resolve_device, _save_json


BRATS_PSEUDO_PROXY_DIMS = {
    "context_proxy_dim": 12,
    "disease_proxy_dim": 6,
    "annotation_proxy_dim": 4,
}


def _proxy_dims_from_args(args: argparse.Namespace) -> dict[str, int]:
    if not bool(getattr(args, "use_pseudo_proxies", False)):
        return {"context_proxy_dim": 0, "disease_proxy_dim": 0, "annotation_proxy_dim": 0}
    return dict(BRATS_PSEUDO_PROXY_DIMS)


def _cache_signature(csv_path: str | Path, args: argparse.Namespace, split: str) -> dict[str, object]:
    return {
        "dataset": "brats_h5",
        "split": split,
        "csv_path": str(csv_path),
        "data_root": str(args.data_root),
        "volume_size": args.volume_size,
        "crop_margin": args.crop_margin,
        "path_col": args.path_col,
        "volume_col": args.volume_col,
        "slice_col": args.slice_col,
        "h5_image_key": args.h5_image_key,
        "h5_mask_key": args.h5_mask_key,
    }


def _build_model(args: argparse.Namespace) -> CausalMedNeXt:
    proxy_dims = _proxy_dims_from_args(args)
    return build_causal_mednext(
        model_id=args.model_id,
        kernel_size=args.kernel_size,
        latent_dim=args.latent_dim,
        num_classes=3,
        base_channels=args.base_channels,
        modulation_scale=args.modulation_scale,
        causal_residual_scale=args.causal_residual_scale,
        contrastive_dim=args.contrastive_dim,
        spatial_refiner_scale=args.spatial_refiner_scale,
        region_fusion_scale=args.region_fusion_scale,
        prototype_dim=args.prototype_dim,
        prototype_fusion_scale=args.prototype_fusion_scale,
        prototype_temperature=args.prototype_temperature,
        category_confounder_scale=args.category_confounder_scale,
        category_confounder_temperature=args.category_confounder_temperature,
        modality_prior_scale=args.modality_prior_scale,
        logit_calibration_scale=args.logit_calibration_scale,
        cascade_refiner_scale=args.cascade_refiner_scale,
        frontdoor_mediator_scale=args.frontdoor_mediator_scale,
        frontdoor_residual_scale=args.frontdoor_residual_scale,
        use_causal_mediator_router=args.use_causal_mediator_router,
        use_nested_causal_intervention=args.use_nested_causal_intervention,
        nested_causal_gate_scale=args.nested_causal_gate_scale,
        region_causal_bottleneck_scale=args.region_causal_bottleneck_scale,
        region_causal_background_leak=args.region_causal_background_leak,
        region_causal_base=args.region_causal_base,
        region_causal_mask_source=args.region_causal_mask_source,
        region_volume_scale=args.region_volume_scale,
        et_volume_veto_scale=args.et_volume_veto_scale,
        et_volume_veto_multiplier=args.et_volume_veto_multiplier,
        et_volume_veto_min_fraction=args.et_volume_veto_min_fraction,
        et_volume_veto_max_bias=args.et_volume_veto_max_bias,
        **proxy_dims,
    )


def _load_baseline_backbone(model: CausalMedNeXt, checkpoint_path: str | Path) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint)
    model.load_baseline_state_dict(state_dict, strict_backbone=True)
    return checkpoint


def _unwrap_brats_h5_dataset(dataset: torch.utils.data.Dataset) -> torch.utils.data.Dataset:
    source = dataset
    while not hasattr(source, "volumes") and hasattr(source, "dataset"):
        source = source.dataset
    return source


def _cache_payload_hash(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()[:12]


def _brats_h5_fraction_cache_dir(
    dataset: torch.utils.data.Dataset,
    args: argparse.Namespace,
) -> Path | None:
    cache_root = getattr(args, "disk_cache_dir", None)
    if cache_root is None or str(cache_root).strip() == "":
        return None
    source = _unwrap_brats_h5_dataset(dataset)
    signature = {
        "dataset": "brats_h5_raw_mask_region_fractions_v1",
        "csv_path": str(getattr(source, "csv_path", "")),
        "data_root": str(getattr(source, "data_root", "")),
        "path_col": str(getattr(source, "path_col", "path")),
        "slice_col": str(getattr(source, "slice_col", "slice")),
        "mask_key": str(getattr(source, "mask_key", "mask")),
    }
    return Path(cache_root).expanduser() / "brats_h5_raw_mask_region_fractions" / _cache_payload_hash(signature)


def _raw_mask_fraction_cache_path(
    dataset: torch.utils.data.Dataset,
    index: int,
    cache_dir: Path | None,
) -> Path | None:
    if cache_dir is None:
        return None
    volume_id, _ = dataset.volumes[int(index)]
    return _raw_mask_fraction_cache_path_for_volume(int(volume_id), cache_dir)


def _raw_mask_fraction_cache_path_for_volume(volume_id: int, cache_dir: Path) -> Path:
    digest = hashlib.sha1(str(volume_id).encode("utf-8")).hexdigest()[:12]
    return cache_dir / f"volume_{volume_id}_{digest}.pt"


def _metadata_mask_region_fractions_from_rows(rows: list[dict[str, object]]) -> tuple[float, float, float] | None:
    count_columns = ("label0_pxl_cnt", "label1_pxl_cnt", "label2_pxl_cnt")
    if not rows or any(column not in rows[0] for column in (*count_columns, "background_ratio")):
        return None
    counts: list[tuple[float, float, float]] = []
    slice_voxel_candidates: list[int] = []
    for row in rows:
        try:
            label_counts = tuple(float(row[column]) for column in count_columns)
            background_ratio = float(row["background_ratio"])
        except (TypeError, ValueError):
            return None
        if any(not np.isfinite(value) or value < 0.0 for value in label_counts):
            return None
        if not np.isfinite(background_ratio) or background_ratio < 0.0 or background_ratio > 1.0:
            return None
        counts.append(label_counts)
        foreground_pixels = float(sum(label_counts))
        if foreground_pixels > 0.0 and background_ratio < 1.0:
            slice_voxels = int(round(foreground_pixels / max(1.0 - background_ratio, 1e-12)))
            if slice_voxels > 0:
                slice_voxel_candidates.append(slice_voxels)

    region_counts = np.asarray(
        (
            sum(label0 + label1 + label2 for label0, label1, label2 in counts),
            sum(label0 + label2 for label0, _, label2 in counts),
            sum(label2 for _, _, label2 in counts),
        ),
        dtype=np.float64,
    )
    if float(region_counts.sum()) <= 0.0:
        return (0.0, 0.0, 0.0)
    if not slice_voxel_candidates:
        return None
    slice_voxels = int(round(float(np.median(np.asarray(slice_voxel_candidates, dtype=np.float64)))))
    total_voxels = slice_voxels * len(rows)
    if total_voxels <= 0:
        return None
    return tuple(float(value / total_voxels) for value in region_counts)


def _load_raw_mask_fraction_cache(path: Path, volume_id: int) -> tuple[float, float, float] | None:
    if not path.exists():
        return None
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    except Exception:
        return None
    if not isinstance(payload, dict) or int(payload.get("volume_id", -1)) != int(volume_id):
        return None
    fractions = payload.get("fractions")
    if not isinstance(fractions, (list, tuple)) or len(fractions) != 3:
        return None
    return tuple(float(value) for value in fractions)


def _write_raw_mask_fraction_cache(
    path: Path,
    volume_id: int,
    fractions: tuple[float, float, float],
    num_slices: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "volume_id": int(volume_id),
        "fractions": [float(value) for value in fractions],
        "num_slices": int(num_slices),
    }
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.stem}.", suffix=".tmp", delete=False) as handle:
        tmp_path = Path(handle.name)
    try:
        torch.save(payload, tmp_path)
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _raw_mask_slice_region_counts(array: np.ndarray) -> tuple[np.ndarray, int]:
    if array.ndim == 3 and array.shape[-1] == 3:
        channel_0 = array[..., 0] > 0
        channel_2 = array[..., 2] > 0
        wt = np.any(array > 0, axis=-1)
        tc = np.logical_or(channel_0, channel_2)
        et = channel_2
        return np.asarray((float(wt.sum()), float(tc.sum()), float(et.sum())), dtype=np.float64), int(wt.size)
    if array.ndim == 3 and array.shape[0] == 3:
        channel_0 = array[0] > 0
        channel_2 = array[2] > 0
        wt = np.any(array > 0, axis=0)
        tc = np.logical_or(channel_0, channel_2)
        et = channel_2
        return np.asarray((float(wt.sum()), float(tc.sum()), float(et.sum())), dtype=np.float64), int(wt.size)
    if array.ndim == 2:
        ncr_net = array == 1
        enhancing = (array == 4) | (array == 3)
        wt = array > 0
        tc = np.logical_or(ncr_net, enhancing)
        et = enhancing
        return np.asarray((float(wt.sum()), float(tc.sum()), float(et.sum())), dtype=np.float64), int(wt.size)
    mask = _mask_slice_to_channels(array).astype(bool, copy=False)
    wt = mask.any(axis=0)
    tc = np.logical_or(mask[0], mask[2])
    et = mask[2]
    return np.asarray((float(wt.sum()), float(tc.sum()), float(et.sum())), dtype=np.float64), int(wt.size)


def _raw_mask_region_fractions_from_rows(
    volume_id: int,
    rows: list[dict[str, object]],
    data_root: str,
    path_col: str,
    mask_key: str,
    cache_path: str | None,
) -> tuple[float, float, float]:
    import h5py

    cache_file = Path(cache_path) if cache_path else None
    if cache_file is not None:
        cached = _load_raw_mask_fraction_cache(cache_file, int(volume_id))
        if cached is not None:
            return cached
    metadata_fractions = _metadata_mask_region_fractions_from_rows(rows)
    if metadata_fractions is not None:
        if cache_file is not None:
            _write_raw_mask_fraction_cache(cache_file, int(volume_id), metadata_fractions, num_slices=len(rows))
        return metadata_fractions
    data_root_path = Path(data_root).expanduser() if data_root else None
    region_counts = np.zeros(3, dtype=np.float64)
    total_voxels = 0
    for row in rows:
        path = _resolve_path(row[path_col], data_root_path)
        with h5py.File(path, "r") as handle:
            if mask_key not in handle:
                raise KeyError(f"{path} must contain mask key {mask_key!r}; found {list(handle.keys())}")
            slice_counts, slice_voxels = _raw_mask_slice_region_counts(np.asarray(handle[mask_key]))
        region_counts += slice_counts
        total_voxels += slice_voxels
    fractions = (0.0, 0.0, 0.0) if total_voxels <= 0 else tuple(float(value / total_voxels) for value in region_counts)
    if cache_file is not None:
        _write_raw_mask_fraction_cache(cache_file, int(volume_id), fractions, num_slices=len(rows))
    return fractions


def _raw_mask_region_fractions_task(payload: tuple[int, list[dict[str, object]], str, str, str, str | None]) -> tuple[float, float, float]:
    return _raw_mask_region_fractions_from_rows(*payload)


def _raw_mask_region_fractions(
    dataset: torch.utils.data.Dataset,
    index: int,
    cache_dir: Path | None = None,
) -> tuple[float, float, float]:
    volume_id, group = dataset.volumes[int(index)]
    cache_path = _raw_mask_fraction_cache_path(dataset, index, cache_dir)
    if cache_path is not None:
        cached = _load_raw_mask_fraction_cache(cache_path, int(volume_id))
        if cached is not None:
            return cached
    ordered = group.sort_values(dataset.slice_col)
    return _raw_mask_region_fractions_from_rows(
        int(volume_id),
        ordered.to_dict("records"),
        str(dataset.data_root or ""),
        str(dataset.path_col),
        str(dataset.mask_key),
        str(cache_path) if cache_path is not None else None,
    )


def _brats_h5_hard_case_fraction_list(
    dataset: torch.utils.data.Dataset,
    args: argparse.Namespace,
    cache_dir: Path | None,
) -> list[tuple[float, float, float]]:
    fractions: list[tuple[float, float, float] | None] = [None] * len(dataset)
    tasks: list[tuple[int, tuple[int, list[dict[str, object]], str, str, str, str | None]]] = []
    for index in range(len(dataset)):
        volume_id, group = dataset.volumes[int(index)]
        cache_path = _raw_mask_fraction_cache_path_for_volume(int(volume_id), cache_dir) if cache_dir is not None else None
        if cache_path is not None:
            cached = _load_raw_mask_fraction_cache(cache_path, int(volume_id))
            if cached is not None:
                fractions[index] = cached
                continue
        tasks.append(
            (
                index,
                (
                    int(volume_id),
                    group.sort_values(dataset.slice_col).to_dict("records"),
                    str(dataset.data_root or ""),
                    str(dataset.path_col),
                    str(dataset.mask_key),
                    str(cache_path) if cache_path is not None else None,
                ),
            )
        )
    if not tasks:
        return [tuple(value) for value in fractions if value is not None]

    missing = len(tasks)
    worker_count = min(4, os.cpu_count() or 1, missing) if cache_dir is not None and missing >= 4 else 1
    desc = f"brats-hard-case-fractions:{missing}miss/{len(dataset)}"
    if worker_count > 1:
        payloads = [payload for _, payload in tasks]
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            iterator = tqdm(executor.map(_raw_mask_region_fractions_task, payloads), total=missing, desc=desc, leave=False)
            for (index, _), value in zip(tasks, iterator):
                fractions[index] = value
    else:
        for index, payload in tqdm(tasks, total=missing, desc=desc, leave=False):
            fractions[index] = _raw_mask_region_fractions_task(payload)
    if any(value is None for value in fractions):
        raise RuntimeError("Failed to compute all BraTS hard-case region fractions.")
    return [tuple(value) for value in fractions if value is not None]


def _brats_h5_hard_case_sampler_weights(dataset: torch.utils.data.Dataset, args: argparse.Namespace) -> Tensor | None:
    emphasis = float(getattr(args, "hard_case_sampler_emphasis", 0.0) or 0.0)
    if emphasis <= 0.0:
        return None
    source = _unwrap_brats_h5_dataset(dataset)
    if not hasattr(source, "volumes"):
        return None
    references = np.asarray(
        _parse_channel_weights(
            getattr(args, "hard_case_sampler_reference_fractions", "0.015,0.006,0.003"),
            3,
            "--hard-case-sampler-reference-fractions",
        ),
        dtype=np.float64,
    )
    region_weights = np.asarray(
        _parse_channel_weights(
            getattr(args, "hard_case_sampler_region_weights", "0.5,1.0,1.5"),
            3,
            "--hard-case-sampler-region-weights",
        ),
        dtype=np.float64,
    )
    references = np.maximum(references, 1e-6)
    region_weights = np.maximum(region_weights, 0.0)
    denominator = max(float(region_weights.sum()), 1e-6)
    max_weight = max(1.0, float(getattr(args, "hard_case_sampler_max_weight", 3.0) or 3.0))
    fraction_cache_dir = _brats_h5_fraction_cache_dir(source, args)
    weights: list[float] = []
    for item in _brats_h5_hard_case_fraction_list(source, args, fraction_cache_dir):
        fractions = np.asarray(item, dtype=np.float64)
        rarity = np.clip((references - fractions) / references, 0.0, 1.0)
        rarity_score = float((rarity * region_weights).sum() / denominator)
        non_empty = 1.0 if float(fractions[0]) > 0.0 else 0.0
        weights.append(min(max_weight, max(1.0, 1.0 + emphasis * rarity_score * non_empty)))
    if not weights:
        return None
    return torch.as_tensor(weights, dtype=torch.double).clamp_min(1e-6)


def _brats_h5_hard_case_sampler(dataset: torch.utils.data.Dataset, args: argparse.Namespace) -> WeightedRandomSampler | None:
    weights = _brats_h5_hard_case_sampler_weights(dataset, args)
    if weights is None:
        return None
    multiplier = max(float(getattr(args, "hard_case_sampler_epoch_multiplier", 1.0) or 1.0), 1e-6)
    num_samples = max(1, int(round(len(weights) * multiplier)))
    generator = torch.Generator()
    generator.manual_seed(int(getattr(args, "seed", 7)))
    return WeightedRandomSampler(weights, num_samples=num_samples, replacement=True, generator=generator)


def _make_causal_loader(dataset: torch.utils.data.Dataset, args: argparse.Namespace, shuffle: bool) -> DataLoader:
    sampler = _brats_h5_hard_case_sampler(dataset, args) if shuffle else None
    if sampler is None and shuffle:
        sampler = _hard_case_sampler(dataset, args)
    if sampler is None:
        return _make_plain_loader(dataset, args, shuffle=shuffle)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train causal MedNeXt on BraTS2020 HDF5 volumes.")
    parser.add_argument("--config-json", help="Load saved trainer arguments before applying explicit CLI overrides.")
    parser.add_argument("--baseline-checkpoint", default="runs/mednext_brats_h5_s_k3/best.pt")
    parser.add_argument("--train-csv", default="data/brats/brats_train.csv")
    parser.add_argument("--val-csv", default="data/brats/brats_val.csv")
    parser.add_argument("--data-root", default="data/brats/archive/BraTS2020_training_data/content/data")
    parser.add_argument("--teacher-checkpoint")
    parser.add_argument("--output-dir", default="runs/mednext_brats_h5_causal_s_k3")
    parser.add_argument("--model-id", choices=["S", "B", "M", "L"], default="S")
    parser.add_argument("--kernel-size", type=int, choices=[3, 5], default=3)
    parser.add_argument("--base-channels", type=int)
    parser.add_argument("--volume-size", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--backbone-lr", type=float)
    parser.add_argument("--causal-lr", type=float)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--limit-train-volumes", type=int)
    parser.add_argument("--limit-val-volumes", type=int)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    parser.add_argument("--max-context-bank-batches", type=int)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--crop-margin", type=int, default=8)
    parser.add_argument("--path-col", default="path")
    parser.add_argument("--volume-col", default="volume")
    parser.add_argument("--slice-col", default="slice")
    parser.add_argument("--h5-image-key", default="image")
    parser.add_argument("--h5-mask-key", default="mask")
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--disk-cache-dir", help="Optional MedNeXt-only disk cache for preprocessed dataset items.")
    parser.add_argument(
        "--use-pseudo-proxies",
        action="store_true",
        help="Use dataset-agnostic BraTS image, disease-region, and scan-geometry proxies when metadata is unavailable.",
    )
    parser.add_argument("--warm-disk-cache-only", action="store_true", help="Warm preprocessed disk cache entries and exit before model setup.")
    parser.add_argument("--warm-disk-cache-split", choices=["train", "val", "both"], default="train")
    parser.add_argument("--warm-disk-cache-start-index", type=int, default=0)
    parser.add_argument("--warm-disk-cache-max-items", type=int)
    parser.add_argument("--warm-disk-cache-include-existing", action="store_true")
    _add_causal_args(parser)
    parser.set_defaults(lambda_context_proxy=0.0, lambda_disease_proxy=0.0, lambda_annotation_proxy=0.0)
    config_args, _ = parser.parse_known_args()
    if config_args.config_json:
        with Path(config_args.config_json).expanduser().open("r", encoding="utf-8") as handle:
            saved_config = json.load(handle)
        destinations = {action.dest for action in parser._actions}
        parser.set_defaults(
            **{
                key: value
                for key, value in saved_config.items()
                if key in destinations and key != "config_json"
            }
        )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_ids = _volume_ids(args.train_csv, args.volume_col, args.limit_train_volumes)
    val_ids = _volume_ids(args.val_csv, args.volume_col, args.limit_val_volumes)
    splits = {"train": train_ids, "val": val_ids}
    _save_json(splits, output_dir / "splits.json")
    _save_json(vars(args), output_dir / "config.json")
    _save_json(asdict(default_utsw_scm()), output_dir / "scm.json")

    train_dataset = maybe_disk_cache_dataset(
        _make_dataset(args.train_csv, train_ids, args),
        args.disk_cache_dir,
        namespace="brats_h5_train",
        signature=_cache_signature(args.train_csv, args, "train"),
    )
    val_dataset = maybe_disk_cache_dataset(
        _make_dataset(args.val_csv, val_ids, args),
        args.disk_cache_dir,
        namespace="brats_h5_val",
        signature=_cache_signature(args.val_csv, args, "val"),
    )
    if args.warm_disk_cache_only:
        warmup: dict[str, object] = {}
        if args.warm_disk_cache_split in ("train", "both"):
            warmup["train"] = warm_disk_cache_dataset(
                train_dataset,
                start_index=args.warm_disk_cache_start_index,
                max_items=args.warm_disk_cache_max_items,
                missing_only=not args.warm_disk_cache_include_existing,
            )
        if args.warm_disk_cache_split in ("val", "both"):
            warmup["val"] = warm_disk_cache_dataset(
                val_dataset,
                start_index=args.warm_disk_cache_start_index,
                max_items=args.warm_disk_cache_max_items,
                missing_only=not args.warm_disk_cache_include_existing,
            )
        _save_json(warmup, output_dir / "disk_cache_warmup.json")
        print({"disk_cache_warmup": warmup, "output_dir": str(output_dir)})
        return
    train_loader: DataLoader = _make_causal_loader(train_dataset, args, shuffle=True)
    val_loader: DataLoader = _make_causal_loader(val_dataset, args, shuffle=False)
    bank_loader: DataLoader = _make_causal_loader(train_dataset, args, shuffle=False)

    device = _resolve_device(args.device)
    model = _build_model(args)
    _load_baseline_backbone(model, args.baseline_checkpoint)
    init_report = _load_causal_init_checkpoint(model, args.init_checkpoint)
    if init_report:
        _save_json(init_report, output_dir / "init_checkpoint.json")
    model.to(device)
    optimizer = _build_optimizer(model, args)
    proxy_dims = _proxy_dims_from_args(args)
    teacher_checkpoint = args.teacher_checkpoint
    if teacher_checkpoint is None and (
        args.lambda_teacher_distill > 0.0
        or args.lambda_adjusted_teacher_distill > 0.0
        or args.lambda_context_swap_teacher_distill > 0.0
        or args.lambda_region_causal_teacher_distill > 0.0
    ):
        teacher_checkpoint = args.baseline_checkpoint
    teacher_model = _build_teacher_model(teacher_checkpoint, device)

    best_monitor = float("-inf")
    best_monitor_key = "adjusted/brats/mean_dice"
    contrastive_bank: dict[str, Tensor] | None = None
    lesion_bank = (
        LesionInterventionBank(
            max_patches=args.lesion_bank_size,
            min_voxels=args.lesion_min_voxels,
            edge_softening=args.lesion_edge_softening,
            min_brain_coverage=args.lesion_min_brain_coverage,
            placement_attempts=args.lesion_placement_attempts,
            match_recipient_moments=args.lesion_match_recipient_moments,
        )
        if any(
            float(value) > 0.0
            for value in (
                args.lambda_lesion_paste_seg,
                args.lambda_lesion_erase_seg,
                args.lambda_lesion_intervention_effect,
            )
        )
        else None
    )
    lesion_prefill_report = prefill_lesion_intervention_bank(
        lesion_bank,
        bank_loader,
        max_batches=args.lesion_prefill_batches,
    )
    for epoch in range(1, args.epochs + 1):
        effective_et_veto_scale = _set_et_volume_veto_scale_for_epoch(model, args, epoch)
        _set_backbone_trainable(model, epoch > args.freeze_backbone_epochs)
        category_report = build_category_confounder_dictionary(
            model,
            bank_loader,
            device,
            max_batches=args.max_category_confounder_batches
            if args.max_category_confounder_batches is not None
            else args.max_context_bank_batches,
            threshold=args.threshold,
        )
        if contrastive_bank is None or (args.context_bank_refresh_epochs > 0 and (epoch - 1) % args.context_bank_refresh_epochs == 0):
            contrastive_bank = build_sdd_cite_bank(
                model,
                bank_loader,
                device,
                max_contexts=args.context_bank_size,
                max_batches=args.max_context_bank_batches,
                sampling=args.context_bank_sampling,
                seed=args.seed + epoch,
            )
        context_bank = None if contrastive_bank is None else contrastive_bank["z_c"]
        train_metrics = _run_train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            args,
            context_bank,
            contrastive_bank,
            proxy_layout=None,
            lesion_bank=lesion_bank,
            teacher_model=teacher_model,
        )
        val_metrics = _run_eval_epoch(
            model,
            val_loader,
            device,
            args,
            context_bank,
            contrastive_bank,
            proxy_layout=None,
        )
        train_metrics["schedule/et_volume_veto_scale"] = float(effective_et_veto_scale)
        val_metrics["schedule/et_volume_veto_scale"] = float(effective_et_veto_scale)
        monitor, monitor_key = checkpoint_monitor(val_metrics, args, prefer_adjusted=True)
        _save_json(
            {
                "epoch": epoch,
                "category_confounders": category_report,
                "lesion_bank": lesion_prefill_report,
                "train": train_metrics,
                "val": val_metrics,
            },
            output_dir / f"epoch_{epoch:03d}.json",
        )
        print(
            {
                "epoch": epoch,
                "train_loss": train_metrics.get("loss/total"),
                "val_loss": val_metrics.get("loss/total"),
                "val_factual_mean_dice": val_metrics.get("brats/mean_dice"),
                "val_adjusted_mean_dice": val_metrics.get("adjusted/brats/mean_dice"),
                "monitor": monitor,
                "monitor_key": monitor_key,
                "et_volume_veto_scale": effective_et_veto_scale,
            }
        )

        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": vars(args),
            "splits": splits,
            "proxy_dims": proxy_dims,
            "proxy_layout": None,
            "category_confounders": category_report,
            "val_metrics": val_metrics,
        }
        torch.save(checkpoint, output_dir / "last.pt")
        if monitor > best_monitor:
            best_monitor = monitor
            best_monitor_key = monitor_key
            torch.save(checkpoint, output_dir / "best.pt")

    print({"best_val_monitor": best_monitor, "best_val_monitor_key": best_monitor_key, "output_dir": str(output_dir)})


if __name__ == "__main__":
    main()

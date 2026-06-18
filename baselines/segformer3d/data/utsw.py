from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset

from baselines.segformer3d.causal.scm import default_utsw_scm


UTSW_MODALITIES = ("flair", "t1", "t1ce", "t2")
UTSW_SUBREGION_LABELS = {
    "ncr_net": 1,
    "edema": 2,
    "enhancing_tumor": 4,
}
UTSW_METADATA_FILENAME = "UTSW_Glioma_Metadata-2-1.tsv"
UTSW_CONTEXT_NUMERIC_COLUMNS = ("Age at Imaging", "Scanner Strength")
UTSW_DISEASE_NUMERIC_COLUMNS = ("Tumor Grade",)


def _load_nifti(path: Path) -> np.ndarray:
    if path.name.endswith(".gz"):
        with path.open("rb") as handle:
            if handle.read(2) != b"\x1f\x8b":
                raise nib.filebasedimages.ImageFileError(f"{path} is not a gzip file")
    return np.asarray(nib.load(str(path)).dataobj)


def _load_depth_first_nifti(path: Path) -> np.ndarray:
    return _to_depth_first(_load_nifti(path))


def _to_depth_first(volume: np.ndarray) -> np.ndarray:
    """Convert NIfTI array `(X, Y, Z)` to network volume `(D, H, W)`."""
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D NIfTI volume, got shape {volume.shape}")
    return np.moveaxis(volume, -1, 0)


def _normalize_mri(volume: np.ndarray) -> np.ndarray:
    volume = volume.astype(np.float32, copy=False)
    foreground = volume[np.abs(volume) > 0]
    if foreground.size == 0:
        return np.zeros_like(volume, dtype=np.float32)
    mean = float(foreground.mean())
    std = float(foreground.std())
    if std < 1e-6:
        return np.zeros_like(volume, dtype=np.float32)
    volume = (volume - mean) / std
    return np.clip(volume, -5.0, 5.0).astype(np.float32, copy=False)


def _subregion_mask(segmentation: np.ndarray) -> np.ndarray:
    segmentation = segmentation.astype(np.int16, copy=False)
    ncr_net = segmentation == UTSW_SUBREGION_LABELS["ncr_net"]
    edema = segmentation == UTSW_SUBREGION_LABELS["edema"]
    enhancing = (segmentation == UTSW_SUBREGION_LABELS["enhancing_tumor"]) | (segmentation == 3)
    return np.stack([ncr_net, edema, enhancing], axis=0).astype(np.float32)


def _crop_to_foreground(image: np.ndarray, mask: np.ndarray, margin: int) -> tuple[np.ndarray, np.ndarray]:
    foreground = np.any(np.abs(image) > 0, axis=0)
    if not np.any(foreground):
        return image, mask

    coords = np.argwhere(foreground)
    starts = np.maximum(coords.min(axis=0) - int(margin), 0)
    stops = np.minimum(coords.max(axis=0) + int(margin) + 1, np.asarray(foreground.shape))
    slices = tuple(slice(int(start), int(stop)) for start, stop in zip(starts, stops, strict=True))
    return image[(slice(None), *slices)], mask[(slice(None), *slices)]


def _resize_volume(image: np.ndarray, mask: np.ndarray, volume_size: int) -> tuple[Tensor, Tensor]:
    image_tensor = torch.from_numpy(np.ascontiguousarray(image)).float().unsqueeze(0)
    mask_tensor = torch.from_numpy(np.ascontiguousarray(mask)).float().unsqueeze(0)
    size = (int(volume_size), int(volume_size), int(volume_size))
    image_tensor = F.interpolate(image_tensor, size=size, mode="trilinear", align_corners=False)
    mask_tensor = F.interpolate(mask_tensor, size=size, mode="nearest")
    return image_tensor.squeeze(0), mask_tensor.squeeze(0)


def _unique_existing_paths(paths: Iterable[Path]) -> list[Path]:
    seen: set[Path] = set()
    existing: list[Path] = []
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        if path.exists():
            existing.append(path)
    return existing


def _modality_candidates(case_dir: Path, modality: str, ants: bool) -> list[Path]:
    ants_name = "fl" if modality == "flair" else modality
    ants_path = case_dir / f"brain_{ants_name}_ants.nii.gz"
    native_paths = [
        case_dir / f"brain_{modality}.nii.gz",
        case_dir / f"brain_{modality.upper()}.nii.gz",
    ]
    paths = [ants_path, *native_paths] if ants else [*native_paths, ants_path]
    return _unique_existing_paths(paths)


def _segmentation_candidates(case_dir: Path, prefer_manual: bool) -> list[Path]:
    manual_paths = [
        case_dir / "rtumorseg_manual_correction.nii.gz",
        case_dir / "tumorseg_manual_correction.nii.gz",
    ]
    fets_path = case_dir / "tumorseg_FeTS.nii.gz"
    paths = [*manual_paths, fets_path] if prefer_manual else [fets_path, *manual_paths]
    return _unique_existing_paths(paths)


def _find_modality(case_dir: Path, modality: str, ants: bool) -> Path:
    candidates = _modality_candidates(case_dir, modality, ants)
    for candidate in candidates:
        return candidate
    raise FileNotFoundError(f"Missing {modality} modality in {case_dir}")


def _find_segmentation(case_dir: Path, prefer_manual: bool) -> Path:
    candidates = _segmentation_candidates(case_dir, prefer_manual)
    for candidate in candidates:
        return candidate
    raise FileNotFoundError(f"Missing segmentation in {case_dir}")


def _format_load_errors(errors: list[str]) -> str:
    return "; ".join(errors) if errors else "no candidate files were found"


def _load_modalities(case_dir: Path, ants: bool) -> tuple[list[np.ndarray], list[Path]]:
    candidate_lists = {modality: _modality_candidates(case_dir, modality, ants) for modality in UTSW_MODALITIES}

    preferred: list[tuple[str, Path, np.ndarray]] = []
    preferred_errors: list[str] = []
    preloaded_by_path: dict[Path, np.ndarray] = {}
    failed_by_path: dict[Path, str] = {}
    for modality in UTSW_MODALITIES:
        candidates = candidate_lists[modality]
        if not candidates:
            preferred_errors.append(f"{modality}: no candidate files found")
            break
        path = candidates[0]
        try:
            volume = _load_depth_first_nifti(path)
            preloaded_by_path[path] = volume
            preferred.append((modality, path, volume))
        except Exception as exc:
            error = f"{modality}/{path.name}: {type(exc).__name__}: {exc}"
            failed_by_path[path] = error
            preferred_errors.append(error)
            break

    if len(preferred) == len(UTSW_MODALITIES):
        preferred_shapes = {tuple(volume.shape) for _, _, volume in preferred}
        if len(preferred_shapes) == 1:
            return [volume for _, _, volume in preferred], [path for _, path, _ in preferred]

    loaded_by_modality: dict[str, list[tuple[int, Path, np.ndarray]]] = {}
    load_errors: list[str] = list(preferred_errors)
    for modality in UTSW_MODALITIES:
        loaded: list[tuple[int, Path, np.ndarray]] = []
        for rank, path in enumerate(candidate_lists[modality]):
            if path in preloaded_by_path:
                loaded.append((rank, path, preloaded_by_path[path]))
                continue
            if path in failed_by_path:
                load_errors.append(failed_by_path[path])
                continue
            try:
                volume = _load_depth_first_nifti(path)
                preloaded_by_path[path] = volume
                loaded.append((rank, path, volume))
            except Exception as exc:
                error = f"{modality}/{path.name}: {type(exc).__name__}: {exc}"
                failed_by_path[path] = error
                load_errors.append(error)
        if not loaded:
            raise RuntimeError(
                f"Could not load any {modality} modality for {case_dir.name}: {_format_load_errors(load_errors)}"
            )
        loaded_by_modality[modality] = loaded

    common_shapes: set[tuple[int, ...]] | None = None
    for loaded in loaded_by_modality.values():
        shapes = {tuple(volume.shape) for _, _, volume in loaded}
        common_shapes = shapes if common_shapes is None else common_shapes & shapes
    if not common_shapes:
        shape_report = []
        for modality, loaded in loaded_by_modality.items():
            shapes = ", ".join(f"{path.name}:{tuple(volume.shape)}" for _, path, volume in loaded)
            shape_report.append(f"{modality} [{shapes}]")
        raise ValueError(
            f"No common modality shape for {case_dir.name}; "
            f"{'; '.join(shape_report)}. Load errors: {_format_load_errors(load_errors)}"
        )

    def shape_score(shape: tuple[int, ...]) -> tuple[int, tuple[int, ...]]:
        score = 0
        for modality in UTSW_MODALITIES:
            ranks = [rank for rank, _, volume in loaded_by_modality[modality] if tuple(volume.shape) == shape]
            score += min(ranks)
        return score, shape

    selected_shape = min(common_shapes, key=shape_score)
    selected: list[tuple[Path, np.ndarray]] = []
    for modality in UTSW_MODALITIES:
        matches = [
            (rank, path, volume)
            for rank, path, volume in loaded_by_modality[modality]
            if tuple(volume.shape) == selected_shape
        ]
        _, path, volume = min(matches, key=lambda item: item[0])
        selected.append((path, volume))
    return [volume for _, volume in selected], [path for path, _ in selected]


def _load_segmentation(case_dir: Path, prefer_manual: bool, image_shape: tuple[int, ...]) -> tuple[np.ndarray, Path]:
    candidates = _segmentation_candidates(case_dir, prefer_manual)
    load_errors: list[str] = []
    mismatched_shapes: list[str] = []
    for path in candidates:
        try:
            segmentation = _load_depth_first_nifti(path)
        except Exception as exc:
            load_errors.append(f"{path.name}: {type(exc).__name__}: {exc}")
            continue
        if tuple(segmentation.shape) == tuple(image_shape):
            return segmentation, path
        mismatched_shapes.append(f"{path.name}:{tuple(segmentation.shape)}")
    if not candidates:
        raise FileNotFoundError(f"Missing segmentation in {case_dir}")
    raise ValueError(
        f"No segmentation for {case_dir.name} matches image shape {tuple(image_shape)}; "
        f"mismatched shapes: {', '.join(mismatched_shapes) or 'none loaded'}. "
        f"Load errors: {_format_load_errors(load_errors)}"
    )


def _clean_category(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float) and np.isnan(value):
        return "NA"
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "na", "n/a"}:
        return "NA"
    return text


def _to_float(value: Any) -> float | None:
    text = _clean_category(value)
    if text == "NA":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _default_metadata_path(root: Path) -> Path:
    return root.parent.parent / UTSW_METADATA_FILENAME


class UTSWMetadataEncoder:
    """Encodes observed UTSW proxy variables by causal role.

    The encoder is fit on the full metadata table, not on a split subset, so
    train/validation/test datasets share identical feature dimensions.
    """

    def __init__(self, metadata_path: str | Path) -> None:
        self.metadata_path = Path(metadata_path)
        frame = pd.read_csv(self.metadata_path, sep="\t")
        self.scm = default_utsw_scm()
        self.scm.validate_metadata_columns(frame.columns)
        self.frame = frame.set_index("Subject ID", drop=False)
        self.context_numeric = UTSW_CONTEXT_NUMERIC_COLUMNS
        self.disease_numeric = UTSW_DISEASE_NUMERIC_COLUMNS
        self.context_categorical = tuple(
            column for column in self.scm.context_columns if column not in self.context_numeric
        )
        self.disease_categorical = tuple(
            column for column in self.scm.disease_columns if column not in self.disease_numeric
        )
        self.annotation_categorical = self.scm.annotation_columns
        self.numeric_stats = {
            column: self._numeric_stats(frame[column])
            for column in self.context_numeric + self.disease_numeric
        }
        self.vocab = {
            column: sorted({_clean_category(value) for value in frame[column].tolist()})
            for column in self.context_categorical + self.disease_categorical + self.annotation_categorical
        }
        self.context_dim = len(self.context_numeric) + sum(len(self.vocab[column]) for column in self.context_categorical)
        self.disease_dim = len(self.disease_numeric) + sum(len(self.vocab[column]) for column in self.disease_categorical)
        self.annotation_dim = sum(len(self.vocab[column]) for column in self.annotation_categorical)
        self.treatment_dim = 2

    def _layout(
        self,
        numeric_columns: Iterable[str],
        categorical_columns: Iterable[str],
    ) -> list[dict[str, Any]]:
        layout: list[dict[str, Any]] = []
        offset = 0
        for column in numeric_columns:
            layout.append({"name": column, "kind": "numeric", "start": offset, "end": offset + 1})
            offset += 1
        for column in categorical_columns:
            width = len(self.vocab[column])
            layout.append(
                {
                    "name": column,
                    "kind": "categorical",
                    "start": offset,
                    "end": offset + width,
                    "categories": list(self.vocab[column]),
                }
            )
            offset += width
        return layout

    def proxy_layout(self) -> dict[str, list[dict[str, Any]]]:
        return {
            "context": self._layout(self.context_numeric, self.context_categorical),
            "disease": self._layout(self.disease_numeric, self.disease_categorical),
            "annotation": self._layout((), self.annotation_categorical),
        }

    @staticmethod
    def _numeric_stats(series: pd.Series) -> tuple[float, float]:
        values = np.asarray([_to_float(value) for value in series.tolist()], dtype=object)
        values_float = np.asarray([value for value in values if value is not None], dtype=np.float32)
        if values_float.size == 0:
            return 0.0, 1.0
        mean = float(values_float.mean())
        std = float(values_float.std())
        return mean, std if std > 1e-6 else 1.0

    def _numeric_feature(self, row: pd.Series, column: str) -> list[float]:
        mean, std = self.numeric_stats[column]
        value = _to_float(row.get(column))
        if value is None:
            value = mean
        return [float((value - mean) / std)]

    def _categorical_feature(self, row: pd.Series, column: str) -> list[float]:
        value = _clean_category(row.get(column))
        vocab = self.vocab[column]
        return [1.0 if value == option else 0.0 for option in vocab]

    def _encode_group(
        self,
        row: pd.Series,
        numeric_columns: tuple[str, ...],
        categorical_columns: tuple[str, ...],
    ) -> Tensor:
        features: list[float] = []
        for column in numeric_columns:
            features.extend(self._numeric_feature(row, column))
        for column in categorical_columns:
            features.extend(self._categorical_feature(row, column))
        return torch.tensor(features, dtype=torch.float32)

    def encode(self, subject_id: str) -> dict[str, Any]:
        if subject_id not in self.frame.index:
            raise KeyError(f"Subject {subject_id!r} is missing from {self.metadata_path}")
        row = self.frame.loc[subject_id]
        scanner_strength = _to_float(row.get("Scanner Strength"))
        if scanner_strength is None:
            scanner_strength = self.numeric_stats["Scanner Strength"][0]
        treatment_label = int(scanner_strength >= 3.0)
        treatment = torch.zeros(self.treatment_dim, dtype=torch.float32)
        treatment[treatment_label] = 1.0
        return {
            "observed_context": self._encode_group(row, self.context_numeric, self.context_categorical),
            "observed_disease": self._encode_group(row, self.disease_numeric, self.disease_categorical),
            "observed_annotation": self._encode_group(row, (), self.annotation_categorical),
            "observed_treatment": treatment,
            "observed_treatment_label": torch.tensor(treatment_label, dtype=torch.long),
            "metadata_raw": {column: _clean_category(row.get(column)) for column in self.scm.context_columns + self.scm.disease_columns + self.scm.annotation_columns},
        }


class UTSWGliomaDataset(Dataset):
    """UTSW glioma NIfTI adapter for the SegFormer3D baseline.

    Returns images as `(4, S, S, S)` and BraTS subregion masks as `(3, S, S, S)`.
    The upstream SegFormer3D code assumes cubic feature grids, so this adapter
    crops foreground and resizes each case to a cubic `volume_size`.
    """

    def __init__(
        self,
        root: str | Path,
        volume_size: int = 128,
        case_ids: list[str] | None = None,
        limit: int | None = None,
        crop_margin: int = 8,
        prefer_manual_seg: bool = False,
        use_ants_modalities: bool = False,
        metadata_path: str | Path | None = None,
        include_metadata: bool = True,
    ) -> None:
        self.root = Path(root).expanduser()
        if not self.root.exists():
            raise FileNotFoundError(f"UTSW root does not exist: {self.root}")
        self.volume_size = int(volume_size)
        self.crop_margin = int(crop_margin)
        self.prefer_manual_seg = bool(prefer_manual_seg)
        self.use_ants_modalities = bool(use_ants_modalities)
        self.metadata_encoder: UTSWMetadataEncoder | None = None
        if include_metadata:
            resolved_metadata = Path(metadata_path) if metadata_path is not None else _default_metadata_path(self.root)
            if resolved_metadata.exists():
                self.metadata_encoder = UTSWMetadataEncoder(resolved_metadata)

        selected = set(case_ids or [])
        cases = sorted(path for path in self.root.iterdir() if path.is_dir())
        if selected:
            cases = [path for path in cases if path.name in selected]
        if limit is not None:
            cases = cases[: int(limit)]
        if not cases:
            raise ValueError(f"No UTSW cases found under {self.root}")
        self.cases = cases

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, index: int) -> dict[str, Any]:
        case_dir = self.cases[index]
        volumes, modality_paths = _load_modalities(case_dir, self.use_ants_modalities)
        volumes = [_normalize_mri(volume) for volume in volumes]
        image = np.stack(volumes, axis=0)

        image_shape = image.shape[1:]
        segmentation, seg_path = _load_segmentation(case_dir, self.prefer_manual_seg, image_shape)

        mask = _subregion_mask(segmentation)
        image, mask = _crop_to_foreground(image, mask, self.crop_margin)
        image_tensor, mask_tensor = _resize_volume(image, mask, self.volume_size)
        mask_tensor = mask_tensor.clamp(0.0, 1.0)

        item: dict[str, Any] = {
            "case_id": case_dir.name,
            "image": image_tensor,
            "mask": mask_tensor,
            "source_shape": torch.tensor(image_shape, dtype=torch.long),
            "segmentation_path": str(seg_path),
            "modality_paths": [str(path) for path in modality_paths],
        }
        if self.metadata_encoder is not None:
            item.update(self.metadata_encoder.encode(case_dir.name))
        return item

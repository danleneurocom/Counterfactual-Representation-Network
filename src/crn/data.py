from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F


def _normalise_array(array: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return array.astype(np.float32, copy=False)
    if mode == "minmax":
        array = array.astype(np.float32, copy=False)
        min_value = float(np.nanmin(array))
        max_value = float(np.nanmax(array))
        if max_value <= min_value:
            return np.zeros_like(array, dtype=np.float32)
        return (array - min_value) / (max_value - min_value)
    if mode == "auto":
        array = array.astype(np.float32, copy=False)
        if np.nanmax(array) > 1.0:
            return array / 255.0
        return array
    raise ValueError(f"Unknown image normalization mode: {mode}")


def _resolve_path(root: str | Path | None, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute() or root in (None, ""):
        return path
    return Path(root) / path


def _pil_mode(in_channels: int) -> str:
    if in_channels == 1:
        return "L"
    if in_channels == 3:
        return "RGB"
    if in_channels == 4:
        return "RGBA"
    raise ValueError("PIL inputs only support 1, 3, or 4 channels. Use .npy for other channel counts.")


def _array_to_tensor(array: np.ndarray, channels: int | None = None, normalization: str = "auto") -> Tensor:
    array = _normalise_array(array, normalization)
    if array.ndim == 2:
        array = array[None, :, :]
    elif array.ndim == 3 and array.shape[0] in {1, 3, 4}:
        pass
    elif array.ndim == 3:
        array = np.moveaxis(array, -1, 0)
    elif array.ndim == 4:
        pass  # volume: (C, D, H, W)
    else:
        raise ValueError(f"Expected 2D, 3D or 4D image array, got shape {array.shape}")

    tensor = torch.from_numpy(np.ascontiguousarray(array)).float()
    if channels is not None and tensor.shape[0] != channels:
        if tensor.shape[0] == 1 and channels > 1:
            # For 3D volumes, expand only spatial dims, keep depth
            if tensor.ndim == 4:
                tensor = tensor.expand(channels, -1, -1, -1)
            else:
                tensor = tensor.expand(channels, -1, -1)
        else:
            raise ValueError(f"Expected {channels} channels, got {tensor.shape[0]}")
    return tensor.clamp(0.0, 1.0)


def _resize_tensor(tensor: Tensor, image_size: tuple[int, int], is_mask: bool) -> Tensor:
    if tensor.shape[-2:] == image_size:
        return tensor
    if tensor.ndim == 4:
        mode = "nearest" if is_mask else "trilinear"
        kwargs: dict[str, Any] = {} if is_mask else {"align_corners": False}
        target_size = (tensor.shape[-3], *image_size)
        return F.interpolate(tensor.unsqueeze(0), size=target_size, mode=mode, **kwargs).squeeze(0)
    mode = "nearest" if is_mask else "bilinear"
    kwargs = {} if is_mask else {"align_corners": False}
    return F.interpolate(tensor.unsqueeze(0), size=image_size, mode=mode, **kwargs).squeeze(0)


def _mask_to_tensor(array: np.ndarray, mask_mode: str = "any", mask_channel: int | None = None) -> Tensor:
    tensor = _array_to_tensor(array, normalization="none")
    if mask_channel is not None:
        tensor = tensor[mask_channel : mask_channel + 1]
    elif mask_mode == "any" and tensor.shape[0] > 1:
        tensor = (tensor > 0).any(dim=0, keepdim=True).float()
    elif mask_mode == "brats_subregions":
        tensor = (tensor > 0).float()
    elif mask_mode == "all_channels":
        tensor = (tensor > 0).float()
    elif mask_mode != "any":
        raise ValueError(f"Unknown mask mode: {mask_mode}")
    return (tensor > 0).float()


class ImagingCsvDataset(Dataset):
    def __init__(
        self,
        csv_path: str | Path,
        image_root: str | Path | None,
        image_col: str,
        label_cols: list[str] | None,
        mask_col: str | None = None,
        image_size: tuple[int, int] = (128, 128),
        in_channels: int = 1,
        uncertain_value: float = 0.0,
        missing_label_value: float = 0.0,
        h5_image_key: str = "image",
        h5_mask_key: str = "mask",
        mask_mode: str = "any",
        mask_channel: int | None = None,
        image_normalization: str = "auto",
        slice_context: int = 1,
        slice_context_layout: str = "channels",
    ) -> None:
        self.frame = pd.read_csv(csv_path)
        sort_cols = [column for column in ("volume", "slice") if column in self.frame.columns]
        if sort_cols:
            self.frame = self.frame.sort_values(sort_cols).reset_index(drop=True)
        self.image_root = image_root
        self.image_col = image_col
        self.label_cols = label_cols or []
        self.mask_col = mask_col
        self.image_size = tuple(image_size)
        self.in_channels = in_channels
        self.uncertain_value = uncertain_value
        self.missing_label_value = missing_label_value
        self.h5_image_key = h5_image_key
        self.h5_mask_key = h5_mask_key
        self.mask_mode = mask_mode
        self.mask_channel = mask_channel
        self.image_normalization = image_normalization
        self.slice_context = int(slice_context)
        self.slice_context_layout = str(slice_context_layout).lower()
        if self.slice_context <= 0 or self.slice_context % 2 == 0:
            raise ValueError(f"slice_context must be a positive odd integer, got {self.slice_context}")
        if self.slice_context_layout not in {"channels", "depth"}:
            raise ValueError(
                "slice_context_layout must be either 'channels' or 'depth', "
                f"got {self.slice_context_layout!r}"
            )

        missing_cols = [col for col in [image_col, mask_col, *self.label_cols] if col and col not in self.frame]
        if missing_cols:
            raise ValueError(f"Missing columns in {csv_path}: {missing_cols}")

        self._volume_slice_lookup: dict[tuple[int, int], int] = {}
        self._volume_slice_bounds: dict[int, tuple[int, int]] = {}
        if self.slice_context > 1:
            required_context_cols = {"volume", "slice"}
            missing_context_cols = required_context_cols.difference(self.frame.columns)
            if missing_context_cols:
                raise ValueError(
                    f"slice_context={self.slice_context} requires metadata columns {sorted(required_context_cols)}; "
                    f"missing {sorted(missing_context_cols)}"
                )
            grouped = self.frame.groupby("volume", sort=False)["slice"]
            self._volume_slice_bounds = {
                int(volume): (int(values.min()), int(values.max()))
                for volume, values in grouped
            }
            self._volume_slice_lookup = {
                (int(row.volume), int(row.slice)): int(index)
                for index, row in self.frame.loc[:, ["volume", "slice"]].reset_index(drop=True).iterrows()
            }

    def __len__(self) -> int:
        return len(self.frame)

    def _load_image(self, path: Path, is_mask: bool = False) -> Tensor:
        if path.suffix.lower() in {".h5", ".hdf5"}:
            try:
                import h5py
            except ImportError as exc:
                raise ImportError("Install h5py to read BraTS .h5 slices: python -m pip install h5py") from exc

            with h5py.File(path, "r") as handle:
                key = self.h5_mask_key if is_mask else self.h5_image_key
                if key not in handle:
                    raise KeyError(f"{path} does not contain HDF5 key {key!r}; available keys: {list(handle.keys())}")
                array = np.asarray(handle[key])

            if is_mask:
                tensor = _mask_to_tensor(array, mask_mode=self.mask_mode, mask_channel=self.mask_channel)
            else:
                tensor = _array_to_tensor(array, self.in_channels, normalization=self.image_normalization)
            return _resize_tensor(tensor, self.image_size, is_mask=is_mask)

        if path.suffix.lower() == ".npy":
            if is_mask:
                tensor = _mask_to_tensor(np.load(path), mask_mode=self.mask_mode, mask_channel=self.mask_channel)
            else:
                tensor = _array_to_tensor(np.load(path), self.in_channels, normalization=self.image_normalization)
            return _resize_tensor(tensor, self.image_size, is_mask=is_mask)

        with Image.open(path) as image:
            if is_mask:
                image = image.convert("L")
                resample = Image.Resampling.NEAREST
            else:
                image = image.convert(_pil_mode(self.in_channels))
                resample = Image.Resampling.BILINEAR
            image = image.resize((self.image_size[1], self.image_size[0]), resample=resample)
            tensor = _array_to_tensor(np.asarray(image), None if is_mask else self.in_channels)
            if is_mask:
                tensor = (tensor > 0.5).float()
            return tensor

    def _labels(self, row: pd.Series) -> Tensor:
        values = row[self.label_cols].astype(float).to_numpy(copy=True)
        values = np.nan_to_num(values, nan=self.missing_label_value)
        values[values < 0] = self.uncertain_value
        return torch.tensor(values, dtype=torch.float32)

    def _context_image(self, row: pd.Series) -> Tensor:
        if self.slice_context == 1 or "volume" not in row.index or "slice" not in row.index:
            image_path = _resolve_path(self.image_root, row[self.image_col])
            image = self._load_image(image_path)
            if self.slice_context_layout == "depth":
                if image.ndim == 3:
                    return image.unsqueeze(1)
                # image is already a 4D volume (C, D, H, W)
                return image
            return image

        half_window = self.slice_context // 2
        volume_id = int(row["volume"])
        center_slice = int(row["slice"])
        min_slice, max_slice = self._volume_slice_bounds[volume_id]

        # Check if data is stored as per-volume files (4D arrays in one file)
        first_path = _resolve_path(self.image_root, row[self.image_col])
        first_slice = self._load_image(first_path)
        if first_slice.ndim == 4:
            # Extract window directly from the volume
            start = max(center_slice - half_window, min_slice)
            end = min(center_slice + half_window, max_slice) + 1
            extracted: list[Tensor] = [first_slice[:, s] for s in range(start, end)]
            while len(extracted) < self.slice_context:
                if center_slice - min_slice < max_slice - center_slice:
                    extracted.insert(0, extracted[0])
                else:
                    extracted.append(extracted[-1])
            if self.slice_context_layout == "depth":
                return torch.stack(extracted, dim=1)
            return torch.cat(extracted, dim=0)

        slices: list[Tensor] = []
        for delta in range(-half_window, half_window + 1):
            slice_id = min(max(center_slice + delta, min_slice), max_slice)
            lookup_index = self._volume_slice_lookup[(volume_id, slice_id)]
            context_row = self.frame.iloc[lookup_index]
            image_path = _resolve_path(self.image_root, context_row[self.image_col])
            slices.append(self._load_image(image_path))
        if self.slice_context_layout == "depth":
            return torch.stack(slices, dim=1)
        return torch.cat(slices, dim=0)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        image_path = _resolve_path(self.image_root, row[self.image_col])
        image = self._context_image(row)
        # If the loaded data is a full volume (C, D, H, W) and we only want one slice, extract it
        if self.slice_context == 1 and image.ndim == 4 and "slice" in row.index:
            image = image[:, int(row["slice"])]
            if self.slice_context_layout == "depth":
                image = image.unsqueeze(1)
        item: dict[str, Any] = {
            "image": image,
            "index": int(index),
            "path": str(image_path),
        }
        for key in ("volume", "slice"):
            if key in row.index:
                item[key] = int(row[key])
        if self.label_cols:
            item["label"] = self._labels(row)
        if self.mask_col:
            mask_path = _resolve_path(self.image_root, row[self.mask_col])
            mask = self._load_image(mask_path, is_mask=True)
            if mask.ndim == 4 and "slice" in row.index:
                mask = mask[:, int(row["slice"])]
            item["mask"] = mask
        return item


def make_dataloader(config: dict[str, Any], split: str, shuffle: bool) -> DataLoader:
    csv_key = f"{split}_csv"
    if csv_key not in config or not config[csv_key]:
        raise ValueError(f"Missing data.{csv_key} in config")

    dataset = ImagingCsvDataset(
        csv_path=config[csv_key],
        image_root=config.get("image_root"),
        image_col=config.get("image_col", "path"),
        label_cols=config.get("label_cols"),
        mask_col=config.get("mask_col"),
        image_size=tuple(config.get("image_size", (128, 128))),
        in_channels=int(config.get("in_channels", 1)),
        uncertain_value=float(config.get("uncertain_value", 0.0)),
        missing_label_value=float(config.get("missing_label_value", 0.0)),
        h5_image_key=config.get("h5_image_key", "image"),
        h5_mask_key=config.get("h5_mask_key", "mask"),
        mask_mode=config.get("mask_mode", "any"),
        mask_channel=config.get("mask_channel"),
        image_normalization=config.get("image_normalization", "auto"),
        slice_context=int(config.get("slice_context", 1)),
        slice_context_layout=config.get("slice_context_layout", "channels"),
    )
    return DataLoader(
        dataset,
        batch_size=int(config.get("batch_size", 8)),
        shuffle=shuffle,
        num_workers=int(config.get("num_workers", 0)),
        pin_memory=bool(config.get("pin_memory", False)),
    )

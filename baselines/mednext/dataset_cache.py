from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import re
import tempfile
from time import perf_counter
from typing import Any

import torch
from torch.utils.data import Dataset


def _safe_name(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return name.strip("._") or "dataset"


def _signature_hash(signature: Mapping[str, Any] | None) -> str:
    if not signature:
        return "default"
    payload = json.dumps(signature, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:12]


def _identity_path_component(value: Any) -> str:
    text = str(value)
    safe = _safe_name(text)
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
    return f"{safe[:64]}-{digest}"


def _dataset_item_key(dataset: Dataset, index: int) -> str:
    if hasattr(dataset, "cache_key"):
        return str(dataset.cache_key(index))
    volumes = getattr(dataset, "volumes", None)
    if volumes is not None:
        volume = volumes[int(index)]
        if isinstance(volume, tuple) and volume:
            return f"volume:{volume[0]}"
    cases = getattr(dataset, "cases", None)
    if cases is not None:
        case = cases[int(index)]
        return f"case:{getattr(case, 'name', case)}"
    return f"index:{int(index)}"


def _cached_item_key(item: Any) -> str | None:
    if not isinstance(item, Mapping):
        return None
    if "volume" in item:
        return f"volume:{item['volume']}"
    if "volume_id" in item:
        return f"volume:{item['volume_id']}"
    if "case_id" in item:
        return f"case:{item['case_id']}"
    if "case" in item:
        return f"case:{item['case']}"
    return None


def _tensor_volume_shape_matches(item: Any, signature: Mapping[str, Any] | None) -> bool:
    volume_size = None if signature is None else signature.get("volume_size")
    if volume_size is None:
        return True
    if isinstance(volume_size, int):
        expected = (int(volume_size),) * 3
    elif isinstance(volume_size, (list, tuple)) and len(volume_size) == 3:
        expected = tuple(int(value) for value in volume_size)
    else:
        return True
    if not isinstance(item, Mapping):
        return True
    for key in ("image", "mask"):
        value = item.get(key)
        if isinstance(value, torch.Tensor) and value.ndim >= 3 and tuple(value.shape[-3:]) != expected:
            return False
    return True


class DiskCachedDataset(Dataset):
    """Lazy on-disk cache for expensive MedNeXt volume dataset items."""

    def __init__(
        self,
        dataset: Dataset,
        cache_dir: str | Path,
        namespace: str,
        signature: Mapping[str, Any] | None = None,
    ) -> None:
        self.dataset = dataset
        self.signature = dict(signature or {})
        self.namespace_dir = Path(cache_dir).expanduser() / _safe_name(namespace)
        self.cache_dir = self.namespace_dir / _signature_hash(signature)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.legacy_dirs = tuple(
            path
            for path in sorted(self.namespace_dir.iterdir())
            if path.is_dir() and path != self.cache_dir
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self.dataset, name)

    def __len__(self) -> int:
        return len(self.dataset)

    def _path(self, index: int) -> Path:
        return self.cache_dir / f"{_identity_path_component(_dataset_item_key(self.dataset, int(index)))}.pt"

    def cache_path(self, index: int) -> Path:
        return self._path(index)

    def is_cached(self, index: int) -> bool:
        return self._path(index).exists()

    def cache_status(self, start_index: int = 0) -> dict[str, Any]:
        start = max(0, int(start_index))
        cached_indices: list[int] = []
        missing_indices: list[int] = []
        for index in range(start, len(self)):
            if self.is_cached(index):
                cached_indices.append(index)
            else:
                missing_indices.append(index)
        return {
            "enabled": True,
            "cache_dir": str(self.cache_dir),
            "start_index": start,
            "total_items": len(self),
            "cached_count": len(cached_indices),
            "missing_count": len(missing_indices),
            "cached_indices": cached_indices,
            "missing_indices": missing_indices,
        }

    def warm_cache(
        self,
        *,
        start_index: int = 0,
        max_items: int | None = None,
        missing_only: bool = True,
    ) -> dict[str, Any]:
        start = max(0, int(start_index))
        limit = None if max_items is None else max(0, int(max_items))
        before = self.cache_status(start_index=start)
        warmed: list[dict[str, Any]] = []
        skipped_cached = 0
        started = perf_counter()
        for index in range(start, len(self)):
            if limit is not None and len(warmed) >= limit:
                break
            if missing_only and self.is_cached(index):
                skipped_cached += 1
                continue
            item_start = perf_counter()
            _ = self[index]
            warmed.append(
                {
                    "index": int(index),
                    "key": _dataset_item_key(self.dataset, int(index)),
                    "path": str(self._path(index)),
                    "elapsed_sec": float(perf_counter() - item_start),
                }
            )
        after = self.cache_status(start_index=start)
        return {
            "enabled": True,
            "cache_dir": str(self.cache_dir),
            "start_index": start,
            "max_items": limit,
            "missing_only": bool(missing_only),
            "before": before,
            "after": after,
            "warmed_count": len(warmed),
            "skipped_cached": skipped_cached,
            "elapsed_sec": float(perf_counter() - started),
            "warmed": warmed,
        }

    def _load_cache_file(self, path: Path) -> Any:
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(path, map_location="cpu")

    def _write_cache_file(self, item: Any, path: Path, index: int) -> None:
        with tempfile.NamedTemporaryFile(dir=self.cache_dir, prefix=f".{int(index):08d}.", suffix=".tmp", delete=False) as handle:
            tmp_path = Path(handle.name)
        try:
            torch.save(item, tmp_path)
            tmp_path.replace(path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    @staticmethod
    def _unlink_cache_file(path: Path) -> None:
        try:
            path.unlink()
        except FileNotFoundError:
            pass

    def _valid_legacy_item(self, legacy_path: Path, expected_key: str) -> Any | None:
        if not legacy_path.exists():
            return None
        try:
            item = self._load_cache_file(legacy_path)
        except Exception:
            return None
        if _cached_item_key(item) != expected_key:
            return None
        if not _tensor_volume_shape_matches(item, self.signature):
            return None
        return item

    def _legacy_index_item(self, index: int) -> Any | None:
        expected_key = _dataset_item_key(self.dataset, int(index))
        legacy_name = f"{int(index):08d}.pt"
        identity_name = f"{_identity_path_component(expected_key)}.pt"
        for legacy_dir in self.legacy_dirs:
            checked_paths: set[Path] = set()
            for candidate_name in (identity_name, legacy_name):
                candidate = legacy_dir / candidate_name
                checked_paths.add(candidate)
                item = self._valid_legacy_item(candidate, expected_key)
                if item is not None:
                    return item
            for legacy_path in sorted(legacy_dir.glob("*.pt")):
                if legacy_path in checked_paths or re.fullmatch(r"\d{8}\.pt", legacy_path.name) is None:
                    continue
                item = self._valid_legacy_item(legacy_path, expected_key)
                if item is not None:
                    return item
        return None

    def __getitem__(self, index: int) -> Any:
        path = self._path(index)
        if path.exists():
            try:
                return self._load_cache_file(path)
            except Exception:
                self._unlink_cache_file(path)
        legacy_item = self._legacy_index_item(int(index))
        if legacy_item is not None:
            self._write_cache_file(legacy_item, path, int(index))
            return legacy_item
        item = self.dataset[index]
        self._write_cache_file(item, path, int(index))
        return item


def maybe_disk_cache_dataset(
    dataset: Dataset,
    cache_dir: str | Path | None,
    namespace: str,
    signature: Mapping[str, Any] | None = None,
) -> Dataset:
    if cache_dir is None or str(cache_dir).strip() == "":
        return dataset
    return DiskCachedDataset(dataset, cache_dir=cache_dir, namespace=namespace, signature=signature)


def warm_disk_cache_dataset(
    dataset: Dataset,
    *,
    start_index: int = 0,
    max_items: int | None = None,
    missing_only: bool = True,
) -> dict[str, Any]:
    if not isinstance(dataset, DiskCachedDataset):
        return {"enabled": False, "reason": "dataset is not disk cached"}
    return dataset.warm_cache(start_index=start_index, max_items=max_items, missing_only=missing_only)

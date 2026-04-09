from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _path_candidates(raw_path: str, data_root: Path | None) -> list[Path]:
    raw = Path(raw_path)
    candidates = [raw]
    if data_root is None:
        return candidates

    parts = raw.parts
    suffix_from_content = None
    if "content" in parts:
        suffix_from_content = Path(*parts[parts.index("content") :])
    suffix_from_data = None
    if "data" in parts:
        suffix_from_data = Path(*parts[parts.index("data") :])

    candidates.extend(
        [
            data_root / raw_path,
            data_root / raw.name,
            data_root / "content" / "data" / raw.name,
            data_root / "data" / raw.name,
        ]
    )
    if suffix_from_content is not None:
        candidates.append(data_root / suffix_from_content)
    if suffix_from_data is not None:
        candidates.append(data_root / suffix_from_data)
    return candidates


def resolve_slice_path(raw_path: str, data_root: Path | None) -> Path:
    candidates = _path_candidates(raw_path, data_root)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    if data_root is not None:
        return (data_root / Path(raw_path).name).resolve()
    return Path(raw_path)


def split_by_volume(frame: pd.DataFrame, val_fraction: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    volumes = frame["volume"].drop_duplicates().to_numpy(copy=True)
    rng.shuffle(volumes)
    val_count = max(1, round(len(volumes) * val_fraction))
    val_volumes = set(volumes[:val_count])
    val_mask = frame["volume"].isin(val_volumes)
    return frame.loc[~val_mask].reset_index(drop=True), frame.loc[val_mask].reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare BraTS2020 HDF5 slice metadata for CRN training.")
    parser.add_argument("--metadata", default="BraTS20 Training Metadata.csv")
    parser.add_argument("--data-root", help="Directory containing the .h5 slice files.")
    parser.add_argument("--output-dir", default="data/brats")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--require-files", action="store_true", help="Fail if any resolved .h5 path is missing.")
    args = parser.parse_args()

    metadata = Path(args.metadata)
    data_root = Path(args.data_root).expanduser().resolve() if args.data_root else None
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(metadata)
    required = {"slice_path", "target", "volume", "slice"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required metadata columns: {sorted(missing)}")

    frame = frame.copy()
    frame["path"] = [str(resolve_slice_path(path, data_root)) for path in frame["slice_path"]]
    frame["mask"] = frame["path"]

    exists = frame["path"].map(lambda value: Path(value).exists())
    missing_count = int((~exists).sum())
    if args.require_files and missing_count:
        examples = frame.loc[~exists, "path"].head(5).to_list()
        raise FileNotFoundError(f"{missing_count} slice files are missing. Examples: {examples}")

    train, val = split_by_volume(frame, args.val_fraction, args.seed)
    train_path = output_dir / "brats_train.csv"
    val_path = output_dir / "brats_val.csv"
    train.to_csv(train_path, index=False)
    val.to_csv(val_path, index=False)

    print(f"metadata rows: {len(frame)}")
    print(f"volumes: {frame['volume'].nunique()}")
    print(f"missing resolved files: {missing_count}")
    print(f"train: {len(train)} rows, {train['volume'].nunique()} volumes -> {train_path}")
    print(f"val: {len(val)} rows, {val['volume'].nunique()} volumes -> {val_path}")


if __name__ == "__main__":
    main()

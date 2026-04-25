"""FeTS leave-one-institution-out data helpers — Phase C.5.

FeTS (Pati et al. 2021, arXiv:2105.05874; Pati et al. 2022, Nat Commun
DOI 10.1038/s41467-022-33407-5) distributes 71 sites / 6,314 GBM patients.
Access is gated via Synapse/CBICA and must be requested separately.

This module's responsibility is bookkeeping:

* read a federated partition CSV (`site, volume, slice, path, mask, ...`) that
  the user has already assembled from the FeTS release;
* produce per-site train/test CSV splits under a LOIO scheme so the main
  training and evaluation entrypoints consume them unchanged;
* nothing here converts NIfTI → HDF5 — reuse
  `src/crn/data_brats_africa.py::convert_case` if needed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(f"Partition CSV is missing required columns: {missing}. Has: {list(frame.columns)}")


def write_loio_splits(
    partition_csv: Path,
    output_dir: Path,
    site_column: str = "site",
) -> list[dict[str, object]]:
    """Write one (train, test) CSV pair per held-out site.

    For each distinct site `s`:
        train_<s>.csv = all slices whose site != s
        test_<s>.csv  = all slices whose site == s

    Returns a list of site metadata dicts (site_id, n_train_volumes,
    n_test_volumes, train_csv, test_csv) suitable for a manifest JSON.
    """
    frame = pd.read_csv(partition_csv)
    _require_columns(frame, (site_column, "volume", "path", "mask"))
    output_dir.mkdir(parents=True, exist_ok=True)

    sites = sorted(frame[site_column].unique())
    manifest: list[dict[str, object]] = []
    for site in sites:
        test_frame = frame.loc[frame[site_column] == site].reset_index(drop=True)
        train_frame = frame.loc[frame[site_column] != site].reset_index(drop=True)
        train_csv = output_dir / f"train_holdout_{site}.csv"
        test_csv = output_dir / f"test_holdout_{site}.csv"
        train_frame.to_csv(train_csv, index=False)
        test_frame.to_csv(test_csv, index=False)
        manifest.append(
            {
                "site": site,
                "n_train_volumes": int(train_frame["volume"].nunique()),
                "n_test_volumes": int(test_frame["volume"].nunique()),
                "train_csv": str(train_csv),
                "test_csv": str(test_csv),
            }
        )
    manifest_path = output_dir / "loio_manifest.csv"
    pd.DataFrame(manifest).to_csv(manifest_path, index=False)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Write FeTS leave-one-institution-out splits.")
    parser.add_argument("--partition-csv", required=True, help="Full FeTS partition CSV with a 'site' column.")
    parser.add_argument("--output-dir", default="data/fets/loio")
    parser.add_argument("--site-column", default="site")
    args = parser.parse_args()
    manifest = write_loio_splits(Path(args.partition_csv), Path(args.output_dir), args.site_column)
    print(f"Wrote {len(manifest)} LOIO splits to {args.output_dir}.")


if __name__ == "__main__":
    main()

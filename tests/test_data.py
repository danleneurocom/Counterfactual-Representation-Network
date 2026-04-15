from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from crn.data import ImagingCsvDataset


def test_slice_context_stacks_neighbor_slices(tmp_path: Path) -> None:
    rows = []
    for slice_id, value in enumerate((0.1, 0.5, 0.9), start=10):
        image = np.full((6, 6, 1), value, dtype=np.float32)
        mask = np.ones((6, 6, 1), dtype=np.uint8)
        image_path = tmp_path / f"slice_{slice_id}.npy"
        mask_path = tmp_path / f"slice_{slice_id}_mask.npy"
        np.save(image_path, image)
        np.save(mask_path, mask)
        rows.append(
            {
                "path": str(image_path),
                "mask": str(mask_path),
                "target": 1,
                "volume": 5,
                "slice": slice_id,
            }
        )

    csv_path = tmp_path / "dataset.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    dataset = ImagingCsvDataset(
        csv_path=csv_path,
        image_root=None,
        image_col="path",
        label_cols=["target"],
        mask_col="mask",
        image_size=(6, 6),
        in_channels=1,
        image_normalization="none",
        slice_context=3,
    )

    middle = dataset[1]
    assert middle["image"].shape == (3, 6, 6)
    assert torch.allclose(middle["image"][0], torch.full((6, 6), 0.1))
    assert torch.allclose(middle["image"][1], torch.full((6, 6), 0.5))
    assert torch.allclose(middle["image"][2], torch.full((6, 6), 0.9))

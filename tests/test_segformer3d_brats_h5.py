from __future__ import annotations

from pathlib import Path

import numpy as np

from baselines.segformer3d.evaluate_causal_brats_h5 import (
    _image_slice_to_channels,
    _mask_slice_to_channels,
    _resolve_path,
)


def test_brats_h5_image_slice_channel_layouts() -> None:
    channels_last = np.zeros((8, 9, 4), dtype=np.float32)
    channels_first = np.zeros((4, 8, 9), dtype=np.float32)

    assert _image_slice_to_channels(channels_last).shape == (4, 8, 9)
    assert _image_slice_to_channels(channels_first).shape == (4, 8, 9)


def test_brats_h5_mask_slice_accepts_labels_and_binary_channels() -> None:
    labels = np.array([[0, 1, 2, 4, 3]], dtype=np.uint8)
    mask = _mask_slice_to_channels(labels)

    assert mask.shape == (3, 1, 5)
    assert mask[:, 0, 1].tolist() == [1.0, 0.0, 0.0]
    assert mask[:, 0, 2].tolist() == [0.0, 1.0, 0.0]
    assert mask[:, 0, 3].tolist() == [0.0, 0.0, 1.0]
    assert mask[:, 0, 4].tolist() == [0.0, 0.0, 1.0]

    channels_last = np.zeros((8, 9, 3), dtype=np.float32)
    channels_last[..., 2] = 1.0
    assert _mask_slice_to_channels(channels_last).shape == (3, 8, 9)


def test_resolve_path_uses_data_root_file_name(tmp_path: Path) -> None:
    target = tmp_path / "volume_1_slice_2.h5"
    target.write_bytes(b"fake")

    resolved = _resolve_path("/missing/content/data/volume_1_slice_2.h5", tmp_path)

    assert resolved == target

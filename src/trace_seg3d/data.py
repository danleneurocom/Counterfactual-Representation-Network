"""Datasets, proxy encoding and augmentation for preprocessed ``.npz`` cases."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset

from trace_seg3d.preprocess import load_case

# Context = acquisition / institution only. Patient covariates (age, sex, race) and
# operation status are NOT context: they are related to the disease / anatomy.
CONTEXT_FIELDS = {"utsw": ("scanner_make", "field_strength"), "brats": ("site",)}
DISEASE_FIELDS = {"utsw": ("grade",), "brats": ("grade",)}
VOLUME_SCALE = 1000.0


def label_to_subregions(label: np.ndarray | Tensor) -> Tensor:
    """{0,1,2,3} label map -> float [3, ...] channels (NCR/NET, ED, ET)."""

    label_t = torch.as_tensor(np.asarray(label)).long()
    return torch.stack([(label_t == 1), (label_t == 2), (label_t == 3)]).float()


def region_volume_target(sub: Tensor) -> Tensor:
    """log1p(scale * fraction) of WT, TC, ET for a [B,3,...] or [3,...] subregion mask."""

    if sub.ndim == 4:
        sub = sub.unsqueeze(0)
    wt = sub.amax(dim=1)
    tc = torch.maximum(sub[:, 0], sub[:, 2])
    et = sub[:, 2]
    frac = torch.stack([r.flatten(1).mean(1) for r in (wt, tc, et)], dim=1)
    return torch.log1p(VOLUME_SCALE * frac)


class ProxyEncoder:
    """One categorical head per proxy field; vocab is fitted on the TRAINING split only."""

    def __init__(self, fields: tuple[str, ...], vocab: dict[str, list[str]] | None = None) -> None:
        self.fields = tuple(fields)
        self.vocab: dict[str, list[str]] = dict(vocab or {})

    def fit(self, metas: list[dict[str, Any]]) -> "ProxyEncoder":
        for f in self.fields:
            values = sorted({str(m.get(f, "NA")) for m in metas} - {"NA"})
            self.vocab[f] = values
        return self

    @property
    def active_fields(self) -> list[str]:
        return [f for f in self.fields if len(self.vocab.get(f, [])) >= 2]

    @property
    def sizes(self) -> list[int]:
        return [len(self.vocab[f]) for f in self.active_fields]

    def encode(self, meta: dict[str, Any]) -> Tensor:
        """Class index per active field, -100 (= ignore) when unknown / unseen."""

        out = []
        for f in self.active_fields:
            value = str(meta.get(f, "NA"))
            out.append(self.vocab[f].index(value) if value in self.vocab[f] else -100)
        return torch.tensor(out, dtype=torch.long)

    def state(self) -> dict[str, Any]:
        return {"fields": list(self.fields), "vocab": self.vocab}

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "ProxyEncoder":
        return cls(tuple(state["fields"]), state["vocab"])


def read_meta(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        return json.loads(str(data["meta"]))


class CaseDataset(Dataset):
    def __init__(
        self,
        data_dir: str | Path,
        case_ids: list[str],
        *,
        train: bool = False,
        patch_size: int | None = None,
        fg_prob: float = 0.5,
        context_encoder: ProxyEncoder | None = None,
        disease_encoder: ProxyEncoder | None = None,
        cache_in_memory: bool = False,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.case_ids = list(case_ids)
        missing = [c for c in self.case_ids if not (self.data_dir / f"{c}.npz").exists()]
        if missing:
            raise FileNotFoundError(f"{len(missing)} cases missing in {self.data_dir}, e.g. {missing[:3]}")
        self.train = train
        self.patch_size = patch_size
        self.fg_prob = fg_prob
        self.context_encoder = context_encoder
        self.disease_encoder = disease_encoder
        self.cache_in_memory = cache_in_memory
        self._cache: dict[int, tuple[dict[str, np.ndarray], dict[str, Any]]] = {}

    def __len__(self) -> int:
        return len(self.case_ids)

    def metas(self) -> list[dict[str, Any]]:
        return [read_meta(self.data_dir / f"{c}.npz") for c in self.case_ids]

    def _load(self, index: int) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        if index in self._cache:
            return self._cache[index]
        item = load_case(self.data_dir / f"{self.case_ids[index]}.npz")
        if self.cache_in_memory:
            self._cache[index] = item
        return item

    def _crop(self, image: Tensor, sub: Tensor, brain: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        size = image.shape[-1]
        p = self.patch_size
        if p is None or p >= size:
            return image, sub, brain
        fg = sub.amax(0)
        if random.random() < self.fg_prob and fg.any():
            idx = torch.nonzero(fg)
            center = idx[random.randrange(idx.shape[0])]
        else:
            center = torch.randint(0, size, (3,))
        start = [int(min(max(int(c) - p // 2, 0), size - p)) for c in center]
        sl = tuple(slice(s, s + p) for s in start)
        return image[(slice(None), *sl)], sub[(slice(None), *sl)], brain[sl]

    def __getitem__(self, index: int) -> dict[str, Any]:
        arrays, meta = self._load(index)
        image = torch.from_numpy(arrays["image"].astype(np.float32))
        sub = label_to_subregions(arrays["label"])
        brain = torch.from_numpy(arrays["brain"].astype(bool))
        if self.train:
            image, sub, brain = self._crop(image, sub, brain)
            for axis in (1, 2, 3):
                if random.random() < 0.5:
                    image, sub, brain = image.flip(axis), sub.flip(axis), brain.flip(axis - 1)
            scale = 1.0 + 0.1 * (2 * torch.rand(4, 1, 1, 1) - 1)
            shift = 0.1 * (2 * torch.rand(4, 1, 1, 1) - 1)
            image = torch.where(brain.unsqueeze(0), image * scale + shift, image)
        item: dict[str, Any] = {
            "image": image.contiguous(),
            "target": sub.contiguous(),
            "brain": brain.contiguous(),
            "case_id": meta["case_id"],
            "dataset": meta.get("dataset", "NA"),
            "spacing": float(meta.get("spacing_mm", 1.0)),
        }
        item["context_label"] = self.context_encoder.encode(meta) if self.context_encoder else torch.zeros(0, dtype=torch.long)
        item["disease_label"] = self.disease_encoder.encode(meta) if self.disease_encoder else torch.zeros(0, dtype=torch.long)
        return item


# ----------------------------------------------------------------------------- style intervention


def _smooth_field(shape: tuple[int, ...], device: torch.device, strength: float) -> Tensor:
    """Low-frequency multiplicative bias field in [1-strength, 1+strength]."""

    b = shape[0]
    coarse = torch.rand((b, 1, 4, 4, 4), device=device) * 2 - 1
    field = F.interpolate(coarse, size=shape[2:], mode="trilinear", align_corners=True)
    return 1.0 + strength * field


def style_augment(image: Tensor, brain: Tensor, p: float = 0.8) -> Tensor:
    """Image-level context intervention: random per-modality gamma, contrast, bias field, noise.

    Keeps the lesion geometry (labels unchanged) and only changes acquisition-like appearance.
    ``image`` [B,4,D,H,W] z-scored in brain, ``brain`` [B,D,H,W] bool.
    """

    if random.random() > p:
        return image
    b, c = image.shape[:2]
    mask = brain.unsqueeze(1).float()
    lo = image.amin(dim=(2, 3, 4), keepdim=True)
    hi = image.amax(dim=(2, 3, 4), keepdim=True)
    unit = ((image - lo) / (hi - lo).clamp_min(1e-6)).clamp(0, 1)
    gamma = torch.exp(torch.empty((b, c, 1, 1, 1), device=image.device).uniform_(math.log(0.6), math.log(1.6)))
    unit = unit.pow(gamma)
    unit = unit * _smooth_field(image.shape, image.device, 0.25)
    out = unit * (hi - lo) + lo
    contrast = torch.empty((b, c, 1, 1, 1), device=image.device).uniform_(0.75, 1.25)
    out = out * contrast + 0.05 * torch.randn_like(out)
    # re-standardise inside the brain so the network still sees z-scored inputs
    denom = mask.sum(dim=(2, 3, 4), keepdim=True).clamp_min(1.0)
    mean = (out * mask).sum(dim=(2, 3, 4), keepdim=True) / denom
    std = (((out - mean) ** 2 * mask).sum(dim=(2, 3, 4), keepdim=True) / denom).sqrt().clamp_min(1e-6)
    out = ((out - mean) / std).clamp(-5, 5) * mask
    return out


def load_splits(path: str | Path) -> dict[str, list[str]]:
    data = json.loads(Path(path).read_text())
    return {k: v for k, v in data.items() if not k.startswith("_")}

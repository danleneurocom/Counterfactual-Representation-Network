"""Data adapters for the SegFormer3D baseline."""

from .utsw import UTSWGliomaDataset, UTSW_MODALITIES, UTSW_SUBREGION_LABELS

__all__ = ["UTSWGliomaDataset", "UTSW_MODALITIES", "UTSW_SUBREGION_LABELS"]

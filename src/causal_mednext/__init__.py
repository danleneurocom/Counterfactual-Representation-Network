"""Causal MedNeXt public model package.

The implementation lives in ``src`` so the research model can be imported as a
library. The ``baselines.mednext`` modules remain as compatibility wrappers for
the original training and evaluation scripts.
"""

from .backbone import MedNeXtSegmenter, build_mednext_segmenter
from .causal_model import CausalMedNeXt, build_causal_mednext

__all__ = [
    "CausalMedNeXt",
    "MedNeXtSegmenter",
    "build_causal_mednext",
    "build_mednext_segmenter",
]

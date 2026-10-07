"""Compatibility wrapper for the public Causal MedNeXt model package."""

from causal_mednext.causal_model import CausalMedNeXt, build_causal_mednext
from causal_mednext.mechanisms.gradient_reversal import gradient_reverse

__all__ = [
    "CausalMedNeXt",
    "build_causal_mednext",
    "gradient_reverse",
]

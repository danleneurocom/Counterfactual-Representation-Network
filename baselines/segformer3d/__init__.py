"""SegFormer3D baseline replication from OSUPCVLab/SegFormer3D."""

from .architectures.segformer3d import SegFormer3D, build_segformer3d_model

__all__ = ["SegFormer3D", "build_segformer3d_model"]

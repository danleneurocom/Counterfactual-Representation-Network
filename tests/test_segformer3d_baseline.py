import torch

from baselines.segformer3d import SegFormer3D


def test_segformer3d_baseline_forward_tiny_volume() -> None:
    model = SegFormer3D(
        in_channels=4,
        sr_ratios=[4, 2, 1, 1],
        embed_dims=[4, 8, 16, 32],
        patch_kernel_size=[7, 3, 3, 3],
        patch_stride=[4, 2, 2, 2],
        patch_padding=[3, 1, 1, 1],
        mlp_ratios=[2, 2, 2, 2],
        num_heads=[1, 1, 2, 4],
        depths=[1, 1, 1, 1],
        decoder_head_embedding_dim=8,
        num_classes=3,
        decoder_dropout=0.0,
    )
    model.eval()

    with torch.no_grad():
        output = model(torch.randn(1, 4, 32, 32, 32))

    assert output.shape == (1, 3, 32, 32, 32)

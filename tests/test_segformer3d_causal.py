from pathlib import Path

import pandas as pd
import torch

from baselines.segformer3d.causal import build_causal_segformer3d_tiny, default_utsw_scm
from baselines.segformer3d.data.utsw import UTSWMetadataEncoder
from baselines.segformer3d.train_causal_utsw import _proxy_loss, _subsample_context_bank, _typed_proxy_loss


def test_utsw_scm_matches_metadata_columns() -> None:
    metadata_path = Path("data/brats/UTSW_Glioma_Metadata-2-1.tsv")
    frame = pd.read_csv(metadata_path, sep="\t")
    scm = default_utsw_scm()

    scm.validate_metadata_columns(frame.columns)

    assert scm.question.estimand.startswith("P(M | do(Z_d")
    assert ("Z_d", "M_hat") in scm.edges
    assert ("Z_c", "M_hat") in scm.edges


def test_utsw_metadata_encoder_separates_proxy_roles() -> None:
    encoder = UTSWMetadataEncoder("data/brats/UTSW_Glioma_Metadata-2-1.tsv")
    encoded = encoder.encode("BT0001")

    assert encoded["observed_context"].shape == (encoder.context_dim,)
    assert encoded["observed_disease"].shape == (encoder.disease_dim,)
    assert encoded["observed_annotation"].shape == (encoder.annotation_dim,)
    assert encoded["observed_treatment"].shape == (encoder.treatment_dim,)
    assert int(encoded["observed_treatment_label"]) in (0, 1)
    assert torch.isfinite(encoded["observed_context"]).all()
    assert torch.isfinite(encoded["observed_disease"]).all()
    assert torch.isfinite(encoded["observed_annotation"]).all()
    assert torch.isfinite(encoded["observed_treatment"]).all()
    assert "Scanner Make" in encoded["metadata_raw"]
    assert "Tumor Grade" in encoded["metadata_raw"]

    layout = encoder.proxy_layout()
    assert layout["context"][0]["kind"] == "numeric"
    assert any(spec["kind"] == "categorical" for spec in layout["context"])
    assert layout["context"][-1]["end"] == encoder.context_dim
    assert layout["disease"][-1]["end"] == encoder.disease_dim
    assert layout["annotation"][-1]["end"] == encoder.annotation_dim


def test_causal_segformer3d_exposes_intervention_hooks() -> None:
    torch.manual_seed(7)
    model = build_causal_segformer3d_tiny(
        latent_dim=8,
        context_proxy_dim=3,
        disease_proxy_dim=2,
        annotation_proxy_dim=1,
    )
    model.eval()

    with torch.no_grad():
        outputs = model(torch.randn(1, 4, 32, 32, 32))

    assert outputs["logits"].shape == (1, 3, 32, 32, 32)
    assert outputs["z_d"].shape == (1, 8)
    assert outputs["z_c"].shape == (1, 8)
    assert outputs["context_proxy_logits"].shape == (1, 3)
    assert outputs["disease_proxy_logits"].shape == (1, 2)
    assert outputs["annotation_proxy_logits"].shape == (1, 1)

    with torch.no_grad():
        torch.nn.init.normal_(model.modulator.proj.weight, mean=0.0, std=0.05)
        features = outputs["features"]
        z_d = outputs["z_d"]
        z_c = outputs["z_c"]
        factual = model.segment_from_latents(features, z_d, z_c)
        intervened = model.segment_from_latents(features, z_d, z_c + 1.0)
        context_bank = torch.stack([z_c.squeeze(0), z_c.squeeze(0) + 1.0], dim=0)
        adjusted = model.backdoor_adjusted_logits(features, z_d, context_bank)

    assert factual.shape == intervened.shape == adjusted.shape
    assert not torch.allclose(factual, intervened)


def test_typed_proxy_loss_and_hard_context_bank_sampling() -> None:
    prediction = torch.tensor([[0.2, 2.0, -1.0, 0.5]], dtype=torch.float32)
    target = torch.tensor([[0.0, 1.0, 0.0, 1.0]], dtype=torch.float32)
    layout = [
        {"name": "numeric", "kind": "numeric", "start": 0, "end": 1},
        {"name": "categorical", "kind": "categorical", "start": 1, "end": 4},
    ]

    loss = _typed_proxy_loss(prediction, target, layout)
    mse_loss = _proxy_loss(prediction, target, layout, mode="mse")
    typed_loss = _proxy_loss(prediction, target, layout, mode="typed")

    assert loss is not None
    assert torch.isfinite(loss)
    assert mse_loss is not None
    assert typed_loss is not None
    assert torch.isfinite(mse_loss)
    assert torch.isfinite(typed_loss)
    assert torch.allclose(loss, typed_loss)
    bank = torch.arange(20, dtype=torch.float32).view(10, 2)
    farthest = _subsample_context_bank(bank, max_contexts=4, strategy="farthest", seed=7)
    random = _subsample_context_bank(bank, max_contexts=4, strategy="random", seed=7)

    assert farthest.shape == random.shape == (4, 2)
    assert torch.equal(farthest[0], bank[0])

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from trace_seg3d.audit import aurc, auroc
from trace_seg3d.evaluate import load_calibration
from trace_seg3d.losses import segmentation_loss, stability_loss, subregion_to_region_prob
from trace_seg3d.metrics import HD95_EMPTY_PENALTY_MM, case_metrics, hd95_mm, structural_prior, threshold_regions
from trace_seg3d.model import ContextBank, TraceMedNeXt
from trace_seg3d.preprocess import RawCase, brats_labels_to_canonical, harmonise
from trace_seg3d.splits import make_splits


def _ball(shape, center, r):
    g = np.ogrid[tuple(slice(0, s) for s in shape)]
    return sum((a - c) ** 2 for a, c in zip(g, center)) <= r**2


def _raw_case(shape=(60, 70, 50)):
    brain = _ball(shape, [s // 2 for s in shape], 18)
    img = np.stack([np.where(brain, 300 + 50 * np.random.rand(*shape), 0) for _ in range(4)]).astype(np.float32)
    lab = np.zeros(shape, np.uint8)
    lab[_ball(shape, [s // 2 for s in shape], 6)] = 2
    lab[_ball(shape, [s // 2 for s in shape], 3)] = 3
    return RawCase("c1", "utsw", img, lab, (1.0, 1.0, 1.0)), brain


def test_preprocess_crops_on_raw_brain_and_keeps_background_zero():
    case, brain = _raw_case()
    arrays, meta = harmonise(case, size=32, resample_mm=1.0, margin=2)
    img, br = arrays["image"].astype(np.float32), arrays["brain"].astype(bool)
    assert img.shape == (4, 32, 32, 32)
    # brain fills the cube (old bug: crop was a no-op and brain was ~half of the box)
    extent = np.ptp(np.argwhere(br), axis=0) + 1
    assert extent.min() >= 26
    assert np.all(img[:, ~br] == 0)
    assert abs(float(img[0][br].mean())) < 0.1
    assert meta["spacing_mm"] == pytest.approx(case_side_mm(brain, 2) / 32, rel=1e-3)
    assert meta["et_voxels"] > 0


def case_side_mm(brain, margin):
    coords = np.argwhere(brain)
    lo = np.maximum(coords.min(0) - margin, 0)
    hi = np.minimum(coords.max(0) + margin + 1, brain.shape)
    return float((hi - lo).max())


def test_brats_label_codes():
    out = brats_labels_to_canonical(np.array([0, 1, 2, 4, 3]))
    assert out.tolist() == [0, 1, 2, 3, 3]
    with pytest.raises(ValueError):
        brats_labels_to_canonical(np.array([5]))


def _tiny_model(**kw):
    torch.manual_seed(0)
    return TraceMedNeXt(base_channels=4, context_sizes=(2,), disease_sizes=(2,), **kw)


def test_transport_with_own_moments_is_identity_and_other_moments_change_output():
    model = _tiny_model().eval()
    x = torch.randn(1, 4, 32, 32, 32)
    feats, stats = model.encode(x)
    same = model.transport(feats, stats, stats)
    for a, b in zip(feats, same):
        assert torch.allclose(a, b, atol=1e-4)
    other = [(mu + 1.0, sigma * 2.0) for mu, sigma in stats]
    base = model.decode(feats, (32, 32, 32))
    moved = model.decode(model.transport(feats, stats, other), (32, 32, 32))
    assert (base - moved).abs().mean() > 1e-3  # the decoder cannot ignore the intervention


def test_forward_outputs_and_losses_are_finite():
    model = _tiny_model()
    x = torch.randn(2, 4, 32, 32, 32)
    y = (torch.rand(2, 3, 32, 32, 32) > 0.9).float()
    _, stats = model.encode(x)
    out = model(x, target_stats=[(m.detach().flip(0), s.detach().flip(0)) for m, s in stats])
    loss = segmentation_loss(out["logits"], y) + stability_loss(out["logits"], out["transported_logits"], 0.03)
    loss.backward()
    assert torch.isfinite(loss)
    assert out["z_d"].shape == out["z_c"].shape


def test_cct_inference_shapes():
    model = _tiny_model().eval()
    x = torch.randn(1, 4, 32, 32, 32)
    _, stats = model.encode(x)
    bank = ContextBank([torch.cat([m + i for i in range(5)]) for m, _ in stats], [torch.cat([s * (1 + i) for i in range(5)]) for _, s in stats], dataset="utsw")
    res = model.cct(x, bank, k=3)
    assert res["transported"].shape[0] == 3
    assert res["instability"].shape == res["factual"].shape
    assert len({tuple(e[0][0].flatten()[:3].tolist()) for e in bank.select(3)}) == 3


def test_region_map_is_probabilistic_or():
    p = torch.tensor([0.5, 0.5, 0.0]).view(1, 3, 1, 1, 1)
    r = subregion_to_region_prob(p).flatten()
    assert r.tolist() == pytest.approx([0.75, 0.5, 0.0])


def test_metrics_empty_conventions_and_mm():
    empty = np.zeros((8, 8, 8), bool)
    cube = empty.copy()
    cube[2:5, 2:5, 2:5] = True
    shifted = np.roll(cube, 2, axis=0)
    assert hd95_mm(empty, empty, 1.0) == 0.0
    assert hd95_mm(cube, empty, 1.0) == HD95_EMPTY_PENALTY_MM
    assert hd95_mm(cube, shifted, 2.0) == pytest.approx(2 * hd95_mm(cube, shifted, 1.0))
    pred = {"WT": cube, "TC": cube, "ET": empty}
    ref = {"WT": cube, "TC": cube, "ET": empty}
    m = case_metrics(pred, ref, 1.0)
    assert m["ET_dice"] == 1.0 and m["ET_ref_empty"] == 1.0


def test_structural_prior_removes_small_et_and_keeps_hierarchy():
    prob = np.zeros((3, 16, 16, 16), np.float32)
    prob[2, 2:6, 2:6, 2:6] = 0.9  # 64-voxel ET blob
    prob[2, 12, 12, 12] = 0.9  # isolated 1-voxel ET island
    masks = structural_prior(threshold_regions(prob, {"WT": 0.5, "TC": 0.5, "ET": 0.5}), 8)
    assert masks["ET"].sum() == 64
    assert np.all(masks["ET"] <= masks["TC"]) and np.all(masks["TC"] <= masks["WT"])


def test_calibration_must_come_from_source_val(tmp_path: Path):
    ckpt = tmp_path / "run" / "best.pt"
    ckpt.parent.mkdir()
    good = {"ckpt": str(ckpt), "source_dataset": "utsw", "split": "val", "thresholds": {}, "min_et_voxels": 0}
    path = tmp_path / "run" / "calib.json"
    path.write_text(json.dumps(good))
    assert load_calibration(path, {"source_dataset": "utsw"}, ckpt)["calibrated"]
    path.write_text(json.dumps({**good, "split": "test"}))
    with pytest.raises(AssertionError):
        load_calibration(path, {"source_dataset": "utsw"}, ckpt)
    path.write_text(json.dumps({**good, "source_dataset": "brats"}))
    with pytest.raises(AssertionError):
        load_calibration(path, {"source_dataset": "utsw"}, ckpt)


def test_splits_disjoint_and_stratified():
    rows = [{"case_id": f"c{i}", "et_present": str(i % 2), "grade": "NA", "label_source": "manual" if i % 5 else "auto_fets"} for i in range(50)]
    s = make_splits(rows, seed=1, exclude_auto_labels=True)
    ids = s["train"] + s["val"] + s["test"]
    assert len(ids) == len(set(ids)) == 50
    assert all(not c.endswith(("0", "5")) for c in s["test"])  # auto labels kept out of test


def test_audit_metrics():
    dice = np.array([0.9, 0.8, 0.2, 0.1])
    perfect = 1 - dice
    assert auroc(perfect, dice < 0.5) == 1.0
    assert aurc(perfect, 1 - dice) <= aurc(np.zeros(4), 1 - dice) + 1e-9


def test_shifts_are_deterministic_and_keep_normalisation():
    import torch

    from trace_seg3d.shifts import SEVERITY, apply_shift, parse_shift

    g = torch.Generator().manual_seed(0)
    image = torch.randn((4, 24, 24, 24), generator=g)
    brain = torch.zeros((24, 24, 24), dtype=torch.bool)
    brain[4:20, 4:20, 4:20] = True
    image = image * brain
    for name in SEVERITY:
        shift = parse_shift(f"{name}:2")
        a = apply_shift(image, brain, "case1", shift)
        b = apply_shift(image, brain, "case1", shift)
        assert torch.equal(a, b)
        assert a.shape == image.shape and float(a[:, ~brain].abs().max()) == 0.0
        inside = a[:, brain]
        assert torch.allclose(inside.mean(1), torch.zeros(4), atol=1e-4) and torch.allclose(inside.std(1), torch.ones(4), atol=1e-2)
        assert not torch.allclose(a, image)
    assert parse_shift("none") is None


def test_bootstrap_audit_reports_cis_and_paired_tests():
    import numpy as np

    from trace_seg3d.audit import bootstrap_audit

    rng = np.random.default_rng(0)
    dice = rng.uniform(0.3, 0.95, 60)
    rows = [{"case_id": str(i), "final_mean_dice": str(d), "good": str(1 - d + rng.normal(0, 0.02)), "entropy_mean": str(rng.uniform())}
            for i, d in enumerate(dice)]
    res = bootstrap_audit([rows], "final_mean_dice", None, 0.2, ["good", "entropy_mean"], refs=("entropy_mean",), n_boot=200)
    good = res["signals"]["good"]
    assert good["auroc"]["value"] > 0.9 and good["auroc"]["ci95"][0] <= good["auroc"]["value"] <= good["auroc"]["ci95"][1]
    assert good["delta_auroc_vs_entropy_mean"]["p"] < 0.05

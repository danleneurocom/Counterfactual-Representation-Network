"""Check the causal assumption behind CCT: "feature moments carry the acquisition context, the
moment-normalised content carries the disease".

For one checkpoint we extract, per case and without labels,
  context descriptor  z_c = [mu, log sigma] of the encoder features at every transport level
                            (exactly what CCT swaps)
  content descriptor  z_d = statistics of the moment-NORMALISED features (what CCT keeps):
                            mean |f_hat| per channel, and mean / second moment of f_hat inside the
                            model's own predicted whole tumour (bottleneck and the level above)
and fit simple linear probes on the SOURCE train split, scored on the test split:
  context targets  site / scanner / field strength, and source-vs-target domain if both are available
  disease targets  grade, log whole-tumour volume
Both descriptors are reduced to the same number of PCA components (fit on train) so the comparison
is not about dimensionality. Expected if the assumption holds: context targets are predicted
better from z_c, disease targets better from z_d.

    python -m trace_seg3d.probe --ckpt runs/isbi/brats/trace_s0/best.pt --data-dir data/processed/brats_128 \
        --splits splits/brats.json [--other-data-dir data/processed/utsw_128 --other-splits splits/utsw.json] --out .../probe
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from trace_seg3d.data import CaseDataset, load_splits
from trace_seg3d.evaluate import load_checkpoint
from trace_seg3d.model import feature_moments

CONTEXT_TARGETS = ("site", "scanner", "field_strength")
DISEASE_TARGETS = ("grade",)


@torch.no_grad()
def descriptors(model, loader, device, amp: bool) -> tuple[list[str], np.ndarray, np.ndarray]:
    ids, zc, zd = [], [], []
    for batch in loader:
        x = batch["image"].to(device)
        with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            features, stats = model.encode(x)
            logits = model.decode(features, tuple(x.shape[-3:]))
        wt = (torch.sigmoid(logits.float()).amax(1, keepdim=True) > 0.5).float()  # model's own WT, no labels
        zc.append(torch.cat([torch.cat([mu, torch.log(s)], 1) for mu, s in stats], 1).float().cpu().numpy()[0])
        parts = []
        for level in (len(features) - 1, len(features) - 2):
            f = features[level].float()
            mu, sigma = feature_moments(f)
            fh = (f - mu[..., None, None, None]) / sigma[..., None, None, None]
            m = torch.nn.functional.interpolate(wt, size=f.shape[-3:], mode="trilinear", align_corners=False)
            w = m / m.sum().clamp_min(1e-6)
            parts += [fh.abs().mean(dim=(2, 3, 4)), (fh * w).sum(dim=(2, 3, 4)), (fh.pow(2) * w).sum(dim=(2, 3, 4))]
        zd.append(torch.cat(parts, 1).cpu().numpy()[0])
        ids.append(batch["case_id"][0])
    return ids, np.stack(zc), np.stack(zd)


def read_index(data_dir: Path) -> dict[str, dict[str, str]]:
    with open(data_dir / "index.csv", newline="") as handle:
        return {r["case_id"]: r for r in csv.DictReader(handle)}


def _clf_score(Xtr, ytr, Xte, yte, n_pca: int) -> float:
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    k = min(n_pca, Xtr.shape[0] - 1, Xtr.shape[1])
    pipe = make_pipeline(StandardScaler(), PCA(k, random_state=0), LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0))
    pipe.fit(Xtr, ytr)
    return float(balanced_accuracy_score(yte, pipe.predict(Xte)))


def _reg_score(Xtr, ytr, Xte, yte, n_pca: int) -> float:
    from sklearn.decomposition import PCA
    from sklearn.linear_model import RidgeCV
    from sklearn.metrics import r2_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    k = min(n_pca, Xtr.shape[0] - 1, Xtr.shape[1])
    pipe = make_pipeline(StandardScaler(), PCA(k, random_state=0), RidgeCV(alphas=np.logspace(-2, 3, 12)))
    pipe.fit(Xtr, ytr)
    return float(r2_score(yte, pipe.predict(Xte)))


def categorical(index, ids_tr, ids_te, field: str, min_count: int):
    ytr = np.array([index[c].get(field, "NA") for c in ids_tr])
    yte = np.array([index[c].get(field, "NA") for c in ids_te])
    values, counts = np.unique(ytr[ytr != "NA"], return_counts=True)
    keep = set(values[counts >= min_count])
    if len(keep) < 2:
        return None
    mtr, mte = np.isin(ytr, list(keep)), np.isin(yte, list(keep))
    if len(set(yte[mte])) < 2:
        return None
    return mtr, ytr, mte, yte, len(keep)


def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True, type=Path)
    p.add_argument("--data-dir", required=True, type=Path)
    p.add_argument("--splits", required=True, type=Path)
    p.add_argument("--other-data-dir", type=Path, help="target dataset for the domain probe (analysis only)")
    p.add_argument("--other-splits", type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--max-train", type=int, default=200)
    p.add_argument("--n-pca", type=int, default=16)
    p.add_argument("--min-class", type=int, default=8)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--amp", action="store_true")
    args = p.parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config, _ = load_checkpoint(args.ckpt, device)
    splits = load_splits(args.splits)
    ids_tr_all, ids_te_all = splits["train"][: args.max_train], splits["test"]

    def extract(data_dir, ids):
        loader = DataLoader(CaseDataset(data_dir, ids), batch_size=1, num_workers=args.workers)
        return descriptors(model, loader, device, args.amp)

    ids_tr, zc_tr, zd_tr = extract(args.data_dir, ids_tr_all)
    ids_te, zc_te, zd_te = extract(args.data_dir, ids_te_all)
    index = read_index(args.data_dir)
    results: dict = {"ckpt": str(args.ckpt), "source": config["source_dataset"], "mode": config["mode"], "n_train": len(ids_tr), "n_test": len(ids_te),
                     "n_pca": args.n_pca, "probes": {}}
    for field in CONTEXT_TARGETS + DISEASE_TARGETS:
        cat = categorical(index, ids_tr, ids_te, field, args.min_class)
        if cat is None:
            continue
        mtr, ytr, mte, yte, k = cat
        results["probes"][field] = {
            "kind": "context" if field in CONTEXT_TARGETS else "disease", "metric": "balanced_acc", "chance": 1.0 / k, "classes": k,
            "from_moments": _clf_score(zc_tr[mtr], ytr[mtr], zc_te[mte], yte[mte], args.n_pca),
            "from_content": _clf_score(zd_tr[mtr], ytr[mtr], zd_te[mte], yte[mte], args.n_pca),
        }
    vol_tr = np.log1p(np.array([float(index[c]["wt_voxels"]) for c in ids_tr]))
    vol_te = np.log1p(np.array([float(index[c]["wt_voxels"]) for c in ids_te]))
    results["probes"]["log_wt_volume"] = {
        "kind": "disease", "metric": "R2", "chance": 0.0,
        "from_moments": _reg_score(zc_tr, vol_tr, zc_te, vol_te, args.n_pca),
        "from_content": _reg_score(zd_tr, vol_tr, zd_te, vol_te, args.n_pca),
    }
    if args.other_data_dir and args.other_splits and (args.other_data_dir / "index.csv").exists():
        osplits = load_splits(args.other_splits)
        o_tr, ozc_tr, ozd_tr = extract(args.other_data_dir, osplits["train"][: args.max_train])
        o_te, ozc_te, ozd_te = extract(args.other_data_dir, osplits["test"])
        y_tr = np.array([0] * len(ids_tr) + [1] * len(o_tr))
        y_te = np.array([0] * len(ids_te) + [1] * len(o_te))
        results["probes"]["domain"] = {
            "kind": "context", "metric": "balanced_acc", "chance": 0.5, "classes": 2,
            "from_moments": _clf_score(np.vstack([zc_tr, ozc_tr]), y_tr, np.vstack([zc_te, ozc_te]), y_te, args.n_pca),
            "from_content": _clf_score(np.vstack([zd_tr, ozd_tr]), y_tr, np.vstack([zd_te, ozd_te]), y_te, args.n_pca),
        }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "probe.json").write_text(json.dumps(results, indent=2))
    lines = [f"Probe {config['source_dataset']}/{config['mode']} (train {len(ids_tr)}, test {len(ids_te)}, PCA {args.n_pca})", "",
             "| target | kind | metric | chance | from moments (z_c) | from content (z_d) |", "|---|---|---|---|---|---|"]
    for name, r in results["probes"].items():
        lines.append(f"| {name} | {r['kind']} | {r['metric']} | {r['chance']:.2f} | {r['from_moments']:.3f} | {r['from_content']:.3f} |")
    (args.out / "probe.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return results


if __name__ == "__main__":
    main()

"""Train MedNeXt baselines and TRACE-Seg3D on one SOURCE dataset.

Modes
-----
baseline  plain MedNeXt (seg loss only)
styleaug  baseline + image-level style intervention (strong DG baseline)
trace     + feature-moment CCT loss + stability hinge + proxy anchoring (+ adversaries)

Ablations for ``trace``: --no-cct-loss, --no-stability, --no-proxies, --no-adversary.

Model selection uses ONLY the source validation split (mean region Dice at threshold 0.5).
After training the source-train context bank is computed with the best weights and stored
inside the checkpoint, so evaluation can never build a bank from the target domain.

    python -m trace_seg3d.train --data-dir data/processed/utsw_128 --splits splits/utsw.json \
        --mode trace --out runs/isbi/utsw/trace_s0 --seed 0 --amp
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from trace_seg3d.data import (
    CONTEXT_FIELDS,
    DISEASE_FIELDS,
    CaseDataset,
    ProxyEncoder,
    load_splits,
    region_volume_target,
    style_augment,
)
from trace_seg3d.losses import (
    multi_ce,
    orthogonality_loss,
    segmentation_loss,
    stability_loss,
    subregion_to_region_prob,
    subregion_to_region_target,
)
from trace_seg3d.model import StatsQueue, TraceMedNeXt, build_bank


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def infer_dataset_name(ds: CaseDataset) -> str:
    names = {m.get("dataset", "NA") for m in ds.metas()[:20]}
    if len(names) != 1:
        raise ValueError(f"mixed datasets in source split: {names}")
    return names.pop()


def _worker_init(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)


@torch.no_grad()
def validate(model: TraceMedNeXt, loader: DataLoader, device: torch.device, amp: bool) -> dict[str, float]:
    model.eval()
    dices = []
    for batch in loader:
        x = batch["image"].to(device)
        y = batch["target"].to(device)
        with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            logits = model(x)["logits"]
        pred = subregion_to_region_prob(torch.sigmoid(logits.float())) >= 0.5
        ref = subregion_to_region_target(y) > 0.5
        dims = (2, 3, 4)
        inter = (pred & ref).sum(dims).float()
        denom = pred.sum(dims).float() + ref.sum(dims).float()
        dice = torch.where(denom > 0, 2 * inter / denom.clamp_min(1), torch.ones_like(denom))
        dices.append(dice.cpu())
    d = torch.cat(dices).mean(0)
    return {"val/WT": float(d[0]), "val/TC": float(d[1]), "val/ET": float(d[2]), "val/mean": float(d.mean())}


def _atomic_save(obj: Any, path: Path) -> None:
    """Write to a temp file and rename, so a session killed mid-write never leaves a corrupt checkpoint."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", required=True, type=Path)
    p.add_argument("--splits", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--mode", choices=["baseline", "styleaug", "trace"], default="trace")
    p.add_argument("--style-aug", action="store_true", help="also apply image-level style intervention in trace mode")
    p.add_argument("--no-cct-loss", action="store_true")
    p.add_argument("--no-stability", action="store_true")
    p.add_argument("--no-proxies", action="store_true")
    p.add_argument("--no-adversary", action="store_true")
    p.add_argument("--model-id", default="S", choices=["S", "B", "M", "L"])
    p.add_argument("--kernel-size", type=int, default=3)
    p.add_argument("--base-channels", type=int)
    p.add_argument("--patch-size", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--cct-start-epoch", type=int, default=10, help="CCT losses switch on after this epoch")
    p.add_argument("--lambda-region", type=float, default=1.0)
    p.add_argument("--lambda-cct", type=float, default=0.5)
    p.add_argument("--lambda-stab", type=float, default=0.1)
    p.add_argument("--stab-margin", type=float, default=0.03)
    p.add_argument("--lambda-proxy", type=float, default=0.05)
    p.add_argument("--lambda-adv", type=float, default=0.02)
    p.add_argument("--lambda-orth", type=float, default=0.01)
    p.add_argument("--queue-size", type=int, default=64)
    p.add_argument("--val-every", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="auto")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--grad-clip", type=float, default=12.0)
    p.add_argument("--max-train-cases", type=int, help="smoke tests only")
    p.add_argument("--max-val-cases", type=int, help="smoke tests only")
    p.add_argument("--cache-in-memory", action="store_true")
    p.add_argument("--no-resume", action="store_true", help="ignore OUT/resume.pt and start from scratch")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    seed_everything(args.seed)
    device = resolve_device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)
    splits = load_splits(args.splits)
    train_ids = splits["train"][: args.max_train_cases]
    val_ids = splits["val"][: args.max_val_cases]

    probe = CaseDataset(args.data_dir, train_ids)
    dataset_name = infer_dataset_name(probe)
    use_trace = args.mode == "trace"
    use_proxies = use_trace and not args.no_proxies
    metas = probe.metas()
    ctx_enc = ProxyEncoder(CONTEXT_FIELDS.get(dataset_name, ())).fit(metas) if use_proxies else None
    dis_enc = ProxyEncoder(DISEASE_FIELDS.get(dataset_name, ())).fit(metas) if use_proxies else None

    train_ds = CaseDataset(args.data_dir, train_ids, train=True, patch_size=args.patch_size, context_encoder=ctx_enc, disease_encoder=dis_enc, cache_in_memory=args.cache_in_memory)
    val_ds = CaseDataset(args.data_dir, val_ids, cache_in_memory=args.cache_in_memory)
    bank_ds = CaseDataset(args.data_dir, train_ids)
    g = torch.Generator()
    g.manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=len(train_ds) > args.batch_size, num_workers=args.workers, worker_init_fn=_worker_init, generator=g, pin_memory=device.type == "cuda", persistent_workers=args.workers > 0)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=args.workers)
    bank_loader = DataLoader(bank_ds, batch_size=1, num_workers=args.workers)

    model = TraceMedNeXt(
        model_id=args.model_id,
        kernel_size=args.kernel_size,
        base_channels=args.base_channels,
        context_sizes=ctx_enc.sizes if ctx_enc else (),
        disease_sizes=dis_enc.sizes if dis_enc else (),
        proxies=use_proxies,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = max(len(train_loader), 1)
    total_steps = args.epochs * steps_per_epoch
    warmup = args.warmup_epochs * steps_per_epoch

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(total_steps - warmup, 1)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    use_fp16 = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)
    queue = StatsQueue(args.queue_size)

    config = {
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "source_dataset": dataset_name,
        "mode": args.mode,
        "model": model.config,
        "context_encoder": ctx_enc.state() if ctx_enc else None,
        "disease_encoder": dis_enc.state() if dis_enc else None,
        "train_ids": train_ids,
        "val_ids": val_ids,
    }
    (args.out / "config.json").write_text(json.dumps(config, indent=2))
    best, history, start_epoch = -1.0, [], 1
    resume_path = args.out / "resume.pt"
    if resume_path.exists() and not args.no_resume:
        ck = torch.load(resume_path, map_location=device, weights_only=False)
        # paths and loader settings may change between sessions (Colab -> Kaggle, new state dir) without affecting training
        volatile = {"workers", "no_resume", "device", "data_dir", "splits", "out", "cache_in_memory"}
        same = {k: v for k, v in ck["config"]["args"].items() if k not in volatile} == {k: v for k, v in config["args"].items() if k not in volatile}
        if same:
            model.load_state_dict(ck["model"])
            optimizer.load_state_dict(ck["optimizer"])
            scheduler.load_state_dict(ck["scheduler"])
            scaler.load_state_dict(ck["scaler"])
            queue.mus, queue.sigmas = ck["queue"]
            best, history, start_epoch = ck["best"], ck["history"], ck["epoch"] + 1
            torch.set_rng_state(ck["rng_cpu"])
            if device.type == "cuda" and ck.get("rng_cuda") is not None:
                torch.cuda.set_rng_state(ck["rng_cuda"])
            print(json.dumps({"resumed_from_epoch": ck["epoch"], "best_val_mean_dice": best}))
        else:
            print("!! resume.pt ignored: training arguments differ from the interrupted run")
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        t0 = time.time()
        sums: dict[str, float] = {}
        n = 0
        cct_on = use_trace and epoch > args.cct_start_epoch
        for batch in train_loader:
            x = batch["image"].to(device, non_blocking=True)
            y = batch["target"].to(device, non_blocking=True)
            if args.mode == "styleaug" or (use_trace and args.style_aug):
                x = style_augment(x, batch["brain"].to(device))
            target_stats = queue.sample(x.shape[0]) if (cct_on and len(queue) >= 2 and not args.no_cct_loss) else None
            with torch.autocast(device_type=device.type, enabled=use_fp16):
                out = model(x, target_stats=target_stats)
            terms: dict[str, torch.Tensor] = {"seg": segmentation_loss(out["logits"], y, args.lambda_region)}
            if target_stats is not None:
                terms["cct"] = args.lambda_cct * segmentation_loss(out["transported_logits"], y, args.lambda_region)
                if not args.no_stability:
                    terms["stab"] = args.lambda_stab * stability_loss(out["logits"], out["transported_logits"], args.stab_margin)
            if use_proxies:
                vol = region_volume_target(y)
                terms["proxy_vol"] = args.lambda_proxy * torch.nn.functional.smooth_l1_loss(out["volume"].float(), vol)
                ce = multi_ce(out["context_logits"], batch["context_label"].to(device))
                if ce is not None:
                    terms["proxy_ctx"] = args.lambda_proxy * ce
                ce = multi_ce(out["disease_logits"], batch["disease_label"].to(device))
                if ce is not None:
                    terms["proxy_dis"] = args.lambda_proxy * ce
                terms["orth"] = args.lambda_orth * orthogonality_loss(out["z_d"].float(), out["z_c"].float())
                if not args.no_adversary:
                    terms["adv_vol"] = args.lambda_adv * torch.nn.functional.smooth_l1_loss(out["adv_volume"].float(), vol)
                    ce = multi_ce(out["adv_context_logits"], batch["context_label"].to(device))
                    if ce is not None:
                        terms["adv_ctx"] = args.lambda_adv * ce
            loss = sum(terms.values())
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch {epoch}: { {k: float(v) for k, v in terms.items()} }")
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            if use_trace:
                queue.push(out["stats"])  # type: ignore[arg-type]
            for k, v in terms.items():
                sums[k] = sums.get(k, 0.0) + float(v.detach())
            sums["total"] = sums.get("total", 0.0) + float(loss.detach())
            n += 1
        log: dict[str, Any] = {"epoch": epoch, "lr": scheduler.get_last_lr()[0], "time_s": round(time.time() - t0, 1)}
        if device.type == "cuda":
            log["gpu_peak_gb"] = round(torch.cuda.max_memory_allocated(device) / 1024**3, 2)
        log.update({f"train/{k}": v / max(n, 1) for k, v in sums.items()})
        if epoch % args.val_every == 0 or epoch == args.epochs:
            log.update(validate(model, val_loader, device, args.amp))
            state = {"model": model.state_dict(), "config": config, "epoch": epoch, "val": log}
            _atomic_save(state, args.out / "last.pt")
            if log["val/mean"] > best:
                best = log["val/mean"]
                _atomic_save(state, args.out / "best.pt")
        history.append(log)
        print(json.dumps(log))
        (args.out / "history.json").write_text(json.dumps(history, indent=1))
        # full training state every epoch -> a disconnected Colab session continues where it stopped
        _atomic_save({
            "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(), "queue": (queue.mus, queue.sigmas), "best": best, "history": history,
            "epoch": epoch, "config": config, "rng_cpu": torch.get_rng_state(),
            "rng_cuda": torch.cuda.get_rng_state() if device.type == "cuda" else None,
        }, resume_path)

    # source-train context bank with the selected weights, stored in the checkpoint
    for name in ("best.pt", "last.pt"):
        path = args.out / name
        state = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        bank = build_bank(model, bank_loader, device, dataset_name, args.amp)
        state["bank"] = bank.state()
        _atomic_save(state, path)
    summary = {"best_val_mean_dice": best, "out": str(args.out), "source_dataset": dataset_name, "mode": args.mode}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    resume_path.unlink(missing_ok=True)
    print(json.dumps(summary))
    return summary


if __name__ == "__main__":
    main()

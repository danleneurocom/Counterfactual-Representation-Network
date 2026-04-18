from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_RUNS = {
    "causal_region": Path("runs/brats_segonly_unet_causal_regions/best_val_metrics.json"),
    "full_contrastive": Path("runs/brats_segonly_unet_causal_contrastive_continue/best_val_metrics.json"),
    "no_contrastive": Path("runs/ablations/no_contrastive/best_val_metrics.json"),
    "no_region_adjustment": Path("runs/ablations/no_region_adjustment/best_val_metrics.json"),
    "no_region_context_stability": Path("runs/ablations/no_region_context_stability/best_val_metrics.json"),
    "no_region_disease_swap": Path("runs/ablations/no_region_disease_swap/best_val_metrics.json"),
}


METRIC_KEYS = {
    "epoch": "eval/epoch",
    "threshold": "sweep/best_threshold",
    "mean_dice": "sweep_best_volume/brats/mean_dice",
    "mean_hd95": "sweep_best_volume/brats/mean_hd95",
    "WT": "sweep_best_volume/brats/WT/dice",
    "TC": "sweep_best_volume/brats/TC/dice",
    "ET": "sweep_best_volume/brats/ET/dice",
}


def _load_metrics(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_best_epoch_metrics(run_dir: Path) -> dict[str, Any] | None:
    epoch_files = sorted(run_dir.glob("epoch_*.json"))
    best: dict[str, Any] | None = None
    best_dice: float | None = None
    best_hd95: float | None = None
    for path in epoch_files:
        data = json.loads(path.read_text(encoding="utf-8"))
        val = data.get("val", {})
        dice = val.get("sweep_best_volume/brats/mean_dice")
        hd95 = val.get("sweep_best_volume/brats/mean_hd95")
        if not isinstance(dice, (float, int)):
            continue
        if best is None or float(dice) > float(best_dice) + 1e-8 or (
            abs(float(dice) - float(best_dice)) <= 1e-8 and isinstance(hd95, (float, int)) and isinstance(best_hd95, (float, int)) and float(hd95) < float(best_hd95) - 1e-8
        ):
            best = val
            best_dice = float(dice)
            best_hd95 = float(hd95) if isinstance(hd95, (float, int)) else None
            best["eval/epoch"] = int(data.get("epoch", -1))
    return best


def _resolve_run_metrics(path: Path) -> dict[str, Any] | None:
    metrics = _load_metrics(path)
    if metrics is not None:
        return metrics
    run_dir = path.parent
    if run_dir.exists():
        return _load_best_epoch_metrics(run_dir)
    return None


def _format_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _row(name: str, metrics: dict[str, Any] | None, reference: dict[str, Any] | None) -> list[str]:
    if metrics is None:
        return [name, "missing", "-", "-", "-", "-", "-", "-", "-"]
    mean_dice = metrics.get(METRIC_KEYS["mean_dice"])
    reference_dice = reference.get(METRIC_KEYS["mean_dice"]) if reference else None
    delta = mean_dice - reference_dice if isinstance(mean_dice, float) and isinstance(reference_dice, float) else None
    return [
        name,
        _format_value(metrics.get(METRIC_KEYS["epoch"])),
        _format_value(metrics.get(METRIC_KEYS["threshold"])),
        _format_value(mean_dice),
        _format_value(delta),
        _format_value(metrics.get(METRIC_KEYS["mean_hd95"])),
        _format_value(metrics.get(METRIC_KEYS["WT"])),
        _format_value(metrics.get(METRIC_KEYS["TC"])),
        _format_value(metrics.get(METRIC_KEYS["ET"])),
    ]


def _markdown_table(rows: list[list[str]]) -> str:
    headers = ["run", "epoch", "thr", "mean Dice", "Delta Dice", "mean HD95", "WT", "TC", "ET"]
    separator = ["---", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(separator) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def summarize(output: Path | None = None) -> str:
    loaded = {name: _resolve_run_metrics(path) for name, path in DEFAULT_RUNS.items()}
    reference = loaded.get("full_contrastive")
    rows = [_row(name, metrics, reference) for name, metrics in loaded.items()]
    table = _markdown_table(rows)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(table + "\n", encoding="utf-8")
    return table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize CRN ablation metrics into a markdown table.")
    parser.add_argument("--output", type=Path, help="Optional markdown output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(summarize(args.output))


if __name__ == "__main__":
    main()

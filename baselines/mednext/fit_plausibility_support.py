from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from baselines.mednext.calibration import fit_plausibility_support_thresholds


def _load_per_case(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        payload = json.load(handle)
    records = payload.get("per_case")
    if not isinstance(records, list):
        raise ValueError(f"{path} does not contain a per_case list.")
    return [dict(record) for record in records]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit H76 plausibility gates from per-case support records.")
    parser.add_argument("--support-json", action="append", required=True, help="JSON file with per_case support records.")
    parser.add_argument(
        "--validation-json",
        action="append",
        help="JSON file with per_case validation records used for the WT support ceiling.",
    )
    parser.add_argument("--prefix", default="registered_tta_plausibility_region_calibrated")
    parser.add_argument("--low-stability-threshold", type=float, default=0.90)
    parser.add_argument("--low-stability-wt-margin", type=float, default=0.95)
    parser.add_argument("--tc-collapse-tc-margin", type=float, default=0.50)
    parser.add_argument("--tc-collapse-wt-ratio-min", type=float, default=0.0)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    support_paths = [Path(path) for path in args.support_json]
    validation_paths = [Path(path) for path in (args.validation_json or [])]
    support_records = [record for path in support_paths for record in _load_per_case(path)]
    validation_records = [record for path in validation_paths for record in _load_per_case(path)] if validation_paths else None
    thresholds = fit_plausibility_support_thresholds(
        support_records,
        validation_records=validation_records,
        prefix=args.prefix,
        low_stability_threshold=args.low_stability_threshold,
        low_stability_wt_margin=args.low_stability_wt_margin,
        tc_collapse_tc_margin=args.tc_collapse_tc_margin,
        tc_collapse_wt_ratio_min=args.tc_collapse_wt_ratio_min,
    )
    output = {
        "support_json": [str(path) for path in support_paths],
        "validation_json": [str(path) for path in validation_paths],
        "prefix": str(args.prefix),
        **thresholds,
        "cli": {
            "--plausibility-low-stability-wt-ratio-threshold": thresholds[
                "plausibility/low_stability_wt_ratio_threshold"
            ],
            "--plausibility-low-stability-threshold": thresholds["plausibility/low_stability_threshold"],
            "--plausibility-tc-collapse-wt-ratio-min": thresholds["plausibility/tc_collapse_wt_ratio_min"],
            "--plausibility-tc-collapse-wt-ratio-max": thresholds["plausibility/tc_collapse_wt_ratio_max"],
            "--plausibility-tc-collapse-tc-ratio-threshold": thresholds[
                "plausibility/tc_collapse_tc_ratio_threshold"
            ],
        },
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        json.dump(output, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(output)


if __name__ == "__main__":
    main()

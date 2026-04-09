from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from crn.train import build_model
from crn.utils import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Instantiate CRN from a config.")
    parser.add_argument("config")
    args = parser.parse_args()
    config = load_yaml(args.config)
    model = build_model(config)
    print(model.__class__.__name__)
    print(f"head_uses_context={model.head_uses_context}")
    print(f"latent_dim={model.latent_dim}")


if __name__ == "__main__":
    main()


from __future__ import annotations

import os
from pathlib import Path
from typing import Any

try:
    import wandb
except ImportError:  # pragma: no cover - optional dependency.
    wandb = None


_REPO_ROOT = Path(__file__).resolve().parents[2]
_CODE_EXTENSIONS = {".py", ".md", ".toml", ".yaml", ".yml", ".json"}
_EXCLUDED_PARTS = {".git", ".pytest_cache", ".mypy_cache", ".ruff_cache", "__pycache__", ".venv", "runs"}


def _to_serializable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(item) for item in value]
    return value


def wandb_is_available() -> bool:
    return wandb is not None and os.environ.get("WANDB_DISABLED", "").lower() != "true"


def should_enable_wandb(config: dict[str, Any]) -> bool:
    if not wandb_is_available():
        return False
    wandb_config = config.get("wandb") or {}
    if "enabled" in wandb_config:
        return bool(wandb_config.get("enabled"))
    return bool(os.environ.get("WANDB_API_KEY") or os.environ.get("WANDB_MODE"))


def init_wandb_run(
    *,
    config: dict[str, Any],
    output_dir: str | Path,
    run_name: str,
    job_type: str,
    project: str | None = None,
) -> Any | None:
    if not should_enable_wandb(config):
        return None

    wandb_config = config.get("wandb") or {}
    run = wandb.init(
        project=wandb_config.get("project", project or "counterfactual-representation-network"),
        entity=wandb_config.get("entity"),
        name=wandb_config.get("name", run_name),
        job_type=wandb_config.get("job_type", job_type),
        dir=str(output_dir),
        config=_to_serializable(config),
        settings=wandb.Settings(code_dir=str(_REPO_ROOT)),
        save_code=True,
        reinit=True,
    )
    log_code_snapshot(run)
    return run


def log_code_snapshot(run: Any) -> None:
    if run is None:
        return
    code_artifact = wandb.Artifact("source-code", type="source")
    for path in sorted(_REPO_ROOT.rglob("*")):
        if not path.is_file():
            continue
        relative_path = path.relative_to(_REPO_ROOT)
        if any(part in _EXCLUDED_PARTS for part in relative_path.parts):
            continue
        if path.name not in {"pyproject.toml", "README.md"} and path.suffix not in _CODE_EXTENSIONS:
            continue
        code_artifact.add_file(str(path), name=str(relative_path))
    run.log_artifact(code_artifact)


def log_path_artifact(
    run: Any,
    path: str | Path,
    *,
    artifact_name: str,
    artifact_type: str,
    aliases: list[str] | None = None,
) -> None:
    if run is None:
        return
    resolved_path = Path(path)
    if not resolved_path.exists():
        return
    artifact = wandb.Artifact(artifact_name, type=artifact_type)
    if resolved_path.is_dir():
        artifact.add_dir(str(resolved_path))
    else:
        artifact.add_file(str(resolved_path), name=resolved_path.name)
    run.log_artifact(artifact, aliases=aliases or [])


def log_metrics(run: Any, metrics: dict[str, Any], *, step: int | None = None) -> None:
    if run is None:
        return
    run.log(_to_serializable(metrics), step=step)


def finish_wandb_run(run: Any) -> None:
    if run is not None:
        run.finish()
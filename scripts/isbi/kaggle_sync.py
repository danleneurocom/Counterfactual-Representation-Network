"""Carry the experiment state between Kaggle versions without any manual download/upload.

Heavy state lives in two PRIVATE Kaggle datasets owned by you (created automatically on first push):
  <user>/trace-isbi-data : processed/ + splits/        (pushed once, again only if a dataset is added)
  <user>/trace-isbi-runs : runs/ results/ logs/ ...    (pushed at the end of every version)
Light results (tables, audit, logs, histories, per-case CSVs; no checkpoints) can also be pushed to a
GitHub branch so they can be read without opening Kaggle.

    python scripts/isbi/kaggle_sync.py pull --state /kaggle/working/trace_state
    python scripts/isbi/kaggle_sync.py push --state /kaggle/working/trace_state --note "version 3"
    python scripts/isbi/kaggle_sync.py push-git --state ... --repo owner/name --branch isbi-results

Credentials come from the environment: KAGGLE_USERNAME + KAGGLE_API_TOKEN (new KGAT_ token, kaggle>=1.8) or KAGGLE_KEY (legacy), and GITHUB_TOKEN.
For tests, TRACE_SYNC_LOCAL=<dir> replaces the Kaggle API by a local folder.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
import time
from pathlib import Path

DATA_PARTS = ("processed", "splits")
RUNS_PARTS = ("runs", "results", "logs", "STATUS.md", "state_meta.json", "check_brats.png", "check_utsw.png")
LIGHT_SUFFIXES = (".json", ".csv", ".md", ".tex", ".png", ".log", ".txt")


def log(msg: str) -> None:
    print(f"[sync {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------------------------------ backends
class KaggleBackend:
    def __init__(self) -> None:
        from kaggle.api.kaggle_api_extended import KaggleApi  # needs KAGGLE_USERNAME / KAGGLE_KEY

        self.api = KaggleApi()
        self.api.authenticate()
        self.user = os.environ.get("KAGGLE_USERNAME") or self.api.get_config_value("username")

    def ref(self, slug: str) -> str:
        return f"{self.user}/{slug}"

    def exists(self, slug: str) -> bool:
        try:
            self.api.dataset_list_files(self.ref(slug))
            return True
        except Exception:
            return False

    def download(self, slug: str, dest: Path) -> bool:
        if not self.exists(slug):
            return False
        self.api.dataset_download_files(self.ref(slug), path=str(dest), unzip=True, quiet=False)
        return True

    def upload(self, slug: str, folder: Path, note: str) -> None:
        meta = {"title": slug, "id": self.ref(slug), "licenses": [{"name": "CC0-1.0"}]}
        (folder / "dataset-metadata.json").write_text(json.dumps(meta))
        if self.exists(slug):
            self.api.dataset_create_version(str(folder), version_notes=note, quiet=False, dir_mode="skip")
        else:
            self.api.dataset_create_new(str(folder), public=False, quiet=False, dir_mode="skip")


class LocalBackend:
    """Test double: a folder per dataset slug."""

    def __init__(self, root: str) -> None:
        self.root = Path(root)

    def exists(self, slug: str) -> bool:
        return (self.root / slug).exists()

    def download(self, slug: str, dest: Path) -> bool:
        if not self.exists(slug):
            return False
        shutil.copytree(self.root / slug, dest, dirs_exist_ok=True)
        return True

    def upload(self, slug: str, folder: Path, note: str) -> None:
        shutil.rmtree(self.root / slug, ignore_errors=True)
        shutil.copytree(folder, self.root / slug)


def backend():
    return LocalBackend(os.environ["TRACE_SYNC_LOCAL"]) if os.environ.get("TRACE_SYNC_LOCAL") else KaggleBackend()


# ---------------------------------------------------------------------------------- pull / push
def _tar(state: Path, parts: tuple[str, ...], out: Path, skip_suffixes: tuple[str, ...] = (".tmp",)) -> list[str]:
    present = [p for p in parts if (state / p).exists()]
    with tarfile.open(out, "w") as tar:  # no compression: checkpoints and float16 volumes barely compress
        for p in present:
            tar.add(state / p, arcname=p, filter=lambda ti: None if ti.name.endswith(skip_suffixes) else ti)
    return present


def _manifest(state: Path) -> dict:
    proc = state / "processed"
    return {"datasets": sorted(d.name for d in proc.iterdir() if (d / "index.csv").exists()) if proc.exists() else []}


def pull(args) -> None:
    be = backend()
    args.state.mkdir(parents=True, exist_ok=True)
    for slug in (args.data_slug, args.runs_slug):
        with tempfile.TemporaryDirectory() as tmp:
            if not be.download(slug, Path(tmp)):
                log(f"{slug}: not found (first version?) -> nothing to pull")
                continue
            for tf in sorted([*Path(tmp).rglob("*.tar"), *Path(tmp).rglob("*.bin")]):
                with tarfile.open(tf) as tar:
                    try:
                        tar.extractall(args.state, filter="data")
                    except TypeError:  # Python without extraction filters
                        tar.extractall(args.state)
                log(f"{slug}: restored {tf.name} ({tf.stat().st_size / 1e9:.2f} GB)")
            # in case Kaggle unpacked the archives on its side: copy the expected folders/files directly
            for part in DATA_PARTS + RUNS_PARTS:
                for hit in sorted(Path(tmp).rglob(part), key=lambda q: len(q.parts))[:1]:
                    if hit.is_dir():
                        shutil.copytree(hit, args.state / part, dirs_exist_ok=True)
                    elif hit.is_file():
                        shutil.copy2(hit, args.state / part)
            if not [*Path(tmp).rglob("*.tar"), *Path(tmp).rglob("*.bin")]:
                log(f"{slug}: downloaded but no archive inside ({[q.name for q in Path(tmp).rglob('*')][:10]}) -> nothing restored")
            m = Path(tmp) / "manifest.json"
            if m.exists():
                shutil.copy2(m, args.state / f".{slug}.manifest.json")


def push(args) -> None:
    be = backend()
    state = args.state
    # data: only when the set of preprocessed datasets changed since the last push
    current = _manifest(state)
    old_path = state / f".{args.data_slug}.manifest.json"
    old = json.loads(old_path.read_text()) if old_path.exists() else None
    if current["datasets"] and current != old:
        with tempfile.TemporaryDirectory() as tmp:
            parts = _tar(state, DATA_PARTS, Path(tmp) / "data.bin")  # .bin: Kaggle must not unpack it
            (Path(tmp) / "manifest.json").write_text(json.dumps(current))
            log(f"uploading {args.data_slug} ({parts}, {(Path(tmp) / 'data.bin').stat().st_size / 1e9:.2f} GB) ...")
            be.upload(args.data_slug, Path(tmp), args.note)
            old_path.write_text(json.dumps(current))
    else:
        log(f"{args.data_slug}: unchanged ({current['datasets']}) -> not re-uploaded")
    if not (state / "runs").exists():
        log(f"{args.runs_slug}: no runs yet -> nothing to upload")
        return
    with tempfile.TemporaryDirectory() as tmp:
        parts = _tar(state, RUNS_PARTS, Path(tmp) / "runs.bin")
        size = (Path(tmp) / "runs.bin").stat().st_size / 1e9
        log(f"uploading {args.runs_slug} ({parts}, {size:.2f} GB) ...")
        be.upload(args.runs_slug, Path(tmp), args.note)
    log("push done")


def push_git(args) -> None:
    token = os.environ.get("GITHUB_TOKEN", "")
    url = f"https://x-access-token:{token}@github.com/{args.repo}.git" if token else f"https://github.com/{args.repo}.git"
    if args.remote_url:  # tests
        url = args.remote_url

    def git(*cmd: str, cwd: Path, check: bool = True) -> subprocess.CompletedProcess:
        env = dict(os.environ, GIT_LFS_SKIP_SMUDGE="1", GIT_TERMINAL_PROMPT="0")
        r = subprocess.run(["git", *cmd], cwd=cwd, env=env, capture_output=True, text=True)
        if check and r.returncode != 0:
            msg = (r.stderr + r.stdout).replace(token, "***") if token else r.stderr + r.stdout
            raise RuntimeError(f"git {cmd[0]} failed: {msg[-1500:]}")
        return r

    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp) / "repo"
        r = git("clone", "--depth", "1", "--branch", args.branch, url, str(repo), cwd=Path(tmp), check=False)
        if r.returncode != 0:  # branch does not exist yet -> new orphan branch
            repo.mkdir()
            git("init", "-q", cwd=repo)
            git("checkout", "-q", "--orphan", args.branch, cwd=repo)
            git("remote", "add", "origin", url, cwd=repo)
        dest = repo / args.subdir
        shutil.rmtree(dest, ignore_errors=True)
        n = 0
        for f in args.state.rglob("*"):
            rel = f.relative_to(args.state)
            if f.is_file() and rel.parts[0] != "processed" and f.suffix in LIGHT_SUFFIXES and f.stat().st_size < 20e6:
                out = dest / rel
                out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, out)
                n += 1
        git("add", "-A", cwd=repo)
        if not git("status", "--porcelain", cwd=repo).stdout.strip():
            log("git: nothing new to push")
            return
        git("-c", "user.name=trace-kaggle", "-c", "user.email=trace-kaggle@users.noreply.github.com",
            "commit", "-q", "-m", args.note or "update results", cwd=repo)
        git("push", "-q", "origin", f"HEAD:{args.branch}", cwd=repo)
        log(f"git: pushed {n} light files to {args.repo}@{args.branch}/{args.subdir}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("pull", "push", "push-git"):
        s = sub.add_parser(name)
        s.add_argument("--state", type=Path, default=Path("/kaggle/working/trace_state"))
        s.add_argument("--data-slug", default="trace-isbi-data")
        s.add_argument("--runs-slug", default="trace-isbi-runs")
        s.add_argument("--note", default="")
        if name == "push-git":
            s.add_argument("--repo", required=True, help="owner/name")
            s.add_argument("--branch", default="isbi-results")
            s.add_argument("--subdir", default="trace_results")
            s.add_argument("--remote-url", help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    {"pull": pull, "push": push, "push-git": push_git}[args.cmd](args)


if __name__ == "__main__":
    main()

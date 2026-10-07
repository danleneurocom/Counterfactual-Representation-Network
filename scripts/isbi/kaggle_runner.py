"""Run the whole ISBI matrix on Kaggle (2x T4), across as many 12-hour sessions as it takes.

Each Kaggle version:
  1. merges the state saved by earlier versions (any attached input that contains processed/, splits/ or runs/,
     e.g. the output of the previous version of this notebook, or the Colab Drive folder ``trace_isbi``)
     into ``--state`` (= /kaggle/working/trace_state, which Kaggle saves as the version output);
  2. prepares missing datasets: BraTS 2020 NIfTI and UTSW (raw or already preprocessed ``utsw_128``),
     runs check_data (stops a dataset whose modality check fails) and creates the fixed splits;
  3. schedules jobs on all GPUs, one job per GPU:
       train+calib (source, method, seed)  ->  eval (source, method) once all its seeds are trained;
     training resumes from ``resume.pt``, so a run cut by the time limit continues in the next version;
  4. stops everything before ``--hours`` so Kaggle can save the output, then writes audit tables,
     result tables and ``STATUS.md``.

Local dry run (CPU, synthetic data):
  python scripts/isbi/kaggle_runner.py --input-root <dir> --state <dir> --hours 0.5 --epochs 2 \
      --extra "--base-channels 8 --patch-size 32" --gpus cpu
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

DATASETS = ("brats", "utsw")
ABLATION_FLAGS = {"trace_nocct", "trace_nostab", "trace_noproxy"}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ----------------------------------------------------------------------------------------- state
def find_state_inputs(root: Path) -> list[Path]:
    """Dirs (depth <= 4) that look like a saved state: contain processed/, splits/ or runs/."""

    found = []
    if not root.exists():
        return found
    for dirpath, dirnames, _ in os.walk(root, followlinks=True):
        depth = len(Path(dirpath).relative_to(root).parts)
        if depth > 4:
            dirnames[:] = []
            continue
        if {"processed", "splits", "runs"} & set(dirnames):
            found.append(Path(dirpath))
            dirnames[:] = []
    return found


def state_version(path: Path) -> int:
    try:
        return int(json.loads((path / "state_meta.json").read_text())["version"])
    except Exception:
        return 0


def merge_tree(src: Path, dst: Path) -> int:
    """Copy files from src into dst (skips files of identical size). Callers merge oldest first, so newest wins."""

    n = 0
    for f in src.rglob("*"):
        if f.is_dir() or f.name.endswith(".tmp"):
            continue
        out = dst / f.relative_to(src)
        if out.exists() and out.stat().st_size == f.stat().st_size:
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, out)
        n += 1
    return n


def restore_state(input_root: Path, state: Path) -> int:
    inputs = sorted(find_state_inputs(input_root), key=state_version)
    state.mkdir(parents=True, exist_ok=True)
    for src in inputs:  # oldest first, the newest version wins
        if src.resolve() == state.resolve():
            continue
        n = merge_tree(src, state)
        log(f"restored {n} files from {src} (state version {state_version(src)})")
    return max([state_version(s) for s in inputs] + [state_version(state)])


# ------------------------------------------------------------------------------------- datasets
def _subdirs(root: Path, depth: int = 5):
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        if len(Path(dirpath).relative_to(root).parts) >= depth:
            dirnames[:] = []
        yield Path(dirpath), dirnames, filenames


def find_brats_raw(input_root: Path) -> Path | None:
    for d, dirnames, _ in _subdirs(input_root):
        if d.name == "MICCAI_BraTS2020_TrainingData" and any(n.startswith("BraTS20_Training_") for n in dirnames):
            return d
    return None


def find_utsw_processed(input_root: Path, state: Path) -> Path | None:
    for d, _, filenames in _subdirs(input_root):
        if "index.csv" in filenames and state not in d.parents:
            with open(d / "index.csv", newline="") as h:
                row = next(csv.DictReader(h), None)
            if row and row.get("dataset") == "utsw":
                return d
    return None


def find_utsw_raw(input_root: Path) -> tuple[Path | None, Path | None]:
    root = meta = None
    for d, dirnames, filenames in _subdirs(input_root, depth=6):
        if root is None and sum(1 for n in dirnames if n.startswith("BT") and n[2:].isdigit()) >= 10:
            root = d
        for f in filenames:
            if meta is None and f.endswith(".tsv") and "utsw" in f.lower():
                meta = d / f
    return root, meta


def run(cmd: list[str], env: dict, log_path: Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(r.stdout + "\n--- stderr ---\n" + r.stderr)
    if check and r.returncode != 0:
        raise RuntimeError(f"command failed ({r.returncode}): {' '.join(cmd)}\n{r.stderr[-3000:]}")
    return r


def prepare_data(args, env: dict) -> list[str]:
    state = args.state
    proc = state / "processed"
    cpu = str(max(1, os.cpu_count() or 2))
    py = [args.python, "-m"]
    # BraTS
    if not (proc / "brats_128" / "index.csv").exists():
        raw = find_brats_raw(args.input_root)
        if raw:
            log(f"preprocessing BraTS from {raw} (~20-40 min) ...")
            run(py + ["trace_seg3d.preprocess", "brats-nifti", "--root", str(raw), "--out", str(proc / "brats_128"), "--workers", cpu, "--size", str(args.size)], env, state / "logs" / "preprocess_brats.log")
        else:
            log("!! BraTS not found: attach the Kaggle dataset 'awsaf49/brats20-dataset-training-validation'")
    # UTSW
    if not (proc / "utsw_128" / "index.csv").exists():
        pre = find_utsw_processed(args.input_root, state)
        root, meta = find_utsw_raw(args.input_root)
        if pre:
            log(f"using preprocessed UTSW from {pre}")
            merge_tree(pre, proc / "utsw_128")
        elif root:
            log(f"preprocessing UTSW from {root} (metadata: {meta}) ...")
            cmd = py + ["trace_seg3d.preprocess", "utsw", "--root", str(root), "--out", str(proc / "utsw_128"), "--workers", cpu, "--size", str(args.size)]
            if meta:
                cmd += ["--metadata", str(meta)]
            run(cmd, env, state / "logs" / "preprocess_utsw.log")
        else:
            log("UTSW not attached -> only BraTS runs (no OOD). Attach raw UTSW or a preprocessed utsw_128 folder.")
    available = []
    for ds in DATASETS:
        d = proc / f"{ds}_128"
        if not (d / "index.csv").exists():
            continue
        r = run(py + ["trace_seg3d.check_data", str(d), "--png", str(state / f"check_{ds}.png")], env, state / "logs" / f"check_{ds}.log", check=False)
        print(r.stdout.strip(), flush=True)
        if r.returncode != 0 or "!!" in r.stdout:
            log(f"!! {ds}: data check FAILED -> dataset not used. See logs/check_{ds}.log and check_{ds}.png")
            continue
        split = state / "splits" / f"{ds}.json"
        if not split.exists():
            cmd = py + ["trace_seg3d.splits", "--index", str(d / "index.csv"), "--out", str(split)]
            if ds == "utsw":
                cmd.append("--exclude-auto-labels")
            run(cmd, env)
        available.append(ds)
    return available


# -------------------------------------------------------------------------------------- jobs
@dataclass
class Job:
    kind: str  # "train" | "eval"
    src: str
    method: str
    seeds: list[str]
    name: str = ""
    proc: subprocess.Popen | None = None
    gpu: str = ""
    started: float = 0.0
    log_path: Path | None = None
    status: str = "todo"  # todo | running | done | failed | stopped
    rc: int | None = None
    extra_env: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.name = f"{self.kind}:{self.src}/{self.method}_s{'+'.join(self.seeds)}"


def run_dir(args, src: str, method: str, seed: str) -> Path:
    return args.state / "runs" / src / f"{method}_s{seed}"


def trained(args, src, method, seed) -> bool:
    d = run_dir(args, src, method, seed)
    return (d / "summary.json").exists() and (d / "calib.json").exists()


def eval_marker(args, src, method) -> Path:
    return args.state / "runs" / src / f".eval_{method}.json"


def eval_done(args, job: Job, targets: list[str]) -> bool:
    m = eval_marker(args, job.src, job.method)
    if not m.exists():
        return False
    info = json.loads(m.read_text())
    if info.get("seeds") != job.seeds:
        return False
    return all((run_dir(args, job.src, job.method, s) / f"eval_{t}_test" / "summary.json").exists() for s in job.seeds for t in targets)


def eval_target(args, group: Job) -> list[str] | None:
    """Seeds to evaluate now: the first seed as soon as it is trained (early results), then all planned seeds
    once they are all trained (re-evaluated so the seed-ensemble audit signal is included)."""

    done = [s for s in group.seeds if trained(args, group.src, group.method, s)]
    if done == group.seeds:
        return group.seeds
    if group.seeds[0] in done:
        return [group.seeds[0]]
    return None


def eval_complete(args, src: str, method: str, seeds: list[str], targets: list[str]) -> bool:
    return eval_done(args, Job("eval", src, method, seeds), targets)


def build_jobs(args, available: list[str]) -> tuple[list[Job], list[Job]]:
    seeds = args.seeds.split()
    trains, evals = [], []
    for seed in seeds:
        for src in available:
            for m in args.methods.split():
                trains.append(Job("train", src, m, [seed]))
    abl_src = args.abl_source if args.abl_source in available else (available[0] if available else None)
    abl = args.ablations.split()
    if abl_src:
        for a in abl:
            trains.append(Job("train", abl_src, a, [args.abl_seed]))
    for src in available:
        for m in args.methods.split():
            evals.append(Job("eval", src, m, seeds))
        if src == abl_src:
            for a in abl:
                evals.append(Job("eval", src, a, [args.abl_seed]))
    return trains, evals


def job_env(args, job: Job, base_env: dict) -> dict:
    env = dict(base_env)
    st = args.state
    env.update({
        "PYTHON_BIN": args.python,
        "DATA_BRATS": str(st / "processed" / "brats_128"), "SPLIT_BRATS": str(st / "splits" / "brats.json"),
        "DATA_UTSW": str(st / "processed" / "utsw_128"), "SPLIT_UTSW": str(st / "splits" / "utsw.json"),
        "RUNS": str(st / "runs"), "RESULTS": str(st / "results"),
        "SOURCES": job.src, "EPOCHS": str(args.epochs), "WORKERS": str(args.workers_per_job),
        "AMP": "0" if args.gpus == "cpu" else "1", "CCT_K": str(args.cct_k), "SAVE_MAPS": str(args.save_maps),
        "EXTRA_TRAIN_ARGS": args.extra, "SKIP_EXISTING": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })
    is_abl = job.method in ABLATION_FLAGS
    env["MAIN_METHODS"] = "" if is_abl else job.method
    env["ABLATIONS"] = job.method if is_abl else ""
    env["SEEDS"] = " ".join(job.seeds) if not is_abl else args.abl_seed
    env["ABL_SEEDS"] = " ".join(job.seeds) if is_abl else ""
    env["STEPS"] = "train calib" if job.kind == "train" else "eval"
    if job.gpu not in ("", "cpu"):
        env["CUDA_VISIBLE_DEVICES"] = job.gpu
    return env


def start(args, job: Job, gpu: str, base_env: dict) -> None:
    job.gpu = gpu
    if job.kind == "eval":
        m = eval_marker(args, job.src, job.method)
        old = json.loads(m.read_text()).get("seeds") if m.exists() else None
        if old is not None and old != job.seeds:  # seed set changed -> ensemble signal changed -> redo eval
            for s in job.seeds:
                for ev in run_dir(args, job.src, job.method, s).glob("eval_*_test"):
                    shutil.rmtree(ev)
    job.log_path = args.state / "logs" / (job.name.replace(":", "_").replace("/", "_") + ".log")
    job.log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(job.log_path, "a")
    handle.write(f"\n===== {time.ctime()} start on gpu {gpu}\n")
    handle.flush()
    job.proc = subprocess.Popen(["bash", str(args.code / "scripts/isbi/run_all.sh")], cwd=args.code, env=job_env(args, job, base_env),
                                stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
    job.started = time.time()
    job.status = "running"
    log(f"START {job.name} on gpu {gpu}")


def stop(job: Job) -> None:
    if job.proc and job.proc.poll() is None:
        os.killpg(job.proc.pid, signal.SIGTERM)
        try:
            job.proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(job.proc.pid, signal.SIGKILL)
            job.proc.wait()
    job.status = "stopped"


def train_progress(args, job: Job) -> str:
    h = run_dir(args, job.src, job.method, job.seeds[0]) / "history.json"
    try:
        last = json.loads(h.read_text())[-1]
        val = [x for x in json.loads(h.read_text()) if "val/mean" in x]
        best = max((x["val/mean"] for x in val), default=float("nan"))
        return f"epoch {last['epoch']}/{args.epochs} ({last['time_s'] / 60:.1f} min/epoch, best val {best:.3f})"
    except Exception:
        return ""


def tail(path: Path | None, n: int = 25) -> str:
    if not path or not path.exists():
        return ""
    return "\n".join(path.read_text(errors="replace").splitlines()[-n:])


# ------------------------------------------------------------------------------------------ main
def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-root", type=Path, default=Path("/kaggle/input"))
    p.add_argument("--state", type=Path, default=Path("/kaggle/working/trace_state"))
    p.add_argument("--code", type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument("--hours", type=float, default=11.0, help="stop all jobs this many hours after start")
    p.add_argument("--methods", default="baseline styleaug trace")
    p.add_argument("--seeds", default="0")
    p.add_argument("--ablations", default="")
    p.add_argument("--abl-source", default="utsw")
    p.add_argument("--abl-seed", default="0")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--extra", default="--patch-size 96 --batch-size 2 --val-every 5")
    p.add_argument("--gpus", default="auto", help="'auto', 'cpu', or a list like '0,1'")
    p.add_argument("--workers-per-job", type=int, default=2)
    p.add_argument("--cct-k", type=int, default=4)
    p.add_argument("--save-maps", type=int, default=1)
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--size", type=int, default=128, help="preprocessing cube size (keep 128; smaller only for tests)")
    p.add_argument("--min-train-min", type=float, default=15, help="do not start a train job with less time left")
    p.add_argument("--min-eval-min", type=float, default=45, help="do not start an eval job with less time left")
    p.add_argument("--poll-s", type=float, default=30)
    args = p.parse_args(argv)

    t0 = time.time()
    deadline = t0 + args.hours * 3600
    base_env = dict(os.environ)
    base_env["PYTHONPATH"] = f"{args.code}:{args.code / 'src'}" + (f":{base_env['PYTHONPATH']}" if base_env.get("PYTHONPATH") else "")

    version = restore_state(args.input_root, args.state)
    log(f"state: {args.state} (previous version {version})")
    available = prepare_data(args, base_env)
    log(f"datasets ready: {available or 'NONE'}")
    if not available:
        raise SystemExit("no dataset available")

    if args.gpus == "auto":
        try:
            import torch

            n = torch.cuda.device_count()
        except Exception:
            n = 0
        gpus = [str(i) for i in range(n)] or ["cpu"]
    elif args.gpus == "cpu":
        gpus = ["cpu"]
    else:
        gpus = args.gpus.split(",")
    log(f"GPU slots: {gpus}")

    trains, evals = build_jobs(args, available)
    for j in trains:
        if trained(args, j.src, j.method, j.seeds[0]):
            j.status = "done"
    def eval_status(g: Job) -> str:
        tgt = eval_target(args, g)
        if tgt == g.seeds and eval_complete(args, g.src, g.method, g.seeds, available):
            return "done"
        if tgt and eval_complete(args, g.src, g.method, tgt, available):
            return f"partial (seeds {'+'.join(tgt)})"
        return "todo"

    log(f"jobs: {sum(j.status == 'todo' for j in trains)} train + {sum(eval_status(g) != 'done' for g in evals)} eval to do "
        f"({sum(j.status == 'done' for j in trains)} train / {sum(eval_status(g) == 'done' for g in evals)} eval already done)")
    failed: set[str] = set()

    running: dict[str, Job] = {}
    last_report = 0.0
    while True:
        now = time.time()
        left_min = (deadline - now) / 60
        # finished jobs
        for gpu, job in list(running.items()):
            rc = job.proc.poll() if job.proc else 0
            if rc is None:
                continue
            job.rc = rc
            del running[gpu]
            if rc == 0:
                job.status = "done"
                if job.kind == "eval":
                    eval_marker(args, job.src, job.method).write_text(json.dumps({"seeds": job.seeds, "targets": available}))
                log(f"DONE  {job.name} ({(now - job.started) / 60:.0f} min)")
            else:
                job.status = "failed"
                failed.add(job.name)
                log(f"FAIL  {job.name} rc={rc}; last log lines:\n{tail(job.log_path)}")
        if left_min <= 0:
            for job in running.values():
                log(f"time limit -> stopping {job.name} (training resumes next version)")
                stop(job)
            running.clear()
            break
        # schedule
        for gpu in gpus:
            if gpu in running:
                continue
            busy = {j.name for j in running.values()}
            nxt = None
            if left_min >= args.min_eval_min:
                for g in evals:
                    tgt = eval_target(args, g)
                    if not tgt or eval_complete(args, g.src, g.method, tgt, available):
                        continue
                    cand = Job("eval", g.src, g.method, tgt)
                    if cand.name not in busy and cand.name not in failed and not any(
                            r.kind == "eval" and r.src == g.src and r.method == g.method for r in running.values()):
                        nxt = cand
                        break
            if nxt is None and left_min >= args.min_train_min:
                for j in trains:
                    if j.status == "todo" and j.name not in busy:
                        nxt = j
                        break
            if nxt:
                start(args, nxt, gpu, base_env)
                running[gpu] = nxt
        if not running:
            todo = [j for j in trains if j.status == "todo"] + [g for g in evals if eval_status(g) != "done"]
            if todo:
                log(f"{len(todo)} job(s) not started: not enough time left in this version")
            break
        if now - last_report > 600:
            last_report = now
            for job in running.values():
                extra = train_progress(args, job) if job.kind == "train" else ""
                log(f"  ... {job.name} on gpu {job.gpu}: {max(0.0, now - job.started) / 60:.0f} min {extra} | {left_min:.0f} min left in this version")
        time.sleep(args.poll_s)

    # audit + tables over everything finished so far (CPU, quick)
    env = job_env(args, Job("eval", " ".join(available), args.methods, args.seeds.split()), base_env)
    env.update({"SOURCES": " ".join(available), "MAIN_METHODS": args.methods, "SEEDS": args.seeds,
                "ABLATIONS": args.ablations, "ABL_SEEDS": args.abl_seed, "STEPS": "audit tables"})
    env.pop("CUDA_VISIBLE_DEVICES", None)
    r = run(["bash", str(args.code / "scripts/isbi/run_all.sh")], env, args.state / "logs" / "tables.log", check=False)
    if r.returncode != 0:
        log(f"tables step failed (normal if no eval finished yet):\n{r.stderr[-1500:]}")

    # status report
    lines = [f"# TRACE ISBI – status after version {version + 1}", "", f"datasets: {', '.join(available)}", "",
             "| job | status | progress |", "|---|---|---|"]
    for j in trains:
        st = "done" if trained(args, j.src, j.method, j.seeds[0]) else j.status
        lines.append(f"| {j.name} | {st} | {train_progress(args, j) if st != 'done' else ''} |")
    for g in evals:
        lines.append(f"| {g.name} | {eval_status(g)} |  |")
    remaining = sum(1 for line in lines[6:] if "| done |" not in line)
    lines += ["", f"**{remaining} job(s) left.** " + ("Everything finished." if remaining == 0 else
              "Save a new version of the notebook to continue.")]
    (args.state / "STATUS.md").write_text("\n".join(lines) + "\n")
    (args.state / "state_meta.json").write_text(json.dumps({"version": version + 1, "time": time.ctime()}))
    print("\n".join(lines), flush=True)
    table = args.state / "results" / "final" / "table.md"
    if table.exists():
        print("\n" + table.read_text(), flush=True)


if __name__ == "__main__":
    main()

# Public Release Checklist

Use this checklist before pushing the repository to a public branch.

## Keep In Git

Core model and OOD pipeline:

```text
README.md
pyproject.toml
baselines/README.md
baselines/run_mednext_ood_baselines.sh
baselines/mednext/
baselines/segformer3d/
scripts/
src/crn/mednext_blocks.py
src/crn/metrics.py
tests/test_mednext_baseline.py
tests/test_metrics.py
docs/REPRODUCING_MEDNEXT_OOD.md
docs/PUBLIC_RELEASE_CHECKLIST.md
```

Supporting files such as `src/crn/__init__.py` may remain tracked so imports
work normally.

## Keep Local, Do Not Push

```text
data/
runs/
experiments/
to_human/
paper/
literature/
findings.md
research-log.md
research-state.yaml
*.pt
*.ckpt
*.nii.gz
*.h5
```

## Untrack Data Without Deleting It

`.gitignore` does not remove files that are already tracked. This repository has
historically tracked some local UTSW medical image files. To remove them from
the git index while keeping the files on disk, run:

```bash
git rm --cached -r --ignore-unmatch data
```

If you also want the public branch to exclude legacy prototype configs and
non-mainline code while keeping them locally:

```bash
git rm --cached -r --ignore-unmatch configs src/cpa_seg3d
git rm --cached --ignore-unmatch findings.md research-log.md research-state.yaml
```

Review carefully after untracking:

```bash
git status --short
```

The status will show staged deletions for files removed from git tracking. That
is expected; the files remain in your local working directory.

## Safe Add Command For The Current Public Mainline

```bash
git add \
  README.md \
  .gitignore \
  pyproject.toml \
  baselines/README.md \
  baselines/run_mednext_ood_baselines.sh \
  baselines/mednext \
  baselines/segformer3d \
  scripts \
  src/crn/__init__.py \
  src/crn/mednext_blocks.py \
  src/crn/metrics.py \
  tests/test_mednext_baseline.py \
  tests/test_metrics.py \
  docs/REPRODUCING_MEDNEXT_OOD.md \
  docs/PUBLIC_RELEASE_CHECKLIST.md
```

Do not use `git add -A` on a machine that contains local data/checkpoints.

## External Artifact To Share With Teammates

The current best checkpoint should be shared outside git:

```text
runs/_ood_causal_adapt_brats_v5_et_precision_e2/best.pt
```

After downloading it into the same path, the reproduction command in
`docs/REPRODUCING_MEDNEXT_OOD.md` should produce the current best OOD smoke row.

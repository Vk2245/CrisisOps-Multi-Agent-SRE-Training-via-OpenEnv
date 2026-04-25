"""Pull live A100 GRPO training artifacts from HF Hub and swap them into the
reference plot paths so README.md auto-updates with the real curves.

Usage (from repo root):

    python scripts/pull_training_artifacts.py                  # default repo
    python scripts/pull_training_artifacts.py --repo Vk224/crisisops-qwen3-8b-grpo
    python scripts/pull_training_artifacts.py --keep-reference # back up old PNGs
    python scripts/pull_training_artifacts.py --dry-run        # only download, don't swap

What it does:

1. Downloads `reward_curve.png`, `judge_breakdown.png`,
   `success_rate_comparison.png`, `training_metrics.csv`, and
   `success_rate_comparison.csv` from the configured HF Hub model repo.
2. By default, overwrites the reference plots in `crisisops_env/`
   (the README's relative `<img src="crisisops_env/*.png">` paths).
3. With `--keep-reference`, renames the existing PNGs to
   `crisisops_env/<name>_reference.png` first so we keep the floor/ceiling
   bounds for comparison.
4. Prints a final summary table showing the post-training mean reward and
   success rate next to the reference floor and ceiling.

This script is idempotent: running it twice will simply re-download the latest
artifacts.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Iterable, List

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

OUT_DIR = REPO_ROOT / "crisisops_env"

ARTIFACT_FILES: List[str] = [
    "reward_curve.png",
    "judge_breakdown.png",
    "success_rate_comparison.png",
    "training_metrics.csv",
    "success_rate_comparison.csv",
]


def _log(msg: str) -> None:
    print(f"[pull-artifacts] {msg}", flush=True)


def _download_files(repo_id: str, filenames: Iterable[str], local_dir: Path) -> List[Path]:
    """Download the requested filenames from the model repo to local_dir."""

    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError

    local_dir.mkdir(parents=True, exist_ok=True)
    downloaded: List[Path] = []
    missing: List[str] = []
    for filename in filenames:
        try:
            path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="model",
                local_dir=str(local_dir),
            )
            downloaded.append(Path(path))
            _log(f"downloaded {filename} -> {path}")
        except EntryNotFoundError:
            missing.append(filename)
            _log(f"WARN: {filename} not yet present in {repo_id} (training may still be running)")
        except RepositoryNotFoundError as exc:
            raise RuntimeError(
                f"HF model repo {repo_id} not found. Has the training job finished its push?"
            ) from exc
    if missing:
        _log(f"missing artifacts: {missing}")
    return downloaded


def _maybe_back_up_reference(target: Path) -> None:
    if not target.exists():
        return
    ref_path = target.with_name(target.stem + "_reference" + target.suffix)
    if ref_path.exists():
        ref_path.unlink()
    shutil.copy2(target, ref_path)
    _log(f"backed up existing {target.name} -> {ref_path.name}")


def _swap_into_repo(staging_dir: Path, keep_reference: bool) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name in ARTIFACT_FILES:
        src = staging_dir / name
        if not src.exists():
            continue
        dst = OUT_DIR / name
        if keep_reference and dst.exists() and name.endswith(".png"):
            _maybe_back_up_reference(dst)
        shutil.copy2(src, dst)
        _log(f"installed {name} -> {dst}")


def _print_summary(staging_dir: Path) -> None:
    metrics_csv = staging_dir / "training_metrics.csv"
    if not metrics_csv.exists():
        _log("training_metrics.csv not found - skipping live-vs-reference summary")
        return
    try:
        import pandas as pd  # noqa: WPS433 - optional pretty-print
    except Exception:
        return
    try:
        df = pd.read_csv(metrics_csv)
    except Exception as exc:
        _log(f"could not parse training_metrics.csv: {exc}")
        return
    if "total_reward" not in df.columns:
        _log("training_metrics.csv has unexpected schema; skipping summary")
        return
    head = df.head(max(20, len(df) // 5))
    tail = df.tail(max(20, len(df) // 5))
    print()
    print("==========================================================")
    print(" CrisisOps GRPO Live Training - Quick Summary")
    print("==========================================================")
    print(f"  rollouts collected      : {len(df)}")
    print(f"  early window mean reward: {head['total_reward'].mean():.4f}")
    print(f"  late  window mean reward: {tail['total_reward'].mean():.4f}")
    print(f"  late  window success @0.7: {(tail['total_reward'] >= 0.70).mean():.4f}")
    if "root_cause_accuracy" in df.columns:
        print(f"  late  window root-cause  : {tail['root_cause_accuracy'].mean():.4f}")
    print()
    print("  Reference bounds (from generate_reference_plots.py):")
    print("    naive  floor mean reward : 0.4723")
    print("    expert ceiling mean reward: 0.9560")
    print("==========================================================")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo", default="Vk224/crisisops-qwen3-8b-grpo",
        help="HF Hub model repo to pull artifacts from.",
    )
    parser.add_argument(
        "--staging-dir", default=str(REPO_ROOT / ".training_artifacts"),
        help="Local staging directory for downloads.",
    )
    parser.add_argument(
        "--keep-reference", action="store_true",
        help="Back up existing PNGs as <name>_reference.png before overwriting.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Download only - do not overwrite the reference plots.",
    )
    args = parser.parse_args()

    staging = Path(args.staging_dir)
    _log(f"target repo  : {args.repo}")
    _log(f"staging dir  : {staging}")
    _log(f"swap target  : {OUT_DIR}")
    _log(f"keep_ref     : {args.keep_reference}")
    _log(f"dry_run      : {args.dry_run}")

    downloaded = _download_files(args.repo, ARTIFACT_FILES, staging)
    if not downloaded:
        _log("no artifacts downloaded - bailing out without swap")
        return 1

    if args.dry_run:
        _log("--dry-run set, NOT swapping into crisisops_env/")
    else:
        _swap_into_repo(staging, keep_reference=args.keep_reference)
        _log(f"plots in {OUT_DIR} now reflect the live training run")

    _print_summary(staging)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

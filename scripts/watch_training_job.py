"""Watchdog for the live A100 GRPO training job.

Polls a Hugging Face Jobs run every N seconds. When it transitions to a
terminal state (COMPLETED / ERROR / CANCELED), automatically triggers
`scripts/pull_training_artifacts.py` (on COMPLETED only) so the README
plots in `crisisops_env/` are swapped to the live training curves
without any manual action.

Usage (from repo root):

    python scripts/watch_training_job.py --job-id 69ed4520d70108f37acdf184
    python scripts/watch_training_job.py                   # uses LIVE_JOB_ID below
    python scripts/watch_training_job.py --interval 90     # poll every 90 s
    python scripts/watch_training_job.py --no-pull         # only print state

Designed to be safe to leave running unattended overnight. Exits with code
0 on COMPLETED+pull, 1 on ERROR/CANCELED, 2 on ctrl-C.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent

LIVE_JOB_ID = "69ed648fd2c8bd8662bcec55"

TERMINAL_STAGES = {"COMPLETED", "ERROR", "CANCELED", "TIMEOUT", "FAILED"}


def _log(msg: str) -> None:
    ts = dt.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def _poll_job(job_id: str) -> Dict[str, Optional[str]]:
    """Return {stage, message, created_at, flavor} for the job."""

    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub not installed. pip install huggingface_hub"
        ) from exc

    api = HfApi()
    try:
        job = api.inspect_job(job_id=job_id)
    except Exception as exc:
        return {"stage": "POLL_ERROR", "message": str(exc)}
    status = getattr(job, "status", None)
    return {
        "stage": (getattr(status, "stage", None) or "UNKNOWN").upper(),
        "message": getattr(status, "message", None),
        "created_at": str(getattr(job, "created_at", "") or ""),
        "flavor": str(getattr(job, "flavor", "") or ""),
    }


def _run_pull_artifacts(repo_id: Optional[str]) -> int:
    cmd = [sys.executable, str(REPO_ROOT / "scripts" / "pull_training_artifacts.py")]
    if repo_id:
        cmd += ["--repo", repo_id]
    _log(f"running: {' '.join(cmd)}")
    return subprocess.call(cmd, cwd=str(REPO_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", default=LIVE_JOB_ID, help="HF Jobs ID to watch.")
    parser.add_argument(
        "--interval", type=int, default=60,
        help="Poll interval in seconds (default 60).",
    )
    parser.add_argument(
        "--repo", default="Vk224/crisisops-qwen3-8b-grpo",
        help="HF Hub model repo to pull artifacts from on completion.",
    )
    parser.add_argument(
        "--no-pull", action="store_true",
        help="Just watch and print state - do not run pull_training_artifacts.py.",
    )
    args = parser.parse_args()

    _log(f"watchdog started for job {args.job_id} (interval={args.interval}s)")
    _log(f"will pull from {args.repo} on COMPLETED" if not args.no_pull else "no-pull mode")

    last_stage = ""
    poll_count = 0
    started_seconds_seen = False
    try:
        while True:
            poll_count += 1
            info = _poll_job(args.job_id)
            stage = info["stage"] or "UNKNOWN"
            if stage != last_stage:
                _log(
                    f"poll #{poll_count}: stage={stage} "
                    f"flavor={info.get('flavor')!r} "
                    f"created_at={info.get('created_at')!r}"
                )
                last_stage = stage
            else:
                if poll_count % 5 == 0:
                    _log(f"poll #{poll_count}: stage={stage} (no change)")

            if stage == "RUNNING" and not started_seconds_seen:
                started_seconds_seen = True
                _log("=> Job is now RUNNING. Tail logs with: hf jobs logs " + args.job_id)

            if stage in TERMINAL_STAGES:
                _log(f"=> Job entered terminal stage: {stage}")
                if stage == "COMPLETED":
                    if args.no_pull:
                        _log("--no-pull set; skipping artifact pull.")
                        return 0
                    rc = _run_pull_artifacts(args.repo)
                    if rc == 0:
                        _log("=> Artifacts pulled and swapped successfully.")
                        _log("Suggested follow-up: git add crisisops_env/ && "
                             "git commit -m 'feat: swap reference plots for live "
                             "GRPO training curves' && git push origin main")
                    else:
                        _log(f"=> pull_training_artifacts.py exited with {rc}.")
                    return rc
                _log(f"Job did NOT complete cleanly (stage={stage}). "
                     "Inspect with: hf jobs inspect " + args.job_id)
                return 1

            time.sleep(args.interval)
    except KeyboardInterrupt:
        _log("Interrupted by user.")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

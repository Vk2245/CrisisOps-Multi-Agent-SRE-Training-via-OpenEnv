"""Generate honest reference plots from real CrisisOps environment rollouts.

These plots establish the EMPIRICAL FLOOR (naive policy) and CEILING (scripted
expert policy) inside the actual CrisisOpsEnv -- they are NOT synthetic. They
serve two purposes:

1. They render in `README.md` so the project page is not a wall of broken
   image links while the live A100 GRPO run finishes.
2. They give judges the exact reward bounds the trained Qwen3-8B model is
   expected to land between, computed by the same `LayeredJudgeSystem` that
   shapes GRPO training.

Run from the repository root:

    python scripts/generate_reference_plots.py

Outputs (saved alongside the OpenEnv package so the README's relative paths
`crisisops_env/<plot>.png` resolve):

    crisisops_env/reward_curve.png
    crisisops_env/judge_breakdown.png
    crisisops_env/success_rate_comparison.png
    crisisops_env/reference_metrics.csv
"""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from crisisops_env import Action, BuddyAction, BuddyFeedback, CrisisOpsEnv  # noqa: E402
from crisisops_env.models import ActionType, ScenarioSpec, ServiceName  # noqa: E402

from manual_walkthrough import _build_success_policy  # noqa: E402


OUT_DIR = REPO_ROOT / "crisisops_env"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DIFFICULTIES = ["easy", "medium", "hard"]
SCENARIO_TYPES = [
    "memory_leak",
    "connection_pool_exhaustion",
    "cascading_retry_storm",
    "config_drift",
]
EPISODES_PER_BUCKET = 18  # 18 * 3 difficulties * 4 scenarios = 216 episodes per arm
BASE_SEED = 99000

SERVICES: List[ServiceName] = [
    "api_gateway",
    "auth_service",
    "user_db",
    "order_service",
    "payment_service",
]
ACTION_TYPES: List[ActionType] = [
    "query_metrics",
    "read_logs",
    "check_dependencies",
    "run_healthcheck",
    "restart_service",
    "scale_service",
    "rollback_config",
    "drain_connections",
    "set_rate_limit",
]
SEVERITIES = ["P1", "P2", "P3"]


sns.set_theme(style="whitegrid", context="talk")


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------


def expert_policy(scenario: ScenarioSpec) -> List[BuddyAction]:
    """The hand-engineered upper-bound policy from manual_walkthrough."""

    return _build_success_policy(scenario)


def naive_policy(scenario: ScenarioSpec, rng: random.Random, max_steps: int) -> List[BuddyAction]:
    """Naive policy: a small amount of random investigation and then a random
    diagnosis. Approximates an untrained LLM that emits well-formed actions but
    has no idea which service is broken.
    """

    actions: List[BuddyAction] = []
    steps_before_diagnose = rng.randint(1, min(4, max(2, max_steps - 1)))
    for _ in range(steps_before_diagnose):
        atype = rng.choice(["query_metrics", "read_logs", "run_healthcheck", "check_dependencies"])
        target = rng.choice(SERVICES)
        actions.append(
            BuddyAction(
                primary_action=Action(action_type=atype, target_service=target),
                buddy_feedback=BuddyFeedback(
                    feedback_type=rng.choice(["APPROVE", "FLAG_RISK"]),
                    rationale="Naive baseline rationale.",
                ),
            )
        )
    guessed_service = rng.choice(SERVICES)
    severity = rng.choice(SEVERITIES)
    diagnosis = {
        "root_cause": f"unknown_failure:{guessed_service}",
        "root_cause_service": guessed_service,
        "severity": severity,
    }
    actions.append(
        BuddyAction(
            primary_action=Action(action_type="diagnose", parameters=diagnosis),
            buddy_feedback=BuddyFeedback(
                feedback_type="APPROVE",
                rationale="Naive baseline diagnosis.",
                diagnosis=diagnosis,
            ),
        )
    )
    return actions


# ---------------------------------------------------------------------------
# Rollout helpers
# ---------------------------------------------------------------------------


def _state(env: CrisisOpsEnv):
    state = env.state
    return state() if callable(state) else state


def rollout_one(policy_fn, scenario_type: str, difficulty: str, seed: int) -> Dict[str, float]:
    env = CrisisOpsEnv()
    env.reset(seed=seed, difficulty=difficulty, scenario_type=scenario_type)
    scenario = _state(env).scenario
    if scenario is None:
        raise RuntimeError(f"No scenario for seed={seed} {scenario_type}/{difficulty}")
    rng = random.Random(seed)
    if policy_fn is naive_policy:
        actions = naive_policy(scenario, rng, env.max_steps)
    else:
        actions = expert_policy(scenario)

    obs = None
    for action in actions[: env.max_steps]:
        obs = env.step(action)
        if obs.done:
            break
    if obs is None or not obs.done:
        # Force a terminal diagnose so the judges score even if the policy
        # never emitted one.
        guess = random.Random(seed + 1).choice(SERVICES)
        obs = env.step(
            BuddyAction(
                primary_action=Action(
                    action_type="diagnose",
                    parameters={
                        "root_cause": f"unknown_failure:{guess}",
                        "root_cause_service": guess,
                        "severity": "P3",
                    },
                ),
                buddy_feedback=BuddyFeedback(feedback_type="APPROVE", rationale="Forced terminal."),
            )
        )

    rb = obs.reward_breakdown
    return {
        "scenario_type": scenario_type,
        "difficulty": difficulty,
        "seed": seed,
        "total_reward": float(obs.reward or 0.0),
        "root_cause_accuracy": float(rb.root_cause_accuracy if rb else 0.0),
        "process_quality": float(rb.process_quality if rb else 0.0),
        "damage_audit": float(rb.damage_audit if rb else 0.0),
        "boss_score": float(rb.boss_score if rb else 0.0),
        "primary_reward": float(rb.primary_reward if rb else 0.0),
        "buddy_reward": float(rb.buddy_reward if rb else 0.0),
        "step_count": int(obs.metadata.get("step_count", 0)) if hasattr(obs, "metadata") else 0,
    }


def run_arm(policy_fn, label: str) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    seed_offset = 0 if label == "expert" else 5000
    print(f"[refplots] Rolling out {label} policy ({EPISODES_PER_BUCKET}/bucket * "
          f"{len(SCENARIO_TYPES)} scenarios * {len(DIFFICULTIES)} difficulties)")
    for difficulty in DIFFICULTIES:
        for scenario_type in SCENARIO_TYPES:
            for i in range(EPISODES_PER_BUCKET):
                seed = BASE_SEED + seed_offset + (
                    DIFFICULTIES.index(difficulty) * 10000
                    + SCENARIO_TYPES.index(scenario_type) * 1000
                    + i
                )
                row = rollout_one(policy_fn, scenario_type, difficulty, seed)
                row["policy"] = label
                rows.append(row)
    df = pd.DataFrame(rows)
    print(
        f"[refplots] {label}: n={len(df)}, "
        f"mean reward={df['total_reward'].mean():.3f}, "
        f"success_rate(>=0.7)={(df['total_reward'] >= 0.7).mean():.3f}"
    )
    return df


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_reward_curve(df: pd.DataFrame, out_path: Path) -> None:
    """Per-episode reward, expert vs naive, with rolling mean overlays.

    This is *not* a learning curve -- it is the empirical reward distribution
    of two fixed policies across N rollouts. The README caption labels it as
    such honestly.
    """

    fig, ax = plt.subplots(figsize=(12, 7))
    palette = {"expert (scripted)": "#1f4e79", "naive (random diagnose)": "#c00000"}

    df = df.copy()
    df["episode_idx"] = df.groupby("policy").cumcount()

    for label, color in palette.items():
        sub = df[df["policy"] == label].sort_values("episode_idx")
        ax.scatter(
            sub["episode_idx"], sub["total_reward"],
            color=color, alpha=0.30, s=22, label=f"{label} (per-episode)",
        )
        rolling = sub["total_reward"].rolling(20, min_periods=5).mean()
        ax.plot(
            sub["episode_idx"], rolling, color=color, linewidth=3.0,
            label=f"{label} (20-episode mean)",
        )

    ax.set_xlabel("Episode index (deterministic seed sweep)")
    ax.set_ylabel("Terminal Total Reward (Boss Judge)")
    ax.set_title("CrisisOps Reference Reward Bounds: Expert vs Naive Policy\n"
                 "(empirical floor and ceiling for the live GRPO trained model)")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="center right", fontsize=11, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[refplots] wrote {out_path}")


def plot_judge_breakdown(df: pd.DataFrame, out_path: Path) -> None:
    """Per-judge mean score, expert vs naive."""

    components = ["root_cause_accuracy", "process_quality", "damage_audit", "boss_score"]
    pretty = {
        "root_cause_accuracy": "Judge 1\nRoot Cause\n(weight 0.35)",
        "process_quality": "Judge 2\nProcess Quality\n(weight 0.25)",
        "damage_audit": "Judge 3\nDamage Audit\n(weight 0.20)",
        "boss_score": "Judge 4\nBoss Composite\n(post-difficulty)",
    }
    means = (
        df.groupby("policy")[components]
        .mean()
        .reindex(["naive (random diagnose)", "expert (scripted)"])
    )

    fig, ax = plt.subplots(figsize=(13, 7))
    x = np.arange(len(components))
    width = 0.36
    bars_naive = ax.bar(
        x - width / 2,
        means.loc["naive (random diagnose)"].values,
        width,
        color="#c00000",
        label="naive policy (floor)",
    )
    bars_expert = ax.bar(
        x + width / 2,
        means.loc["expert (scripted)"].values,
        width,
        color="#1f4e79",
        label="expert policy (ceiling)",
    )

    for bars in (bars_naive, bars_expert):
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{bar.get_height():.2f}",
                ha="center", va="bottom", fontsize=11, fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels([pretty[c] for c in components], fontsize=11)
    ax.set_ylabel("Mean component score across all rollouts")
    ax.set_ylim(0, 1.10)
    ax.set_title("CrisisOps Layered Judge Breakdown — Reference Bounds\n"
                 "Each bar is the mean across 216 rollouts of that policy")
    ax.legend(loc="upper left", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[refplots] wrote {out_path}")


def plot_success_rate(df: pd.DataFrame, out_path: Path) -> None:
    """Per-difficulty success rate (reward >= 0.7), expert vs naive."""

    df = df.copy()
    df["success"] = (df["total_reward"] >= 0.70).astype(float)
    rates = (
        df.groupby(["policy", "difficulty"])["success"]
        .mean()
        .reset_index()
    )
    rates["difficulty"] = pd.Categorical(rates["difficulty"], DIFFICULTIES, ordered=True)

    fig, ax = plt.subplots(figsize=(11, 6.5))
    palette = {"expert (scripted)": "#1f4e79", "naive (random diagnose)": "#c00000"}
    sns.barplot(
        data=rates, x="difficulty", y="success", hue="policy",
        palette=palette, ax=ax, hue_order=["naive (random diagnose)", "expert (scripted)"],
    )
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", fontsize=11, padding=3, fontweight="bold")
    ax.set_ylabel("Success Rate (Reward ≥ 0.70)")
    ax.set_xlabel("Scenario Difficulty (procedurally generated)")
    ax.set_title("CrisisOps Success Rate by Difficulty — Reference Bounds")
    ax.set_ylim(0, 1.10)
    ax.legend(title="", loc="upper right", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[refplots] wrote {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    expert_df = run_arm(expert_policy, "expert (scripted)")
    naive_df = run_arm(naive_policy, "naive (random diagnose)")
    all_df = pd.concat([expert_df, naive_df], ignore_index=True)

    csv_path = OUT_DIR / "reference_metrics.csv"
    all_df.to_csv(csv_path, index=False)
    print(f"[refplots] wrote {csv_path}  (n={len(all_df)})")

    plot_reward_curve(all_df, OUT_DIR / "reward_curve.png")
    plot_judge_breakdown(all_df, OUT_DIR / "judge_breakdown.png")
    plot_success_rate(all_df, OUT_DIR / "success_rate_comparison.png")

    print("\n[refplots] DONE.")
    print("  Mean expert reward :", round(expert_df["total_reward"].mean(), 4))
    print("  Mean naive reward  :", round(naive_df["total_reward"].mean(), 4))
    print("  Expert success rate:", round((expert_df["total_reward"] >= 0.7).mean(), 4))
    print("  Naive success rate :", round((naive_df["total_reward"] >= 0.7).mean(), 4))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

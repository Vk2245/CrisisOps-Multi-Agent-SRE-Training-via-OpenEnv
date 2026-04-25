"""Tiny smoke test for the CrisisOps notebook reward bridge.

This mirrors the parsing + environment rollout logic in
`notebooks/crisisops_grpo_training.ipynb` cells 12 and 14, but does not load
Qwen3-8B. It verifies:

1. A well-formed scripted completion produces a high terminal reward.
2. A malformed completion gracefully returns 0.0 with a parse error tag.
3. Hard-difficulty config-drift trajectories are scorable end-to-end.
4. Buddy-correction (SUGGEST_ALTERNATIVE) trajectories produce a
   non-zero buddy Difference Reward.

Run from the repository root:

    python scripts/notebook_smoke_test.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from crisisops_env import Action, BuddyAction, BuddyFeedback, CrisisOpsEnv  # noqa: E402


def _completion_to_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        if completion and isinstance(completion[-1], dict):
            return str(completion[-1].get("content", ""))
        return "\n".join(map(str, completion))
    if isinstance(completion, dict):
        return str(completion.get("content", completion))
    return str(completion)


def _extract_json_actions(text: str) -> Tuple[List[BuddyAction], Optional[str]]:
    match = re.search(r"<actions>\s*(.*?)\s*</actions>", text, flags=re.DOTALL | re.IGNORECASE)
    raw = match.group(1) if match else text
    if not match:
        bracket = re.search(r"(\[.*\]|\{.*\})", text, flags=re.DOTALL)
        raw = bracket.group(1) if bracket else text
    try:
        payload = json.loads(raw)
    except Exception as exc:
        return [], f"json_parse_error: {exc}"
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list):
        return [], "actions_payload_not_list"
    actions: List[BuddyAction] = []
    for item in payload:
        try:
            actions.append(BuddyAction.model_validate(item))
        except Exception as exc:
            return actions, f"buddy_action_validation_error: {exc}"
    return actions, None


def _forced_wrong_diagnosis() -> BuddyAction:
    return BuddyAction(
        primary_action=Action(
            action_type="diagnose",
            parameters={
                "root_cause": "unknown",
                "root_cause_service": "api_gateway",
                "severity": "P3",
            },
        ),
        buddy_feedback=BuddyFeedback(
            feedback_type="APPROVE", rationale="Forced terminal fallback."
        ),
    )


def _state(env: CrisisOpsEnv):
    state = env.state
    return state() if callable(state) else state


def run_completion_in_env(
    completion_text: str, seed: int, difficulty: str
) -> Dict[str, Any]:
    env = CrisisOpsEnv()
    env.reset(seed=int(seed), difficulty=difficulty)
    actions, parse_error = _extract_json_actions(completion_text)
    scenario = _state(env).scenario
    if parse_error:
        return {
            "total_reward": 0.0,
            "root_cause_accuracy": 0.0,
            "process_quality": 0.0,
            "damage_audit": 0.0,
            "parse_error": parse_error,
            "scenario_type": scenario.scenario_type if scenario else "unknown",
            "difficulty": difficulty,
        }
    obs = None
    for action in actions[: env.max_steps]:
        obs = env.step(action)
        if obs.done:
            break
    if obs is None or not obs.done:
        obs = env.step(_forced_wrong_diagnosis())
    rb = obs.reward_breakdown
    return {
        "total_reward": float(obs.reward or 0.0),
        "root_cause_accuracy": float(rb.root_cause_accuracy if rb else 0.0),
        "process_quality": float(rb.process_quality if rb else 0.0),
        "damage_audit": float(rb.damage_audit if rb else 0.0),
        "boss_score": float(rb.boss_score if rb else 0.0),
        "primary_reward": float(rb.primary_reward if rb else 0.0),
        "buddy_reward": float(rb.buddy_reward if rb else 0.0),
        "scenario_type": _state(env).scenario.scenario_type,
        "difficulty": difficulty,
        "parse_error": None,
    }


# Scenario-aware remediation map. The notebook's GRPO model is expected to
# learn this mapping from experience; the smoke test hard-codes the optimal
# remediation per scenario family so we can prove the reward pipeline rewards
# correct behavior end-to-end.
SCENARIO_REMEDIATION: Dict[str, str] = {
    "memory_leak": "restart_service",
    "connection_pool_exhaustion": "drain_connections",
    "cascading_retry_storm": "set_rate_limit",
    "config_drift": "rollback_config",
}
SCENARIO_SEVERITY: Dict[str, str] = {
    "memory_leak": "P2",
    "connection_pool_exhaustion": "P1",
    "cascading_retry_storm": "P1",
    "config_drift": "P1",
}


def build_scripted_completion(seed: int, difficulty: str) -> Tuple[str, str, str]:
    """Inspect the procedurally generated scenario and produce an optimal
    buddy trajectory targeting the actual root_cause_service.

    Returns (completion_text, scenario_type, root_cause_service).
    """
    env = CrisisOpsEnv()
    env.reset(seed=seed, difficulty=difficulty)
    scenario = _state(env).scenario
    if scenario is None:
        raise RuntimeError(f"No scenario generated for seed={seed} difficulty={difficulty}")
    root = scenario.root_cause_service
    scenario_type = scenario.scenario_type
    remediation = SCENARIO_REMEDIATION[scenario_type]
    severity = SCENARIO_SEVERITY[scenario_type]
    diagnosis_payload = {
        "root_cause": f"{scenario_type}:{root}",
        "root_cause_service": root,
        "severity": severity,
    }
    actions = [
        {
            "primary_action": {"action_type": "query_metrics", "target_service": "api_gateway", "parameters": {}},
            "buddy_feedback": {"feedback_type": "APPROVE", "rationale": "Start at customer symptom."},
        },
        {
            "primary_action": {"action_type": "query_metrics", "target_service": root, "parameters": {}},
            "buddy_feedback": {"feedback_type": "APPROVE", "rationale": "Check suspected root."},
        },
        {
            "primary_action": {"action_type": "read_logs", "target_service": root, "parameters": {}},
            "buddy_feedback": {"feedback_type": "APPROVE", "rationale": "Read root logs before remediation."},
        },
        {
            "primary_action": {"action_type": "check_dependencies", "target_service": root, "parameters": {}},
            "buddy_feedback": {"feedback_type": "APPROVE", "rationale": "Confirm dependency cascade."},
        },
        {
            "primary_action": {"action_type": remediation, "target_service": root, "parameters": {}},
            "buddy_feedback": {
                "feedback_type": "FLAG_RISK",
                "rationale": "State-changing but supported by evidence on root cause.",
                "risk_flags": ["state_change"],
            },
        },
        {
            "primary_action": {
                "action_type": "diagnose",
                "target_service": None,
                "parameters": diagnosis_payload,
            },
            "buddy_feedback": {
                "feedback_type": "APPROVE",
                "rationale": "Evidence supports diagnosis.",
                "diagnosis": diagnosis_payload,
            },
        },
    ]
    text = "<think>Investigate, remediate root, then diagnose.</think>\n<actions>" + json.dumps(actions) + "</actions>"
    return text, scenario_type, root


def malformed_completion() -> str:
    return "<think>I am confused</think>\n<actions>not json at all { broken</actions>"


def build_buddy_corrected_completion(seed: int, difficulty: str) -> Tuple[str, str, str]:
    """Primary proposes risky restart of api_gateway; buddy redirects to the
    real root cause service. Exercises the Difference Reward (D_i) path."""
    env = CrisisOpsEnv()
    env.reset(seed=seed, difficulty=difficulty)
    scenario = _state(env).scenario
    if scenario is None:
        raise RuntimeError(f"No scenario for seed={seed} difficulty={difficulty}")
    root = scenario.root_cause_service
    scenario_type = scenario.scenario_type
    remediation = SCENARIO_REMEDIATION[scenario_type]
    severity = SCENARIO_SEVERITY[scenario_type]
    diagnosis_payload = {
        "root_cause": f"{scenario_type}:{root}",
        "root_cause_service": root,
        "severity": severity,
    }
    actions = [
        {
            "primary_action": {"action_type": "query_metrics", "target_service": "api_gateway", "parameters": {}},
            "buddy_feedback": {"feedback_type": "APPROVE", "rationale": "Start at customer symptom."},
        },
        {
            "primary_action": {"action_type": "read_logs", "target_service": root, "parameters": {}},
            "buddy_feedback": {"feedback_type": "APPROVE", "rationale": "Check upstream logs."},
        },
        {
            "primary_action": {"action_type": "query_metrics", "target_service": root, "parameters": {}},
            "buddy_feedback": {"feedback_type": "APPROVE", "rationale": "Validate metrics on suspected root."},
        },
        {
            "primary_action": {"action_type": "restart_service", "target_service": "api_gateway", "parameters": {}},
            "buddy_feedback": {
                "feedback_type": "SUGGEST_ALTERNATIVE",
                "rationale": (
                    "Gateway is healthy; restarting it causes avoidable damage. "
                    f"Apply {remediation} to {root} instead."
                ),
                "suggested_action": {"action_type": remediation, "target_service": root, "parameters": {}},
                "use_suggestion": True,
                "risk_flags": ["wrong_target", "avoidable_damage"],
            },
        },
        {
            "primary_action": {
                "action_type": "diagnose",
                "target_service": None,
                "parameters": diagnosis_payload,
            },
            "buddy_feedback": {
                "feedback_type": "APPROVE",
                "rationale": "Evidence supports diagnosis.",
                "diagnosis": diagnosis_payload,
            },
        },
    ]
    text = "<think>Buddy redirects.</think>\n<actions>" + json.dumps(actions) + "</actions>"
    return text, scenario_type, root


def _find_seed_for_scenario(scenario_type: str, difficulty: str, base: int = 20260000, span: int = 2000) -> int:
    """Search for a seed that produces a given scenario_type at the requested
    difficulty. Keeps the smoke test deterministic without hard-coding seeds
    that may rotate when scenarios.py changes."""
    for offset in range(span):
        seed = base + offset
        env = CrisisOpsEnv()
        env.reset(seed=seed, difficulty=difficulty)
        scenario = _state(env).scenario
        if scenario and scenario.scenario_type == scenario_type:
            return seed
    raise RuntimeError(f"No seed in [{base}, {base + span}) produced {scenario_type} at {difficulty}")


def _print_block(title: str, payload: Dict[str, Any]) -> None:
    print(f"--- {title} ---")
    for key, value in payload.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    print()


def main() -> int:
    failures: List[str] = []

    leak_seed = _find_seed_for_scenario("memory_leak", "easy")
    leak_text, _, leak_root = build_scripted_completion(leak_seed, "easy")
    leak = run_completion_in_env(leak_text, seed=leak_seed, difficulty="easy")
    _print_block(f"Test 1: scripted memory leak easy (seed={leak_seed}, root={leak_root})", leak)
    if leak["total_reward"] <= 0.55:
        failures.append(f"memory leak reward too low: {leak['total_reward']:.4f}")
    if leak["root_cause_accuracy"] < 1.0:
        failures.append(f"memory leak root_cause_accuracy < 1.0: {leak['root_cause_accuracy']:.2f}")

    bad = run_completion_in_env(malformed_completion(), seed=leak_seed, difficulty="easy")
    _print_block("Test 2: malformed completion (parser graceful failure)", bad)
    if bad["total_reward"] != 0.0:
        failures.append(f"malformed completion should reward 0.0, got {bad['total_reward']:.4f}")
    if not bad["parse_error"]:
        failures.append("malformed completion should report parse_error")

    drift_seed = _find_seed_for_scenario("config_drift", "hard")
    drift_text, _, drift_root = build_scripted_completion(drift_seed, "hard")
    drift = run_completion_in_env(drift_text, seed=drift_seed, difficulty="hard")
    _print_block(f"Test 3: scripted config drift hard (seed={drift_seed}, root={drift_root})", drift)
    if drift["total_reward"] <= 0.55:
        failures.append(f"config drift reward too low: {drift['total_reward']:.4f}")

    pool_seed = _find_seed_for_scenario("connection_pool_exhaustion", "medium")
    pool_text, _, pool_root = build_scripted_completion(pool_seed, "medium")
    pool = run_completion_in_env(pool_text, seed=pool_seed, difficulty="medium")
    _print_block(f"Test 4: scripted connection pool medium (seed={pool_seed}, root={pool_root})", pool)
    if pool["total_reward"] <= 0.55:
        failures.append(f"connection pool reward too low: {pool['total_reward']:.4f}")

    buddy_text, _, buddy_root = build_buddy_corrected_completion(leak_seed, "easy")
    buddy = run_completion_in_env(buddy_text, seed=leak_seed, difficulty="easy")
    _print_block(
        f"Test 5: buddy-corrected memory leak (seed={leak_seed}, root={buddy_root})", buddy
    )
    if buddy["total_reward"] <= 0.55:
        failures.append(f"buddy-corrected reward too low: {buddy['total_reward']:.4f}")
    if buddy["buddy_reward"] < buddy["primary_reward"]:
        failures.append(
            f"buddy reward should >= primary when buddy provides correction: "
            f"primary={buddy['primary_reward']:.4f} buddy={buddy['buddy_reward']:.4f}"
        )

    print("================================================================")
    if failures:
        print(f"FAILED: {len(failures)} assertion(s)")
        for entry in failures:
            print(f"  - {entry}")
        return 1
    print("PASSED: all 5 reward-bridge smoke checks succeeded.")
    print("Reward bridge is wired correctly; safe to launch GRPO training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

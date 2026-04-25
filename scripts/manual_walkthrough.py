"""Manual walkthroughs for validating CrisisOps Phase 3 behavior."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from crisisops_env import Action, BuddyAction, BuddyFeedback, CrisisOpsEnv
from crisisops_env.models import ScenarioSpec, ServiceName


REMEDIATION_BY_SCENARIO: Dict[str, str] = {
    "memory_leak": "restart_service",
    "connection_pool_exhaustion": "drain_connections",
    "cascading_retry_storm": "set_rate_limit",
    "config_drift": "rollback_config",
}


def run_successful_buddy_response(scenario_type: str, seed: int) -> None:
    """Run a buddy-pair policy that investigates, restores, and diagnoses."""

    env = CrisisOpsEnv()
    obs = env.reset(seed=seed, scenario_type=scenario_type)
    scenario = env.state().scenario
    if scenario is None:
        raise RuntimeError("Expected active scenario after reset")
    print(f"\n=== BUDDY SUCCESS: {scenario_type.upper()} ===")
    print(obs.action_result)
    print(
        f"Ground truth for verifier only: {scenario.root_cause} "
        f"severity={scenario.severity} max_steps={scenario.max_steps}"
    )
    for buddy_action in _build_success_policy(scenario):
        obs = env.step(buddy_action)
        executed = buddy_action.executed_action
        print(f"\nStep {obs.step_count}: {executed.action_type}:{executed.target_service}")
        print(obs.action_result)
    print("\nReward:", obs.reward)
    print("Breakdown:", obs.reward_breakdown.model_dump() if obs.reward_breakdown else None)
    print("Evidence:", env.state().discovered_evidence)


def run_buddy_correction_case() -> None:
    """Run a case where buddy overrides a risky primary action."""

    env = CrisisOpsEnv()
    env.reset(seed=21, scenario_type="config_drift")
    scenario = env.state().scenario
    if scenario is None:
        raise RuntimeError("Expected active scenario after reset")
    root = scenario.root_cause_service
    print("\n=== BUDDY CORRECTION: CONFIG DRIFT ===")
    first_step = BuddyAction(
        primary_action=Action(action_type="restart_service", target_service="api_gateway"),
        buddy_feedback=BuddyFeedback(
            feedback_type="SUGGEST_ALTERNATIVE",
            rationale="Restarting gateway is risky before checking the changed service logs.",
            suggested_action=Action(action_type="read_logs", target_service=root),
            use_suggestion=True,
            risk_flags=["restart_without_evidence"],
        ),
    )
    obs = env.step(first_step)
    print(obs.action_result)
    for buddy_action in [
        BuddyAction(
            primary_action=Action(action_type="query_metrics", target_service=root),
            buddy_feedback=BuddyFeedback(
                feedback_type="APPROVE",
                rationale="Metrics confirm the changed service is unhealthy.",
            ),
        ),
        BuddyAction(
            primary_action=Action(action_type="rollback_config", target_service=root),
            buddy_feedback=BuddyFeedback(
                feedback_type="FLAG_RISK",
                rationale="Rollback is risky but matches config mismatch evidence.",
                risk_flags=["config_change"],
            ),
        ),
        _diagnosis_step(scenario),
    ]:
        obs = env.step(buddy_action)
        print(f"\nStep {obs.step_count}: {buddy_action.executed_action.action_type}")
        print(obs.action_result)
    print("\nReward:", obs.reward)
    print("Breakdown:", obs.reward_breakdown.model_dump() if obs.reward_breakdown else None)


def run_bad_primary_buddy_catches() -> None:
    """Run a diagnosis where primary is wrong and buddy is closer."""

    env = CrisisOpsEnv()
    env.reset(seed=9, scenario_type="connection_pool_exhaustion")
    scenario = env.state().scenario
    if scenario is None:
        raise RuntimeError("Expected active scenario after reset")
    obs = env.step(
        BuddyAction(
            primary_action=Action(
                action_type="diagnose",
                parameters={
                    "root_cause": "memory leak in api_gateway",
                    "root_cause_service": "api_gateway",
                    "severity": "P3",
                },
            ),
            buddy_feedback=BuddyFeedback(
                feedback_type="FLAG_RISK",
                rationale="Open connection saturation better explains hanging callers.",
                risk_flags=["primary_diagnosis_under_supported"],
                diagnosis={
                    "root_cause": scenario.root_cause,
                    "root_cause_service": scenario.root_cause_service,
                    "severity": scenario.severity,
                },
            ),
        )
    )
    print("\n=== BUDDY CATCHES BAD PRIMARY DIAGNOSIS ===")
    print(obs.action_result)
    print("Reward:", obs.reward)
    print("Breakdown:", obs.reward_breakdown.model_dump() if obs.reward_breakdown else None)


def _build_success_policy(scenario: ScenarioSpec) -> List[BuddyAction]:
    """Build a scenario-aware scripted policy for manual verification."""

    root = scenario.root_cause_service
    dependency_probe = _dependency_probe_service(scenario)
    actions: List[BuddyAction] = [
        _approved(Action(action_type="query_metrics", target_service=scenario.affected_services[-1])),
        _approved(Action(action_type="query_metrics", target_service=root)),
        _approved(Action(action_type="read_logs", target_service=root)),
    ]
    if dependency_probe is not None:
        actions.append(_approved(Action(action_type="check_dependencies", target_service=dependency_probe)))
    remediation = REMEDIATION_BY_SCENARIO[scenario.scenario_type]
    parameters = (
        {"limit_per_minute": 1200}
        if remediation == "set_rate_limit"
        else {}
    )
    actions.append(
        BuddyAction(
            primary_action=Action(
                action_type=remediation,
                target_service=root,
                parameters=parameters,
            ),
            buddy_feedback=BuddyFeedback(
                feedback_type="FLAG_RISK",
                rationale="This is a state-changing action, but evidence points to the root service.",
                risk_flags=["state_changing_remediation"],
            ),
        )
    )
    actions.append(_diagnosis_step(scenario))
    return actions


def _approved(action: Action) -> BuddyAction:
    """Wrap an action with an approving buddy review."""

    return BuddyAction(
        primary_action=action,
        buddy_feedback=BuddyFeedback(
            feedback_type="APPROVE",
            rationale="Action gathers relevant evidence before remediation.",
        ),
    )


def _diagnosis_step(scenario: ScenarioSpec) -> BuddyAction:
    """Create a paired diagnosis step with independent matching diagnoses."""

    diagnosis = {
        "root_cause": scenario.root_cause,
        "root_cause_service": scenario.root_cause_service,
        "severity": scenario.severity,
    }
    return BuddyAction(
        primary_action=Action(action_type="diagnose", parameters=diagnosis),
        buddy_feedback=BuddyFeedback(
            feedback_type="APPROVE",
            rationale="Evidence and remediation support this diagnosis.",
            diagnosis=diagnosis,
        ),
    )


def _dependency_probe_service(scenario: ScenarioSpec) -> ServiceName | None:
    """Return the service whose dependency check reveals root-cause linkage."""

    for evidence in scenario.required_evidence:
        if evidence.startswith("dependency:"):
            caller = evidence.split(":", 1)[1].split("->", 1)[0]
            return caller  # type: ignore[return-value]
    return None


def main() -> None:
    """Run all manual walkthroughs."""

    run_successful_buddy_response("memory_leak", seed=7)
    run_successful_buddy_response("connection_pool_exhaustion", seed=8)
    run_successful_buddy_response("cascading_retry_storm", seed=10)
    run_successful_buddy_response("config_drift", seed=11)
    run_buddy_correction_case()
    run_bad_primary_buddy_catches()


if __name__ == "__main__":
    main()

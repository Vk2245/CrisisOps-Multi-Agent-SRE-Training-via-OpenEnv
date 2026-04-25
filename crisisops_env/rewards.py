"""Compatibility reward engine backed by the layered judge system."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .judges import LayeredJudgeSystem
from .models import Action, ActionRecord, RewardBreakdown, ScenarioSpec
from .simulator import ServiceSimulator


class RewardEngine:
    """Legacy facade that now delegates reward scoring to layered judges."""

    def __init__(self, max_steps: int = 20) -> None:
        """Initialize the compatibility reward engine."""

        self.max_steps = max_steps
        self.judge_system = LayeredJudgeSystem()

    def compute_total_reward(
        self,
        *,
        scenario: ScenarioSpec,
        simulator: ServiceSimulator,
        action_history: List[ActionRecord],
        diagnosis_action: Optional[Action],
        step_count: int,
        damage_score: int,
        terminal_reason: str,
    ) -> RewardBreakdown:
        """Compute a RewardBreakdown using the Phase 3 judge stack."""

        del damage_score
        judge_result = self.judge_system.score(
            scenario=scenario,
            action_history=action_history,
            damage_log=simulator.damage_log,
            primary_diagnosis=diagnosis_action,
            buddy_diagnosis=None,
            steps_taken=step_count,
            max_steps=self.max_steps,
            time_used=step_count,
            red_herring_visits=simulator.red_herring_visits(action_history),
        )
        judge_scores: Dict[str, float] = judge_result["judge_scores"]
        restoration_score = 1.0 if simulator.is_system_restored() else 0.0
        total = judge_result["primary_reward"]
        notes = list(judge_result["notes"])
        if diagnosis_action is None:
            notes.append("No diagnosis submitted before terminal state.")
            total = min(total, 0.10)
        if diagnosis_action is not None and restoration_score == 0.0:
            notes.append("Diagnosis submitted before full system restoration.")
            total = min(total, 0.55)
        if terminal_reason == "max_steps":
            notes.append("Step budget exhausted before diagnosis.")
        return RewardBreakdown(
            root_cause_accuracy=judge_scores.get("root_cause", 0.0),
            triage_correctness=self._triage_score(scenario, diagnosis_action),
            resolution_efficiency=judge_scores.get("efficiency", 0.0),
            no_collateral_damage=judge_scores.get("damage", 0.0),
            system_restoration=restoration_score,
            process_quality=judge_scores.get("process", 0.0),
            damage_audit=judge_scores.get("damage", 0.0),
            efficiency=judge_scores.get("efficiency", 0.0),
            boss_score=judge_result["boss_score"],
            primary_reward=judge_result["primary_reward"],
            buddy_reward=judge_result["buddy_reward"],
            cooperation_bonus=judge_result["cooperation_bonus"],
            competition_bonus=judge_result["competition_bonus"],
            judge_scores=judge_scores,
            total=round(max(0.0, min(1.0, total)), 4),
            notes=notes,
        )

    def _triage_score(
        self, scenario: ScenarioSpec, diagnosis_action: Optional[Action]
    ) -> float:
        """Score severity accuracy with P/SEV alias support."""

        if diagnosis_action is None:
            return 0.0
        submitted = str(diagnosis_action.parameters.get("severity", "")).lower()
        expected = str(scenario.severity).lower()
        if submitted == expected:
            return 1.0
        aliases: Dict[str, str] = {
            "sev1": "p1",
            "sev2": "p2",
            "sev3": "p3",
            "p1": "p1",
            "p2": "p2",
            "p3": "p3",
        }
        rank = {"p1": 1, "p2": 2, "p3": 3}
        submitted_rank = rank.get(aliases.get(submitted, submitted))
        expected_rank = rank.get(aliases.get(expected, expected))
        if submitted_rank is None or expected_rank is None:
            return 0.0
        return 0.5 if abs(submitted_rank - expected_rank) == 1 else 0.0

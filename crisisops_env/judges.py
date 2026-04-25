"""Layered judge system for CrisisOps trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .models import Action, ActionRecord, Difficulty, ScenarioSpec, ServiceName


RISKY_ACTIONS = {
    "restart_service",
    "scale_service",
    "rollback_config",
    "drain_connections",
    "set_rate_limit",
}
EVIDENCE_ACTIONS = {
    "query_metrics",
    "read_logs",
    "check_dependencies",
    "run_healthcheck",
}
DIFFICULTY_MULTIPLIER: Dict[Difficulty, float] = {
    "easy": 0.70,
    "medium": 1.00,
    "hard": 1.30,
    "expert": 1.50,
}


@dataclass(frozen=True)
class BuddyReward:
    """Primary and buddy rewards after cooperation and competition shaping."""

    primary_reward: float
    buddy_reward: float
    cooperation_bonus: float
    competition_bonus: float
    notes: List[str]


class Judge1_RootCauseVerifier:
    """Binary verification: did the agent find the correct root cause?"""

    weight = 0.35

    def score(self, diagnosis: Optional[Any], ground_truth: ScenarioSpec) -> float:
        """Score root cause accuracy with partial credit for service match."""

        root_cause, service = _diagnosis_parts(diagnosis)
        expected = ground_truth.root_cause.lower()
        expected_service = ground_truth.root_cause_service.lower()
        if not root_cause and not service:
            return 0.0
        if root_cause == expected:
            return 1.0
        names_service = expected_service in root_cause or service == expected_service
        names_failure_mode = _failure_mode_match(root_cause, ground_truth.scenario_type)
        if names_service and names_failure_mode:
            return 1.0
        if names_service:
            return 0.3
        return 0.0


class Judge2_ProcessQuality:
    """Was the diagnostic process logical and evidence-based?"""

    weight = 0.25

    def score(
        self,
        action_history: Sequence[ActionRecord],
        scenario: Optional[ScenarioSpec] = None,
        diagnosis: Optional[Any] = None,
    ) -> float:
        """Score evidence gathering, relevance, non-repetition, and support."""

        if not action_history:
            return 0.0

        evidence_records = [
            record for record in action_history if record.action.action_type in EVIDENCE_ACTIONS
        ]
        risky_records = [
            record for record in action_history if record.action.action_type in RISKY_ACTIONS
        ]
        first_risky_step = min((record.step for record in risky_records), default=None)
        evidence_before_risky = [
            record
            for record in evidence_records
            if first_risky_step is None or record.step < first_risky_step
        ]

        score = 0.20
        score += min(0.30, 0.10 * len(evidence_before_risky))
        score += self._relevance_score(action_history, scenario) * 0.20
        score += self._repeat_penalty_credit(action_history) * 0.15
        score += self._diagnosis_support_score(action_history, scenario, diagnosis) * 0.15
        return _clamp(score)

    def _relevance_score(
        self,
        action_history: Sequence[ActionRecord],
        scenario: Optional[ScenarioSpec],
    ) -> float:
        """Return the ratio of actions targeting affected or dependent services."""

        service_actions = [
            record.action
            for record in action_history
            if record.action.target_service is not None
        ]
        if not service_actions:
            return 0.5
        if scenario is None:
            return 0.7
        relevant = set(scenario.affected_services)
        relevant.add(scenario.root_cause_service)
        hits = sum(
            1 for action in service_actions if action.target_service in relevant
        )
        return hits / len(service_actions)

    def _repeat_penalty_credit(self, action_history: Sequence[ActionRecord]) -> float:
        """Return high credit when the trajectory avoids repeated identical actions."""

        signatures = [_action_signature(record.action) for record in action_history]
        if not signatures:
            return 0.0
        repeated = len(signatures) - len(set(signatures))
        return _clamp(1.0 - repeated / max(1, len(signatures)))

    def _diagnosis_support_score(
        self,
        action_history: Sequence[ActionRecord],
        scenario: Optional[ScenarioSpec],
        diagnosis: Optional[Any],
    ) -> float:
        """Legacy support score; replaced by formal PBRS in LayeredJudgeSystem."""
        return 0.5


class Judge3_DamageAuditor:
    """Did actions cause unnecessary collateral damage?"""

    weight = 0.20

    def score(
        self,
        damage_log: Sequence[str],
        action_history: Optional[Sequence[ActionRecord]] = None,
    ) -> float:
        """Score collateral damage with heavy penalties for avoidable outages."""

        if not damage_log:
            return 1.0
        score = 1.0
        for entry in damage_log:
            lowered = entry.lower()
            if "healthy" in lowered or "wrong" in lowered or "avoidable" in lowered:
                score -= 0.35
            elif "rollback" in lowered or "rate_limit" in lowered:
                score -= 0.25
            else:
                score -= 0.20
        if action_history:
            repeated_restarts = sum(
                1
                for record in action_history
                if record.action.action_type == "restart_service"
            )
            if repeated_restarts > 2:
                score -= 0.15 * (repeated_restarts - 2)
        return _clamp(score)


class Judge4_EfficiencyScorer:
    """Fewer steps and less wasted investigation are better."""

    weight = 0.20

    def score(
        self,
        steps_taken: int,
        max_steps: int,
        time_used: int,
        max_time: int,
        red_herring_visits: int = 0,
    ) -> float:
        """Score step efficiency and red-herring avoidance."""

        if max_steps <= 1 or max_time <= 1:
            return 1.0
        step_score = 1.0 - max(0, steps_taken - 1) / float(max_steps - 1)
        time_score = 1.0 - max(0, time_used - 1) / float(max_time - 1)
        red_herring_penalty = min(0.35, 0.10 * red_herring_visits)
        return _clamp(0.60 * step_score + 0.40 * time_score - red_herring_penalty)


class BossJudge:
    """Meta-judge that calibrates and validates the four judges."""

    def compute_final(
        self,
        judge_scores: Dict[str, float],
        difficulty: Difficulty,
    ) -> float:
        """Apply weighted aggregation, difficulty normalization, and consistency."""

        weighted = (
            Judge1_RootCauseVerifier.weight * judge_scores.get("root_cause", 0.0)
            + Judge2_ProcessQuality.weight * judge_scores.get("process", 0.0)
            + Judge3_DamageAuditor.weight * judge_scores.get("damage", 0.0)
            + Judge4_EfficiencyScorer.weight * judge_scores.get("efficiency", 0.0)
        )
        # Apply Potential-Based Reward Shaping (PBRS)
        pbrs_bonus = judge_scores.get("pbrs", 0.0) * 0.15
        
        multiplier = DIFFICULTY_MULTIPLIER.get(difficulty, 1.0)
        disagreement = _score_disagreement(judge_scores.values())
        consistency_penalty = 0.10 if disagreement > 0.55 else 0.0
        return _clamp((weighted + pbrs_bonus) * multiplier - consistency_penalty)


class LayeredJudgeSystem:
    """Coordinates specialized judges, boss calibration, and buddy shaping."""

    def __init__(self) -> None:
        """Create all judge instances."""

        self.root_cause = Judge1_RootCauseVerifier()
        self.process = Judge2_ProcessQuality()
        self.damage = Judge3_DamageAuditor()
        self.efficiency = Judge4_EfficiencyScorer()
        self.boss = BossJudge()

    def score(
        self,
        *,
        scenario: ScenarioSpec,
        action_history: Sequence[ActionRecord],
        damage_log: Sequence[str],
        primary_diagnosis: Optional[Any],
        buddy_diagnosis: Optional[Any],
        steps_taken: int,
        max_steps: int,
        time_used: int,
        red_herring_visits: int = 0,
    ) -> Dict[str, Any]:
        """Score a full trajectory and return judge plus buddy reward details."""

        primary_root = self.root_cause.score(primary_diagnosis, scenario)
        buddy_root = self.root_cause.score(buddy_diagnosis, scenario)
        primary_scores = {
            "root_cause": primary_root,
            "process": self.process.score(action_history, scenario, primary_diagnosis),
            "damage": self.damage.score(damage_log, action_history),
            "efficiency": self.efficiency.score(
                steps_taken, max_steps, time_used, max_steps, red_herring_visits
            ),
            "pbrs": self._compute_pbrs(action_history, scenario),
        }
        primary_base = self.boss.compute_final(primary_scores, scenario.difficulty)
        
        score_without_buddy = self._score_without_buddy(
            scenario=scenario,
            action_history=action_history,
            damage_log=damage_log,
            primary_diagnosis=primary_diagnosis,
            steps_taken=steps_taken,
            max_steps=max_steps,
            time_used=time_used,
            red_herring_visits=red_herring_visits,
        )

        buddy_reward = self._buddy_rewards(
            primary_base=primary_base,
            score_without_buddy=score_without_buddy,
            primary_root=primary_root,
            buddy_root=buddy_root,
            action_history=action_history,
            primary_diagnosis=primary_diagnosis,
            buddy_diagnosis=buddy_diagnosis,
        )
        return {
            "judge_scores": primary_scores,
            "boss_score": primary_base,
            "primary_reward": buddy_reward.primary_reward,
            "buddy_reward": buddy_reward.buddy_reward,
            "cooperation_bonus": buddy_reward.cooperation_bonus,
            "competition_bonus": buddy_reward.competition_bonus,
            "notes": buddy_reward.notes,
        }

    def _compute_pbrs(
        self, action_history: Sequence[ActionRecord], scenario: ScenarioSpec
    ) -> float:
        """Formal Potential-Based Reward Shaping (PBRS) to guarantee Policy Invariance.
        Phi(s) = fraction of required evidence discovered.
        """
        required = set(scenario.required_evidence)
        if not required:
            return 0.0
        discovered = {
            evidence
            for record in action_history
            for evidence in record.evidence_found
        }
        return len(required.intersection(discovered)) / len(required)

    def _score_without_buddy(
        self,
        *,
        scenario: ScenarioSpec,
        action_history: Sequence[ActionRecord],
        damage_log: Sequence[str],
        primary_diagnosis: Optional[Any],
        steps_taken: int,
        max_steps: int,
        time_used: int,
        red_herring_visits: int,
    ) -> float:
        """Evaluate the counterfactual trajectory to compute Difference Rewards (D_i)."""
        cf_history = []
        cf_damage_log = list(damage_log)
        
        for record in action_history:
            # If buddy suggested an alternative and primary accepted it, counterfactual is the original risky action.
            if record.buddy_feedback and record.buddy_feedback.feedback_type == "SUGGEST_ALTERNATIVE" and record.proposed_action:
                cf_action = record.proposed_action
                if cf_action.action_type in RISKY_ACTIONS:
                    cf_damage_log.append("avoidable damage from counterfactual primary action")
            else:
                cf_action = record.action
                
            cf_history.append(ActionRecord(
                step=record.step,
                action=cf_action,
                result=record.result,
                actor="primary",
                proposed_action=None,
                buddy_feedback=None,
                caused_damage=record.caused_damage,
                evidence_found=record.evidence_found
            ))
            
        cf_scores = {
            "root_cause": self.root_cause.score(primary_diagnosis, scenario),
            "process": self.process.score(cf_history, scenario, primary_diagnosis),
            "damage": self.damage.score(cf_damage_log, cf_history),
            "efficiency": self.efficiency.score(steps_taken, max_steps, time_used, max_steps, red_herring_visits),
            "pbrs": self._compute_pbrs(cf_history, scenario),
        }
        return self.boss.compute_final(cf_scores, scenario.difficulty)

    def _buddy_rewards(
        self,
        *,
        primary_base: float,
        score_without_buddy: float,
        primary_root: float,
        buddy_root: float,
        action_history: Sequence[ActionRecord],
        primary_diagnosis: Optional[Any],
        buddy_diagnosis: Optional[Any],
    ) -> BuddyReward:
        """Compute cooperation and Difference Rewards (D_i) shaping for the buddy pair."""

        notes: List[str] = []
        cooperation_bonus = self._cooperation_bonus(action_history)
        competition_bonus = 0.0
        
        # Difference Reward (D_i) = Score(with buddy) - Score(without buddy)
        difference_reward = primary_base - score_without_buddy
        
        primary_reward = primary_base + cooperation_bonus
        # Formal Multi-Agent Credit Assignment
        buddy_reward = primary_base + difference_reward + cooperation_bonus

        if difference_reward > 0.05:
            notes.append(f"Buddy provided critical Difference Reward contribution (+{difference_reward:.2f}).")

        if primary_diagnosis is not None and buddy_diagnosis is not None:
            if primary_root >= 1.0 and buddy_root >= 1.0:
                notes.append("Both agents agreed on the correct root cause.")
            elif primary_root < buddy_root:
                competition_bonus = 0.05
                buddy_reward += competition_bonus
                primary_reward = max(0.0, primary_reward - 0.05)
                notes.append("Buddy caught a better diagnosis than the primary.")
            elif primary_root > buddy_root:
                competition_bonus = 0.05
                primary_reward += competition_bonus
                buddy_reward = max(0.0, buddy_reward - 0.05)
                notes.append("Primary was right despite weaker buddy diagnosis.")
        elif primary_diagnosis is not None:
            buddy_reward = max(0.0, buddy_reward - 0.05)
            notes.append("Buddy did not submit an independent diagnosis.")

        risk_flags = sum(
            len(record.buddy_feedback.risk_flags)
            for record in action_history
            if record.buddy_feedback is not None
        )
        if risk_flags and primary_root >= 1.0:
            notes.append("Buddy provided risk review during investigation.")

        return BuddyReward(
            primary_reward=_clamp(primary_reward),
            buddy_reward=_clamp(buddy_reward),
            cooperation_bonus=cooperation_bonus,
            competition_bonus=competition_bonus,
            notes=notes,
        )

    def _cooperation_bonus(self, action_history: Sequence[ActionRecord]) -> float:
        """Reward complementary primary/buddy investigation behavior."""

        feedback_records = [
            record for record in action_history if record.buddy_feedback is not None
        ]
        if not feedback_records:
            return 0.0
        useful_feedback = [
            record
            for record in feedback_records
            if record.buddy_feedback is not None
            and (
                record.buddy_feedback.feedback_type != "APPROVE"
                or bool(record.buddy_feedback.rationale)
                or bool(record.buddy_feedback.risk_flags)
            )
        ]
        evidence_count = sum(len(record.evidence_found) for record in action_history)
        if useful_feedback and evidence_count >= 2:
            return 0.05
        if useful_feedback:
            return 0.025
        return 0.0


def prune_trajectories(
    trajectories: Sequence[Dict[str, Any]],
    top_k_ratio: float = 0.3,
    min_batch_size: int = 4,
) -> List[Dict[str, Any]]:
    """Keep top-k% by reward while removing near-duplicate action patterns."""

    if not trajectories:
        return []
    target_count = max(1, int(round(len(trajectories) * top_k_ratio)))
    sorted_trajectories = sorted(
        trajectories,
        key=lambda item: float(item.get("reward", item.get("total_reward", 0.0))),
        reverse=True,
    )
    kept: List[Dict[str, Any]] = []
    seen_prefixes: set[tuple[str, ...]] = set()
    for trajectory in sorted_trajectories:
        prefix = _trajectory_prefix(trajectory)
        if prefix in seen_prefixes:
            continue
        kept.append(trajectory)
        seen_prefixes.add(prefix)
        if len(kept) >= target_count:
            break

    if len(kept) < min_batch_size:
        kept_ids = {id(item) for item in kept}
        for trajectory in sorted_trajectories:
            if id(trajectory) in kept_ids:
                continue
            kept.append(trajectory)
            kept_ids.add(id(trajectory))
            if len(kept) >= min(min_batch_size, len(sorted_trajectories)):
                break
    return kept


def _diagnosis_parts(diagnosis: Optional[Any]) -> tuple[str, str]:
    """Extract root cause and service strings from an action or dict."""

    if diagnosis is None:
        return "", ""
    if isinstance(diagnosis, Action):
        payload = diagnosis.parameters
    elif isinstance(diagnosis, dict):
        payload = diagnosis
    else:
        payload = getattr(diagnosis, "parameters", {})
    root_cause = str(payload.get("root_cause", "")).strip().lower()
    service = str(payload.get("root_cause_service", "")).strip().lower()
    return root_cause, service


def _failure_mode_match(root_cause: str, scenario_type: str) -> bool:
    """Return whether natural-language diagnosis names the failure mode."""

    tokens_by_type: Dict[str, tuple[str, ...]] = {
        "memory_leak": ("memory", "leak"),
        "connection_pool_exhaustion": ("connection", "pool"),
        "cascading_retry_storm": ("retry", "storm"),
        "config_drift": ("config",),
    }
    return all(token in root_cause for token in tokens_by_type.get(scenario_type, ()))


def _score_disagreement(scores: Iterable[float]) -> float:
    """Return max-min disagreement among judge scores."""

    score_list = list(scores)
    if not score_list:
        return 0.0
    return max(score_list) - min(score_list)


def _trajectory_prefix(trajectory: Dict[str, Any]) -> tuple[str, ...]:
    """Return first five action signatures for diversity pruning."""

    raw_actions = trajectory.get("actions", trajectory.get("action_history", []))
    signatures: List[str] = []
    for item in raw_actions[:5]:
        if isinstance(item, ActionRecord):
            signatures.append(_action_signature(item.action))
        elif isinstance(item, Action):
            signatures.append(_action_signature(item))
        elif isinstance(item, dict):
            action = item.get("action", item)
            if isinstance(action, dict):
                signatures.append(
                    f"{action.get('action_type')}:{action.get('target_service')}"
                )
            else:
                signatures.append(str(action))
        else:
            signatures.append(str(item))
    return tuple(signatures)


def _action_signature(action: Action) -> str:
    """Return compact action type and target signature."""

    return f"{action.action_type}:{action.target_service or '-'}"


def _clamp(value: float) -> float:
    """Clamp a float to the reward range."""

    return max(0.0, min(1.0, round(value, 4)))

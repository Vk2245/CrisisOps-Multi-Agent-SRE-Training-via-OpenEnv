"""OpenEnv-compatible CrisisOps environment."""

from __future__ import annotations

from typing import Any, Generic, Optional, TypeVar
from uuid import uuid4

from pydantic import BaseModel, ConfigDict

from .judges import LayeredJudgeSystem
from .models import (
    Action,
    ActionRecord,
    BuddyAction,
    BuddyFeedback,
    Observation,
    RewardBreakdown,
    State,
)
from .scenarios import ScenarioGenerator
from .simulator import AVAILABLE_ACTIONS, ActionOutcome, ServiceSimulator

try:
    from openenv.core.env_server.interfaces import Environment as OpenEnvEnvironment
    from openenv.core.env_server.types import EnvironmentMetadata
except ImportError:
    ActT = TypeVar("ActT")
    ObsT = TypeVar("ObsT")
    StateT = TypeVar("StateT")

    class EnvironmentMetadata(BaseModel):
        """Local fallback for OpenEnv EnvironmentMetadata."""

        name: str
        description: str
        readme_content: Optional[str] = None
        version: Optional[str] = None
        author: Optional[str] = None
        documentation_url: Optional[str] = None
        model_config = ConfigDict(extra="forbid", validate_assignment=True)

    class OpenEnvEnvironment(Generic[ActT, ObsT, StateT]):
        """Local fallback matching the OpenEnv Environment surface."""

        SUPPORTS_CONCURRENT_SESSIONS = False
        REQUIRES_SINGLE_THREAD_EXECUTOR = False

        def __init__(self, transform: Any = None, rubric: Any = None) -> None:
            """Store optional transform and rubric hooks."""

            self.transform = transform
            self.rubric = rubric

        def close(self) -> None:
            """Release environment resources."""

            return None


class CrisisOpsEnv(OpenEnvEnvironment[BuddyAction, Observation, State]):
    """RL environment that trains agents to handle production incidents."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(
        self,
        *,
        max_steps: int = 20,
        scenario_generator: Optional[ScenarioGenerator] = None,
    ) -> None:
        """Create a CrisisOps environment instance."""

        super().__init__()
        self.default_max_steps = max_steps
        self.max_steps = max_steps
        self.scenario_generator = scenario_generator or ScenarioGenerator()
        self.judge_system = LayeredJudgeSystem()
        self.simulator = ServiceSimulator()
        self._episode_id: Optional[str] = None
        self._step_count = 0
        self._done = False
        self._terminal_reason: Optional[str] = None
        self._action_history: list[ActionRecord] = []
        self._discovered_evidence: list[str] = []
        self._damage_score = 0
        self._primary_diagnosis: Optional[dict[str, Any]] = None
        self._buddy_diagnosis: Optional[dict[str, Any]] = None
        self._buddy_mode_used = False
        self._last_reward_breakdown: Optional[RewardBreakdown] = None

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Observation:
        """Start a new incident episode and return the first observation."""

        self._episode_id = episode_id or str(uuid4())
        self._step_count = 0
        self._done = False
        self._terminal_reason = None
        self._action_history = []
        self._discovered_evidence = []
        self._damage_score = 0
        self._primary_diagnosis = None
        self._buddy_diagnosis = None
        self._buddy_mode_used = False
        self._last_reward_breakdown = None
        self._action_counts: dict[str, int] = {}
        self.simulator.reset(seed=seed)
        scenario = self.scenario_generator.generate(
            seed=seed,
            difficulty=kwargs.get("difficulty"),
            scenario_type=kwargs.get("scenario_type"),
        )
        self.max_steps = int(kwargs.get("max_steps", scenario.max_steps or self.default_max_steps))
        self.simulator.apply_scenario(scenario)
        return self._observation(
            action_result=(
                f"PagerDuty page: {scenario.initial_symptoms.get('alert', 'production incident')}. "
                "Primary SRE proposes actions; buddy SRE can approve, suggest, or flag risk. "
                "Restore service, then submit independent diagnoses."
            ),
            reward=0.0,
            done=False,
        )

    def step(
        self,
        action: Action | BuddyAction | dict[str, Any],
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        """Apply one action and return the next observation."""

        del timeout_s, kwargs
        buddy_mode = self._is_buddy_payload(action)
        if buddy_mode:
            buddy_action = (
                action if isinstance(action, BuddyAction) else BuddyAction.model_validate(action)
            )
        else:
            primary_action = (
                action if isinstance(action, Action) else Action.model_validate(action)
            )
            buddy_action = BuddyAction(primary_action=primary_action)
        if self._episode_id is None or self.simulator.active_scenario is None:
            self.reset()
        if self._done:
            return self._observation(
                action_result="Episode is already done. Call reset() for a new incident.",
                reward=self._last_reward_breakdown.total
                if self._last_reward_breakdown
                else 0.0,
                done=True,
                reward_breakdown=self._last_reward_breakdown,
            )

        self._step_count += 1
        self._buddy_mode_used = self._buddy_mode_used or buddy_mode
        executed_action = buddy_action.executed_action
        buddy_feedback = buddy_action.buddy_feedback if buddy_mode else None
        
        signature = f"{executed_action.action_type}:{executed_action.target_service or '-'}"
        self._action_counts[signature] = self._action_counts.get(signature, 0) + 1
        
        # Count-based Exploration Bonus (Intrinsic Reward)
        # Beta = 0.02
        intrinsic_reward = round(0.02 / ((self._action_counts[signature]) ** 0.5), 4)

        if executed_action.action_type == "diagnose":
            self._primary_diagnosis = dict(executed_action.parameters)
            self._buddy_diagnosis = self._extract_buddy_diagnosis(buddy_action)
            outcome = ActionOutcome(result=self._diagnosis_result(buddy_action))
            self._record_action(
                executed_action,
                outcome,
                proposed_action=buddy_action.primary_action,
                buddy_feedback=buddy_feedback,
            )
            self._done = True
            self._terminal_reason = "diagnosed"
            reward_breakdown = self._compute_terminal_reward(executed_action)
            self._last_reward_breakdown = reward_breakdown
            return self._observation(
                action_result=outcome.result,
                reward=reward_breakdown.total + intrinsic_reward,
                done=True,
                reward_breakdown=reward_breakdown,
            )

        outcome = self.simulator.execute(executed_action)
        self._record_action(
            executed_action,
            outcome,
            proposed_action=buddy_action.primary_action,
            buddy_feedback=buddy_feedback,
        )
        self.simulator.advance_time()
        if self._step_count >= self.max_steps:
            self._done = True
            self._terminal_reason = "max_steps"
            reward_breakdown = self._compute_terminal_reward(None)
            self._last_reward_breakdown = reward_breakdown
            return self._observation(
                action_result=(
                    f"{outcome.result}\nStep budget exhausted before diagnosis."
                ),
                reward=reward_breakdown.total + intrinsic_reward,
                done=True,
                reward_breakdown=reward_breakdown,
            )

        return self._observation(
            action_result=self._render_step_result(buddy_action, outcome, buddy_mode),
            reward=intrinsic_reward,
            done=False,
        )

    @property
    def state(self) -> State:
        """Return full environment state, including hidden scenario details."""

        return State(
            episode_id=self._episode_id,
            step_count=self._step_count,
            services=self.simulator.snapshots(),
            scenario=self.simulator.active_scenario,
            recent_alerts=self.simulator.get_recent_alerts(),
            action_history=list(self._action_history),
            discovered_evidence=list(self._discovered_evidence),
            damage_score=self._damage_score,
            damage_log=list(self.simulator.damage_log),
            resolved=self.simulator.is_system_restored(),
            primary_diagnosis=self._primary_diagnosis,
            buddy_diagnosis=self._buddy_diagnosis,
            terminal_reason=self._terminal_reason,
            last_reward_breakdown=self._last_reward_breakdown,
        )

    def get_metadata(self) -> EnvironmentMetadata:
        """Return OpenEnv metadata for discovery and web UI display."""

        return EnvironmentMetadata(
            name="crisisops_env",
            description=(
                "CrisisOps trains LLM agents to diagnose cascading "
                "microservice incidents under partial observability."
            ),
            version="0.1.0",
            author="CrisisOps Hackathon Team",
        )

    def _record_action(
        self,
        action: Action,
        outcome: ActionOutcome,
        *,
        proposed_action: Optional[Action] = None,
        buddy_feedback: Optional[BuddyFeedback] = None,
    ) -> None:
        """Persist action outcome, evidence, and damage tracking."""

        if outcome.caused_damage:
            self._damage_score += 1
            if outcome.damage_reason and outcome.damage_reason not in self.simulator.damage_log:
                self.simulator.damage_log.append(outcome.damage_reason)
        for evidence in outcome.evidence_found:
            if evidence not in self._discovered_evidence:
                self._discovered_evidence.append(evidence)
        self._action_history.append(
            ActionRecord(
                step=self._step_count,
                action=action,
                result=outcome.result,
                actor="primary",
                proposed_action=proposed_action,
                buddy_feedback=buddy_feedback,
                caused_damage=outcome.caused_damage,
                evidence_found=outcome.evidence_found,
            )
        )

    def _compute_terminal_reward(
        self, diagnosis_action: Optional[Action]
    ) -> RewardBreakdown:
        """Compute the reward breakdown for a terminal episode."""

        if self.simulator.active_scenario is None:
            raise RuntimeError("No active scenario to score")
        scenario = self.simulator.active_scenario
        judge_result = self.judge_system.score(
            scenario=self.simulator.active_scenario,
            action_history=self._action_history,
            damage_log=self.simulator.damage_log,
            primary_diagnosis=diagnosis_action,
            buddy_diagnosis=self._buddy_diagnosis,
            steps_taken=self._step_count,
            max_steps=self.max_steps,
            time_used=self._step_count,
            red_herring_visits=self.simulator.red_herring_visits(self._action_history),
        )
        judge_scores = judge_result["judge_scores"]
        triage_score = self._triage_score(diagnosis_action)
        restoration_score = 1.0 if self.simulator.is_system_restored() else 0.0
        team_total = (
            (judge_result["primary_reward"] + judge_result["buddy_reward"]) / 2.0
            if self._buddy_mode_used
            else judge_result["primary_reward"]
        )
        notes = list(judge_result["notes"])
        if diagnosis_action is None:
            notes.append("No primary diagnosis submitted before terminal state.")
            team_total = min(team_total, 0.10)
        if diagnosis_action is not None and restoration_score == 0.0:
            notes.append("Diagnosis submitted before full system restoration.")
            team_total = min(team_total, 0.55)
        if self._terminal_reason == "max_steps":
            notes.append("Step budget exhausted before diagnosis.")
        return RewardBreakdown(
            root_cause_accuracy=judge_scores.get("root_cause", 0.0),
            triage_correctness=triage_score,
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
            total=round(max(0.0, min(1.0, team_total)), 4),
            notes=notes,
        )

    def _diagnosis_result(self, buddy_action: BuddyAction) -> str:
        """Render the submitted diagnosis without revealing correctness."""

        primary = buddy_action.executed_action.parameters
        buddy = self._extract_buddy_diagnosis(buddy_action)
        return (
            f"Primary diagnosis submitted: root_cause={primary.get('root_cause')!r}, "
            f"severity={primary.get('severity')!r}. "
            f"Buddy diagnosis submitted: {buddy or 'none'}. "
            "Episode ended for layered judge scoring."
        )

    def _observation(
        self,
        *,
        action_result: str,
        reward: float,
        done: bool,
        reward_breakdown: Optional[RewardBreakdown] = None,
    ) -> Observation:
        """Build an agent-visible observation."""

        return Observation(
            done=done,
            reward=reward,
            system_overview=self.simulator.get_system_overview(),
            recent_alerts=self.simulator.get_recent_alerts(),
            action_result=action_result,
            step_count=self._step_count,
            time_remaining=max(0, self.max_steps - self._step_count),
            available_actions=list(AVAILABLE_ACTIONS),
            buddy_context={
                "protocol": "primary_action + buddy_feedback per step",
                "feedback_options": ["APPROVE", "SUGGEST_ALTERNATIVE", "FLAG_RISK"],
                "discovered_evidence_count": len(self._discovered_evidence),
                "damage_events": self._damage_score,
            },
            reward_breakdown=reward_breakdown,
        )

    def _is_buddy_payload(self, action: Action | BuddyAction | dict[str, Any]) -> bool:
        """Return whether an input uses the buddy-pair schema."""

        if isinstance(action, BuddyAction):
            return True
        return isinstance(action, dict) and "primary_action" in action

    def _extract_buddy_diagnosis(
        self, buddy_action: BuddyAction
    ) -> Optional[dict[str, Any]]:
        """Extract an independent buddy diagnosis from feedback."""

        feedback = buddy_action.buddy_feedback
        if feedback.diagnosis:
            return dict(feedback.diagnosis)
        if (
            feedback.suggested_action is not None
            and feedback.suggested_action.action_type == "diagnose"
        ):
            return dict(feedback.suggested_action.parameters)
        return None

    def _render_step_result(
        self,
        buddy_action: BuddyAction,
        outcome: ActionOutcome,
        buddy_mode: bool,
    ) -> str:
        """Render primary execution plus buddy review for observations."""

        if not buddy_mode:
            return outcome.result
        feedback = buddy_action.buddy_feedback
        executed = buddy_action.executed_action
        review = (
            f"Buddy review={feedback.feedback_type}; "
            f"rationale={feedback.rationale or 'none'}; "
            f"executed={executed.action_type}:{executed.target_service or '-'}"
        )
        if feedback.risk_flags:
            review += f"; risk_flags={feedback.risk_flags}"
        return f"{review}\n{outcome.result}"

    def _triage_score(self, diagnosis_action: Optional[Action]) -> float:
        """Score severity accuracy against the active scenario."""

        if diagnosis_action is None or self.simulator.active_scenario is None:
            return 0.0
        submitted = str(diagnosis_action.parameters.get("severity", "")).lower()
        expected = str(self.simulator.active_scenario.severity).lower()
        if submitted == expected:
            return 1.0
        aliases = {
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

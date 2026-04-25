"""Pydantic models for the CrisisOps OpenEnv contract."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

try:
    from openenv.core.env_server.types import (
        Action as OpenEnvAction,
        Observation as OpenEnvObservation,
        State as OpenEnvState,
    )
except ImportError:

    class OpenEnvAction(BaseModel):
        """Local fallback for OpenEnv Action when openenv-core is unavailable."""

        metadata: Dict[str, Any] = Field(default_factory=dict)
        model_config = ConfigDict(
            arbitrary_types_allowed=True,
            extra="forbid",
            validate_assignment=True,
        )

    class OpenEnvObservation(BaseModel):
        """Local fallback for OpenEnv Observation when openenv-core is unavailable."""

        done: bool = False
        reward: Optional[float] = 0.0
        metadata: Dict[str, Any] = Field(default_factory=dict)
        model_config = ConfigDict(
            arbitrary_types_allowed=True,
            extra="forbid",
            validate_assignment=True,
        )

    class OpenEnvState(BaseModel):
        """Local fallback for OpenEnv State when openenv-core is unavailable."""

        episode_id: Optional[str] = None
        step_count: int = 0
        model_config = ConfigDict(
            arbitrary_types_allowed=True,
            extra="allow",
            validate_assignment=True,
        )


ServiceName = Literal[
    "api_gateway",
    "auth_service",
    "user_db",
    "order_service",
    "payment_service",
]
HealthState = Literal["healthy", "degraded", "down"]
LogLevel = Literal["info", "warn", "error", "fatal"]
Severity = Literal["sev1", "sev2", "sev3", "sev4", "P1", "P2", "P3"]
Difficulty = Literal["easy", "medium", "hard", "expert"]
FeedbackType = Literal["APPROVE", "SUGGEST_ALTERNATIVE", "FLAG_RISK"]
ActionType = Literal[
    "query_metrics",
    "read_logs",
    "check_dependencies",
    "run_healthcheck",
    "restart_service",
    "scale_service",
    "rollback_config",
    "drain_connections",
    "set_rate_limit",
    "diagnose",
]


class Action(OpenEnvAction):
    """Action submitted by an incident-response agent."""

    action_type: ActionType
    target_service: Optional[ServiceName] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_target_service(self) -> "Action":
        """Ensure service-scoped actions include a target service."""

        service_scoped_actions = {
            "query_metrics",
            "read_logs",
            "run_healthcheck",
            "restart_service",
            "scale_service",
            "rollback_config",
            "drain_connections",
            "set_rate_limit",
        }
        if self.action_type in service_scoped_actions and self.target_service is None:
            raise ValueError(f"{self.action_type} requires target_service")
        if self.action_type == "diagnose":
            missing = {"root_cause", "severity"} - set(self.parameters)
            if missing:
                raise ValueError(
                    "diagnose requires parameters: root_cause and severity"
                )
        return self


class BuddyFeedback(BaseModel):
    """Buddy SRE review of the primary SRE's proposed action."""

    feedback_type: FeedbackType = "APPROVE"
    rationale: str = ""
    suggested_action: Optional[Action] = None
    use_suggestion: bool = False
    risk_flags: List[str] = Field(default_factory=list)
    diagnosis: Optional[Dict[str, Any]] = None

    @model_validator(mode="after")
    def validate_suggestion(self) -> "BuddyFeedback":
        """Require an alternative action when the buddy asks to switch."""

        if self.feedback_type == "SUGGEST_ALTERNATIVE" and self.suggested_action is None:
            raise ValueError("SUGGEST_ALTERNATIVE requires suggested_action")
        return self


class BuddyAction(OpenEnvAction):
    """One environment step containing primary action plus buddy review."""

    primary_action: Action
    buddy_feedback: BuddyFeedback = Field(default_factory=BuddyFeedback)

    @model_validator(mode="before")
    @classmethod
    def wrap_single_action(cls, data: Any) -> Any:
        """Allow legacy single-action payloads to validate as buddy actions."""

        if isinstance(data, Action):
            return {"primary_action": data}
        if isinstance(data, dict) and "primary_action" not in data and "action_type" in data:
            return {"primary_action": data}
        return data

    @property
    def executed_action(self) -> Action:
        """Return the action that should execute after buddy review."""

        if (
            self.buddy_feedback.feedback_type == "SUGGEST_ALTERNATIVE"
            and self.buddy_feedback.use_suggestion
            and self.buddy_feedback.suggested_action is not None
        ):
            return self.buddy_feedback.suggested_action
        return self.primary_action


class ServiceMetrics(BaseModel):
    """Point-in-time service telemetry visible through query_metrics."""

    cpu_percent: float = Field(ge=0.0, le=100.0)
    memory_percent: float = Field(ge=0.0, le=100.0)
    latency_ms: float = Field(ge=0.0)
    error_rate: float = Field(ge=0.0, le=1.0)
    request_count: int = Field(ge=0)
    open_connections: int = Field(ge=0)
    replica_count: int = Field(ge=1)


class LogEntry(BaseModel):
    """Timestamped service log entry."""

    timestamp: str
    service: ServiceName
    level: LogLevel
    message: str


class ServiceSnapshot(BaseModel):
    """Serializable service state for observations and internal state."""

    name: ServiceName
    health: HealthState
    metrics: ServiceMetrics
    dependencies: List[ServiceName] = Field(default_factory=list)
    logs: List[LogEntry] = Field(default_factory=list)
    config_version: str = "v1"
    rate_limit_per_minute: int = Field(default=1200, ge=1)


class ScenarioSpec(BaseModel):
    """Ground-truth incident definition kept out of normal observations."""

    scenario_id: str
    scenario_type: str
    difficulty: Difficulty
    root_cause_service: ServiceName
    root_cause: str
    affected_services: List[ServiceName] = Field(default_factory=list)
    severity: Severity
    initial_symptoms: Dict[str, Any] = Field(default_factory=dict)
    max_steps: int = Field(default=20, ge=1)
    description: str
    required_evidence: List[str] = Field(default_factory=list)
    recommended_actions: List[ActionType] = Field(default_factory=list)
    red_herrings: List[str] = Field(default_factory=list)


class RewardBreakdown(BaseModel):
    """Reward score with legacy fields plus layered judge details."""

    root_cause_accuracy: float = Field(ge=0.0, le=1.0)
    triage_correctness: float = Field(default=0.0, ge=0.0, le=1.0)
    resolution_efficiency: float = Field(default=0.0, ge=0.0, le=1.0)
    no_collateral_damage: float = Field(default=0.0, ge=0.0, le=1.0)
    system_restoration: float = Field(default=0.0, ge=0.0, le=1.0)
    process_quality: float = Field(default=0.0, ge=0.0, le=1.0)
    damage_audit: float = Field(default=0.0, ge=0.0, le=1.0)
    efficiency: float = Field(default=0.0, ge=0.0, le=1.0)
    boss_score: float = Field(default=0.0, ge=0.0, le=1.0)
    primary_reward: float = Field(default=0.0, ge=0.0, le=1.0)
    buddy_reward: float = Field(default=0.0, ge=0.0, le=1.0)
    cooperation_bonus: float = Field(default=0.0, ge=0.0, le=0.1)
    competition_bonus: float = Field(default=0.0, ge=0.0, le=0.1)
    judge_scores: Dict[str, float] = Field(default_factory=dict)
    total: float
    notes: List[str] = Field(default_factory=list)


class ActionRecord(BaseModel):
    """Audit record for an action taken during the episode."""

    step: int
    action: Action
    result: str
    actor: Literal["primary", "buddy", "system"] = "primary"
    proposed_action: Optional[Action] = None
    buddy_feedback: Optional[BuddyFeedback] = None
    caused_damage: bool = False
    evidence_found: List[str] = Field(default_factory=list)


class Observation(OpenEnvObservation):
    """Agent-visible environment observation without ground-truth leakage."""

    system_overview: Dict[ServiceName, HealthState]
    recent_alerts: List[str] = Field(default_factory=list)
    action_result: str = ""
    step_count: int = 0
    time_remaining: int = 20
    available_actions: List[ActionType] = Field(default_factory=list)
    available_feedback: List[FeedbackType] = Field(
        default_factory=lambda: ["APPROVE", "SUGGEST_ALTERNATIVE", "FLAG_RISK"]
    )
    buddy_context: Dict[str, Any] = Field(default_factory=dict)
    reward_breakdown: Optional[RewardBreakdown] = None


class State(OpenEnvState):
    """Full environment state, including hidden ground truth for evaluation."""

    services: Dict[ServiceName, ServiceSnapshot] = Field(default_factory=dict)
    scenario: Optional[ScenarioSpec] = None
    recent_alerts: List[str] = Field(default_factory=list)
    action_history: List[ActionRecord] = Field(default_factory=list)
    discovered_evidence: List[str] = Field(default_factory=list)
    damage_score: int = 0
    damage_log: List[str] = Field(default_factory=list)
    resolved: bool = False
    primary_diagnosis: Optional[Dict[str, Any]] = None
    buddy_diagnosis: Optional[Dict[str, Any]] = None
    terminal_reason: Optional[str] = None
    last_reward_breakdown: Optional[RewardBreakdown] = None

    def __call__(self) -> "State":
        """Return self so both env.state and env.state() work locally."""

        return self

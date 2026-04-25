"""Incident scenario generation for CrisisOps."""

from __future__ import annotations

import random
from typing import Callable, Dict, List, Optional
from uuid import uuid4

from .models import Difficulty, ScenarioSpec, ServiceName


DIFFICULTY_SCENARIOS: Dict[Difficulty, List[str]] = {
    "easy": ["memory_leak"],
    "medium": ["connection_pool_exhaustion", "cascading_retry_storm"],
    "hard": ["config_drift"],
    "expert": [
        "memory_leak",
        "connection_pool_exhaustion",
        "cascading_retry_storm",
        "config_drift",
    ],
}
MAX_STEPS_BY_DIFFICULTY: Dict[Difficulty, int] = {
    "easy": 20,
    "medium": 18,
    "hard": 16,
    "expert": 14,
}
SERVICE_UPSTREAMS: Dict[ServiceName, List[ServiceName]] = {
    "api_gateway": [],
    "auth_service": ["api_gateway"],
    "user_db": ["auth_service", "api_gateway"],
    "order_service": ["api_gateway"],
    "payment_service": ["order_service", "api_gateway"],
}


class ScenarioGenerator:
    """Procedurally generates incident scenarios with configurable difficulty."""

    def __init__(self, default_difficulty: Optional[Difficulty] = None) -> None:
        """Initialize the generator with optional curriculum difficulty."""

        self.default_difficulty = default_difficulty
        self._builders: Dict[str, Callable[[random.Random, Difficulty], ScenarioSpec]] = {
            "memory_leak": self._memory_leak,
            "connection_pool_exhaustion": self._connection_pool_exhaustion,
            "cascading_retry_storm": self._cascading_retry_storm,
            "config_drift": self._config_drift,
        }

    def generate(
        self,
        *,
        seed: Optional[int] = None,
        difficulty: Optional[Difficulty] = None,
        scenario_type: Optional[str] = None,
    ) -> ScenarioSpec:
        """Generate a scenario, filtering by difficulty when requested."""

        rng = random.Random(seed)
        selected_difficulty = difficulty or self.default_difficulty
        if scenario_type is not None:
            if scenario_type not in self._builders:
                raise ValueError(f"Unsupported scenario_type: {scenario_type}")
            scenario_difficulty = selected_difficulty or self._difficulty_for(scenario_type)
            return self._builders[scenario_type](rng, scenario_difficulty)

        available = (
            DIFFICULTY_SCENARIOS[selected_difficulty]
            if selected_difficulty is not None
            else list(self._builders)
        )
        selected_type = rng.choice(available)
        scenario_difficulty = selected_difficulty or self._difficulty_for(selected_type)
        return self._builders[selected_type](rng, scenario_difficulty)

    def _memory_leak(self, rng: random.Random, difficulty: Difficulty) -> ScenarioSpec:
        """Create a memory saturation scenario with upstream cascade."""

        root = rng.choice(["user_db", "auth_service", "payment_service"])
        affected = self._affected_services(root)
        leak_source = rng.choice(["heap growth", "cache bloat", "allocator churn"])
        severity = rng.choice(["P2", "P2", "P3"])
        return ScenarioSpec(
            scenario_id=f"memleak-{uuid4().hex[:8]}",
            scenario_type="memory_leak",
            difficulty=difficulty,
            root_cause_service=root,
            root_cause=f"memory_leak:{root}",
            affected_services=affected,
            severity=severity,
            initial_symptoms={
                "trigger_minute": rng.randint(6, 14),
                "alert": f"{affected[-1]} elevated latency and 5xx errors",
                "memory_percent": rng.randint(90, 98),
            },
            max_steps=MAX_STEPS_BY_DIFFICULTY[difficulty],
            description=(
                f"{root} has {leak_source}; upstream services degrade as calls "
                "timeout and retries pile up."
            ),
            required_evidence=[
                f"metric:{root}:memory_saturation",
                f"log:{root}:heap_growth",
                self._dependency_evidence(root, affected),
            ],
            recommended_actions=["query_metrics", "read_logs", "restart_service"],
            red_herrings=self._sample_red_herrings(
                rng,
                [
                    "payment_service saw a one-minute card network jitter spike",
                    "order_service retry queue drained after a short deploy window",
                    "api_gateway access logs show a noisy crawler already rate limited",
                ],
            ),
        )

    def _connection_pool_exhaustion(
        self, rng: random.Random, difficulty: Difficulty
    ) -> ScenarioSpec:
        """Create a connection pool exhaustion scenario."""

        root = rng.choice(["payment_service", "user_db", "auth_service"])
        affected = self._affected_services(root)
        severity = rng.choice(["P1", "P2", "P2"])
        return ScenarioSpec(
            scenario_id=f"connpool-{uuid4().hex[:8]}",
            scenario_type="connection_pool_exhaustion",
            difficulty=difficulty,
            root_cause_service=root,
            root_cause=f"connection_pool_exhaustion:{root}",
            affected_services=affected,
            severity=severity,
            initial_symptoms={
                "trigger_minute": rng.randint(4, 10),
                "alert": f"{root} max open connections reached",
                "open_connections": rng.randint(480, 650),
            },
            max_steps=MAX_STEPS_BY_DIFFICULTY[difficulty],
            description=(
                f"{root} exhausted its connection pool, causing callers to hang "
                "and queue requests."
            ),
            required_evidence=[
                f"metric:{root}:connection_saturation",
                f"log:{root}:pool_exhausted",
                self._dependency_evidence(root, affected),
            ],
            recommended_actions=[
                "query_metrics",
                "read_logs",
                "drain_connections",
                "scale_service",
            ],
            red_herrings=self._sample_red_herrings(
                rng,
                [
                    "user_db vacuum job completed normally before the alert",
                    "api_gateway TLS certificate renewal succeeded with no errors",
                    "auth_service warnings mention expired sessions from test traffic",
                ],
            ),
        )

    def _cascading_retry_storm(
        self, rng: random.Random, difficulty: Difficulty
    ) -> ScenarioSpec:
        """Create a retry storm started by rate limiting or overload."""

        root = rng.choice(["auth_service", "order_service", "api_gateway"])
        affected = self._affected_services(root)
        if root != "api_gateway" and "api_gateway" not in affected:
            affected.append("api_gateway")
        severity = rng.choice(["P1", "P2"])
        return ScenarioSpec(
            scenario_id=f"retrystorm-{uuid4().hex[:8]}",
            scenario_type="cascading_retry_storm",
            difficulty=difficulty,
            root_cause_service=root,
            root_cause=f"cascading_retry_storm:{root}",
            affected_services=affected,
            severity=severity,
            initial_symptoms={
                "trigger_minute": rng.randint(3, 9),
                "alert": f"{root} retry volume caused CPU saturation",
                "cpu_percent": rng.randint(88, 99),
                "retry_multiplier": rng.choice([3, 5, 8]),
            },
            max_steps=MAX_STEPS_BY_DIFFICULTY[difficulty],
            description=(
                f"{root} began returning throttles/timeouts; upstream clients "
                "retried aggressively and amplified load."
            ),
            required_evidence=[
                f"metric:{root}:cpu_saturation",
                f"log:{root}:retry_storm",
                f"rate_limit:{root}:throttling",
            ],
            recommended_actions=[
                "query_metrics",
                "read_logs",
                "set_rate_limit",
                "scale_service",
            ],
            red_herrings=self._sample_red_herrings(
                rng,
                [
                    "payment_service fraud rule refresh produced noisy warnings",
                    "user_db checkpoint latency rose briefly but stayed under SLO",
                    "order_service deploy marker appears near the incident window",
                ],
            ),
        )

    def _config_drift(self, rng: random.Random, difficulty: Difficulty) -> ScenarioSpec:
        """Create a hard config drift scenario with misleading logs."""

        root = rng.choice(["auth_service", "order_service", "payment_service"])
        affected = self._affected_services(root)
        severity = rng.choice(["P1", "P2"])
        config_key = rng.choice(["SECRET_KEY", "PAYMENT_REGION", "JWT_AUDIENCE"])
        return ScenarioSpec(
            scenario_id=f"configdrift-{uuid4().hex[:8]}",
            scenario_type="config_drift",
            difficulty=difficulty,
            root_cause_service=root,
            root_cause=f"config_drift:{root}",
            affected_services=affected,
            severity=severity,
            initial_symptoms={
                "trigger_minute": rng.randint(1, 7),
                "alert": f"{root} error rate spiked after config rollout",
                "config_key": config_key,
                "error_rate": round(rng.uniform(0.18, 0.38), 2),
            },
            max_steps=MAX_STEPS_BY_DIFFICULTY[difficulty],
            description=(
                f"{root} is running with drifted {config_key}; symptoms mimic "
                "dependency failure but rollback is the correct fix."
            ),
            required_evidence=[
                f"log:{root}:config_mismatch",
                f"metric:{root}:error_spike",
                f"config:{root}:drift_detected",
            ],
            recommended_actions=["read_logs", "query_metrics", "rollback_config"],
            red_herrings=self._sample_red_herrings(
                rng,
                [
                    "user_db reports slow query warnings from an unrelated analytics job",
                    "payment_service logs card decline errors from synthetic tests",
                    "api_gateway logs show client disconnects from mobile network churn",
                    "order_service worker restart completed successfully after deploy",
                ],
            ),
        )

    def _difficulty_for(self, scenario_type: str) -> Difficulty:
        """Return the natural curriculum difficulty for a scenario type."""

        for difficulty, scenario_types in DIFFICULTY_SCENARIOS.items():
            if difficulty == "expert":
                continue
            if scenario_type in scenario_types:
                return difficulty
        return "expert"

    def _affected_services(self, root: ServiceName) -> List[ServiceName]:
        """Return root plus upstream services impacted by the incident."""

        affected = [root]
        for upstream in SERVICE_UPSTREAMS[root]:
            if upstream not in affected:
                affected.append(upstream)
        return affected

    def _sample_red_herrings(
        self, rng: random.Random, candidates: List[str]
    ) -> List[str]:
        """Sample a non-empty red-herring subset for noisy observability."""

        count = rng.randint(1, min(3, len(candidates)))
        return rng.sample(candidates, count)

    def _dependency_evidence(
        self, root: ServiceName, affected: List[ServiceName]
    ) -> str:
        """Return direct dependency evidence for the root service."""

        if len(affected) > 1:
            return f"dependency:{affected[1]}->{root}"
        return f"health:{root}:degraded"

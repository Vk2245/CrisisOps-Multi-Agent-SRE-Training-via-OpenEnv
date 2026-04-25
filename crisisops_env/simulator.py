"""Deterministic microservice simulator for CrisisOps incidents."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from .models import (
    Action,
    ActionType,
    HealthState,
    LogEntry,
    LogLevel,
    ServiceMetrics,
    ServiceName,
    ServiceSnapshot,
    ScenarioSpec,
)


SERVICE_DEPENDENCIES: Dict[ServiceName, List[ServiceName]] = {
    "api_gateway": ["auth_service", "order_service"],
    "auth_service": ["user_db"],
    "user_db": [],
    "order_service": ["payment_service"],
    "payment_service": [],
}

AVAILABLE_ACTIONS: List[ActionType] = [
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


class ServiceRuntime(BaseModel):
    """Mutable service state used internally by the simulator."""

    name: ServiceName
    health: HealthState = "healthy"
    metrics: ServiceMetrics
    dependencies: List[ServiceName] = Field(default_factory=list)
    logs: List[LogEntry] = Field(default_factory=list)
    config_version: str = "v1"
    rate_limit_per_minute: int = 1200
    restart_count: int = 0


@dataclass
class ActionOutcome:
    """Result returned by the simulator after an action."""

    result: str
    caused_damage: bool = False
    evidence_found: List[str] = field(default_factory=list)
    damage_reason: Optional[str] = None


class ServiceSimulator:
    """Simulates five dependent production services and their telemetry."""

    def __init__(self, seed: Optional[int] = None) -> None:
        """Create a simulator with deterministic randomness."""

        self.rng = random.Random(seed)
        self.services: Dict[ServiceName, ServiceRuntime] = {}
        self.recent_alerts: List[str] = []
        self.damage_log: List[str] = []
        self.active_scenario: Optional[ScenarioSpec] = None
        self.elapsed_minutes = 0
        self.reset(seed=seed)

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset all services to a healthy baseline."""

        if seed is not None:
            self.rng.seed(seed)
        self.elapsed_minutes = 0
        self.recent_alerts = []
        self.damage_log = []
        self.active_scenario = None
        self.services = {
            "api_gateway": self._service(
                "api_gateway",
                cpu=35,
                memory=42,
                latency=95,
                error_rate=0.01,
                request_count=2400,
                open_connections=180,
                replicas=4,
            ),
            "auth_service": self._service(
                "auth_service",
                cpu=30,
                memory=45,
                latency=80,
                error_rate=0.005,
                request_count=1400,
                open_connections=95,
                replicas=3,
            ),
            "user_db": self._service(
                "user_db",
                cpu=28,
                memory=48,
                latency=45,
                error_rate=0.002,
                request_count=900,
                open_connections=70,
                replicas=2,
            ),
            "order_service": self._service(
                "order_service",
                cpu=32,
                memory=40,
                latency=105,
                error_rate=0.006,
                request_count=1200,
                open_connections=110,
                replicas=3,
            ),
            "payment_service": self._service(
                "payment_service",
                cpu=25,
                memory=38,
                latency=130,
                error_rate=0.004,
                request_count=850,
                open_connections=60,
                replicas=2,
            ),
        }
        for service in self.services:
            self._add_log(service, "info", "service healthy after baseline reset")

    def apply_scenario(self, scenario: ScenarioSpec) -> None:
        """Inject a scenario into the simulated infrastructure."""

        self.active_scenario = scenario
        if scenario.scenario_type == "memory_leak":
            self._apply_memory_leak(scenario)
        elif scenario.scenario_type == "connection_pool_exhaustion":
            self._apply_connection_pool_exhaustion(scenario)
        elif scenario.scenario_type == "cascading_retry_storm":
            self._apply_retry_storm(scenario)
        elif scenario.scenario_type == "config_drift":
            self._apply_config_drift(scenario)
        else:
            raise ValueError(f"Unsupported scenario: {scenario.scenario_type}")
        self._inject_red_herrings(scenario)

    def execute(self, action: Action) -> ActionOutcome:
        """Execute a non-diagnosis action against the simulated services."""

        if action.action_type == "query_metrics":
            return self.get_metrics(action.target_service)
        if action.action_type == "read_logs":
            return self.get_logs(action.target_service, action.parameters)
        if action.action_type == "check_dependencies":
            return self.check_dependencies(action.target_service)
        if action.action_type == "run_healthcheck":
            return self.run_healthcheck(action.target_service)
        if action.action_type == "restart_service":
            return self.restart_service(action.target_service)
        if action.action_type == "scale_service":
            return self.scale_service(action.target_service)
        if action.action_type == "rollback_config":
            return self.rollback_config(action.target_service)
        if action.action_type == "drain_connections":
            return self.drain_connections(action.target_service)
        if action.action_type == "set_rate_limit":
            return self.set_rate_limit(action.target_service, action.parameters)
        return ActionOutcome(result=f"{action.action_type} is handled by the env")

    def get_metrics(self, service_name: Optional[ServiceName]) -> ActionOutcome:
        """Return visible metrics for one service."""

        service = self._get_service(service_name)
        metrics = service.metrics
        evidence = self._metric_evidence(service)
        result = (
            f"{service.name} metrics: cpu={metrics.cpu_percent:.1f}%, "
            f"mem={metrics.memory_percent:.1f}%, latency={metrics.latency_ms:.0f}ms, "
            f"error_rate={metrics.error_rate:.3f}, reqs={metrics.request_count}, "
            f"open_conns={metrics.open_connections}, replicas={metrics.replica_count}"
        )
        return ActionOutcome(result=result, evidence_found=evidence)

    def get_logs(
        self, service_name: Optional[ServiceName], parameters: Dict[str, object]
    ) -> ActionOutcome:
        """Return recent logs for one service."""

        service = self._get_service(service_name)
        limit = int(parameters.get("limit", 5))
        logs = service.logs[-limit:]
        evidence = self._log_evidence(service, logs)
        rendered = [
            f"{entry.timestamp} {entry.level.upper()} {entry.service}: {entry.message}"
            for entry in logs
        ]
        return ActionOutcome(result="\n".join(rendered), evidence_found=evidence)

    def check_dependencies(
        self, service_name: Optional[ServiceName] = None
    ) -> ActionOutcome:
        """Return dependency graph information."""

        if service_name is None:
            graph = ", ".join(
                f"{service}->{deps or ['none']}"
                for service, deps in SERVICE_DEPENDENCIES.items()
            )
            return ActionOutcome(result=f"Dependency graph: {graph}")
        service = self._get_service(service_name)
        upstreams = [
            parent
            for parent, deps in SERVICE_DEPENDENCIES.items()
            if service.name in deps
        ]
        evidence = self._dependency_evidence(service)
        result = (
            f"{service.name} depends_on={service.dependencies or ['none']}; "
            f"upstream_callers={upstreams or ['none']}"
        )
        return ActionOutcome(result=result, evidence_found=evidence)

    def run_healthcheck(self, service_name: Optional[ServiceName]) -> ActionOutcome:
        """Return a synthetic healthcheck result for one service."""

        service = self._get_service(service_name)
        if service.health == "healthy":
            status = "200 OK"
        elif service.health == "degraded":
            status = "200 OK but dependency checks are degraded"
        else:
            status = "503 unavailable"
        evidence = []
        if self.active_scenario and service.name in self.active_scenario.affected_services:
            evidence.append(f"health:{service.name}:{service.health}")
        return ActionOutcome(
            result=f"{service.name} healthcheck: {status}",
            evidence_found=evidence,
        )

    def restart_service(self, service_name: Optional[ServiceName]) -> ActionOutcome:
        """Restart a service, possibly restoring or damaging the system."""

        service = self._get_service(service_name)
        service.restart_count += 1
        self._add_log(service.name, "warn", "operator requested service restart")
        if self._is_root_scenario(service.name, "memory_leak"):
            service.metrics.memory_percent = 46.0
            service.metrics.cpu_percent = 32.0
            service.metrics.latency_ms = 55.0
            service.metrics.error_rate = 0.002
            service.metrics.open_connections = 72
            self._recover_scenario(
                evidence=f"mitigation:{service.name}:restart_recovered",
                result=(
                    f"restart_service {service.name}: heap cleared and upstream "
                    "dependency checks returned healthy"
                ),
            )
            return ActionOutcome(
                result=(
                    f"restart_service {service.name}: heap cleared and upstream "
                    "dependency checks returned healthy"
                ),
                evidence_found=[f"mitigation:{service.name}:restart_recovered"],
            )

        reason = (
            f"wrong restart on {service.name}: disrupted capacity while root cause "
            "remained active"
        )
        service.health = "degraded"
        service.metrics.error_rate = min(1.0, service.metrics.error_rate + 0.04)
        service.metrics.latency_ms += 80
        self._add_log(service.name, "error", reason)
        self.damage_log.append(reason)
        return ActionOutcome(
            result=(
                f"restart_service {service.name}: service restarted but symptoms "
                "returned because the root cause was not fixed"
            ),
            caused_damage=True,
            damage_reason=reason,
        )

    def scale_service(self, service_name: Optional[ServiceName]) -> ActionOutcome:
        """Scale a service replica count as a mitigation action."""

        service = self._get_service(service_name)
        service.metrics.replica_count += 1
        self._add_log(service.name, "info", "operator scaled replicas by +1")
        if self._is_root_scenario(service.name, "cascading_retry_storm"):
            service.metrics.cpu_percent = max(55.0, service.metrics.cpu_percent - 22)
            service.metrics.error_rate = max(0.05, service.metrics.error_rate - 0.08)
            return ActionOutcome(
                result=(
                    f"scale_service {service.name}: CPU pressure reduced, but "
                    "retry throttling policy still needs correction"
                ),
                evidence_found=[f"mitigation:{service.name}:scale_reduced_retry_load"],
            )
        if self._is_root_scenario(service.name, "connection_pool_exhaustion"):
            service.metrics.open_connections = max(220, service.metrics.open_connections - 160)
            service.metrics.latency_ms = max(170.0, service.metrics.latency_ms - 180)
            return ActionOutcome(
                result=(
                    f"scale_service {service.name}: added pool capacity; drain "
                    "connections to fully clear stuck callers"
                ),
                evidence_found=[f"mitigation:{service.name}:scale_added_pool_capacity"],
            )
        if self._is_root_scenario(service.name, "memory_leak"):
            service.metrics.memory_percent = max(78.0, service.metrics.memory_percent - 12)
            service.metrics.latency_ms = max(120.0, service.metrics.latency_ms - 110)
            return ActionOutcome(
                result=(
                    f"scale_service {service.name}: pressure reduced but memory "
                    "leak still requires restart or rollback"
                ),
                evidence_found=[f"mitigation:{service.name}:scale_reduced_pressure"],
            )
        reason = f"unnecessary scale on {service.name}: capacity changed without fixing root cause"
        self.damage_log.append(reason)
        return ActionOutcome(
            result=f"scale_service {service.name}: capacity increased, alerts did not clear",
            caused_damage=True,
            damage_reason=reason,
        )

    def rollback_config(self, service_name: Optional[ServiceName]) -> ActionOutcome:
        """Rollback a service config version."""

        service = self._get_service(service_name)
        if self._is_root_scenario(service.name, "config_drift"):
            service.config_version = "v_previous"
            self._recover_scenario(
                evidence=f"mitigation:{service.name}:config_rollback_recovered",
                result=f"rollback_config {service.name}: drifted config restored",
            )
            return ActionOutcome(
                result=f"rollback_config {service.name}: drifted config restored",
                evidence_found=[f"mitigation:{service.name}:config_rollback_recovered"],
            )
        reason = f"avoidable rollback on {service.name}: no config drift found"
        self._add_log(service.name, "error", reason)
        self.damage_log.append(reason)
        return ActionOutcome(
            result=(
                f"rollback_config {service.name}: no config drift found; rollback "
                "introduced avoidable risk"
            ),
            caused_damage=True,
            damage_reason=reason,
        )

    def drain_connections(self, service_name: Optional[ServiceName]) -> ActionOutcome:
        """Drain connection pools on a service."""

        service = self._get_service(service_name)
        before = service.metrics.open_connections
        service.metrics.open_connections = max(10, before // 2)
        if self._is_root_scenario(service.name, "connection_pool_exhaustion"):
            self._recover_scenario(
                evidence=f"mitigation:{service.name}:connections_drained",
                result=f"drain_connections {service.name}: pool pressure cleared",
            )
            return ActionOutcome(
                result=(
                    f"drain_connections {service.name}: open connections "
                    f"{before}->{service.metrics.open_connections}; callers recovered"
                ),
                evidence_found=[f"mitigation:{service.name}:connections_drained"],
            )
        return ActionOutcome(
            result=(
                f"drain_connections {service.name}: open connections "
                f"{before}->{service.metrics.open_connections}"
            )
        )

    def set_rate_limit(
        self, service_name: Optional[ServiceName], parameters: Dict[str, object]
    ) -> ActionOutcome:
        """Set a per-minute rate limit for a service."""

        service = self._get_service(service_name)
        new_limit = int(parameters.get("limit_per_minute", service.rate_limit_per_minute))
        service.rate_limit_per_minute = new_limit
        if self._is_root_scenario(service.name, "cascading_retry_storm"):
            if 500 <= new_limit <= 1800:
                service.metrics.cpu_percent = 45.0
                service.metrics.error_rate = 0.015
                self._recover_scenario(
                    evidence=f"mitigation:{service.name}:retry_rate_limit_fixed",
                    result=(
                        f"set_rate_limit {service.name}: retry storm dampened and "
                        "upstream services recovered"
                    ),
                )
                return ActionOutcome(
                    result=(
                        f"set_rate_limit {service.name}: retry storm dampened and "
                        "upstream services recovered"
                    ),
                    evidence_found=[f"mitigation:{service.name}:retry_rate_limit_fixed"],
                )
            reason = f"wrong rate_limit on {service.name}: limit {new_limit}/min amplified errors"
            self.damage_log.append(reason)
            return ActionOutcome(
                result=(
                    f"set_rate_limit {service.name}: limit {new_limit}/min was "
                    "too aggressive and increased throttling"
                ),
                caused_damage=True,
                damage_reason=reason,
            )
        if service.name == "api_gateway" and new_limit < 500:
            service.metrics.error_rate = min(1.0, service.metrics.error_rate + 0.08)
            reason = "wrong rate_limit on api_gateway: legitimate traffic throttled"
            self.damage_log.append(reason)
            return ActionOutcome(
                result=(
                    "set_rate_limit api_gateway: limit too low; legitimate traffic "
                    "is now being throttled"
                ),
                caused_damage=True,
                damage_reason=reason,
            )
        return ActionOutcome(
            result=f"set_rate_limit {service.name}: limit set to {new_limit}/min"
        )

    def advance_time(self) -> None:
        """Advance the simulated incident by one minute."""

        self.elapsed_minutes += 1
        scenario = self.active_scenario
        if not scenario or self.is_system_restored():
            return
        root = self.services[scenario.root_cause_service]
        if scenario.scenario_type == "memory_leak":
            root.metrics.memory_percent = min(
                100.0, root.metrics.memory_percent + self.rng.randint(1, 3)
            )
            if root.metrics.memory_percent >= 99:
                self._degrade_services(scenario.affected_services, down=True)
                self.recent_alerts.append(
                    f"{scenario.severity}: {root.name} OOM caused upstream outage"
                )
        elif scenario.scenario_type == "connection_pool_exhaustion":
            root.metrics.open_connections += self.rng.randint(15, 45)
            root.metrics.latency_ms += self.rng.randint(20, 60)
        elif scenario.scenario_type == "cascading_retry_storm":
            root.metrics.cpu_percent = min(100.0, root.metrics.cpu_percent + 2)
            root.metrics.request_count += self.rng.randint(150, 450)
        elif scenario.scenario_type == "config_drift":
            root.metrics.error_rate = min(1.0, root.metrics.error_rate + 0.02)

    def get_system_overview(self) -> Dict[ServiceName, HealthState]:
        """Return service health visible to the agent."""

        return {name: service.health for name, service in self.services.items()}

    def get_recent_alerts(self) -> List[str]:
        """Return recent alert strings visible to the agent."""

        return self.recent_alerts[-5:]

    def is_system_restored(self) -> bool:
        """Return whether all services have recovered to healthy state."""

        return all(service.health == "healthy" for service in self.services.values())

    def red_herring_visits(self, action_history: List[object]) -> int:
        """Count actions that targeted services outside the affected scenario."""

        if not self.active_scenario:
            return 0
        affected = set(self.active_scenario.affected_services)
        count = 0
        for item in action_history:
            action = getattr(item, "action", item)
            target = getattr(action, "target_service", None)
            if target is not None and target not in affected:
                count += 1
        return count

    def snapshots(self) -> Dict[ServiceName, ServiceSnapshot]:
        """Return serializable service snapshots for state inspection."""

        return {
            name: ServiceSnapshot(
                name=service.name,
                health=service.health,
                metrics=service.metrics.model_copy(deep=True),
                dependencies=list(service.dependencies),
                logs=list(service.logs[-8:]),
                config_version=service.config_version,
                rate_limit_per_minute=service.rate_limit_per_minute,
            )
            for name, service in self.services.items()
        }

    def _apply_memory_leak(self, scenario: ScenarioSpec) -> None:
        """Inject memory leak telemetry and logs."""

        root = self.services[scenario.root_cause_service]
        root.health = "degraded"
        root.metrics.memory_percent = float(scenario.initial_symptoms["memory_percent"])
        root.metrics.cpu_percent = float(self.rng.randint(66, 82))
        root.metrics.latency_ms = float(self.rng.randint(260, 520))
        root.metrics.error_rate = 0.06
        root.metrics.open_connections = self.rng.randint(120, 180)
        self._add_log(root.name, "warn", "heap growth detected; resident memory rising")
        self._add_log(root.name, "error", "workers stalled waiting for memory compaction")
        self._degrade_upstream_callers(scenario, f"timeout calling {root.name}")
        self.recent_alerts = [
            f"{scenario.severity}: {scenario.initial_symptoms['alert']}",
            f"{root.name} memory above {scenario.initial_symptoms['memory_percent']} percent",
        ]

    def _apply_connection_pool_exhaustion(self, scenario: ScenarioSpec) -> None:
        """Inject connection pool exhaustion telemetry and logs."""

        root = self.services[scenario.root_cause_service]
        root.health = "degraded"
        root.metrics.open_connections = int(scenario.initial_symptoms["open_connections"])
        root.metrics.latency_ms = float(self.rng.randint(700, 1300))
        root.metrics.error_rate = 0.14
        root.metrics.cpu_percent = float(self.rng.randint(58, 74))
        self._add_log(root.name, "error", "connection pool exhausted; callers blocked")
        self._add_log(root.name, "warn", "pool wait queue exceeded safe threshold")
        self._degrade_upstream_callers(scenario, f"hanging on pooled calls to {root.name}")
        self.recent_alerts = [
            f"{scenario.severity}: {scenario.initial_symptoms['alert']}",
            f"{root.name} open connections above pool limit",
        ]

    def _apply_retry_storm(self, scenario: ScenarioSpec) -> None:
        """Inject cascading retry storm telemetry and logs."""

        root = self.services[scenario.root_cause_service]
        root.health = "degraded"
        root.metrics.cpu_percent = float(scenario.initial_symptoms["cpu_percent"])
        root.metrics.request_count *= int(scenario.initial_symptoms["retry_multiplier"])
        root.metrics.error_rate = 0.21
        root.metrics.latency_ms = float(self.rng.randint(480, 900))
        self._add_log(root.name, "warn", "429 throttling responses triggered retry storm")
        self._add_log(root.name, "error", "retry storm amplified CPU and queue depth")
        self._degrade_services(scenario.affected_services)
        self.recent_alerts = [
            f"{scenario.severity}: {scenario.initial_symptoms['alert']}",
            f"{root.name} retry multiplier {scenario.initial_symptoms['retry_multiplier']}x",
        ]

    def _apply_config_drift(self, scenario: ScenarioSpec) -> None:
        """Inject config drift telemetry and misleading symptoms."""

        root = self.services[scenario.root_cause_service]
        root.health = "degraded"
        root.config_version = "v_bad"
        root.metrics.error_rate = float(scenario.initial_symptoms["error_rate"])
        root.metrics.latency_ms = float(self.rng.randint(300, 760))
        root.metrics.cpu_percent = float(self.rng.randint(44, 70))
        self._add_log(
            root.name,
            "error",
            f"config mismatch detected for {scenario.initial_symptoms['config_key']}",
        )
        self._add_log(root.name, "warn", "dependency failures may be downstream noise")
        self._degrade_upstream_callers(scenario, f"500s from {root.name} config mismatch")
        self.recent_alerts = [
            f"{scenario.severity}: {scenario.initial_symptoms['alert']}",
            f"{root.name} config version differs from desired state",
        ]

    def _inject_red_herrings(self, scenario: ScenarioSpec) -> None:
        """Add noisy but non-causal logs to unrelated services."""

        unrelated = [
            name for name in self.services if name not in set(scenario.affected_services)
        ] or list(self.services)
        for message in scenario.red_herrings:
            target = self.rng.choice(unrelated)
            self._add_log(target, "warn", message)

    def _degrade_upstream_callers(self, scenario: ScenarioSpec, message: str) -> None:
        """Degrade services upstream of the root cause."""

        for service_name in scenario.affected_services:
            if service_name == scenario.root_cause_service:
                continue
            service = self.services[service_name]
            service.health = "degraded"
            service.metrics.latency_ms += self.rng.randint(180, 520)
            service.metrics.error_rate = min(1.0, service.metrics.error_rate + 0.10)
            service.metrics.cpu_percent = min(100.0, service.metrics.cpu_percent + 18)
            self._add_log(service.name, "error", message)

    def _degrade_services(self, service_names: List[ServiceName], down: bool = False) -> None:
        """Degrade or take down a list of services."""

        for service_name in service_names:
            service = self.services[service_name]
            service.health = "down" if down else "degraded"
            service.metrics.error_rate = min(1.0, service.metrics.error_rate + 0.12)
            service.metrics.latency_ms += self.rng.randint(120, 360)

    def _recover_scenario(self, evidence: str, result: str) -> None:
        """Recover all affected services for the active scenario."""

        del evidence, result
        if not self.active_scenario:
            return
        for service_name in self.active_scenario.affected_services:
            service = self.services[service_name]
            service.health = "healthy"
            service.metrics.cpu_percent = min(service.metrics.cpu_percent, 42.0)
            service.metrics.memory_percent = min(service.metrics.memory_percent, 55.0)
            service.metrics.latency_ms = min(service.metrics.latency_ms, 120.0)
            service.metrics.error_rate = min(service.metrics.error_rate, 0.01)
            service.metrics.open_connections = min(service.metrics.open_connections, 140)
            self._add_log(service.name, "info", "dependency checks green after recovery")
        self.recent_alerts = ["RECOVERY: affected customer paths are healthy"]

    def _metric_evidence(self, service: ServiceRuntime) -> List[str]:
        """Return evidence discovered by querying metrics for a service."""

        scenario = self.active_scenario
        if not scenario or service.name != scenario.root_cause_service:
            return []
        if scenario.scenario_type == "memory_leak" and service.metrics.memory_percent >= 90:
            return [f"metric:{service.name}:memory_saturation"]
        if (
            scenario.scenario_type == "connection_pool_exhaustion"
            and service.metrics.open_connections >= 400
        ):
            return [f"metric:{service.name}:connection_saturation"]
        if scenario.scenario_type == "cascading_retry_storm" and service.metrics.cpu_percent >= 85:
            return [f"metric:{service.name}:cpu_saturation"]
        if scenario.scenario_type == "config_drift" and service.metrics.error_rate >= 0.15:
            return [f"metric:{service.name}:error_spike"]
        return []

    def _log_evidence(
        self, service: ServiceRuntime, logs: List[LogEntry]
    ) -> List[str]:
        """Return evidence discovered by reading service logs."""

        scenario = self.active_scenario
        if not scenario or service.name != scenario.root_cause_service:
            return []
        messages = " ".join(log.message.lower() for log in logs)
        if scenario.scenario_type == "memory_leak" and "heap growth" in messages:
            return [f"log:{service.name}:heap_growth"]
        if (
            scenario.scenario_type == "connection_pool_exhaustion"
            and "pool exhausted" in messages
        ):
            return [f"log:{service.name}:pool_exhausted"]
        if scenario.scenario_type == "cascading_retry_storm" and "retry storm" in messages:
            return [
                f"log:{service.name}:retry_storm",
                f"rate_limit:{service.name}:throttling",
            ]
        if scenario.scenario_type == "config_drift" and "config mismatch" in messages:
            return [
                f"log:{service.name}:config_mismatch",
                f"config:{service.name}:drift_detected",
            ]
        return []

    def _dependency_evidence(self, service: ServiceRuntime) -> List[str]:
        """Return dependency evidence discovered by checking a service."""

        scenario = self.active_scenario
        if not scenario:
            return []
        evidence = []
        for dependency in service.dependencies:
            candidate = f"dependency:{service.name}->{dependency}"
            if candidate in scenario.required_evidence:
                evidence.append(candidate)
        return evidence

    def _service(
        self,
        name: ServiceName,
        *,
        cpu: float,
        memory: float,
        latency: float,
        error_rate: float,
        request_count: int,
        open_connections: int,
        replicas: int,
    ) -> ServiceRuntime:
        """Build a healthy service runtime object."""

        return ServiceRuntime(
            name=name,
            metrics=ServiceMetrics(
                cpu_percent=cpu,
                memory_percent=memory,
                latency_ms=latency,
                error_rate=error_rate,
                request_count=request_count,
                open_connections=open_connections,
                replica_count=replicas,
            ),
            dependencies=list(SERVICE_DEPENDENCIES[name]),
        )

    def _add_log(self, service_name: ServiceName, level: LogLevel, message: str) -> None:
        """Append a timestamped log entry to a service."""

        service = self.services[service_name]
        service.logs.append(
            LogEntry(
                timestamp=f"T+{self.elapsed_minutes:02d}m",
                service=service_name,
                level=level,
                message=message,
            )
        )

    def _get_service(self, service_name: Optional[ServiceName]) -> ServiceRuntime:
        """Fetch a service or raise a clear validation error."""

        if service_name is None:
            raise ValueError("target_service is required for this action")
        return self.services[service_name]

    def _is_root_scenario(self, service_name: ServiceName, scenario_type: str) -> bool:
        """Return whether a service is root cause for a scenario type."""

        return (
            self.active_scenario is not None
            and self.active_scenario.scenario_type == scenario_type
            and service_name == self.active_scenario.root_cause_service
        )

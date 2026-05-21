from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolPlan:
    route: str
    plan_steps: list[str] = field(default_factory=list)
    implementation_spec: dict[str, Any] = field(default_factory=dict)
    render_contract: dict[str, Any] = field(default_factory=dict)
    self_checks: list[str] = field(default_factory=list)


@dataclass
class ExecutionResult:
    image: Any
    artifacts: dict[str, Any] = field(default_factory=dict)
    trace: dict[str, Any] = field(default_factory=dict)


class RoutePlanner:
    def plan(self, context: dict[str, Any]) -> ToolPlan:
        raise NotImplementedError


class SynthesisTool:
    route: str = ""

    def plan(self, context: dict[str, Any]) -> ToolPlan:
        raise NotImplementedError

    def execute(self, plan: ToolPlan, context: dict[str, Any]) -> ExecutionResult:
        raise NotImplementedError

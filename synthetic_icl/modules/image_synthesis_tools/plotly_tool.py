from __future__ import annotations

import json
from typing import Any

from synthetic_icl.backbone import MLLMBackbone
from synthetic_icl.json_utils import llm_json_call_with_retry
from synthetic_icl.modules.image_synthesis_tools.base import ExecutionResult, SynthesisTool, ToolPlan


class PlotlySynthesisTool(SynthesisTool):
    route = "plotly_python"

    def __init__(self, backbone: MLLMBackbone) -> None:
        self.backbone = backbone

    def plan(self, context: dict[str, Any]) -> ToolPlan:
        prompt = f"""
You are a plotly planner for scientific chart synthesis.
Return JSON with fields: plan_steps (list), implementation_spec (object), render_contract (object), self_checks (list), code (string).
Context:\n{json.dumps(context, ensure_ascii=False, indent=2)}
Constraints:
- Use plotly (and helpers such as PIL/io).
- Produce Python code that sets variable output_image to a PIL Image.
- Prioritize task-answerability, then moderate style alignment.
""".strip()
        parsed = llm_json_call_with_retry(lambda: self.backbone.generate_response_text(prompt), max_attempts=3)
        if not isinstance(parsed, dict):
            raise ValueError("plotly planner did not return JSON object")
        spec = dict(parsed.get("implementation_spec") or {})
        spec["code"] = str(parsed.get("code", ""))
        return ToolPlan(
            route=self.route,
            plan_steps=[str(x) for x in parsed.get("plan_steps", [])],
            implementation_spec=spec,
            render_contract=dict(parsed.get("render_contract") or {}),
            self_checks=[str(x) for x in parsed.get("self_checks", [])],
        )

    def execute(self, plan: ToolPlan, context: dict[str, Any]) -> ExecutionResult:
        _ = context
        code = str(plan.implementation_spec.get("code", "")).strip()
        if not code:
            raise ValueError("plotly planner returned empty code")
        scope: dict[str, Any] = {}
        exec(code, {"__builtins__": __builtins__}, scope)
        image = scope.get("output_image")
        if image is None:
            raise ValueError("plotly code did not define output_image")
        return ExecutionResult(image=image, artifacts={"code": code}, trace={"route": self.route})

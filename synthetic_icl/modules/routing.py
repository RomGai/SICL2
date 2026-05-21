"""Route synthesis modality via MLLM from predefined route keys."""

from __future__ import annotations

import json
from typing import Any

from synthetic_icl.backbone import MLLMBackbone
from synthetic_icl.json_utils import llm_json_call_with_retry
from synthetic_icl.schemas import TaskIR



class SynthesisRouterModule:
    @staticmethod
    def _safe_float(value: Any, default: float = 0.5) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def __init__(self, backbone: MLLMBackbone, routes: list[str] | None = None) -> None:
        self.backbone = backbone
        self.routes = routes or ["matplotlib_python", "plotly_python"]

    def run(self, task_ir: TaskIR, understanding: dict[str, Any], query: str) -> dict[str, Any]:
        routes = self.routes
        prompt = f"""
Select one route key from allowed_routes for synthetic image synthesis.
allowed_routes={json.dumps(routes, ensure_ascii=False)}
Rules:
- prioritize task answerability and query alignment
- keep visual style moderately aligned with original image
- fallback is disabled

Query: {json.dumps(query, ensure_ascii=False)}
TaskIR: {json.dumps(task_ir.to_dict(), ensure_ascii=False)}
Understanding: {json.dumps(understanding, ensure_ascii=False)}

Return strict JSON:
{{
  "selected_route": "matplotlib_python",
  "route_confidence": 0.8,
  "route_reason": "...",
  "style_alignment_notes": ["..."],
  "constraints": ["..."],
  "fallback_allowed": false
}}
""".strip()
        parsed = llm_json_call_with_retry(lambda: self.backbone.generate_response_text(prompt), max_attempts=3)
        if not isinstance(parsed, dict):
            parsed = {}
        selected = str(parsed.get("selected_route", "")).strip()
        if selected not in routes:
            selected = routes[0]
        return {
            "selected_route": selected,
            "route_confidence": self._safe_float(parsed.get("route_confidence"), default=0.5),
            "route_reason": str(parsed.get("route_reason", "fallback to first allowed route")),
            "style_alignment_notes": [str(x) for x in parsed.get("style_alignment_notes", [])] if isinstance(parsed.get("style_alignment_notes"), list) else [],
            "constraints": [str(x) for x in parsed.get("constraints", [])] if isinstance(parsed.get("constraints"), list) else [],
            "fallback_allowed": False,
        }

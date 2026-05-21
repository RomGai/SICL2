"""Route synthesis modality via MLLM from predefined route keys."""

from __future__ import annotations

import json
from typing import Any

from synthetic_icl.backbone import MLLMBackbone
from synthetic_icl.json_utils import robust_json_parse
from synthetic_icl.schemas import TaskIR


class SynthesisRouterModule:
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
        raw = self.backbone.generate_response_text(prompt)
        parsed = robust_json_parse(raw)
        if not isinstance(parsed, dict):
            parsed = {}
        selected = str(parsed.get("selected_route", "")).strip()
        if selected not in routes:
            selected = routes[0]
        return {
            "selected_route": selected,
            "route_confidence": float(parsed.get("route_confidence", 0.5) or 0.5),
            "route_reason": str(parsed.get("route_reason", "fallback to first allowed route")),
            "style_alignment_notes": [str(x) for x in parsed.get("style_alignment_notes", [])] if isinstance(parsed.get("style_alignment_notes"), list) else [],
            "constraints": [str(x) for x in parsed.get("constraints", [])] if isinstance(parsed.get("constraints"), list) else [],
            "fallback_allowed": False,
        }

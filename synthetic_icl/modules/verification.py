"""Synthetic image verification module."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from PIL import Image

from synthetic_icl.backbone import MLLMBackbone
from synthetic_icl.json_utils import llm_json_call_with_retry
from synthetic_icl.schemas import AnswerSpec, ScenarioSpec, TaskIR


class VerificationModule:
    """Verify generated images with route-aware checks."""

    def __init__(self, backbone: MLLMBackbone) -> None:
        self.backbone = backbone

    def run(
        self,
        synthetic_image: Image.Image | None,
        evaluation_query: str,
        known_answer: str,
        task_ir: TaskIR,
        scenario: ScenarioSpec,
        answer_spec: AnswerSpec,
        source_query: str | None = None,
        synthesis_trace: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if synthetic_image is None:
            return {"status": "failed", "pass": False, "reason": "no image produced", "is_valid_demo": False}

        prompt = f"""
You are verifying one synthetic multimodal ICL demonstration image.

Evaluation query:
{json.dumps(evaluation_query, ensure_ascii=False)}
Known answer:
{json.dumps(known_answer, ensure_ascii=False)}
Source/original query:
{json.dumps(source_query or task_ir.original_query, ensure_ascii=False)}
TaskIR:
{json.dumps(task_ir.to_dict(), ensure_ascii=False, indent=2)}
Scenario:
{json.dumps(scenario.to_dict(), ensure_ascii=False, indent=2)}
AnswerSpec:
{json.dumps(answer_spec.to_dict(), ensure_ascii=False, indent=2)}
Synthesis trace (contains selected route + planner/executor data):\n{json.dumps(synthesis_trace or {}, ensure_ascii=False, indent=2)}

Return strict JSON:
{{"status":"completed","pass":true,"predicted_answer":"...","matches_known_answer":true,
"ambiguity_score":0.2,"reason":"...","issues":["..."],"improvement_actions":["..."],"is_valid_demo":true,"confidence":0.8}}
""".strip()
        parsed = llm_json_call_with_retry(
            lambda: self.backbone.generate_response_multimodal_single(synthetic_image, prompt),
            max_attempts=3,
        )
        if not isinstance(parsed, dict):
            raise ValueError("VerificationModule expected JSON object")
        parsed.setdefault("status", "completed")
        parsed.setdefault("pass", bool(parsed.get("matches_known_answer")))
        parsed.setdefault("is_valid_demo", bool(parsed.get("pass")))
        parsed.setdefault("confidence", 0.5)
        return parsed

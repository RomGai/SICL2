"""End-to-end pipeline for query-driven synthetic demonstrations."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, TypeVar

if TYPE_CHECKING:
    from PIL import Image

from synthetic_icl.backbone import MLLMBackbone
from synthetic_icl.modules import (
    AnswerSamplingModule,
    DemonstrationSelectionModule,
    GenerationPromptConstructionModule,
    ImageGenerationModule,
    ImageQueryUnderstandingModule,
    ScenarioExpansionModule,
    SynthesisRouterModule,
    TaskInductionModule,
    VerificationModule,
)
from synthetic_icl.schemas import SyntheticExample


class SyntheticICLPipeline:
    def __init__(self, backbone: MLLMBackbone, image_generation_module: ImageGenerationModule | None = None) -> None:
        self.backbone = backbone
        self.image_understanding_module = ImageQueryUnderstandingModule(backbone)
        self.task_induction_module = TaskInductionModule(backbone)
        self.scenario_expansion_module = ScenarioExpansionModule(backbone)
        self.answer_sampling_module = AnswerSamplingModule(backbone)
        self.prompt_construction_module = GenerationPromptConstructionModule(backbone)
        self.router_module = SynthesisRouterModule(backbone, routes=["matplotlib_python", "plotly_python"])
        self.image_generation_module = image_generation_module or ImageGenerationModule(backbone)
        self.verification_module = VerificationModule(backbone)
        self.selection_module = DemonstrationSelectionModule(backbone)
        self.last_candidates: list[SyntheticExample] = []
        self.last_run_log: dict[str, Any] = {}
        self.last_attempt_traces: list[dict[str, Any]] = []
        self.logger = logging.getLogger(__name__)

    _T = TypeVar("_T")

    def _run_with_retry(
        self,
        operation: Callable[[], _T],
        *,
        label: str,
        max_retries: int,
    ) -> _T:
        """Run operation with bounded retries for transient model/runtime failures."""
        last_error: Exception | None = None
        for attempt_idx in range(max_retries + 1):
            try:
                return operation()
            except Exception as exc:  # noqa: BLE001 - attempt-level resiliency across model/runtime errors
                last_error = exc
                if attempt_idx >= max_retries:
                    break
                self.logger.warning(
                    "Retrying %s after failure (%d/%d): %s",
                    label,
                    attempt_idx + 1,
                    max_retries + 1,
                    exc,
                )
        assert last_error is not None
        raise last_error

    def run(
        self,
        original_image: Image.Image,
        original_query: str,
        num_scenarios: int = 5,
        num_answers_per_scenario: int = 1,
        top_k: int = 3,
        dry_run: bool = False,
        preserve_original_query: bool = True,
        scenario_regen_rounds: int = 3,
        max_replan_rounds: int = 2,
        generation_retry_rounds: int = 1,
        verification_retry_rounds: int = 1,
    ) -> list[SyntheticExample]:
        understanding = self.image_understanding_module.run(original_image, original_query)
        task_ir = self.task_induction_module.run(original_query, understanding)
        scenarios = self.scenario_expansion_module.run(task_ir, num_scenarios, max_regen_rounds=scenario_regen_rounds)
        answer_specs = self.answer_sampling_module.run(task_ir, scenarios, num_answers_per_scenario, preserve_original_query=preserve_original_query)
        router_decision = self.router_module.run(task_ir=task_ir, understanding=understanding, query=original_query)

        scenarios_by_id = {s.scenario_id: s for s in scenarios}
        candidates: list[SyntheticExample] = []
        traces: list[dict[str, Any]] = []

        for answer_spec in answer_specs:
            scenario = scenarios_by_id.get(answer_spec.scenario_id)
            if scenario is None:
                continue
            prompt_spec = self.prompt_construction_module.run(original_image, task_ir, scenario, answer_spec, answer_spec.query or original_query)

            last_verification: dict[str, Any] = {"is_valid_demo": False, "reason": "not_run"}
            final_bundle = None
            feedback: dict[str, Any] = {}
            for try_idx in range(1 if dry_run else max_replan_rounds + 1):
                if dry_run:
                    break
                run_context = dict(router_decision)
                run_context["replan_round"] = try_idx + 1
                run_context["verification_feedback"] = feedback
                try:
                    bundle = self._run_with_retry(
                        lambda: self.image_generation_module.generate(original_image, prompt_spec, synthesis_context=run_context),
                        label="image generation",
                        max_retries=generation_retry_rounds,
                    )
                    verification = self._run_with_retry(
                        lambda: self.verification_module.run(
                            synthetic_image=bundle.image,
                            evaluation_query=answer_spec.query or original_query,
                            known_answer=answer_spec.answer,
                            task_ir=task_ir,
                            scenario=scenario,
                            answer_spec=answer_spec,
                            source_query=original_query,
                            synthesis_trace={"router": router_decision, "route": bundle.route, "plan": bundle.plan, "trace": bundle.trace},
                        ),
                        label="verification",
                        max_retries=verification_retry_rounds,
                    )
                except Exception as exc:  # noqa: BLE001 - replan should absorb per-attempt failures
                    error_msg = f"{type(exc).__name__}: {exc}"
                    last_verification = {"is_valid_demo": False, "reason": "attempt_error", "error": error_msg}
                    traces.append(
                        {
                            "scenario_id": scenario.scenario_id,
                            "try": try_idx + 1,
                            "error": error_msg,
                            "verification": last_verification,
                        }
                    )
                    feedback = {"issues": [{"category": "runtime_failure", "details": error_msg}], "reason": "attempt_error"}
                    continue
                last_verification = verification
                traces.append({"scenario_id": scenario.scenario_id, "route": bundle.route, "try": try_idx + 1, "verification": verification})
                if bool(verification.get("is_valid_demo")):
                    final_bundle = bundle
                    break
                feedback = {"issues": verification.get("issues", []), "reason": verification.get("reason", "")}

            if dry_run:
                candidates.append(SyntheticExample(image=None, query=answer_spec.query or original_query, answer=answer_spec.answer, task_ir=task_ir, scenario=scenario, answer_spec=answer_spec, generation_prompt=prompt_spec, verification_result={"status": "skipped_dry_run"}, selected=False))
                continue
            if final_bundle is None:
                continue

            candidates.append(SyntheticExample(image=final_bundle.image, query=answer_spec.query or original_query, answer=answer_spec.answer, task_ir=task_ir, scenario=scenario, answer_spec=answer_spec, generation_prompt=prompt_spec, verification_result=last_verification, selected=False))

        selected = self.selection_module.run(candidates, top_k)
        self.last_candidates = candidates
        self.last_attempt_traces = traces
        self.last_run_log = {"router": router_decision, "selected_examples": [e.to_metadata_dict() for e in selected]}
        return selected

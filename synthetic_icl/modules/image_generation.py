"""Route-aware synthesis executor (legacy filename kept for compatibility)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from PIL import Image

from synthetic_icl.backbone import MLLMBackbone
from synthetic_icl.modules.image_synthesis_tools.registry import create_tool_registry
from synthetic_icl.schemas import GenerationPromptSpec


@dataclass
class SynthesisExecutionBundle:
    image: Image.Image
    route: str
    plan: dict[str, Any]
    artifacts: dict[str, Any]
    trace: dict[str, Any]


class ImageGenerationModule:
    def __init__(self, backbone: MLLMBackbone) -> None:
        self.registry = create_tool_registry(backbone)

    def generate(
        self,
        original_image: Image.Image,
        generation_prompt_spec: GenerationPromptSpec,
        synthesis_context: dict[str, Any] | None = None,
    ) -> SynthesisExecutionBundle:
        selected_route = str((synthesis_context or {}).get("selected_route", "matplotlib_python"))
        tool = self.registry.get(selected_route)
        if tool is None:
            raise ValueError(f"No synthesis tool for route: {selected_route}")
        context = {
            "router": synthesis_context or {},
            "generation_prompt_spec": generation_prompt_spec.to_dict(),
            "original_image_size": getattr(original_image, "size", None),
        }
        plan = tool.plan(context)
        result = tool.execute(plan, context)
        return SynthesisExecutionBundle(
            image=result.image,
            route=selected_route,
            plan={
                "route": plan.route,
                "plan_steps": plan.plan_steps,
                "implementation_spec": plan.implementation_spec,
                "render_contract": plan.render_contract,
                "self_checks": plan.self_checks,
            },
            artifacts=result.artifacts,
            trace=result.trace,
        )


def create_image_generation_module(backbone: MLLMBackbone, coding_backbone: MLLMBackbone | None = None) -> ImageGenerationModule:
    return ImageGenerationModule(backbone=coding_backbone or backbone)

from __future__ import annotations

from synthetic_icl.backbone import MLLMBackbone
from synthetic_icl.modules.image_synthesis_tools.base import SynthesisTool
from synthetic_icl.modules.image_synthesis_tools.matplotlib_tool import MatplotlibSynthesisTool
from synthetic_icl.modules.image_synthesis_tools.plotly_tool import PlotlySynthesisTool


def create_tool_registry(backbone: MLLMBackbone) -> dict[str, SynthesisTool]:
    return {
        "matplotlib_python": MatplotlibSynthesisTool(backbone),
        "plotly_python": PlotlySynthesisTool(backbone),
    }

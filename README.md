# Query-driven Synthetic Demonstration Generation for Multimodal ICL

This repository builds synthetic multimodal ICL demonstrations from an `original_image` and `original_query` while preserving query invariance.

## Pipeline overview

Current flow is route-aware and extensible:
1. Image-Query Understanding
2. Task Induction
3. Scenario Expansion
4. Answer Sampling
5. Generation Prompt Construction
6. **Synthesis Routing** (`matplotlib_python` / `plotly_python`)
7. **Route-specific Planning + Execution** through `image_synthesis_tools`
8. Route-aware Verification (with improvement actions)
9. Selection

Fallback is currently disabled; verification failure triggers same-route replan retries.

## Installation

```bash
pip install -r requirements.txt
```

## Config

Use `synthetic_icl/demo_config.example.json`.

- `mllm`: general reasoning backbone (understanding/induction/routing/verification).
- `coding_mllm`: optional separate backbone for coding planners/executors (e.g. `gemini-3.5-flash`).
- `run`: runtime options. Removed obsolete fields that are not used by current route-aware pipeline.

## Run

```bash
python -m synthetic_icl.demo --config synthetic_icl/demo_config.example.json
```

## Extensibility

See `ARCHITECTURE_EXTENSIBILITY.md` for extension interfaces and future route directions.

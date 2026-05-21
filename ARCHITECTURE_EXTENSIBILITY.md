# Extensibility Blueprint for Route-aware Synthesis

## Current Core Flow
1. Understanding / Task induction / Scenario + answer sampling.
2. Router selects a concrete route key (`matplotlib_python` or `plotly_python`).
3. Route tool performs route-specific planning (`plan_steps + implementation_spec`) and execution.
4. Verifier evaluates image-query-answer alignment and returns improvement actions.
5. Pipeline can re-plan on the **same route** using verifier feedback.

## Stable Extension Interfaces

### 1) Router Interface (`SynthesisRouterModule`)
- Input: `TaskIR`, understanding payload, query.
- Output contract:
  - `selected_route`
  - `route_confidence`
  - `route_reason`
  - `style_alignment_notes`
  - `constraints`
  - `fallback_allowed`
- Future extension: add route keys for edit/generative chains without changing orchestrator shape.

### 2) Tool Interface (`image_synthesis_tools/base.py`)
- `SynthesisTool.plan(context) -> ToolPlan`
- `SynthesisTool.execute(plan, context) -> ExecutionResult`
- `ToolPlan` carries route-specific planner output in a generic envelope:
  - `plan_steps`
  - `implementation_spec`
  - `render_contract`
  - `self_checks`

### 3) Executor Contract
- Pipeline calls tool plan/execute via route registry.
- Execution output includes:
  - final image
  - artifacts (e.g., code, prompts, intermediate metadata)
  - trace (tool-chain runtime trace)
- Future routes can use non-code execution (edit-model tool composition, generative model orchestration).

### 4) Verifier Contract
- Input includes `synthesis_trace` with route + plan/execution context.
- Output includes:
  - validity signals (`is_valid_demo`, `pass`, ambiguity/confidence)
  - `issues`
  - `improvement_actions`
- Route-aware verification rules can be specialized by route key.

## Candidate Future Routes

### A. `edit_model_chain` (future)
- Planner output: edit strategy graph (global style edit, structural edits, local fixes).
- Executor: ordered model/tool calls (masking, localized edits, consistency post-check).
- Verifier: detect semantic drift and style mismatch; return edit-target actions.

### B. `gen_model_chain` (future)
- Planner output: generation setup (prompt family, control signals, seeds, constraints).
- Executor: generation + optional post-processing tools.
- Verifier: query-answer evidence confidence and hallucination checks.

### C. Hybrid routes
- `matplotlib_then_edit` or `plotly_then_edit` for style polishing while preserving task evidence.

## Practical Guidance for Adding a New Route
1. Add route key to router allowed routes.
2. Implement a new tool class under `image_synthesis_tools/`.
3. Register it in `registry.py`.
4. Ensure tool planner returns `ToolPlan` fields.
5. Extend verifier prompt/rules for the new route semantics.
6. (Optional) add route-specific replan heuristics.

## Current Policy Choices
- Fallback disabled by design.
- Route chosen by MLLM from predefined list.
- Priority: task answerability/alignment > style similarity, with moderate style preservation.

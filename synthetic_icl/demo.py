"""CLI entry point for running the synthetic ICL pipeline."""

from __future__ import annotations

import argparse
import io
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from synthetic_icl.backbone import MLLMBackbone
from synthetic_icl.modules.image_generation import create_image_generation_module
from synthetic_icl.pipeline import SyntheticICLPipeline
from synthetic_icl.schemas import SyntheticExample


def _print_json(title: str, payload: object) -> None:
    print(f"\n=== {title} ===")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _save_generated_images(examples: list[SyntheticExample], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for idx, example in enumerate(examples):
        if example.image is None:
            continue
        safe_scenario_id = example.scenario.scenario_id or f"example_{idx:03d}"
        image_path = output_dir / f"{idx:03d}_{safe_scenario_id}.png"
        example.image.save(image_path)
        example.verification_result.setdefault("saved_image_path", str(image_path.resolve()))


def _save_dataset_case_assets(case_dir: Path, case_idx: int, image_bytes: bytes, prompt: str, groundtruth: str) -> None:
    import importlib.util

    if importlib.util.find_spec("PIL") is None:
        raise ImportError("Pillow is required to load images. Install it with: pip install Pillow")

    from PIL import Image

    case_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(io.BytesIO(image_bytes)) as gt_img:
        gt_img.convert("RGB").save(case_dir / f"{case_idx}_gt.png")
    _save_json_log(
        {
            "prompt": prompt,
            "groundtruth": groundtruth,
        },
        case_dir / "meta.json",
    )


def _iter_test_pt_cases(test_pt_path: Path):
    import torch

    if not test_pt_path.exists():
        raise FileNotFoundError(f"Test pt file not found: {test_pt_path}")

    dataset = torch.load(test_pt_path, map_location="cpu", weights_only=False)
    if not isinstance(dataset, list):
        raise ValueError("Loaded test .pt data must be a list of dict entries.")

    for idx, item in enumerate(dataset):
        if not isinstance(item, dict):
            raise ValueError(f"Entry at index {idx} is not a dict.")
        prompt = item.get("prompt")
        groundtruth = item.get("groundtruth")
        image_bytes = item.get("image")
        if not isinstance(prompt, str) or not prompt:
            raise ValueError(f"Entry at index {idx} has invalid prompt.")
        if not isinstance(groundtruth, str):
            raise ValueError(f"Entry at index {idx} has invalid groundtruth.")
        if not isinstance(image_bytes, (bytes, bytearray)):
            raise ValueError(f"Entry at index {idx} has invalid image bytes.")
        yield idx, prompt, groundtruth, bytes(image_bytes)



def _save_json_log(log_payload: dict[str, Any], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as fp:
        json.dump(log_payload, fp, ensure_ascii=False, indent=2)

def _load_config(config_path: str | None) -> dict[str, Any]:
    if not config_path:
        return {}
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    if not isinstance(data, dict):
        raise ValueError("Config file root must be a JSON object.")
    return data


def _coalesce(arg_value: Any, config: dict[str, Any], key: str) -> Any:
    return arg_value if arg_value is not None else config.get(key)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the synthetic multimodal ICL pipeline.")
    parser.add_argument("--config", help="Path to a JSON config file for MLLM and pipeline parameters.")
    parser.add_argument("--image", help="Path to the original image.")
    parser.add_argument("--query", help="Original query. It will not be rewritten.")
    parser.add_argument("--num-scenarios", type=int, help="Number of scenarios to expand.")
    parser.add_argument("--num-answers-per-scenario", type=int, help="Answers per scenario.")
    parser.add_argument("--scenario-regen-rounds", type=int, help="Max regeneration rounds to refill aligned scenarios.")
    parser.add_argument("--max-replan-rounds", type=int, help="Max replan rounds per attempt.")
    parser.add_argument("--generation-retry-rounds", type=int, help="Retries for generation failures per attempt.")
    parser.add_argument("--verification-retry-rounds", type=int, help="Retries for verification failures per attempt.")
    parser.add_argument("--top-k", type=int, help="Number of selected examples.")
    parser.add_argument(
        "--preserve-original-query",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Whether synthetic examples must keep the exact original query text.",
    )
    parser.add_argument(
        "--image-generation-pipe",
        choices=["code_synthesis"],
        help="Image generation backend.",
    )
    parser.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Skip real image generation. Defaults to false for code_synthesis."
        ),
    )
    parser.add_argument("--output-dir", help="Directory for generated images.")
    parser.add_argument("--log-json-path", help="Path to save detailed intermediate pipeline log JSON.")
    parser.add_argument("--mllm-api-key", help="Override MLLM API key (or set in config/env).")
    parser.add_argument("--mllm-base-url", help="Override MLLM base URL (or set in config/env).")
    parser.add_argument("--mllm-model-name", help="Override MLLM model name (or set in config/env).")
    parser.add_argument("--coding-mllm-api-key", help="Override coding MLLM API key (or set in config/env).")
    parser.add_argument("--coding-mllm-base-url", help="Override coding MLLM base URL (or set in config/env).")
    parser.add_argument("--coding-mllm-model-name", help="Override coding MLLM model name (or set in config/env).")
    args = parser.parse_args()

    config = _load_config(args.config)
    mllm_cfg = config.get("mllm", {}) if isinstance(config.get("mllm"), dict) else {}
    run_cfg = config.get("run", {}) if isinstance(config.get("run"), dict) else {}
    coding_mllm_cfg = config.get("coding_mllm", {}) if isinstance(config.get("coding_mllm"), dict) else {}

    image = _coalesce(args.image, run_cfg, "image")
    query = _coalesce(args.query, run_cfg, "query")
    num_scenarios = int(_coalesce(args.num_scenarios, run_cfg, "num_scenarios") or 5)
    num_answers_per_scenario = int(_coalesce(args.num_answers_per_scenario, run_cfg, "num_answers_per_scenario") or 1)
    scenario_regen_rounds = int(_coalesce(args.scenario_regen_rounds, run_cfg, "scenario_regen_rounds") or 3)
    max_replan_rounds = int(_coalesce(args.max_replan_rounds, run_cfg, "max_replan_rounds") or 2)
    generation_retry_rounds = int(_coalesce(args.generation_retry_rounds, run_cfg, "generation_retry_rounds") or 1)
    verification_retry_rounds = int(_coalesce(args.verification_retry_rounds, run_cfg, "verification_retry_rounds") or 1)
    top_k = int(_coalesce(args.top_k, run_cfg, "top_k") or 3)
    preserve_original_query_cfg = _coalesce(args.preserve_original_query, run_cfg, "preserve_original_query")
    preserve_original_query = True if preserve_original_query_cfg is None else bool(preserve_original_query_cfg)
    answer_sampling_format_retry_times_raw = _coalesce(None, run_cfg, "answer_sampling_format_retry_times")
    answer_sampling_format_retry_times = (
        int(answer_sampling_format_retry_times_raw) if answer_sampling_format_retry_times_raw is not None else 5
    )
    image_generation_pipe = _coalesce(args.image_generation_pipe, run_cfg, "image_generation_pipe") or "code_synthesis"
    output_dir = _coalesce(args.output_dir, run_cfg, "output_dir") or "synthetic_outputs"
    test_pt_path_raw = _coalesce(None, run_cfg, "test_pt_path")
    log_json_path = _coalesce(args.log_json_path, run_cfg, "log_json_path")

    test_pt_path = Path(test_pt_path_raw) if test_pt_path_raw else None
    if test_pt_path is None:
        if not image:
            raise ValueError("Missing image path. Provide --image or run.image in config.")
        if not query:
            raise ValueError("Missing query. Provide --query or run.query in config.")

    dry_run = args.dry_run if args.dry_run is not None else run_cfg.get("dry_run")
    if dry_run is None:
        dry_run = False

    import importlib.util

    if importlib.util.find_spec("PIL") is None:
        raise ImportError("Pillow is required to load images. Install it with: pip install Pillow")

    from PIL import Image

    backbone = MLLMBackbone(
        api_key=_coalesce(args.mllm_api_key, mllm_cfg, "api_key"),
        base_url=_coalesce(args.mllm_base_url, mllm_cfg, "base_url"),
        model=_coalesce(args.mllm_model_name, mllm_cfg, "model_name"),
    )
    coding_backbone = MLLMBackbone(
        api_key=_coalesce(args.coding_mllm_api_key, coding_mllm_cfg, "api_key") or backbone.api_key,
        base_url=_coalesce(args.coding_mllm_base_url, coding_mllm_cfg, "base_url") or backbone.base_url,
        model=_coalesce(args.coding_mllm_model_name, coding_mllm_cfg, "model_name") or backbone.model,
    )
    image_generation_module = create_image_generation_module(backbone, coding_backbone=coding_backbone)
    pipeline = SyntheticICLPipeline(backbone, image_generation_module=image_generation_module)
    if hasattr(pipeline.answer_sampling_module, "format_retry_times"):
        pipeline.answer_sampling_module.format_retry_times = max(0, answer_sampling_format_retry_times)
    output_root = Path(output_dir)

    if test_pt_path is not None:
        for idx, prompt, groundtruth, image_bytes in _iter_test_pt_cases(test_pt_path):
            case_dir = output_root / str(idx)
            _save_dataset_case_assets(case_dir, idx, image_bytes, prompt, groundtruth)

            with Image.open(io.BytesIO(image_bytes)) as img:
                original_image = img.convert("RGB")

            selected_examples = pipeline.run(
                original_image=original_image,
                original_query=prompt,
                num_scenarios=num_scenarios,
                num_answers_per_scenario=num_answers_per_scenario,
                top_k=top_k,
                dry_run=bool(dry_run),
                preserve_original_query=preserve_original_query,
                scenario_regen_rounds=scenario_regen_rounds,
                max_replan_rounds=max_replan_rounds,
                generation_retry_rounds=generation_retry_rounds,
                verification_retry_rounds=verification_retry_rounds,
            )

            if not dry_run:
                _save_generated_images(pipeline.last_candidates, case_dir)
    
            if log_json_path:
                _save_json_log(pipeline.last_run_log, case_dir / Path(log_json_path).name)

            _print_json("Selected Examples Metadata", [example.to_metadata_dict() for example in selected_examples])
    else:
        image_path = Path(image)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        with Image.open(image_path) as img:
            original_image = img.convert("RGB")

        selected_examples = pipeline.run(
            original_image=original_image,
            original_query=query,
            num_scenarios=num_scenarios,
            num_answers_per_scenario=num_answers_per_scenario,
            top_k=top_k,
            dry_run=bool(dry_run),
            preserve_original_query=preserve_original_query,
            scenario_regen_rounds=scenario_regen_rounds,
            max_replan_rounds=max_replan_rounds,
            generation_retry_rounds=generation_retry_rounds,
            verification_retry_rounds=verification_retry_rounds,
        )

        if not dry_run:
            _save_generated_images(pipeline.last_candidates, output_root)

        if log_json_path:
            _save_json_log(pipeline.last_run_log, Path(log_json_path))

        if selected_examples:
            _print_json("TaskIR", selected_examples[0].task_ir.to_dict())

        _print_json("ScenarioSpecs", [example.scenario.to_dict() for example in pipeline.last_candidates])
        _print_json("AnswerSpecs", [example.answer_spec.to_dict() for example in pipeline.last_candidates])
        _print_json(
            "Generation Prompts",
            [asdict(example.generation_prompt) for example in pipeline.last_candidates],
        )
        _print_json("Selected Examples Metadata", [example.to_metadata_dict() for example in selected_examples])


if __name__ == "__main__":
    main()

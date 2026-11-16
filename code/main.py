#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""main.py

Baseline evaluation runner.

This script runs one or more LLM judge models on a paired dataset (canonical + buggy)
under one or more prompt modes, and logs outputs for downstream scoring.

Key features
------------
- No hardcoded API keys (use .env / environment variables)
- Resume-safe checkpointing via append-only JSONL
- Same JSON schema across benchmarks

Example
-------
python main.py \
  --dataset data/humanevalbugs.jsonl \
  --benchmark humaneval \
  --out_dir outputs/humaneval \
  --models gpt-4o gemini-2.0-flash claude-haiku-4-5 \
  --prompt_modes direct direct_explain full

"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from tqdm import tqdm

from model import AnthropicModel, GoogleModel, OpenAIModel, OpenRouterModel
from utils import (
    MissingGeneratedModuleError,
    check_generated_candidate,
    construct_candidate_prompt,
    construct_generated_tests_v2_prompt,
    construct_prompt,
    construct_reference_input_prompt,
    construct_signature_prompt,
    extract_code_block,
    extract_generated_tests,
    function_name_from_signature,
    generate_reference_output_test,
    load_data,
    parse_function_signature,
    parse_response,
    parse_reference_input,
    run_generated_tests,
    run_validated_tests,
    save_json,
    validate_candidate_signature,
    validate_generated_test,
)


DEFAULT_MODELS = [
    "gpt-4o",
    "gemini-2.0-flash",
    "claude-haiku-4-5",
    "meta-llama/llama-3.1-8b-instruct",
    "mistralai/mistral-small-3.2-24b-instruct",
]

DEFAULT_PROMPT_MODES = ["direct", "direct_explain", "full", "generated_tests", "generated_tests_v2"]


def read_jsonl_tolerant(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                rows.append(obj)
        except Exception:
            # tolerate partial/corrupted lines (e.g., crash during write)
            continue
    return rows


def append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def expand_paired_dataset(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Expand each task into {canonical, buggy} samples.

    Expected keys (depending on the benchmark preparation):
      - task_id
      - text or prompt or requirement
      - declaration (optional)
      - canonical_solution
      - buggy_solution
      - bug_type / failure_symptoms (optional)

    We keep additional fields if present.
    """
    out: List[Dict[str, Any]] = []
    for item in data:
        task_id = item.get("task_id")
        if task_id is None:
            # Skip malformed rows
            continue

        requirement = item.get("text") or item.get("prompt") or item.get("requirement") or ""
        decl = item.get("declaration", "") or ""

        canonical = (decl or "") + (item.get("canonical_solution") or "")
        buggy = (decl or "") + (item.get("buggy_solution") or "")

        common = {
            "task_id": task_id,
            "requirement": requirement,
            "declaration": decl,
            "benchmark_test": item.get("test", ""),
            "example_test": item.get("example_test", ""),
        }

        out.append({
            **common,
            "variant": "canonical",
            "label": 1,
            "code": canonical,
            "bug_type": "none",
            "failure_symptoms": "none",
        })

        out.append({
            **common,
            "variant": "buggy",
            "label": 0,
            "code": buggy,
            "bug_type": item.get("bug_type", "unknown"),
            "failure_symptoms": item.get("failure_symptoms", "unknown"),
        })

    return out


def build_model(model_name: str):
    if model_name == "gpt-4o" or model_name.startswith("gpt-"):
        return OpenAIModel(model_name=model_name)
    if model_name.startswith("gemini"):
        return GoogleModel(model_name=model_name)
    if model_name.startswith("claude"):
        return AnthropicModel(model_name=model_name)
    if "/" in model_name:
        # OpenRouter style ids
        return OpenRouterModel(model_name=model_name)
    raise ValueError(f"Unknown model: {model_name}")


def evaluate_generated_tests_v2(
    check_requirement: Callable[[str], str],
    requirement: str,
    submitted_code: str,
    num_tests: int,
    max_attempts: int,
    delay: float,
) -> Tuple[str, Dict[str, Any]]:
    """Generate a reference and collect passing tests before evaluating submitted code."""
    if num_tests < 1 or max_attempts < 1:
        raise ValueError("number of tests and maximum attempts must be positive")
    signature, output_format = parse_function_signature(
        check_requirement(construct_signature_prompt(requirement, submitted_code)), submitted_code
    )
    function_name = function_name_from_signature(signature)
    if delay:
        time.sleep(delay)
    candidate_raw = check_requirement(construct_candidate_prompt(requirement, signature, output_format))
    candidate = extract_code_block(candidate_raw)
    if not candidate:
        raise ValueError("model returned an empty candidate implementation")
    validate_candidate_signature(candidate, signature)
    check_generated_candidate(candidate)

    accepted: List[Tuple[str, str]] = []
    previous_tests: List[str] = []
    fingerprints: Set[str] = set()
    rejected: List[Dict[str, Any]] = []
    for attempt in range(1, max_attempts + 1):
        if delay:
            time.sleep(delay)
        prompt = construct_generated_tests_v2_prompt(
            requirement, signature, output_format, previous_tests, attempt
        )
        response = check_requirement(prompt)
        tests = extract_generated_tests(response)
        result = validate_generated_test(candidate, tests, function_name)
        feedback = {
            "status": result["status"],
            "observations": result.get(
                "observations", "unavailable: test was not run or execution did not complete"
            ),
        }
        if result.get("observations_truncated"):
            feedback["observations_truncated"] = True
        if "observation_note" in result:
            feedback["observation_note"] = result["observation_note"]
        if "observation_error" in result:
            feedback["observation_error"] = result["observation_error"]
        if result["status"] != "passed":
            feedback["error"] = result.get("error", result["status"])
            rejected.append({"attempt": attempt, **feedback})
        elif result["fingerprint"] in fingerprints:
            feedback["status"] = "duplicate"
            feedback["error"] = "duplicate test"
            rejected.append({"attempt": attempt, **feedback})
        else:
            fingerprints.add(result["fingerprint"])
            accepted.append((tests, result["test_name"]))
        previous_tests.append(
            f"{tests or response}\nValidation feedback:\n{json.dumps(feedback, ensure_ascii=False, indent=2)}"
        )
        if len(accepted) == num_tests:
            break

    # if not accepted:
    #     last_error = rejected[-1]["error"] if rejected else "none"
    #     raise RuntimeError(
    #         f"generated {len(accepted)} of {num_tests} valid tests after {max_attempts} attempts; "
    #         f"last rejection: {last_error}"
    #     )

    test_sources = ["llm"] * len(accepted)
    reference_excluded: List[Dict[str, Any]] = []
    previous_inputs: List[str] = []
    seen_inputs: Set[str] = set()
    reference_attempts = 0
    while len(accepted) < num_tests and reference_attempts < max_attempts:
        reference_attempts += 1
        if delay:
            time.sleep(delay)
        input_prompt = construct_reference_input_prompt(
            requirement, signature, [test for test, _ in accepted], previous_inputs, reference_attempts
        )
        input_response = check_requirement(input_prompt)
        try:
            input_args, input_kwargs = parse_reference_input(input_response)
        except ValueError as exc:
            error = f"invalid input: {exc}"
        else:
            input_key = repr((input_args, sorted(input_kwargs.items())))
            if input_key in seen_inputs:
                error = "duplicate input"
            else:
                seen_inputs.add(input_key)
                generated = generate_reference_output_test(
                    candidate, function_name, input_args, input_kwargs, len(accepted) + 1
                )
                if generated["status"] != "passed":
                    error = generated["error"]
                else:
                    result = validate_generated_test(candidate, generated["test"], function_name)
                    if result["status"] != "passed":
                        error = f"reference-output test did not pass validation: {result.get('error', result['status'])}"
                    elif result["fingerprint"] in fingerprints:
                        error = "duplicate test"
                    else:
                        fingerprints.add(result["fingerprint"])
                        accepted.append((generated["test"], generated["test_name"]))
                        test_sources.append("reference_output")
                        previous_inputs.append(f"{input_response}\nAccepted")
                        continue
        reference_excluded.append({"attempt": reference_attempts, "input": input_response, "error": error})
        previous_inputs.append(f"{input_response}\nRejected: {error}")

    report = run_validated_tests(submitted_code, accepted)
    report["generated"] = attempt
    report["excluded"] = rejected
    report["requested_tests"] = num_tests
    report["reference_output_attempts"] = reference_attempts
    report["reference_output_excluded"] = reference_excluded
    pred = "YES" if report["failed"] == 0 else "NO"
    return pred, {
        "function_signature": signature,
        "output_format": output_format,
        "function_name": function_name,
        "candidate_implementation": candidate,
        "generated_tests": [test for test, _ in accepted],
        "test_sources": test_sources,
        "execution": report,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="Path to paired dataset (JSON/JSONL)")
    ap.add_argument("--benchmark", required=True, help="Benchmark name for bookkeeping (e.g., humaneval, mbpp, quixbugs)")
    ap.add_argument("--out_dir", required=True, help="Directory to write outputs")

    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="List of judge models to run")
    ap.add_argument(
        "--prompt_modes",
        nargs="+",
        default=DEFAULT_PROMPT_MODES,
        help="Prompt modes: direct, direct_explain, full, generated_tests, generated_tests_v2",
    )

    ap.add_argument("--delay", type=float, default=0.5, help="Sleep between API calls")
    ap.add_argument("--max_items", type=int, default=0, help="Debug: limit number of expanded samples (0 = all)")
    ap.add_argument("--continue_on_error", action="store_true", help="Continue even if a model call fails")
    ap.add_argument("--num_unit_tests", type=int, default=5, help="Number of passing tests required for generated_tests_v2")
    ap.add_argument("--max_test_attempts", type=int, default=10, help="Maximum test-generation calls for generated_tests_v2 (0 = 10 times the requested count)")

    args = ap.parse_args()
    if args.num_unit_tests < 1 or args.max_test_attempts < 0:
        ap.error("--num_unit_tests must be positive and --max_test_attempts must be non-negative")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_data(args.dataset)
    samples = expand_paired_dataset(data)
    if args.max_items and args.max_items > 0:
        samples = samples[: args.max_items]

    for model_name in args.models:
        print("\n" + "=" * 100)
        print(f"[RUN] model={model_name}")

        model = build_model(model_name)

        # Per-model checkpoint files
        base_name = f"{args.benchmark}__{model_name}__baseline"
        ckpt = out_dir / f"{base_name}.jsonl"
        final_json = out_dir / f"{base_name}.json"

        rows = read_jsonl_tolerant(ckpt)
        done: Set[Tuple[Any, str, str]] = set()
        for r in rows:
            tid = r.get("task_id")
            var = r.get("variant")
            mode = r.get("mode")
            if tid is not None and var and mode:
                done.add((tid, str(var), str(mode)))

        if rows:
            print(f"[RESUME] loaded {len(rows)} rows from {ckpt}")

        total = len(samples) * len(args.prompt_modes)
        pbar = tqdm(total=total, desc=f"Judge={model_name}")

        # Advance progress bar for already-done items
        if done:
            pbar.update(len(done))

        t0 = time.time()
        newly = 0

        for s in samples:
            for mode in args.prompt_modes:
                key = (s["task_id"], s["variant"], mode)
                if key in done:
                    continue

                raw = ""
                pred = ""
                try:
                    if mode == "generated_tests_v2":
                        pred, details = evaluate_generated_tests_v2(
                            model.check_requirement, s["requirement"], s["code"],
                            args.num_unit_tests, args.max_test_attempts,
                            args.delay,
                        )
                        raw = json.dumps(details, ensure_ascii=False)
                    else:
                        prompt = construct_prompt(s["requirement"], s["code"], mode=mode)
                        raw = model.check_requirement(prompt)
                    if mode == "generated_tests":
                        tests = extract_generated_tests(raw)
                        report = run_generated_tests(s["code"], tests)
                        pred = "YES" if report["total"] > 0 and report["failed"] == 0 else "NO"
                        raw = json.dumps(
                            {"generated_tests": raw, "extracted_tests": tests, "execution": report},
                            ensure_ascii=False,
                        )
                    elif mode != "generated_tests_v2":
                        pred = parse_response(raw)
                except MissingGeneratedModuleError:
                    raise
                except Exception as e:
                    raw = f"ERROR: {repr(e)}"
                    pred = ""
                    if not args.continue_on_error:
                        raise
                    print(f"[ERROR] model={model_name} task_id={s['task_id']} variant={s['variant']} mode={mode}")
                    print(traceback.format_exc())

                row = {
                    "benchmark": args.benchmark,
                    "model": model_name,
                    "mode": mode,
                    "task_id": s["task_id"],
                    "variant": s["variant"],
                    "label": s["label"],
                    "bug_type": s.get("bug_type", "unknown"),
                    "failure_symptoms": s.get("failure_symptoms", "unknown"),
                    "pred": pred,
                    "raw": raw,
                }
                rows.append(row)
                append_jsonl(ckpt, row)
                done.add(key)
                newly += 1

                pbar.update(1)

                if args.delay:
                    time.sleep(args.delay)

        pbar.close()

        save_json(rows, str(final_json))
        elapsed = time.time() - t0
        print(f"[DONE] wrote: {ckpt}")
        print(f"[DONE] wrote: {final_json}")
        print(f"[STATS] total_rows={len(rows)} newly_written={newly} elapsed={elapsed/60:.1f} min")


if __name__ == "__main__":
    main()

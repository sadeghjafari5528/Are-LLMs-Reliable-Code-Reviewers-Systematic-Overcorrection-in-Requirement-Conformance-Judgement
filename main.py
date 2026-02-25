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
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from tqdm import tqdm

from model import AnthropicModel, GoogleModel, OpenAIModel, OpenRouterModel
from utils import construct_prompt, load_data, parse_response, save_json


DEFAULT_MODELS = [
    "gpt-4o",
    "gemini-2.0-flash",
    "claude-haiku-4-5",
    "meta-llama/llama-3.1-8b-instruct",
    "mistralai/mistral-small-3.2-24b-instruct",
]

DEFAULT_PROMPT_MODES = ["direct", "direct_explain", "full"]


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="Path to paired dataset (JSON/JSONL)")
    ap.add_argument("--benchmark", required=True, help="Benchmark name for bookkeeping (e.g., humaneval, mbpp, quixbugs)")
    ap.add_argument("--out_dir", required=True, help="Directory to write outputs")

    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="List of judge models to run")
    ap.add_argument("--prompt_modes", nargs="+", default=DEFAULT_PROMPT_MODES, help="Prompt modes: direct, direct_explain, full")

    ap.add_argument("--delay", type=float, default=0.5, help="Sleep between API calls")
    ap.add_argument("--max_items", type=int, default=0, help="Debug: limit number of expanded samples (0 = all)")
    ap.add_argument("--continue_on_error", action="store_true", help="Continue even if a model call fails")

    args = ap.parse_args()

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

                prompt = construct_prompt(s["requirement"], s["code"], mode=mode)

                raw = ""
                pred = ""
                try:
                    raw = model.check_requirement(prompt)
                    pred = parse_response(raw)
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

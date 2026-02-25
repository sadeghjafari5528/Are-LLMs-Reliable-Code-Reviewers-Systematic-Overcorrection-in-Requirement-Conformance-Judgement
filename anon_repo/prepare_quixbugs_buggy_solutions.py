#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prepare QuixBugs buggy dataset in the same jsonl schema as mbpp_buggy_solutions.jsonl.

Input (JSON array):
  /path/to/thesis/FSE/dataset/quixbugs_pairs.json

Output (JSONL):
  one record per line with fields:
    task_id, prompt, canonical_solution, test, text, declaration, example_test,
    buggy_solution, bug_type, failure_symptoms, entry_point

Key differences vs MBPP prep:
- QuixBugs already provides mutate_code => we directly use it as buggy_solution (body-only).
- bug_type and failure_symptoms are produced by LLM (GPT-4o) and normalized to your schemas.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# -----------------------------
# Normalization targets
# -----------------------------
BUG_TYPES = [
    "value misuse",
    "excess logic",
    "missing logic",
    "operator misuse",
    "variable misuse",
    "function misuse",
]

FAILURE_SYMPTOMS = [
    "incorrect output",
    "infinite loop",
    "stackoverflow",
]

# -----------------------------
# OpenAI wrapper (standalone)
# -----------------------------
try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore


class OpenAIModel:
    """
    Minimal wrapper compatible with your existing usage pattern:
        openai_model = OpenAIModel(model_name="gpt-4o")
        text = openai_model.check_requirement(prompt)
    """

    def __init__(self, model_name: str = "gpt-4o"):
        if OpenAI is None:
            raise RuntimeError("openai package not found. Please `pip install openai` (v1+).")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY not found in environment.")
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name

    def check_requirement(self, prompt: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return (resp.choices[0].message.content or "").strip()


# -----------------------------
# IO
# -----------------------------
def load_json(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array in {path}")
    return data


def load_existing_task_ids_jsonl(path: str) -> set:
    p = Path(path)
    if not p.exists():
        return set()
    seen = set()
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
                if "task_id" in obj:
                    seen.add(obj["task_id"])
            except Exception:
                continue
    return seen


def append_jsonl(path: str, record: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False))
        f.write("\n")


# -----------------------------
# Code extraction helpers
# -----------------------------
@dataclass
class ExtractedFunction:
    entry_point: str
    declaration: str      # imports + def line (+ decorators if any), no docstring
    body: str             # function body only (indented), no docstring


def _ensure_trailing_newline(s: str) -> str:
    return s if s.endswith("\n") else (s + "\n")


def extract_first_function(code: str) -> ExtractedFunction:
    """
    Extract the first top-level FunctionDef:
      - declaration: everything up to (but excluding) first stmt in function body
      - body: all lines in function body excluding docstring if exists
    """
    code = _ensure_trailing_newline(code)
    lines = code.splitlines(keepends=True)

    try:
        tree = ast.parse(code)
    except SyntaxError:
        # Fallback: regex extraction for function name; keep crude split
        m = re.search(r"^\s*def\s+(\w+)\s*\(", code, flags=re.M)
        entry = m.group(1) if m else "candidate"
        # declaration = up to def line (inclusive)
        decl = ""
        body = ""
        if m:
            def_line_idx = code[: m.start()].count("\n")
            # get def line content
            decl_lines = lines[: def_line_idx + 1]
            decl = "".join(decl_lines)
            body = "".join(lines[def_line_idx + 1 :])
        else:
            decl = ""
            body = code
        return ExtractedFunction(entry_point=entry, declaration=decl, body=body)

    fn_node = None
    for n in tree.body:
        if isinstance(n, ast.FunctionDef):
            fn_node = n
            break
    if fn_node is None:
        raise ValueError("No top-level function definition found.")

    entry_point = fn_node.name

    # Need end_lineno support (Python 3.8+). If missing, fallback to whole remainder.
    end_lineno = getattr(fn_node, "end_lineno", None)
    if end_lineno is None:
        # crude: from def line to end of file
        def_lineno = fn_node.lineno
        decl = "".join(lines[:def_lineno])
        body = "".join(lines[def_lineno:])
        return ExtractedFunction(entry_point=entry_point, declaration=decl, body=body)

    # declaration ends right before first statement inside function (docstring counts as stmt)
    if not fn_node.body:
        # empty function
        decl = "".join(lines[:fn_node.lineno])
        body = ""
        return ExtractedFunction(entry_point=entry_point, declaration=decl, body=body)

    first_stmt = fn_node.body[0]
    body_first_lineno = getattr(first_stmt, "lineno", fn_node.lineno + 1)

    decl = "".join(lines[: body_first_lineno - 1])

    # skip docstring in body if present
    start_idx = body_first_lineno - 1
    if (
        isinstance(first_stmt, ast.Expr)
        and isinstance(first_stmt.value, ast.Constant)
        and isinstance(first_stmt.value.value, str)
    ):
        doc_end = getattr(first_stmt, "end_lineno", body_first_lineno)
        start_idx = doc_end  # start after docstring line(s), 0-based index = doc_end

    body = "".join(lines[start_idx:end_lineno])

    return ExtractedFunction(entry_point=entry_point, declaration=decl, body=body)


# -----------------------------
# Prompt/test construction (MBPP-like)
# -----------------------------
def indent_block(text: str, spaces: int) -> str:
    pad = " " * spaces
    return "\n".join(pad + ln for ln in text.splitlines())


def build_prompt_from_requirement(declaration: str, requirement: str) -> str:
    """
    Build a MBPP-style 'prompt' = declaration + docstring, no body.
    """
    declaration = _ensure_trailing_newline(declaration)
    req = requirement.strip("\n")
    req_ind = indent_block(req, 4)
    return f"{declaration}    \"\"\"\n{req_ind}\n    \"\"\"\n"


def parse_doctest_pairs(text: str) -> List[Tuple[str, str]]:
    """
    Parse pairs like:
        >>> f(127)
        7
    """
    lines = [ln.rstrip() for ln in text.splitlines()]
    pairs: List[Tuple[str, str]] = []
    i = 0
    while i < len(lines):
        ln = lines[i].strip()
        if ln.startswith(">>>"):
            expr = ln[len(">>>"):].strip()
            j = i + 1
            while j < len(lines) and lines[j].strip() == "":
                j += 1
            if j < len(lines) and not lines[j].strip().startswith(">>>"):
                expected = lines[j].strip()
                pairs.append((expr, expected))
                i = j + 1
                continue
        i += 1
    return pairs


def build_example_test(requirement: str, entry_point: str) -> str:
    """
    Best-effort doctest -> asserts.
    If no examples exist, returns empty string.
    """
    pairs = parse_doctest_pairs(requirement)
    if not pairs:
        return ""

    asserts = []
    for expr, expected in pairs:
        # try literal_eval for expected
        try:
            lit = ast.literal_eval(expected)
            expected_code = repr(lit)
        except Exception:
            # fallback: treat as raw token (e.g., True/False/None) or string
            if expected in ("True", "False", "None"):
                expected_code = expected
            else:
                expected_code = repr(expected)
        asserts.append(f"    assert {expr} == {expected_code}")

    return "\n".join(
        [
            f"def check({entry_point}):",
            *asserts,
            f"check({entry_point})",
            "",
        ]
    )


def build_test_block(requirement: str, entry_point: str) -> str:
    """
    Build MBPP-like `test` field.
    If no examples, still create a no-op check() to keep schema stable.
    """
    example = build_example_test(requirement, entry_point)
    if example.strip():
        body = "\n".join(example.splitlines()[1:-2])  # only assert lines
        return "\n".join(
            [
                "",
                "",
                "METADATA = {",
                "    'author': 'quixbugs',",
                "    'dataset': 'quixbugs'",
                "}",
                "",
                "",
                f"def check({entry_point}):",
                body if body.strip() else "    pass",
                "",
                f"check({entry_point})",
            ]
        )

    return "\n".join(
        [
            "",
            "",
            "METADATA = {",
            "    'author': 'quixbugs',",
            "    'dataset': 'quixbugs'",
            "}",
            "",
            "",
            f"def check({entry_point}):",
            "    # No doctest examples found in requirement; no executable assertions provided.",
            "    pass",
            "",
            f"check({entry_point})",
        ]
    )


# -----------------------------
# LLM classification
# -----------------------------
def _normalize_bug_type(s: str) -> Optional[str]:
    x = (s or "").strip().lower()
    x = x.replace("_", " ").replace("-", " ")
    x = re.sub(r"\s+", " ", x)

    # exact
    if x in BUG_TYPES:
        return x

    # common variants
    aliases = {
        "stack overflow": "stackoverflow",
        "wrong operator": "operator misuse",
        "operator error": "operator misuse",
        "value error": "value misuse",
        "wrong value": "value misuse",
        "missing condition": "missing logic",
        "missing step": "missing logic",
        "extra logic": "excess logic",
        "extra step": "excess logic",
        "variable error": "variable misuse",
        "wrong variable": "variable misuse",
        "function call misuse": "function misuse",
        "api misuse": "function misuse",
    }
    if x in aliases and aliases[x] in BUG_TYPES:
        return aliases[x]

    # heuristic contain
    if "operator" in x:
        return "operator misuse"
    if "value" in x or "constant" in x or "literal" in x or "numeric" in x:
        return "value misuse"
    if "variable" in x:
        return "variable misuse"
    if "function" in x or "call" in x or "api" in x:
        return "function misuse"
    if "excess" in x or "extra" in x or "redundant" in x:
        return "excess logic"
    if "missing" in x or "remove" in x or "deleted" in x:
        return "missing logic"

    return None


def _normalize_failure_symptoms(s: str) -> Optional[str]:
    x = (s or "").strip().lower()
    x = x.replace("_", " ").replace("-", " ")
    x = re.sub(r"\s+", " ", x)

    if x in FAILURE_SYMPTOMS:
        return x
    aliases = {
        "stack overflow": "stackoverflow",
        "recursion error": "stackoverflow",
        "infinite loop": "infinite loop",
        "non terminating": "infinite loop",
        "timeout": "infinite loop",
        "wrong output": "incorrect output",
        "incorrect": "incorrect output",
        "exception": "incorrect output",
        "runtime error": "incorrect output",
    }
    if x in aliases and aliases[x] in FAILURE_SYMPTOMS:
        return aliases[x]

    if "stack" in x or "recursion" in x:
        return "stackoverflow"
    if "loop" in x or "hang" in x or "timeout" in x or "non terminat" in x:
        return "infinite loop"
    if "incorrect" in x or "wrong" in x or "error" in x or "exception" in x:
        return "incorrect output"

    return None


def build_labeling_prompt(requirement: str, correct_code: str, buggy_code: str) -> str:
    """
    Strongly constrained JSON-only output.
    """
    return (
        "You are a software debugging expert.\n"
        "You will be given:\n"
        "1) a problem requirement/specification,\n"
        "2) a correct reference implementation,\n"
        "3) a buggy implementation produced by mutating the correct one.\n\n"
        "Task A — bug_type classification:\n"
        "Choose exactly ONE label from the following set:\n"
        f"{BUG_TYPES}\n\n"
        "Label definitions:\n"
        "- value misuse: wrong numeric literal / constant / threshold / index / base value.\n"
        "- operator misuse: wrong operator or comparator (e.g., + vs -, & vs ^, < vs <=).\n"
        "- variable misuse: wrong variable is used/updated, or variable update is incorrect.\n"
        "- function misuse: wrong function/method/API used, or wrong call/argument usage.\n"
        "- missing logic: essential branch/condition/update/base-case is removed or absent.\n"
        "- excess logic: extra/unnecessary operations/branches that change behavior.\n\n"
        "Task B — failure_symptoms classification:\n"
        "Choose exactly ONE label from:\n"
        f"{FAILURE_SYMPTOMS}\n\n"
        "Rules for failure_symptoms:\n"
        "- If buggy code likely does not terminate on valid inputs => infinite loop.\n"
        "- If buggy code likely causes unbounded recursion / RecursionError => stackoverflow.\n"
        "- Otherwise (including wrong results OR runtime exception) => incorrect output.\n\n"
        "Return STRICT JSON ONLY with exactly these keys:\n"
        '''{"bug_type": "<one of bug_type labels>", "failure_symptoms": "<one of failure_symptoms labels>"}\n\n"'''
        "=== REQUIREMENT ===\n"
        f"{requirement}\n\n"
        "=== CORRECT CODE ===\n"
        f"{correct_code}\n\n"
        "=== BUGGY CODE ===\n"
        f"{buggy_code}\n"
    )


def parse_llm_json(resp: str) -> Dict[str, str]:
    """
    Parse JSON-only response; tolerant to accidental wrapping text.
    """
    resp = (resp or "").strip()
    if not resp:
        raise ValueError("Empty LLM response")

    # try direct JSON
    try:
        obj = json.loads(resp)
        if isinstance(obj, dict):
            return obj  # type: ignore
    except Exception:
        pass

    # extract first {...}
    m = re.search(r"\{[\s\S]*\}", resp)
    if m:
        obj = json.loads(m.group(0))
        if isinstance(obj, dict):
            return obj  # type: ignore

    raise ValueError(f"Cannot parse JSON from LLM response: {resp[:200]}...")


def llm_label(
    model: OpenAIModel,
    requirement: str,
    correct_code: str,
    buggy_code: str,
    retries: int = 2,
    delay_s: float = 0.4,
) -> Tuple[str, str]:
    """
    Returns (bug_type, failure_symptoms) normalized.
    """
    last_err = None
    prompt = build_labeling_prompt(requirement, correct_code, buggy_code)

    for attempt in range(retries + 1):
        if attempt > 0:
            time.sleep(delay_s)

        resp = model.check_requirement(prompt)
        try:
            obj = parse_llm_json(resp)
            bt = _normalize_bug_type(str(obj.get("bug_type", "")))
            fs = _normalize_failure_symptoms(str(obj.get("failure_symptoms", "")))

            if bt not in BUG_TYPES or fs not in FAILURE_SYMPTOMS:
                raise ValueError(f"Out of schema: bug_type={bt}, failure_symptoms={fs}")

            return bt, fs

        except Exception as e:
            last_err = e
            # tighten the instruction for retry
            prompt = (
                "Your previous response did not follow the required JSON schema.\n"
                "Return STRICT JSON ONLY and ONLY using the allowed labels.\n"
                "No extra text.\n\n"
                + prompt
            )

    raise RuntimeError(f"LLM labeling failed after retries: {last_err}")


# -----------------------------
# Main pipeline
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="/path/to/thesis/FSE/dataset/quixbugs_pairs.json")
    ap.add_argument("--output", default="/path/to/thesis/FSE/dataset/quixbugs_buggy_solutions.jsonl")
    ap.add_argument("--model", default="gpt-4o")
    ap.add_argument("--delay_s", type=float, default=0.4, help="Sleep between LLM calls")
    ap.add_argument("--max_items", type=int, default=0, help="If >0, only process first N items")
    ap.add_argument("--resume", action="store_true", help="If set, skip task_ids already in output jsonl")
    args = ap.parse_args()

    data = load_json(args.input)

    openai_model = OpenAIModel(model_name=args.model)

    seen = load_existing_task_ids_jsonl(args.output) if args.resume else set()

    total = 0
    written = 0
    skipped = 0

    t0 = time.time()

    for idx, item in enumerate(data):
        if args.max_items and idx >= args.max_items:
            break

        total += 1
        task_id = f"Python/{idx}"  # keep consistent with your other datasets

        if task_id in seen:
            skipped += 1
            continue

        try:
            requirement = item.get("requirement", "") or ""
            correct_code_raw = item.get("correct_code", "") or ""
            mutate_code_raw = item.get("mutate_code", "") or ""

            # Extract main function from correct + buggy
            correct_fn = extract_first_function(correct_code_raw)
            buggy_fn = extract_first_function(mutate_code_raw)

            # Use correct entry_point as canonical
            entry_point = correct_fn.entry_point

            # Build MBPP-like prompt + tests
            prompt = build_prompt_from_requirement(correct_fn.declaration, requirement)
            example_test = build_example_test(requirement, entry_point)
            test = build_test_block(requirement, entry_point)

            # LLM labeling uses compact extracted code (less noise than raw strings)
            correct_for_llm = correct_fn.declaration + correct_fn.body
            buggy_for_llm = buggy_fn.declaration + buggy_fn.body

            bug_type, failure_symptoms = llm_label(
                model=openai_model,
                requirement=requirement,
                correct_code=correct_for_llm,
                buggy_code=buggy_for_llm,
                retries=2,
                delay_s=args.delay_s,
            )

            record = {
                "task_id": task_id,
                "prompt": prompt,
                "canonical_solution": correct_fn.body,
                "test": test,
                "text": requirement,
                "declaration": correct_fn.declaration,
                "example_test": example_test,
                "buggy_solution": buggy_fn.body,
                "bug_type": bug_type,
                "failure_symptoms": failure_symptoms,
                "entry_point": entry_point,
            }

            append_jsonl(args.output, record)
            written += 1

            # progress heartbeat
            if total == 1 or total % 5 == 0:
                elapsed = time.time() - t0
                rate = total / elapsed if elapsed > 0 else 0
                print(
                    f"[progress] {total} processed | {written} written | {skipped} skipped | "
                    f"rate={rate:.2f} items/s | last={item.get('id', '')} -> {bug_type}, {failure_symptoms}"
                )

            time.sleep(args.delay_s)

        except Exception:
            print(f"[WARN] Failed on task_id={task_id}, quix_id={item.get('id')}", file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)
            continue

    print("==== Done ====")
    print(f"input:  {args.input}")
    print(f"output: {args.output}")
    print(f"total processed: {total}")
    print(f"written: {written}")
    print(f"skipped (resume): {skipped}")


if __name__ == "__main__":
    main()

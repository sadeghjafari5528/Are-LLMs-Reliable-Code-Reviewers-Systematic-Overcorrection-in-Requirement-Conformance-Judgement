#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare a buggy-MBPP dataset with full buggy solutions reconstructed from Buggy-Aware-CodeLM.

Key idea (recommended over LLM completion):
    buggy_solution_full = buggy_prefix_body(from buggy_prompt) + canonical_solution_remainder

Output format (jsonl, one record per line) matches the user's schema:
{
  "task_id": "Python/0",
  "prompt": "...",
  "canonical_solution": "...",   # body-only
  "test": "...",
  "text": "...",                 # docstring content
  "declaration": "...",          # imports + def line (no docstring)
  "example_test": "...",         # doctest-derived mini checks (best-effort)
  "buggy_solution": "...",       # body-only
  "bug_type": "...",             # normalized
  "failure_symptoms": "...",     # normalized
  "entry_point": "func_name"
}

Usage:
  python prepare_buggy_mbpp_dataset.py \
    --input path/to/b_mbppplus.jsonl \
    --output ./mbpp_buggy_solutions.jsonl \
    --timeout_s 2.0

Notes:
- By default, this script keeps ONLY instances whose reconstructed buggy_solution fails the tests.
  This avoids "clean" instances in b-MBPP (the paper mentions b-MBPP contains clean instances).
- If you want to keep passing instances too, pass --keep_passing (then failure_symptoms will be "incorrect output" by definition is wrong, so we instead set it to "incorrect output" ONLY if you force it; default is safer).
"""

from __future__ import annotations

import argparse
import ast
import json
import multiprocessing as mp
import re
import sys
import textwrap
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


ALLOWED_BUG_TYPES = {
    "value misuse",
    "excess logic",
    "missing logic",
    "operator misuse",
    "variable misuse",
    "function misuse",
}

ALLOWED_FAILURE_SYMPTOMS = {
    "incorrect output",
    "infinite loop",
    "stackoverflow",
}


# Map Buggy-Aware-CodeLM injection "method" -> your normalized bug types.
# (Repo lists these injection methods; not all 6 categories necessarily appear in MBPP benchmarks.)
METHOD_TO_BUG_TYPE = {
    "NumericValueChangeNodeVisitor": "value misuse",
    "OperatorChangeNodeVisitor": "operator misuse",
    "VariableRenamingNodeVisitor": "variable misuse",
    # The following are best-effort semantic mappings:
    "KeywordRemovalTransformer": "missing logic",
    "ConditionRemovalNodeVisitor": "missing logic",
    "BranchRemovalNodeVisitor": "missing logic",
    "WhileToIfTransformer": "missing logic",
}


def load_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                yield json.loads(s)
            except Exception as e:
                raise ValueError(f"Invalid JSON on line {line_no} in {path}: {e}") from e


def dump_jsonl(path: str, records: Iterable[Dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False))
            f.write("\n")


def normalize_bug_type(method: Optional[str], existing: Optional[str] = None) -> str:
    """
    Normalize bug_type into the user's 6-category schema.
    Priority:
      1) if existing already matches allowed, keep it
      2) else map from method
      3) else heuristic fallback
    """
    if existing and existing in ALLOWED_BUG_TYPES:
        return existing

    if method:
        if method in METHOD_TO_BUG_TYPE:
            return METHOD_TO_BUG_TYPE[method]

        # heuristics for unexpected method names
        m = method.lower()
        if "numeric" in m or "value" in m or "constant" in m:
            return "value misuse"
        if "operator" in m:
            return "operator misuse"
        if "variable" in m or "rename" in m:
            return "variable misuse"
        if "function" in m or "call" in m:
            return "function misuse"
        if "insert" in m or "add" in m or "extra" in m:
            return "excess logic"
        if "remove" in m or "delete" in m or "branch" in m or "condition" in m or "while" in m:
            return "missing logic"

    # last resort
    return "missing logic"


def extract_declaration(prompt: str) -> str:
    """
    Extract "imports + def line" from the MBPP prompt.
    We stop after the first 'def ...:' line.
    """
    lines = prompt.splitlines(keepends=True)
    decl: List[str] = []
    def_pat = re.compile(r"^\s*def\s+\w+\s*\(.*\)\s*:\s*$")
    for line in lines:
        decl.append(line)
        if def_pat.match(line):
            break
    return "".join(decl)


def extract_docstring_text(prompt: str) -> str:
    """
    Extract docstring content between triple quotes.
    Returns empty string if not found.
    """
    m = re.search(r'"""(.*?)"""', prompt, flags=re.S)
    if not m:
        m = re.search(r"'''(.*?)'''", prompt, flags=re.S)
    if not m:
        return ""
    return m.group(1).strip("\n")


def _safe_literal_eval(s: str) -> Optional[Any]:
    try:
        return ast.literal_eval(s)
    except Exception:
        return None


def build_example_test_from_doctest(doc_text: str, func_name: str) -> str:
    """
    Best-effort conversion of doctest-style examples into a simple check() function.

    We look for patterns like:
        >>> f(1,2)
        3
    and convert them to:
        assert f(1,2) == 3

    If nothing can be parsed, return "".
    """
    if not doc_text.strip():
        return ""

    lines = [ln.rstrip() for ln in doc_text.splitlines()]
    pairs: List[Tuple[str, str]] = []

    i = 0
    while i < len(lines):
        ln = lines[i].strip()
        if ln.startswith(">>>"):
            expr = ln[len(">>>"):].strip()
            # find next non-empty line as expected
            j = i + 1
            while j < len(lines) and lines[j].strip() == "":
                j += 1
            if j < len(lines) and not lines[j].strip().startswith(">>>"):
                expected = lines[j].strip()
                pairs.append((expr, expected))
                i = j + 1
                continue
        i += 1

    if not pairs:
        return ""

    asserts: List[str] = []
    for expr, expected in pairs:
        # expected may be "True"/"False"/numbers/lists/strings etc.
        lit = _safe_literal_eval(expected)
        if lit is not None:
            expected_code = repr(lit)
        else:
            # fallback: compare to raw string (rare)
            expected_code = repr(expected)
        # keep expr as-is; it typically already calls the correct function name
        asserts.append(f"    assert {expr} == {expected_code}")

    check = [
        f"def check({func_name}):",
        *asserts,
        f"check({func_name})",
        "",
    ]
    return "\n".join(check)


def extract_body_after_docstring(code_with_def_and_doc: str) -> str:
    """
    Given code that includes def + docstring + (maybe) body,
    return the substring AFTER the first docstring block.

    If no docstring is found, return substring after first def line.
    """
    m = re.search(r'"""[\s\S]*?"""', code_with_def_and_doc)
    if not m:
        m = re.search(r"'''[\s\S]*?'''", code_with_def_and_doc)
    if m:
        return code_with_def_and_doc[m.end():]

    # fallback: after first def line
    lines = code_with_def_and_doc.splitlines(keepends=True)
    def_pat = re.compile(r"^\s*def\s+\w+\s*\(.*\)\s*:\s*$")
    out: List[str] = []
    seen_def = False
    for ln in lines:
        if not seen_def:
            if def_pat.match(ln):
                seen_def = True
            continue
        out.append(ln)
    return "".join(out)


def reconstruct_buggy_body(canonical_body: str, buggy_prompt: str) -> str:
    """
    Reconstruct a full buggy body (body-only) by:
      1) taking buggy body prefix from buggy_prompt (after docstring)
      2) aligning it against canonical_body prefix (linewise, tolerant to blank-line shifts)
      3) appending the remaining canonical_body lines as suffix

    This avoids calling an LLM and keeps the completion aligned with the reference solution.
    """
    buggy_prefix_body = extract_body_after_docstring(buggy_prompt)

    can_lines = canonical_body.splitlines(keepends=True)
    bug_lines = buggy_prefix_body.splitlines(keepends=True)

    # align bug_lines (prefix) against can_lines (prefix), allowing blank-line skipping
    i = 0  # index in canonical lines consumed
    j = 0  # index in buggy prefix lines consumed

    while j < len(bug_lines) and i < len(can_lines):
        b = bug_lines[j].rstrip("\n").rstrip()
        c = can_lines[i].rstrip("\n").rstrip()

        if b == c:
            i += 1
            j += 1
            continue

        # tolerate blank-line mismatches
        if (not c.strip()) and b.strip():
            i += 1
            continue
        if (not b.strip()) and c.strip():
            j += 1
            continue

        # treat as a transformed line at the same relative position
        i += 1
        j += 1

    prefix = "".join(bug_lines)
    suffix = "".join(can_lines[i:])

    # ensure newline boundary
    if prefix and (not prefix.endswith("\n")) and suffix and (not suffix.startswith("\n")):
        prefix += "\n"

    return prefix + suffix


@dataclass
class TestOutcome:
    status: str  # pass | assertion | recursion | exception | timeout
    detail: str


def _test_worker(q: mp.Queue, full_code: str, test_code: str, entry_point: str) -> None:
    env: Dict[str, Any] = {}
    try:
        exec(full_code, env)
        exec(test_code, env)

        fn = env.get(entry_point)
        if fn is None or not callable(fn):
            # fallback: find any callable with same name ignoring case
            for k, v in env.items():
                if callable(v) and k.lower() == entry_point.lower():
                    fn = v
                    break

        check = env.get("check")
        if callable(check):
            check(fn)
        # If no check() is present, we assume the test code ran assertions itself (rare).
        q.put(TestOutcome(status="pass", detail=""))
    except AssertionError as e:
        q.put(TestOutcome(status="assertion", detail=repr(e)))
    except RecursionError as e:
        q.put(TestOutcome(status="recursion", detail=repr(e)))
    except Exception:
        q.put(TestOutcome(status="exception", detail=traceback.format_exc()))


def run_tests_with_timeout(
    declaration: str,
    body: str,
    test_code: str,
    entry_point: str,
    timeout_s: float,
) -> TestOutcome:
    """
    Execute code+tests in a separate process to detect infinite loops via timeout.
    """
    full_code = declaration + body
    q: mp.Queue = mp.Queue()

    p = mp.Process(target=_test_worker, args=(q, full_code, test_code, entry_point))
    p.start()
    p.join(timeout=timeout_s)

    if p.is_alive():
        p.terminate()
        p.join(timeout=0.2)
        return TestOutcome(status="timeout", detail=f"timeout>{timeout_s}s")

    if not q.empty():
        return q.get()

    # If worker exited without writing outcome (should be rare), treat as exception
    return TestOutcome(status="exception", detail="No outcome returned from worker process.")


def normalize_failure_symptoms(outcome: TestOutcome) -> Optional[str]:
    """
    Normalize to one of:
      - incorrect output
      - infinite loop
      - stackoverflow

    Returns None if outcome is "pass".
    """
    if outcome.status == "pass":
        return None
    if outcome.status == "timeout":
        return "infinite loop"
    if outcome.status == "recursion":
        return "stackoverflow"
    # assertion / exception / syntax error etc.
    return "incorrect output"


def should_skip_as_clean(item: Dict[str, Any]) -> bool:
    """
    Heuristic: in b-MBPP, some instances are clean.
    If 'method' is missing/empty, we treat it as clean and skip by default.
    """
    method = (item.get("method") or "").strip()
    return method == ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to b_mbpp.jsonl or b_mbppplus.jsonl")
    ap.add_argument("--output", required=True, help="Output jsonl path")
    ap.add_argument("--timeout_s", type=float, default=2.0, help="Per-instance test timeout in seconds")
    ap.add_argument("--max_items", type=int, default=0, help="If >0, process only first N items")
    ap.add_argument(
        "--keep_passing",
        action="store_true",
        help="If set, keep instances even if reconstructed buggy_solution passes tests (NOT recommended).",
    )
    ap.add_argument(
        "--include_clean_instances",
        action="store_true",
        help="If set, do not skip items that look clean (method empty). Default skips them.",
    )

    args = ap.parse_args()

    input_path = args.input
    output_path = args.output

    out_records: List[Dict[str, Any]] = []
    total = 0
    kept = 0
    skipped_clean = 0
    skipped_passing = 0
    skipped_errors = 0

    for idx, item in enumerate(load_jsonl(input_path)):
        total += 1
        if args.max_items and total > args.max_items:
            break

        if (not args.include_clean_instances) and should_skip_as_clean(item):
            skipped_clean += 1
            continue

        try:
            prompt = item["prompt"]
            canonical_solution = item["canonical_solution"]
            test_code = item["test"]
            buggy_prompt = item["buggy_prompt"]
            entry_point = item.get("entry_point") or item.get("entry_point".lower())  # just in case
            if not entry_point:
                # attempt parse from def line in prompt
                m = re.search(r"^\s*def\s+(\w+)\s*\(", prompt, flags=re.M)
                entry_point = m.group(1) if m else "candidate"

            declaration = extract_declaration(prompt)
            text = extract_docstring_text(prompt)
            example_test = build_example_test_from_doctest(text, entry_point)

            buggy_solution_body = reconstruct_buggy_body(canonical_solution, buggy_prompt)

            outcome = run_tests_with_timeout(
                declaration=declaration,
                body=buggy_solution_body,
                test_code=test_code,
                entry_point=entry_point,
                timeout_s=args.timeout_s,
            )

            failure_symptoms = normalize_failure_symptoms(outcome)
            if failure_symptoms is None and (not args.keep_passing):
                skipped_passing += 1
                continue

            if failure_symptoms is None and args.keep_passing:
                # You asked for 3-category normalization; passing instances do not fit.
                # We keep the instance but set failure_symptoms to "incorrect output" only if forced.
                failure_symptoms = "incorrect output"

            assert failure_symptoms in ALLOWED_FAILURE_SYMPTOMS

            bug_type = normalize_bug_type(method=item.get("method"), existing=item.get("bug_type"))
            assert bug_type in ALLOWED_BUG_TYPES

            out_records.append(
                {
                    "task_id": f"Python/{kept}",
                    "prompt": prompt,
                    "canonical_solution": canonical_solution,
                    "test": test_code,
                    "text": text,
                    "declaration": declaration,
                    "example_test": example_test,
                    "buggy_solution": buggy_solution_body,
                    "bug_type": bug_type,
                    "failure_symptoms": failure_symptoms,
                    "entry_point": entry_point,
                }
            )
            kept += 1

        except Exception:
            skipped_errors += 1
            print(f"[WARN] Failed to process item #{idx}: {item.get('instance_id', '(no instance_id)')}", file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)
            continue

    dump_jsonl(output_path, out_records)

    print("==== Done ====")
    print(f"input: {input_path}")
    print(f"output: {output_path}")
    print(f"total read: {total}")
    print(f"kept: {kept}")
    print(f"skipped clean: {skipped_clean}")
    print(f"skipped passing: {skipped_passing}")
    print(f"skipped errors: {skipped_errors}")


if __name__ == "__main__":
    main()

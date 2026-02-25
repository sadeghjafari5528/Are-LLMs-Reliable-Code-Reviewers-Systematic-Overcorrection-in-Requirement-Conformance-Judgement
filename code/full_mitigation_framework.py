#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""full_mitigation_framework.py

Fix-guided Verification Filter for FULL prompting.

This script runs a judge model to produce a YES/NO verdict and (if NO) a proposed fix.
When the trigger condition holds (verdict=NO and a fix is present), we apply a
verification filter that executes both the original and fixed implementations on:
  (1) the benchmark's reference unit tests, and
  (2) a GPT-4o generated spec-constrained augmented input set.

The filter then updates the verdict based on executable evidence (four cases).
Outputs are logged to a resume-safe JSONL checkpoint and a final JSON array.

WARNING: This script executes model-generated code in a local sandbox process.
Run it inside an isolated environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import textwrap
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import multiprocessing as mp
from tqdm import tqdm

from utils import load_data, construct_prompt, parse_response  # type: ignore
from model import OpenAIModel, GoogleModel, AnthropicModel, OpenRouterModel  # type: ignore

try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore


# ----------------------------
# Small utilities: JSONL checkpointing (resume-safe)
# ----------------------------
def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read JSONL tolerantly (skips corrupted/partial lines)."""
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
            continue
    return rows


def _preview(obj: Any, max_chars: int = 400) -> str:
    """Best-effort short string for logging; always JSON-safe."""
    try:
        s = repr(obj)
    except Exception:
        s = f"<unreprable:{type(obj).__name__}>"
    s = s.replace('\n', '\\n')
    if len(s) > max_chars:
        s = s[:max_chars] + '...'
    return s


def to_jsonable(obj: Any, *, max_depth: int = 8, max_list: int = 80, max_str: int = 8000) -> Any:
    """Recursively convert obj into something that json.dumps can always handle.

    This prevents checkpoint crashes when runtime values include tuples/sets/custom objects
    or dicts with non-string keys (common when logging function outputs).
    """
    if max_depth <= 0:
        return _preview(obj)

    # Primitives
    if obj is None or isinstance(obj, (str, int, float, bool)):
        if isinstance(obj, str) and len(obj) > max_str:
            return obj[:max_str] + '...'
        return obj

    # Containers
    if isinstance(obj, (list, tuple, set)):
        seq = list(obj)
        out = [to_jsonable(x, max_depth=max_depth-1, max_list=max_list, max_str=max_str) for x in seq[:max_list]]
        if len(seq) > max_list:
            out.append(f"<truncated +{len(seq)-max_list} items>")
        return out

    if isinstance(obj, dict):
        new: Dict[str, Any] = {}
        items = list(obj.items())
        for k, v in items[:max_list]:
            # JSON requires string keys
            if isinstance(k, (str, int, float, bool)) or k is None:
                kk = str(k) if not isinstance(k, str) else k
            else:
                kk = _preview(k, max_chars=200)
            if kk in new:
                kk = f"{kk}__dup{len(new)}"
            new[kk] = to_jsonable(v, max_depth=max_depth-1, max_list=max_list, max_str=max_str)
        if len(items) > max_list:
            new['<truncated_keys>'] = f"+{len(items)-max_list} more keys"
        return new

    # Fallback
    return _preview(obj)


def append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    """Append one JSON object per line. Never crashes on serialization errors."""
    path.parent.mkdir(parents=True, exist_ok=True)
    safe = to_jsonable(obj)
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(safe, ensure_ascii=False) + '\n')



CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*(.*?)```", flags=re.IGNORECASE | re.DOTALL)
DEF_RE = re.compile(r"^\s*def\s+([A-Za-z_]\w*)\s*\(", flags=re.MULTILINE)


# ----------------------------
# Dataset expansion
# ----------------------------
def extract_func_name(declaration: str) -> str:
    m = DEF_RE.search(declaration or "")
    return m.group(1) if m else ""


def expand_x_bugs(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    for item in data:
        requirement = item.get("text", item.get("prompt", ""))
        decl = item.get("declaration", "")
        fn = extract_func_name(decl)

        samples.append({
            "task_id": item["task_id"],
            "variant": "canonical",
            "label": 1,
            "requirement": requirement,
            "declaration": decl,
            "func_name": fn,
            "code": (decl or "") + (item.get("canonical_solution") or ""),
            "benchmark_test": item.get("test", ""),
            "example_test": item.get("example_test", ""),
        })

        samples.append({
            "task_id": item["task_id"],
            "variant": "buggy",
            "label": 0,
            "requirement": requirement,
            "declaration": decl,
            "func_name": fn,
            "code": (decl or "") + (item.get("buggy_solution") or ""),
            "benchmark_test": item.get("test", ""),
            "example_test": item.get("example_test", ""),
            "bug_type": item.get("bug_type", "unknown"),
            "failure_symptoms": item.get("failure_symptoms", "unknown"),
        })
    return samples


# ----------------------------
# Fix extraction / assembly
# ----------------------------
def extract_fix_code(raw: str) -> str:
    if not isinstance(raw, str):
        return ""
    blocks = CODE_BLOCK_RE.findall(raw)
    if blocks:
        return blocks[-1].strip()
    m = DEF_RE.search(raw)
    if m:
        return raw[m.start():].strip()
    return ""


def assemble_function_code(declaration: str, body_or_full: str) -> str:
    s = (body_or_full or "").strip("\n")
    if DEF_RE.search(s):
        return s + "\n"
    decl = declaration or ""
    if decl and not decl.endswith("\n"):
        decl += "\n"
    body = textwrap.dedent(s).strip("\n") or "pass"
    body = textwrap.indent(body, "    ")
    return decl + body + "\n"


# ----------------------------
# Sandbox execution with timeouts
# ----------------------------
@dataclass
class ExecResult:
    ok: bool
    status: str
    error: str = ""


def _worker_run_benchmark(code: str, test_code: str, q: mp.Queue) -> None:
    try:
        env: Dict[str, Any] = {}
        exec(code, env, env)
        exec(test_code, env, env)
        q.put({"status": "ok"})
    except Exception as e:
        q.put({"status": "exception", "error": repr(e)})


def run_benchmark_tests(code: str, test_code: str, timeout_s: float = 3.0) -> ExecResult:
    q: mp.Queue = mp.Queue()
    p = mp.Process(target=_worker_run_benchmark, args=(code, test_code, q))
    p.start()
    p.join(timeout=timeout_s)
    if p.is_alive():
        p.terminate()
        p.join()
        return ExecResult(ok=False, status="timeout", error=f"timeout>{timeout_s}s")
    if q.empty():
        return ExecResult(ok=False, status="exception", error="no result")
    out = q.get()
    if out.get("status") == "ok":
        return ExecResult(ok=True, status="ok")
    return ExecResult(ok=False, status="exception", error=str(out.get("error", "")))


@dataclass
class CallResult:
    ok: bool
    status: str
    value: Any = None
    error: str = ""


def _worker_call_func(code: str, func_name: str, args: List[Any], q: mp.Queue) -> None:
    try:
        env: Dict[str, Any] = {}
        exec(code, env, env)
        fn = env.get(func_name)
        if fn is None:
            raise NameError(f"Function {func_name} not found")
        q.put({"status": "ok", "value": fn(*args)})
    except Exception as e:
        q.put({"status": "exception", "error": repr(e)})


def call_func(code: str, func_name: str, args: List[Any], timeout_s: float = 2.0) -> CallResult:
    q: mp.Queue = mp.Queue()
    p = mp.Process(target=_worker_call_func, args=(code, func_name, args, q))
    p.start()
    p.join(timeout=timeout_s)
    if p.is_alive():
        p.terminate()
        p.join()
        return CallResult(ok=False, status="timeout", error=f"timeout>{timeout_s}s")
    if q.empty():
        return CallResult(ok=False, status="exception", error="no result")
    out = q.get()
    if out.get("status") == "ok":
        return CallResult(ok=True, status="ok", value=out.get("value"))
    return CallResult(ok=False, status="exception", error=str(out.get("error", "")))


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def deep_equal(a: Any, b: Any, *, rel_tol: float = 1e-9, abs_tol: float = 1e-9) -> bool:
    if _is_number(a) and _is_number(b):
        if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
            return True
        return math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=abs_tol)
    if type(a) != type(b):
        return False
    if isinstance(a, (str, bool, type(None))):
        return a == b
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(deep_equal(x, y, rel_tol=rel_tol, abs_tol=abs_tol) for x, y in zip(a, b))
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(deep_equal(a[k], b[k], rel_tol=rel_tol, abs_tol=abs_tol) for k in a.keys())
    return a == b


# ----------------------------
# GPT-4o augmented input generation (diff-guided)
# ----------------------------
@dataclass
class AugCase:
    args: List[Any]
    why: str


class AugInputGenerator:
    """
    Generate spec-constrained *inputs* that are likely to distinguish original vs fix,
    while staying within the task spec.
    """

    def __init__(self, model_name: str = "gpt-4o", cache_path: Optional[str] = None):
        if OpenAI is None:
            raise ImportError("openai SDK not installed. pip install openai>=1.0.0")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY not set")

        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        self.cache_path = Path(cache_path) if cache_path else None
        self.cache: Dict[str, List[AugCase]] = {}
        if self.cache_path and self.cache_path.exists():
            self._load_cache()

        self.system = (
            "You generate additional SPEC-CONSTRAINED test inputs for a Python function.\n"
            "You are given: spec + examples + (original code, proposed fix).\n"
            "Goal: propose inputs that are still implied by the spec BUT are likely to reveal behavioral differences.\n"
            "Rules:\n"
            "- Do NOT introduce requirements not implied by spec/examples.\n"
            "- Inputs must be within the described domain.\n"
            "- Return JSON only.\n"
        )

    def _key(self, task_id: str, spec: str, example_test: str, orig: str, fix: str, n: int) -> str:
        obj = {"task_id": task_id, "spec": spec, "ex": example_test, "orig": orig, "fix": fix, "n": n}
        s = json.dumps(obj, ensure_ascii=False)
        return hashlib.sha1(s.encode("utf-8")).hexdigest()

    def _load_cache(self) -> None:
        assert self.cache_path is not None
        for line in self.cache_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except Exception:
                # tolerate partial/corrupted lines if the process was interrupted
                continue
            k = o.get("key")
            tests = o.get("tests", [])
            if not k or not isinstance(tests, list):
                continue
            parsed: List[AugCase] = []
            for t in tests:
                if not isinstance(t, dict):
                    continue
                args = t.get("args")
                if isinstance(args, list):
                    parsed.append(AugCase(args=args, why=str(t.get("why", ""))))
            if parsed:
                self.cache[k] = parsed

    def _append_cache(self, key: str, cases: List[AugCase]) -> None:
        if not self.cache_path:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"key": key, "tests": [{"args": c.args, "why": c.why} for c in cases]}
        with self.cache_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    @staticmethod
    def _truncate(s: str, max_chars: int = 1800) -> str:
        s = s or ""
        return s if len(s) <= max_chars else (s[:max_chars] + "\n# ... truncated ...\n")

    def generate(
        self,
        *,
        task_id: str,
        spec: str,
        example_test: str,
        orig_code: str,
        fix_code: str,
        n: int = 12,
    ) -> List[AugCase]:
        k = self._key(task_id, spec, example_test, orig_code, fix_code, n)
        if k in self.cache:
            return self.cache[k]

        payload = {
            "task_id": task_id,
            "spec": spec,
            "examples": example_test,
            "original_code": self._truncate(orig_code),
            "proposed_fix": self._truncate(fix_code),
            "n": n,
            "output_format": {"tests": [{"args": [], "why": "cite spec/examples to justify this input"}]},
            "instructions": (
                "Generate exactly n additional test INPUTS (positional args only).\n"
                "Each input must be justified by citing a constraint/behavior from spec/examples.\n"
                "Prefer boundary/edge cases and cases likely to distinguish original vs fix.\n"
                "Return JSON: {tests:[{args:[...], why:'...'}]}."
            ),
        }

        # NOTE: even in JSON mode, responses can be truncated if max_tokens is too small,
        # resulting in invalid JSON. We therefore (1) request concise rationales and
        # (2) robustly parse with retries.
        def _extract_json_obj(s: str) -> Optional[Dict[str, Any]]:
            s = (s or "").strip()
            if not s:
                return None
            # Fast path
            try:
                obj = json.loads(s)
                return obj if isinstance(obj, dict) else None
            except Exception:
                pass

            # Strip common markdown fences
            if s.startswith("```"):
                m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", s, flags=re.DOTALL | re.IGNORECASE)
                if m:
                    cand = m.group(1).strip()
                    try:
                        obj = json.loads(cand)
                        return obj if isinstance(obj, dict) else None
                    except Exception:
                        pass

            # Heuristic: first '{' to last '}'
            i = s.find("{")
            j = s.rfind("}")
            if 0 <= i < j:
                cand = s[i : j + 1]
                try:
                    obj = json.loads(cand)
                    return obj if isinstance(obj, dict) else None
                except Exception:
                    return None
            return None

        # Encourage very short justifications to prevent truncation
        payload["instructions"] += "\n- Keep each why <= 20 words."

        data: Optional[Dict[str, Any]] = None
        max_tokens = 1500
        for attempt in range(2):
            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
                temperature=0,
                response_format={"type": "json_object"},
                max_tokens=max_tokens,
            )
            txt = resp.choices[0].message.content or ""
            data = _extract_json_obj(txt)
            if data is not None:
                break

            # If the model hit the token limit, retry with a larger max_tokens.
            finish_reason = getattr(resp.choices[0], "finish_reason", None)
            if finish_reason == "length":
                max_tokens = min(3000, max_tokens + 1000)
                continue

            # Otherwise, retry once with a stricter reminder.
            payload["instructions"] += "\n- Output MUST be a single valid JSON object. No trailing commas."

        if data is None:
            # Fail closed: no augmented tests.
            return []
        tests = data.get("tests", [])
        out: List[AugCase] = []
        if isinstance(tests, list):
            for t in tests:
                if not isinstance(t, dict):
                    continue
                args = t.get("args")
                if not isinstance(args, list):
                    continue
                out.append(AugCase(args=args, why=str(t.get("why", ""))))

        out = out[:n]
        if out:
            self.cache[k] = out
            self._append_cache(k, out)
        return out


# ----------------------------
# Filter logic
# ----------------------------
@dataclass
class FilterOutcome:
    final_pred: str
    case: str
    details: Dict[str, Any]


def apply_filter(
    *,
    sample: Dict[str, Any],
    fix_extracted: str,
    aug_gen: Optional[AugInputGenerator],
    enable_aug: bool,
    n_aug: int,
    max_rounds: int,
    timeout_base: float,
    timeout_call: float,
    flip_case3: bool,
    fallback_to_base_if_inconclusive: bool,
) -> FilterOutcome:

    declaration = sample["declaration"]
    func_name = sample["func_name"]
    orig_code = sample["code"]
    fix_code = assemble_function_code(declaration, fix_extracted)

    base_orig = run_benchmark_tests(orig_code, sample["benchmark_test"], timeout_s=timeout_base)
    base_fix = run_benchmark_tests(fix_code,  sample["benchmark_test"], timeout_s=timeout_base)

    details: Dict[str, Any] = {
        "base_orig": base_orig.__dict__,
        "base_fix": base_fix.__dict__,
    }

    # Case 4 (base): orig fails, fix passes -> bug fixed -> keep NO
    if (not base_orig.ok) and base_fix.ok:
        return FilterOutcome(final_pred="NO", case="case4_repair_success_base", details=details)

    # Case 3 (base): orig passes, fix fails -> ambiguous; conservative default keeps NO
    if base_orig.ok and (not base_fix.ok):
        if flip_case3:
            return FilterOutcome(final_pred="YES", case="case3_overrepair_base_flip", details=details)
        return FilterOutcome(final_pred="NO", case="case3_fix_failed_base_keep_no", details=details)

    # Both fail base -> keep NO
    if (not base_orig.ok) and (not base_fix.ok):
        return FilterOutcome(final_pred="NO", case="both_fail_base_keep_no", details=details)

    # Both pass base
    if not enable_aug or aug_gen is None:
        if fallback_to_base_if_inconclusive:
            return FilterOutcome(final_pred="YES", case="no_aug_fallback_base_flip_yes", details=details)
        return FilterOutcome(final_pred="NO", case="no_aug_keep_no", details=details)

    aug_round_logs: List[Dict[str, Any]] = []
    details["aug_rounds"] = aug_round_logs

    for rd in range(max_rounds):
        cases = aug_gen.generate(
            task_id=sample["task_id"],
            spec=sample["requirement"],
            example_test=sample["example_test"],
            orig_code=orig_code,
            fix_code=fix_code,
            n=n_aug,
        )

        log: Dict[str, Any] = {
            "round": rd,
            "n_aug": len(cases),
            "example_inputs": [{"args": c.args, "why": c.why} for c in cases[:3]],
        }

        if not cases:
            aug_round_logs.append(log)
            continue

        n_ok = 0
        n_equal = 0
        n_orig_err = 0
        n_fix_err = 0
        mismatches: List[Dict[str, Any]] = []
        err_examples: List[Dict[str, Any]] = []

        for c in cases:
            r1 = call_func(orig_code, func_name, c.args, timeout_s=timeout_call)
            r2 = call_func(fix_code,  func_name, c.args, timeout_s=timeout_call)

            if not r1.ok:
                n_orig_err += 1
                if len(err_examples) < 2:
                    err_examples.append({"where": "orig", "args": c.args, "status": r1.status, "error": r1.error})
            if not r2.ok:
                n_fix_err += 1
                if len(err_examples) < 2:
                    err_examples.append({"where": "fix", "args": c.args, "status": r2.status, "error": r2.error})

            if r1.ok and r2.ok:
                n_ok += 1
                if deep_equal(r1.value, r2.value):
                    n_equal += 1
                else:
                    if len(mismatches) < 3:
                        mismatches.append({"args": c.args, "orig": r1.value, "fix": r2.value})

        log.update({
            "n_ok": n_ok,
            "n_equal": n_equal,
            "n_orig_err": n_orig_err,
            "n_fix_err": n_fix_err,
            "mismatch_examples": mismatches,
            "error_examples": err_examples,
        })
        aug_round_logs.append(log)

        # Case 4 (aug): orig errors but fix ok -> keep NO
        if n_fix_err == 0 and n_orig_err > 0:
            return FilterOutcome(final_pred="NO", case="case4_repair_success_aug", details=details)

        # Case 3 (aug): fix errors but orig ok -> ambiguous (conservative default keeps NO)
        if n_orig_err == 0 and n_fix_err > 0:
            if flip_case3:
                return FilterOutcome(final_pred="YES", case="case3_overrepair_aug_flip", details=details)
            return FilterOutcome(final_pred="NO", case="case3_fix_failed_aug_keep_no", details=details)

        # Case 1: all ok and all equal -> flip YES
        if n_ok == len(cases) and n_equal == len(cases):
            return FilterOutcome(final_pred="YES", case="case1_equivalent_behavior_flip_yes", details=details)

        # If we found any behavioral differences on spec-supported inputs -> keep NO
        if mismatches:
            return FilterOutcome(final_pred="NO", case="behavior_diff_keep_no", details=details)

        # Otherwise: inconclusive -> regenerate next round
        # (Typically because too many timeouts/errors for both sides, or too few comparable runs.)
        continue

    # After K rounds, still inconclusive.
    if fallback_to_base_if_inconclusive:
        return FilterOutcome(final_pred="YES", case="case2_inconclusive_fallback_base_flip_yes", details=details)
    return FilterOutcome(final_pred="NO", case="case2_inconclusive_keep_no", details=details)


# ----------------------------
# Metrics for analysis
# ----------------------------
@dataclass
class Rates:
    tp: int
    fn: int
    fp: int
    tn: int

    @property
    def fnr(self) -> float:
        return self.fn / (self.tp + self.fn) if (self.tp + self.fn) else float("nan")

    @property
    def fpr(self) -> float:
        return self.fp / (self.fp + self.tn) if (self.fp + self.tn) else float("nan")


def compute_rates(rows: List[Dict[str, Any]], pred_field: str) -> Rates:
    tp = fn = fp = tn = 0
    for r in rows:
        y = r.get("label")
        p = r.get(pred_field, "")
        if y not in (0, 1) or p not in ("YES", "NO"):
            continue
        if y == 1 and p == "YES":
            tp += 1
        elif y == 1 and p == "NO":
            fn += 1
        elif y == 0 and p == "YES":
            fp += 1
        elif y == 0 and p == "NO":
            tn += 1
    return Rates(tp, fn, fp, tn)


# ----------------------------
# Model factory (same set you used)
# ----------------------------
def build_judge(model_name: str):
    if model_name == "gpt-4o":
        return OpenAIModel(model_name="gpt-4o")
    if model_name == "gemini-2.0-flash":
        return GoogleModel(model_name="gemini-2.0-flash")
    if model_name in ("claude-3-5-sonnet-20240620", "claude-haiku-4-5", "Claude-3.5-sonnet"):
        return AnthropicModel(model_name="claude-3-5-sonnet-20240620")
    if model_name.startswith("meta-llama/") or model_name.startswith("mistralai/"):
        return OpenRouterModel(model_name=model_name)
    raise ValueError(f"Unknown model: {model_name}")


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="*_bugs.jsonl path")
    ap.add_argument("--benchmark", default="humaneval", help="bookkeeping only")
    ap.add_argument("--model", required=True, help="judge model name")
    ap.add_argument("--out", required=True, help="output JSON path")
    ap.add_argument("--delay", type=float, default=0.5)

    # framework options
    ap.add_argument("--enable_filter", action="store_true")
    ap.add_argument("--enable_aug", action="store_true")
    ap.add_argument("--n_aug", type=int, default=12)
    ap.add_argument("--max_rounds", type=int, default=2)
    ap.add_argument("--timeout_base", type=float, default=3.0)
    ap.add_argument("--timeout_call", type=float, default=2.0)
    ap.add_argument("--aug_cache", default="", help="optional JSONL cache for augmented inputs")

    # IMPORTANT knobs
    ap.add_argument("--flip_case3", action="store_true",
                    help="aggressive: allow NO→YES flip when original passes tests but fix fails (can increase FPR)")
    ap.add_argument("--fallback_to_base", action="store_true",
                    help="if augmented testing is inconclusive after K rounds, fall back to base tests and flip YES")

    args = ap.parse_args()

    try:
        mp.set_start_method("fork")
    except Exception:
        pass

    data = load_data(args.dataset)
    samples = expand_x_bugs(data)

    judge = build_judge(args.model)
    aug_gen = None
    if args.enable_filter and args.enable_aug:
        aug_gen = AugInputGenerator(model_name="gpt-4o", cache_path=(args.aug_cache or None))

    # ----------------------------
    # Resume-safe checkpointing
    # ----------------------------
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    # Always checkpoint to JSONL (append-only) to survive crashes.
    ckpt = outp if outp.suffix.lower() == ".jsonl" else outp.with_suffix(".jsonl")
    rows: List[Dict[str, Any]] = read_jsonl(ckpt)
    done = {(r.get("task_id"), r.get("variant")) for r in rows if r.get("task_id") and r.get("variant")}
    if rows:
        print(f"[RESUME] Loaded {len(rows)} rows from checkpoint: {ckpt}")

    t0 = time.time()

    newly_written = 0
    for idx, s in enumerate(tqdm(samples, desc=f"Judge={args.model}")):
        key = (s.get("task_id"), s.get("variant"))
        if key in done:
            continue
        prompt = construct_prompt(s["requirement"], s["code"], mode="full")

        raw = ""
        pred = ""
        fix = ""
        final_pred = ""
        filter_case = ""
        filter_details: Dict[str, Any] = {}

        try:
            raw = judge.check_requirement(prompt)
            pred = parse_response(raw)
            final_pred = pred
            if pred == "NO":
                fix = extract_fix_code(raw)
        except Exception as e:
            raw = f"ERROR: {repr(e)}"
            pred = ""
            final_pred = ""
            print(f"[ERROR] call failed task={s['task_id']} variant={s['variant']}")
            print(traceback.format_exc())

        # Apply framework filter
        if args.enable_filter and pred == "NO" and fix.strip():
            try:
                outc = apply_filter(
                    sample=s,
                    fix_extracted=fix,
                    aug_gen=aug_gen,
                    enable_aug=bool(args.enable_aug),
                    n_aug=int(args.n_aug),
                    max_rounds=int(args.max_rounds),
                    timeout_base=float(args.timeout_base),
                    timeout_call=float(args.timeout_call),
                    flip_case3=bool(args.flip_case3),
                    fallback_to_base_if_inconclusive=bool(args.fallback_to_base),
                )
                final_pred = outc.final_pred
                filter_case = outc.case
                filter_details = outc.details
            except Exception as e:
                # Fail closed: keep original prediction (usually NO)
                final_pred = pred
                filter_case = "filter_error"
                filter_details = {"error": repr(e)}

        row = {
            "benchmark": args.benchmark,
            "model": args.model,
            "mode": "full",
            "task_id": s["task_id"],
            "variant": s["variant"],
            "label": s["label"],
            "pred": pred,
            "raw": raw,
            "fix_extracted": fix,
            "final_pred": final_pred,
            "filter_case": filter_case,
            "filter_details": filter_details,
        }
        row_safe = to_jsonable(row)
        rows.append(row_safe)
        append_jsonl(ckpt, row_safe)
        newly_written += 1
        done.add(key)

        if args.delay:
            time.sleep(args.delay)
        if (idx + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"[{args.model}] processed={idx+1}/{len(samples)} | newly_written={newly_written} | elapsed={elapsed/60:.1f} min")

    # Write final JSON (array) for convenient downstream analysis.
    final_json = outp if outp.suffix.lower() == ".json" else outp.with_suffix(".json")
    final_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[DONE] wrote checkpoint: {ckpt}")
    print(f"[DONE] wrote final json : {final_json}")

    before = compute_rates(rows, "pred")
    after  = compute_rates(rows, "final_pred")
    print(f"Before: TP={before.tp} FN={before.fn} FP={before.fp} TN={before.tn} | FNR={before.fnr*100:.2f}% FPR={before.fpr*100:.2f}%")
    print(f"After : TP={after.tp} FN={after.fn} FP={after.fp} TN={after.tn} | FNR={after.fnr*100:.2f}% FPR={after.fpr*100:.2f}%")


if __name__ == "__main__":
    main()

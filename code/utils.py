"""utils.py

Shared utilities:
- load_data: read JSON/JSONL datasets
- construct_prompt: build prompts for {direct, direct_explain, full, generated_tests}
- parse_response: robust YES/NO extraction
- run_generated_tests: execute model-generated tests with per-test timeouts
- save_json: convenience writer

Note: Prompt templates are kept *compatible* with the original experiments.
If you change prompt wording, results may differ.
"""

from __future__ import annotations

import json
import ast
import builtins
import math
import multiprocessing as mp
import re
import reprlib
from pathlib import Path
from queue import Empty
from typing import Any, Dict, List, Literal, Optional, Tuple


PromptMode = Literal["direct", "direct_explain", "full", "generated_tests"]


class MissingGeneratedModuleError(ModuleNotFoundError):
    """A module needed to execute generated or evaluated code is unavailable."""


def _raise_missing_module(result: Dict[str, Any]) -> None:
    if result.get("status") == "missing_module":
        name = result.get("module")
        raise MissingGeneratedModuleError(
            f"{result['error']}. Install the required package in the Python environment running "
            "this script and rerun; an import name may differ from its package name.",
            name=name,
        )


def load_data(path: str) -> List[Dict[str, Any]]:
    """Load a dataset from JSON (.json) or JSON Lines (.jsonl)."""
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        if p.suffix.lower() == ".jsonl":
            return [json.loads(line) for line in f if line.strip()]
        return json.load(f)


def construct_prompt(requirement: str, code: str, mode: PromptMode) -> str:
    """Construct the judge prompt.

    The three prompt modes correspond to the paper:
    - direct: verdict only
    - direct_explain: verdict + rationale
    - full: verdict + rationale + fix (if NO)
    """

    base = (
        "You are a senior software engineer specializing in code review and requirement conformance.\n\n"
        "Below is a requirement describing what a Python function is expected to do:\n\n"
        f"{requirement}\n\n"
        "And here is the implementation of the function:\n\n"
        f"```python\n{code}\n```\n\n"
    )

    if mode == "direct":
        return base + (
            "Please ignore any indentation error and lack of type hinting! Does the code satisfy the requirement?\n\n"
            "Answer only with YES or NO."
        )

    if mode == "direct_explain":
        return base + (
            "Step 1: Conformance Check\n\n"
            "Please ignore any indentation error and lack of type hinting! Does the code satisfy the requirement?\n\n"
            "Answer only with YES or NO.\n\n"
            "Step 2: Justification\n\n"
            "If your answer was NO, explain why the code does not meet the requirement.\n\n"
            "Be specific and refer to particular lines, conditions, or behaviors."
        )

    if mode == "full":
        # Keep the original structure used in experiments (verdict -> justification -> fix).
        return base + (
            "Step 1: Conformance Check\n\n"
            "Please ignore any indentation error and lack of type hinting! Does the code satisfy the requirement?\n\n"
            "Answer only with YES or NO.\n\n"
            "Step 2: Justification\n\n"
            "If your answer was NO, explain why the code does not meet the requirement.\n\n"
            "Be specific and refer to particular lines, conditions, or behaviors.\n\n"
            "Step 3: Fix\n\n"
            "If your answer was NO, please fix the code so that it meets the requirement.\n"
            "Return the fixed code in a single Python code block."
        )

    if mode == "generated_tests":
        return base + (
            "Generate 5 to 10 focused Python tests for this requirement and implementation.\n"
            "The tests must be executable after the implementation has been defined.\n"
            "Use plain Python functions named test_1, test_2, and so on; each function must "
            "raise an AssertionError when the implementation is incorrect.\n"
            "Cover normal cases, boundary cases, and relevant invalid or empty inputs when "
            "the requirement specifies their behavior.\n"
            "Do not redefine or modify the implementation.\n"
            "Return only the test functions in one Python code block, with no explanation."
        )

    raise ValueError(f"Unknown mode: {mode}")


def construct_signature_prompt(requirement: str, code: str) -> str:
    """Ask the model for the entry-point signature plus its output type and format."""
    return (
        "You are a senior Python engineer.\n\n"
        f"Requirement:\n{requirement}\n\n"
        f"Implementation:\n```python\n{code}\n```\n\n"
        "Identify the function that implements the requirement (the entry point, not a helper). "
        "Please ignore any indentation error.\n"
        "Step 1: Write its signature. Copy the function name, parameter names and order, "
        "default values, and parameter type annotations exactly as written in the implementation. "
        "Do not add, remove, rename, or reorder parameters. Include a return annotation: copy it "
        "if present, otherwise infer the precise return type from the requirement and implementation. "
        "Do not include the body, docstring, decorators, or imports.\n"
        "Step 2: Describe the output type and format of the returned value in one line: the exact "
        "type, its structure and element types, and any format details (for example ordering, "
        "rounding, string format, or what is returned in edge cases).\n"
        "Respond in exactly this form, with nothing else:\n"
        "```python\ndef name(first, second=1) -> list[str]:\n```\n"
        "OUTPUT: a list of strings sorted in ascending order; an empty list if there is no match"
    )


def _parameter_names(args: ast.arguments) -> List[str]:
    names = [arg.arg for arg in args.posonlyargs + args.args]
    if args.vararg:
        names.append("*" + args.vararg.arg)
    names += [arg.arg for arg in args.kwonlyargs]
    if args.kwarg:
        names.append("**" + args.kwarg.arg)
    return names


def _first_function(source: str) -> Optional[ast.FunctionDef]:
    for text in (source, source.rstrip() + "\n    pass\n"):
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                return node
    return None


def parse_function_signature(response: str, submitted_code: str) -> Tuple[str, str]:
    """Normalize the model's signature and output format and check them against the submitted code."""
    output_match = re.search(r"(?im)^[ \t]*OUTPUT[ \t]*:[ \t]*(\S.*?)[ \t]*$", response)
    if output_match is None:
        raise ValueError("model did not describe the output type and format")
    output_format = output_match.group(1)
    node = _first_function(extract_code_block(response.replace(output_match.group(0), "")))
    if node is None:
        raise ValueError("model did not return a valid function signature")
    if node.returns is None:
        raise ValueError("model did not specify the return type in the signature")
    signature = f"def {node.name}({ast.unparse(node.args)}) -> {ast.unparse(node.returns)}:"

    try:
        tree = ast.parse(submitted_code)
    except SyntaxError:
        if not re.search(rf"\bdef\s+{re.escape(node.name)}\s*\(", submitted_code):
            raise ValueError(f"extracted function {node.name!r} is not defined in the submitted code")
        return signature, output_format
    definitions = [
        item for item in ast.walk(tree)
        if isinstance(item, ast.FunctionDef) and item.name == node.name
    ]
    if not definitions:
        raise ValueError(f"extracted function {node.name!r} is not defined in the submitted code")
    if not any(
        _parameter_names(item.args) == _parameter_names(node.args)
        and (item.returns is None or ast.unparse(item.returns) == ast.unparse(node.returns))
        for item in definitions
    ):
        raise ValueError(f"extracted signature for {node.name!r} does not match the submitted code")
    return signature, output_format


def function_name_from_signature(signature: str) -> str:
    node = _first_function(signature)
    if node is None:
        raise ValueError("invalid function signature")
    return node.name


def validate_candidate_signature(candidate: str, signature: str) -> None:
    """Require the generated reference to define the expected function and parameters."""
    expected = _first_function(signature)
    if expected is None:
        raise ValueError("invalid function signature")
    try:
        tree = ast.parse(candidate)
    except SyntaxError as exc:
        raise ValueError("candidate implementation has invalid syntax") from exc
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == expected.name:
            if _parameter_names(node.args) != _parameter_names(expected.args):
                raise ValueError(
                    f"candidate function {expected.name!r} does not use the required parameters: "
                    f"expected {_parameter_names(expected.args)}, got {_parameter_names(node.args)}"
                )
            return
    raise ValueError(f"candidate does not define the required function {expected.name!r}")


def construct_candidate_prompt(requirement: str, signature: str, output_format: str) -> str:
    """Ask for a reference implementation without exposing the submitted code."""
    return (
        "You are implementing a Python function from its requirement.\n\n"
        f"Requirement:\n{requirement}\n\n"
        f"Required function signature:\n{signature}\n\n"
        f"Required output type and format:\n{output_format}\n\n"
        "Write a complete reference implementation that uses exactly this function name and "
        "these parameters; do not rename, add, remove, or reorder them. The returned value must "
        "have exactly the required output type and format. Include any imports needed, such as "
        "for type annotations.\n"
        "Return only one Python code block containing the implementation."
    )


def construct_generated_tests_v2_prompt(
    requirement: str, signature: str, output_format: str, previous_tests: List[str], attempt: int
) -> str:
    """Ask for one new test without exposing either implementation."""
    for test in previous_tests:
        if '"error":' in test:
            if '"error": "AssertionError()"' in test:
                continue
            if '"error": "duplicate test"' in test:
                continue
            raise ValueError("previous test contains an error other than AssertionError()")
    history = "\n\n---\n\n".join(previous_tests) or "(none)"
    return (
        "You are creating black-box tests for a Python function.\n\n"
        f"Requirement:\n{requirement}\n\n"
        f"Function signature:\n{signature}\n\n"
        f"Output type and format:\n{output_format}\n\n"
        f"Generation attempt: {attempt}\n"
        f"Previously generated tests and responses:\n{history}\n\n"
        "Generate exactly one new, distinct test function named test_case.\n"
        "Do not repeat any previous test, including rejected tests; "
        "choose a different test case from the list above.\n"
        "It must contain an assert statement and call the function exactly as defined by "
        "the signature, with matching parameters. The expected value in the assertion must "
        "have exactly the output type and format described above.\n"
        "Use only imports and the test function at the top level. "
        "Cover behavior supported by the requirement.\n"
        "The observed outputs come from an AI-generated implementation and may be incorrect; "
        "derive expected behavior from the requirement, not just those outputs.\n"
        "Do not inspect any implementation.\n"
        "Return only one Python code block containing the test."
    )


def construct_reference_input_prompt(
    requirement: str, signature: str, previous_tests: List[str], previous_inputs: List[str], attempt: int
) -> str:
    """Request inputs, not an expected output, for a reference-output test."""
    return (
        "Choose a new valid input for this Python function based on its requirement.\n\n"
        f"Requirement:\n{requirement}\n\n"
        f"Function signature:\n{signature}\n\n"
        f"Existing tests:\n{chr(10).join(previous_tests)}\n\n"
        f"Previous input attempts:\n{chr(10).join(previous_inputs) or '(none)'}\n\n"
        f"Input attempt: {attempt}\n"
        'Return only a Python literal dictionary with "args" as a list of positional arguments '
        'and "kwargs" as a dictionary of keyword arguments, matching the signature. Use only literal values. '
        "Choose a distinct input and do not provide an expected output."
    )


def parse_reference_input(response: str) -> Tuple[List[Any], Dict[str, Any]]:
    """Parse literal positional and keyword arguments proposed by the model."""
    match = re.fullmatch(r"\s*```(?:json|python|py)?\s*(.*?)\s*```\s*", response, flags=re.IGNORECASE | re.DOTALL)
    text = match.group(1) if match else response
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        try:
            data = ast.literal_eval(text)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(f"input is not a valid JSON or Python literal: {exc}") from exc
    if not isinstance(data, dict) or set(data) != {"args", "kwargs"}:
        raise ValueError('input must be a dictionary with exactly "args" and "kwargs"')
    if not isinstance(data["args"], list) or not isinstance(data["kwargs"], dict):
        raise ValueError('"args" must be a list and "kwargs" must be a dictionary')
    if not all(isinstance(key, str) for key in data["kwargs"]) or not _literal_reference_value(data):
        raise ValueError("input must contain only supported finite literal values and string keyword names")
    return data["args"], data["kwargs"]


def parse_response(response_text: str) -> str:
    """Extract the first YES/NO token from a model response."""
    if not response_text:
        return ""
    m = re.search(r"\b(YES|NO)\b", response_text, flags=re.IGNORECASE)
    return m.group(1).upper() if m else ""


def extract_generated_tests(response_text: str) -> str:
    """Extract test code from a model response."""
    if not response_text:
        return ""
    blocks = re.findall(r"```(?:python|py)?\s*(.*?)```", response_text, flags=re.IGNORECASE | re.DOTALL)
    if blocks:
        return blocks[-1].strip()
    if re.search(r"^\s*def\s+test_[A-Za-z0-9_]\w*\s*\(", response_text, flags=re.MULTILINE):
        return response_text.strip()
    return ""


def extract_code_block(response_text: str) -> str:
    """Extract the first Python code block from a model response."""
    if not response_text:
        return ""
    blocks = re.findall(r"```(?:python|py)?\s*(.*?)```", response_text, flags=re.IGNORECASE | re.DOTALL)
    return blocks[0].strip() if blocks else response_text.strip()


def _worker_run_generated_test(
    code: str, tests: str, test_name: str, queue: mp.Queue, function_name: Optional[str] = None
) -> None:
    observations: List[Dict[str, str]] = []
    observation_count = 0
    formatter = reprlib.Repr()
    formatter.maxstring = 256
    formatter.maxother = 256
    result: Dict[str, Any]

    def observe_call(function: Any, /, *args: Any, **kwargs: Any) -> Any:
        nonlocal observation_count
        observation_count += 1
        if observation_count > 10:
            return function(*args, **kwargs)
        observation = {"function": function_name or "", "args": formatter.repr(args), "kwargs": formatter.repr(kwargs)}
        try:
            value = function(*args, **kwargs)
        except Exception as exc:
            observation["exception"] = formatter.repr(exc)
            observations.append(observation)
            raise
        observation["return_value"] = formatter.repr(value)
        observations.append(observation)
        return value

    class ObserveCalls(ast.NodeTransformer):
        def visit_Call(self, node: ast.Call) -> ast.AST:
            self.generic_visit(node)
            if isinstance(node.func, ast.Name) and node.func.id == function_name:
                node.args.insert(0, node.func)
                node.func = ast.copy_location(ast.Name(id="__observe_candidate_call__", ctx=ast.Load()), node.func)
            return node

    try:
        env: Dict[str, Any] = {}
        exec(code, env, env)
        if function_name:
            env["__observe_candidate_call__"] = observe_call
            tree = ObserveCalls().visit(ast.parse(tests))
            exec(compile(ast.fix_missing_locations(tree), "<generated tests>", "exec"), env, env)
        else:
            exec(tests, env, env)
        test = env.get(test_name)
        if not callable(test):
            raise NameError(f"Generated test {test_name} not found")
        test()
        result = {"status": "passed"}
    except AssertionError as exc:
        result = {"status": "assertion_failed", "error": repr(exc)}
    except ModuleNotFoundError as exc:
        result = {"status": "missing_module", "error": str(exc), "module": exc.name}
    except Exception as exc:
        result = {"status": "failed", "error": repr(exc)}
    if function_name:
        result["observations"] = observations
        result["observations_truncated"] = observation_count > len(observations)
    queue.put(result)


def run_generated_tests(code: str, tests: str, timeout_s: float = 3.0) -> Dict[str, Any]:
    """Run each generated test in a fresh process and return an execution report."""
    if not tests:
        return {"total": 0, "passed": 0, "failed": 0, "errors": ["no test code extracted"]}

    try:
        namespace: Dict[str, Any] = {}
        exec(tests, namespace, namespace)
        test_names = sorted(
            name for name, value in namespace.items()
            if name.startswith("test_") and callable(value)
        )
    except Exception as exc:
        return {"total": 0, "passed": 0, "failed": 0, "errors": [f"test definition error: {exc!r}"]}

    report: Dict[str, Any] = {"total": len(test_names), "passed": 0, "failed": 0, "errors": []}
    for test_name in test_names:
        queue: mp.Queue = mp.Queue()
        process = mp.Process(target=_worker_run_generated_test, args=(code, tests, test_name, queue))
        process.start()
        process.join(timeout=timeout_s)
        if process.is_alive():
            process.terminate()
            process.join()
            report["failed"] += 1
            report["errors"].append(f"{test_name}: timeout>{timeout_s}s")
            continue
        if queue.empty():
            report["failed"] += 1
            report["errors"].append(f"{test_name}: no result")
            continue
        result = queue.get()
        if result.get("status") == "passed":
            report["passed"] += 1
        else:
            report["failed"] += 1
            report["errors"].append(f"{test_name}: {result.get('error', '')}")
    return report


def _generated_test_names(tests: str) -> List[str]:
    """Find test functions without executing model-generated definitions."""
    matches = re.findall(r"^\s*def\s+(test_[A-Za-z0-9_]\w*)\s*\(", tests, flags=re.MULTILINE)
    return sorted(set(matches))


def _run_generated_test_case(
    code: str, tests: str, test_name: str, timeout_s: float, function_name: Optional[str] = None
) -> Dict[str, Any]:
    queue: mp.Queue = mp.Queue()
    process = mp.Process(target=_worker_run_generated_test, args=(code, tests, test_name, queue, function_name))
    process.start()
    try:
        result = queue.get(timeout=timeout_s)
    except Empty:
        if process.is_alive():
            process.terminate()
            process.join()
            return {"status": "incompatible", "error": f"timeout>{timeout_s}s"}
        process.join()
        return {"status": "incompatible", "error": "no result"}
    process.join(timeout=timeout_s)
    if process.is_alive():
        process.terminate()
        process.join()
        return {"status": "incompatible", "error": f"timeout>{timeout_s}s"}
    _raise_missing_module(result)
    if result.get("status") == "failed":
        result["status"] = "incompatible"
    return result


def check_generated_candidate(candidate_code: str, timeout_s: float = 3.0) -> None:
    """Check that the generated reference can be loaded before requesting tests."""
    result = _run_generated_test_case(candidate_code, "def test_load():\n    pass", "test_load", timeout_s)
    if result["status"] != "passed":
        raise RuntimeError(f"Unable to load generated reference: {result.get('error', result['status'])}")


def _observe_invalid_test(
    candidate_code: str, tests: str, test: ast.FunctionDef, function_name: str, timeout_s: float
) -> Dict[str, Any]:
    """Collect diagnostic outputs without accepting or changing a rejected test."""
    try:
        candidate_tree = ast.parse(candidate_code)
    except SyntaxError:
        return {"observation_note": "Reference implementation has invalid syntax; it could not be run."}
    defined_names = {
        node.name for node in candidate_tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    calls = [
        node for node in ast.walk(test)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]
    called_names = [node.func.id for node in calls if isinstance(node.func, ast.Name)]
    observed_name = next(
        (name for name in [function_name, *called_names] if name in defined_names and name in called_names),
        None,
    )
    if observed_name is not None:
        result = _run_generated_test_case(candidate_code, tests, test.name, timeout_s, observed_name)
        note = f"Ran the rejected test for feedback only, observing reference function {observed_name!r}."
    else:
        missing_names = {
            name for name in called_names if name not in defined_names and not hasattr(builtins, name)
        }
        if function_name not in defined_names or len(missing_names) != 1:
            return {"observation_note": "No matching reference function call could be observed."}
        missing_name = next(iter(missing_names))
        statements = []
        for call in calls:
            if not isinstance(call.func, ast.Name) or call.func.id != missing_name:
                continue
            try:
                args = [ast.literal_eval(arg) for arg in call.args]
                kwargs = {keyword.arg: ast.literal_eval(keyword.value) for keyword in call.keywords}
            except (ValueError, SyntaxError):
                continue
            if None in kwargs or not _literal_reference_value([args, kwargs]):
                continue
            statements.append(
                f"    try:\n        {function_name}(*{args!r}, **{kwargs!r})\n"
                "    except ModuleNotFoundError:\n        raise\n    except Exception:\n        pass"
            )
            if len(statements) == 10:
                break
        if not statements:
            return {"observation_note": f"Function {missing_name!r} is not defined and its inputs could not be extracted as literals."}
        probe = "def test_feedback():\n" + "\n".join(statements)
        result = _run_generated_test_case(candidate_code, probe, "test_feedback", timeout_s, function_name)
        note = (
            f"Function {missing_name!r} is not defined in the reference. For feedback only, "
            f"called {function_name!r} with the test's literal arguments; these are not returns from {missing_name!r}."
        )
    details: Dict[str, Any] = {"observation_note": note}
    if "observations" in result:
        details["observations"] = result["observations"]
        details["observations_truncated"] = result["observations_truncated"]
    if result["status"] != "passed":
        details["observation_error"] = result.get("error", result["status"])
    return details


def validate_generated_test(
    candidate_code: str, tests: str, function_name: str, timeout_s: float = 3.0
) -> Dict[str, Any]:
    """Check that one test is meaningful and passes the generated implementation."""
    try:
        tree = ast.parse(tests)
    except SyntaxError as exc:
        return {
            "status": "invalid", "error": f"test syntax error: {exc}",
            "observation_note": "The test has invalid syntax and could not be run.",
        }

    definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if (
        len(definitions) != 1
        or not definitions[0].name.startswith("test_")
        or any(not isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef)) for node in tree.body)
    ):
        test_definitions = [node for node in definitions if node.name.startswith("test_")]
        details: Dict[str, Any] = {"observation_note": "Could not identify a single test function to run."}
        if len(test_definitions) == 1:
            isolated = ast.Module(
                body=[node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
                + test_definitions,
                type_ignores=[],
            )
            details = _observe_invalid_test(
                candidate_code, ast.unparse(isolated), test_definitions[0], function_name, timeout_s
            )
            details["observation_note"] = (
                "Only the test function and imports were used for feedback; "
                "other top-level code was excluded. " + details["observation_note"]
            )
        return {"status": "invalid", "error": "expected one test function and optional imports", **details}
    test = definitions[0]
    nodes = list(ast.walk(test))
    has_assert = any(isinstance(node, ast.Assert) for node in nodes)
    has_target_call = any(
        isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == function_name
        for node in nodes
    )
    if not has_assert or not has_target_call:
        error = "test must contain an assertion" if not has_assert else f"test must call the target function {function_name!r}"
        return {
            "status": "invalid", "error": error,
            **_observe_invalid_test(candidate_code, tests, test, function_name, timeout_s),
        }

    test_name = test.name
    result = _run_generated_test_case(candidate_code, tests, test_name, timeout_s, function_name)
    if result["status"] != "passed":
        return result
    test.name = "test_case"
    return {
        "status": "passed",
        "test_name": test_name,
        "fingerprint": ast.dump(tree, include_attributes=False),
        "observations": result["observations"],
        "observations_truncated": result["observations_truncated"],
    }


def _literal_reference_value(value: Any) -> bool:
    if type(value) in (type(None), bool, int, str, bytes):
        return True
    if type(value) is float:
        return math.isfinite(value)
    if type(value) in (list, tuple, set):
        return all(_literal_reference_value(item) for item in value)
    if type(value) is dict:
        return all(_literal_reference_value(key) and _literal_reference_value(item) for key, item in value.items())
    return False


def _worker_reference_output(
    code: str, function_name: str, args: List[Any], kwargs: Dict[str, Any], queue: mp.Queue
) -> None:
    try:
        env: Dict[str, Any] = {}
        exec(code, env, env)
        value = env[function_name](*args, **kwargs)
        if not _literal_reference_value(value):
            raise ValueError("reference output cannot be represented as a supported Python literal")
        output = repr(value)
        if len(output) > 16000:
            raise ValueError("reference output is too large to save as a test")
        ast.literal_eval(output)
        queue.put({"status": "passed", "output": output})
    except ModuleNotFoundError as exc:
        queue.put({"status": "missing_module", "error": str(exc), "module": exc.name})
    except Exception as exc:
        queue.put({"status": "failed", "error": repr(exc)})


def generate_reference_output_test(
    candidate_code: str, function_name: str, args: List[Any], kwargs: Dict[str, Any],
    test_number: int, timeout_s: float = 3.0
) -> Dict[str, Any]:
    """Build a test from an actual reference return value in a timed process."""
    queue: mp.Queue = mp.Queue()
    process = mp.Process(
        target=_worker_reference_output, args=(candidate_code, function_name, args, kwargs, queue)
    )
    process.start()
    try:
        result = queue.get(timeout=timeout_s)
    except Empty:
        if process.is_alive():
            process.terminate()
            process.join()
            return {"status": "failed", "error": f"timeout>{timeout_s}s"}
        process.join()
        return {"status": "failed", "error": "no result"}
    process.join(timeout=timeout_s)
    if process.is_alive():
        process.terminate()
        process.join()
        return {"status": "failed", "error": f"timeout>{timeout_s}s"}
    _raise_missing_module(result)
    if result["status"] == "passed":
        test_name = f"test_reference_output_{test_number}"
        result["test_name"] = test_name
        result["test"] = (
            f"def {test_name}():\n"
            f"    assert {function_name}(*{args!r}, **{kwargs!r}) == {result['output']}"
        )
    return result


def run_validated_tests(
    submitted_code: str, tests: List[Tuple[str, str]], timeout_s: float = 3.0
) -> Dict[str, Any]:
    """Run separately validated test snippets against the submitted implementation."""
    report: Dict[str, Any] = {
        "eligible": len(tests), "passed": 0, "failed": 0, "failures": []
    }
    for test_code, test_name in tests:
        result = _run_generated_test_case(submitted_code, test_code, test_name, timeout_s)
        if result["status"] == "passed":
            report["passed"] += 1
        else:
            report["failed"] += 1
            report["failures"].append({"test": test_name, "error": result.get("error", "")})
    return report


def run_generated_tests_v2(
    candidate_code: str, submitted_code: str, tests: str, timeout_s: float = 3.0
) -> Dict[str, Any]:
    """Validate tests against a candidate, discard unusable tests, then test submitted code."""
    test_names = _generated_test_names(tests)
    report: Dict[str, Any] = {
        "generated": len(test_names),
        "eligible": 0,
        "passed": 0,
        "failed": 0,
        "excluded": [],
    }
    if not candidate_code or not test_names:
        report["excluded"].append("missing candidate implementation or test functions")
        return report

    eligible_tests: List[str] = []
    for test_name in test_names:
        result = _run_generated_test_case(candidate_code, tests, test_name, timeout_s)
        if result.get("status") == "passed":
            eligible_tests.append(test_name)
        else:
            report["excluded"].append({"test": test_name, "error": result.get("error", "")})

    report["eligible"] = len(eligible_tests)
    for test_name in eligible_tests:
        result = _run_generated_test_case(submitted_code, tests, test_name, timeout_s)
        if result.get("status") == "passed":
            report["passed"] += 1
        else:
            report["failed"] += 1
            report.setdefault("failures", []).append({"test": test_name, "error": result.get("error", "")})
    return report


def save_json(obj: Any, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

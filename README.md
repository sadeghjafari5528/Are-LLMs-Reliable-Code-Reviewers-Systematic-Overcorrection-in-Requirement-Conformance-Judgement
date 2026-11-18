# Uncovering-Prompt-Induced-Bias-and-Explanation-Failures-in-LLM-Based-Code-Conformance-Judgement.

This repository contains code to reproduce experiments from our paper on **LLM requirement–code conformance judgement** and the proposed **Fix-guided Verification Filter** to mitigate *over-correction* under rationale+fix prompting.

> **Important:** Configure keys via environment variables or a local `.env` file.

---

## Repository structure

- `main.py` – baseline runner (Direct / Direct+Explain / Full) with **resume JSONL** logging.
- `full_mitigation_framework.py` – single-model **Full+Filter** pipeline.
- `run_full_mitigation_batch.py` – runs **Full+Filter** for a list of models sequentially.
- `model.py` – provider wrappers (OpenAI / Gemini / Anthropic / OpenRouter).
- `utils.py` – dataset loading, prompt construction, verdict parsing.
- `prepare_buggy_mbpp_dataset.py` – dataset preparation utility for paired MBPP.
- `prepare_quixbugs_buggy_solutions.py` – dataset preparation utility for paired QuixBugs.
- `factory.py` – small helper to build model wrappers by name.

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

(Optional) load environment variables from a `.env` file:

```bash
cp .env.example .env
# edit .env
```

---

## API keys / environment variables

Set the following variables depending on the models you run:

- `OPENAI_API_KEY` (OpenAI + also used for GPT-4o augmentation in the filter)
- `GOOGLE_API_KEY` (Gemini)
- `ANTHROPIC_API_KEY` (Claude)
- `OPENROUTER_API_KEY` (OpenRouter for Llama/Mistral)

You may use a `.env` file locally; **do not commit it**.

---

## Expected dataset format

The runners expect a **paired** dataset where each task has:

- `task_id`
- a natural-language requirement in `text` (or `prompt` / `requirement`)
- optional `declaration` (function signature)
- `canonical_solution` (correct implementation)
- `buggy_solution` (bug-injected implementation)
- optional `test` and `example_test` fields (used by `full_mitigation_framework.py`)

All scripts accept `--dataset` as `.json` or `.jsonl`.

---

## Baseline experiments (no filter)

Run multiple models and prompt modes, with automatic resume:

```bash
python main.py \
  --dataset path/to/humanevalbugs.jsonl \
  --benchmark humaneval \
  --out_dir outputs/humaneval \
  --models gpt-4o gemini-2.0-flash claude-haiku-4-5 meta-llama/llama-3.1-8b-instruct mistralai/mistral-small-3.2-24b-instruct \
  --prompt_modes direct direct_explain full \
  --continue_on_error
```

Outputs:
- `outputs/<benchmark>/<benchmark>__<model>__baseline.jsonl` (append-only checkpoint)
- `outputs/<benchmark>/<benchmark>__<model>__baseline.json` (final array)

### Generated tests v2

Use `--prompt_modes generated_tests_v2 --num_unit_tests 5` to generate a
reference implementation and then request tests one at a time. First, one extra
LLM call extracts the entry-point signature, including its return type, and a
one-line description of the output type and format (structure, element types,
ordering, rounding, edge cases) from the submitted implementation (the
requirement is also given so it can choose the right function). If the
implementation has no return annotation, the model infers one. The signature is
checked against the submitted code (the function must exist with the same
parameter names, and the return annotation must agree if the code has one) and,
together with the output description, is then given to the
reference-implementation and test prompts; the signature is also given to the
fallback-input prompt. All of them therefore use the same function name,
parameters, and output format. The generated reference must define that function
with those parameters, otherwise the run raises an error before any tests are
requested. The signature and description are saved as `function_signature` and
`output_format` (with `function_name`) in the result; annotations and defaults
are normalized and may be reformatted. The output description comes from the
model, so it can be wrong where the submitted code is buggy or unannotated.
Note that the submitted code is sent to the model for this extraction step.
Only distinct
tests that pass the generated reference are kept. If fewer than the requested
number pass but at least one is accepted, the runner requests literal inputs and
uses the reference's actual return values to construct additional tests. It
then runs the available tests on the submitted implementation: all must pass
for a YES verdict. Passing the generated reference does not prove a test or
reference is correct; reference-output tests in particular can reproduce
mistakes in the reference implementation.
Each new test prompt includes all previous attempts, including rejected tests,
and asks for a different test case. It also includes validation failures and
observed return values or exceptions from the generated reference. These
observations may be wrong and should not replace the requirement as the oracle.
Feedback records up to ten calls per test; long values may be abbreviated.
Observations are unavailable if validation cannot run or times out.
For rejected tests, feedback runs the test for diagnosis where possible without
accepting it. If the test names a function absent from the reference, feedback
may instead call the required function with literal arguments from the test;
this is explicitly labeled and is not presented as the missing function's return.

The default limit is ten test-generation attempts per requested test; override
it with `--max_test_attempts`. If needed, up to the same number of additional
attempts are used to generate reference-output tests. These inputs and reference
outputs must be representable as supported finite Python literals.
If the shortfall cannot be filled, evaluation proceeds with the tests available;
if no LLM-generated test passes, it is an error, not a NO verdict. In each
saved `raw` result, `test_sources` corresponds by index to `generated_tests`:
`llm` identifies LLM-written tests and `reference_output` identifies tests
created from reference returns. The execution report records the requested
count and rejected fallback attempts. With `--continue_on_error`, errors are
saved with an empty prediction. Checkpoints are keyed by task, variant and mode,
not test count or retry limit; use a fresh output directory to rerun with
different settings or retry a saved error.

**Response format.** Every prompt in these modes specifies an exact tagged
output format: `<signature>` and `<output_format>` for the signature step,
`<code>` for the reference implementation, `<test>` for a generated test, and
`<input>` / `<inputs>` for inputs (inputs are written as `{"args": [...],
"kwargs": {...}}`). The parsers look for the tagged content anywhere in the
response, so explanations before or after it are ignored; tags may wrap a
fenced code block. If the tags are missing they fall back to a fenced code block
(and, for the signature step, the older `OUTPUT:` line). Inputs may be JSON or
Python literals, including comments, `true`/`false`/`null`, and `'a' * 3`. A
parse error includes the start of the response it received. A response that was
cut off by the model's token limit cannot be recovered and raises such an error.

**Security:** generated implementations and tests execute arbitrary Python in
subprocesses with the current user's permissions. Timeouts are not a security
sandbox; run this mode only in an appropriately isolated environment.

If a generated reference, test, or submitted implementation requires a missing
Python module, the run stops immediately, even with `--continue_on_error`.
The error names the missing module so you can install the corresponding package
in the environment running the script and rerun. Import names and package names
can differ. Top-level imports are checked before requesting tests; imports
inside functions are detected when those functions are called.

### Generated tests v3

`--prompt_modes generated_tests_v3` never asks the model to write expected
values. It uses the same signature and output-format extraction and the same
generated reference implementation as v2, then asks the model for
`--num_unit_tests` inputs (as `args`/`kwargs`) in a single request. The
reference is run on each input, and each test asserts the reference's actual
return value, e.g. `assert double(3) == 6`. All tests are then run on the
submitted implementation: all must pass for a YES verdict. A normal run makes
three model calls: signature, reference implementation, and inputs.

Inputs are not filtered or retried: the model must return exactly
`--num_unit_tests` valid inputs in one response, and the reference must return
a plain Python literal (finite numbers, strings, bytes, lists, tuples, sets,
dicts, booleans, `None`) for every one of them. Otherwise the run raises an
error naming the problem (for example, the reference raising on an input),
which with `--continue_on_error` is saved with an empty prediction. Saved
results include `inputs`, `generated_tests`, `reference_outputs`, and
`test_sources` (always `reference_output`) next to the execution report.

Because expected values come from the reference, a mistake in the reference
implementation becomes a wrong expectation, and the tests only check that the
submitted code agrees with the reference. Exact equality is used, so correct
implementations that differ in floating-point rounding can fail. The security
and missing-module notes above apply. Checkpoints are keyed by task, variant and
mode, so a fresh output directory is needed to rerun with different settings.

---

## Full+Filter for a single model

```bash
python full_mitigation_framework.py \
  --dataset path/to/mbpp_buggy_solutions.jsonl \
  --benchmark mbpp \
  --model gpt-4o \
  --out outputs/mbpp/mbpp__gpt-4o__full_filtered.json \
  --enable_filter \
  --enable_aug \
  --aug_cache outputs/mbpp/aug_cache_mbpp.jsonl \
  --n_aug 12 \
  --max_rounds 2
```

The script logs a checkpoint next to `--out` (same name with `.jsonl`).
If the process is interrupted, re-running the exact command will resume.

---

## Full+Filter for all five models (batch)

```bash
python run_full_mitigation_batch.py \
  --dataset path/to/mbpp_buggy_solutions.jsonl \
  --benchmark mbpp \
  --out_dir outputs/mbpp \
  --enable_filter \
  --enable_aug \
  --aug_cache outputs/mbpp/aug_cache_mbpp.jsonl \
  --n_aug 12 \
  --max_rounds 2 \
  --continue_on_error
```

Each model writes its own resume-safe JSONL checkpoint in `--out_dir`.


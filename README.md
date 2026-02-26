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


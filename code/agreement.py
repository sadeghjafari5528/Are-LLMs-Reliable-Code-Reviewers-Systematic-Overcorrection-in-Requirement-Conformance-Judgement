#!/usr/bin/env python3

"""Calculate majority-vote results from saved prompt-method evaluations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


METHODS = ("direct", "generated_tests_v3")
VERDICTS = {"YES", "NO"}
DEFAULT_BASELINE_RESULTS = (
    Path(__file__).resolve().parents[1]
    / "outputs/humaneval/humaneval__gpt-4o__baseline_v1.jsonl"
)
DEFAULT_GENERATED_TESTS_V3_RESULTS = (
    Path(__file__).resolve().parents[1]
    / "outputs/humaneval/humaneval__gpt-4o__baseline.jsonl"
)


def read_results(path: Path) -> list[dict[str, Any]]:
    """Read result objects from a JSON array/object or JSONL file."""
    try:
        if path.suffix.lower() == ".json":
            content = json.loads(path.read_text(encoding="utf-8"))
            rows = content if isinstance(content, list) else [content]
        else:
            rows = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read valid results from {path}: {error}") from error

    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"Expected every result in {path} to be a JSON object")
    return rows


def row_key(row: dict[str, Any], path: Path, row_number: int) -> tuple[str, str]:
    task_id = row.get("task_id")
    variant = row.get("variant")
    if task_id is None or variant is None:
        raise ValueError(f"{path}: row {row_number} is missing task_id or variant")
    return str(task_id), str(variant).strip().lower()


def predictions_by_method(
    baseline_path: Path, generated_tests_v3_path: Path
) -> tuple[
    dict[str, dict[tuple[str, str], str]],
    dict[tuple[str, str], str],
]:
    results = {method: {} for method in METHODS}
    expected_verdicts: dict[tuple[str, str], str] = {}

    for path in (baseline_path, generated_tests_v3_path):
        for row_number, row in enumerate(read_results(path), start=1):
            mode = row.get("mode")
            if mode not in results:
                continue
            if mode == "generated_tests_v3" and path != generated_tests_v3_path:
                raise ValueError(
                    f"{path}: generated_tests_v3 results should be supplied via "
                    "--generated-tests-v3-results"
                )
            if mode != "generated_tests_v3" and path != baseline_path:
                raise ValueError(
                    f"{path}: {mode!r} results should be supplied via "
                    "--baseline-results"
                )

            key = row_key(row, path, row_number)
            if key in results[mode]:
                raise ValueError(f"{path}: duplicate {mode} result for {key}")
            label = row.get("label")
            if label not in (0, 1, "0", "1"):
                raise ValueError(
                    f"{path}: row {row_number} has invalid label {label!r}; "
                    "expected 0 or 1"
                )
            expected_verdict = "YES" if int(label) == 1 else "NO"
            previous_expected = expected_verdicts.setdefault(key, expected_verdict)
            if previous_expected != expected_verdict:
                raise ValueError(f"{path}: inconsistent labels for {key}")

            prediction = str(row.get("pred", "")).strip().upper()
            if prediction not in VERDICTS:
                raise ValueError(
                    f"{path}: row {row_number} has invalid {mode} prediction "
                    f"{row.get('pred')!r}; expected YES or NO"
                )
            results[mode][key] = prediction

    missing_modes = [mode for mode, rows in results.items() if not rows]
    if missing_modes:
        raise ValueError(f"No results found for methods: {', '.join(missing_modes)}")
    return results, expected_verdicts


def majority_verdict(predictions: list[str]) -> str:
    """Return YES only when more than half of the methods predict YES."""
    return "YES" if predictions.count("YES") > len(predictions) / 2 else "NO"


def rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def print_report(
    results: dict[str, dict[tuple[str, str], str]],
    expected_verdicts: dict[tuple[str, str], str],
) -> None:
    common_keys = set.intersection(*(set(results[method]) for method in METHODS))
    if not common_keys:
        raise ValueError("The selected methods have no task/variant rows in common")

    keys = sorted(common_keys)
    print(f"Methods: {', '.join(METHODS)}")
    print(f"Rows available: {', '.join(f'{m}={len(results[m])}' for m in METHODS)}")
    print(f"Complete matched rows: {len(keys)}")
    print(
        "Unmatched rows excluded by method: "
        + ", ".join(f"{method}={len(results[method]) - len(keys)}" for method in METHODS)
    )

    variants = sorted({key[1] for key in keys})
    for title, selected_keys in [
        ("Overall", keys),
        *[
            (f"Variant: {variant}", [key for key in keys if key[1] == variant])
            for variant in variants
        ],
    ]:
        yes_votes = 0
        ties = 0
        true_positives = 0
        true_negatives = 0
        false_positives = 0
        false_negatives = 0
        for key in selected_keys:
            row_predictions = [results[method][key] for method in METHODS]
            yes_count = row_predictions.count("YES")
            no_count = len(row_predictions) - yes_count
            ties += yes_count == no_count
            verdict = majority_verdict(row_predictions)
            yes_votes += verdict == "YES"
            expected = expected_verdicts[key]
            true_positives += verdict == "YES" and expected == "YES"
            true_negatives += verdict == "NO" and expected == "NO"
            false_positives += verdict == "YES" and expected == "NO"
            false_negatives += verdict == "NO" and expected == "YES"

        count = len(selected_keys)
        correct = true_positives + true_negatives
        false_positive_rate = rate(false_positives, false_positives + true_negatives)
        false_negative_rate = rate(false_negatives, false_negatives + true_positives)
        print(
            f"\n{title}: accuracy={correct}/{count} ({rate(correct, count):.4%}), "
            f"FP={false_positives}, FN={false_negatives}, "
            f"FPR={false_positive_rate:.4%}, FNR={false_negative_rate:.4%}, "
            f"majority YES={yes_votes}, NO={count - yes_votes}, "
            f"ties resolved as NO={ties}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calculate agreement from saved method predictions; no LLM calls."
    )
    parser.add_argument(
        "--baseline-results",
        type=Path,
        default=DEFAULT_BASELINE_RESULTS,
        help="JSON or JSONL containing direct results",
    )
    parser.add_argument(
        "--generated-tests-v3-results",
        type=Path,
        default=DEFAULT_GENERATED_TESTS_V3_RESULTS,
        help="JSON or JSONL containing generated_tests_v3 results",
    )
    args = parser.parse_args()
    results, expected_verdicts = predictions_by_method(
        args.baseline_results, args.generated_tests_v3_results
    )
    print_report(results, expected_verdicts)


if __name__ == "__main__":
    main()

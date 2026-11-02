#!/usr/bin/env python3

"""Calculate requirement-conformance accuracy from main.py JSONL output."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def is_correct(row: dict[str, Any]) -> bool:
	"""Return whether the prediction matches the sample's expected verdict."""
	variant = str(row.get("variant", "")).strip().lower()
	prediction = str(row.get("pred", "")).strip().upper()
	expected = "YES" if variant == "canonical" else "NO" if variant == "buggy" else ""
	return bool(expected) and prediction == expected


def is_false_positive(row: dict[str, Any]) -> bool:
	return (
		str(row.get("variant", "")).strip().lower() == "buggy"
		and str(row.get("pred", "")).strip().upper() == "YES"
	)


def is_false_negative(row: dict[str, Any]) -> bool:
	return (
		str(row.get("variant", "")).strip().lower() == "canonical"
		and str(row.get("pred", "")).strip().upper() == "NO"
	)


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
	with path.open(encoding="utf-8") as file:
		for line_number, line in enumerate(file, start=1):
			if not line.strip():
				continue
			try:
				row = json.loads(line)
			except json.JSONDecodeError as error:
				raise ValueError(f"Invalid JSON on line {line_number}") from error
			if not isinstance(row, dict):
				raise ValueError(f"Expected an object on line {line_number}")
			yield row


def accuracy(rows: list[dict[str, Any]]) -> float:
	return sum(is_correct(row) for row in rows) / len(rows) if rows else 0.0


def rate(numerator: int, denominator: int) -> float:
	return numerator / denominator if denominator else 0.0


def print_metrics(name: str, rows: list[dict[str, Any]]) -> None:
	correct = sum(is_correct(row) for row in rows)
	false_positives = sum(is_false_positive(row) for row in rows)
	false_negatives = sum(is_false_negative(row) for row in rows)
	true_positives = sum(
		str(row.get("variant", "")).strip().lower() == "canonical"
		and str(row.get("pred", "")).strip().upper() == "YES"
		for row in rows
	)
	true_negatives = sum(
		str(row.get("variant", "")).strip().lower() == "buggy"
		and str(row.get("pred", "")).strip().upper() == "NO"
		for row in rows
	)
	false_positive_rate = rate(false_positives, false_positives + true_negatives)
	false_negative_rate = rate(false_negatives, false_negatives + true_positives)
	print(
		f"{name}: accuracy={correct}/{len(rows)} ({accuracy(rows):.4%}), "
		f"FP={false_positives}, FN={false_negatives}, "
		f"FPR={false_positive_rate:.4%}, FNR={false_negative_rate:.4%}"
	)


def main() -> None:
	path = Path("outputs/humaneval/humaneval__gpt-4o__baseline.jsonl")

	rows = list(read_jsonl(path))
	print_metrics("Overall", rows)

	by_mode: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
	for row in rows:
		by_mode[str(row.get("mode", "unknown"))].append(row)

	for mode, mode_rows in by_mode.items():
		print_metrics(mode, mode_rows)


if __name__ == "__main__":
	main()

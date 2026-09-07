#!/usr/bin/env python3
"""Fail closed when comparing compatible, independently repeated benchmark runs."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import statistics

MIN_RUNS = 5


def key(record):
    return record["case"], record["backend"], record["threads"], record["phase"]


def validate(document):
    if document.get("schema_version") != 1:
        raise ValueError("unsupported result schema")
    if not document.get("records"):
        raise ValueError("no benchmark records")
    seen = set()
    for row in document["records"]:
        identity = key(row)
        if identity in seen:
            raise ValueError(f"duplicate record: {identity}")
        seen.add(identity)
        if row.get("correctness") != "passed":
            raise ValueError(f"correctness not established: {identity}")
        values = row.get("run_medians_ns", [])
        if not values or any(not math.isfinite(x) or x <= 0 for x in values):
            raise ValueError(f"invalid timing samples: {identity}")
    expected = {tuple(k) for k in document.get("expected_records", [])}
    if not expected or seen != expected:
        raise ValueError("missing or unexpected benchmark records")


def ratio_interval(reference, candidate, resamples=10000):
    """Percentile bootstrap over independent process medians, never inner samples."""
    rng = random.Random(4137)
    ratios = sorted(
        statistics.median(rng.choices(candidate, k=len(candidate)))
        / statistics.median(rng.choices(reference, k=len(reference)))
        for _ in range(resamples)
    )
    return ratios[int(0.025 * resamples)], ratios[int(0.975 * resamples)]


def comparison(reference, candidate, tolerance):
    ratio = statistics.median(candidate) / statistics.median(reference)
    if min(len(reference), len(candidate)) < MIN_RUNS:
        return dict(ratio=ratio, status="insufficient_runs", confidence_interval=None)
    low, high = ratio_interval(reference, candidate)
    limit = 1 + tolerance
    status = "pass" if high <= limit else "regression" if low > limit else "inconclusive"
    return dict(ratio=ratio, status=status, confidence_interval=[low, high])


def compare(reference, candidate, tolerance=0.05):
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative")
    validate(reference)
    validate(candidate)
    for field in ["suite_sha256", "environment"]:
        if reference.get(field) != candidate.get(field) or not reference.get(field):
            raise ValueError(f"incompatible {field}; remeasure both revisions on the same setup")
    old = {key(row): row for row in reference["records"]}
    new = {key(row): row for row in candidate["records"]}
    if old.keys() != new.keys():
        raise ValueError("benchmark coverage changed; missing cases cannot pass")
    rows = []
    for identity in sorted(old):
        result = comparison(old[identity]["run_medians_ns"], new[identity]["run_medians_ns"], tolerance)
        rows.append(dict(case=identity[0], backend=identity[1], threads=identity[2], phase=identity[3], **result))
    return dict(schema_version=1, tolerance=tolerance,
                status="pass" if all(r["status"] == "pass" for r in rows) else "fail", records=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--tolerance", type=float, default=0.05)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        report = compare(json.loads(args.baseline.read_text()), json.loads(args.candidate.read_text()), args.tolerance)
    except (ValueError, KeyError, TypeError) as error:
        parser.exit(2, f"Cannot compare: {error}\n")
    content = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(content)
    for row in report["records"]:
        print(f'{row["status"]:18} {row["case"]:36} {row["backend"]:12} {row["threads"]}t  {row["ratio"]:.3f}×')
    parser.exit(0 if report["status"] == "pass" else 1)


if __name__ == "__main__":
    main()

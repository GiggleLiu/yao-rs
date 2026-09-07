#!/usr/bin/env python3
"""Assess warmed native execution against every measured CPU competitor."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from compare import comparison, validate


def qualify(document, tolerance=0.05):
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative")
    validate(document)
    groups = {}
    for row in document['records']:
        if row['phase'] == 'execute':
            groups.setdefault((row['case'], row['threads']), {})[row['backend']] = row
    records = []
    for (case, threads), backends in sorted(groups.items()):
        if 'native' not in backends or len(backends) < 2:
            raise ValueError(f'Missing native or competitor: {case}, {threads}t')
        for name, rival in sorted(backends.items()):
            if name == 'native':
                continue
            result = comparison(rival['run_medians_ns'], backends['native']['run_medians_ns'], tolerance)
            if result['status'] == 'regression':
                result['status'] = 'slower'
            records.append(dict(case=case, threads=threads, competitor=name, **result))
    if not records:
        raise ValueError('No matched comparisons')
    return dict(schema_version=1, tolerance=tolerance,
                scope='Warmed native CPU execution against named measured competitors; other phases and devices require separate qualification.',
                status='pass' if all(r['status'] == 'pass' for r in records) else 'fail', records=records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path)
    parser.add_argument('--tolerance', type=float, default=0.05)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    try:
        report = qualify(json.loads(args.results.read_text()), args.tolerance)
    except (ValueError, KeyError, TypeError) as error:
        parser.exit(2, f'Cannot qualify: {error}\n')
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + '\n')
    for row in report['records']:
        print(f"{row['status']:18} {row['case']:50} {row['competitor']:14} {row['ratio']:.3f}× native/competitor")
    parser.exit(0 if report['status'] == 'pass' else 1)


if __name__ == '__main__':
    main()

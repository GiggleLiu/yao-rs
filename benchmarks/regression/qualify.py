#!/usr/bin/env python3
"""Assess warmed native execution against every measured CPU competitor."""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from compare import comparison, validate


def qualify(document, tolerance=0.05):
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative")
    validate(document)
    groups = {}
    for row in document['records']:
        group = groups.setdefault((row['case'], row['threads']), dict(native={}, rivals={}))
        if row['backend'] == 'native' and row['phase'] in ['execute', 'fused2_execute', 'fused4_execute']:
            group['native'][row['phase']] = row
        elif row['backend'] != 'native' and row['phase'] == 'execute':
            group['rivals'][row['backend']] = row
    records = []
    for (case, threads), group in sorted(groups.items()):
        if 'execute' not in group['native'] or not group['rivals']:
            raise ValueError(f'Missing native or competitor: {case}, {threads}t')
        native_mode, native = min(group['native'].items(), key=lambda pair: statistics.median(pair[1]['run_medians_ns']))
        for name, rival in sorted(group['rivals'].items()):
            result = comparison(rival['run_medians_ns'], native['run_medians_ns'], tolerance)
            if result['status'] == 'regression':
                result['status'] = 'slower'
            records.append(dict(case=case, threads=threads, native_mode=native_mode, competitor=name, **result))
    if not records:
        raise ValueError('No matched comparisons')
    return dict(schema_version=1, tolerance=tolerance,
                scope='Fastest measured native CPU execution mode against every named competitor; preparation is excluded. Other phases and devices require separate qualification.',
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
        print(f"{row['status']:18} {row['case']:50} {row['native_mode']:16} {row['competitor']:14} {row['ratio']:.3f}× native/competitor")
    parser.exit(0 if report['status'] == 'pass' else 1)


if __name__ == '__main__':
    main()

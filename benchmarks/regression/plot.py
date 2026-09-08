#!/usr/bin/env python3
"""Render the documentation scaling chart from a saved curated run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results")
    parser.add_argument("output")
    parser.add_argument("--stacked", action="store_true", help="Stack panels for narrow screens")
    args = parser.parse_args()
    data = json.load(open(args.results))
    rows = data["records"]
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 10,
                         "svg.fonttype": "none", "svg.hashsalt": "yao-curated-cpu"})
    shape, size = ((2, 1), (5.2, 7.2)) if args.stacked else ((1, 2), (10, 3.8))
    fig, axes = plt.subplots(*shape, figsize=size, layout="constrained")
    styles = [
        ("yao-rs, direct", "#729888", "--", ["native"], ["execute"]),
        ("yao-rs, fastest mode", "#087b57", "-", ["native"], ["execute", "fused2_execute", "fused4_execute"]),
        ("Yao.jl", "#3b67ad", "-", ["julia"], ["execute"]),
        ("Qulacs, fastest mode", "#bd702c", "-", ["qulacs", "qulacs_fused4"], ["execute"]),
    ]
    for ax, (prefix, title) in zip(axes, [("qft_", "Quantum Fourier transform"),
                                         ("layers100_", "100 Ry–CX layers")], strict=True):
        cases = sorted({r["case"] for r in rows if r["case"].startswith("circuits/" + prefix)},
                       key=lambda case: int(case.rsplit("_", 1)[1]))
        if not cases:
            raise ValueError(f"Missing chart cases: {prefix}")
        xs = [int(case.rsplit("_", 1)[1]) for case in cases]
        for label, color, style, backends, phases in styles:
            ys = [min(statistics.median(r["run_medians_ns"]) for r in rows
                      if r["case"] == case and r["threads"] == 1
                      and r["backend"] in backends and r["phase"] in phases) / 1e6
                  for case in cases]
            ax.plot(xs, ys, label=label, color=color, linestyle=style,
                    marker="o", markersize=4, linewidth=1.8)
        ax.set(title=title, xlabel="Qubits", ylabel="Execution time (ms · log scale)", yscale="log", xticks=xs)
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.grid(axis="y", alpha=.18)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["bottom", "left"]].set_color("#c5cec9")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, frameon=False)
    fig.savefig(args.output, metadata={"Date": None,
        "Description": "Source: " + data["source_commit"] + "; medians of independent processes. Fusion preparation excluded."})
    output = Path(args.output)
    if output.suffix.lower() == '.svg':
        output.write_text('\n'.join(line.rstrip() for line in output.read_text().splitlines()) + '\n')


if __name__ == "__main__":
    main()

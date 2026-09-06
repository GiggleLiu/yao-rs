#!/usr/bin/env python3
"""Export standalone expectation and slicing time/memory comparisons."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument("directory", type=Path)
a = p.parse_args()
rows = json.loads((a.directory / "summary.json").read_text())
rows = [r for r in rows if r["threads"] == 1]
lookup = {(r["id"], r["backend"]): r for r in rows}
ids = sorted({r["id"] for r in rows if r["backend"] == "native"})
fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
for ax, noisy in zip(axes, [False, True]):
    selected = [case for case in ids if ("_dm_" in case) == noisy]
    for backend, label, color in [
        ("native", "Native", "C0"),
        ("julia", "Yao", "C1"),
        ("tenferro_unsliced_warm", "Tenferro unsliced", "C2"),
        ("tenferro_term_sliced_warm", "Tenferro term slices", "C3"),
        ("omeinsum_unsliced", "omeinsum unsliced", "C4"),
        ("omeinsum_term_sliced", "omeinsum term slices", "C5"),
    ]:
        values = [lookup[(case, backend)] for case in selected]
        ax.errorbar(
            [4, 6],
            [v["median_ns"] / 1e6 for v in values],
            yerr=[
                [(v["median_ns"] - v["min_run_median_ns"]) / 1e6 for v in values],
                [(v["max_run_median_ns"] - v["median_ns"]) / 1e6 for v in values],
            ],
            marker="o",
            label=label,
            color=color,
            capsize=3,
        )
    ax.set(
        yscale="log",
        xlabel="Qubits",
        ylabel="Expectation (ms)",
        xticks=[4, 6],
        title="Noisy density matrix" if noisy else "Pure state",
    )
    ax.grid(alpha=0.2)
axes[0].legend(fontsize=7)
fig.suptitle("Five-term complex polynomial · 1 thread · bars = range of run medians")
fig.savefig(a.directory / "observable-costs.svg")
fig.savefig(a.directory / "observable-costs.png", dpi=180)

memory = json.loads((a.directory / "tensor-memory.json").read_text())
fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
workloads = [
    ("chain", 32),
    ("chain", 128),
    ("chain", 256),
    ("outer", 32),
    ("outer", 64),
]
for ax, (kind, n) in zip(axes.flat, workloads):
    for backend, color in [("tenferro", "C2"), ("omeinsum", "C4")]:
        for mode, marker in [
            ("unsliced", "o"),
            ("fixed_output", "s"),
            ("auto", "^"),
            ("greedy", "D"),
        ]:
            selected = [
                r
                for r in memory
                if r["dimension"] == n
                and r["kind"] == kind
                and r["mode"] == mode
                and r["backend"] == backend
            ]
            if not selected:
                continue
            row = selected[0]
            name = f"{backend}_{mode}" + ("_warm" if backend == "tenferro" else "")
            timing = lookup[(f"matrix_{kind}_{n}", name)]
            ax.scatter(
                timing["median_ns"] / 1e6,
                row["peak_rss_bytes"] / 1048576,
                color=color,
                marker=marker,
                label=f"{backend} {mode}",
                s=55,
            )
    ax.set(
        xscale="log",
        xlabel="Contraction (ms)",
        ylabel="Process peak RSS (MiB)",
        title=f"{n} × {n} · {kind} path",
    )
    ax.grid(alpha=0.2)
handles, labels = [], []
for ax in axes.flat:
    h, l = ax.get_legend_handles_labels()
    for item, label in zip(h, l):
        if label not in labels:
            handles.append(item)
            labels.append(label)
axes[1, 2].axis("off")
axes[1, 2].legend(handles, labels, fontsize=8, loc="center")
fig.suptitle(
    "Complex128 · 1 thread · fixed-path slices vs separately labelled replanning"
)
fig.savefig(a.directory / "slicing-tradeoff.svg")
fig.savefig(a.directory / "slicing-tradeoff.png", dpi=180)
for name in ["observable-costs.svg", "slicing-tradeoff.svg"]:
    path = a.directory / name
    path.write_text(
        "\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n"
    )

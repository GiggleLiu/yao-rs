#!/usr/bin/env python3
"""Standalone circuit AD cost and memory plots, with incomplete runs labelled."""

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
cases = {c["id"]: c for c in json.loads((a.directory / "cases.json").read_text())}
series = [
    ("native", "Native", "C0"),
    ("julia", "Yao", "C1"),
    ("tenferro_circuit_ad", "Tenferro circuit", "C2"),
    ("tenferro_composed_ad", "Tensor composition", "C3"),
]
fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
for ax, n in zip(axes, [8, 12, 16]):
    for backend, label, color in series:
        selected = sorted(
            (
                r
                for r in rows
                if r["threads"] == 1
                and r["backend"] == backend
                and r["id"] in cases
                and cases[r["id"]]["circuit"]["num_qubits"] == n
            ),
            key=lambda r: cases[r["id"]]["custom_loss"]["depth"],
        )
        ax.errorbar(
            [cases[r["id"]]["custom_loss"]["depth"] for r in selected],
            [r["median_ns"] / 1e6 for r in selected],
            yerr=[
                [(r["median_ns"] - r["min_run_median_ns"]) / 1e6 for r in selected],
                [(r["max_run_median_ns"] - r["median_ns"]) / 1e6 for r in selected],
            ],
            marker="o",
            color=color,
            label=label,
            capsize=3,
        )
    ax.set(
        xscale="log",
        yscale="log",
        xlabel="Layers",
        ylabel="Value + gradients (ms)",
        title=f"{n} qubits",
        xticks=[10, 100],
        xticklabels=["10", "100"],
    )
    ax.grid(alpha=0.2)
axes[0].legend(fontsize=8)
fig.suptitle(
    "Complex128 · 1 thread · same loss/parameter/input gradients · bars = run range"
)
fig.savefig(a.directory / "circuit-ad-costs.svg")
fig.savefig(a.directory / "circuit-ad-costs.png", dpi=180)

memory = json.loads((a.directory / "circuit-ad-memory.json").read_text())
fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
for ax, n in zip(axes.flat, [8, 12, 16]):
    for backend, label, color in [
        ("native", "Native", "C0"),
        ("custom", "Tenferro circuit", "C2"),
        ("composed", "Tensor composition", "C3"),
    ]:
        selected = sorted(
            (
                r
                for r in memory
                if r["qubits"] == n
                and r["backend"] == backend
                and r["status"] == "complete"
            ),
            key=lambda r: r["depth"],
        )
        ax.plot(
            [r["depth"] for r in selected],
            [r["peak_rss_bytes"] / 1048576 for r in selected],
            marker="o",
            color=color,
            label=label,
        )
        for r in memory:
            if (
                r["qubits"] == n
                and r["backend"] == backend
                and r["status"] != "complete"
                and "peak_rss_bytes" in r
            ):
                ax.scatter(
                    [r["depth"]],
                    [r["peak_rss_bytes"] / 1048576],
                    marker="^",
                    facecolors="none",
                    edgecolors=color,
                )
                ax.annotate(
                    "CPU limit; incomplete",
                    (r["depth"], r["peak_rss_bytes"] / 1048576),
                    xytext=(-115, -18),
                    textcoords="offset points",
                    fontsize=8,
                )
    ax.set(
        xscale="log",
        yscale="log",
        xticks=[10, 100],
        xticklabels=["10", "100"],
        xlabel="Layers",
        ylabel="Process peak RSS (MiB)",
        title=f"{n} qubits",
    )
    ax.grid(alpha=0.2)
axes[0, 0].legend(fontsize=8)
ax = axes[1, 1]
for backend, label, color in [
    ("custom", "Tenferro circuit", "C2"),
    ("composed", "Tensor composition", "C3"),
]:
    selected = sorted(
        (
            r
            for r in memory
            if r["qubits"] == 16
            and r["backend"] == backend
            and r.get("forward_retained_bytes") is not None
        ),
        key=lambda r: r["depth"],
    )
    ax.plot(
        [r["depth"] for r in selected],
        [r["forward_retained_bytes"] / 1048576 for r in selected],
        marker="o",
        color=color,
        label=label,
    )
ax.set(
    xscale="log",
    yscale="log",
    xticks=[10, 100],
    xticklabels=["10", "100"],
    xlabel="Layers",
    ylabel="Retained Rust heap (MiB)",
    title="16 qubits · forward tape/input storage",
)
ax.grid(alpha=0.2)
ax.legend(fontsize=8)
fig.suptitle(
    "Isolated memory probes · incomplete composition has no completed backward result"
)
fig.savefig(a.directory / "circuit-ad-memory.svg")
fig.savefig(a.directory / "circuit-ad-memory.png", dpi=180)

# Matplotlib emits trailing spaces inside path data; normalize generated text.
for name in ["circuit-ad-costs.svg", "circuit-ad-memory.svg"]:
    path = a.directory / name
    path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")

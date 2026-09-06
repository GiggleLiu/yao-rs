#!/usr/bin/env python3
"""Standalone trajectory accuracy/time and memory figures from raw reports."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("directory", type=Path)
a = parser.parse_args()
summary = json.loads((a.directory / "summary.json").read_text())
times = {(r["threads"], r["id"], r["backend"]): r for r in summary}
stats = json.loads((a.directory / "trajectory-statistics.json").read_text())
fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
for ax, threads in zip(axes, [1, 4]):
    for n, color in [(4, "#4477aa"), (6, "#228833"), (8, "#cc6677")]:
        rows = [
            r
            for r in stats
            if r["threads"] == threads and r["id"] == f"noisy_expectation_{n}"
        ]
        rows.sort(key=lambda r: r["trajectories"])
        x = [
            times[(threads, r["id"], f"trajectory_{r['trajectories']}")]["median_ns"]
            / 1e6
            for r in rows
        ]
        ax.plot(x, [r["rmse"] for r in rows], "o-", color=color, label=f"{n}q observed")
        ax.plot(
            x,
            [r["predicted_rms_error"] for r in rows],
            "--",
            color=color,
            label=f"{n}q from SE",
        )
        if n == 6:
            for t, row in zip(x, rows):
                ax.annotate(
                    str(row["trajectories"]),
                    (t, row["rmse"]),
                    xytext=(3, 6),
                    textcoords="offset points",
                    fontsize=7,
                )
    ax.set(
        xscale="log",
        yscale="log",
        xlabel="Ensemble execution (ms)",
        ylabel="Complex expectation error",
        title=f"{threads} trajectory worker(s)",
    )
    ax.grid(alpha=0.2)
fig.legend(
    *axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=3, fontsize=8
)
fig.suptitle(
    "Entangled noise · 6q: eight seeds; 4q/8q: one seed · labels = trajectories"
)
for ext in ["svg", "png"]:
    fig.savefig(a.directory / f"trajectory-error-time.{ext}", dpi=180)
plt.close(fig)

memory = json.loads((a.directory / "trajectory-memory.json").read_text())
fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
for backend, count, threads, label, color in [
    ("density", 0, 1, "Exact density, 1 worker", "#cc6677"),
    ("trajectory", 256, 1, "256 trajectories, 1 worker", "#4477aa"),
    ("trajectory", 256, 4, "256 trajectories, 4 workers", "#228833"),
]:
    rows = sorted(
        [
            r
            for r in memory
            if r["backend"] == backend
            and r["trajectories"] == count
            and r["threads"] == threads
        ],
        key=lambda r: r["qubits"],
    )
    axes[0].plot(
        [r["qubits"] for r in rows],
        [r["execution_peak_bytes"] / 2**20 for r in rows],
        "o-",
        label=label,
        color=color,
    )
axes[0].set(
    yscale="log",
    xlabel="Qubits",
    ylabel="Additional execution heap (MiB)",
    title="Execution buffers beyond live inputs",
)
axes[0].legend(fontsize=8)
for threads, color in [(1, "#4477aa"), (4, "#228833")]:
    rows = sorted(
        [
            r
            for r in memory
            if r["backend"] == "trajectory"
            and r["qubits"] == 16
            and r["threads"] == threads
        ],
        key=lambda r: r["trajectories"],
    )
    x = [r["trajectories"] for r in rows]
    axes[1].plot(
        x,
        [r["execution_peak_bytes"] / 2**20 for r in rows],
        "o-",
        color=color,
        label=f"{threads} worker(s), execution heap",
    )
    axes[1].plot(
        x,
        [r["peak_rss_bytes"] / 2**20 for r in rows],
        "s--",
        color=color,
        label=f"{threads} worker(s), process RSS",
    )
axes[1].set(
    xscale="log",
    yscale="log",
    xlabel="Trajectories",
    ylabel="Memory (MiB)",
    title="16-qubit product fixture: sample-count scaling",
)
axes[1].legend(fontsize=8)
for ax in axes:
    ax.grid(alpha=0.2)
fig.suptitle(
    "Isolated probes · 4/8q entangled; 10/12/16q product · heap and RSS differ"
)
for ext in ["svg", "png"]:
    fig.savefig(a.directory / f"trajectory-memory.{ext}", dpi=180)
for path in a.directory.glob("trajectory-*.svg"):
    path.write_text(
        "\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n"
    )

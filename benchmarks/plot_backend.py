#!/usr/bin/env python3
"""Standalone publication artifacts from measured baseline data (requires matplotlib)."""

import argparse
import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument("directory", type=Path)
a = p.parse_args()
data = json.loads((a.directory / "summary.json").read_text())
fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
for ax, prefix in zip(axes, ["rx_", "qft_"]):
    for backend, label in [("native", "yao-rs native"), ("julia", "Yao.jl")]:
        points = sorted(
            (int(r["id"].rsplit("_", 1)[1]), r["median_ns"] / 1e6)
            for r in data
            if r["threads"] == 1
            and r["backend"] == backend
            and r["id"].startswith(prefix)
        )
        if points:
            ax.plot(*zip(*points), marker="o", label=label)
    ax.set(
        xlabel="Qubits",
        ylabel="Time (ms)",
        yscale="log",
        title=prefix.rstrip("_") + " · complex128 · 1 thread",
    )
    ax.grid(alpha=0.2)
    ax.legend()
fig.savefig(a.directory / "cpu-scaling.svg")
fig.savefig(a.directory / "cpu-scaling.png", dpi=180)
plt.close(fig)
fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
for n in [8, 12, 16]:
    points = []
    for depth in [10, 100]:
        path = a.directory / f"memory-{n}-{depth}.log"
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            if line.startswith("{"):
                row = json.loads(line)
                if row.get("phase") == "ad_forward_tape":
                    points.append(
                        (depth, row["retained_additional_rust_heap_bytes"] / 2**20)
                    )
    if points:
        ax.plot(
            *zip(*points),
            marker="o",
            color="C0",
            label="Conjugation, 8–16 qubits" if n == 16 else None,
        )
points = []
for depth in [10, 100]:
    path = a.directory / f"memory-nonlinear-16-{depth}.log"
    if path.exists():
        for line in path.read_text().splitlines():
            if line.startswith("{"):
                row = json.loads(line)
                if row.get("phase") == "ad_forward_tape":
                    points.append(
                        (depth, row["retained_additional_rust_heap_bytes"] / 2**20)
                    )
if points:
    ax.plot(*zip(*points), marker="s", linestyle="--", label="16 qubits, scaled sin")
ax.set(
    xlabel="Layers (conjugation unless labeled)",
    ylabel="Retained Rust heap (MiB)",
    yscale="log",
    title="Tenferro eager complex AD tape",
)
ax.grid(alpha=0.2)
ax.legend()
fig.savefig(a.directory / "ad-memory.svg")
fig.savefig(a.directory / "ad-memory.png", dpi=180)

plt.close(fig)
fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
lookup = {(r["id"], r["backend"]): r for r in data if r["threads"] == 1}
cases = [("tensor_state_8", "8-qubit state"), ("noisy_4", "4-qubit density")]
for offset, (backend, label) in enumerate(
    [
        ("native", "Native circuit"),
        ("omeinsum", "omeinsum, from arrays"),
        ("tenferro_from_arrays", "tenferro, from arrays"),
        ("tenferro_warm", "tenferro, prepared"),
    ]
):
    rows = [lookup[(case, backend)] for case, _ in cases]
    heights = [r["median_ns"] / 1000 for r in rows]
    errors = [
        [max(0, r["median_ns"] - r["min_run_median_ns"]) / 1000 for r in rows],
        [max(0, r["max_run_median_ns"] - r["median_ns"]) / 1000 for r in rows],
    ]
    ax.bar(
        [i + (offset - 1.5) * 0.18 for i in range(len(cases))],
        heights,
        width=0.18,
        label=label,
        yerr=errors,
        capsize=2,
    )
ax.set(
    xticks=range(len(cases)),
    xticklabels=[label for _, label in cases],
    ylabel="Time (µs)",
    yscale="log",
    title="CPU circuit and contraction costs · 1 thread",
)
ax.legend(fontsize=8)
ax.grid(axis="y", alpha=0.2)
fig.savefig(a.directory / "tensor-costs.svg")
fig.savefig(a.directory / "tensor-costs.png", dpi=180)

#!/usr/bin/env python3
"""Plot product-formula accuracy versus measured cost; no dense/sparse cost conflation."""

import argparse
import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

p = argparse.ArgumentParser()
p.add_argument("directory", type=Path)
a = p.parse_args()
rows = json.loads((a.directory / "summary.json").read_text())
cases = {
    c["id"]: c["evolution"]
    for c in json.loads((a.directory / "cases.json").read_text())
    if "evolution" in c
}
fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
for ax, model in zip(axes, ["ising", "heisenberg"]):
    for order, color in [(1, "C0"), (2, "C1")]:
        for backend, style in [("native", "-"), ("julia", "--")]:
            selected = sorted(
                (
                    r
                    for r in rows
                    if r["threads"] == 1
                    and r["backend"] == backend
                    and r["id"] in cases
                    and cases[r["id"]]["model"] == model
                    and cases[r["id"]]["order"] == order
                ),
                key=lambda r: cases[r["id"]]["steps"],
            )
            ax.errorbar(
                [r["median_ns"] / 1000 for r in selected],
                [r["relative_state_error"] for r in selected],
                xerr=[
                    [
                        (r["median_ns"] - r["min_run_median_ns"]) / 1000
                        for r in selected
                    ],
                    [
                        (r["max_run_median_ns"] - r["median_ns"]) / 1000
                        for r in selected
                    ],
                ],
                linestyle=style,
                capsize=2,
                color=color,
                marker="o",
                label=f"{'Rust' if backend == 'native' else 'Yao'} · order {order}",
            )
            for r in selected:
                if backend == "julia":
                    continue
                ax.annotate(
                    str(cases[r["id"]]["steps"]),
                    (r["median_ns"] / 1000, r["relative_state_error"]),
                    fontsize=7,
                    xytext=(3, 3),
                    textcoords="offset points",
                )
    ax.set(
        xscale="log",
        yscale="log",
        xlabel="Execution time (µs)",
        ylabel="Relative state error",
        title=f"{model.title()} · 3 qubits · t = 0.8",
    )
    ax.grid(alpha=0.2)
    ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.legend(fontsize=8)
fig.suptitle(
    "Same formulas · complex128 · 1 thread · labels = steps · bars = run range"
)
fig.savefig(a.directory / "evolution-error-time.svg")
fig.savefig(a.directory / "evolution-error-time.png", dpi=180)

#!/usr/bin/env python3
"""Standalone latency and observed device-memory figures from CUDA report data."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save(fig, directory, name):
    fig.tight_layout()
    for extension in ["svg", "png"]:
        target = directory / f"{name}.{extension}"
        fig.savefig(target, dpi=180, bbox_inches="tight")
        if extension == "svg":
            target.write_text(
                "\n".join(line.rstrip() for line in target.read_text().splitlines())
                + "\n"
            )
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    directory = args.directory
    cases = json.loads((directory / "cases.json").read_text())
    summary = json.loads((directory / "summary.json").read_text())
    lookup = {(row["id"], row["backend"]): row for row in summary}
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    backends = ["native", "julia", "cuda_resident", "cuda_transfer_inclusive"]
    names = ["Native CPU", "Yao CPU", "GPU resident", "GPU + transfers"]
    colors = ["#666666", "#7b3294", "#008837", "#c47b00"]
    for ax, mode, title in zip(
        axes,
        ["state", "custom_gradient", "density"],
        ["Unitary state", "Loss + all gradients", "Exact noisy density"],
    ):
        selected = [case for case in cases if case["mode"] == mode]
        y = np.arange(len(selected))
        for i, (backend, name, color) in enumerate(zip(backends, names, colors)):
            rows = [lookup[(case["id"], backend)] for case in selected]
            values = np.array([row["median_ns"] / 1e6 for row in rows])
            lower = values - np.array([row["min_run_median_ns"] / 1e6 for row in rows])
            upper = np.array([row["max_run_median_ns"] / 1e6 for row in rows]) - values
            ax.barh(
                y + (i - 1.5) * 0.2,
                values,
                height=0.19,
                color=color,
                label=name,
                xerr=[lower, upper],
                capsize=2,
            )
        ax.set_yticks(
            y,
            [
                case["id"]
                .replace("cuda_", "")
                .replace("custom_", "")
                .replace(mode + "_", "")
                .replace("gradient_", "")
                .replace("_depth", "q / depth ")
                for case in selected
            ],
        )
        ax.set_xscale("log")
        ax.set_xlabel("Milliseconds (log scale)")
        ax.set_title(title)
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.2)
    axes[0].legend(fontsize=8, loc="best")
    fig.suptitle(
        "Complex128 · GPU and host CPU · medians and range of process medians"
    )
    save(fig, directory, "cuda-latency")

    records = json.loads((directory / "cuda-qualification.json").read_text())
    memory = {row["id"]: row for row in records}
    gradient_cases = [case for case in cases if case["mode"] == "custom_gradient"]
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for n in sorted({case["circuit"]["num_qubits"] for case in gradient_cases}):
        group = sorted(
            (case for case in gradient_cases if case["circuit"]["num_qubits"] == n),
            key=lambda case: int(case["id"].split("depth")[1]),
        )
        depths = [int(case["id"].split("depth")[1]) for case in group]
        sizes = [
            memory[case["id"]]["snapshots"]["after_repeats"] / 1048576 for case in group
        ]
        ax.plot(depths, sizes, marker="o", label=f"{n} qubits")
    ax.set_xlabel("Circuit depth (4 gates per layer)")
    ax.set_ylabel("GPU process memory after repeats (MiB)")
    ax.set_title("Composed AD: observed memory versus depth")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.text(
        0.5,
        -0.02,
        "Isolated process per point; includes context, workspaces and allocator pool.\nSnapshots are not continuous peaks or exact live tensor storage.",
        ha="center",
        fontsize=9,
    )
    save(fig, directory, "cuda-memory")


if __name__ == "__main__":
    main()

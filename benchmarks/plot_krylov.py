#!/usr/bin/env python3
"""Standalone achieved-error and memory figures for the Krylov CPU suite."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save(fig, directory, name):
    fig.savefig(directory / f"{name}.svg", bbox_inches="tight")
    fig.savefig(directory / f"{name}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    directory = args.directory
    rows = json.loads((directory / "summary.json").read_text())
    lookup = {(r["threads"], r["id"], r["backend"]): r for r in rows}
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    colors = {"native": "#2166ac", "julia": "#d6604d"}
    labels = {"native": "Rust Krylov", "julia": "Yao TimeEvolution"}
    fig, axes = plt.subplots(3, 2, figsize=(11, 10), layout="constrained")
    for i, n in enumerate((8, 12, 16)):
        for j, model in enumerate(("ising", "heisenberg")):
            ax = axes[i, j]
            for backend in ("native", "julia"):
                points = [lookup[(1, f"krylov_{model}_{n}q_tol{p}", backend)] for p in (4, 7, 10)]
                ax.loglog([r["median_ns"] / 1e6 for r in points],
                          [max(r["relative_state_error"], 1e-16) for r in points],
                          "o-", color=colors[backend], label=labels[backend])
            ax.set(title=f"{model.title()} · {n} qubits", xlabel="Execution time (ms)", ylabel="Relative state error")
            ax.grid(True, which="major", alpha=0.2)
            if i == j == 0:
                ax.legend()
    fig.suptitle("CPU evolution: achieved error versus time · one thread\nThree tolerance settings per curve; solver stopping criteria differ", fontsize=13)
    save(fig, directory, "krylov-error-time")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), layout="constrained")
    for ax, model in zip(axes, ("ising", "heisenberg")):
        for backend, label, color in [("native", "Rust Suzuki", "#2166ac"), ("julia", "Yao Suzuki", "#d6604d"),
                                      ("supported_warm", "Tenferro Suzuki (warm)", "#1b7837"),
                                      ("omeinsum_fixed_tree", "Omeinsum Suzuki (fixed tree)", "#762a83")]:
            points = [lookup[(1, f"krylov_product_{model}_4q_steps{s}", backend)] for s in (2, 8, 32)]
            errors = [lookup[(1, r["id"], "native")]["relative_state_error"] for r in points]
            ax.loglog([r["median_ns"] / 1e3 for r in points], errors, "o-", color=color, label=label)
        for backend in ("native", "julia"):
            points = [lookup[(1, f"krylov_{model}_4q_tol{p}", backend)] for p in (4, 7, 10)]
            ax.loglog([r["median_ns"] / 1e3 for r in points],
                      [max(r["relative_state_error"], 1e-16) for r in points],
                      "x--", color=colors[backend], label=labels[backend])
        ax.set(title=f"{model.title()} · 4 qubits", xlabel="Execution time (µs)", ylabel="Relative state error")
        ax.grid(True, alpha=0.2)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="outside lower center", ncol=2, fontsize=9)
    fig.suptitle("Same Hamiltonian, zero input and t=0.8 · one thread\nTensor curves execute Suzuki circuits; approximation errors differ", fontsize=12)
    save(fig, directory, "krylov-product-comparison")

    memory = json.loads((directory / "krylov-memory.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7), layout="constrained")
    for model, style in (("ising", "-"), ("heisenberg", "--")):
        for k, color in ((8, "#2166ac"), (20, "#d6604d"), (40, "#1b7837")):
            points = sorted((r for r in memory if r["model"] == model and r["krylov_dim"] == k), key=lambda r: r["qubits"])
            for ax, field in zip(axes, ("execution_peak_bytes", "peak_rss_bytes")):
                ax.semilogy([r["qubits"] for r in points], [r[field] / 2**20 for r in points],
                            marker="o", linestyle=style, color=color, label=f"{model}, cap {k}")
                ax.set(xlabel="Qubits", ylabel="MiB", xticks=[4, 8, 12, 16])
                ax.grid(True, alpha=0.2)
    axes[0].set_title("Additional Rust execution heap")
    axes[1].set_title("Whole-process peak RSS")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="outside lower center", ncol=3, fontsize=9)
    fig.suptitle("Krylov memory at rtol=1e-8\nBasis caps are maxima; convergence can use fewer vectors", fontsize=12)
    save(fig, directory, "krylov-memory")


if __name__ == "__main__":
    main()

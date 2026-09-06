#!/usr/bin/env python3
"""Standalone publication artifacts from measured baseline data (requires matplotlib)."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

p=argparse.ArgumentParser();p.add_argument("directory",type=Path);a=p.parse_args()
data=json.loads((a.directory/"summary.json").read_text())
fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
for ax,prefix in zip(axes,["rx_","qft_"]):
    for backend,label in [("native","yao-rs native"),("julia","Yao.jl")]:
        points=sorted((int(r["id"].rsplit("_",1)[1]),r["median_ns"]/1e6) for r in data if r["threads"]==1 and r["backend"]==backend and r["id"].startswith(prefix))
        if points: ax.plot(*zip(*points),marker="o",label=label)
    ax.set(xlabel="Qubits",ylabel="Time (ms)",yscale="log",title=prefix.rstrip("_")+" · complex128 · 1 thread")
    ax.grid(alpha=.2);ax.legend()
fig.savefig(a.directory/"cpu-scaling.svg")
fig.savefig(a.directory/"cpu-scaling.png",dpi=180)
plt.close(fig)
fig,ax=plt.subplots(figsize=(6,4),constrained_layout=True)
for n in [8,12,16]:
    points=[]
    for depth in [10,100]:
        path=a.directory/f"memory-{n}-{depth}.log"
        if not path.exists():continue
        for line in path.read_text().splitlines():
            if line.startswith('{'):
                row=json.loads(line)
                if row.get("phase")=="ad_forward_tape":points.append((depth,row["retained_additional_rust_heap_bytes"]/2**20))
    if points:ax.plot(*zip(*points),marker="o",label=f"{n} qubits")
ax.set(xlabel="Conjugation layers",ylabel="Retained Rust heap (MiB)",title="Tenferro eager complex AD tape")
ax.grid(alpha=.2);ax.legend()
fig.savefig(a.directory/"ad-memory.svg")
fig.savefig(a.directory/"ad-memory.png",dpi=180)

"""
Benchmark figure from a benchmark.csv written by tools/benchmark.sh.

(a) CPU strong scaling: XBeach MPI cost per cell-step versus ranks for each grid size, with the GPU
    cost as horizontal lines (XBeach MPI needs at least 2 ranks, so there is no 1-rank point).
(b) Cost per cell and time step versus grid size: best CPU rank count, GPU fp64, GPU fp32.
Usage: python tools/plot_benchmark.py <benchmark.csv> [out.png]
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Arial"]
import matplotlib.pyplot as plt
import pandas as pd

import sys
CSV = Path(sys.argv[1])
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else CSV.with_suffix(".png")
COL = {600: "#1f4e79", 2400: "#2a9d8f", 9600: "#8ab17d", 33400: "#c0392b"}


def main():
    df = pd.read_csv(CSV)
    cpu = df[df.code == "cpu_mpi"]
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.4))
    for ny, g in cpu.groupby("ny"):
        g = g.sort_values("ranks")
        c = COL.get(ny, "k")
        lab = f"{g.cells.iloc[0] / 1e6:.2f} M cells"
        ax[0].plot(g.ranks, g.s_per_cellstep, "o-", color=c, ms=4, lw=1.2, label=f"CPU, {lab}")
        for code, ls in (("gpu_fp64", "--"), ("gpu_fp32", ":")):
            gg = df[(df.code == code) & (df.ny == ny)]
            if len(gg):
                ax[0].axhline(gg.s_per_cellstep.iloc[0], color=c, ls=ls, lw=1.0)
    r = cpu[cpu.ny == 600].sort_values("ranks")
    ax[0].plot(r.ranks, r.s_per_cellstep.iloc[0] * 2 / r.ranks, color="0.6", lw=0.8, label="Ideal (from 2 ranks)")
    ax[0].plot([], [], "k--", lw=1.0, label="GPU fp64")
    ax[0].plot([], [], "k:", lw=1.0, label="GPU fp32")
    ax[0].set_xscale("log", base=2)
    ax[0].set_xticks([2, 4, 8, 16, 24, 32], ["2", "4", "8", "16", "24", "32"])
    ax[0].set_yscale("log")
    ax[0].set_xlabel("MPI ranks")
    ax[0].set_ylabel("Wall time per cell and step (s)")
    ax[0].set_title("(a) CPU strong scaling vs GPU", fontsize=9, loc="left")
    ax[0].legend(fontsize=6, loc="upper right", framealpha=0.9)
    best = cpu.loc[cpu.groupby("ny").s_per_cellstep.idxmin()].sort_values("cells")
    ax[1].plot(best.cells, best.s_per_cellstep, "o-", color="#1f4e79", ms=4,
               label="CPU MPI, best rank count")
    for code, c, m in (("gpu_fp64", "#e69f00", "s"), ("gpu_fp32", "#c0392b", "^")):
        g = df[df.code == code].sort_values("cells")
        ax[1].plot(g.cells, g.s_per_cellstep, m + "-", color=c, ms=4, label=code.replace("_", " ").upper())
    for r in best.itertuples():
        ax[1].annotate(f"{r.ranks}", (r.cells, r.s_per_cellstep), xytext=(3, 4), textcoords="offset points", fontsize=6)
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("Grid cells")
    ax[1].set_ylabel("Wall time per cell and step (s)")
    ax[1].set_title("(b) Cost per cell-step", fontsize=9, loc="left")
    ax[1].legend(fontsize=6, loc="center right", framealpha=0.9)
    ax[1].set_ylim(1.5e-8, 2.5e-7)
    for a in ax:
        a.grid(True, which="major", lw=0.5, alpha=0.5)
        a.grid(True, which="minor", lw=0.3, alpha=0.25)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=600, bbox_inches="tight")
    print("written", OUT)


if __name__ == "__main__":
    main()

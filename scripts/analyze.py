import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullLocator

from plot import SERIES, draw, setup_plots

# the example words for the neighbour table
WORDS = ["king", "water", "music", "war", "computer", "science"]


def n_of(path):
    return int(os.path.basename(path)[len("edges_perk_"):-len(".txt")])


def read_edges(path, ids):
    # the largest file has about 22.6 million lines, so it is read line by line and only the
    # degree of every word and the neighbours of the example words are kept
    deg = np.zeros(len(ids), np.int64)
    near = {w: [] for w in WORDS}
    edges = 0
    with open(path) as f:
        for line in f:
            a, b, dist = line.split()
            deg[ids[a]] += 1
            deg[ids[b]] += 1
            if a in near:
                near[a].append((float(dist), b))
            if b in near:
                near[b].append((float(dist), a))
            edges += 1
    return edges, deg, near


def degree_figure(deg, n, path):
    degrees, counts = np.unique(deg, return_counts=True)
    fig, ax = plt.subplots(figsize=(4.5, 3.0))
    draw(ax, degrees, counts, SERIES[0], "none", "o")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ends = [degrees[0], degrees[-1]]
    ax.set_xticks(ends)
    ax.set_xticklabels([str(k) for k in ends])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("degree k")
    ax.set_ylabel("number of words with degree k")
    ax.set_title(f"degree distribution at n = {n}", loc="left")
    fig.tight_layout()
    fig.savefig(path)


def hubs_table(deg, vocab, path):
    # the stable sort keeps words with the same degree in id order
    top = np.argsort(-deg, kind="stable")[:15]
    with open(path, "w") as f:
        print(r"\begin{tabular}{rlrr}", file=f)
        print(r"\toprule", file=f)
        print(r"rank & word & degree & frequency rank \\", file=f)
        print(r"\midrule", file=f)
        for rank, i in enumerate(top, 1):
            print(rf"{rank} & {vocab[i]} & {deg[i]} & {i + 1} \\", file=f)
        print(r"\bottomrule", file=f)
        print(r"\end{tabular}", file=f)


def neighbours_table(near, path):
    with open(path, "w") as f:
        print(r"\begin{tabular}{lrp{0.65\linewidth}}", file=f)
        print(r"\toprule", file=f)
        print(r"word & degree & 10 closest neighbours (distance) \\", file=f)
        print(r"\midrule", file=f)
        for w in WORDS:
            closest = sorted(near[w])[:10]
            cells = ", ".join(f"{b} ({dist:.3f})" for dist, b in closest)
            print(rf"{w} & {len(near[w])} & {cells} \\", file=f)
        print(r"\bottomrule", file=f)
        print(r"\end{tabular}", file=f)


def main():
    vocab_path, edges_dir, out_dir = sys.argv[1:4]
    fig_dir = os.path.join(out_dir, "figures")
    tab_dir = os.path.join(out_dir, "tables")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(tab_dir, exist_ok=True)

    with open(vocab_path) as f:
        vocab = f.read().splitlines()
    ids = {w: i for i, w in enumerate(vocab)}

    paths = sorted(glob.glob(os.path.join(edges_dir, "edges_perk_*.txt")), key=n_of)
    for path in paths:
        n = n_of(path)
        edges, deg, near = read_edges(path, ids)
        print(f"n = {n}: {edges} edges, average degree {2 * edges / n:.2f}")
    # the figure and the tables use the last file, the one with the largest n
    deg = deg[:n]

    setup_plots()
    degree_figure(deg, n, os.path.join(fig_dir, "degree.pdf"))
    hubs_table(deg, vocab, os.path.join(tab_dir, "hubs.tex"))
    neighbours_table(near, os.path.join(tab_dir, "neighbours.tex"))


if __name__ == "__main__":
    main()

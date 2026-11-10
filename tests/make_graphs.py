import os

import numpy as np


def triangle(n, d01, d12, d02):
    D = np.full((n, n), np.inf)
    np.fill_diagonal(D, 0)
    D[0, 1] = D[1, 0] = d01
    D[1, 2] = D[2, 1] = d12
    D[0, 2] = D[2, 0] = d02
    return D


def random_graph(n):
    rng = np.random.default_rng(1)
    W = rng.random((n, n))
    W[rng.random((n, n)) < 0.2] = np.inf
    # keep the upper triangle and mirror it: symmetric, zero diagonal
    D = np.triu(W, 1)
    return D + D.T


def sparse_graph(n):
    rng = np.random.default_rng(1)
    # scaling by pi lets a path's sum go above 1, where it rounds, so the result depends on the order of the
    # additions and the blocked variant can differ from serial in the last bits
    W = rng.random((n, n)) * np.pi
    W[rng.random((n, n)) < 0.9] = np.inf
    D = np.triu(W, 1)
    return D + D.T


def save(name, D):
    with open(f"out/graphs/{name}.bin", "wb") as f:
        np.array([len(D)], np.int32).tofile(f)
        D.tofile(f)


def main():
    os.makedirs("out/graphs", exist_ok=True)
    save("shortcut32", triangle(32, 0.2, 0.2, 0.5))
    save("shortcut64", triangle(64, 0.2, 0.2, 0.5))
    save("tie32", triangle(32, 0.25, 0.25, 0.5))
    for n in (32, 96, 160):
        save(f"rand{n}", random_graph(n))
    for n in (96, 160):
        save(f"sparse{n}", sparse_graph(n))


if __name__ == "__main__":
    main()

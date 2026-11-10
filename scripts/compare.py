import argparse
import filecmp
import sys

import numpy as np
from scipy.sparse.csgraph import csgraph_from_dense, floyd_warshall

TOL = 1e-12
EDGE_TOL = 1e-12  # same value as in src/common.h


def read_graph(path):
    n = int(np.fromfile(path, np.int32, 1)[0])
    return np.fromfile(path, np.float64, n * n, offset=4).reshape(n, n)


def read_pairs(path):
    pairs = []
    with open(path) as f:
        for line in f:
            i, j = line.split()[:2]
            pairs.append((int(i), int(j)))
    return pairs


def edge_rule(D0, D):
    # same rule as find_edges in src/common.cpp
    keep = np.zeros(D0.shape, bool)
    finite = np.isfinite(D0)
    keep[finite] = D0[finite] - D[finite] <= EDGE_TOL
    i, j = np.nonzero(np.triu(keep, 1))
    return list(zip(i.tolist(), j.tolist()))


def matrix_diff(A, B):
    # inf - inf is nan, so only subtract where A is not inf. A nan in A is kept on purpose: it makes max_diff nan,
    # which fails the check
    inf = np.isinf(A)
    same_inf = np.array_equal(inf, np.isinf(B))
    return same_inf, np.abs(A[~inf] - B[~inf]).max()


def oracle(args):
    D0 = read_graph(args.graph)
    D = read_graph(args.dump)
    ref = floyd_warshall(csgraph_from_dense(D0, null_value=np.inf), directed=False)
    inf_ok, max_diff = matrix_diff(D, ref)

    got = read_pairs(args.edges)
    edges_ok = got == edge_rule(D0, ref)

    ok = inf_ok and max_diff <= TOL and edges_ok
    line = (f"oracle n={len(D0)} max_diff={max_diff:.3g} inf_ok={inf_ok} "
            f"edges={len(got)} edges_ok={edges_ok}")
    if args.expect is not None:
        want = [tuple(map(int, p.split("-"))) for p in args.expect.split(",")]
        expect_ok = got == want
        ok = ok and expect_ok
        line += f" expect_ok={expect_ok}"
    return ok, line


def diff(args):
    A = read_graph(args.dump_a)
    B = read_graph(args.dump_b)
    inf_ok, max_diff = matrix_diff(A, B)
    same_edges = filecmp.cmp(args.edges_a, args.edges_b, shallow=False)

    ok = inf_ok and max_diff <= args.tol and same_edges
    return ok, (f"diff n={len(A)} max_diff={max_diff:.3g} tol={args.tol:g} "
                f"inf_ok={inf_ok} edges_identical={same_edges}")


def props(args):
    D = read_graph(args.dump)
    n = len(D)
    max_diag = np.abs(np.diag(D)).max()
    inf_sym, max_asym = matrix_diff(D, D.T)

    if n <= 100:
        via = (D[:, :, None] + D[None, :, :]).min(axis=1)
        direct = D
    else:
        # n^3 triples is too many, so check a sample
        i, k, j = np.random.default_rng(1).integers(0, n, (3, 10000))
        via = D[i, k] + D[k, j]
        direct = D[i, j]
    tri_bad = np.count_nonzero(direct > via + TOL)

    ok = max_diag == 0 and inf_sym and max_asym <= TOL and tri_bad == 0
    line = (f"props n={n} max_diag={max_diag:.3g} inf_sym={inf_sym} max_asym={max_asym:.3g} "
            f"tri_bad={tri_bad}/{direct.size}")
    if args.graph is not None:
        above = np.count_nonzero(D > read_graph(args.graph))
        ok = ok and above == 0
        line += f" above_graph={above}"
    return ok, line


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("oracle")
    p.add_argument("graph")
    p.add_argument("dump")
    p.add_argument("edges")
    p.add_argument("--expect")
    p.set_defaults(func=oracle)

    p = sub.add_parser("diff")
    p.add_argument("dump_a")
    p.add_argument("dump_b")
    p.add_argument("edges_a")
    p.add_argument("edges_b")
    p.add_argument("--tol", type=float, required=True)
    p.set_defaults(func=diff)

    p = sub.add_parser("props")
    p.add_argument("dump")
    p.add_argument("--graph")
    p.set_defaults(func=props)

    args = parser.parse_args()
    ok, line = args.func(args)
    print(("PASS " if ok else "FAIL ") + line)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

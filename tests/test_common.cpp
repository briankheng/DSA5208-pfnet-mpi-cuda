#include <cassert>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "common.h"

int main()
{
    // count_cooc: tokens {0,1,-1,2,0}, window 5, a == b pairs and -1 skipped
    assert(count_cooc({0, 1, -1, 2, 0}, 3, 3) == std::vector<double>({0, 2, 2, 2, 0, 1, 2, 1, 0}));
    // n < m keeps only the rows below n
    assert(count_cooc({0, 1, -1, 2, 0}, 2, 3) == std::vector<double>({0, 2, 2, 2, 0, 1}));

    // ppmi: off-diagonal ln 2, rows normalised to 1, zero row stays zero
    std::vector<double> G = {0, 2, 2, 0, 0, 0};
    ppmi_normalise(G, 3, 2);
    assert(G == std::vector<double>({0, 1, 1, 0, 0, 0}));
    // negative PMI clamps to 0: [[1,1],[1,3]] -> row 0 becomes [1, 0]
    std::vector<double> H = {1, 1, 1, 3};
    ppmi_normalise(H, 2, 2);
    assert(H[0] == 1 && H[1] == 0);

    // distance rule
    double a[2] = {1, 0}, b[2] = {0, 1}, c[2] = {0.6, 0.8};
    assert(sim_distance(a, b, 2) == INFINITY);
    assert(sim_distance(c, c, 2) >= 0 && sim_distance(c, c, 2) < 1e-15);
    assert(sim_distance(a, c, 2) == 1 / 0.6 - 1);

    // graph file round trip and partial read
    double D[9] = {0, 0.5, INFINITY, 0.5, 0, 0.2, INFINITY, 0.2, 0};
    write_graph("out/t.bin", D, 3);
    assert(graph_size("out/t.bin") == 3);
    assert(read_graph_rows("out/t.bin", 3, 1, 2) == std::vector<double>(D + 3, D + 9));

    // edges: (0,2) shortened from 0.9 to 0.4 is dropped, (0,1) and (1,2) kept
    double D0[9] = {0, 0.2, 0.9, 0.2, 0, 0.2, 0.9, 0.2, 0};
    double Ds[9] = {0, 0.2, 0.4, 0.2, 0, 0.2, 0.4, 0.2, 0};
    std::vector<Edge> e;
    find_edges(D0, Ds, 0, 3, 3, e);
    assert(e.size() == 2 && e[0].i == 0 && e[0].j == 1 && e[1].i == 1 && e[1].j == 2);
    // a slice holding only row 1 keeps global indices
    e.clear();
    find_edges(D0 + 3, Ds + 3, 1, 1, 3, e);
    assert(e.size() == 1 && e[0].i == 1 && e[0].j == 2);

    // CSV: two appends -> one header line plus two rows
    std::remove("out/t.csv");
    Times t;
    append_csv("out/t.csv", "serial", "serial", 3, 2, 1, t, 2);
    append_csv("out/t.csv", "serial", "serial", 3, 2, 1, t, 2);
    std::ifstream in("out/t.csv");
    std::string line, first;
    int lines = 0;
    while (std::getline(in, line)) {
        if (lines == 0)
            first = line;
        lines++;
    }
    assert(lines == 3 && first == "program,variant,n,m,p,t_load,t_count,t_ppmi,t_sim,t_fw,t_fw_comm,t_edges,t_total,edges");

    std::printf("common ok\n");
}

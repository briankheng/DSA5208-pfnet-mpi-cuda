#pragma once

#include <cmath>
#include <string>
#include <vector>

// sim_distance is called from both the CPU code and the CUDA kernel
#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

const int WINDOW = 5;
const double EDGE_TOL = 1e-12;

struct Edge { int i, j; double w; };
struct Times { double load = 0, count = 0, ppmi = 0, sim = 0, fw = 0, fw_comm = 0, edges = 0, total = 0; };

HD inline double sim_distance(const double *a, const double *b, int m)
{
    double s = 0;
    for (int c = 0; c < m; c++)
        s += a[c] * b[c];
    if (s <= 0)
        return INFINITY;
    // 1/s - 1 makes weak similarities long, so a chain of close words can beat a weak direct link
    double d = 1 / s - 1;
    // rounding can push s just above 1 for identical rows, and distances must stay >= 0
    return d > 0 ? d : 0;
}

std::vector<int> load_tokens(const std::string &path);
std::vector<std::string> load_vocab(const std::string &path);
std::vector<double> count_cooc(const std::vector<int> &tokens, int n, int m);  // n x m, row-major
void ppmi_normalise(std::vector<double> &F, int n, int m);                      // in place
int graph_size(const std::string &path);
std::vector<double> read_graph_rows(const std::string &path, int n, int row0, int nrows);
void write_graph(const std::string &path, const double *D, int n);
void find_edges(const double *D0, const double *D, int row0, int nrows, int n, std::vector<Edge> &edges);
void write_edges(const std::string &path, const std::vector<Edge> &edges, const std::vector<std::string> &names);
void append_csv(const std::string &path, const std::string &program, const std::string &variant,
                int n, int m, int p, const Times &t, long edges);

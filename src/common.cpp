#include "common.h"

#include <algorithm>
#include <cstdio>
#include <fstream>

std::vector<int> load_tokens(const std::string &path)
{
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    std::vector<int> tokens(in.tellg() / sizeof(int));
    in.seekg(0);
    in.read((char *)tokens.data(), tokens.size() * sizeof(int));
    return tokens;
}

std::vector<std::string> load_vocab(const std::string &path)
{
    std::ifstream in(path);
    std::vector<std::string> vocab;
    std::string word;
    while (std::getline(in, word))
        vocab.push_back(word);
    return vocab;
}

std::vector<double> count_cooc(const std::vector<int> &tokens, int n, int m)
{
    std::vector<double> F((size_t)n * m);
    size_t len = tokens.size();
    for (size_t t1 = 0; t1 < len; t1++) {
        int a = tokens[t1];
        if (a < 0)
            continue;
        for (size_t t2 = t1 + 1; t2 <= t1 + WINDOW && t2 < len; t2++) {
            int b = tokens[t2];
            if (b < 0 || a == b)
                continue;
            if (a < n && b < m)
                F[(size_t)a * m + b] += 1;
            if (b < n && a < m)
                F[(size_t)b * m + a] += 1;
        }
    }
    return F;
}

void ppmi_normalise(std::vector<double> &F, int n, int m)
{
    double T = 0;
    std::vector<double> R(n), C(m);
    for (int a = 0; a < n; a++)
        for (int c = 0; c < m; c++) {
            double f = F[(size_t)a * m + c];
            T += f;
            R[a] += f;
            C[c] += f;
        }

    for (int a = 0; a < n; a++)
        for (int c = 0; c < m; c++) {
            size_t k = (size_t)a * m + c;
            if (F[k] > 0)
                F[k] = std::max(0.0, std::log(F[k] * T / (R[a] * C[c])));
        }

    for (int a = 0; a < n; a++) {
        double *row = F.data() + (size_t)a * m;
        double norm = 0;
        for (int c = 0; c < m; c++)
            norm += row[c] * row[c];
        norm = std::sqrt(norm);
        if (norm > 0)
            for (int c = 0; c < m; c++)
                row[c] /= norm;
    }
}

int graph_size(const std::string &path)
{
    FILE *f = std::fopen(path.c_str(), "rb");
    int n = 0;
    size_t got = std::fread(&n, sizeof(int), 1, f);  // fread is declared warn_unused_result
    (void)got;
    std::fclose(f);
    return n;
}

std::vector<double> read_graph_rows(const std::string &path, int n, int row0, int nrows)
{
    std::vector<double> D((size_t)nrows * n);
    FILE *f = std::fopen(path.c_str(), "rb");
    // 4 bytes for n, then 8 bytes per double; long keeps the offset from overflowing at large n
    std::fseek(f, 4 + (long)row0 * n * 8, SEEK_SET);
    size_t got = std::fread(D.data(), sizeof(double), D.size(), f);  // fread is declared warn_unused_result
    (void)got;
    std::fclose(f);
    return D;
}

void write_graph(const std::string &path, const double *D, int n)
{
    FILE *f = std::fopen(path.c_str(), "wb");
    std::fwrite(&n, sizeof(int), 1, f);
    std::fwrite(D, sizeof(double), (size_t)n * n, f);
    std::fclose(f);
}

void find_edges(const double *D0, const double *D, int row0, int nrows, int n, std::vector<Edge> &edges)
{
    for (int r = 0; r < nrows; r++) {
        int i = row0 + r;
        const double *d0 = D0 + (size_t)r * n;
        const double *d = D + (size_t)r * n;
        for (int j = i + 1; j < n; j++)
            if (std::isfinite(d0[j]) && d0[j] - d[j] <= EDGE_TOL)
                edges.push_back({i, j, d0[j]});
    }
}

void write_edges(const std::string &path, const std::vector<Edge> &edges, const std::vector<std::string> &names)
{
    FILE *f = std::fopen(path.c_str(), "w");
    // on WSL, writes to /mnt/c in the default 4 KB chunks run at about 4 MB/s, so buffer 4 MB
    std::vector<char> buffer(1 << 22);
    std::setvbuf(f, buffer.data(), _IOFBF, buffer.size());
    for (const Edge &e : edges) {
        std::string a = names.empty() ? std::to_string(e.i) : names[e.i];
        std::string b = names.empty() ? std::to_string(e.j) : names[e.j];
        std::fprintf(f, "%s %s %.17g\n", a.c_str(), b.c_str(), e.w);
    }
    std::fclose(f);
}

void append_csv(const std::string &path, const std::string &program, const std::string &variant,
                int n, int m, int p, const Times &t, long edges)
{
    bool exists = std::ifstream(path).good();
    FILE *f = std::fopen(path.c_str(), "a");
    if (!exists)
        std::fprintf(f, "program,variant,n,m,p,t_load,t_count,t_ppmi,t_sim,t_fw,t_fw_comm,t_edges,t_total,edges\n");
    std::fprintf(f, "%s,%s,%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%ld\n",
                 program.c_str(), variant.c_str(), n, m, p,
                 t.load, t.count, t.ppmi, t.sim, t.fw, t.fw_comm, t.edges, t.total, edges);
    std::fclose(f);
}

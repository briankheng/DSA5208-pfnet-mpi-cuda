#include "common.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

double now()
{
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

void similarity(const std::vector<double> &F, int n, int m, std::vector<double> &D0)
{
    D0.resize((size_t)n * n);
    for (int i = 0; i < n; i++) {
        // Floyd-Warshall needs an exact 0 on the diagonal. resize already zero-fills it, this line makes that explicit
        D0[(size_t)i * n + i] = 0;
        const double *Fi = F.data() + (size_t)i * m;
        for (int j = i + 1; j < n; j++) {
            double d = sim_distance(Fi, F.data() + (size_t)j * m, m);
            D0[(size_t)i * n + j] = d;
            D0[(size_t)j * n + i] = d;
        }
    }
}

void floyd_warshall(std::vector<double> &D, int n)
{
    for (int k = 0; k < n; k++) {
        const double *Dk = D.data() + (size_t)k * n;
        for (int i = 0; i < n; i++) {
            double *Di = D.data() + (size_t)i * n;
            // D[i][k] cannot change during step k because D[k][k] = 0, so it is read once per row
            double dik = Di[k];
            for (int j = 0; j < n; j++) {
                double t = dik + Dk[j];
                if (t < Di[j])
                    Di[j] = t;
            }
        }
    }
}

int main(int argc, char **argv)
{
    std::string tokens_path = "data/tokens.bin", vocab_path = "data/vocab.txt";
    std::string graph_path, out_path = "edges.txt", csv_path, dump_path, fw = "serial";
    int n = 0, m = 2000;
    for (int a = 1; a < argc; a++) {
        std::string opt = argv[a];
        if (opt == "--tokens")
            tokens_path = argv[++a];
        else if (opt == "--vocab")
            vocab_path = argv[++a];
        else if (opt == "-n")
            n = std::atoi(argv[++a]);
        else if (opt == "-m")
            m = std::atoi(argv[++a]);
        else if (opt == "--graph")
            graph_path = argv[++a];
        else if (opt == "--out")
            out_path = argv[++a];
        else if (opt == "--csv")
            csv_path = argv[++a];
        else if (opt == "--dump")
            dump_path = argv[++a];
        else if (opt == "--fw")
            fw = argv[++a];
    }

    Times t;
    double mark = now();
    double start = mark;
    auto lap = [&]() {
        double prev = mark;
        mark = now();
        return mark - prev;
    };

    std::vector<double> D0;
    std::vector<std::string> names;  // stays empty in graph mode, so the edge file gets ids
    if (graph_path.empty()) {
        std::vector<std::string> vocab = load_vocab(vocab_path);
        if (n > (int)vocab.size()) {
            std::fprintf(stderr, "serial: n = %d is larger than the vocabulary (%d words)\n", n, (int)vocab.size());
            return 1;
        }
        std::vector<int> tokens = load_tokens(tokens_path);
        names.assign(vocab.begin(), vocab.begin() + n);
        t.load = lap();

        std::vector<double> F = count_cooc(tokens, n, m);
        t.count = lap();
        ppmi_normalise(F, n, m);
        t.ppmi = lap();
        similarity(F, n, m, D0);
        t.sim = lap();
    } else {
        n = graph_size(graph_path);
        D0 = read_graph_rows(graph_path, n, 0, n);
        m = 0;
        t.load = lap();
    }

    std::vector<double> D = D0;
    if (fw != "none")
        floyd_warshall(D, n);
    t.fw = lap();

    std::vector<Edge> edges;
    find_edges(D0.data(), D.data(), 0, n, n, edges);
    write_edges(out_path, edges, names);
    t.edges = lap();
    t.total = now() - start;

    if (!dump_path.empty())
        write_graph(dump_path, D.data(), n);
    if (!csv_path.empty())
        append_csv(csv_path, "serial", fw, n, m, 1, t, (long)edges.size());
    std::printf("serial n=%d edges=%ld t_total=%.6f\n", n, (long)edges.size(), t.total);
}

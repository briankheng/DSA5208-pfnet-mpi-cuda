#include <mpi.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "common.h"
#include "fw.h"

__global__ void sim_kernel(const double *F, double *D, int n, int m, int row0, int nloc)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int i = row0 + r;
    // the diagonal is set to an exact 0, as in the serial program: sim_distance(row, row) can be a tiny positive number
    D[(size_t)r * n + j] = i == j ? 0 : sim_distance(F + (size_t)i * m, F + (size_t)j * m, m);
}

// every rank reaches the same verdict, so rank 0 prints and all ranks finalize and return together
int reject(int rank, const std::string &why)
{
    if (rank == 0)
        std::fprintf(stderr, "pfnet_mpi: %s\n", why.c_str());
    MPI_Finalize();
    return 1;
}

// only rank 0 gets the edges; rows are split in order, so the list is already sorted by i then j
std::vector<Edge> gather_edges(const std::vector<Edge> &mine, int p)
{
    int count = mine.size();
    std::vector<int> ei(count), ej(count);
    std::vector<double> ew(count);
    for (int e = 0; e < count; e++) {
        ei[e] = mine[e].i;
        ej[e] = mine[e].j;
        ew[e] = mine[e].w;
    }

    // counts is only filled in on rank 0, so on the other ranks total is 0 and the receive arrays stay empty
    std::vector<int> counts(p), displs(p);
    MPI_Gather(&count, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    int total = 0;
    for (int r = 0; r < p; r++) {
        displs[r] = total;
        total += counts[r];
    }

    std::vector<int> all_i(total), all_j(total);
    std::vector<double> all_w(total);
    MPI_Gatherv(ei.data(), count, MPI_INT, all_i.data(), counts.data(), displs.data(), MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gatherv(ej.data(), count, MPI_INT, all_j.data(), counts.data(), displs.data(), MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gatherv(ew.data(), count, MPI_DOUBLE, all_w.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    std::vector<Edge> edges(total);
    for (int e = 0; e < total; e++)
        edges[e] = {all_i[e], all_j[e], all_w[e]};
    return edges;
}

// rank 0 gets the whole n x n matrix, the other ranks get an empty vector
std::vector<double> gather_rows(const std::vector<double> &D, int n, int p, int rank)
{
    std::vector<int> counts(p), displs(p);
    for (int r = 0; r < p; r++) {
        int r0, nl;
        split_rows(n, p, r, r0, nl);
        counts[r] = nl * n;
        displs[r] = r0 * n;
    }
    std::vector<double> all(rank == 0 ? (size_t)n * n : 0);
    MPI_Gatherv(D.data(), counts[rank], MPI_DOUBLE, all.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    return all;
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank, p;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    std::string tokens_path = "data/tokens.bin", vocab_path = "data/vocab.txt";
    std::string graph_path, out_path = "edges.txt", csv_path, dump_path, fw = "perk";
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

    if (fw != "none" && fw != "perk" && fw != "blocked")
        return reject(rank, "unknown --fw " + fw);
    if (!graph_path.empty())
        n = graph_size(graph_path);
    if (n % TILE != 0)
        return reject(rank, "n = " + std::to_string(n) + " is not a multiple of " + std::to_string(TILE));
    if (n / TILE < p)
        return reject(rank, "n = " + std::to_string(n) + " has only " + std::to_string(n / TILE) +
                                " blocks of " + std::to_string(TILE) + " rows for " + std::to_string(p) + " ranks");

    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    CUDA_CHECK(cudaSetDevice(rank % device_count));
    CUDA_CHECK(cudaFree(0));  // creates the CUDA context now, so it is not counted in t_load

    int row0, nloc;
    split_rows(n, p, rank, row0, nloc);
    size_t slice_bytes = (size_t)nloc * n * sizeof(double);

    // every phase boundary waits for the GPU and for the slowest rank, so a phase time is the slowest rank's
    auto now = []() {
        CUDA_CHECK(cudaDeviceSynchronize());
        MPI_Barrier(MPI_COMM_WORLD);
        return MPI_Wtime();
    };
    Times t;
    double mark = now();
    double start = mark;
    auto lap = [&]() {
        double prev = mark;
        mark = now();
        return mark - prev;
    };

    std::vector<double> D0;  // this rank's rows of the direct distances, kept on the host for the edge test
    double *d_D;
    std::vector<std::string> names;  // stays empty in graph mode, so the edge file gets ids
    if (graph_path.empty()) {
        std::vector<std::string> vocab = load_vocab(vocab_path);
        if (n > (int)vocab.size())
            return reject(rank, "n = " + std::to_string(n) + " is larger than the vocabulary (" +
                                    std::to_string(vocab.size()) + " words)");
        std::vector<int> tokens = load_tokens(tokens_path);
        names.assign(vocab.begin(), vocab.begin() + n);
        t.load = lap();

        std::vector<double> F = count_cooc(tokens, n, m);
        t.count = lap();
        ppmi_normalise(F, n, m);
        t.ppmi = lap();

        double *d_F;
        size_t f_bytes = (size_t)n * m * sizeof(double);
        CUDA_CHECK(cudaMalloc(&d_F, f_bytes));
        CUDA_CHECK(cudaMalloc(&d_D, slice_bytes));
        CUDA_CHECK(cudaMemcpy(d_F, F.data(), f_bytes, cudaMemcpyHostToDevice));
        // n and nloc are multiples of 32, so the grid covers the slice exactly and the kernel needs no bounds check
        dim3 block(32, 8), grid(n / 32, nloc / 8);
        sim_kernel<<<grid, block>>>(d_F, d_D, n, m, row0, nloc);
        CUDA_CHECK(cudaGetLastError());
        D0.resize((size_t)nloc * n);
        CUDA_CHECK(cudaMemcpy(D0.data(), d_D, slice_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_F));
        t.sim = lap();
    } else {
        D0 = read_graph_rows(graph_path, n, row0, nloc);
        CUDA_CHECK(cudaMalloc(&d_D, slice_bytes));
        CUDA_CHECK(cudaMemcpy(d_D, D0.data(), slice_bytes, cudaMemcpyHostToDevice));
        m = 0;
        t.load = lap();
    }

    // fw "none" leaves d_D equal to D0, which checks the similarity step and the gathers on their own
    double t_comm = 0;
    if (fw == "perk")
        fw_perk(d_D, n, row0, nloc, rank, p, t_comm);
    else if (fw == "blocked")
        fw_blocked(d_D, n, row0, nloc, rank, p, t_comm);
    t.fw = lap();

    std::vector<double> D(D0.size());
    CUDA_CHECK(cudaMemcpy(D.data(), d_D, slice_bytes, cudaMemcpyDeviceToHost));
    std::vector<Edge> mine;
    find_edges(D0.data(), D.data(), row0, nloc, n, mine);
    std::vector<Edge> edges = gather_edges(mine, p);
    if (rank == 0)
        write_edges(out_path, edges, names);
    t.edges = lap();
    t.total = MPI_Wtime() - start;

    if (!dump_path.empty()) {
        std::vector<double> all = gather_rows(D, n, p, rank);
        if (rank == 0)
            write_graph(dump_path, all.data(), n);
    }
    MPI_Reduce(&t_comm, &t.fw_comm, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        if (!csv_path.empty())
            append_csv(csv_path, "pfnet_mpi", fw, n, m, p, t, (long)edges.size());
        std::printf("pfnet_mpi %s n=%d p=%d edges=%ld t_fw=%.6f t_total=%.6f\n",
                    fw.c_str(), n, p, (long)edges.size(), t.fw, t.total);
    }

    CUDA_CHECK(cudaFree(d_D));
    MPI_Finalize();
}

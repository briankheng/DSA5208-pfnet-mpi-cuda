#pragma once

#include <cstdio>

const int TILE = 32;

// rows go out in whole 32-row blocks: rank r gets nb/p blocks, plus one if r < nb % p
inline void split_rows(int n, int p, int r, int &row0, int &nloc)
{
    int nb = n / TILE;
    int base = nb / p, extra = nb % p;
    int first = r * base + (r < extra ? r : extra);
    int blocks = base + (r < extra ? 1 : 0);
    row0 = first * TILE;
    nloc = blocks * TILE;
}

inline int owner_of_row(int row, int n, int p)
{
    int row0, nloc;
    for (int r = 0; r < p - 1; r++) {
        split_rows(n, p, r, row0, nloc);
        if (row < row0 + nloc)
            return r;
    }
    return p - 1;
}

// d_D is this rank's slice of D on the device: nloc x n, row-major, local row r is global row row0 + r
void fw_perk(double *d_D, int n, int row0, int nloc, int rank, int p, double &t_comm);
void fw_blocked(double *d_D, int n, int row0, int nloc, int rank, int p, double &t_comm);

// the .cu files include mpi.h and cuda_runtime.h before this header, so that test_split.cpp still builds with plain g++
#define CUDA_CHECK(call)                                                                  \
    do {                                                                                  \
        cudaError_t cuda_err = (call);                                                    \
        if (cuda_err != cudaSuccess) {                                                    \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,         \
                         cudaGetErrorString(cuda_err));                                   \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                 \
        }                                                                                 \
    } while (0)

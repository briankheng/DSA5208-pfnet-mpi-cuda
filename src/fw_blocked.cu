#include <mpi.h>
#include <cuda_runtime.h>

#include "fw.h"

// In phase1, phase2_row and phase2_col the values read in step k are never written in step k, because the diagonal
// tile has zeros on its diagonal and t = x + 0 is not < x. So the only sync needed is the one at the end of each step.
__global__ void phase1(double *D, int n, int lrow, int K)
{
    __shared__ double s[TILE][TILE];
    int tx = threadIdx.x, ty = threadIdx.y;
    double *x = D + (size_t)(lrow + ty) * n + K * TILE + tx;
    s[ty][tx] = *x;
    __syncthreads();
    for (int k = 0; k < TILE; k++) {
        double t = s[ty][k] + s[k][tx];
        if (t < s[ty][tx])
            s[ty][tx] = t;
        __syncthreads();
    }
    *x = s[ty][tx];
}

__global__ void phase2_row(double *D, int n, int lrow, int K)
{
    __shared__ double sd[TILE][TILE], st[TILE][TILE];
    int J = blockIdx.x;
    if (J == K)
        return;
    int tx = threadIdx.x, ty = threadIdx.y;
    size_t row = (size_t)(lrow + ty) * n;
    sd[ty][tx] = D[row + K * TILE + tx];
    st[ty][tx] = D[row + J * TILE + tx];
    __syncthreads();
    for (int k = 0; k < TILE; k++) {
        double t = sd[ty][k] + st[k][tx];
        if (t < st[ty][tx])
            st[ty][tx] = t;
        __syncthreads();
    }
    D[row + J * TILE + tx] = st[ty][tx];
}

// the diagonal tile comes from the broadcast panel, so ranks that do not own block-row K can use it too
__global__ void phase2_col(double *D, const double *panel, int n, int row0, int K)
{
    __shared__ double sd[TILE][TILE], st[TILE][TILE];
    int lI = blockIdx.y;
    if (row0 / TILE + lI == K)
        return;
    int tx = threadIdx.x, ty = threadIdx.y;
    size_t row = (size_t)(lI * TILE + ty) * n;
    st[ty][tx] = D[row + K * TILE + tx];
    sd[ty][tx] = panel[(size_t)ty * n + K * TILE + tx];
    __syncthreads();
    for (int k = 0; k < TILE; k++) {
        double t = st[ty][k] + sd[k][tx];
        if (t < st[ty][tx])
            st[ty][tx] = t;
        __syncthreads();
    }
    D[row + K * TILE + tx] = st[ty][tx];
}

// both inputs are final for this round, so unlike the phase 2 kernels the k loop needs no sync
__global__ void phase3(double *D, const double *panel, int n, int row0, int K)
{
    __shared__ double a[TILE][TILE], b[TILE][TILE];
    int J = blockIdx.x, lI = blockIdx.y;
    if (J == K || row0 / TILE + lI == K)
        return;
    int tx = threadIdx.x, ty = threadIdx.y;
    size_t row = (size_t)(lI * TILE + ty) * n;
    a[ty][tx] = D[row + K * TILE + tx];
    b[ty][tx] = panel[(size_t)ty * n + J * TILE + tx];
    __syncthreads();
    double cur = D[row + J * TILE + tx];
    for (int k = 0; k < TILE; k++) {
        double t = a[ty][k] + b[k][tx];
        if (t < cur)
            cur = t;
    }
    D[row + J * TILE + tx] = cur;
}

void fw_blocked(double *d_D, int n, int row0, int nloc, int rank, int p, double &t_comm)
{
    size_t panel_bytes = (size_t)TILE * n * sizeof(double);
    double *h_panel, *d_panel;
    CUDA_CHECK(cudaMallocHost(&h_panel, panel_bytes));
    CUDA_CHECK(cudaMalloc(&d_panel, panel_bytes));
    // n and nloc are multiples of 32, so the grids cover the slice exactly and the kernels need no bounds check
    dim3 block(TILE, TILE), col_grid(1, nloc / TILE), grid(n / TILE, nloc / TILE);

    for (int K = 0; K < n / TILE; K++) {
        int owner = owner_of_row(K * TILE, n, p);
        // block-row K starts at this local row on the owner, the other ranks do not use it
        int lrow = K * TILE - row0;
        if (rank == owner) {
            phase1<<<1, block>>>(d_D, n, lrow, K);
            CUDA_CHECK(cudaGetLastError());
            phase2_row<<<n / TILE, block>>>(d_D, n, lrow, K);
            CUDA_CHECK(cudaGetLastError());
        }

        // sync before the timer starts, so this rank's own kernels are not counted as communication. The other ranks
        // wait inside MPI_Bcast while the owner runs phase1 and phase2_row, and that wait is counted in t_comm
        CUDA_CHECK(cudaDeviceSynchronize());
        double t0 = MPI_Wtime();
        if (rank == owner) {
            CUDA_CHECK(cudaMemcpy(h_panel, d_D + (size_t)lrow * n, panel_bytes, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(d_panel, d_D + (size_t)lrow * n, panel_bytes, cudaMemcpyDeviceToDevice));
        }
        MPI_Bcast(h_panel, TILE * n, MPI_DOUBLE, owner, MPI_COMM_WORLD);
        if (rank != owner)
            CUDA_CHECK(cudaMemcpy(d_panel, h_panel, panel_bytes, cudaMemcpyHostToDevice));
        t_comm += MPI_Wtime() - t0;

        phase2_col<<<col_grid, block>>>(d_D, d_panel, n, row0, K);
        CUDA_CHECK(cudaGetLastError());
        phase3<<<grid, block>>>(d_D, d_panel, n, row0, K);
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaFreeHost(h_panel));
    CUDA_CHECK(cudaFree(d_panel));
}

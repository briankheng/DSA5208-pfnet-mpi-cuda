#include <mpi.h>
#include <cuda_runtime.h>

#include "fw.h"

__global__ void fw_perk_kernel(double *D, const double *rowk, int n, int nloc, int k)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    size_t row = (size_t)i * n;
    // D[i][k] cannot change during step k because D[k][k] = 0, so it is safe to read while the rest of the row updates
    double t = D[row + k] + rowk[j];
    if (t < D[row + j])
        D[row + j] = t;
}

void fw_perk(double *d_D, int n, int row0, int nloc, int rank, int p, double &t_comm)
{
    size_t row_bytes = n * sizeof(double);
    double *h_row, *d_row;
    CUDA_CHECK(cudaMallocHost(&h_row, row_bytes));
    CUDA_CHECK(cudaMalloc(&d_row, row_bytes));
    // n and nloc are multiples of 32, so the grid covers the slice exactly and the kernel needs no bounds check
    dim3 block(32, 8), grid(n / 32, nloc / 8);

    for (int k = 0; k < n; k++) {
        int owner = owner_of_row(k, n, p);
        // sync before the timer starts, so the previous kernel is not counted as communication
        CUDA_CHECK(cudaDeviceSynchronize());
        double t0 = MPI_Wtime();
        if (rank == owner)
            CUDA_CHECK(cudaMemcpy(h_row, d_D + (size_t)(k - row0) * n, row_bytes, cudaMemcpyDeviceToHost));
        MPI_Bcast(h_row, n, MPI_DOUBLE, owner, MPI_COMM_WORLD);
        CUDA_CHECK(cudaMemcpy(d_row, h_row, row_bytes, cudaMemcpyHostToDevice));
        t_comm += MPI_Wtime() - t0;

        fw_perk_kernel<<<grid, block>>>(d_D, d_row, n, nloc, k);
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaFreeHost(h_row));
    CUDA_CHECK(cudaFree(d_row));
}

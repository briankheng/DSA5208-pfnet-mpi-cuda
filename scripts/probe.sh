#!/bin/bash
echo "== hostname"
hostname
echo "== date"
date
echo "== nvidia-smi -L"
nvidia-smi -L
echo "== nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv
echo "== nvidia-smi topo -m"
nvidia-smi topo -m
echo "== nvcc --version"
nvcc --version
echo "== mpirun --version"
mpirun --version
echo "== ompi_info --parsable --all | grep mpi_built_with_cuda_support:value"
ompi_info --parsable --all | grep mpi_built_with_cuda_support:value
echo "== lscpu | head -20"
lscpu | head -20
echo "== free -g"
free -g
echo "== python3 -c 'import numpy, scipy; print(\"numpy\", numpy.__version__, \"scipy\", scipy.__version__)'"
python3 -c 'import numpy, scipy; print("numpy", numpy.__version__, "scipy", scipy.__version__)'
echo "== module list 2>&1"
module list 2>&1

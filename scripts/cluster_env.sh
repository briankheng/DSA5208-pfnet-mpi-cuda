# Vanda A40 nodes: 2 GPUs per node, which is also the most a GPU job can ask for
module load cuda12.4/toolkit/12.4.1
module load openmpi/gcc/64/4.1.8
export ARCH=sm_86
export NGPU=2

# Pathfinder network of words on MPI and CUDA

These programs build a Pathfinder network (PFNET) of words from the text8 corpus. The similarity s of two words is the cosine of their PPMI (positive pointwise mutual information) co-occurrence vectors, their distance is 1/s - 1, and Floyd-Warshall finds the shortest paths, so a pair of words stays linked only if no path through other words is shorter than the direct distance. `bin/serial` is the CPU version, and `bin/pfnet_mpi` is the MPI+CUDA version: it splits the rows of the distance matrix over the ranks (one GPU each) and has two Floyd-Warshall variants, `perk` and `blocked`.

## Requirements

I developed on Ubuntu 22.04 under WSL2 with an RTX 4060 laptop GPU, using g++ 11.4, nvcc 11.5, Open MPI 4.1.2 and Python 3.10 with numpy 1.24, scipy 1.14, matplotlib 3.9 and nltk 3.10.4.

```
sudo apt install build-essential nvidia-cuda-toolkit openmpi-bin libopenmpi-dev python3-pip wget unzip
python3 -m pip install --user nltk numpy scipy matplotlib
python3 -c "import nltk; nltk.download('stopwords')"
```

I call pip through `python3 -m pip` because on my WSL `pip3` belongs to Python 3.11 while `python3` is 3.10, so anything installed with `pip3` is invisible to `python3`.

On the cluster the jobs need a CUDA module, an Open MPI module (`scripts/cluster_env.sh` loads both) and Python 3 with numpy and scipy (the validation job runs `scripts/compare.py`). Open MPI doesn't have to be CUDA-aware, because the broadcasts go through host memory. The jobs don't use nltk or matplotlib.

## Data

text8 is 100 MB of cleaned English Wikipedia text: lowercase words, no punctuation, digits spelled out. Fetch it and run the preparation script from the repository root:

```
mkdir -p data
wget -q http://mattmahoney.net/dc/text8.zip -O data/text8.zip && unzip -o -q data/text8.zip -d data
python3 prep/prepare.py data/text8 data/
```

The script drops the NLTK English stopwords and the ten digit words (zero to nine), ranks the remaining words by count (ties in alphabetical order) and keeps the top 32,768. It writes `data/vocab.txt`, one word per line in rank order, and `data/tokens.bin`, one int32 per remaining token. Each int32 is the word's rank counting from 0, or -1 if the word fell outside the top 32,768. It prints

```
tokens kept: 9104966
vocabulary size: 32768
count at id 1999: 852
count at id 4095: 391
count at id 8191: 162
count at id 16383: 59
```

The last four lines are the corpus counts of the 2000th, 4096th, 8192nd and 16384th most frequent words.

I prepared the data with nltk 3.10.4, whose English stopword list has 198 words. A different list changes the vocabulary and every number after it, so if your counts differ from the ones above, look there first.

## Build

```
make
```

This builds `bin/serial` with g++ and `bin/pfnet_mpi` with nvcc (using `mpicxx` as the host compiler), and creates `bin/` and `out/`. `ARCH` is the nvcc target and defaults to `sm_86`. That's what I use locally: the RTX 4060 is `sm_89`, which nvcc 11.5 can't target, and it runs `sm_86` code fine. On the cluster, take `ARCH` from the compute capability the probe job reports (8.0 is `sm_80`, 8.6 is `sm_86`, 8.9 is `sm_89`, 9.0 is `sm_90`):

```
make ARCH=sm_80
```

The Makefile doesn't track `ARCH`, so run `make clean` before you rebuild for a different one. Keep `-ffp-contract=off` and `-fmad=false` in the Makefile. They stop the compilers from fusing multiplies and adds, and the exact match between `perk` and the serial program depends on that.

If `make` stops at `bin/pfnet_mpi` because nvcc can't use `mpicxx` as its host compiler (`bin/serial` is already built by then), build that one by hand. Compile with plain nvcc, taking the MPI include flags from `mpicxx --showme:compile`, then link with `mpicxx` and the CUDA runtime. The other flags are the Makefile's `NVFLAGS` without `-ccbin mpicxx`. Run `source scripts/cluster_env.sh` first so that `ARCH` is set. The object files land in the current directory.

```
nvcc -O2 -std=c++17 -fmad=false -arch=$ARCH -Xcompiler -ffp-contract=off $(mpicxx --showme:compile) -Isrc -c src/pfnet_mpi.cu src/fw_perk.cu src/fw_blocked.cu src/common.cpp
mpicxx pfnet_mpi.o fw_perk.o fw_blocked.o common.o -L$(dirname $(which nvcc))/../lib64 -lcudart -o bin/pfnet_mpi
```

## Running

Both programs read the prepared data and take the same options:

- `-n N`: use the N most frequent words, which are ids 0 to N-1. It is required in text mode: without it `bin/serial` runs with n = 0 and writes an empty edge file
- `-m M`: number of context words (columns) in the co-occurrence matrix, default 2000
- `--tokens F`, `--vocab F`: the prepared data, default `data/tokens.bin` and `data/vocab.txt`
- `--out F`: where to write the edge list, default `edges.txt` in the current directory
- `--csv F`: append one row of timings to F (columns below)
- `--dump F`: write the final distance matrix to F
- `--graph F`: read the distance matrix from F instead of building it from the text; the tests use this
- `--fw V`: the Floyd-Warshall variant. `bin/serial` takes `serial` (default) or `none`, and `bin/pfnet_mpi` takes `perk` (default), `blocked` or `none`. `perk` broadcasts one row per step and `blocked` broadcasts a 32-row block per round. `none` skips Floyd-Warshall, which lets the tests check the other phases on their own.

`pfnet_mpi` needs n to be a multiple of 32 and n/32 to be at least the number of ranks, and in text mode both programs need n to be at most 32768. A violation prints one line to stderr and exits with status 1. Rank r uses GPU `r % (number of GPUs)`.

One example of each, after `make`:

```
bin/serial -n 1024 --csv out/serial.csv --out out/edges_serial_1024.txt
mpirun --oversubscribe -np 2 bin/pfnet_mpi --fw blocked -n 1024 --csv out/mpi.csv --out out/edges_blocked_1024.txt
```

Each takes a few seconds and prints a summary line, for example `serial n=1024 edges=101951 t_total=1.556634`. The two edge files come out identical. An edge file has one line per edge, `word_i word_j distance`, sorted by the words' ranks (with `--graph` the words are replaced by node numbers):

```
also many 1.9080169350692184
also used 1.5024444045301597
also see 1.314770178872442
```

Locally I run MPI with `mpirun --oversubscribe -np P`. All ranks then share the one GPU, so local timings say nothing about scaling. WSL2's GPU driver spills device memory into host RAM instead of failing, and my WSL has about 7 GB of RAM, so I keep local text8 runs at n <= 4096.

The network keeps 16.9% to 28.9% of the pairs of words over the sizes I ran: 101,951 edges at n = 1024 (3.4 MB), 449,304 at n = 2048 (15 MB) and 2,212,510 at n = 4096 (76 MB, 26% of the 8,386,560 pairs). The cluster runs gave 9,691,205 edges at n = 8192 (339 MB) and 22,618,439 at n = 16384 (799 MB).

The graph files for `--graph` and `--dump` hold an int32 n followed by n*n float64 values in row-major order, with inf for "no link".

## Validation

```
make test
python3 prep/test_prepare.py
```

`make test` builds and runs two unit test programs, which print `common ok` and `split ok`. The second command checks the two helper functions in `prep/prepare.py` and prints `ok`.

The main check is the validation script. It needs the binaries from `make` and the prepared text8 data:

```
bash tests/validate_local.sh
```

It deletes `out/v` and runs four sections. That took about 5 minutes in my project folder under `/mnt/c` (a clone on WSL's own disk is faster). Every comparison, done by `scripts/compare.py`, prints a PASS or FAIL line, and the script stops at the first failure. Each section prints `PASS serial`, `PASS none`, `PASS perk` or `PASS blocked` when it finishes, so a good run ends with `PASS blocked`.

What the sections check:

- `serial` compares `bin/serial` with scipy's `floyd_warshall` on eight small graphs made by `tests/make_graphs.py` (hand-made triangles, random graphs and two sparse random graphs with about 10% of the pairs linked). It also tests each result's properties (zero diagonal, no distance longer than the direct one, symmetry, the triangle inequality) and checks that an n larger than the vocabulary is refused with one line and exit status 1. The triangle inequality is checked for every k when n is at most 100 (six of the eight graphs) and on 10,000 sampled triples above that. The text8 results for n = 1024, 2048 and 4096 are the references for the later sections.
- `none` runs `pfnet_mpi --fw none` on 1, 2 and 4 ranks where n/32 allows it (`shortcut32` runs on 1 rank and `rand96` on 1 and 2) and requires exactly the matrix and edges of `bin/serial --fw none`. That tests the distance kernel, the row split and the gathers without Floyd-Warshall. It also checks that three bad argument combinations are refused with one line and exit status 1.
- `perk` and `blocked` run the variants on 1, 2 and 4 ranks (fewer on the graphs that are too small), on the eight small graphs and on text8 at n = 1024, 2048 and 4096, and compare with the serial results. `perk` has to match exactly, with a largest difference of 0 and an identical edge file. `blocked` has to match to within 1e-12 with an identical edge file, and it is also compared with `perk` at n = 2048. On the two sparse graphs `blocked` does differ from the serial result, by about 4e-16 to 9e-16. Their weights are scaled by pi, which makes the path sums go above 1, where they round, so the order of the additions changes the result. `perk` is still exact there. On text8 `blocked` differs too, by about 1.8e-15, because the shortest paths there can take several hops and the order of the additions changes the rounding. That stays well within the 1e-12 tolerance.

You can run the sections separately, for example `bash tests/validate_local.sh serial`. With no argument the script deletes `out/v` first; with an argument it leaves `out/v` alone. Only `serial` works on an empty `out/v`. `none` and `perk` read the serial results it leaves there, and `blocked` also reads the `perk` results at n = 2048. The sections run in the order `serial`, `none`, `perk`, `blocked`, so run them one at a time in that order or name several in one quoted argument:

```
bash tests/validate_local.sh serial
bash tests/validate_local.sh "perk blocked"
```

## Cluster jobs

The job scripts are set up for NUS Vanda (PBS Pro). They go to the `auto_free` routing queue, which sends GPU jobs to the free `gpu` queue (nodes with 2x A40, so p goes up to 2); submitting to `gpu` directly is denied. On another cluster, change `scripts/cluster_env.sh` and the `#PBS` lines (queue, `select`, walltime). They start with `cd "$PBS_O_WORKDIR"` and use paths relative to the repository root, so submit them from there. The scripts as committed go to `auto_free`, but I ran the measurement jobs (validate, strong, weak) with `qsub -P <project> -q auto ...` because the free GPU pool was full, and that route gives each job a whole node. Home has a 40 GB quota, so the jobs run from `/scratch/USER/pfnet`. Copy the repository there without `bin/`, `out/` and `data/text8`. The jobs do need `data/vocab.txt` and `data/tokens.bin`:

```
rsync -av --exclude .git --exclude bin --exclude out --exclude old-project --exclude 'data/text8*' ./ USER@HOST:/scratch/USER/pfnet/
```

Then, on the cluster, in this order:

1. Install the Python packages once with `python3 -m pip install --user numpy==1.26.4 scipy==1.13.1` (the system `python3` is 3.9 and needs no module). On another cluster, check that the queue in `scripts/probe.pbs` and the module names in `scripts/cluster_env.sh` exist (`qstat -Q` lists the queues, `module avail` the modules). Then run `qsub scripts/probe.pbs`. It runs `scripts/probe.sh` on one GPU and writes `results/env.txt`: the hostname and date, the GPU model, memory and compute capability, the GPU topology, the nvcc and Open MPI versions, whether Open MPI was built with CUDA support, the CPU and memory, the numpy and scipy versions and the loaded modules.
2. Check `env.txt` against the scripts. On Vanda the values are already set: `cluster_env.sh` loads `cuda12.4/toolkit/12.4.1` and `openmpi/gcc/64/4.1.8`, with `ARCH=sm_86` (the A40 is compute capability 8.6) and `NGPU=2`. On another cluster, set the `module load` lines, `ARCH` and `NGPU` (the number of GPUs you can use on a node) there, and in every `scripts/*.pbs` set the queue (`-q`), the `select` line (memory included) and the `walltime`. In `run_validate.pbs`, `run_strong.pbs` and `run_weak.pbs`, `ngpus` in `select` should equal NGPU, one GPU per rank; the scripts ask for `ncpus=4`. The `#PBS` lines can't read NGPU, so you edit those by hand.
3. Build on the login node with `source scripts/cluster_env.sh && make clean && make ARCH=$ARCH`. The clean matters after you sync new sources: rsync keeps the files' timestamps, so a plain `make` can miss changed sources and leave a stale binary. The jobs don't build anything.
4. `qsub scripts/run_validate.pbs` runs `tests/validate_local.sh`, then repeats the check at n = 8192 with NGPU ranks. It writes `results/validate.txt`, where every `compare.py` line should start with PASS, and leaves the matrices and edge lists in `out/v/`. Look at it before you submit the measurement jobs.
5. `qsub scripts/run_serial.pbs` times `bin/serial` on one core at n = 2048, 4096 and 8192, three runs each. It writes `results/serial.csv` and `out/edges_serial_<n>.txt`.
6. `qsub scripts/run_strong.pbs` times `perk` and `blocked` at n = 16384, 8192 and 4096 (largest first, so a walltime overrun loses the cheap runs) with p = 1, 2, 4 and 8 ranks (up to NGPU), three runs each. It writes `results/strong.csv` and `out/edges_<variant>_<n>.txt`, which every repeat overwrites.
7. `qsub scripts/run_weak.pbs` times both variants with n growing with p so that n^3/p stays about constant: (n, p) = (8192, 1), (10336, 2), (12992, 4) and (16384, 8), up to NGPU, three runs each. It writes `results/weak.csv` and `out/edges_weak.txt`, which every run overwrites.

The CSVs are appended to, so delete a CSV before you resubmit the job that writes it. A complete set of runs leaves 10 lines in `serial.csv`, 73 in `strong.csv` and 25 in `weak.csv`; the last two are shorter when NGPU is below 8, because p stops at NGPU (with NGPU = 2 on Vanda they come to 37 and 13 lines).

Check your disk quota before you submit. Together the jobs left 7.5 GB under `out/` on my run, 3.1 GB of it edge files and most of the rest validation dumps. The largest single file is the edge list at n = 16384 (799 MB); `out/edges_weak.txt` is 487 MB. `run_strong.pbs` and `run_weak.pbs` ask for more host memory than the other scripts, because rank 0 keeps host copies of the matrices and collects every edge, several GB at n = 16384.

## Figures

Copy the results back from the cluster. `scripts/analyze.py` reads every `out/edges_perk_<n>.txt` it finds, so delete any local ones first:

```
rm -f out/edges_perk_*.txt
rsync -av USER@HOST:/scratch/USER/pfnet/results/ results/
rsync -av 'USER@HOST:/scratch/USER/pfnet/out/edges_perk_*.txt' out/
python3 scripts/plot.py results report
python3 scripts/analyze.py data/vocab.txt out/ report/
```

`plot.py` takes the median over the repeats of each configuration. It writes `strong_speedup.pdf`, `strong_efficiency.pdf`, `weak_efficiency.pdf` and `breakdown.pdf` to `report/figures/` and `serial_vs_gpu.tex` to `report/tables/`. `analyze.py` prints the edge count and average degree of each edge file and writes `degree.pdf` (the degree distribution at the largest n) to `report/figures/`, plus `hubs.tex` (the 15 words with the highest degree) and `neighbours.tex` (the 10 closest neighbours of the example words in the `WORDS` list at the top of the script) to `report/tables/`.

To try both scripts before there are cluster results, `bash scripts/run_local_smoke.sh` writes small CSVs and edge files to `out/smoke/` in about 20 seconds:

```
python3 scripts/plot.py out/smoke out/plot_test
python3 scripts/analyze.py data/vocab.txt out/smoke out/plot_test
```

The smoke runs put all ranks on one GPU, so the plots are only good for checking that the scripts run.

## CSV columns

`--csv F` appends one row per run to F, and writes the header first if F doesn't exist yet:

```
program,variant,n,m,p,t_load,t_count,t_ppmi,t_sim,t_fw,t_fw_comm,t_edges,t_total,edges
```

`program` is `serial` or `pfnet_mpi`, `variant` is the `--fw` value, `n` and `m` are the number of words and of context words (`m` is 0 with `--graph`), `p` is the number of ranks (1 for serial) and `edges` is the number of lines in the edge file. The other columns are seconds:

- `t_load`: reading `tokens.bin` and `vocab.txt` (with `--graph`, loading the graph)
- `t_count`: the co-occurrence counts
- `t_ppmi`: PPMI weighting and row normalisation
- `t_sim`: the distance matrix (on the GPU in `pfnet_mpi`, including the copies to and from the device)
- `t_fw`: Floyd-Warshall
- `t_fw_comm`: the part of `t_fw` spent moving the row or block to every rank (the copies to and from the host and the broadcast, including the wait for the owner); the largest value over the ranks, and 0 for serial
- `t_edges`: copying the result back to the host (in `pfnet_mpi`), applying the edge rule, gathering the edges on rank 0 and writing the edge file
- `t_total`: the run from the start of `t_load` to the end of `t_edges`, without MPI and CUDA startup and without the `--dump` file

In `pfnet_mpi` every phase boundary waits for the GPU and for all ranks, so a phase time is the slowest rank's.

## Files

```
Makefile                    build rules: all, test, clean
prep/prepare.py             text8 -> data/vocab.txt and data/tokens.bin
prep/test_prepare.py        tests for the helper functions in prepare.py
src/common.h                types, constants, the distance function and the declarations shared by both programs
src/common.cpp              loading, co-occurrence counts, PPMI, graph files, edge rule, edge and CSV output
src/serial.cpp              the CPU program
src/pfnet_mpi.cu            the MPI+CUDA program: row split, distance kernel, gathers, main
src/fw.h                    row split helpers and the declarations of the two variants
src/fw_perk.cu              perk: one row broadcast and one kernel launch per step
src/fw_blocked.cu           blocked: 32x32 tiles, one block-row broadcast per round
tests/test_common.cpp       unit tests for src/common.cpp
tests/test_split.cpp        unit tests for the row split in src/fw.h
tests/make_graphs.py        writes the small test graphs to out/graphs
tests/validate_local.sh     the validation script
scripts/compare.py          checks results against scipy, against each other, and for basic properties
scripts/probe.pbs           cluster probe job, runs probe.sh
scripts/probe.sh            prints the GPU, compiler, MPI and CPU details that go to results/env.txt
scripts/cluster_env.sh      modules, ARCH and NGPU for the job scripts
scripts/run_validate.pbs    validation job
scripts/run_serial.pbs      serial timing job
scripts/run_strong.pbs      strong scaling job
scripts/run_weak.pbs        weak scaling job
scripts/run_local_smoke.sh  small local run for trying plot.py and analyze.py
scripts/plot.py             scaling figures and the serial-versus-GPU table from the CSVs
scripts/analyze.py          degree distribution, hubs and neighbours from the edge files
results/                    written by the cluster jobs: env.txt, validate.txt and the three CSVs
report/                     main.pdf is the report; plot.py and analyze.py write figures/ and tables/ there
bin/                        binaries built by make (git-ignored)
out/                        run outputs: test graphs, validation files, edge lists (git-ignored)
data/                       text8 and the prepared vocab.txt and tokens.bin (git-ignored)
```

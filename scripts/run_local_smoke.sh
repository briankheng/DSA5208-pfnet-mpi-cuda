#!/bin/bash
set -e
cd "$(dirname "$0")/.."
mkdir -p out/smoke
# rows are appended to the csv files, so the line counts only hold on a fresh directory
rm -f out/smoke/*.csv

for n in 256 512; do
    bin/serial -n $n --csv out/smoke/serial.csv --out out/smoke/edges_serial_$n.txt
done

for n in 512 1024; do
    for v in perk blocked; do
        for p in 1 2; do
            mpirun --oversubscribe -np $p bin/pfnet_mpi --fw $v -n $n --csv out/smoke/strong.csv --out out/smoke/edges_${v}_$n.txt
        done
    done
done

for pair in "512 1" "640 2"; do
    read n p <<< "$pair"
    for v in perk blocked; do
        mpirun --oversubscribe -np $p bin/pfnet_mpi --fw $v -n $n --csv out/smoke/weak.csv --out out/smoke/edges_weak.txt
    done
done

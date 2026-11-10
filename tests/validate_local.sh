#!/bin/bash
set -e
cd "$(dirname "$0")/.."
sections=${1:-"serial none perk blocked"}
want() { [[ " $sections " == *" $1 "* ]]; }
[ -n "$1" ] || rm -rf out/v
mkdir -p out/v
python3 tests/make_graphs.py

if want serial; then
    for g in shortcut32 shortcut64 tie32 rand32 rand96 rand160 sparse96 sparse160; do
        bin/serial --graph out/graphs/$g.bin --out out/v/$g.serial.edges --dump out/v/$g.serial.bin
        python3 scripts/compare.py oracle out/graphs/$g.bin out/v/$g.serial.bin out/v/$g.serial.edges
        python3 scripts/compare.py props out/v/$g.serial.bin --graph out/graphs/$g.bin
    done
    python3 scripts/compare.py oracle out/graphs/shortcut32.bin out/v/shortcut32.serial.bin out/v/shortcut32.serial.edges --expect 0-1,1-2
    python3 scripts/compare.py oracle out/graphs/tie32.bin out/v/tie32.serial.bin out/v/tie32.serial.edges --expect 0-1,0-2,1-2
    # text8 references for the GPU sections
    for n in 1024 2048 4096; do
        bin/serial -n $n --out out/v/t$n.serial.edges --dump out/v/t$n.serial.bin
    done
    python3 scripts/compare.py props out/v/t1024.serial.bin
    rc=0
    bin/serial -n 40000 2> out/v/err.txt || rc=$?
    if [ "$rc" != 1 ] || [ "$(wc -l < out/v/err.txt)" != 1 ]; then echo "FAIL n > vocab not rejected with one line and status 1"; exit 1; fi
    echo "PASS serial"
fi

if want none; then
    for g in shortcut32 rand96 rand160; do
        n=${g//[!0-9]/}
        bin/serial --fw none --graph out/graphs/$g.bin --out out/v/$g.none.s.edges --dump out/v/$g.none.s.bin
        for P in 1 2 4; do
            [ $((n / 32)) -ge $P ] || continue
            mpirun --oversubscribe -np $P bin/pfnet_mpi --fw none --graph out/graphs/$g.bin --out out/v/$g.none.$P.edges --dump out/v/$g.none.$P.bin
            python3 scripts/compare.py diff out/v/$g.none.s.bin out/v/$g.none.$P.bin out/v/$g.none.s.edges out/v/$g.none.$P.edges --tol 0
        done
    done
    bin/serial --fw none -n 1024 --out out/v/t1024.none.s.edges --dump out/v/t1024.none.s.bin
    python3 scripts/compare.py props out/v/t1024.serial.bin --graph out/v/t1024.none.s.bin
    for P in 1 2 4; do
        mpirun --oversubscribe -np $P bin/pfnet_mpi --fw none -n 1024 --out out/v/t1024.none.$P.edges --dump out/v/t1024.none.$P.bin
        python3 scripts/compare.py diff out/v/t1024.none.s.bin out/v/t1024.none.$P.bin out/v/t1024.none.s.edges out/v/t1024.none.$P.edges --tol 0
    done
    # bad arguments must fail cleanly: status 1 and one line of the program's own message (mpirun adds a banner of its own)
    for args in "-np 1 bin/pfnet_mpi --fw none -n 1000" "-np 4 bin/pfnet_mpi --fw none --graph out/graphs/rand96.bin" "-np 1 bin/pfnet_mpi --fw none -n 40000"; do
        rc=0
        mpirun --oversubscribe $args > out/v/err.txt 2>&1 || rc=$?
        if [ "$rc" != 1 ] || [ "$(grep -c '^pfnet_mpi:' out/v/err.txt)" != 1 ] || grep -q MPI_ABORT out/v/err.txt; then
            echo "FAIL not rejected with one line and status 1: $args"; exit 1
        fi
    done
    echo "PASS none"
fi

check_variant() {
    v=$1; tol=$2
    for g in shortcut32 shortcut64 tie32 rand32 rand96 rand160 sparse96 sparse160; do
        n=${g//[!0-9]/}
        for P in 1 2 4; do
            [ $((n / 32)) -ge $P ] || continue
            mpirun --oversubscribe -np $P bin/pfnet_mpi --fw $v --graph out/graphs/$g.bin --out out/v/$g.$v.$P.edges --dump out/v/$g.$v.$P.bin
            python3 scripts/compare.py diff out/v/$g.serial.bin out/v/$g.$v.$P.bin out/v/$g.serial.edges out/v/$g.$v.$P.edges --tol $tol
        done
    done
    for n in 1024 2048 4096; do
        for P in 1 2 4; do
            mpirun --oversubscribe -np $P bin/pfnet_mpi --fw $v -n $n --out out/v/t$n.$v.$P.edges --dump out/v/t$n.$v.$P.bin
            python3 scripts/compare.py diff out/v/t$n.serial.bin out/v/t$n.$v.$P.bin out/v/t$n.serial.edges out/v/t$n.$v.$P.edges --tol $tol
        done
    done
    python3 scripts/compare.py props out/v/t4096.$v.4.bin
}

if want perk; then
    check_variant perk 0
    echo "PASS perk"
fi

if want blocked; then
    check_variant blocked 1e-12
    # the two variants agree with each other as well
    python3 scripts/compare.py diff out/v/t2048.perk.2.bin out/v/t2048.blocked.2.bin out/v/t2048.perk.2.edges out/v/t2048.blocked.2.edges --tol 1e-12
    echo "PASS blocked"
fi

ARCH ?= sm_86
CXXFLAGS = -O2 -std=c++17 -ffp-contract=off
NVFLAGS = -O2 -std=c++17 -fmad=false -arch=$(ARCH) -ccbin mpicxx -Xcompiler -ffp-contract=off

all: bin/serial bin/pfnet_mpi

bin/test_common: tests/test_common.cpp src/common.cpp src/common.h
	@mkdir -p bin out
	g++ $(CXXFLAGS) -Isrc tests/test_common.cpp src/common.cpp -o $@

bin/test_split: tests/test_split.cpp src/fw.h
	@mkdir -p bin out
	g++ $(CXXFLAGS) -Isrc tests/test_split.cpp -o $@

bin/serial: src/serial.cpp src/common.cpp src/common.h
	@mkdir -p bin out
	g++ $(CXXFLAGS) src/serial.cpp src/common.cpp -o $@

bin/pfnet_mpi: src/pfnet_mpi.cu src/fw_perk.cu src/fw_blocked.cu src/common.cpp src/common.h src/fw.h
	@mkdir -p bin out
	nvcc $(NVFLAGS) -Isrc src/pfnet_mpi.cu src/fw_perk.cu src/fw_blocked.cu src/common.cpp -o $@

test: bin/test_common bin/test_split
	bin/test_common
	bin/test_split

clean:
	rm -rf bin

.PHONY: all test clean

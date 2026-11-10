#include <cassert>
#include <cstdio>

#include "fw.h"

int main()
{
    int row0, nloc;
    split_rows(96, 2, 0, row0, nloc);    assert(row0 == 0 && nloc == 64);
    split_rows(96, 2, 1, row0, nloc);    assert(row0 == 64 && nloc == 32);
    split_rows(160, 4, 0, row0, nloc);   assert(row0 == 0 && nloc == 64);
    split_rows(160, 4, 3, row0, nloc);   assert(row0 == 128 && nloc == 32);
    split_rows(12992, 4, 2, row0, nloc); assert(row0 == 6528 && nloc == 3232);
    assert(owner_of_row(63, 96, 2) == 0 && owner_of_row(64, 96, 2) == 1);
    assert(owner_of_row(64, 160, 4) == 1 && owner_of_row(159, 160, 4) == 3);
    assert(owner_of_row(6527, 12992, 4) == 1 && owner_of_row(6528, 12992, 4) == 2);
    // ranges are contiguous, cover all rows, and agree with owner_of_row
    for (int p = 1; p <= 8; p++) {
        int next = 0;
        for (int r = 0; r < p; r++) {
            split_rows(4096, p, r, row0, nloc);
            assert(row0 == next && nloc % 32 == 0);
            for (int i = row0; i < row0 + nloc; i++)
                assert(owner_of_row(i, 4096, p) == r);
            next = row0 + nloc;
        }
        assert(next == 4096);
    }

    std::printf("split ok\n");
}

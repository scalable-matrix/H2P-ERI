#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>

#include "H2ERI.h"

int main(int argc, char **argv)
{
    simint_init();
    
    H2ERI_t h2eri;
    H2ERI_init(&h2eri, 1e-10, 1e-10, 1e-6);
    
    // 1. Read molecular file
    CMS_read_mol_file(argv[1], &h2eri->nshell, &h2eri->shells);
    
    // 2. Process input shells for H2 partitioning
    H2ERI_process_shells(h2eri);
    
    // 3. H2 partition of uncontracted shell pair centers
    H2ERI_partition(h2eri);
    
    int num_target = 8, num_sp = 10, num_pair_idx;
    int target_rows[8] = {0, 1, 8, 29, 36, 59, 60, 72};
    int am1[10] = {0, 0, 1, 2, 1, 1, 1, 2, 2, 0};
    int am2[10] = {0, 1, 1, 0, 1, 1, 2, 0, 1, 2};
    int pair_idx[10], row_idx[8];
    int workbuf[10 * 5 + 8 + 2];
    H2ERI_extract_shell_pair_idx(
        num_target, target_rows, num_sp, am1, am2,
        workbuf, &num_pair_idx, pair_idx, row_idx
    );
    for (int i = 0; i < num_pair_idx; i++) printf("%d, ", pair_idx[i] + 1);
    printf("\n");
    for (int i = 0; i < num_target; i++) printf("%d, ", row_idx[i] + 1);
    printf("\n");
    // Output should be:
    // 1, 2, 3, 6, 8, 9
    // 1, 2, 9, 15, 22, 27, 28, 40
    
    simint_finalize();
    return 0;
}
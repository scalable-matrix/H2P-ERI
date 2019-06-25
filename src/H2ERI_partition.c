#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

#include "CMS.h"
#include "H2ERI_typedef.h"
#include "H2Pack_partition.h"

// Partition uncontracted shell pair centers (as points) for H2 tree
void H2ERI_partition_unc_sp_centers(H2ERI_t h2eri, int max_leaf_points, double max_leaf_size)
{
    // 1. Partition uncontracted shell pair centers
    int num_unc_sp = h2eri->num_unc_sp;
    double *unc_sp_center = h2eri->unc_sp_center;
    if (max_leaf_points <= 0)   max_leaf_points = 300;
    if (max_leaf_size   <= 0.0) max_leaf_size   = 2.0;
    H2P_partition_points(
        h2eri->h2pack, num_unc_sp, unc_sp_center, 
        max_leaf_points, max_leaf_size
    );
    
    // 2. Permute the uncontracted shell pairs and their extents according to 
    // the permutation of their center coordinate
    int *coord_idx = h2eri->h2pack->coord_idx;
    shell_t *unc_sp = h2eri->unc_sp;
    shell_t *unc_sp_new = (shell_t *) malloc(sizeof(shell_t) * num_unc_sp * 2);
    double *unc_sp_extent = h2eri->unc_sp_extent;
    double *unc_sp_extent_new = (double *) malloc(sizeof(double) * num_unc_sp);
    assert(unc_sp_new != NULL && unc_sp_extent_new != NULL);
    for (int i = 0; i < num_unc_sp; i++)
    {
        int cidx_i = coord_idx[i];
        int i20 = i * 2, i21 = i * 2 + 1;
        int cidx_i20 = cidx_i * 2, cidx_i21 = cidx_i * 2 + 1;
        unc_sp_extent_new[i] = unc_sp_extent[cidx_i];
        simint_initialize_shell(&unc_sp_new[i20]);
        simint_initialize_shell(&unc_sp_new[i21]);
        simint_allocate_shell(1, &unc_sp_new[i20]);
        simint_allocate_shell(1, &unc_sp_new[i21]);
        simint_copy_shell(&unc_sp[cidx_i20], &unc_sp_new[i20]);
        simint_copy_shell(&unc_sp[cidx_i21], &unc_sp_new[i21]);
    }
    CMS_destroy_shells(num_unc_sp * 2, h2eri->unc_sp);
    free(h2eri->unc_sp);
    free(h2eri->unc_sp_extent);
    h2eri->unc_sp = unc_sp_new;
    h2eri->unc_sp_extent = unc_sp_extent_new;
}


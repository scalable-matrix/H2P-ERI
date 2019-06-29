#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

#include "CMS.h"
#include "H2ERI_typedef.h"
#include "H2Pack_partition.h"

// Partition uncontracted shell pair centers (as points) for H2 tree
// Input parameters:
//   h2eri->num_unc_sp    : Number of fully uncontracted shell pairs (FUSP)
//   h2eri->unc_sp_center : Array, size 3 * num_unc_sp, centers of FUSP
//   h2eri->unc_sp_extent : Array, size num_unc_sp, extents of FUSP
//   max_leaf_points      : Maximum number of point in a leaf node's box. If <= 0, 
//                          will use 300.
//   max_leaf_size        : Maximum size of a leaf node's box. 
// Output parameter:
//   h2eri->h2pack        : H2Pack structure with point partitioning info
//   h2eri->unc_sp_center : Array, size 3 * num_unc_sp, sorted centers of FUSP
//   h2eri->unc_sp_extent : Array, size num_unc_sp, sorted extents of FUSP
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
    memcpy(unc_sp_center, h2eri->h2pack->coord, sizeof(double) * 3 * num_unc_sp);
    
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

// Calculate the basis function indices information of shells and shell pairs 
// Input parameters:
//   h2eri->nshell     : Number of original shells 
//   h2eri->shells     : Array, size nshell, original shells
//   h2eri->num_unc_sp : Number of shell pairs
//   h2eri->unc_sp     : Array, size num_sp * 2, each row is a shell pair
// Output parameters:
//   h2eri->shell_bf_sidx  : Array, size nshell+1, indices of each shell's first basis function
//   h2eri->unc_sp_bf_sidx : Array, size num_unc_sp+1, indices of each FUSP first basis function 
void H2ERI_calc_bf_sidx(H2ERI_t h2eri)
{
    int nshell      = h2eri->nshell;
    int num_unc_sp  = h2eri->num_unc_sp;
    shell_t *shells = h2eri->shells;
    shell_t *unc_sp = h2eri->unc_sp;
    
    h2eri->shell_bf_sidx  = (int *) malloc(sizeof(int) * (nshell + 1));
    h2eri->unc_sp_bf_sidx = (int *) malloc(sizeof(int) * (num_unc_sp + 1));
    assert(h2eri->shell_bf_sidx != NULL && h2eri->unc_sp_bf_sidx != NULL);
    
    h2eri->shell_bf_sidx[0] = 0;
    for (int i = 0; i < nshell; i++)
    {
        int am_i  = shells[i].am;
        int nbf_i = NCART(am_i);
        h2eri->max_am = MAX(h2eri->max_am, am_i);
        h2eri->shell_bf_sidx[i + 1] = h2eri->shell_bf_sidx[i] + nbf_i;
    }
    
    h2eri->unc_sp_bf_sidx[0] = 0;
    for (int i = 0; i < num_unc_sp; i++)
    {
        int nbf_i20 = NCART(unc_sp[i * 2].am);
        int nbf_i21 = NCART(unc_sp[i * 2 + 1].am);
        int nbf_i   = nbf_i20 * nbf_i21;
        h2eri->unc_sp_bf_sidx[i + 1] = h2eri->unc_sp_bf_sidx[i] + nbf_i;
    }
}

// Calculate the max extent of shell pairs in each H2 box
// Input parameters:
//   h2eri->h2pack        : H2 tree partitioning info
//   h2eri->num_unc_sp    : Number of fully uncontracted shell pairs (FUSP)
//   h2eri->unc_sp_center : Array, size 3 * num_unc_sp, centers of FUSP, sorted
//   h2eri->unc_sp_extent : Array, size num_unc_sp, extents of FUSP, sorted
// Output parameter:
//   h2eri->box_extent : Array, size h2pack->n_node, extent of each H2 node box
void H2ERI_calc_box_extent(H2ERI_t h2eri)
{
    H2Pack_t h2pack = h2eri->h2pack;
    int    n_node         = h2pack->n_node;
    int    max_level      = h2pack->max_level;
    int    max_child      = h2pack->max_child;
    int    n_leaf_node    = h2pack->n_leaf_node;
    int    *cluster       = h2pack->cluster;
    int    *children      = h2pack->children;
    int    *n_child       = h2pack->n_child;
    int    *level_nodes   = h2pack->level_nodes;
    int    *level_n_node  = h2pack->level_n_node;
    double *enbox         = h2pack->enbox;
    int    num_unc_sp     = h2eri->num_unc_sp;
    double *unc_sp_center = h2eri->unc_sp_center;
    double *unc_sp_extent = h2eri->unc_sp_extent;
    
    h2eri->box_extent = (double *) malloc(sizeof(double) * n_node);
    assert(h2eri->box_extent != NULL);
    double *box_extent = h2eri->box_extent;
    
    for (int i = max_level; i >= 1; i--)
    {
        int *level_i_nodes = level_nodes + i * n_leaf_node;
        int level_i_n_node = level_n_node[i];
        for (int j = 0; j < level_i_n_node; j++)
        {
            int node = level_i_nodes[j];
            double *node_enbox = enbox + 6 * node;
            double enbox_center[3];
            enbox_center[0] = node_enbox[0] + 0.5 * node_enbox[3];
            enbox_center[1] = node_enbox[1] + 0.5 * node_enbox[4];
            enbox_center[2] = node_enbox[2] + 0.5 * node_enbox[5];
            
            int n_child_node = n_child[node];
            if (n_child_node == 0)
            {
                int s_index = cluster[2 * node];
                int e_index = cluster[2 * node + 1];
                int n_point = e_index - s_index + 1;
                double box_extent_node = 0.0;
                for (int d = 0; d < 3; d++)
                {
                    double *center_d = unc_sp_center + d * num_unc_sp;
                    double enbox_width_d = node_enbox[3 + d];
                    for (int k = s_index; k <= e_index; k++)
                    {
                        double tmp_extent_d_k;
                        // Distance of shell pair center to the upper limit along each dimension
                        tmp_extent_d_k  = fabs(center_d[k] - enbox_center[d]);
                        tmp_extent_d_k  = 0.5 * enbox_width_d - tmp_extent_d_k;
                        // Outreach of each extent box
                        tmp_extent_d_k  = unc_sp_extent[k] - tmp_extent_d_k;
                        // Ratio of extent box outreach over enclosing box size
                        tmp_extent_d_k /= enbox_width_d;
                        // Negative means this dimension of extent box is inside the
                        // enclosing box, make it 0.1 to make sure the box_extent >= 1
                        tmp_extent_d_k  = MAX(tmp_extent_d_k, 0.1);
                        box_extent_node = MAX(box_extent_node, tmp_extent_d_k);
                    }
                }
                box_extent[node] = ceil(box_extent_node);
            } else {
                // Since the out-reach width is the same, the extent of this box 
                // (outreach / box width) is just half of the largest sub-box extent.
                double box_extent_node = 0.0;
                int *child_nodes = children + node * max_child;
                for (int k = 0; k < n_child_node; k++)
                {
                    int child_k = child_nodes[k];
                    box_extent_node = MAX(box_extent_node, box_extent[child_k]);
                }
                box_extent[node] = ceil(0.5 * box_extent_node);
            }  // End of "if (n_child_node == 0)"
        }  // End of j loop
    }  // End of i loop
}

// Calculate the matvec cluster for H2 nodes
// Input parameters:
//   h2eri->h2pack         : H2 tree partitioning info
//   h2eri->unc_sp_bf_sidx : Array, size num_unc_sp+1, indices of each FUSP first basis function 
// Output parameter:
//   h2eri->h2pack->mat_cluster : Array, size h2pack->n_node * 2, matvec cluster for H2 nodes
void H2ERI_calc_mat_cluster(H2ERI_t h2eri)
{
    H2Pack_t h2pack = h2eri->h2pack;
    int n_node    = h2pack->n_node;
    int max_child = h2pack->max_child;
    int *cluster  = h2pack->cluster;
    int *children = h2pack->children;
    int *n_child  = h2pack->n_child;
    int *mat_cluster = h2pack->mat_cluster;
    int *unc_sp_bf_sidx = h2eri->unc_sp_bf_sidx;
    
    int offset = 0;
    for (int i = 0; i < n_node; i++)
    {
        int i20 = i * 2;
        int i21 = i * 2 + 1;
        int n_child_i = n_child[i];
        if (n_child_i == 0)
        {
            int s_index = cluster[2 * i];
            int e_index = cluster[2 * i + 1];
            int node_nbf = unc_sp_bf_sidx[e_index + 1] - unc_sp_bf_sidx[s_index];
            mat_cluster[i20] = offset;
            mat_cluster[i21] = offset + node_nbf - 1;
            offset += node_nbf;
        } else {
            int *i_childs = children + i * max_child;
            int child_0 = i_childs[0];
            int child_n = i_childs[n_child_i - 1];
            mat_cluster[i20] = mat_cluster[2 * child_0];
            mat_cluster[i21] = mat_cluster[2 * child_n + 1];
        }
    }
}

// H2 partition of uncontracted shell pair centers
void H2ERI_partition(H2ERI_t h2eri)
{
    H2ERI_partition_unc_sp_centers(h2eri, 0, 0.0);
    H2ERI_calc_bf_sidx(h2eri);
    H2ERI_calc_box_extent(h2eri);
    H2ERI_calc_mat_cluster(h2eri);
    
    int n_thread = h2eri->h2pack->n_thread;
    h2eri->simint_buffs = (simint_buff_t *) malloc(sizeof(simint_buff_t) * n_thread);
    assert(h2eri->simint_buffs != NULL);
    for (int i = 0; i < n_thread; i++)
        CMS_init_Simint_buff(h2eri->max_am, &h2eri->simint_buffs[i]);
}

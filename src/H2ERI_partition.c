#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

#include "CMS.h"
#include "H2ERI_typedef.h"
#include "H2Pack_partition.h"
#include "H2Pack_utils.h"
#include "H2ERI_build_exchange.h"

// Partition screened shell pair centers (as points) for H2 tree
// Input parameters:
//   h2eri->num_sp    : Number of screened shell pairs (SSP)
//   h2eri->sp        : Array, size 2 * num_sp, SSP
//   h2eri->sp_center : Array, size 3 * num_sp, centers of SSP
//   h2eri->sp_extent : Array, size num_sp, extents of SSP
//   max_leaf_points  : Maximum number of point in a leaf node's box. If <= 0, 
//                      will use 300.
//   max_leaf_size    : Maximum size of a leaf node's box. 
// Output parameter:
//   h2eri->h2pack    : H2Pack structure with point partitioning info
//   h2eri->sp        : Array, size 2 * num_sp, sorted SSP
//   h2eri->sp_center : Array, size 3 * num_sp, sorted centers of SSP
//   h2eri->sp_extent : Array, size num_sp, sorted extents of SSP
void H2ERI_partition_sp_centers(H2ERI_p h2eri, int max_leaf_points, double max_leaf_size)
{
    // 1. Partition screened shell pair centers
    int num_sp = h2eri->num_sp;
    double *sp_center = h2eri->sp_center;
    if (max_leaf_points <= 0)   max_leaf_points = 300;
    if (max_leaf_size   <= 0.0) max_leaf_size   = 1.0;
    // Manually set the kernel matrix size for h2eri->h2pack->tb allocation.
    shell_t *sp_shells = h2eri->sp_shells;
    int num_sp_bfp = 0;
    for (int i = 0; i < num_sp; i++)
    {
        int am0 = sp_shells[i].am;
        int am1 = sp_shells[i + num_sp].am;
        num_sp_bfp += NCART(am0) * NCART(am1);
    }
    h2eri->h2pack->krnl_mat_size = num_sp_bfp; 
    h2eri->h2pack->is_H2ERI = 1;  // Tell H2Pack not to set krnl_mat_size and mat_cluster
    H2P_partition_points(
        h2eri->h2pack, num_sp, sp_center, 
        max_leaf_points, max_leaf_size
    );
    memcpy(sp_center, h2eri->h2pack->coord, sizeof(double) * 3 * num_sp);
    
    // 2. Permute the screened shell pairs and their extents according to 
    // the permutation of their center coordinate
    int *coord_idx = h2eri->h2pack->coord_idx;
    double  *sp_extent    = h2eri->sp_extent;
    int     *sp_shell_idx = h2eri->sp_shell_idx;
    shell_t *sp_shells_new    = (shell_t *) malloc(sizeof(shell_t) * num_sp * 2);
    double  *sp_extent_new    = (double *)  malloc(sizeof(double)  * num_sp);
    int     *sp_shell_idx_new = (int *)     malloc(sizeof(int)     * num_sp * 2);
    assert(sp_shells_new != NULL && sp_extent_new != NULL);
    assert(sp_shell_idx_new != NULL);
    for (int i = 0; i < num_sp; i++)
    {
        int cidx_i = coord_idx[i];
        int i20 = i, i21 = i + num_sp;
        int cidx_i20 = cidx_i, cidx_i21 = cidx_i + num_sp;
        sp_extent_new[i] = sp_extent[cidx_i];
        simint_initialize_shell(&sp_shells_new[i20]);
        simint_initialize_shell(&sp_shells_new[i21]);
        simint_allocate_shell(sp_shells[cidx_i20].nprim, &sp_shells_new[i20]);
        simint_allocate_shell(sp_shells[cidx_i21].nprim, &sp_shells_new[i21]);
        simint_copy_shell(&sp_shells[cidx_i20], &sp_shells_new[i20]);
        simint_copy_shell(&sp_shells[cidx_i21], &sp_shells_new[i21]);
        sp_shell_idx_new[i20] = sp_shell_idx[cidx_i20];
        sp_shell_idx_new[i21] = sp_shell_idx[cidx_i21];
    }
    CMS_destroy_shells(num_sp * 2, h2eri->sp_shells);
    free(h2eri->sp_shells);
    free(h2eri->sp_extent);
    free(h2eri->sp_shell_idx);
    h2eri->sp_shells    = sp_shells_new;
    h2eri->sp_extent    = sp_extent_new;
    h2eri->sp_shell_idx = sp_shell_idx_new;
    
    // 3. Initialize shell pairs. Note that Simint MATLAB code uses (NM|QP) instead
    // of the normal (MN|PQ) order for ERI. We follow this for the moment.
    h2eri->sp = (multi_sp_t *) malloc(sizeof(multi_sp_t) * num_sp);
    h2eri->index_seq = (int *) malloc(sizeof(int) * num_sp);
    assert(h2eri->sp != NULL && h2eri->index_seq != NULL);
    for (int i = 0; i < num_sp; i++)
    {
        h2eri->index_seq[i] = i;
        simint_initialize_multi_shellpair(&h2eri->sp[i]);
        simint_create_multi_shellpair(
            1, &h2eri->sp_shells[i + num_sp], 
            1, &h2eri->sp_shells[i], &h2eri->sp[i], SIMINT_SCREEN_NONE
        );
    }
}

// Calculate the basis function indices information of shells and shell pairs 
// Input parameters:
//   h2eri->num_sp : Number of shell pairs
//   h2eri->sp     : Array, size num_sp * 2, each row is a screened shell pair
// Output parameters:
//   h2eri->shell_bf_sidx : Array, size nshell, indices of each shell's first basis function
//   h2eri->sp_nbfp       : Array, size num_sp, number of basis function pairs of each SSP
//   h2eri->sp_bfp_sidx   : Array, size num_sp+1, indices of each SSP first basis function pair
void H2ERI_calc_bf_bfp_info(H2ERI_p h2eri)
{
    int nshell = h2eri->nshell;
    int num_sp = h2eri->num_sp;
    shell_t *shells = h2eri->shells;
    shell_t *sp_shells = h2eri->sp_shells;
    
    h2eri->shell_bf_sidx = (int *) malloc(sizeof(int) * (nshell + 1));
    h2eri->sp_nbfp       = (int *) malloc(sizeof(int) * num_sp);
    h2eri->sp_bfp_sidx   = (int *) malloc(sizeof(int) * (num_sp + 1));
    assert(h2eri->shell_bf_sidx != NULL && h2eri->sp_nbfp != NULL);
    assert(h2eri->sp_bfp_sidx != NULL);
    
    h2eri->shell_bf_sidx[0] = 0;
    for (int i = 0; i < nshell; i++)
    {
        int am = shells[i].am;
        h2eri->shell_bf_sidx[i + 1] = h2eri->shell_bf_sidx[i] + NCART(am);
        h2eri->max_am = MAX(h2eri->max_am, am);
    }
    h2eri->num_bf = h2eri->shell_bf_sidx[nshell];
    h2eri->max_shell_nbf = NCART(h2eri->max_am);
    
    h2eri->sp_bfp_sidx[0] = 0;
    for (int i = 0; i < num_sp; i++)
    {
        int am0  = sp_shells[i].am;
        int am1  = sp_shells[i + num_sp].am;
        int nbf0 = NCART(am0);
        int nbf1 = NCART(am1);
        int nbfp = nbf0 * nbf1;
        h2eri->sp_nbfp[i] = nbfp;
        h2eri->sp_bfp_sidx[i + 1] = h2eri->sp_bfp_sidx[i] + nbfp;
    }
    h2eri->num_sp_bfp = h2eri->sp_bfp_sidx[num_sp];
}

// Calculate the max extent of shell pairs in each H2 box
// Input parameters:
//   h2eri->h2pack    : H2 tree partitioning info
//   h2eri->num_sp    : Number of SSP
//   h2eri->sp_center : Array, size 3 * num_sp, centers of SSP, sorted
//   h2eri->sp_extent : Array, size num_sp, extents of SSP, sorted
// Output parameter:
//   h2eri->box_extent : Array, size h2pack->n_node, extent of each H2 node box
void H2ERI_calc_box_extent(H2ERI_p h2eri)
{
    H2Pack_p h2pack = h2eri->h2pack;
    int    n_node        = h2pack->n_node;
    int    max_level     = h2pack->max_level;
    int    max_child     = h2pack->max_child;
    int    n_leaf_node   = h2pack->n_leaf_node;
    int    *pt_cluster   = h2pack->pt_cluster;
    int    *children     = h2pack->children;
    int    *n_child      = h2pack->n_child;
    int    *level_nodes  = h2pack->level_nodes;
    int    *level_n_node = h2pack->level_n_node;
    double *enbox        = h2pack->enbox;
    int    num_sp        = h2eri->num_sp;
    double *sp_center    = h2eri->sp_center;
    double *sp_extent    = h2eri->sp_extent;
    
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
                int pt_s = pt_cluster[2 * node];
                int pt_e = pt_cluster[2 * node + 1];
                double box_extent_node = 0.0;
                for (int d = 0; d < 3; d++)
                {
                    double *center_d = sp_center + d * num_sp;
                    double enbox_width_d = node_enbox[3 + d];
                    for (int k = pt_s; k <= pt_e; k++)
                    {
                        double tmp_extent_d_k;
                        // Distance of shell pair center to the upper limit along each dimension
                        tmp_extent_d_k  = fabs(center_d[k] - enbox_center[d]);
                        tmp_extent_d_k  = 0.5 * enbox_width_d - tmp_extent_d_k;
                        // Outreach of each extent box
                        tmp_extent_d_k  = sp_extent[k] - tmp_extent_d_k;
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
//   h2eri->h2pack      : H2 tree partitioning info
//   h2eri->sp_bfp_sidx : Array, size num_sp+1, indices of each SSP first basis function pair
// Output parameter:
//   h2eri->h2pack->mat_cluster : Array, size h2pack->n_node * 2, matvec cluster for H2 nodes
void H2ERI_calc_mat_cluster(H2ERI_p h2eri)
{
    H2Pack_p h2pack  = h2eri->h2pack;
    int n_node       = h2pack->n_node;
    int max_child    = h2pack->max_child;
    int *pt_cluster  = h2pack->pt_cluster;
    int *children    = h2pack->children;
    int *n_child     = h2pack->n_child;
    int *mat_cluster = h2pack->mat_cluster;
    int *sp_bfp_sidx = h2eri->sp_bfp_sidx;
    
    int offset = 0;
    for (int i = 0; i < n_node; i++)
    {
        int i20 = i * 2;
        int i21 = i * 2 + 1;
        int n_child_i = n_child[i];
        if (n_child_i == 0)
        {
            int pt_s = pt_cluster[2 * i];
            int pt_e = pt_cluster[2 * i + 1];
            int node_nbf = sp_bfp_sidx[pt_e + 1] - sp_bfp_sidx[pt_s];
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

void H2ERI_calc_tree_interaction(
    const int pt_dim, const int max_child, const int *children, const int *n_child,
    const double alpha, const double *enbox, const int row_node, const int col_node,
    H2P_int_vec_p adm_pairs, H2P_int_vec_p inadm_pairs
)
{
    int n_child_r = n_child[row_node];
    int n_child_c = n_child[col_node];

    // 1. Admissible pair
    const double *enbox_r = enbox + row_node * pt_dim * 2;
    const double *enbox_c = enbox + col_node * pt_dim * 2;
    if (H2P_check_box_admissible(enbox_r, enbox_c, pt_dim, alpha))
    {
        H2P_int_vec_push_back(adm_pairs, row_node);
        H2P_int_vec_push_back(adm_pairs, col_node);
        return;
    }
    
    // 2. Two inadmissible leaf node
    if ((n_child_r == 0) && (n_child_c == 0))
    {
        H2P_int_vec_push_back(inadm_pairs, row_node);
        H2P_int_vec_push_back(inadm_pairs, col_node);
        return;
    }
    
    // 3. row_node is leaf node, col_node is non-leaf node
    if ((n_child_r == 0) && (n_child_c > 0))
    {
        const int *col_children = children + col_node * max_child;
        for (int j = 0; j < n_child_c; j++)
        {
            int col_child_j = col_children[j];
            H2ERI_calc_tree_interaction(
                pt_dim, max_child, children, n_child,
                alpha, enbox, row_node, col_child_j,
                adm_pairs, inadm_pairs
            );
        }
        return;
    }
    
    // 4. row_node is non-leaf node, col_node is leaf node
    if ((n_child_r > 0) && (n_child_c == 0))
    {
        const int *row_children = children + row_node * max_child;
        for (int i = 0; i < n_child_r; i++)
        {
            int row_child_i = row_children[i];
            H2ERI_calc_tree_interaction(
                pt_dim, max_child, children, n_child,
                alpha, enbox, row_child_i, col_node,
                adm_pairs, inadm_pairs
            );
        }
        return;
    }
    
    // 5. Neither row_node nor col_node is leaf node
    if ((n_child_r > 0) && (n_child_c > 0))
    {
        const int *row_children = children + row_node * max_child;
        const int *col_children = children + col_node * max_child;
        for (int i = 0; i < n_child_r; i++)
        {
            int row_child_i = row_children[i];
            for (int j = 0; j < n_child_c; j++)
            {
                int col_child_j = col_children[j];
                H2ERI_calc_tree_interaction(
                    pt_dim, max_child, children, n_child,
                    alpha, enbox, row_child_i, col_child_j,
                    adm_pairs, inadm_pairs
                );
            }
        }
    }
}

// Find each node's admissible and inadmissible pair nodes
// Input parameter:
//   h2eri->h2pack : H2 tree partitioning info
// Output parameters:
//   h2eri->node_adm_pairs        : Array, size unknown, each node's admissible node pairs
//   h2eri->node_adm_pairs_sidx   : Array, size h2pack->n_node+1, index of each node's first admissible node pair
//   h2eri->node_inadm_pairs      : Array, size unknown, each node's inadmissible node pairs
//   h2eri->node_inadm_pairs_sidx : Array, size h2pack->n_node+1, index of each node's first inadmissible node pair
void H2ERI_calc_node_adm_inadm_pairs(H2ERI_p h2eri)
{
    H2Pack_p h2pack = h2eri->h2pack;

    int    pt_dim    = h2pack->pt_dim;
    int    max_child = h2pack->max_child;
    int    root_idx  = h2pack->root_idx;
    int    *children = h2pack->children;
    int    *n_child  = h2pack->n_child;
    double *enbox    = h2pack->enbox;
    H2P_int_vec_p adm_pairs, inadm_pairs;
    H2P_int_vec_init(&adm_pairs, 8192);
    H2P_int_vec_init(&inadm_pairs, 8192);
    H2ERI_calc_tree_interaction(
        pt_dim, max_child, children, n_child,
        ALPHA_H2, enbox, root_idx, root_idx,
        adm_pairs, inadm_pairs
    );

    int n_node       = h2pack->n_node;
    int n_node1      = n_node + 1;
    int n_adm_pair   = adm_pairs->length   / 2;
    int n_inadm_pair = inadm_pairs->length / 2;
    int *node_adm_pairs_sidx   = (int *) malloc(sizeof(int) * n_node1);
    int *node_inadm_pairs_sidx = (int *) malloc(sizeof(int) * n_node1);
    int *node_adm_pairs        = (int *) malloc(sizeof(int) * n_adm_pair);
    int *node_adm_pairs_idx    = (int *) malloc(sizeof(int) * n_adm_pair);
    int *node_inadm_pairs      = (int *) malloc(sizeof(int) * n_inadm_pair);
    int *node_inadm_pairs_idx  = (int *) malloc(sizeof(int) * n_inadm_pair);
    ASSERT_PRINTF(node_adm_pairs_sidx   != NULL, "Failed to allocate node_adm_pairs_sidx   of size %d\n", n_node1);
    ASSERT_PRINTF(node_inadm_pairs_sidx != NULL, "Failed to allocate node_inadm_pairs_sidx of size %d\n", n_node1);
    ASSERT_PRINTF(node_adm_pairs        != NULL, "Failed to allocate node_adm_pairs        of size %d\n", n_adm_pair);
    ASSERT_PRINTF(node_adm_pairs_idx    != NULL, "Failed to allocate node_adm_pairs_idx    of size %d\n", n_adm_pair);
    ASSERT_PRINTF(node_inadm_pairs      != NULL, "Failed to allocate node_inadm_pairs      of size %d\n", n_inadm_pair);
    ASSERT_PRINTF(node_inadm_pairs_idx  != NULL, "Failed to allocate node_inadm_pairs_idx  of size %d\n", n_inadm_pair);

    memset(node_adm_pairs_sidx, 0, sizeof(int) * n_node1);
    for (int i = 0; i < n_adm_pair; i++)
    {
        int node0 = adm_pairs->data[2 * i];
        int node1 = adm_pairs->data[2 * i + 1];
        node_adm_pairs_sidx[node0 + 1]++;
    }
    for (int i = 2; i <= n_node; i++)
        node_adm_pairs_sidx[i] += node_adm_pairs_sidx[i - 1];
    for (int i = 0; i < n_adm_pair; i++)
    {
        int node0 = adm_pairs->data[2 * i];
        int node1 = adm_pairs->data[2 * i + 1];
        int idx0  = node_adm_pairs_sidx[node0];
        node_adm_pairs[idx0] = node1;
        node_adm_pairs_sidx[node0]++;
    }
    for (int i = n_node; i >= 1; i--)
        node_adm_pairs_sidx[i] = node_adm_pairs_sidx[i - 1];
    node_adm_pairs_sidx[0] = 0;

    memset(node_inadm_pairs_sidx, 0, sizeof(int) * n_node1);
    for (int i = 0; i < n_inadm_pair; i++)
    {
        int node0 = inadm_pairs->data[2 * i];
        int node1 = inadm_pairs->data[2 * i + 1];
        node_inadm_pairs_sidx[node0 + 1]++;
    }
    for (int i = 2; i <= n_node; i++)
        node_inadm_pairs_sidx[i] += node_inadm_pairs_sidx[i - 1];
    for (int i = 0; i < n_inadm_pair; i++)
    {
        int node0 = inadm_pairs->data[2 * i];
        int node1 = inadm_pairs->data[2 * i + 1];
        int idx0  = node_inadm_pairs_sidx[node0];
        node_inadm_pairs[idx0] = node1;
        node_inadm_pairs_sidx[node0]++;
    }
    for (int i = n_node; i >= 1; i--)
        node_inadm_pairs_sidx[i] = node_inadm_pairs_sidx[i - 1];
    node_inadm_pairs_sidx[0] = 0;

    h2eri->node_adm_pairs_sidx   = node_adm_pairs_sidx;
    h2eri->node_inadm_pairs_sidx = node_inadm_pairs_sidx;
    h2eri->node_adm_pairs        = node_adm_pairs;
    h2eri->node_inadm_pairs      = node_inadm_pairs;
    h2eri->node_adm_pairs_idx    = node_adm_pairs_idx;
    h2eri->node_inadm_pairs_idx  = node_inadm_pairs_idx;

    H2P_int_vec_destroy(&adm_pairs);
    H2P_int_vec_destroy(&inadm_pairs);
}

// Calculate plist, plist_idx, plist_sidx for exchange matrix construction 
// Input parameters:
//   h2eri->nshell       : Number of shells
//   h2eri->num_sp       : Number of screened shell pairs (SSP)
//   h2eri->sp_shell_idx : Array, size 2 * num_sp, each row is the contracted shell indices of a SSP
// Output parameters:
//   h2eri->plist      : Array, size unknown, each shell's screened pair shells
//   h2eri->plist_idx  : Array, size unknown, corresponding indices of each shell's screened pair shells in sp_bfp_sidx
//   h2eri->plist_sidx : Array, size nshell+1, index of each node's first item in plist & plist_idx
void H2ERI_build_plist(H2ERI_p h2eri)
{
    int nshell        = h2eri->nshell;
    int nshell1       = nshell + 1;
    int num_sp        = h2eri->num_sp;
    int *sp_shell_idx = h2eri->sp_shell_idx;

    int *plist      = (int *) malloc(sizeof(int) * 2 * num_sp);
    int *plist_idx  = (int *) malloc(sizeof(int) * 2 * num_sp);
    int *plist_sidx = (int *) malloc(sizeof(int) * nshell1);
    ASSERT_PRINTF(plist      != NULL, "Failed to allocate plist      of size %d\n", 2 * num_sp);
    ASSERT_PRINTF(plist_idx  != NULL, "Failed to allocate plist_idx  of size %d\n", 2 * num_sp);
    ASSERT_PRINTF(plist_sidx != NULL, "Failed to allocate plist_sidx of size %d\n", nshell1);
    memset(plist_sidx, 0, sizeof(int) * nshell1);
    for (int i = 0; i < num_sp; i++)
    {
        int i20 = i, i21 = i + num_sp;
        int shell0 = sp_shell_idx[i20];
        int shell1 = sp_shell_idx[i21];
        plist_sidx[shell0 + 1]++;
        if (shell0 != shell1) plist_sidx[shell1 + 1]++;
    }
    for (int i = 2; i <= nshell; i++)
        plist_sidx[i] += plist_sidx[i - 1];
    for (int i = 0; i < num_sp; i++)
    {
        int i20 = i, i21 = i + num_sp;
        int shell0 = sp_shell_idx[i20];
        int shell1 = sp_shell_idx[i21];
        int idx0 = plist_sidx[shell0];
        plist[idx0] = shell1;
        plist_idx[idx0] = i;
        plist_sidx[shell0]++;
        if (shell0 != shell1)
        {
            int idx1 = plist_sidx[shell1];
            plist[idx1] = shell0;
            plist_idx[idx1] = i;
            plist_sidx[shell1]++;
        }
    }
    for (int i = nshell; i >= 1; i--)
        plist_sidx[i] = plist_sidx[i - 1];
    plist_sidx[0] = 0;

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < nshell; i++)
    {
        int sidx = plist_sidx[i];
        int len  = plist_sidx[i + 1] - sidx;
        H2P_qsort_int_key_val(plist + sidx, plist_idx + sidx, 0, len - 1);
    }

    h2eri->plist      = plist;
    h2eri->plist_idx  = plist_idx;
    h2eri->plist_sidx = plist_sidx;
}

// H2 partition of screened shell pair centers
void H2ERI_partition(H2ERI_p h2eri)
{
    H2ERI_partition_sp_centers(h2eri, 0, 0.0);
    H2ERI_calc_bf_bfp_info(h2eri);
    H2ERI_calc_box_extent (h2eri);
    H2ERI_calc_mat_cluster(h2eri);
    H2ERI_calc_node_adm_inadm_pairs(h2eri);
    H2ERI_build_plist(h2eri);
    H2ERI_exchange_workbuf_init(h2eri);
    
    // Initialize thread local Simint buffer
    int n_thread = h2eri->h2pack->n_thread;
    h2eri->simint_buffs    = (simint_buff_p*)    malloc(sizeof(simint_buff_p)    * n_thread);
    h2eri->eri_batch_buffs = (eri_batch_buff_p*) malloc(sizeof(eri_batch_buff_p) * n_thread);
    assert(h2eri->simint_buffs    != NULL);
    assert(h2eri->eri_batch_buffs != NULL);
    for (int i = 0; i < n_thread; i++)
    {
        CMS_init_Simint_buff(h2eri->max_am, &h2eri->simint_buffs[i]);
        CMS_init_eri_batch_buff(h2eri->max_am, 4, &h2eri->eri_batch_buffs[i]);
    }
}

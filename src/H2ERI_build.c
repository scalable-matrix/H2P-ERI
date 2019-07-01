#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <omp.h>

#ifdef USE_MKL
#include <mkl.h>
#endif

#include "CMS.h"
#include "H2ERI_typedef.h"
#include "H2Pack_utils.h"
#include "H2Pack_build.h"
#include "H2Pack_aux_structs.h"
#include "H2Pack_ID_compress.h"

// Partition the ring area (r1 < r < r2) using multiple layers of 
// box surface and generate the same number of uniformly distributed 
// proxy points on each box surface layer [-r, r]^3. 
// Input parameters:
//   r1, r2     : Radius of ring area
//   nlayer     : Number of layers
//   npts_layer : Minimum number of proxy points on each layer
// Output parameters:
//   pp : H2P_dense_mat structure, contains coordinates of proxy points
void H2ERI_generate_proxy_point_layers(
    const double r1, const double r2, const int nlayer, 
    int npts_layer, H2P_dense_mat_t pp
)
{
    // 1. Decide the number of proxy points on each layer
    int npts_face = npts_layer / 6;
    int npts_axis = (int) ceil(sqrt((double) npts_face));
    npts_layer = 6 * npts_axis * npts_axis;
    int npts_total = nlayer * npts_layer;
    H2P_dense_mat_resize(pp, 3, npts_total);
    
    // 2. Generate a layer of proxy points on a standard [-1, 1]^3 box surface
    double h = 2.0 / (double) (npts_axis + 1);
    double *x = pp->data;
    double *y = pp->data + npts_total;
    double *z = pp->data + npts_total * 2;
    int index = 0;
    for (int i = 0; i < npts_axis; i++)
    {
        double h_i = h * (i + 1) - 1.0;
        for (int j = 0; j < npts_axis; j++)
        {
            double h_j = h * (j + 1) - 1.0;
            
            x[index + 0] = h_i;
            y[index + 0] = h_j;
            z[index + 0] = -1.0;
            
            x[index + 1] = h_i;
            y[index + 1] = h_j;
            z[index + 1] = 1.0;
            
            x[index + 2] = h_i;
            y[index + 2] = -1.0;
            z[index + 2] = h_j;
            
            x[index + 3] = h_i;
            y[index + 3] = 1.0;
            z[index + 3] = h_j;
            
            x[index + 4] = -1.0;
            y[index + 4] = h_i;
            z[index + 4] = h_j;
            
            x[index + 5] = 1.0;
            y[index + 5] = h_i;
            z[index + 5] = h_j;
            
            index += 6;
        }
    }
    // Copy the proxy points on the standard [-1, 1]^3 box surface to each layer
    size_t layer_msize = sizeof(double) * npts_layer;
    for (int i = 1; i < nlayer; i++)
    {
        memcpy(x + i * npts_layer, x, layer_msize);
        memcpy(y + i * npts_layer, y, layer_msize);
        memcpy(z + i * npts_layer, z, layer_msize);
    }
    
    // 3. Scale each layer
    int nlayer1 = MAX(nlayer - 1, 1);
    double dr = ((r2 - r1) / r1) / (double) nlayer1;
    for (int i = 0; i < nlayer; i++)
    {
        double *x_i = x + i * npts_layer;
        double *y_i = y + i * npts_layer;
        double *z_i = z + i * npts_layer;
        double r = r1 * (1.0 + i * dr);
        #pragma omp simd
        for (int j = 0; j < npts_layer; j++)
        {
            x_i[j] *= r;
            y_i[j] *= r;
            z_i[j] *= r;
        }
    }
}

// For all nodes, find shell pairs in idx_in that:
//   1. Are admissible from i-th node;
//   2. Their extents overlap with i-th node's near field boxes (super cell).
// Input parameters:
//   h2eri->h2pack        : H2 tree partitioning info
//   h2eri->num_unc_sp    : Number of fully uncontracted shell pairs (FUSP)
//   h2eri->unc_sp_center : Array, size 3 * num_unc_sp, centers of FUSP, sorted
//   h2eri->unc_sp_extent : Array, size num_unc_sp, extents of FUSP, sorted
// Output parameters:
//   h2eri->ovlp_ff_idx : Array, size h2pack->n_node, i-th vector contains
//                        FUSP indices that satisfy the requirements.
void H2ERI_calc_ovlp_ff_idx(H2ERI_t h2eri)
{
    H2Pack_t h2pack = h2eri->h2pack;
    int    n_node         = h2pack->n_node;
    int    root_idx       = h2pack->root_idx;
    int    n_point        = h2pack->n_point;    // == h2eri->num_unc_sp
    int    min_adm_level  = h2pack->min_adm_level; 
    int    max_level      = h2pack->max_level;  // level = [0, max_level], total max_level+1 levels
    int    max_child      = h2pack->max_child;
    int    n_leaf_node    = h2pack->n_leaf_node;
    int    *cluster       = h2pack->cluster;
    int    *children      = h2pack->children;
    int    *n_child       = h2pack->n_child;
    int    *level_nodes   = h2pack->level_nodes;
    int    *level_n_node  = h2pack->level_n_node;
    double *enbox         = h2pack->enbox;
    double *center   = h2eri->unc_sp_center;
    double *extent   = h2eri->unc_sp_extent;
    double *center_x = center;
    double *center_y = center + n_point;
    double *center_z = center + n_point * 2;
    
    // 1. Initialize ovlp_ff_idx
    h2eri->ovlp_ff_idx = (H2P_int_vec_t *) malloc(sizeof(H2P_int_vec_t) * n_node);
    assert(h2eri->ovlp_ff_idx != NULL);
    H2P_int_vec_t *ovlp_ff_idx = h2eri->ovlp_ff_idx;
    for (int i = 0; i < n_node; i++)
        H2P_int_vec_init(&ovlp_ff_idx[i], n_point);  // Won't exceed n_point
    ovlp_ff_idx[root_idx]->length = n_point;
    for (int i = 0; i < n_point; i++)
        ovlp_ff_idx[root_idx]->data[i] = i;
    
    // 2. Hierarchical partition of all centers
    for (int i = 0; i < max_level; i++)
    {
        int *level_i_nodes = level_nodes + i * n_leaf_node;
        int level_i_n_node = level_n_node[i];
        for (int j = 0; j < level_i_n_node; j++)
        {
            int node = level_i_nodes[j];
            int node_n_child = n_child[node];
            if (node_n_child == 0) continue;
            int *child_nodes = children + node * max_child;
            
            H2P_int_vec_t tmp_ff_idx = ovlp_ff_idx[node];
            int n_tmp_ff_idx = tmp_ff_idx->length;
            for (int k = 0; k < node_n_child; k++)
            {
                int child_k = child_nodes[k];
                double *enbox_k = enbox + 6 * child_k;
                double enbox_center[3];
                enbox_center[0] = enbox_k[0] + 0.5 * enbox_k[3];
                enbox_center[1] = enbox_k[1] + 0.5 * enbox_k[4];
                enbox_center[2] = enbox_k[2] + 0.5 * enbox_k[5];
                
                // Half width of current node's super cell
                double sup_coef = 0.5 + ALPHA_SUP;
                double sup_semi_L[3];
                sup_semi_L[0] = sup_coef * enbox_k[3];
                sup_semi_L[1] = sup_coef * enbox_k[4];
                sup_semi_L[2] = sup_coef * enbox_k[5];
                
                for (int l = 0; l < n_tmp_ff_idx; l++)
                {
                    int ff_idx_l = tmp_ff_idx->data[l];
                    double extent_l = extent[ff_idx_l];
                    
                    // Left corner of each center's extent box to the left 
                    // corner of current child node's super cell box
                    double rel_x = center_x[ff_idx_l] - enbox_center[0];
                    double rel_y = center_y[ff_idx_l] - enbox_center[1];
                    double rel_z = center_z[ff_idx_l] - enbox_center[2];
                    rel_x += sup_semi_L[0] - extent_l;
                    rel_y += sup_semi_L[1] - extent_l;
                    rel_z += sup_semi_L[2] - extent_l;
                    
                    int left_x  = (rel_x <  0);
                    int left_y  = (rel_y <  0);
                    int left_z  = (rel_z <  0);
                    int right_x = (rel_x >= 0);
                    int right_y = (rel_y >= 0);
                    int right_z = (rel_z >= 0);
                    int adm_left_x  = (fabs(rel_x) >= 2.0 * extent_l - 1e-8);
                    int adm_left_y  = (fabs(rel_y) >= 2.0 * extent_l - 1e-8);
                    int adm_left_z  = (fabs(rel_z) >= 2.0 * extent_l - 1e-8);
                    int adm_right_x = (fabs(rel_x) >= 2.0 * sup_semi_L[0] - 1e-8);
                    int adm_right_y = (fabs(rel_y) >= 2.0 * sup_semi_L[1] - 1e-8);
                    int adm_right_z = (fabs(rel_z) >= 2.0 * sup_semi_L[2] - 1e-8);
                    int adm_x = ((left_x && adm_left_x) || (right_x && adm_right_x));
                    int adm_y = ((left_y && adm_left_y) || (right_y && adm_right_y));
                    int adm_z = ((left_z && adm_left_z) || (right_z && adm_right_z));
                    int inadm = (!(adm_x || adm_y || adm_z));
                    if (inadm)
                    {
                        int tail = ovlp_ff_idx[child_k]->length;
                        ovlp_ff_idx[child_k]->data[tail] = ff_idx_l;
                        ovlp_ff_idx[child_k]->length++;
                    }
                }  // End of l loop
            }  // End of k loop
        }  // End of j loop
    }  // End of i loop
    
    // 3. Remove centers that are in each node's inadmissible neighbor nodes
    int *tmp_ff_idx = (int *) malloc(sizeof(int) * n_point);
    assert(tmp_ff_idx != NULL);
    for (int i = min_adm_level; i <= max_level; i++)
    {
        int *level_i_nodes = level_nodes + i * n_leaf_node;
        int level_i_n_node = level_n_node[i];
        for (int j = 0; j < level_i_n_node; j++)
        {
            int node = level_i_nodes[j];
            double enbox_center[3], adm_semi_L[3];
            double *node_enbox = enbox + 6 * node;
            double adm_coef = 0.5 + ALPHA_H2;
            enbox_center[0] = node_enbox[0] + 0.5 * node_enbox[3];
            enbox_center[1] = node_enbox[1] + 0.5 * node_enbox[4];
            enbox_center[2] = node_enbox[2] + 0.5 * node_enbox[5];
            adm_semi_L[0] = adm_coef * node_enbox[3];
            adm_semi_L[1] = adm_coef * node_enbox[4];
            adm_semi_L[2] = adm_coef * node_enbox[5];
            
            int *ff_idx  = ovlp_ff_idx[node]->data;
            int n_ff_idx = ovlp_ff_idx[node]->length;
            int ff_cnt = 0;
            for (int l = 0; l < n_ff_idx; l++)
            {
                int ff_idx_l = ff_idx[l];
                double rel_x = fabs(center_x[ff_idx_l] - enbox_center[0]);
                double rel_y = fabs(center_y[ff_idx_l] - enbox_center[1]);
                double rel_z = fabs(center_z[ff_idx_l] - enbox_center[2]);
                int adm_x = (rel_x > adm_semi_L[0]);
                int adm_y = (rel_y > adm_semi_L[1]);
                int adm_z = (rel_z > adm_semi_L[2]);
                if (adm_x || adm_y || adm_z)
                {
                    tmp_ff_idx[ff_cnt] = ff_idx_l;
                    ff_cnt++;
                }
            }
            memcpy(ff_idx, tmp_ff_idx, sizeof(int) * ff_cnt);
            ovlp_ff_idx[node]->length = ff_cnt;
        }  // End of j loop
    }  // End of i loop
    free(tmp_ff_idx);
}

// Extract shell pair and row indices of a target row index set from
// a given set of FISP
// Input parameters:
//   unc_sp   : Array, size num_unc_sp, FUSP
//   row_idx  : Vector, target row indices set
//   sp_idx   : Vector, given FUSP set 
//   work_buf : Vector, work buffer
// Output parameters:
//   pair_idx    : Vector, FUSP indices that contains target row indices set
//   row_idx_new : Vector, target row new indices in pair_idx FUSP
void H2ERI_extract_shell_pair_idx(
    const multi_sp_t *unc_sp, H2P_int_vec_t row_idx,
    H2P_int_vec_t sp_idx,   H2P_int_vec_t work_buf,
    H2P_int_vec_t pair_idx, H2P_int_vec_t row_idx_new
)
{
    int num_target = row_idx->length;
    int num_sp = sp_idx->length;
    
    H2P_int_vec_set_capacity(work_buf, num_sp * 5 + num_target + 2);
    int *nbf1    = work_buf->data;
    int *nbf2    = nbf1 + num_sp;
    int *off12   = nbf2 + num_sp;
    int *sp_flag = off12 + (num_sp + 1);
    int *tmp_idx = sp_flag + num_sp;
    int *idx_off = tmp_idx + num_target;
    
    off12[0] = 0;
    for (int i = 0; i < num_sp; i++)
    {
        const multi_sp_t *sp_i = unc_sp + sp_idx->data[i];
        nbf1[i] = NCART(sp_i->am1);
        nbf2[i] = NCART(sp_i->am2);
        off12[i + 1] = off12[i] + nbf1[i] * nbf2[i];
    }
    
    memset(sp_flag, 0, sizeof(int) * num_sp);
    for (int i = 0; i < num_target; i++)
    {
        int j = 0, x = row_idx->data[i];
        for (j = 0; j < num_sp; j++) 
            if (off12[j] <= x && x < off12[j + 1]) break;
        tmp_idx[i] = j;
        sp_flag[j] = 1;
    }
    
    H2P_int_vec_set_capacity(pair_idx, num_sp);
    int npair = 0;
    for (int i = 0; i < num_sp; i++)
    {
        if (sp_flag[i])
        {
            pair_idx->data[npair] = i;
            sp_flag[i] = npair;
            npair++;
        }
    }
    pair_idx->length = npair;
    
    idx_off[0] = 0;
    for (int i = 0; i < npair; i++) 
    {
        int spidx = pair_idx->data[i];
        idx_off[i + 1] = idx_off[i] + nbf1[spidx] * nbf2[spidx];
    }
    
    H2P_int_vec_set_capacity(row_idx_new, num_target);
    for (int i = 0; i < num_target; i++)
    {
        int sp_idx1 = tmp_idx[i];
        int sp_idx2 = sp_flag[sp_idx1];
        row_idx_new->data[i] = row_idx->data[i] - off12[sp_idx1] + idx_off[sp_idx2];
    }
    row_idx_new->length = num_target;
}

// Generate normal distribution random number, Marsaglia polar method
// Input parameters:
//   mu, sigma : Normal distribution parameters
//   nelem     : Number of random numbers to be generated
// Output parameter:
//   x : Array, size nelem, generated random numbers
void H2ERI_generate_normal_distribution(
    const double mu, const double sigma,
    const int nelem, double *x
)
{
    double u1, u2, w, mult, x1, x2;
    for (int i = 0; i < nelem - 1; i += 2)
    {
        do 
        {
            u1 = ((double) rand () / RAND_MAX) * 2.0 - 1.0;
            u2 = ((double) rand () / RAND_MAX) * 2.0 - 1.0;
            w  = u1 * u1 + u2 * u2;
        } while (w >= 1.0 || w <= 1e-15);
        mult = sqrt((-2.0 * log(w)) / w);
        x1 = u1 * mult;
        x2 = u2 * mult;
        x[i]   = mu + sigma * x1;
        x[i+1] = mu + sigma * x2;
    }
    if (nelem % 2)
    {
        do 
        {
            u1 = ((double) rand () / RAND_MAX) * 2.0 - 1.0;
            u2 = ((double) rand () / RAND_MAX) * 2.0 - 1.0;
            w  = u1 * u1 + u2 * u2;
        } while (w >= 1.0 || w <= 1e-15);
        mult = sqrt((-2.0 * log(w)) / w);
        x1 = u1 * mult;
        x[nelem - 1] = mu + sigma * x1;
    }
}

int H2ERI_gather_sum(const int *arr, H2P_int_vec_t idx)
{
    int res = 0;
    for (int i = 0; i < idx->length; i++) 
        res += arr[idx->data[i]];
    return res;
}

void H2ERI_mark_flags(int *flag, const int n_pos, const int *pos)
{
    for (int i = 0; i < n_pos; i++) flag[pos[i]] = 1;
}

// Build H2 projection matrices using proxy points
// Input parameter:
//   h2eri : H2ERI structure with point partitioning & shell pair info
// Output parameter:
//   h2eri : H2ERI structure with H2 projection blocks
void H2ERI_build_UJ_proxy(H2ERI_t h2eri)
{
    H2Pack_t h2pack = h2eri->h2pack;
    int    n_thread       = h2pack->n_thread;
    int    n_point        = h2pack->n_point;
    int    n_node         = h2pack->n_node;
    int    n_leaf_node    = h2pack->n_leaf_node;
    int    min_adm_level  = h2pack->min_adm_level;
    int    max_level      = h2pack->max_level;
    int    max_child      = h2pack->max_child;
    int    num_unc_sp     = h2eri->num_unc_sp;
    int    pp_npts_layer  = h2eri->pp_npts_layer;
    int    pp_nlayer_ext  = h2eri->pp_nlayer_ext;
    int    *children      = h2pack->children;
    int    *n_child       = h2pack->n_child;
    int    *level_nodes   = h2pack->level_nodes;
    int    *level_n_node  = h2pack->level_n_node;
    int    *node_level    = h2pack->node_level;
    int    *leaf_nodes    = h2pack->height_nodes;
    int    *cluster       = h2pack->cluster;
    int    *unc_sp_nbfp   = h2eri->unc_sp_nbfp;
    int    *index_seq     = h2eri->index_seq;
    double *enbox         = h2pack->enbox;
    double *box_extent    = h2eri->box_extent;
    double *unc_sp_center = h2eri->unc_sp_center;
    double *unc_sp_extent = h2eri->unc_sp_extent;
    void   *stop_param    = &h2pack->QR_stop_tol;
    multi_sp_t *unc_sp = h2eri->unc_sp;
    shell_t *unc_sp_shells = h2eri->unc_sp_shells;
    
    // 1. Allocate U and J
    h2pack->n_UJ  = n_node;
    h2pack->U     = (H2P_dense_mat_t*) malloc(sizeof(H2P_dense_mat_t) * n_node);
    h2eri->J_pair = (H2P_int_vec_t*)   malloc(sizeof(H2P_int_vec_t)   * n_node);
    h2eri->J_row  = (H2P_int_vec_t*)   malloc(sizeof(H2P_int_vec_t)   * n_node);
    assert(h2pack->U != NULL && h2eri->J_pair != NULL && h2eri->J_row != NULL);
    for (int i = 0; i < h2pack->n_UJ; i++)
    {
        h2pack->U[i]     = NULL;
        h2eri->J_pair[i] = NULL;
        h2eri->J_row[i]  = NULL;
    }
    H2P_dense_mat_t *U      = h2pack->U;
    H2P_int_vec_t   *J_pair = h2eri->J_pair;
    H2P_int_vec_t   *J_row  = h2eri->J_row;
    
    // 2. Calculate overlapping far field (admissible) shell pairs
    //    and auxiliary information for updating skel_flag on each level
    // skel_flag  : Marks if a point is a skeleton point on the current level
    // lvl_leaf   : Leaf nodes above the i-th level
    // lvl_n_leaf : Number of leaf nodes above the i-th level
    int n_level = max_level + 1;
    int *skel_flag  = (int *) malloc(sizeof(int) * n_point);
    int *lvl_leaf   = (int *) malloc(sizeof(int) * n_leaf_node * n_level);
    int *lvl_n_leaf = (int *) malloc(sizeof(int) * n_level);
    assert(skel_flag != NULL && lvl_leaf != NULL && lvl_n_leaf != NULL);
    // At the leaf-node level, all points are skeleton points
    for (int i = 0; i < n_point; i++) skel_flag[i] = 1;
    memset(lvl_n_leaf, 0, sizeof(int) * n_level);
    for (int i = 0; i < n_leaf_node; i++)
    {
        int leaf_i  = leaf_nodes[i];
        int level_i = node_level[leaf_i];
        for (int j = level_i + 1; j <= max_level; j++)
        {
            int idx = lvl_n_leaf[j];
            lvl_leaf[level_i * n_leaf_node + idx] = leaf_i;
            lvl_n_leaf[j]++;
        }
    }
    H2ERI_calc_ovlp_ff_idx(h2eri);
    H2P_int_vec_t *ovlp_ff_idx = h2eri->ovlp_ff_idx;
    
    // 3. Hierarchical construction level by level
    for (int i = max_level; i >= min_adm_level; i--)
    {
        int *level_i_nodes = level_nodes + i * n_leaf_node;
        int level_i_n_node = level_n_node[i];
        
        int tid = omp_get_thread_num();
        H2P_thread_buf_t thread_buf = h2pack->tb[tid];
        H2P_int_vec_t   pair_idx    = thread_buf->idx0;
        H2P_int_vec_t   row_idx     = thread_buf->idx1;
        H2P_int_vec_t   node_ff_idx = thread_buf->idx2;
        H2P_int_vec_t   ID_buff     = thread_buf->idx2;
        H2P_int_vec_t   sub_idx     = thread_buf->idx3;
        H2P_int_vec_t   sub_row_idx = thread_buf->idx2;
        H2P_int_vec_t   work_buf    = thread_buf->idx3;
        H2P_int_vec_t   sub_pair    = thread_buf->idx4;
        H2P_dense_mat_t pp          = thread_buf->mat0;
        H2P_dense_mat_t A_ff_pp     = thread_buf->mat1;
        H2P_dense_mat_t randn_mat   = thread_buf->mat0; 
        H2P_dense_mat_t A_block     = thread_buf->mat2;
        H2P_dense_mat_t QR_buff     = thread_buf->mat0;  
        simint_buff_t   buff        = h2eri->simint_buffs[tid];
        
        // A. Compress at the i-th level
        for (int j = 0; j < level_i_n_node; j++)
        {
            int node = level_i_nodes[j];
            int node_n_child = n_child[node];
            int *child_nodes = children + node * max_child;
            
            // (1) Construct row subset for this node
            if (node_n_child == 0)
            {
                int s_index = cluster[2 * node];
                int e_index = cluster[2 * node + 1];
                int node_npts = e_index - s_index + 1;
                H2P_int_vec_set_capacity(pair_idx, node_npts);
                memcpy(pair_idx->data, index_seq + s_index, sizeof(int) * node_npts);
                pair_idx->length = node_npts;
                
                int nbfp = H2ERI_gather_sum(unc_sp_nbfp, pair_idx);
                H2P_int_vec_set_capacity(row_idx, nbfp);
                for (int k = 0; k < nbfp; k++) row_idx->data[k] = k;
                row_idx->length = nbfp;
            } else {
                int row_idx_offset = 0;
                pair_idx->length = 0;
                row_idx->length  = 0;
                for (int k = 0; k < node_n_child; k++)
                {
                    int child_k = child_nodes[k];
                    H2P_int_vec_concatenate(pair_idx, J_pair[child_k]);
                    int row_idx_spos = row_idx->length;
                    int row_idx_epos = row_idx_spos + J_row[child_k]->length;
                    H2P_int_vec_concatenate(row_idx, J_row[child_k]);
                    for (int l = row_idx_spos; l < row_idx_epos; l++)
                        row_idx->data[l] += row_idx_offset;
                    row_idx_offset += H2ERI_gather_sum(unc_sp_nbfp, J_pair[child_k]);
                }
            }  // End of "if (node_n_child == 0)"
            
            // (2) Generate proxy points
            double *node_enbox = enbox + 6 * node;
            double width  = node_enbox[3];
            double extent = box_extent[node];
            double r1 = width * (0.5 + ALPHA_SUP);
            double r2 = width * (0.5 + extent);
            double d_nlayer = (extent - ALPHA_SUP) * (pp_nlayer_ext - 1);
            int nlayer_node = 1 + ceil(d_nlayer);
            H2ERI_generate_proxy_point_layers(r1, r2, nlayer_node, pp_npts_layer, pp);
            int num_pp = pp->ncol;
            double *pp_x = pp->data;
            double *pp_y = pp->data + num_pp;
            double *pp_z = pp->data + num_pp * 2;
            double center_x = node_enbox[0] + 0.5 * node_enbox[3];
            double center_y = node_enbox[1] + 0.5 * node_enbox[4];
            double center_z = node_enbox[2] + 0.5 * node_enbox[5];
            #pragma omp simd
            for (int k = 0; k < num_pp; k++)
            {
                pp_x[k] += center_x;
                pp_y[k] += center_y;
                pp_z[k] += center_z;
            }
            
            // (3) Prepare current node's overlapping far field point list
            int n_ff_idx0 = ovlp_ff_idx[node]->length;
            int *ff_idx0  = ovlp_ff_idx[node]->data;
            int n_ff_idx  = H2ERI_gather_sum(skel_flag, ovlp_ff_idx[node]);
            H2P_int_vec_set_capacity(node_ff_idx, n_ff_idx);
            n_ff_idx = 0;
            for (int k = 0; k < n_ff_idx0; k++)
            {
                int l = ff_idx0[k];
                if (skel_flag[l] == 1)
                {
                    node_ff_idx->data[n_ff_idx] = l;
                    n_ff_idx++;
                }
            }
            node_ff_idx->length = n_ff_idx;
            
            // (4) Construct NAI and ERI blocks
            // A_ff : A_blk_nrow-by-A_ff_ncol
            // A_pp : A_pp_ncol-by-A_blk_nrow, need to be transposed in gemm
            int A_blk_nrow = H2ERI_gather_sum(unc_sp_nbfp, pair_idx);
            int A_ff_ncol  = H2ERI_gather_sum(unc_sp_nbfp, node_ff_idx);
            int A_pp_ncol  = num_pp;
            H2P_dense_mat_resize(A_ff_pp, A_blk_nrow, A_ff_ncol + A_pp_ncol);
            double *A_ff = A_ff_pp->data;
            double *A_pp = A_ff_pp->data + A_blk_nrow * A_ff_ncol;
            H2ERI_calc_ERI_pairs_to_mat(
                unc_sp, pair_idx->length, node_ff_idx->length,
                pair_idx->data, node_ff_idx->data, buff, A_ff, A_ff_ncol
            );
            H2ERI_calc_NAI_pairs_to_mat(
                unc_sp_shells, num_unc_sp, pair_idx->length, pair_idx->data, 
                num_pp, pp_x, pp_y, pp_z, A_pp, A_blk_nrow
            );
            
            // (5) Randomized normalization for NAI and ERI blocks
            // randn_pp: A_pp_ncol-by-A_blk_nrow
            // randn_ff: A_ff_ncol-by-A_blk_nrow
            int randn_size = (A_pp_ncol + A_ff_ncol) * A_blk_nrow;
            int A_blk_ncol = 2 * A_blk_nrow;
            H2P_dense_mat_resize(randn_mat, 1, randn_size);
            H2P_dense_mat_resize(A_block, A_blk_nrow, A_blk_ncol);
            H2ERI_generate_normal_distribution(0.0, 1.0, randn_size, randn_mat->data);
            double *randn_pp = randn_mat->data;
            double *randn_ff = randn_mat->data + A_pp_ncol * A_blk_nrow;
            double *A_blk_pp = A_block->data;
            double *A_blk_ff = A_block->data + A_blk_nrow;
            CBLAS_GEMM(
                CblasRowMajor, CblasTrans, CblasNoTrans, 
                A_blk_nrow, A_blk_nrow, A_pp_ncol,
                1.0, A_pp, A_blk_nrow, randn_pp, A_blk_nrow,
                0.0, A_blk_pp, A_blk_ncol
            );
            CBLAS_GEMM(
                CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                A_blk_nrow, A_blk_nrow, A_ff_ncol,
                1.0, A_ff, A_ff_ncol, randn_ff, A_blk_nrow,
                0.0, A_blk_ff, A_blk_ncol
            );
            H2P_dense_mat_normalize_columns(A_block, randn_mat);
            
            // (5) ID compression
            H2P_dense_mat_select_rows(A_block, row_idx);
            H2P_dense_mat_resize(QR_buff, 1, 2 * A_block->nrow);
            H2P_int_vec_set_capacity(ID_buff, 4 * A_block->nrow);
            H2P_ID_compress(
                A_block, QR_REL_NRM, stop_param, &U[node], sub_idx, 
                1, QR_buff->data, ID_buff->data
            );
            H2P_int_vec_gather(row_idx, sub_idx, sub_row_idx);
            H2P_int_vec_init(&J_pair[node], pair_idx->length);
            H2P_int_vec_init(&J_row[node],  sub_row_idx->length);
            H2ERI_extract_shell_pair_idx(
                unc_sp, sub_row_idx, pair_idx, 
                work_buf, sub_pair, J_row[node]
            );
            H2P_int_vec_gather(pair_idx, sub_pair, J_pair[node]);
        }  // End of j loop
        
        // B. Update skeleton points after the compression at the i-th level.
        //    At the (i-1)-th level, only need to consider overlapping FF shell pairs
        //    inside the skeleton shell pairs at i-th level. Note that the skeleton
        //    shell pairs of leaf nodes at i-th level are all shell pairs in leaf nodes.
        memset(skel_flag, 0, sizeof(int) * n_point);
        for (int j = 0; j < level_i_n_node; j++)
        {
            int node = level_i_nodes[j];
            H2ERI_mark_flags(skel_flag, J_pair[node]->length, J_pair[node]->data);
        }
        for (int j = 0; j < lvl_n_leaf[i]; j++)
        {
            int leaf_j = lvl_leaf[i * n_leaf_node + j];
            int s_index = cluster[2 * leaf_j];
            int n_index = cluster[2 * leaf_j + 1] - s_index + 1;
            H2ERI_mark_flags(skel_flag, n_index, index_seq + s_index);
        }
    }  // End of i loop
    
    // 4. Initialize other not touched U J & add statistic info
    for (int i = 0; i < h2pack->n_UJ; i++)
    {
        if (U[i] == NULL)
        {
            H2P_dense_mat_init(&U[i], 1, 1);
            U[i]->nrow = 0;
            U[i]->ncol = 0;
            U[i]->ld   = 0;
        } else {
            h2pack->mat_size[0] += U[i]->nrow * U[i]->ncol;
            h2pack->mat_size[3] += U[i]->nrow * U[i]->ncol;
            h2pack->mat_size[3] += U[i]->nrow + U[i]->ncol;
            h2pack->mat_size[5] += U[i]->nrow * U[i]->ncol;
            h2pack->mat_size[5] += U[i]->nrow + U[i]->ncol;
        }
        if (J_row[i]  == NULL) H2P_int_vec_init(&J_row[i], 1);
        if (J_pair[i] == NULL) H2P_int_vec_init(&J_pair[i], 1);
        printf("%4d, %4d\n", U[i]->nrow, U[i]->ncol);
    }
}

// Build H2 generator matrices
// Input parameter:
//   h2eri : H2ERI structure with point partitioning & shell pair info
// Output parameter:
//   h2eri : H2ERI structure with H2 generator blocks
void H2ERI_build_B(H2ERI_t h2eri)
{
    
}

// Build dense blocks in the original matrices
// Input parameter:
//   h2eri : H2ERI structure with point partitioning & shell pair info
// Output parameter:
//   h2eri : H2ERI structure with H2 dense blocks
void H2ERI_build_D(H2ERI_t h2eri)
{
    H2Pack_t h2pack = h2eri->h2pack;
    int n_thread         = h2pack->n_thread;
    int n_point          = h2pack->n_point;
    int n_leaf_node      = h2pack->n_leaf_node;
    int n_r_inadm_pair   = h2pack->n_r_inadm_pair;
    int num_unc_sp       = h2eri->num_unc_sp;
    int *leaf_nodes      = h2pack->height_nodes;
    int *cluster         = h2pack->cluster;
    int *r_inadm_pairs   = h2pack->r_inadm_pairs;
    int *unc_sp_bfp_sidx = h2eri->unc_sp_bfp_sidx;
    int *index_seq       = h2eri->index_seq;
    H2P_int_vec_t D_blk0 = h2pack->D_blk0;
    H2P_int_vec_t D_blk1 = h2pack->D_blk1;
    multi_sp_t *unc_sp   = h2eri->unc_sp;
    
    // 1. Allocate D
    h2pack->n_D = n_leaf_node + n_r_inadm_pair;
    h2pack->D_nrow = (int*)    malloc(sizeof(int)    * h2pack->n_D);
    h2pack->D_ncol = (int*)    malloc(sizeof(int)    * h2pack->n_D);
    h2pack->D_ptr  = (size_t*) malloc(sizeof(size_t) * (h2pack->n_D + 1));
    int    *D_nrow = h2pack->D_nrow;
    int    *D_ncol = h2pack->D_ncol;
    size_t *D_ptr  = h2pack->D_ptr;
    assert(h2pack->D_nrow != NULL && h2pack->D_ncol != NULL && h2pack->D_ptr != NULL);
    
    // 2. Partition D matrices into multiple blocks s.t. each block has approximately
    //    the same total size of D matrices in a block
    D_ptr[0] = 0;
    size_t D0_total_size = 0;
    for (int i = 0; i < n_leaf_node; i++)
    {
        int node = leaf_nodes[i];
        int s_index = cluster[2 * node];
        int e_index = cluster[2 * node + 1];
        int node_nbfp = unc_sp_bfp_sidx[e_index + 1] - unc_sp_bfp_sidx[s_index];
        size_t Di_size = (size_t) node_nbfp * (size_t) node_nbfp;
        D_nrow[i] = node_nbfp;
        D_ncol[i] = node_nbfp;
        //Di_size = (Di_size + N_DTYPE_64B - 1) / N_DTYPE_64B * N_DTYPE_64B;
        D_ptr[i + 1] = Di_size;
        D0_total_size += Di_size;
        // Add statistic info
        h2pack->mat_size[6] += node_nbfp * node_nbfp;
        h2pack->mat_size[6] += node_nbfp + node_nbfp;
    }
    H2P_partition_workload(n_leaf_node, D_ptr + 1, D0_total_size, n_thread * BD_NTASK_THREAD, D_blk0);
    size_t D1_total_size = 0;
    for (int i = 0; i < n_r_inadm_pair; i++)
    {
        int node0 = r_inadm_pairs[2 * i];
        int node1 = r_inadm_pairs[2 * i + 1];
        int s_index0 = cluster[2 * node0];
        int s_index1 = cluster[2 * node1];
        int e_index0 = cluster[2 * node0 + 1];
        int e_index1 = cluster[2 * node1 + 1];
        int node0_nbfp = unc_sp_bfp_sidx[e_index0 + 1] - unc_sp_bfp_sidx[s_index0];
        int node1_nbfp = unc_sp_bfp_sidx[e_index1 + 1] - unc_sp_bfp_sidx[s_index1];
        size_t Di_size = (size_t) node0_nbfp * (size_t) node1_nbfp;
        D_nrow[i + n_leaf_node] = node0_nbfp;
        D_ncol[i + n_leaf_node] = node1_nbfp;
        //Di_size = (Di_size + N_DTYPE_64B - 1) / N_DTYPE_64B * N_DTYPE_64B;
        D_ptr[n_leaf_node + 1 + i] = Di_size;
        D1_total_size += Di_size;
        // Add statistic info
        h2pack->mat_size[6] += 2 * (node0_nbfp * node1_nbfp);
        h2pack->mat_size[6] += 2 * (node0_nbfp + node1_nbfp);
    }
    H2P_partition_workload(n_r_inadm_pair, D_ptr + n_leaf_node + 1, D1_total_size, n_thread * BD_NTASK_THREAD, D_blk1);
    for (int i = 1; i <= n_leaf_node + n_r_inadm_pair; i++) D_ptr[i] += D_ptr[i - 1];
    h2pack->mat_size[2] = D0_total_size + D1_total_size;
    
    h2pack->D_data = (DTYPE*) H2P_malloc_aligned(sizeof(DTYPE) * (D0_total_size + D1_total_size));
    assert(h2pack->D_data != NULL);
    DTYPE *D_data = h2pack->D_data;
    const int n_D0_blk = D_blk0->length;
    const int n_D1_blk = D_blk1->length;
    #pragma omp parallel num_threads(n_thread)
    {
        int tid = omp_get_thread_num();
        simint_buff_t buff = h2eri->simint_buffs[tid];
        
        h2pack->tb[tid]->timer = -H2P_get_wtime_sec();
        
        // 3. Generate diagonal blocks (leaf node self interaction)
        #pragma omp for schedule(dynamic) nowait
        for (int i_blk0 = 0; i_blk0 < n_D0_blk; i_blk0++)
        {
            int blk_s_index = D_blk0->data[i_blk0];
            int blk_e_index = D_blk0->data[i_blk0 + 1];
            for (int i = blk_s_index; i < blk_e_index; i++)
            {
                int node = leaf_nodes[i];
                int s_index = cluster[2 * node];
                int e_index = cluster[2 * node + 1];
                int node_npts = e_index - s_index + 1;
                int ld_Di = D_ncol[i];
                DTYPE *Di = D_data + D_ptr[i];
                int *bra_idx = index_seq + s_index;
                int *ket_idx = bra_idx;
                H2ERI_calc_ERI_pairs_to_mat(
                    unc_sp, node_npts, node_npts, 
                    bra_idx, ket_idx, buff, Di, ld_Di
                );
            }
        }  // End of i_blk0 loop
        
        // 4. Generate off-diagonal blocks from inadmissible pairs
        #pragma omp for schedule(dynamic) nowait
        for (int i_blk1 = 0; i_blk1 < n_D1_blk; i_blk1++)
        {
            int s_index = D_blk1->data[i_blk1];
            int e_index = D_blk1->data[i_blk1 + 1];
            for (int i = s_index; i < e_index; i++)
            {
                int node0 = r_inadm_pairs[2 * i];
                int node1 = r_inadm_pairs[2 * i + 1];
                int s_index0 = cluster[2 * node0];
                int s_index1 = cluster[2 * node1];
                int e_index0 = cluster[2 * node0 + 1];
                int e_index1 = cluster[2 * node1 + 1];
                int node0_npts = e_index0 - s_index0 + 1;
                int node1_npts = e_index1 - s_index1 + 1;
                int ld_Di = D_ncol[i + n_leaf_node];
                DTYPE *Di = D_data + D_ptr[i + n_leaf_node];
                int *bra_idx = index_seq + s_index0;
                int *ket_idx = index_seq + s_index1;
                H2ERI_calc_ERI_pairs_to_mat(
                    unc_sp, node0_npts, node1_npts, 
                    bra_idx, ket_idx, buff, Di, ld_Di
                );
            }
        }  // End of i_blk1 loop
        
        h2pack->tb[tid]->timer += H2P_get_wtime_sec();
    }  // End of "pragma omp parallel"
    
    //FILE *ouf = fopen("D.bin", "wb");
    //fwrite(D_data, sizeof(double), h2pack->mat_size[2], ouf);
    //fclose(ouf);
    //printf("Save D results to file done\n");
    
    #ifdef PROFILING_OUTPUT
    double max_t = 0.0, avg_t = 0.0, min_t = 1145141919.0;
    for (int i = 0; i < n_thread; i++)
    {
        double thread_i_timer = h2pack->tb[i]->timer;
        avg_t += thread_i_timer;
        max_t = MAX(max_t, thread_i_timer);
        min_t = MIN(min_t, thread_i_timer);
    }
    avg_t /= (double) n_thread;
    printf("[PROFILING] Build D: min/avg/max thread wall-time = %.3lf, %.3lf, %.3lf (s)\n", min_t, avg_t, max_t);
    #endif
}

// Build H2 representation for ERI tensor
void H2ERI_build(H2ERI_t h2eri)
{
    double st, et;

    // 1. Build projection matrices and skeleton row sets
    st = H2P_get_wtime_sec();
    H2ERI_build_UJ_proxy(h2eri);
    et = H2P_get_wtime_sec();
    h2eri->h2pack->timers[1] = et - st;

    // 2. Build generator matrices
    st = H2P_get_wtime_sec();
    H2ERI_build_B(h2eri);
    et = H2P_get_wtime_sec();
    h2eri->h2pack->timers[2] = et - st;
    
    // 3. Build dense blocks
    st = H2P_get_wtime_sec();
    H2ERI_build_D(h2eri);
    et = H2P_get_wtime_sec();
    h2eri->h2pack->timers[3] = et - st;
}

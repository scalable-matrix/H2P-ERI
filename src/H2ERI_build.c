#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

#include "CMS.h"
#include "H2ERI_typedef.h"
#include "H2Pack_aux_structs.h"

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


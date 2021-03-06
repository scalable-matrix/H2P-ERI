#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>

#include <omp.h>

#include "H2Pack_matvec.h"
#include "H2Pack_utils.h"
#include "H2ERI_typedef.h"
#include "H2ERI_matvec.h"
#include "H2ERI_utils.h"
#include "utils.h"  // In H2Pack

// Perform bi-matvec for a B or D block blk which might be 
// a dense block or a low-rank approximation blk = U * V.
// Input parameters:
//   blk    : Target B or D block
//   tmp_v  : Auxiliary vector for storing the intermediate result 
//   x_in_0 : Array, size blk->ncol, will be multiplied by blk
//   x_in_1 : Array, size blk->nrow, will be multiplied by blk^T
// Output parameters:
//   x_out_0 : Array, size blk->nrow, == blk   * x_in_0
//   x_out_1 : Array, size blk->ncol, == blk^T * x_in_1
void H2ERI_BD_blk_bi_matvec(
    H2P_dense_mat_p blk, H2P_dense_mat_p tmp_v,
    const double *x_in_0, const double *x_in_1, 
    double *x_out_0, double *x_out_1
)
{
    if (blk->ld > 0)
    {
        CBLAS_BI_GEMV(
            blk->nrow, blk->ncol, blk->data, blk->ncol,
            x_in_0, x_in_1, x_out_0, x_out_1
        );
    } else {
        // U: blk->nrow * blk_rank
        // V: blk_rank  * blk->ncol
        // Note: V^T instead of V is stored
        int    blk_rank = -blk->ld;
        double *U_mat   = blk->data;
        double *VT_mat  = U_mat + blk->nrow * blk_rank;
        H2P_dense_mat_resize(tmp_v, blk_rank, 1);
        // x_out_0 = (U * V) * x_in_0 = U * V * x_in_0
        CBLAS_GEMV(
            CblasRowMajor, CblasTrans, blk->ncol, blk_rank, 
            1.0, VT_mat, blk_rank, x_in_0, 1, 0.0, tmp_v->data, 1
        );
        CBLAS_GEMV(
            CblasRowMajor, CblasNoTrans, blk->nrow, blk_rank,
            1.0, U_mat, blk_rank, tmp_v->data, 1, 1.0, x_out_0, 1
        );
        // x_out_1 = (U * V)^T * x_in_1 = V^T * U^T * x_in_1
        CBLAS_GEMV(
            CblasRowMajor, CblasTrans, blk->nrow, blk_rank,
            1.0, U_mat, blk_rank, x_in_1, 1, 0.0, tmp_v->data, 1
        );
        CBLAS_GEMV(
            CblasRowMajor, CblasNoTrans, blk->ncol, blk_rank, 
            1.0, VT_mat, blk_rank, tmp_v->data, 1, 1.0, x_out_1, 1
        );
    }
}

// H2 matvec intermediate multiplication for H2ERI
// All B_{ij} matrices are calculated and stored
void H2ERI_matvec_intmd_mult_AOT(H2ERI_p h2eri, const double *x)
{
    H2Pack_p h2pack  = h2eri->h2pack;
    int n_node       = h2pack->n_node;
    int n_thread     = h2pack->n_thread;
    int *r_adm_pairs = h2pack->r_adm_pairs;
    int *node_level  = h2pack->node_level;
    int *mat_cluster = h2pack->mat_cluster;
    H2P_int_vec_p   B_blk        = h2pack->B_blk;
    H2P_dense_mat_p *y0          = h2pack->y0;
    H2P_thread_buf_p *thread_buf = h2pack->tb;
    H2P_dense_mat_p  *c_B_blks   = h2eri->c_B_blks;
    
    // 1. Initialize y1 
    H2P_matvec_init_y1(h2pack);
    H2P_dense_mat_p *y1 = h2pack->y1;
    
    // 2. Intermediate multiplication
    const int n_B_blk = B_blk->length - 1;
    #pragma omp parallel num_threads(n_thread)
    {
        int tid = omp_get_thread_num();
        double *y = thread_buf[tid]->y;
        thread_buf[tid]->timer = -get_wtime_sec();
        
        H2P_dense_mat_p tmp_v = thread_buf[tid]->mat0;
        
        #pragma omp for schedule(dynamic) nowait
        for (int i_blk = 0; i_blk < n_B_blk; i_blk++)
        {
            int B_blk_s = B_blk->data[i_blk];
            int B_blk_e = B_blk->data[i_blk + 1];
            for (int i = B_blk_s; i < B_blk_e; i++)
            {
                int node0  = r_adm_pairs[2 * i];
                int node1  = r_adm_pairs[2 * i + 1];
                int level0 = node_level[node0];
                int level1 = node_level[node1];
                
                H2P_dense_mat_p Bi = c_B_blks[i];
                
                // (1) Two nodes are of the same level, compress on both sides
                if (level0 == level1)
                {
                    int ncol0 = y1[node0]->ncol;
                    int ncol1 = y1[node1]->ncol;
                    double *y1_dst_0 = y1[node0]->data + tid * ncol0;
                    double *y1_dst_1 = y1[node1]->data + tid * ncol1;

                    const double *x_in_0 = y0[node1]->data;
                    const double *x_in_1 = y0[node0]->data;
                    double *x_out_0 = y1_dst_0;
                    double *x_out_1 = y1_dst_1;
                    H2ERI_BD_blk_bi_matvec(Bi, tmp_v, x_in_0, x_in_1, x_out_0, x_out_1);
                }
                
                // (2) node1 is a leaf node and its level is higher than node0's level, 
                //     only compress on node0's side
                if (level0 > level1)
                {
                    int vec_s1 = mat_cluster[node1 * 2];
                    double       *y_spos = y + vec_s1;
                    const double *x_spos = x + vec_s1;
                    int ncol0        = y1[node0]->ncol;
                    double *y1_dst_0 = y1[node0]->data + tid * ncol0;

                    const double *x_in_0 = x_spos;
                    const double *x_in_1 = y0[node0]->data;
                    double *x_out_0 = y1_dst_0;
                    double *x_out_1 = y_spos;
                    H2ERI_BD_blk_bi_matvec(Bi, tmp_v, x_in_0, x_in_1, x_out_0, x_out_1);
                }
                
                // (3) node0 is a leaf node and its level is higher than node1's level, 
                //     only compress on node1's side
                if (level0 < level1)
                {
                    int vec_s0 = mat_cluster[node0 * 2];
                    double       *y_spos = y + vec_s0;
                    const double *x_spos = x + vec_s0;
                    int ncol1        = y1[node1]->ncol;
                    double *y1_dst_1 = y1[node1]->data + tid * ncol1;

                    const double *x_in_0 = y0[node1]->data;
                    const double *x_in_1 = x_spos;
                    double *x_out_0 = y_spos;
                    double *x_out_1 = y1_dst_1;
                    H2ERI_BD_blk_bi_matvec(Bi, tmp_v, x_in_0, x_in_1, x_out_0, x_out_1);
                }
            }  // End of i loop
        }  // End of i_blk loop
        thread_buf[tid]->timer += get_wtime_sec();
    }  // End of "pragma omp parallel"
    
    // 3. Sum thread-local buffers in y1
    H2P_matvec_sum_y1_thread(h2pack);
    
    #ifdef PROFILING_OUTPUT
    double max_t = 0.0, avg_t = 0.0, min_t = 19241112.0;
    for (int i = 0; i < n_thread; i++)
    {
        double thread_i_timer = thread_buf[i]->timer;
        avg_t += thread_i_timer;
        max_t = MAX(max_t, thread_i_timer);
        min_t = MIN(min_t, thread_i_timer);
    }
    avg_t /= (double) n_thread;
    printf("[PROFILING] MatVec intermediate sweep: min/avg/max thread wall-time = %.3lf, %.3lf, %.3lf (s)\n", min_t, avg_t, max_t);
    #endif
}

// H2 matvec intermediate multiplication for H2ERI
// Need to calculate all B_{ij} matrices before using it
void H2ERI_matvec_intmd_mult_JIT(H2ERI_p h2eri, const double *x)
{
    H2Pack_p h2pack = h2eri->h2pack;
    int n_node            = h2pack->n_node;
    int n_thread          = h2pack->n_thread;
    int *r_adm_pairs      = h2pack->r_adm_pairs;
    int *node_level       = h2pack->node_level;
    int *pt_cluster       = h2pack->pt_cluster;
    int *mat_cluster      = h2pack->mat_cluster;
    int *sp_nbfp          = h2eri->sp_nbfp;
    int *index_seq        = h2eri->index_seq;
    int *B_nrow           = h2pack->B_nrow;
    int *B_ncol           = h2pack->B_ncol;
    multi_sp_t    *sp     = h2eri->sp;
    H2P_int_vec_p B_blk   = h2pack->B_blk;
    H2P_int_vec_p *J_pair = h2eri->J_pair;
    H2P_int_vec_p *J_row  = h2eri->J_row;
    H2P_dense_mat_p *y0   = h2pack->y0;
    H2P_thread_buf_p *thread_buf      = h2pack->tb;
    simint_buff_p    *simint_buffs    = h2eri->simint_buffs;
    eri_batch_buff_p *eri_batch_buffs = h2eri->eri_batch_buffs;
    
    // 1. Initialize y1 
    H2P_matvec_init_y1(h2pack);
    H2P_dense_mat_p *y1 = h2pack->y1;
    
    // 2. Intermediate multiplication
    const int n_B_blk = B_blk->length - 1;
    #pragma omp parallel num_threads(n_thread)
    {
        int tid = omp_get_thread_num();
        
        H2P_dense_mat_p  tmpB           = thread_buf[tid]->mat0;
        simint_buff_p    simint_buff    = simint_buffs[tid];
        eri_batch_buff_p eri_batch_buff = eri_batch_buffs[tid];
        
        double *y = thread_buf[tid]->y;
        
        thread_buf[tid]->timer = -get_wtime_sec();
        
        #pragma omp for schedule(dynamic) nowait
        for (int i_blk = 0; i_blk < n_B_blk; i_blk++)
        {
            int B_blk_s = B_blk->data[i_blk];
            int B_blk_e = B_blk->data[i_blk + 1];
            for (int i = B_blk_s; i < B_blk_e; i++)
            {
                int node0  = r_adm_pairs[2 * i];
                int node1  = r_adm_pairs[2 * i + 1];
                int level0 = node_level[node0];
                int level1 = node_level[node1];
                
                // (1) Two nodes are of the same level, compress on both sides
                if (level0 == level1)
                {
                    int tmpB_nrow  = H2ERI_gather_sum_int(sp_nbfp, J_pair[node0]->length, J_pair[node0]->data);
                    int tmpB_ncol  = H2ERI_gather_sum_int(sp_nbfp, J_pair[node1]->length, J_pair[node1]->data);
                    int n_bra_pair = J_pair[node0]->length;
                    int n_ket_pair = J_pair[node1]->length;
                    int *bra_idx   = J_pair[node0]->data;
                    int *ket_idx   = J_pair[node1]->data;
                    H2P_dense_mat_resize(tmpB, tmpB_nrow, tmpB_ncol);
                    H2ERI_calc_ERI_pairs_to_mat(
                        sp, n_bra_pair, n_ket_pair, bra_idx, ket_idx, 
                        simint_buff, tmpB->data, tmpB->ncol, eri_batch_buff
                    );
                    H2P_dense_mat_select_rows   (tmpB, J_row[node0]);
                    H2P_dense_mat_select_columns(tmpB, J_row[node1]);
                    
                    int ncol0 = y1[node0]->ncol;
                    int ncol1 = y1[node1]->ncol;
                    double *y1_dst_0 = y1[node0]->data + tid * ncol0;
                    double *y1_dst_1 = y1[node1]->data + tid * ncol1;
                    CBLAS_BI_GEMV(
                        tmpB->nrow, tmpB->ncol, tmpB->data, tmpB->ncol,
                        y0[node1]->data, y0[node0]->data, y1_dst_0, y1_dst_1
                    );
                }
                
                // (2) node1 is a leaf node and its level is higher than node0's level, 
                //     only compress on node0's side
                if (level0 > level1)
                {
                    int tmpB_nrow  = H2ERI_gather_sum_int(sp_nbfp, J_pair[node0]->length, J_pair[node0]->data);
                    int tmpB_ncol  = B_ncol[i];
                    int pt_s1      = pt_cluster[2 * node1];
                    int pt_e1      = pt_cluster[2 * node1 + 1];
                    int n_bra_pair = J_pair[node0]->length;
                    int n_ket_pair = pt_e1 - pt_s1 + 1;
                    int *bra_idx   = J_pair[node0]->data;
                    int *ket_idx   = index_seq + pt_s1;
                    H2P_dense_mat_resize(tmpB, tmpB_nrow, tmpB_ncol);
                    H2ERI_calc_ERI_pairs_to_mat(
                        sp, n_bra_pair, n_ket_pair, bra_idx, ket_idx, 
                        simint_buff, tmpB->data, tmpB->ncol, eri_batch_buff
                    );
                    H2P_dense_mat_select_rows(tmpB, J_row[node0]);
                    
                    int vec_s1 = mat_cluster[node1 * 2];
                    double       *y_spos = y + vec_s1;
                    const double *x_spos = x + vec_s1;
                    int ncol0        = y1[node0]->ncol;
                    double *y1_dst_0 = y1[node0]->data + tid * ncol0;
                    CBLAS_BI_GEMV(
                        tmpB->nrow, tmpB->ncol, tmpB->data, tmpB->ncol,
                        x_spos, y0[node0]->data, y1_dst_0, y_spos
                    );
                }
                
                // (3) node0 is a leaf node and its level is higher than node1's level, 
                //     only compress on node1's side
                if (level0 < level1)
                {
                    int tmpB_nrow  = B_nrow[i];
                    int tmpB_ncol  = H2ERI_gather_sum_int(sp_nbfp, J_pair[node1]->length, J_pair[node1]->data);
                    int pt_s0      = pt_cluster[2 * node0];
                    int pt_e0      = pt_cluster[2 * node0 + 1];
                    int n_bra_pair = pt_e0 - pt_s0 + 1;
                    int n_ket_pair = J_pair[node1]->length;
                    int *bra_idx   = index_seq + pt_s0;
                    int *ket_idx   = J_pair[node1]->data;
                    H2P_dense_mat_resize(tmpB, tmpB_nrow, tmpB_ncol);
                    H2ERI_calc_ERI_pairs_to_mat(
                        sp, n_bra_pair, n_ket_pair, bra_idx, ket_idx, 
                        simint_buff, tmpB->data, tmpB->ncol, eri_batch_buff
                    );
                    H2P_dense_mat_select_columns(tmpB, J_row[node1]);
                    
                    int vec_s0 = mat_cluster[node0 * 2];
                    double       *y_spos = y + vec_s0;
                    const double *x_spos = x + vec_s0;
                    int ncol1        = y1[node1]->ncol;
                    double *y1_dst_1 = y1[node1]->data + tid * ncol1;
                    CBLAS_BI_GEMV(
                        tmpB->nrow, tmpB->ncol, tmpB->data, tmpB->ncol,
                        y0[node1]->data, x_spos, y_spos, y1_dst_1
                    );
                }
            }  // End of i loop
        }  // End of i_blk loop
        thread_buf[tid]->timer += get_wtime_sec();
    }  // End of "pragma omp parallel"
    
    // 3. Sum thread-local buffers in y1
    H2P_matvec_sum_y1_thread(h2pack);
    
    #ifdef PROFILING_OUTPUT
    double max_t = 0.0, avg_t = 0.0, min_t = 19241112.0;
    for (int i = 0; i < n_thread; i++)
    {
        double thread_i_timer = thread_buf[i]->timer;
        avg_t += thread_i_timer;
        max_t = MAX(max_t, thread_i_timer);
        min_t = MIN(min_t, thread_i_timer);
    }
    avg_t /= (double) n_thread;
    printf("[PROFILING] MatVec intermediate sweep: min/avg/max thread wall-time = %.3lf, %.3lf, %.3lf (s)\n", min_t, avg_t, max_t);
    #endif
}

// H2 matvec dense multiplication for H2ERI
// All D_{ij} matrices are calculated and stored
void H2ERI_matvec_dense_mult_AOT(H2ERI_p h2eri, const double *x)
{
    H2Pack_p h2pack    = h2eri->h2pack;
    int n_thread       = h2pack->n_thread;
    int n_leaf_node    = h2pack->n_leaf_node;
    int *leaf_nodes    = h2pack->height_nodes;
    int *mat_cluster   = h2pack->mat_cluster;
    int *r_inadm_pairs = h2pack->r_inadm_pairs;
    H2P_int_vec_p    D_blk0      = h2pack->D_blk0;
    H2P_int_vec_p    D_blk1      = h2pack->D_blk1;
    H2P_thread_buf_p *thread_buf = h2pack->tb;
    H2P_dense_mat_p  *c_D_blks   = h2eri->c_D_blks;
    
    const int n_D0_blk = D_blk0->length - 1;
    const int n_D1_blk = D_blk1->length - 1;
    #pragma omp parallel num_threads(n_thread)
    {
        int tid = omp_get_thread_num();
        
        double *y = thread_buf[tid]->y;
        H2P_dense_mat_p tmp_v  = thread_buf[tid]->mat0;
        H2P_dense_mat_p tmp_v2 = thread_buf[tid]->mat1;

        thread_buf[tid]->timer = -get_wtime_sec();
        
        // 1. Diagonal blocks (leaf node self interaction)
        #pragma omp for schedule(dynamic) nowait
        for (int i_blk0 = 0; i_blk0 < n_D0_blk; i_blk0++)
        {
            int D_blk0_s = D_blk0->data[i_blk0];
            int D_blk0_e = D_blk0->data[i_blk0 + 1];
            for (int i = D_blk0_s; i < D_blk0_e; i++)
            {
                H2P_dense_mat_p Di = c_D_blks[i];
                int node  = leaf_nodes[i];
                int vec_s = mat_cluster[node * 2];
                double       *y_spos = y + vec_s;
                const double *x_spos = x + vec_s;

                H2P_dense_mat_resize(tmp_v2, 1, Di->nrow + Di->ncol);
                const double *x_in_0 = x_spos;
                const double *x_in_1 = tmp_v2->data;
                double *x_out_0 = y_spos;
                double *x_out_1 = tmp_v2->data + Di->ncol;
                H2ERI_BD_blk_bi_matvec(Di, tmp_v, x_in_0, x_in_1, x_out_0, x_out_1);
            }
        }  // End of i_blk0 loop
        
        // 2. Off-diagonal blocks from inadmissible pairs
        #pragma omp for schedule(dynamic) nowait
        for (int i_blk1 = 0; i_blk1 < n_D1_blk; i_blk1++)
        {
            int D_blk1_s = D_blk1->data[i_blk1];
            int D_blk1_e = D_blk1->data[i_blk1 + 1];
            for (int i = D_blk1_s; i < D_blk1_e; i++)
            {
                H2P_dense_mat_p Di = c_D_blks[i + n_leaf_node];
                int node0  = r_inadm_pairs[2 * i];
                int node1  = r_inadm_pairs[2 * i + 1];
                int vec_s0 = mat_cluster[2 * node0];
                int vec_s1 = mat_cluster[2 * node1];
                double       *y_spos0 = y + vec_s0;
                double       *y_spos1 = y + vec_s1;
                const double *x_spos0 = x + vec_s0;
                const double *x_spos1 = x + vec_s1;

                const double *x_in_0 = x_spos1;
                const double *x_in_1 = x_spos0;
                double *x_out_0 = y_spos0;
                double *x_out_1 = y_spos1;
                H2ERI_BD_blk_bi_matvec(Di, tmp_v, x_in_0, x_in_1, x_out_0, x_out_1);
            }
        }  // End of i_blk1 loop
        
        thread_buf[tid]->timer += get_wtime_sec();
    }  // End of "pragma omp parallel"
    
    #ifdef PROFILING_OUTPUT
    double max_t = 0.0, avg_t = 0.0, min_t = 19241112.0;
    for (int i = 0; i < n_thread; i++)
    {
        double thread_i_timer = thread_buf[i]->timer;
        avg_t += thread_i_timer;
        max_t = MAX(max_t, thread_i_timer);
        min_t = MIN(min_t, thread_i_timer);
    }
    avg_t /= (double) n_thread;
    printf("[PROFILING] MatVec dense block sweep: min/avg/max thread wall-time = %.3lf, %.3lf, %.3lf (s)\n", min_t, avg_t, max_t);
    #endif
}

// H2 matvec dense multiplication for H2ERI
// Need to calculate all D_{ij} matrices before using it
void H2ERI_matvec_dense_mult_JIT(H2ERI_p h2eri, const double *x)
{
    H2Pack_p h2pack = h2eri->h2pack;
    int n_thread         = h2pack->n_thread;
    int n_leaf_node      = h2pack->n_leaf_node;
    int *pt_cluster      = h2pack->pt_cluster;
    int *leaf_nodes      = h2pack->height_nodes;
    int *mat_cluster     = h2pack->mat_cluster;
    int *r_inadm_pairs   = h2pack->r_inadm_pairs;
    int *D_nrow          = h2pack->D_nrow;
    int *D_ncol          = h2pack->D_ncol;
    int *index_seq       = h2eri->index_seq;
    H2P_int_vec_p D_blk0 = h2pack->D_blk0;
    H2P_int_vec_p D_blk1 = h2pack->D_blk1;
    multi_sp_t *sp       = h2eri->sp;
    H2P_thread_buf_p *thread_buf      = h2pack->tb;
    simint_buff_p    *simint_buffs    = h2eri->simint_buffs;
    eri_batch_buff_p *eri_batch_buffs = h2eri->eri_batch_buffs;
    
    const int n_D0_blk = D_blk0->length - 1;
    const int n_D1_blk = D_blk1->length - 1;
    #pragma omp parallel num_threads(n_thread)
    {
        int tid = omp_get_thread_num();
        
        H2P_dense_mat_p  tmpD           = thread_buf[tid]->mat0;
        simint_buff_p    simint_buff    = simint_buffs[tid];
        eri_batch_buff_p eri_batch_buff = eri_batch_buffs[tid];
        
        double *y = thread_buf[tid]->y;
        
        thread_buf[tid]->timer = -get_wtime_sec();
        
        // 1. Diagonal blocks (leaf node self interaction)
        #pragma omp for schedule(dynamic) nowait
        for (int i_blk0 = 0; i_blk0 < n_D0_blk; i_blk0++)
        {
            int D_blk0_s = D_blk0->data[i_blk0];
            int D_blk0_e = D_blk0->data[i_blk0 + 1];
            for (int i = D_blk0_s; i < D_blk0_e; i++)
            {
                int node      = leaf_nodes[i];
                int pt_s      = pt_cluster[2 * node];
                int pt_e      = pt_cluster[2 * node + 1];
                int node_npts = pt_e - pt_s + 1;
                int Di_nrow   = D_nrow[i];
                int Di_ncol   = D_ncol[i];
                int *bra_idx  = index_seq + pt_s;
                int *ket_idx  = bra_idx;
                H2P_dense_mat_resize(tmpD, Di_nrow, Di_ncol);
                H2ERI_calc_ERI_pairs_to_mat(
                    sp, node_npts, node_npts, bra_idx, ket_idx, 
                    simint_buff, tmpD->data, Di_ncol, eri_batch_buff
                );
                
                int vec_s = mat_cluster[node * 2];
                double       *y_spos = y + vec_s;
                const double *x_spos = x + vec_s;
                CBLAS_GEMV(
                    CblasRowMajor, CblasNoTrans, Di_nrow, Di_ncol,
                    1.0, tmpD->data, Di_ncol, x_spos, 1, 1.0, y_spos, 1
                );
            }
        }  // End of i_blk0 loop
        
        // 2. Off-diagonal blocks from inadmissible pairs
        #pragma omp for schedule(dynamic) nowait
        for (int i_blk1 = 0; i_blk1 < n_D1_blk; i_blk1++)
        {
            int D_blk1_s = D_blk1->data[i_blk1];
            int D_blk1_e = D_blk1->data[i_blk1 + 1];
            for (int i = D_blk1_s; i < D_blk1_e; i++)
            {
                int node0      = r_inadm_pairs[2 * i];
                int node1      = r_inadm_pairs[2 * i + 1];
                int pt_s0      = pt_cluster[2 * node0];
                int pt_s1      = pt_cluster[2 * node1];
                int pt_e0      = pt_cluster[2 * node0 + 1];
                int pt_e1      = pt_cluster[2 * node1 + 1];
                int node0_npts = pt_e0 - pt_s0 + 1;
                int node1_npts = pt_e1 - pt_s1 + 1;
                int Di_nrow    = D_nrow[i + n_leaf_node];
                int Di_ncol    = D_ncol[i + n_leaf_node];
                int *bra_idx   = index_seq + pt_s0;
                int *ket_idx   = index_seq + pt_s1;
                H2P_dense_mat_resize(tmpD, Di_nrow, Di_ncol);
                H2ERI_calc_ERI_pairs_to_mat(
                    sp, node0_npts, node1_npts, bra_idx, ket_idx, 
                    simint_buff, tmpD->data, Di_ncol, eri_batch_buff
                );
                
                int vec_s0 = mat_cluster[2 * node0];
                int vec_s1 = mat_cluster[2 * node1];
                double       *y_spos0 = y + vec_s0;
                double       *y_spos1 = y + vec_s1;
                const double *x_spos0 = x + vec_s0;
                const double *x_spos1 = x + vec_s1;
                CBLAS_BI_GEMV(
                    Di_nrow, Di_ncol, tmpD->data, Di_ncol,
                    x_spos1, x_spos0, y_spos0, y_spos1
                );
            }
        }  // End of i_blk1 loop
        
        thread_buf[tid]->timer += get_wtime_sec();
    }  // End of "pragma omp parallel"
    
    #ifdef PROFILING_OUTPUT
    double max_t = 0.0, avg_t = 0.0, min_t = 19241112.0;
    for (int i = 0; i < n_thread; i++)
    {
        double thread_i_timer = thread_buf[i]->timer;
        avg_t += thread_i_timer;
        max_t = MAX(max_t, thread_i_timer);
        min_t = MIN(min_t, thread_i_timer);
    }
    avg_t /= (double) n_thread;
    printf("[PROFILING] MatVec dense block sweep: min/avg/max thread wall-time = %.3lf, %.3lf, %.3lf (s)\n", min_t, avg_t, max_t);
    #endif
}

// H2 representation for an ERI tensor multiplies a column vector
void H2ERI_matvec(H2ERI_p h2eri, const double *x, double *y)
{
    H2Pack_p h2pack   = h2eri->h2pack;
    int krnl_mat_size = h2pack->krnl_mat_size;
    int n_thread      = h2pack->n_thread;
    H2P_thread_buf_p *thread_buf = h2pack->tb;
    double st, et;
    
    // 1. Reset partial y result in each thread-local buffer to 0
    st = get_wtime_sec();
    #pragma omp parallel num_threads(n_thread)
    {
        int tid = omp_get_thread_num();
        double *tid_y = thread_buf[tid]->y;
        memset(tid_y, 0, sizeof(double) * krnl_mat_size);
        
        #pragma omp for
        for (int i = 0; i < krnl_mat_size; i++) y[i] = 0;
    }
    et = get_wtime_sec();
    h2pack->timers[MV_VOP_TIMER_IDX] += et - st;
    
    // 2. Forward transformation, calculate U_j^T * x_j
    st = get_wtime_sec();
    H2P_matvec_fwd_transform(h2pack, x);
    et = get_wtime_sec();
    h2pack->timers[MV_FWD_TIMER_IDX] += et - st;
    
    // 3. Intermediate multiplication, calculate B_{ij} * (U_j^T * x_j)
    st = get_wtime_sec();
    if (h2eri->h2pack->BD_JIT == 0)
    {
        H2ERI_matvec_intmd_mult_AOT(h2eri, x);
    } else {
        H2ERI_matvec_intmd_mult_JIT(h2eri, x);
    }
    et = get_wtime_sec();
    h2pack->timers[MV_MID_TIMER_IDX] += et - st;
    
    // 4. Backward transformation, calculate U_i * (B_{ij} * (U_j^T * x_j))
    st = get_wtime_sec();
    H2P_matvec_bwd_transform(h2pack, x, y);
    et = get_wtime_sec();
    h2pack->timers[MV_BWD_TIMER_IDX] += et - st;
    
    // 5. Dense multiplication, calculate D_i * x_i
    st = get_wtime_sec();
    if (h2eri->h2pack->BD_JIT == 0)
    {
        H2ERI_matvec_dense_mult_AOT(h2eri, x);
    } else {
        H2ERI_matvec_dense_mult_JIT(h2eri, x);
    }
    et = get_wtime_sec();
    h2pack->timers[MV_DEN_TIMER_IDX] += et - st;
    
    // 6. Reduce sum partial y results
    st = get_wtime_sec();
    #pragma omp parallel num_threads(n_thread)
    {
        int tid = omp_get_thread_num();
        int spos, len;
        calc_block_spos_len(krnl_mat_size, n_thread, tid, &spos, &len);
        
        for (int tid = 0; tid < n_thread; tid++)
        {
            DTYPE *y_src = thread_buf[tid]->y;
            #pragma omp simd
            for (int i = spos; i < spos + len; i++) y[i] += y_src[i];
        }
    }
    h2pack->mat_size[MV_VOP_SIZE_IDX] = (2 * n_thread + 1) * h2pack->krnl_mat_size;
    et = get_wtime_sec();
    h2pack->timers[MV_VOP_TIMER_IDX] += et - st;
    
    h2pack->n_matvec++;
}

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>

#include <omp.h>

#include "H2Pack_matvec.h"
#include "H2ERI_typedef.h"
#include "utils.h"  // In H2Pack

// These five external functions are in H2Pack_matvec.c, but not exposed in H2Pack_matvec.h
extern void CBLAS_BI_GEMV(
    const int nrow, const int ncol, const double *mat, const int ldm,
    const double *x_in_0, const double *x_in_1, double *x_out_0, double *x_out_1
);
extern void H2P_matvec_init_y1(H2Pack_t h2pack);
extern void H2P_matvec_sum_y1_thread(H2Pack_t h2pack);
extern void H2P_matvec_fwd_transform(H2Pack_t h2pack, const double *x, double *y);
extern void H2P_matvec_bwd_transform(H2Pack_t h2pack, const double *x, double *y);

// This external function is in H2ERI_build_H2.c, but not exposed in H2ERI_build_H2.h
extern int H2ERI_gather_sum(const int *arr, H2P_int_vec_t idx);

// "Uncontract" the density matrix according to SSP and unroll 
// the result to a column for H2 matvec.
// Input parameters:
//   den_mat              : Symmetric density matrix, size h2eri->num_bf^2
//   h2eri->num_bf        : Number of basis functions in the system
//   h2eri->num_sp        : Number of screened shell pairs (SSP)
//   h2eri->shell_bf_sidx : Array, size nshell, indices of each shell's 
//                          first basis function
//   h2eri->sp_bfp_sidx   : Array, size num_sp+1, indices of each 
//                          SSP's first basis function pair
//   h2eri->sp_shell_idx  : Array, size 2 * num_sp, each row is 
//                          the contracted shell indices of a SSP
// Output parameter:
//   h2eri->unc_denmat_x  : Array, size num_sp_bfp, uncontracted density matrix
void H2ERI_uncontract_den_mat(H2ERI_t h2eri, const double *den_mat)
{
    int num_bf = h2eri->num_bf;
    int num_sp = h2eri->num_sp;
    int *shell_bf_sidx = h2eri->shell_bf_sidx;
    int *sp_bfp_sidx   = h2eri->sp_bfp_sidx;
    int *sp_shell_idx  = h2eri->sp_shell_idx;
    double *x = h2eri->unc_denmat_x;
    
    #pragma omp parallel for schedule(dynamic, 16)
    for (int i = 0; i < num_sp; i++)
    {
        int x_spos = sp_bfp_sidx[i];
        int shell_idx0 = sp_shell_idx[i];
        int shell_idx1 = sp_shell_idx[i + num_sp];
        int srow = shell_bf_sidx[shell_idx0];
        int erow = shell_bf_sidx[shell_idx0 + 1];
        int scol = shell_bf_sidx[shell_idx1];
        int ecol = shell_bf_sidx[shell_idx1 + 1];
        int nrow = erow - srow;
        int ncol = ecol - scol;
        double sym_coef = (shell_idx0 == shell_idx1) ? 1.0 : 2.0;
        
        // Originally we need to store den_mat[srow:erow-1, scol:ecol-1]
        // column by column to x(x_spos:x_epos-1). Since den_mat is 
        // symmetric, we store den_mat[scol:ecol-1, srow:erow-1] row by 
        // row to x(x_spos:x_epos-1).
        for (int j = 0; j < ncol; j++)
        {
            const double *den_mat_ptr = den_mat + (scol + j) * num_bf + srow;
            double *x_ptr = x + x_spos + j * nrow;
            #pragma omp simd 
            for (int k = 0; k < nrow; k++)
                x_ptr[k] = sym_coef * den_mat_ptr[k];
        }
    }
}

// "Contract" the H2 matvec result according to SSP and reshape
// the result to form a symmetric Coulomb matrix
// Input parameters:
//   h2eri->num_bf        : Number of basis functions in the system
//   h2eri->num_sp        : Number of SSP
//   h2eri->shell_bf_sidx : Array, size nshell, indices of each shell's 
//                          first basis function
//   h2eri->sp_bfp_sidx   : Array, size num_sp+1, indices of each 
//                          SSP's first basis function pair
//   h2eri->sp_shell_idx  : Array, size 2 * num_sp, each row is 
//                          the contracted shell indices of a SSP
//   h2eri->H2_matvec_y   : Array, size num_sp_bfp, H2 matvec result 
// Output parameter:
//   J_mat : Symmetric Coulomb matrix, size h2eri->num_bf^2
void H2ERI_contract_H2_matvec(H2ERI_t h2eri, double *J_mat)
{
    int num_bf = h2eri->num_bf;
    int num_sp = h2eri->num_sp;
    int *shell_bf_sidx = h2eri->shell_bf_sidx;
    int *sp_bfp_sidx   = h2eri->sp_bfp_sidx;
    int *sp_shell_idx  = h2eri->sp_shell_idx;
    double *y = h2eri->H2_matvec_y;
    
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < num_bf * num_bf; i++) J_mat[i] = 0.0;
    
    #pragma omp parallel for schedule(dynamic, 16)
    for (int i = 0; i < num_sp; i++)
    {
        int y_spos = sp_bfp_sidx[i];
        int shell_idx0 = sp_shell_idx[i];
        int shell_idx1 = sp_shell_idx[i + num_sp];
        int srow = shell_bf_sidx[shell_idx0];
        int erow = shell_bf_sidx[shell_idx0 + 1];
        int scol = shell_bf_sidx[shell_idx1];
        int ecol = shell_bf_sidx[shell_idx1 + 1];
        int nrow = erow - srow;
        int ncol = ecol - scol;
        double sym_coef = (shell_idx0 == shell_idx1) ? 0.5 : 1.0;
        
        // Originally we need to reshape y(y_spos:y_epos-1) as a
        // nrow-by-ncol column-major matrix and add it to column-major
        // matrix J_mat[srow:erow-1, scol:ecol-1]. Since J_mat is 
        // symmetric, we reshape y(y_spos:y_epos-1) as a ncol-by-nrow
        // row-major matrix and add it to J_mat[scol:ecol-1, srow:erow-1].
        for (int j = 0; j < ncol; j++)
        {
            double *J_mat_ptr = J_mat + (scol + j) * num_bf + srow;
            double *y_ptr = y + y_spos + j * nrow;
            #pragma omp simd 
            for (int k = 0; k < nrow; k++) J_mat_ptr[k] += sym_coef * y_ptr[k];
        }
    }
    
    // Symmetrize the Coulomb matrix: J_mat = J_mat + J_mat^T
    #pragma omp parallel for schedule(dynamic, 16)
    for (int i = 0; i < num_bf; i++)
    {
        for (int j = 0; j < i; j++)
        {
            int idx0 = i * num_bf + j;
            int idx1 = j * num_bf + i;
            double val = J_mat[idx0] + J_mat[idx1];
            J_mat[idx0] = val;
            J_mat[idx1] = val;
        }
        int idx_ii = i * num_bf + i;
        J_mat[idx_ii] += J_mat[idx_ii];
    }
}

// H2 matvec intermediate multiplication for H2ERI
// All B_{ij} matrices are calculated and stored
void H2ERI_H2_matvec_intmd_mult_AOT(H2ERI_t h2eri, const double *x)
{
    H2Pack_t h2pack  = h2eri->h2pack;
    int n_node       = h2pack->n_node;
    int n_thread     = h2pack->n_thread;
    int *r_adm_pairs = h2pack->r_adm_pairs;
    int *node_level  = h2pack->node_level;
    int *mat_cluster = h2pack->mat_cluster;
    H2P_int_vec_t   B_blk        = h2pack->B_blk;
    H2P_dense_mat_t *y0          = h2pack->y0;
    H2P_thread_buf_t *thread_buf = h2pack->tb;
    H2P_dense_mat_t  *c_B_blks   = h2eri->c_B_blks;
    
    // 1. Initialize y1 
    H2P_matvec_init_y1(h2pack);
    H2P_dense_mat_t *y1 = h2pack->y1;
    
    // 2. Intermediate multiplication
    const int n_B_blk = B_blk->length - 1;
    #pragma omp parallel num_threads(n_thread)
    {
        int tid = omp_get_thread_num();
        double *y = thread_buf[tid]->y;
        thread_buf[tid]->timer = -get_wtime_sec();
        
        #pragma omp for schedule(static)
        for (int i = 0; i < n_node; i++)
        {
            if (y1[i]->ld == 0) continue;
            const int ncol = y1[i]->ncol;
            // Need not to reset all copies of y1 to be 0 here, use the last element in
            // each row as the beta value to rewrite / accumulate y1 results in GEMV
            memset(y1[i]->data, 0, sizeof(double) * ncol);
            for (int j = 1; j < n_thread; j++)
                y1[i]->data[(j + 1) * ncol - 1] = 0.0;
        }
        
        #pragma omp barrier
        
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
                
                H2P_dense_mat_t Bi = c_B_blks[i];
                
                // (1) Two nodes are of the same level, compress on both sides
                if (level0 == level1)
                {
                    int ncol0 = y1[node0]->ncol;
                    int ncol1 = y1[node1]->ncol;
                    double *y1_dst_0 = y1[node0]->data + tid * ncol0;
                    double *y1_dst_1 = y1[node1]->data + tid * ncol1;
                    double beta0 = y1_dst_0[ncol0 - 1];
                    double beta1 = y1_dst_1[ncol1 - 1];
                    y1_dst_0[ncol0 - 1] = 1.0;
                    y1_dst_1[ncol1 - 1] = 1.0;
                    if (beta0 == 0.0) memset(y1_dst_0, 0, sizeof(double) * Bi->nrow);
                    if (beta1 == 0.0) memset(y1_dst_1, 0, sizeof(double) * Bi->ncol);
                    CBLAS_BI_GEMV(
                        Bi->nrow, Bi->ncol, Bi->data, Bi->ncol,
                        y0[node1]->data, y0[node0]->data, y1_dst_0, y1_dst_1
                    );
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
                    double beta0     = y1_dst_0[ncol0 - 1];
                    y1_dst_0[ncol0 - 1] = 1.0;
                    if (beta0 == 0.0) memset(y1_dst_0, 0, sizeof(double) * Bi->nrow);
                    CBLAS_BI_GEMV(
                        Bi->nrow, Bi->ncol, Bi->data, Bi->ncol,
                        x_spos, y0[node0]->data, y1_dst_0, y_spos
                    );
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
                    double beta1     = y1_dst_1[ncol1 - 1];
                    y1_dst_1[ncol1 - 1] = 1.0;
                    if (beta1 == 0.0) memset(y1_dst_1, 0, sizeof(double) * Bi->ncol);
                    CBLAS_BI_GEMV(
                        Bi->nrow, Bi->ncol, Bi->data, Bi->ncol,
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

// H2 matvec intermediate multiplication for H2ERI
// Need to calculate all B_{ij} matrices before using it
void H2ERI_H2_matvec_intmd_mult_JIT(H2ERI_t h2eri, const double *x)
{
    H2Pack_t h2pack = h2eri->h2pack;
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
    H2P_int_vec_t B_blk   = h2pack->B_blk;
    H2P_int_vec_t *J_pair = h2eri->J_pair;
    H2P_int_vec_t *J_row  = h2eri->J_row;
    H2P_dense_mat_t *y0   = h2pack->y0;
    H2P_thread_buf_t *thread_buf      = h2pack->tb;
    simint_buff_t    *simint_buffs    = h2eri->simint_buffs;
    eri_batch_buff_t *eri_batch_buffs = h2eri->eri_batch_buffs;
    
    // 1. Initialize y1 
    H2P_matvec_init_y1(h2pack);
    H2P_dense_mat_t *y1 = h2pack->y1;
    
    // 2. Intermediate multiplication
    const int n_B_blk = B_blk->length - 1;
    #pragma omp parallel num_threads(n_thread)
    {
        int tid = omp_get_thread_num();
        
        H2P_dense_mat_t  tmpB           = thread_buf[tid]->mat0;
        simint_buff_t    simint_buff    = simint_buffs[tid];
        eri_batch_buff_t eri_batch_buff = eri_batch_buffs[tid];
        
        double *y = thread_buf[tid]->y;
        
        thread_buf[tid]->timer = -get_wtime_sec();
        
        #pragma omp for schedule(static)
        for (int i = 0; i < n_node; i++)
        {
            if (y1[i]->ld == 0) continue;
            const int ncol = y1[i]->ncol;
            // Need not to reset all copies of y1 to be 0 here, use the last element in
            // each row as the beta value to rewrite / accumulate y1 results in GEMV
            memset(y1[i]->data, 0, sizeof(double) * ncol);
            for (int j = 1; j < n_thread; j++)
                y1[i]->data[(j + 1) * ncol - 1] = 0.0;
        }
        
        #pragma omp barrier
        
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
                    int tmpB_nrow  = H2ERI_gather_sum(sp_nbfp, J_pair[node0]);
                    int tmpB_ncol  = H2ERI_gather_sum(sp_nbfp, J_pair[node1]);
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
                    double beta0 = y1_dst_0[ncol0 - 1];
                    double beta1 = y1_dst_1[ncol1 - 1];
                    y1_dst_0[ncol0 - 1] = 1.0;
                    y1_dst_1[ncol1 - 1] = 1.0;
                    if (beta0 == 0.0) memset(y1_dst_0, 0, sizeof(double) * tmpB->nrow);
                    if (beta1 == 0.0) memset(y1_dst_1, 0, sizeof(double) * tmpB->ncol);
                    CBLAS_BI_GEMV(
                        tmpB->nrow, tmpB->ncol, tmpB->data, tmpB->ncol,
                        y0[node1]->data, y0[node0]->data, y1_dst_0, y1_dst_1
                    );
                }
                
                // (2) node1 is a leaf node and its level is higher than node0's level, 
                //     only compress on node0's side
                if (level0 > level1)
                {
                    int tmpB_nrow  = H2ERI_gather_sum(sp_nbfp, J_pair[node0]);
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
                    double beta0     = y1_dst_0[ncol0 - 1];
                    y1_dst_0[ncol0 - 1] = 1.0;
                    if (beta0 == 0.0) memset(y1_dst_0, 0, sizeof(double) * tmpB->nrow);
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
                    int tmpB_ncol  = H2ERI_gather_sum(sp_nbfp, J_pair[node1]);
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
                    double beta1     = y1_dst_1[ncol1 - 1];
                    y1_dst_1[ncol1 - 1] = 1.0;
                    if (beta1 == 0.0) memset(y1_dst_1, 0, sizeof(double) * tmpB->ncol);
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
void H2ERI_H2_matvec_dense_mult_AOT(H2ERI_t h2eri, const double *x)
{
    H2Pack_t h2pack    = h2eri->h2pack;
    int n_thread       = h2pack->n_thread;
    int n_leaf_node    = h2pack->n_leaf_node;
    int *leaf_nodes    = h2pack->height_nodes;
    int *mat_cluster   = h2pack->mat_cluster;
    int *r_inadm_pairs = h2pack->r_inadm_pairs;
    H2P_int_vec_t    D_blk0      = h2pack->D_blk0;
    H2P_int_vec_t    D_blk1      = h2pack->D_blk1;
    H2P_thread_buf_t *thread_buf = h2pack->tb;
    H2P_dense_mat_t  *c_D_blks   = h2eri->c_D_blks;
    
    const int n_D0_blk = D_blk0->length - 1;
    const int n_D1_blk = D_blk1->length - 1;
    #pragma omp parallel num_threads(n_thread)
    {
        int tid = omp_get_thread_num();
        
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
                H2P_dense_mat_t Di = c_D_blks[i];
                int node  = leaf_nodes[i];
                int vec_s = mat_cluster[node * 2];
                double       *y_spos = y + vec_s;
                const double *x_spos = x + vec_s;
                CBLAS_GEMV(
                    CblasRowMajor, CblasNoTrans, Di->nrow, Di->ncol,
                    1.0, Di->data, Di->nrow, x_spos, 1, 1.0, y_spos, 1
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
                H2P_dense_mat_t Di = c_D_blks[i + n_leaf_node];
                int node0  = r_inadm_pairs[2 * i];
                int node1  = r_inadm_pairs[2 * i + 1];
                int vec_s0 = mat_cluster[2 * node0];
                int vec_s1 = mat_cluster[2 * node1];
                double       *y_spos0 = y + vec_s0;
                double       *y_spos1 = y + vec_s1;
                const double *x_spos0 = x + vec_s0;
                const double *x_spos1 = x + vec_s1;
                CBLAS_BI_GEMV(
                    Di->nrow, Di->ncol, Di->data, Di->ncol,
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

// H2 matvec dense multiplication for H2ERI
// Need to calculate all D_{ij} matrices before using it
void H2ERI_H2_matvec_dense_mult_JIT(H2ERI_t h2eri, const double *x)
{
    H2Pack_t h2pack = h2eri->h2pack;
    int n_thread         = h2pack->n_thread;
    int n_leaf_node      = h2pack->n_leaf_node;
    int *pt_cluster      = h2pack->pt_cluster;
    int *leaf_nodes      = h2pack->height_nodes;
    int *mat_cluster     = h2pack->mat_cluster;
    int *r_inadm_pairs   = h2pack->r_inadm_pairs;
    int *D_nrow          = h2pack->D_nrow;
    int *D_ncol          = h2pack->D_ncol;
    int *index_seq       = h2eri->index_seq;
    H2P_int_vec_t D_blk0 = h2pack->D_blk0;
    H2P_int_vec_t D_blk1 = h2pack->D_blk1;
    multi_sp_t *sp       = h2eri->sp;
    H2P_thread_buf_t *thread_buf      = h2pack->tb;
    simint_buff_t    *simint_buffs    = h2eri->simint_buffs;
    eri_batch_buff_t *eri_batch_buffs = h2eri->eri_batch_buffs;
    
    const int n_D0_blk = D_blk0->length - 1;
    const int n_D1_blk = D_blk1->length - 1;
    #pragma omp parallel num_threads(n_thread)
    {
        int tid = omp_get_thread_num();
        
        H2P_dense_mat_t  tmpD           = thread_buf[tid]->mat0;
        simint_buff_t    simint_buff    = simint_buffs[tid];
        eri_batch_buff_t eri_batch_buff = eri_batch_buffs[tid];
        
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

// Build the Coulomb matrix using the density matrix, H2 representation
// of the ERI tensor, and H2 matvec
void H2ERI_H2_matvec(H2ERI_t h2eri, const double *x, double *y)
{
    H2Pack_t h2pack   = h2eri->h2pack;
    int krnl_mat_size = h2pack->krnl_mat_size;
    int n_thread      = h2pack->n_thread;
    H2P_thread_buf_t *thread_buf = h2pack->tb;
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
    h2pack->timers[8] += et - st;
    
    // 2. Forward transformation, calculate U_j^T * x_j
    st = get_wtime_sec();
    H2P_matvec_fwd_transform(h2pack, x, y);
    et = get_wtime_sec();
    h2pack->timers[4] += et - st;
    
    // 3. Intermediate multiplication, calculate B_{ij} * (U_j^T * x_j)
    st = get_wtime_sec();
    if (h2eri->h2pack->BD_JIT == 0)
    {
        H2ERI_H2_matvec_intmd_mult_AOT(h2eri, x);
    } else {
        H2ERI_H2_matvec_intmd_mult_JIT(h2eri, x);
    }
    et = get_wtime_sec();
    h2pack->timers[5] += et - st;
    
    // 4. Backward transformation, calculate U_i * (B_{ij} * (U_j^T * x_j))
    st = get_wtime_sec();
    H2P_matvec_bwd_transform(h2pack, x, y);
    et = get_wtime_sec();
    h2pack->timers[6] += et - st;
    
    // 5. Dense multiplication, calculate D_i * x_i
    st = get_wtime_sec();
    if (h2eri->h2pack->BD_JIT == 0)
    {
        H2ERI_H2_matvec_dense_mult_AOT(h2eri, x);
    } else {
        H2ERI_H2_matvec_dense_mult_JIT(h2eri, x);
    }
    et = get_wtime_sec();
    h2pack->timers[7] += et - st;
    
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
    h2pack->mat_size[7] = (2 * n_thread + 1) * h2pack->krnl_mat_size;
    et = get_wtime_sec();
    h2pack->timers[8] += et - st;
    
    h2pack->n_matvec++;
}

// Build the Coulomb matrix using the density matrix, H2 representation
// of the ERI tensor, and H2 matvec
void H2ERI_build_Coulomb(H2ERI_t h2eri, const double *den_mat, double *J_mat)
{
    if (h2eri->unc_denmat_x == NULL)
    {
        size_t vec_msize = sizeof(double) * h2eri->num_sp_bfp;
        h2eri->unc_denmat_x = (double *) malloc(vec_msize);
        h2eri->H2_matvec_y  = (double *) malloc(vec_msize);
        assert(h2eri->unc_denmat_x != NULL && h2eri->H2_matvec_y != NULL);
    }
    
    H2ERI_uncontract_den_mat(h2eri, den_mat);
    
    H2ERI_H2_matvec(h2eri, h2eri->unc_denmat_x, h2eri->H2_matvec_y);
    
    H2ERI_contract_H2_matvec(h2eri, J_mat);
}

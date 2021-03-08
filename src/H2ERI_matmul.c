#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>

#include <omp.h>

#include "H2Pack_matvec.h"
#include "H2Pack_utils.h"
#include "H2ERI_typedef.h"
#include "H2ERI_matmul.h"
#include "utils.h"  // In H2Pack


// Perform matrix multiplication y = blk * x, where blk can be a dense block
// or a low-rank approximation blk = U * V. Note that blk is a row-major 
// matrix in both forms but x and y are column-major matrices
void H2ERI_BD_blk_matmul_cm(
    const int trans_blk, H2P_dense_mat_p blk, H2P_dense_mat_p tmp_v, const int n_vec,
    const double *x, const int ldx, double *y, const int ldy
)
{
    if (blk->ld > 0)
    {
        if (trans_blk == 0)
        {
            CBLAS_GEMM(
                CblasColMajor, CblasTrans, CblasNoTrans, blk->nrow, n_vec, blk->ncol,
                1.0, blk->data, blk->ld, x, ldx, 1.0, y, ldy
            );
        } else {
            CBLAS_GEMM(
                CblasColMajor, CblasNoTrans, CblasNoTrans, blk->ncol, n_vec, blk->nrow,
                1.0, blk->data, blk->ld, x, ldx, 1.0, y, ldy
            );
        }
    } else {
        int    blk_rank = -blk->ld;
        double *U_mat   = blk->data;
        double *VT_mat  = U_mat + blk->nrow * blk_rank;
        // U  : blk->nrow * blk_rank
        // VT : blk->ncol * blk_rank
        // Note: V^T instead of V is stored, VT in row-major == V in column-major
        H2P_dense_mat_resize(tmp_v, blk_rank, n_vec);
        if (trans_blk == 0)
        {
            // y = (U * V) * x = U * (V * x)
            CBLAS_GEMM(
                CblasColMajor, CblasNoTrans, CblasNoTrans, blk_rank, n_vec, blk->ncol,
                1.0, VT_mat, blk_rank, x, ldx, 0.0, tmp_v->data, blk_rank
            );
            CBLAS_GEMM(
                CblasColMajor, CblasTrans, CblasNoTrans, blk->nrow, n_vec, blk_rank,
                1.0, U_mat, blk_rank, tmp_v->data, blk_rank, 1.0, y, ldy
            );
        } else {
            // y = (U * V)^T * x = V^T * (U^T * x)
            CBLAS_GEMM(
                CblasColMajor, CblasNoTrans, CblasNoTrans, blk_rank, n_vec, blk->nrow,
                1.0, U_mat, blk_rank, x, ldx, 0.0, tmp_v->data, blk_rank
            );
            CBLAS_GEMM(
                CblasColMajor, CblasTrans, CblasNoTrans, blk->ncol, n_vec, blk_rank,
                1.0, VT_mat, blk_rank, tmp_v->data, blk_rank, 1.0, y, ldy
            );
        }
    }
}

// H2 matmul forward transformation, calculate U_j^T * x_j
void H2ERI_matmul_fwd_transform(H2ERI_p h2eri, const int n_vec, const double *mat_x, const int ldx)
{
    H2Pack_p h2pack = h2eri->h2pack;
    int n_thread       = h2pack->n_thread;
    int max_child      = h2pack->max_child;
    int max_level      = h2pack->max_level;
    int min_adm_level  = (h2pack->is_HSS) ? h2pack->HSS_min_adm_level : h2pack->min_adm_level;
    int n_leaf_node    = h2pack->n_leaf_node;
    int *children      = h2pack->children;
    int *n_child       = h2pack->n_child;
    int *level_nodes   = h2pack->level_nodes;
    int *level_n_node  = h2pack->level_n_node;
    int *mat_cluster   = h2pack->mat_cluster;

    // 1. Initialize y0 on the first run
    H2P_matmul_init_y0(h2pack, n_vec);

    // 2. Upward sweep
    H2P_dense_mat_p *y0 = h2pack->y0;
    H2P_dense_mat_p *U  = h2pack->U;
    for (int i = max_level; i >= min_adm_level; i--)
    {
        int *level_i_nodes = level_nodes + i * n_leaf_node;
        int level_i_n_node = level_n_node[i];
        int n_thread_i = MIN(level_i_n_node, n_thread);

        #pragma omp parallel num_threads(n_thread_i)
        {
            #pragma omp for schedule(dynamic)
            for (int j = 0; j < level_i_n_node; j++)
            {
                int node = level_i_nodes[j];
                int n_child_node = n_child[node];
                H2P_dense_mat_p U_node = U[node];

                H2P_dense_mat_resize(y0[node], U_node->ncol, n_vec);
                if (n_child_node == 0)
                {
                    // Leaf node, directly multiply x_j with U_j^T
                    int s_row = mat_cluster[2 * node];
                    int e_row = mat_cluster[2 * node + 1];
                    int nrow = e_row - s_row + 1;
                    const double *mat_x_blk = mat_x + s_row;
                    // Originally we need U[node]^T multiplies x, since U[node] is stored in row-major
                    // and we use CblasColMajor here, no need to transpose
                    CBLAS_GEMM(
                        CblasColMajor, CblasNoTrans, CblasNoTrans, U_node->ncol, n_vec, nrow,
                        1.0, U_node->data, U_node->ld, mat_x_blk, ldx, 0.0, y0[node]->data, y0[node]->nrow
                    );
                } else {
                    // Non-leaf node, multiple U{node}^T with each child node y0 directly
                    int *node_children = children + node * max_child;
                    int U_srow = 0;
                    for (int k = 0; k < n_child_node; k++)
                    {
                        int child_k = node_children[k];
                        H2P_dense_mat_p y0_k = y0[child_k];
                        double *U_node_k = U_node->data + U_srow * U_node->ld;
                        double beta = (k == 0) ? 0.0 : 1.0;
                        // Originally we need U[node]^T multiplies y0[child_k], since U[node] is stored in row-major
                        // and we use CblasColMajor here, no need to transpose
                        CBLAS_GEMM(
                            CblasColMajor, CblasNoTrans, CblasNoTrans, U_node->ncol, n_vec, y0_k->nrow,
                            1.0, U_node_k, U_node->ld, y0_k->data, y0_k->nrow, beta, y0[node]->data, y0[node]->nrow
                        );
                        U_srow += y0_k->nrow;
                    }  // End of k loop
                }  // End of "if (n_child_node == 0)"
            }  // End of j loop
        }  // End of "#pragma omp parallel"
    }  // End of i loop
}

void H2ERI_matmul_intmd_mult(
    H2ERI_p h2eri, const int n_vec, 
    const double *mat_x, const int ldx, double *mat_y, const int ldy
)
{
    H2Pack_p h2pack = h2eri->h2pack;
    int n_node        = h2pack->n_node;
    int n_thread      = h2pack->n_thread;
    int *node_level   = h2pack->node_level;
    int *mat_cluster  = h2pack->mat_cluster;
    int *B_p2i_rowptr = h2pack->B_p2i_rowptr;
    int *B_p2i_colidx = h2pack->B_p2i_colidx;
    int *B_p2i_val    = h2pack->B_p2i_val;
    H2P_thread_buf_p *thread_buf = h2pack->tb;
    H2P_dense_mat_p  *y0         = h2pack->y0;
    H2P_dense_mat_p  *c_B_blks   = h2eri->c_B_blks;

    // 1. Initialize y1 on the first run or reset the size of each y1
    H2P_matmul_init_y1(h2pack, n_vec);
    H2P_dense_mat_p *y1 = h2pack->y1;

    // 2. Intermediate sweep
    #pragma omp parallel num_threads(n_thread)
    {
        int tid = omp_get_thread_num();
        H2P_dense_mat_p tmp_mat = thread_buf[tid]->mat0;

        #pragma omp for schedule(dynamic)
        for (int node0 = 0; node0 < n_node; node0++)
        {
            int level0 = node_level[node0];
            
            H2P_dense_mat_p y1_node0 = y1[node0];
            memset(y1_node0->data, 0, sizeof(double) * y1_node0->nrow * y1_node0->ncol);

            for (int i = B_p2i_rowptr[node0]; i < B_p2i_rowptr[node0 + 1]; i++)
            {
                int node1  = B_p2i_colidx[i];
                int level1 = node_level[node1];
                H2P_dense_mat_p y0_node1 = y0[node1];

                int pair_idx_ij = B_p2i_val[i];
                int trans_blk;
                H2P_dense_mat_p Bij;
                if (pair_idx_ij > 0)
                {
                    trans_blk = 0;
                    Bij = c_B_blks[ pair_idx_ij - 1];
                } else {
                    trans_blk = 1;
                    Bij = c_B_blks[-pair_idx_ij - 1];
                }

                // We only handle the update on node0's side, the symmetric operation for
                // updating on node1's side is handled by double counting the inadmissible pairs

                // (1) Two nodes are of the same level, compress on both sides
                if (level0 == level1)
                {
                    H2ERI_BD_blk_matmul_cm(
                        trans_blk, Bij, tmp_mat, n_vec, 
                        y0_node1->data, y0_node1->nrow, y1_node0->data, y1_node0->nrow
                    );
                }

                // (2) node1 is a leaf node and its level is larger than node0, only compress on node0's side
                if (level0 > level1)
                {
                    int mat_x_srow = mat_cluster[node1 * 2];
                    const double *mat_x_spos = mat_x + mat_x_srow;
                    H2ERI_BD_blk_matmul_cm(
                        trans_blk, Bij, tmp_mat, n_vec, 
                        mat_x_spos, ldx, y1_node0->data, y1_node0->nrow
                    );
                }

                // (3) node0 is a leaf node and its level is larger than node1, only compress on node1's side
                if (level0 < level1)
                {
                    int mat_y_srow = mat_cluster[node0 * 2];
                    double *mat_y_spos = mat_y + mat_y_srow;
                    H2ERI_BD_blk_matmul_cm(
                        trans_blk, Bij, tmp_mat, n_vec, 
                        y0_node1->data, y0_node1->nrow, mat_y_spos, ldy
                    );
                }
            }  // End of i loop
        }  // End of node0 loop
    }  // End of "#pragma omp parallel"
}

// H2 matmul backward transformation, calculate U_i * (B_{ij} * (U_j^T * x_j))
void H2ERI_matmul_bwd_transform(H2ERI_p h2eri, const int n_vec, double *mat_y, const int ldy)
{
    H2Pack_p h2pack = h2eri->h2pack;
    int n_thread        = h2pack->n_thread;
    int max_child       = h2pack->max_child;
    int n_leaf_node     = h2pack->n_leaf_node;
    int max_level       = h2pack->max_level;
    int min_adm_level   = (h2pack->is_HSS) ? h2pack->HSS_min_adm_level : h2pack->min_adm_level;
    int *children       = h2pack->children;
    int *n_child        = h2pack->n_child;
    int *level_n_node   = h2pack->level_n_node;
    int *level_nodes    = h2pack->level_nodes;
    int *mat_cluster    = h2pack->mat_cluster;
    H2P_dense_mat_p *U  = h2pack->U;
    H2P_dense_mat_p *y1 = h2pack->y1;
    H2P_thread_buf_p *thread_buf = h2pack->tb;
    
    for (int i = min_adm_level; i <= max_level; i++)
    {
        int *level_i_nodes = level_nodes + i * n_leaf_node;
        int level_i_n_node = level_n_node[i];
        int n_thread_i = MIN(level_i_n_node, n_thread);
        
        #pragma omp parallel num_threads(n_thread_i) 
        {
            int tid = omp_get_thread_num();
            H2P_dense_mat_p y1_tmp = thread_buf[tid]->mat0;
            
            thread_buf[tid]->timer = -get_wtime_sec();
            #pragma omp for schedule(dynamic) nowait
            for (int j = 0; j < level_i_n_node; j++)
            {
                int node = level_i_nodes[j];
                int n_child_node = n_child[node];
                int *child_nodes = children + node * max_child;
                
                if (y1[node]->ld == 0) continue;
                
                H2P_dense_mat_resize(y1_tmp, U[node]->nrow, n_vec);

                // Originally we need U[node] * y1[node], since U[node] is stored in row-major style
                // and we use CblasColMajor for y1[node] and y1_tmp, we need to transpose U[node]
                CBLAS_GEMM(
                    CblasColMajor, CblasTrans, CblasNoTrans, U[node]->nrow, n_vec, U[node]->ncol,
                    1.0, U[node]->data, U[node]->ld, y1[node]->data, y1[node]->nrow, 0.0, y1_tmp->data, y1_tmp->nrow
                );
                
                if (n_child_node == 0)
                {
                    // Leaf node, accumulate final results to output vector
                    int s_row = mat_cluster[2 * node];
                    int e_row = mat_cluster[2 * node + 1];
                    int n_row = e_row - s_row + 1;
                    for (int l = 0; l < n_vec; l++)
                    {
                        double *mat_y_l  = mat_y + l * ldy;
                        double *y1_tmp_l = y1_tmp->data + l * y1_tmp->nrow;
                        #pragma omp simd
                        for (int k = 0; k < n_row; k++)
                            mat_y_l[s_row + k] += y1_tmp_l[k];
                    }
                } else {
                    // Non-leaf node, push down y1 values
                    int y1_tmp_idx = 0;
                    for (int k = 0; k < n_child_node; k++)
                    {
                        int child_k = child_nodes[k];
                        int child_k_len = U[child_k]->ncol;
                        double *y1_tmp_spos = y1_tmp->data + y1_tmp_idx;
                        if (y1[child_k]->ld == 0)
                        {
                            H2P_dense_mat_resize(y1[child_k], child_k_len, n_vec);
                            // copy_matrix_block uses row-major, swap the number of rows & columns
                            copy_matrix_block(sizeof(double), n_vec, child_k_len, y1_tmp_spos, y1_tmp->nrow, y1[child_k]->data, y1[child_k]->nrow);
                        } else {
                            for (int l = 0; l < n_vec; l++)
                            {
                                double *y1_tmp_l = y1_tmp_spos + l * y1_tmp->nrow;
                                double *y1_child_k_l = y1[child_k]->data + l * y1[child_k]->nrow;
                                #pragma omp simd
                                for (int k0 = 0; k0 < child_k_len; k0++)
                                    y1_child_k_l[k0] += y1_tmp_l[k0];
                            }
                        }
                        y1_tmp_idx += child_k_len;
                    }  // End of k loop
                }  // End of "if (n_child_node == 0)"
            }  // End of j loop
            thread_buf[tid]->timer += get_wtime_sec();
        }  // End of "pragma omp parallel"
    }  // End of i loop
}

void H2ERI_matmul_dense_mult(
    H2ERI_p h2eri, const int n_vec, 
    const double *mat_x, const int ldx, double *mat_y, const int ldy
)
{
    H2Pack_p h2pack = h2eri->h2pack;
    int n_node        = h2pack->n_node;
    int n_thread      = h2pack->n_thread;
    int *mat_cluster  = h2pack->mat_cluster;
    int *D_p2i_rowptr = h2pack->D_p2i_rowptr;
    int *D_p2i_colidx = h2pack->D_p2i_colidx;
    int *D_p2i_val    = h2pack->D_p2i_val;
    H2P_thread_buf_p *thread_buf = h2pack->tb;
    H2P_dense_mat_p  *c_D_blks   = h2eri->c_D_blks;

    #pragma omp parallel num_threads(n_thread)
    {
        int tid = omp_get_thread_num();
        H2P_dense_mat_p tmp_mat = thread_buf[tid]->mat0;

        #pragma omp for schedule(dynamic)
        for (int node0 = 0; node0 < n_node; node0++)
        {
            int mat_y_srow = mat_cluster[2 * node0];
            double *mat_y_spos = mat_y + mat_y_srow;

            for (int i = D_p2i_rowptr[node0]; i < D_p2i_rowptr[node0 + 1]; i++)
            {
                int node1 = D_p2i_colidx[i];
                int mat_x_srow = mat_cluster[2 * node1];
                const double *mat_x_spos = mat_x + mat_x_srow;

                int pair_idx_ij = D_p2i_val[i];
                int trans_blk;
                H2P_dense_mat_p Dij;
                if (pair_idx_ij > 0)
                {
                    trans_blk = 0;
                    Dij = c_D_blks[ pair_idx_ij - 1];
                } else {
                    trans_blk = 1;
                    Dij = c_D_blks[-pair_idx_ij - 1];
                }

                // We only handle y_i = D_{ij} * x_j, its symmetric operation
                // y_j = D_{ij}' * x_i is handled by double counting inadmissible pairs
                H2ERI_BD_blk_matmul_cm(
                    trans_blk, Dij, tmp_mat, n_vec, 
                    mat_x_spos, ldx, mat_y_spos, ldy
                );
            }  // End of i loop
        }  // End of node0 loop
    }  // End of "#pragma omp parallel"
}

void H2ERI_matmul(
    H2ERI_p h2eri, const int n_vec, 
    const double *mat_x, const int ldx, double *mat_y, const int ldy
)
{
    ASSERT_PRINTF(h2eri->h2pack->BD_JIT == 0, "%s does not support BD JIT build\n", __FUNCTION__);

    H2Pack_p h2pack = h2eri->h2pack;
    int    krnl_mat_size = h2pack->krnl_mat_size;
    int    mm_max_n_vec  = h2pack->mm_max_n_vec;
    double *timers       = h2pack->timers;
    double st, et;

    BLAS_SET_NUM_THREADS(1);
    for (int i_vec = 0; i_vec < n_vec; i_vec += mm_max_n_vec)
    {
        int curr_n_vec = (i_vec + mm_max_n_vec <= n_vec) ? mm_max_n_vec : (n_vec - i_vec);
        const double *curr_mat_x = mat_x + i_vec * ldx;
        double *curr_mat_y = mat_y + i_vec * ldy;
    
        // 1. Reset output matrix
        st = get_wtime_sec();
        #pragma omp parallel
        {
            for (int i = 0; i < curr_n_vec; i++)
            {
                double *mat_y_i = curr_mat_y + i * ldy;
                #pragma omp for schedule(static)
                for (int j = 0; j < krnl_mat_size; j++) mat_y_i[j] = 0.0;
            }
        }
        et = get_wtime_sec();
        timers[MV_VOP_TIMER_IDX] += et - st;
        
        // 2. Forward transformation, calculate U_j^T * x_j
        st = get_wtime_sec();
        H2ERI_matmul_fwd_transform(h2eri, curr_n_vec, mat_x, ldx);
        et = get_wtime_sec();
        timers[MV_FWD_TIMER_IDX] += et - st;

        // 3. Intermediate multiplication, calculate B_{ij} * (U_j^T * x_j)
        st = get_wtime_sec();
        H2ERI_matmul_intmd_mult(h2eri, curr_n_vec, mat_x, ldx, mat_y, ldy);
        et = get_wtime_sec();
        timers[MV_MID_TIMER_IDX] += et - st;

        // 4. Backward transformation, calculate U_i * (B_{ij} * (U_j^T * x_j))
        st = get_wtime_sec();
        H2ERI_matmul_bwd_transform(h2eri, curr_n_vec, mat_y, ldy);
        et = get_wtime_sec();
        timers[MV_BWD_TIMER_IDX] += et - st;

        // 5. Dense multiplication, calculate D_{ij} * x_j
        st = get_wtime_sec();
        H2ERI_matmul_dense_mult(h2eri, curr_n_vec, mat_x, ldx, mat_y, ldy);
        et = get_wtime_sec();
        timers[MV_DEN_TIMER_IDX] += et - st;

        h2pack->n_matvec++;
    }  // End of i_vec loop
    BLAS_SET_NUM_THREADS(h2pack->n_thread);
}
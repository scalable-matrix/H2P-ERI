#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>

#include <omp.h>

#include "H2Pack_matvec.h"
#include "H2Pack_utils.h"
#include "H2ERI_typedef.h"
#include "H2ERI_build_ACE.h"
#include "H2ERI_utils.h"
#include "H2ERI_matvec.h"
#include "utils.h"  // In H2Pack

// Build the Adaptive Compressed Exchange (ACE) matrix with the Cocc matrix
void H2ERI_build_ACE(H2ERI_p h2eri, const int num_occ, const double *Cocc_mat, double *ACE_mat)
{
    int num_sp          = h2eri->num_sp;
    int num_bf          = h2eri->num_bf;
    int num_sp_bfp      = h2eri->num_sp_bfp;
    int max_shell_nbf   = h2eri->max_shell_nbf;
    int *shell_bf_sidx  = h2eri->shell_bf_sidx;
    int *sp_bfp_sidx    = h2eri->sp_bfp_sidx;
    int *sp_shell_idx   = h2eri->sp_shell_idx;
    H2Pack_p h2pack = h2eri->h2pack;

    double st, et, misc_t = 0.0, gather_t = 0.0, matmul_t = 0.0, acc_t = 0.0, ace_t = 0.0;

    // 1. Determine maximum nvec and total workbuf size, allocate and assign buffers
    st = get_wtime_sec();
    size_t ACE_workbuf_size = 0;
    ACE_workbuf_size += (size_t) num_bf  * (size_t) num_bf + 1;  // Cji
    ACE_workbuf_size += (size_t) num_bf  * (size_t) num_occ;     // Cocc_T
    ACE_workbuf_size += (size_t) num_bf  * (size_t) num_occ * 2; // K_Mi and K_Mi1
    ACE_workbuf_size += (size_t) num_occ * (size_t) num_occ;     // M and L
    size_t max_nvec0 = (1024 * 1024 * 1024 - ACE_workbuf_size) / ((size_t) num_sp_bfp * 2);
    int max_nvec = (int) max_nvec0;
    if (max_nvec > num_occ) max_nvec = num_occ;
    if (max_nvec > 256)     max_nvec = 256;
    ACE_workbuf_size += (size_t) num_sp_bfp * (size_t) max_nvec * 2;  // V_mat & W_mat
    ACE_workbuf_size += (size_t) num_sp_bfp;  // Two integer arrays
    if (ACE_workbuf_size > h2eri->ACE_workbuf_size)
    {
        free(h2eri->ACE_workbuf);
        h2eri->ACE_workbuf_size = ACE_workbuf_size;
        h2eri->ACE_workbuf = (double *) malloc(sizeof(double) * ACE_workbuf_size);
    }
    double *Cji     = h2eri->ACE_workbuf;
    double *Cocc_T  = Cji    + num_bf   * num_bf + 1;
    double *K_Mi    = Cocc_T + num_bf   * num_occ;
    double *K_Mi1   = K_Mi   + num_bf   * num_occ;
    double *M_mat   = K_Mi1  + num_bf   * num_occ;
    double *V_mat   = M_mat  + num_occ  * num_occ;
    double *W_mat   = V_mat  + max_nvec * num_sp_bfp;
    double *int_buf = W_mat  + max_nvec * num_sp_bfp;
    int *Cji2V_idx0 = (int *) int_buf;
    int *Cji2V_idx1 = Cji2V_idx0 + num_sp_bfp;
    et = get_wtime_sec();
    misc_t += et - st;

    // 2. Preparing Cji to V gathering indices
    st = get_wtime_sec();
    for (int k = 0; k < num_sp; k++)
    {
        int P = sp_shell_idx[k];
        int Q = sp_shell_idx[k + num_sp];
        int sp_bfp_idx_s = sp_bfp_sidx[k];
        int sp_bfp_idx_e = sp_bfp_sidx[k + 1];
        int P_bf_idx_s   = shell_bf_sidx[P];
        int P_bf_idx_e   = shell_bf_sidx[P + 1];
        int Q_bf_idx_s   = shell_bf_sidx[Q];
        int Q_bf_idx_e   = shell_bf_sidx[Q + 1];
        int idx = sp_bfp_sidx[k];
        for (int icol = Q_bf_idx_s; icol < Q_bf_idx_e; icol++)
        {
            for (int irow = P_bf_idx_s; irow < P_bf_idx_e; irow++)
            {
                Cji2V_idx0[idx] = icol * num_bf + irow;
                idx++;
            }
        }
        if (P != Q)
        {
            idx = sp_bfp_sidx[k];
            for (int irow = Q_bf_idx_s; irow < Q_bf_idx_e; irow++)
            {
                for (int icol = P_bf_idx_s; icol < P_bf_idx_e; icol++)
                {
                    Cji2V_idx1[idx] = icol * num_bf + irow;
                    idx++;
                }
            }
        } else {
            for (idx = sp_bfp_idx_s; idx < sp_bfp_idx_e; idx++) 
                Cji2V_idx1[idx] = num_bf * num_bf;
        }  // End of "if (P != Q)"
    }  // End of k loop
    et = get_wtime_sec();
    gather_t += et - st;

    // 3. Transpose Cocc_mat to column-major for better access patterns 
    st = get_wtime_sec();
    #pragma omp parallel for 
    for (int icol = 0; icol < num_occ; icol++)
    {
        double *Cocc_T_icol = Cocc_T + icol * num_bf;
        const double *Cocc_icol = Cocc_mat + icol;
        #pragma omp simd
        for (int irow = 0; irow < num_bf; irow++)
            Cocc_T_icol[irow] = Cocc_icol[irow * num_occ];
    }
    et = get_wtime_sec();
    misc_t += et - st;

    // 4. Build K_Mi matrix, K_Mi is a column-major matrix of size num_bf-by-num_occ
    #pragma omp parallel for 
    for (int i = 0; i < num_bf * num_occ; i++) K_Mi[i] = 0.0;
    Cji[num_bf * num_bf] = 0.0;
    for (int i = 0; i < num_occ; i++)
    {
        for (int j_start = 0; j_start < num_occ; j_start += max_nvec)
        {
            int j_end = j_start + max_nvec;
            if (j_end > num_occ) j_end = num_occ;
            int nvec = j_end - j_start;

            // 4.1 Prepare multiplicand V^{i, j}_{P, Q} = C_{P, j} * C_{Q, i}
            //     Due to symmetry of shell pairs, here we actually prepare 
            //     V^{i, j}_{P, Q} = C_{P, j} * C_{Q, i} + C_{Q, j} * C_{P, i}
            st = get_wtime_sec();
            #pragma omp parallel
            {
                for (int j = j_start; j < j_end; j++)
                {
                    // Cji = C(:, j) * C(:, i)';
                    // Cji2V_idx0 and Cji2V_idx1 are based on column-major Cji
                    double *Cocc_i = Cocc_T + i * num_bf;
                    double *Cocc_j = Cocc_T + j * num_bf;
                    #pragma omp barrier
                    #pragma omp for
                    for (int icol = 0; icol < num_bf; icol++)
                    {
                        double *Cji_icol = Cji + icol * num_bf;
                        #pragma omp simd
                        for (int irow = 0; irow < num_bf; irow++)
                            Cji_icol[irow] = Cocc_i[icol] * Cocc_j[irow];
                    }  // End of icol loop

                    // V(:, idx) = Cji1(C_idx0) + Cji1(C_idx1);
                    int idx = j - j_start;
                    double *V_j = V_mat + idx * num_sp_bfp;
                    #pragma omp barrier
                    #pragma omp for simd
                    for (int l = 0; l < num_sp_bfp; l++)
                        V_j[l] = Cji[Cji2V_idx0[l]] + Cji[Cji2V_idx1[l]];
                }  // End of j loop
            }  // End of "#pragma omp parallel"
            et = get_wtime_sec();
            gather_t += et - st;

            // 4.2 Multiplication W^{i, j}_{M, N} = (M, N|P, Q) V^{i, j}_{P, Q}
            st = get_wtime_sec();
            // TODO: implement H2ERI_matmul and use it
            for (int j = j_start; j < j_end; j++)
            {
                int idx = j - j_start;
                double *V_j = V_mat + idx * num_sp_bfp;
                double *W_j = W_mat + idx * num_sp_bfp;
                H2ERI_matvec(h2eri, V_j, W_j);
            }
            et = get_wtime_sec();
            matmul_t += et - st;

            // 4.3 Accumulate K_{M, i} = \sum_{N, j = 1}^{Nbf} W^{i, j}_{M, N} C_{N, j}
            st = get_wtime_sec();
            #pragma omp parallel
            {
                int tid = omp_get_thread_num();
                H2P_dense_mat_p tmp_mat = h2pack->tb[tid]->mat0;
                H2P_dense_mat_resize(tmp_mat, 2, max_shell_nbf);
                double *tmp0 = tmp_mat->data;
                double *tmp1 = tmp0 + max_shell_nbf;

                #pragma omp for
                for (int k = 0; k < num_sp; k++)
                {
                    int M = sp_shell_idx[k];
                    int N = sp_shell_idx[k + num_sp];
                    int sp_bfp_idx_s = sp_bfp_sidx[k];
                    int M_bf_idx_s   = shell_bf_sidx[M];
                    int M_bf_idx_e   = shell_bf_sidx[M + 1];
                    int N_bf_idx_s   = shell_bf_sidx[N];
                    int N_bf_idx_e   = shell_bf_sidx[N + 1];
                    int M_nbf        = M_bf_idx_e - M_bf_idx_s;
                    int N_nbf        = N_bf_idx_e - N_bf_idx_s;

                    // W_M_N = W(sp_bf_idx, :);
                    // C_M_j = C(M_bf_idx, j_start : j_end);
                    // C_N_j = C(N_bf_idx, j_start : j_end);
                    double *W_M_N = W_mat  + sp_bfp_idx_s;
                    double *C_M_j = Cocc_T + M_bf_idx_s + j_start * num_bf;
                    double *C_N_j = Cocc_T + N_bf_idx_s + j_start * num_bf;
                    memset(tmp0, 0, sizeof(double) * M_nbf);
                    memset(tmp1, 0, sizeof(double) * N_nbf);
                    for (int l = 0; l < N_nbf; l++)
                    {
                        // idx_l = ((l-1) * M_nbf + 1) : (l * M_nbf);
                        int idx_l_s = l * M_nbf;
                        int idx_l_e = (l + 1) * M_nbf;
                        double tmp1_l = 0.0;
                        for (int k = 0; k < nvec; k++)
                        {
                            // tmp0    = tmp0    + W_M_N(idx_l, k) .* C_N_j(l, k);
                            // tmp1(l) = tmp1(l) + dot(W_M_N(idx_l, k), C_M_j(:, k));
                            double C_N_j_l_k = C_N_j[l + k * num_bf];
                            double *W_M_N_k = W_M_N + k * num_sp_bfp;
                            double *C_M_j_k = C_M_j + k * num_bf;
                            #pragma omp simd
                            for (int ii = 0; ii < M_nbf; ii++)
                            {
                                double W_M_N_k_ii = W_M_N_k[idx_l_s + ii];
                                tmp0[ii] += W_M_N_k_ii * C_N_j_l_k;
                                tmp1_l   += W_M_N_k_ii * C_M_j_k[ii];
                            }
                        }  // End of k loop
                        tmp1[l] = tmp1_l;
                    }  // End of l loop

                    // K_Mi(M_bf_idx, i) = K_Mi(M_bf_idx, i) + tmp0;
                    double *K_Mi_M = K_Mi + M_bf_idx_s + i * num_bf;
                    for (int ii = 0; ii < M_nbf; ii++)
                        atomic_add_f64(K_Mi_M + ii, tmp0[ii]);
                    if (M != N)
                    {
                        // K_Mi(N_bf_idx, i) = K_Mi(N_bf_idx, i) + tmp1;
                        double *K_Mi_N = K_Mi + N_bf_idx_s + i * num_bf;
                        for (int ii = 0; ii < N_nbf; ii++)
                            atomic_add_f64(K_Mi_N + ii, tmp1[ii]);
                    }
                }  // End of k loop
            }  // End of "#pragma omp parallel"
            et = get_wtime_sec();
            acc_t += et - st;
        }  // End of j loop
        if (h2pack->print_dbginfo) DEBUG_PRINTF("i = %d done, K_{M, i} fro-norm = %e\n", i, calc_2norm(num_occ * num_bf, K_Mi));
    }  // End of i loop

    // 5. Build K_ij matrix
    st = get_wtime_sec();
    // M = C' * K_Mi;
    // [L, flag] = chol(M, 'lower');
    CBLAS_GEMM(
        CblasColMajor, CblasTrans, CblasNoTrans, num_occ, num_occ, num_bf,
        1.0, Cocc_T, num_bf, K_Mi, num_bf, 0.0, M_mat, num_occ
    );
    int info = LAPACK_POTRF(LAPACK_COL_MAJOR, 'L', num_occ, M_mat, num_occ);
    if (info != 0) ASSERT_PRINTF(info == 0, "LAPACK_POTRF returned error code %d\n", info);
    et = get_wtime_sec();
    ace_t += et - st;

    // 6. Build ACE matrix
    st = get_wtime_sec();
    // K_Mi = K_Mi * (inv(L)');
    info = LAPACKE_dtrtri(LAPACK_COL_MAJOR, 'L', 'N', num_occ, M_mat, num_occ);
    if (info != 0) ASSERT_PRINTF(info == 0, "LAPACKE_dtrtri returned error code %d\n", info);
    // Remember to zero out the upper triangle!!
    for (int icol = 1; icol < num_occ; icol++)
    {
        for (int irow = 0; irow < icol; irow++)
            M_mat[irow + icol * num_occ] = 0;
    }
    CBLAS_GEMM(
        CblasColMajor, CblasNoTrans, CblasTrans, num_bf, num_occ, num_occ,
        1.0, K_Mi, num_bf, M_mat, num_occ, 0.0, K_Mi1, num_bf
    );
    // K_ACE = K_Mi * K_Mi';
    // K_ACE is symmetric so it does not matter if it is row-major or column-major
    CBLAS_GEMM(
        CblasColMajor, CblasNoTrans, CblasTrans, num_bf, num_bf, num_occ,
        1.0, K_Mi1, num_bf, K_Mi1, num_bf, 0.0, ACE_mat, num_bf
    );
    et = get_wtime_sec();
    ace_t += et - st;

    if (h2pack->print_timers)
    {
        INFO_PRINTF(
            "Build ACE timers: gather, matmul, acc, ace, misc = %.3lf, %.3lf, %.3lf, %.3lf, %.3lf sec\n", 
            gather_t, matmul_t, acc_t, ace_t, misc_t
        );
    }
}

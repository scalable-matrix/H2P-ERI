#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>

#include <omp.h>

#include "H2ERI_typedef.h"
#include "H2ERI_matvec.h"
#include "H2ERI_build_exchange.h"
#include "H2ERI_utils.h"
#include "H2ERI_aux_structs.h"

struct Kmat_workbuf
{
    int    row_max_level;       // Row subtree max level
    int    row_idx_len;         // Length of row_idx
    int    row_n_leaf_node;     // Number of leaf nodes in row subtree
    int    M_cut;               // Index of the first element > N shell index in plist{N}
    int    col_max_level;       // Column subtree max level
    int    col_idx_len;         // Length of col_idx
    int    col_n_leaf_node;     // Number of leaf nodes in column subtree
    int    P_cut;               // Index of the first element > Q shell index in plist{Q}
    int    P_list_len;          // Length of P_list
    int    vec_out_idx_size;    // Size of vec_out_idx
    int    nvi_nnz_ridx_size;   // Size of nvi_nnz_row_idx
    int    nvo_nnz_ridx_size;   // Size of nvo_nnz_row_idx
    int    int_buffer_size;     // Size of int_buffer
    int    dbl_buffer_size;     // Size of dbl_buffer
    int    nvec;                // Current matmul output matrix number of columns, == N_nbf * Q_nbf
    int    *MN_bfp_sidx;        // Size h2eri->nshell+1, indices of each bra-side shell pair's first basis function pair
    int    *row_idx;            // Size h2eri->{nshell * max_shell_nbf^2}, H2 matvec row indices
    int    *row_leaf_nodes;     // Size h2eri->n_leaf_node, row subtree leaf nodes
    int    *row_node_flag;      // Size h2eri->n_node, mark if a node in h2eri is in row subtree
    int    *row_idx_ascend;     // Size h2eri->{nshell * max_shell_nbf^2}, row_idx sorted in ascending
    int    *row_idx_pmt;        // Size h2eri->{nshell * max_shell_nbf^2}, original indices of row_idx_ascend
    int    *vec_out_sidx;       // Size h2eri->n_node+1, indices of each node's vec_out_idx
    int    *P_list;             // Size h2eri->nshell, list of significant P shells
    int    *P_idx;              // Size h2eri->nshell, indices of P_list[i] in H2ERI_build_exchange->plist
    int    *col_idx;            // Size h2eri->{nshell * max_shell_nbf^2}, H2 matvec column indices
    int    *col_leaf_nodes;     // Size h2eri->n_leaf_node, column subtree leaf nodes
    int    *col_node_flag;      // Size h2eri->n_node, mark if a node in h2eri is in column subtree
    int    *col_idx_pmt;        // Size h2eri->{nshell * max_shell_nbf^2}, original indices of row_idx
    int    *col_idx_ipmt;       // Size h2eri->{nshell * max_shell_nbf^2}, inverse function of col_idx_pmt
    int    *nvi_nnz_row_sidx;   // Size h2eri->n_node+1, indices of each node's nvi_nnz_row_idx
    int    *nvo_nnz_row_sidx;   // Size h2eri->n_node+1, indices of each node's nvo_nnz_row_idx
    int    *nvi_nnz_sidx;       // Size h2eri->n_node+1, indices of each node's node_vec_in 
    int    *nvo_nnz_sidx;       // Size h2eri->n_node+1, indices of each node's node_vec_out
    int    *y0_sidx;            // Size h2eri->n_node+1, indices of each node's y0
    int    *y1_sidx;            // Size h2eri->n_node+1, indices of each node's y1
    int    *tmp_arr;            // Size h2eri->max_leaf_points, temporary array
    int    *int_buffer;         // == all int* arrays above vec_out_idx, to reduce memory fragments 
    int    *vec_out_idx;        // Size unknown, mapping from each node's nvo_nnz to out_vec
    int    *nvi_nnz_row_idx;    // Size unknown, non-zero row indices of each node's input vectors
    int    *nvo_nnz_row_idx;    // Size unknown, non-zero row indices of each node's unrestricted output vector
    double *vec_in;             // Size unknown, partial matmul input vectors
    double *vec_out;            // Size unknown, partial matmul output vectors restricted from each node's output
    double *nvi_nnz;            // Size unknown, non-zero rows of partial matmul input vectors extended to each node
    double *nvo_nnz;            // Size unknown, non-zero rows of partial matmul output vectors
    double *y0;                 // Size unknown, partial matmul intermediate variables
    double *y1;                 // Size unknown, partial matmul intermediate variables
    double *tmp_K;              // Size h2eri->max_shell_nbf, temporary array for K mat accumulation
    double *dbl_buffer;         // == vec_in + vec_out + node_vec_in + node_vec_out + y0 + y1, to reduce memory fragments 
    double timers[5];           // Profiling timers
};
typedef struct Kmat_workbuf  Kmat_workbuf_s;
typedef struct Kmat_workbuf *Kmat_workbuf_p;

typedef enum
{
    BUILD_K_AUX_TIMER_IDX = 0,  // Auxiliary data structure construction
    BUILD_K_MM_FWD_TIMER_IDX,   // H2 partial matmul forward transformation
    BUILD_K_MM_MID_TIMER_IDX,   // H2 partial matmul intermediate multiplication
    BUILD_K_MM_BWD_TIMER_IDX,   // H2 partial matmul backward transformation
    BUILD_K_MM_DEN_TIMER_IDX    // H2 partial matmul dense multiplication
} build_exchange_timer_idx_t;

// Initialize each thread's K mat build work buffer
void H2ERI_exchange_workbuf_init(H2ERI_p h2eri)
{
    int nshell        = h2eri->nshell;
    int max_shell_nbf = h2eri->max_shell_nbf;
    int n_thread      = h2eri->n_thread;
    int n_node        = h2eri->n_node;
    int n_leaf_node   = h2eri->n_leaf_node;
    int num_sp_bfp    = h2eri->num_sp_bfp;
    Kmat_workbuf_p *thread_Kmat_workbuf = (Kmat_workbuf_p *) malloc(sizeof(Kmat_workbuf_p) * n_thread);
    #pragma omp parallel num_threads(n_thread)
    {
        int tid = omp_get_thread_num();
        Kmat_workbuf_p workbuf = (Kmat_workbuf_p) malloc(sizeof(Kmat_workbuf_s));
        memset(workbuf, 0, sizeof(Kmat_workbuf_s));

        int int_buffer_size = 0;
        // MN_bfp_sidx
        int_buffer_size += nshell + 1;
        // row_idx, row_leaf_nodes, row_node_flag
        int_buffer_size += nshell * max_shell_nbf * max_shell_nbf;
        int_buffer_size += n_leaf_node;
        int_buffer_size += n_node;
        // row_idx_ascend, row_idx_pmt
        int_buffer_size += nshell * max_shell_nbf * max_shell_nbf;
        int_buffer_size += nshell * max_shell_nbf * max_shell_nbf;
        // vec_out_sidx, P_list, P_idx
        int_buffer_size += n_node + 1;
        int_buffer_size += nshell;
        int_buffer_size += nshell;
        // col_idx, col_leaf_nodes, col_node_flag, col_idx_pmt, col_idx_ipmt
        int_buffer_size += nshell * max_shell_nbf * max_shell_nbf;
        int_buffer_size += n_leaf_node;
        int_buffer_size += n_node;
        int_buffer_size += nshell * max_shell_nbf * max_shell_nbf;
        int_buffer_size += nshell * max_shell_nbf * max_shell_nbf;
        // nvi_nnz_row_sidx, nvo_nnz_row_sidx, nvi_nnz_sidx, nvo_nnz_sidx
        int_buffer_size += n_node + 1;
        int_buffer_size += n_node + 1;
        int_buffer_size += n_node + 1;
        int_buffer_size += n_node + 1;
        // y0_sidx, y1_sidx, tmp_arr
        int_buffer_size += n_node + 1;
        int_buffer_size += n_node + 1;
        int_buffer_size += num_sp_bfp;   // This is a safe upper bound for tmp_arr

        int *int_buffer = (int *) malloc(sizeof(int) * int_buffer_size);
        ASSERT_PRINTF(int_buffer != NULL, "Failed to allocate int_buffer of size %d\n", int_buffer_size);
        memset(int_buffer, 0, sizeof(int) * int_buffer_size);
        workbuf->int_buffer_size   = int_buffer_size;
        workbuf->int_buffer        = int_buffer;
        workbuf->MN_bfp_sidx       = workbuf->int_buffer;
        workbuf->row_idx           = workbuf->MN_bfp_sidx       + nshell + 1;
        workbuf->row_leaf_nodes    = workbuf->row_idx           + nshell * max_shell_nbf * max_shell_nbf;
        workbuf->row_node_flag     = workbuf->row_leaf_nodes    + n_leaf_node;
        workbuf->row_idx_ascend    = workbuf->row_node_flag     + n_node;
        workbuf->row_idx_pmt       = workbuf->row_idx_ascend    + nshell * max_shell_nbf * max_shell_nbf;
        workbuf->vec_out_sidx      = workbuf->row_idx_pmt       + nshell * max_shell_nbf * max_shell_nbf;
        workbuf->P_list            = workbuf->vec_out_sidx      + n_node + 1;
        workbuf->P_idx             = workbuf->P_list            + nshell;
        workbuf->col_idx           = workbuf->P_idx             + nshell;
        workbuf->col_leaf_nodes    = workbuf->col_idx           + nshell * max_shell_nbf * max_shell_nbf;
        workbuf->col_node_flag     = workbuf->col_leaf_nodes    + n_leaf_node;
        workbuf->col_idx_pmt       = workbuf->col_node_flag     + n_node;
        workbuf->col_idx_ipmt      = workbuf->col_idx_pmt       + nshell * max_shell_nbf * max_shell_nbf;
        workbuf->nvi_nnz_row_sidx  = workbuf->col_idx_ipmt      + nshell * max_shell_nbf * max_shell_nbf;
        workbuf->nvo_nnz_row_sidx  = workbuf->nvi_nnz_row_sidx  + n_node + 1;
        workbuf->nvi_nnz_sidx      = workbuf->nvo_nnz_row_sidx  + n_node + 1;
        workbuf->nvo_nnz_sidx      = workbuf->nvi_nnz_sidx      + n_node + 1;
        workbuf->y0_sidx           = workbuf->nvo_nnz_sidx      + n_node + 1;
        workbuf->y1_sidx           = workbuf->y0_sidx           + n_node + 1;
        workbuf->tmp_arr           = workbuf->y1_sidx           + n_node + 1;

        workbuf->vec_out_idx_size  = 0;
        workbuf->nvi_nnz_ridx_size = 0;
        workbuf->nvo_nnz_ridx_size = 0;
        workbuf->vec_out_idx       = NULL;
        workbuf->nvi_nnz_row_idx   = NULL;
        workbuf->nvo_nnz_row_idx   = NULL;

        workbuf->dbl_buffer_size   = 0;
        workbuf->dbl_buffer        = NULL;
        workbuf->vec_in            = NULL;
        workbuf->vec_out           = NULL;
        workbuf->nvi_nnz           = NULL;
        workbuf->nvo_nnz           = NULL;
        workbuf->y0                = NULL;
        workbuf->y1                = NULL;

        thread_Kmat_workbuf[tid] = workbuf;
    }  // End of "#pragma omp parallel"
    h2eri->thread_Kmat_workbuf = (void **) thread_Kmat_workbuf;
}

// Free each thread's K mat build work buffer
void H2ERI_exchange_workbuf_free(H2ERI_p h2eri)
{
    int n_thread = h2eri->n_thread;
    Kmat_workbuf_p *thread_Kmat_workbuf = (Kmat_workbuf_p *) h2eri->thread_Kmat_workbuf;
    for (int i = 0; i < n_thread; i++)
    {
        Kmat_workbuf_p workbuf = thread_Kmat_workbuf[i];
        free(workbuf->int_buffer);
        free(workbuf->vec_out_idx);
        free(workbuf->nvi_nnz_row_idx);
        free(workbuf->nvo_nnz_row_idx);
        free(workbuf->dbl_buffer);
        free(workbuf);
    }
    free(thread_Kmat_workbuf);
    h2eri->thread_Kmat_workbuf = NULL;
}

// Find the minimal subtree that covers the given point indices
// idx must be in ascending order
static void H2ERI_find_minimal_cover_subtree(
    H2ERI_p h2eri, const int *idx, const int idx_len, 
    int *st_n_leaf_node_, int *st_leaf_nodes, int *st_node_flag, int *st_max_level_
)
{
    int n_leaf_node   = h2eri->n_leaf_node;
    int n_node        = h2eri->n_node;
    int max_level     = h2eri->max_level;
    int *parent       = h2eri->parent;
    int *leaf_nodes   = h2eri->height_nodes;
    int *mat_cluster  = h2eri->mat_cluster;
    int *level_n_node = h2eri->level_n_node;
    int *level_nodes  = h2eri->level_nodes;

    // Find all leaf nodes in the minimal cover subtree
    int st_n_leaf_node = 0;
    int j_start = 0;
    for (int i = 0; i < n_leaf_node; i++)
    {
        int node = leaf_nodes[i];
        int mat_cluster_s = mat_cluster[2 * node];
        int mat_cluster_e = mat_cluster[2 * node + 1];
        for (int j = j_start; j < idx_len; j++)
        {
            if ((mat_cluster_s <= idx[j]) && (idx[j] <= mat_cluster_e))
            {
                st_leaf_nodes[st_n_leaf_node] = node;
                st_n_leaf_node++;
                j_start = j;
                break;
            }
        }
    }
    *st_n_leaf_node_ = st_n_leaf_node;

    // Upward pass to mark the whole subtree
    memset(st_node_flag, 0, sizeof(int) * n_node);
    for (int i = 0; i < st_n_leaf_node; i++)
        st_node_flag[st_leaf_nodes[i]] = 1;
    int st_max_level = 0;
    for (int i = max_level; i >= 1; i--)
    {
        int *level_i_nodes = level_nodes + i * n_leaf_node;
        int level_i_n_node = level_n_node[i];
        for (int j = 0; j < level_i_n_node; j++)
        {
            int node = level_i_nodes[j];
            if (st_node_flag[node] == 1)
            {
                st_node_flag[parent[node]] = 1;
                if (i > st_max_level) st_max_level = i;
            }
        }
    }
    *st_max_level_ = st_max_level;
}

// Find the intersect of two ascending arrays and the corresponding indices in the 1st array
static void H2ERI_find_intersect(
    const int *a, const int a_len, const int *b, const int b_len,
    int *intersect_len_, int *intersect, int *a_idx
)
{
    int cnt = 0, i = 0, j = 0;
    while ((i < a_len) && (j < b_len))
    {
        if (a[i] == b[j])
        {
            intersect[cnt] = a[i];
            a_idx[cnt] = i;
            i++; j++;
            cnt++;
        } else {
            int ai = a[i];
            int bj = b[j];
            if (ai < bj) i++; else j++;
        }
    }
    *intersect_len_ = cnt;
}

// Update a Kmat_workbuf structure with a selected (M_list[], N|
static void H2ERI_exchange_workbuf_update_MN_list(
    H2ERI_p h2eri, Kmat_workbuf_p workbuf, const int N,
    const int num_M, const int *M_list, const int *MN_pair_idx
)
{
    int nshell = h2eri->nshell;
    int *shell_bf_sidx = h2eri->shell_bf_sidx;

    // (M, N| requires M < N, appears in shell pair list as:
    //   (M_list[k], N| for any k <  workbuf->M_cut,
    //   (N, M_list[k]| for any k >= workbuf->M_cut.
    int N_nbf = shell_bf_sidx[N + 1] - shell_bf_sidx[N];
    for (int i = 0; i < num_M; i++)
    {
        if (M_list[i] <= N) continue;
        workbuf->M_cut = i;
        break;
    }
    if (M_list[num_M - 1] <= N) workbuf->M_cut = num_M;

    // Offset for the output vector indexed by M_list * N
    int *MN_bfp_sidx = workbuf->MN_bfp_sidx;
    memset(MN_bfp_sidx, 0, sizeof(int) * (nshell + 1));
    for (int i = 0; i < num_M; i++)
    {
        int M = M_list[i];
        int M_nbf = shell_bf_sidx[M + 1] - shell_bf_sidx[M];
        MN_bfp_sidx[i + 1] = MN_bfp_sidx[i] + M_nbf * N_nbf;
    }

    // Row indices of M_list * N out of shell pair
    int *sp_bfp_sidx = h2eri->sp_bfp_sidx;
    int *row_idx = workbuf->row_idx;
    int row_idx_len = 0;
    for (int i = 0; i < num_M; i++)
    {
        int M = M_list[i];
        int pair_idx = MN_pair_idx[i];
        int num_bfp = sp_bfp_sidx[pair_idx + 1] - sp_bfp_sidx[pair_idx];
        for (int j = 0; j < num_bfp; j++)
            row_idx[row_idx_len + j] = sp_bfp_sidx[pair_idx] + j;
        row_idx_len += num_bfp;
    }
    workbuf->row_idx_len = row_idx_len;

    // Find the minimal covering subtree for row nodes
    int *row_idx_ascend = workbuf->row_idx_ascend;
    int *row_idx_pmt    = workbuf->row_idx_pmt;
    for (int i = 0; i < row_idx_len; i++)
    {
        row_idx_ascend[i] = row_idx[i];
        row_idx_pmt[i]    = i;
    }
    H2E_qsort_int_kv_ascend(row_idx_ascend, row_idx_pmt, 0, row_idx_len - 1);
    H2ERI_find_minimal_cover_subtree(
        h2eri, row_idx_ascend, row_idx_len, 
        &workbuf->row_n_leaf_node, workbuf->row_leaf_nodes, 
        workbuf->row_node_flag, &workbuf->row_max_level
    );

    // Calculate the sizes and offsets of vec_out and nvo_nnz_row
    int n_node            = h2eri->n_node;
    int row_n_leaf_node   = workbuf->row_n_leaf_node;
    int vec_out_idx_size  = 0;
    int *mat_cluster      = h2eri->mat_cluster;
    int *row_leaf_nodes   = workbuf->row_leaf_nodes;
    int *vec_out_sidx     = workbuf->vec_out_sidx;
    int *nvo_nnz_row_sidx = workbuf->nvo_nnz_row_sidx;
    vec_out_idx_size      = 0;
    memset(vec_out_sidx,     0, sizeof(int) * (n_node + 1));
    memset(nvo_nnz_row_sidx, 0, sizeof(int) * (n_node + 1));
    for (int i = 0; i < row_n_leaf_node; i++)
    {
        int node          = row_leaf_nodes[i];
        int nnz_node      = 0;
        int mat_cluster_s = mat_cluster[2 * node];
        int mat_cluster_e = mat_cluster[2 * node + 1];
        for (int j = 0; j < row_idx_len; j++)
            if ((mat_cluster_s <= row_idx_ascend[j]) && (row_idx_ascend[j] <= mat_cluster_e)) nnz_node++;
        vec_out_sidx[node + 1]     = nnz_node;
        nvo_nnz_row_sidx[node + 1] = nnz_node;
        vec_out_idx_size += nnz_node;
    }
    for (int i = 1; i <= n_node; i++)
    {
        vec_out_sidx[i]     += vec_out_sidx[i - 1];
        nvo_nnz_row_sidx[i] += nvo_nnz_row_sidx[i - 1];
    }

    // Calculate vec_out_idx and nvo_nnz_row_idx
    if (vec_out_idx_size > workbuf->vec_out_idx_size)
    {
        free(workbuf->vec_out_idx);
        free(workbuf->nvo_nnz_row_idx);
        workbuf->vec_out_idx_size  = vec_out_idx_size;
        workbuf->nvo_nnz_ridx_size = vec_out_idx_size;
        workbuf->vec_out_idx     = (int *) malloc(sizeof(int) * vec_out_idx_size);
        workbuf->nvo_nnz_row_idx = (int *) malloc(sizeof(int) * vec_out_idx_size);
        ASSERT_PRINTF(workbuf->vec_out_idx     != NULL, "Failed to allocate vec_out_idx     of size %d\n", vec_out_idx_size);
        ASSERT_PRINTF(workbuf->nvo_nnz_row_idx != NULL, "Failed to allocate nvo_nnz_row_idx of size %d\n", vec_out_idx_size);
    }
    int *vec_out_idx      = workbuf->vec_out_idx;
    int *nvo_nnz_row_idx  = workbuf->nvo_nnz_row_idx;
    for (int i = 0; i < row_n_leaf_node; i++)
    {
        int node          = row_leaf_nodes[i];
        int nnz_node      = 0;
        int mat_cluster_s = mat_cluster[2 * node];
        int mat_cluster_e = mat_cluster[2 * node + 1];
        int *vec_out_idx_node     = vec_out_idx     + vec_out_sidx[node];
        int *nvo_nnz_row_idx_node = nvo_nnz_row_idx + nvo_nnz_row_sidx[node];
        for (int j = 0; j < row_idx_len; j++)
        {
            if (mat_cluster_s <= row_idx_ascend[j] && row_idx_ascend[j] <= mat_cluster_e)
            {
                int idx1 = row_idx_pmt[j];
                int idx2 = row_idx_ascend[j] - mat_cluster_s;
                vec_out_idx_node[nnz_node]     = idx1;
                nvo_nnz_row_idx_node[nnz_node] = idx2;
                nnz_node++;
            }
        }  // End of j loop
    }  // End of i loop
}

// Update a Kmat_workbuf structure with a selected |P_list[], Q)
static void H2ERI_exchange_workbuf_update_PQ_list(
    H2ERI_p h2eri, Kmat_workbuf_p workbuf, const int Q,
    const int num_P0, const int *P_list0, const int *PQ_pair_idx, 
    const int num_D,  const int *D_list
)
{
    int nshell = h2eri->nshell;
    int *shell_bf_sidx = h2eri->shell_bf_sidx;

    // Filter out significant P shells
    int num_P   = 0;
    int *P_list = workbuf->P_list;
    int *P_idx  = workbuf->P_idx;
    H2ERI_find_intersect(
        P_list0, num_P0, D_list, num_D,
        &num_P, P_list, P_idx
    );
    workbuf->P_list_len = num_P;
    if (num_P == 0) return;

    // |P, Q) requires P < Q, appears in shell pair list as:
    //   |P_list[k], Q) for any k <  workbuf->P_cut,
    //   |Q, P_list[k]) for any k >= workbuf->P_cut.
    int Q_nbf = shell_bf_sidx[Q + 1] - shell_bf_sidx[Q];
    for (int i = 0; i < num_P; i++)
    {
        if (P_list[i] <= Q) continue;
        workbuf->P_cut = i;
        break;
    }
    if (P_list[num_P - 1] <= Q) workbuf->P_cut = num_P;

    // Column indices of P_list * Q out of shell pair
    int *sp_bfp_sidx = h2eri->sp_bfp_sidx;
    int *col_idx = workbuf->col_idx;
    int col_idx_len = 0;
    for (int i = 0; i < num_P; i++)
    {
        int P = P_list[i];
        int pair_idx = PQ_pair_idx[P_idx[i]];
        int num_bfp = sp_bfp_sidx[pair_idx + 1] - sp_bfp_sidx[pair_idx];
        for (int j = 0; j < num_bfp; j++)
            col_idx[col_idx_len + j] = sp_bfp_sidx[pair_idx] + j;
        col_idx_len += num_bfp;
    }
    workbuf->col_idx_len = col_idx_len;

    // Find the minimal covering subtree for column nodes
    int *col_idx_pmt  = workbuf->col_idx_pmt;
    int *col_idx_ipmt = workbuf->col_idx_ipmt;
    for (int i = 0; i < col_idx_len; i++) col_idx_pmt[i] = i;
    H2E_qsort_int_kv_ascend(col_idx, col_idx_pmt, 0, col_idx_len - 1);
    for (int i = 0; i < col_idx_len; i++) col_idx_ipmt[col_idx_pmt[i]] = i;
    H2ERI_find_minimal_cover_subtree(
        h2eri, col_idx, col_idx_len, 
        &workbuf->col_n_leaf_node, workbuf->col_leaf_nodes, 
        workbuf->col_node_flag, &workbuf->col_max_level
    );

    // Calculate the sizes and offsets of nvi_nnz_row
    int n_node            = h2eri->n_node;
    int col_n_leaf_node   = workbuf->col_n_leaf_node;
    int nvi_nnz_ridx_size = 0;
    int *mat_cluster      = h2eri->mat_cluster;
    int *col_leaf_nodes   = workbuf->col_leaf_nodes;
    int *nvi_nnz_row_sidx = workbuf->nvi_nnz_row_sidx;
    memset(nvi_nnz_row_sidx, 0, sizeof(int) * (n_node + 1));
    for (int i = 0; i < col_n_leaf_node; i++)
    {
        int node          = col_leaf_nodes[i];
        int nnz_node      = 0;
        int mat_cluster_s = mat_cluster[2 * node];
        int mat_cluster_e = mat_cluster[2 * node + 1];
        for (int j = 0; j < col_idx_len; j++)
            if ((mat_cluster_s <= col_idx[j]) && (col_idx[j] <= mat_cluster_e)) nnz_node++;
        nvi_nnz_row_sidx[node + 1] = nnz_node;
        nvi_nnz_ridx_size += nnz_node;
    }
    for (int i = 1; i <= n_node; i++)
        nvi_nnz_row_sidx[i] += nvi_nnz_row_sidx[i - 1];

    // Calculate nvi_nnz_row_idx
    if (nvi_nnz_ridx_size > workbuf->nvi_nnz_ridx_size)
    {
        free(workbuf->nvi_nnz_row_idx);
        workbuf->nvi_nnz_ridx_size = nvi_nnz_ridx_size;
        workbuf->nvi_nnz_row_idx = (int *) malloc(sizeof(int) * nvi_nnz_ridx_size);
        ASSERT_PRINTF(workbuf->nvi_nnz_row_idx != NULL, "Failed to allocate nvi_nnz_row_idx of size %d\n", nvi_nnz_ridx_size);
    }
    int *nvi_nnz_row_idx = workbuf->nvi_nnz_row_idx;
    for (int i = 0; i < col_n_leaf_node; i++)
    {
        int node          = col_leaf_nodes[i];
        int mat_cluster_s = mat_cluster[2 * node];
        int mat_cluster_e = mat_cluster[2 * node + 1];
        int nnz_node      = 0;
        int *nvi_nnz_row_idx_node = nvi_nnz_row_idx + nvi_nnz_row_sidx[node];
        for (int j = 0; j < col_idx_len; j++)
        {
            if ((mat_cluster_s <= col_idx[j]) && (col_idx[j] <= mat_cluster_e))
            {
                nvi_nnz_row_idx_node[nnz_node] = col_idx[j] - mat_cluster_s;
                nnz_node++;
            }
        }
    }
}

// Allocate dbl_buffer and assign corresponding pointers
static void H2ERI_exchange_workbuf_alloc_dbl_buffer(
    H2ERI_p h2eri, Kmat_workbuf_p workbuf, const int N, const int Q,
    const int num_M, const int *M_list
)
{
    int n_node = h2eri->n_node;
    int *shell_bf_sidx = h2eri->shell_bf_sidx;
    int *mat_cluster   = h2eri->mat_cluster;

    int N_nbf = shell_bf_sidx[N + 1] - shell_bf_sidx[N];
    int Q_nbf = shell_bf_sidx[Q + 1] - shell_bf_sidx[Q];
    int nvec  = N_nbf * Q_nbf;
    workbuf->nvec = nvec;

    // Calculate vec_in_size
    int vec_in_nrow = 0;
    int num_P   = workbuf->P_list_len;
    int *P_list = workbuf->P_list;
    for (int i = 0; i < num_P; i++)
    {
        int P = P_list[i];
        int P_nbf = shell_bf_sidx[P + 1] - shell_bf_sidx[P];
        vec_in_nrow += P_nbf * Q_nbf;
    }
    int vec_in_size = vec_in_nrow * nvec;

    // Calculate vec_out_size
    int vec_out_nrow = 0;
    for (int i = 0; i < num_M; i++)
    {
        int M = M_list[i];
        int M_nbf = shell_bf_sidx[M + 1] - shell_bf_sidx[M];
        vec_out_nrow += M_nbf * N_nbf;
    }
    int vec_out_size = vec_out_nrow * nvec;

    // nv{i, o}_nnz are of the same sizes as vec_{in, out}
    int nvi_nnz_size = vec_in_size;
    int nvo_nnz_size = vec_out_size;

    // Set up nvi_nnz_sidx and nvo_nnz_sidx
    int *nvi_nnz_sidx     = workbuf->nvi_nnz_sidx;
    int *nvo_nnz_sidx     = workbuf->nvo_nnz_sidx;
    int *nvi_nnz_row_sidx = workbuf->nvi_nnz_row_sidx;
    int *nvo_nnz_row_sidx = workbuf->nvo_nnz_row_sidx;
    for (int i = 0; i <= h2eri->n_node; i++)
    {
        nvi_nnz_sidx[i] = nvi_nnz_row_sidx[i] * nvec;
        nvo_nnz_sidx[i] = nvo_nnz_row_sidx[i] * nvec;
    }

    // Calculate y0_size
    int y0_size = 0;
    int *col_node_flag = workbuf->col_node_flag;
    int *y0_sidx = workbuf->y0_sidx;
    H2E_dense_mat_p *U = h2eri->U;
    y0_sidx[0] = 0;
    for (int i = 0; i < n_node; i++)
    {
        if (col_node_flag[i] == 1) y0_size += U[i]->ncol * nvec;
        y0_sidx[i + 1] = y0_size;
    }

    // Calculate y1_size
    int y1_size = 0;
    int *row_node_flag = workbuf->row_node_flag;
    int *y1_sidx = workbuf->y1_sidx;
    for (int i = 0; i < n_node; i++)
    {
        if (row_node_flag[i] == 1) y1_size += U[i]->ncol * nvec;
        y1_sidx[i + 1] = y1_size;
    }

    int tmp_K_size = h2eri->max_shell_nbf;

    // Allocate double buffer and assign pointers 
    int dbl_buffer_size = 0;
    dbl_buffer_size += vec_in_size + vec_out_size;
    dbl_buffer_size += nvi_nnz_size + nvo_nnz_size;
    dbl_buffer_size += y0_size + y1_size;
    dbl_buffer_size += tmp_K_size;
    if (dbl_buffer_size > workbuf->dbl_buffer_size)
    {
        free(workbuf->dbl_buffer);
        workbuf->dbl_buffer_size = dbl_buffer_size;
        workbuf->dbl_buffer = (double *) malloc(sizeof(double) * dbl_buffer_size);
        ASSERT_PRINTF(workbuf->dbl_buffer != NULL, "Failed to allocate dbl_buffer of size %d\n", dbl_buffer_size);
    }
    memset(workbuf->dbl_buffer, 0, sizeof(double) * dbl_buffer_size);
    workbuf->vec_in  = workbuf->dbl_buffer;
    workbuf->vec_out = workbuf->vec_in  + vec_in_size;
    workbuf->nvi_nnz = workbuf->vec_out + vec_out_size;
    workbuf->nvo_nnz = workbuf->nvi_nnz + nvi_nnz_size;
    workbuf->y0      = workbuf->nvo_nnz + nvo_nnz_size;
    workbuf->y1      = workbuf->y0      + y0_size;
    workbuf->tmp_K   = workbuf->y1      + y1_size;
}

// Gather H2 partial matmul input vectors from density matrix
static void H2ERI_build_exchange_gather_vec_in(
    H2ERI_p h2eri, Kmat_workbuf_p workbuf, const int N, const int Q,
    const double *den_mat
)
{
    int *shell_bf_sidx = h2eri->shell_bf_sidx;
    int N_bf_sidx      = shell_bf_sidx[N];
    int N_nbf          = shell_bf_sidx[N + 1] - shell_bf_sidx[N];
    int Q_nbf          = shell_bf_sidx[Q + 1] - shell_bf_sidx[Q];
    int nvec           = N_nbf * Q_nbf;

    // From density matrix to vec_in
    int    P_cut         = workbuf->P_cut;
    int    num_bf        = h2eri->num_bf;
    int    num_P         = workbuf->P_list_len;
    int    curr_row      = 0;
    int    *P_list       = workbuf->P_list;
    int    *col_idx_ipmt = workbuf->col_idx_ipmt;
    double *vec_in       = workbuf->vec_in;
    for (int i = 0; i < num_P; i++)
    {
        int P = P_list[i];
        int P_bf_sidx = shell_bf_sidx[P];
        int P_bf_eidx = shell_bf_sidx[P + 1];
        int P_nbf = P_bf_eidx - P_bf_sidx;
        const double *den_mat_ptr = den_mat + P_bf_sidx * num_bf + N_bf_sidx;
        for (int j = 0; j < Q_nbf; j++)
        {
            int row_idx_s, row_idx_e, row_idx_inc, col_idx_s;
            if (i < P_cut)
            {
                row_idx_s   = curr_row + j * P_nbf + 0;
                row_idx_e   = curr_row + j * P_nbf + P_nbf;
                row_idx_inc = 1;
            } else {
                row_idx_s   = curr_row + j + Q_nbf * 0;
                row_idx_e   = curr_row + j + Q_nbf * P_nbf;
                row_idx_inc = Q_nbf;
            }
            col_idx_s = j * N_nbf + 0;
            int k = 0;
            for (int row_idx0 = row_idx_s; row_idx0 < row_idx_e; row_idx0 += row_idx_inc)
            {
                int row_idx1 = col_idx_ipmt[row_idx0];
                double *dst = vec_in + row_idx1 * nvec + col_idx_s;
                const double *src = den_mat_ptr + k * num_bf;
                memcpy(dst, src, sizeof(double) * N_nbf);
                k++;
            }
        }  // End of j loop
        curr_row += P_nbf * Q_nbf;
    }  // End of i loop

    // Copy data from vec_in to nvi_nnz
    int    col_idx_len     = workbuf->col_idx_len;
    int    col_n_leaf_node = workbuf->col_n_leaf_node;
    int    *mat_cluster    = h2eri->mat_cluster;
    int    *col_leaf_nodes = workbuf->col_leaf_nodes;
    int    *col_idx        = workbuf->col_idx;
    int    *nvi_nnz_sidx   = workbuf->nvi_nnz_sidx;
    double *nvi_nnz        = workbuf->nvi_nnz;
    for (int i = 0; i < col_n_leaf_node; i++)
    {
        int    node          = col_leaf_nodes[i];
        int    mat_cluster_s = mat_cluster[2 * node];
        int    mat_cluster_e = mat_cluster[2 * node + 1];
        int    nnz_node      = 0;
        double *nvi_nnz_node = nvi_nnz + nvi_nnz_sidx[node];
        for (int j = 0; j < col_idx_len; j++)
        {
            if ((mat_cluster_s <= col_idx[j]) && (col_idx[j] <= mat_cluster_e))
            {
                double *dst = nvi_nnz_node + nnz_node * nvec;
                double *src = vec_in       + j * nvec;
                memcpy(dst, src, sizeof(double) * nvec);
                nnz_node++;
            }
        }
    }
}

// Scatter H2 partial matmul output vectors to exchange matrix
static void H2ERI_build_exchange_scatter_vec_out(
    H2ERI_p h2eri, Kmat_workbuf_p workbuf, const int N, const int Q,
    const int num_M, const int *M_list, double *K_mat
)
{
    int *shell_bf_sidx = h2eri->shell_bf_sidx;
    int N_nbf          = shell_bf_sidx[N + 1] - shell_bf_sidx[N];
    int Q_nbf          = shell_bf_sidx[Q + 1] - shell_bf_sidx[Q];
    int nvec           = N_nbf * Q_nbf;

    // Copy data from nvo_nnz to vec_out
    int    row_n_leaf_node = workbuf->row_n_leaf_node;
    int    *row_leaf_nodes = workbuf->row_leaf_nodes;
    int    *vec_out_idx    = workbuf->vec_out_idx;
    int    *vec_out_sidx   = workbuf->vec_out_sidx;
    int    *nvo_nnz_sidx   = workbuf->nvo_nnz_sidx;
    double *vec_out        = workbuf->vec_out;
    double *nvo_nnz        = workbuf->nvo_nnz;
    for (int i = 0; i < row_n_leaf_node; i++)
    {
        int    node            = row_leaf_nodes[i];
        int    vec_out_idx_len = vec_out_sidx[node + 1] - vec_out_sidx[node];
        int    *vec_out_idx_i  = vec_out_idx + vec_out_sidx[node];
        double *nvo_nnz_node   = nvo_nnz + nvo_nnz_sidx[node];
        for (int j = 0; j < vec_out_idx_len; j++)
        {
            double *dst = vec_out + vec_out_idx_i[j] * nvec;
            double *src = nvo_nnz_node + j * nvec;
            memcpy(dst, src, sizeof(double) * nvec);
        }
    }

    // From vec_out to K_mat
    int    num_bf       = h2eri->num_bf;
    int    M_cut        = workbuf->M_cut;
    int    *MN_bfp_sidx = workbuf->MN_bfp_sidx;
    double *tmp_K       = workbuf->tmp_K;
    for (int i = 0; i < num_M; i++)
    {
        int M = M_list[i];
        int M_bf_sidx = shell_bf_sidx[M];
        int M_bf_eidx = shell_bf_sidx[M + 1];
        int M_nbf = M_bf_eidx - M_bf_sidx;
        int row_idx0_s, row_idx0_e, s0, s1, s2;
        if (i < M_cut)
        {
            row_idx0_s = MN_bfp_sidx[i];
            row_idx0_e = MN_bfp_sidx[i] + M_nbf;
            s0 = 1;
            s1 = 0;
            s2 = M_nbf;
        } else {
            row_idx0_s = MN_bfp_sidx[i];
            row_idx0_e = MN_bfp_sidx[i] + M_nbf * N_nbf;
            s0 = N_nbf;
            s1 = 0;
            s2 = 1;
        }
        for (int j = 0; j < Q_nbf; j++)
        {
            memset(tmp_K, 0, sizeof(double) * M_nbf);
            for (int k = 0; k < N_nbf; k++)
            {
                int row_idx0_s_k = row_idx0_s + (k - s1) * s2;
                int vec_out_col = j * N_nbf + k;
                for (int l = 0; l < M_nbf; l++)
                {
                    int vec_out_row_offset = (row_idx0_s_k + l * s0) * nvec;
                    tmp_K[l] += vec_out[vec_out_row_offset + vec_out_col];
                }
            }  // End of k loop
            for (int l = 0; l < M_nbf; l++)
            {
                int K_mat_col = shell_bf_sidx[Q] + j;
                int K_mat_row_offset = (M_bf_sidx + l) * num_bf;
                double *K_mat_ptr = K_mat + K_mat_row_offset + K_mat_col;
                atomic_add_f64(K_mat_ptr, tmp_K[l]);
            }
        }  // End of j loop
    }  // End of i loop
}

// Build dlist according to density matrix
static void H2ERI_build_exchange_dlist(H2ERI_p h2eri, const double *den_mat)
{
    int num_bf = h2eri->num_bf;
    int nshell = h2eri->nshell;
    if (h2eri->dlist_sidx == NULL)
    {
        h2eri->dlist_sidx = (int *) malloc(sizeof(int) * (nshell + 1));
        ASSERT_PRINTF(h2eri->dlist_sidx != NULL, "Failed to allocate dlist_sidx of size %d\n", nshell + 1);
    }
    int *dlist_sidx = h2eri->dlist_sidx;

    int *shell_bf_sidx = h2eri->shell_bf_sidx;
    int *dlist0        = (int *) malloc(sizeof(int) * nshell * nshell);
    int *dlist_cnt     = (int *) malloc(sizeof(int) * nshell);
    ASSERT_PRINTF(dlist0    != NULL, "Failed to allocate dlist0    of size %d\n", nshell * nshell);
    ASSERT_PRINTF(dlist_cnt != NULL, "Failed to allocate dlist_cnt of size %d\n", nshell);
    const double DTOL = 1e-10;
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < nshell; i++)
    {
        int srow = shell_bf_sidx[i];
        int erow = shell_bf_sidx[i + 1];
        int cnt  = 0;
        int *dlist0_i = dlist0 + i * nshell;
        for (int j = 0; j < nshell; j++)
        {
            int scol = shell_bf_sidx[j];
            int ecol = shell_bf_sidx[j + 1];
            int flag = 0;
            for (int irow = srow; irow < erow; irow++)
            {
                const double *den_mat_irow = den_mat + irow * num_bf;
                for (int icol = scol; icol < ecol; icol++)
                {
                    if (fabs(den_mat_irow[icol]) > DTOL)
                    {
                        flag = 1;
                        break;
                    }
                }
                if (flag == 1) break;
            }
            if (flag == 1)
            {
                dlist0_i[cnt] = j;
                cnt++;
            }
        }  // End of j loop
        dlist_cnt[i] = cnt;
    }  // End of i loop

    dlist_sidx[0] = 0;
    for (int i = 1; i <= nshell; i++)
        dlist_sidx[i] = dlist_sidx[i - 1] + dlist_cnt[i - 1];
    int dlist_size = dlist_sidx[nshell];
    free(h2eri->dlist);
    int *dlist = (int *) malloc(sizeof(int) * dlist_size);
    ASSERT_PRINTF(dlist != NULL, "Failed to allocate dlist of size %d\n", dlist_size);
    h2eri->dlist = dlist;
    for (int i = 0; i < nshell; i++)
        memcpy(dlist + dlist_sidx[i], dlist0 + i * nshell, sizeof(int) * dlist_cnt[i]);

    free(dlist_cnt);
    free(dlist0);
}

// Copy a sub-matrix of specified rows and columns
static void H2ERI_copy_submatrix(
    const double *mat, const int ldm, const int ncol, double *submat, const int lds,
    const int submat_nrow, const int *submat_row_idx, const int submat_ncol, const int *submat_col_idx
)
{
    if (ncol > submat_ncol)
    {
        for (int i = 0; i < submat_nrow; i++)
        {
            const double *mat_row_i = mat + submat_row_idx[i] * ldm;
            double *submat_row_i = submat + i * lds;
            for (int j = 0; j < submat_ncol; j++)
                submat_row_i[j] = mat_row_i[submat_col_idx[j]];
        }
    } else {
        size_t row_bytes = sizeof(double) * ncol;
        for (int i = 0; i < submat_nrow; i++)
        {
            const double *mat_row_i = mat + submat_row_idx[i] * ldm;
            double *submat_row_i = submat + i * lds;
            memcpy(submat_row_i, mat_row_i, row_bytes);
        }
    }
}

// Perform matmul for a sub-matrix of B or D block blk which might 
// be a dense block or a low-rank approximation blk = U * V
static void H2ERI_BD_blk_submat_matmul(
    const int trans_blk, H2E_dense_mat_p blk, H2E_dense_mat_p tmp_mat,
    int *tmp_idx, const double *mat_in, double *mat_out, const int nvec,
    const int submat_nrow, const int *submat_row_idx, 
    const int submat_ncol, const int *submat_col_idx
)
{
    if (blk->ld > 0)
    {
        if (trans_blk == 0)
        {
            H2E_dense_mat_resize(tmp_mat, submat_nrow, submat_ncol);
            H2ERI_copy_submatrix(
                blk->data, blk->ld, blk->ncol, tmp_mat->data, tmp_mat->ld, 
                submat_nrow, submat_row_idx, submat_ncol, submat_col_idx
            );
            CBLAS_GEMM(
                CblasRowMajor, CblasNoTrans, CblasNoTrans, submat_nrow, nvec, submat_ncol,
                1.0, tmp_mat->data, tmp_mat->ld, mat_in, nvec, 1.0, mat_out, nvec
            );
        } else {
            H2E_dense_mat_resize(tmp_mat, submat_ncol, submat_nrow);
            H2ERI_copy_submatrix(
                blk->data, blk->ld, blk->ncol, tmp_mat->data, tmp_mat->ld, 
                submat_ncol, submat_col_idx, submat_nrow, submat_row_idx
            );
            CBLAS_GEMM(
                CblasRowMajor, CblasTrans, CblasNoTrans, submat_nrow, nvec, submat_ncol,
                1.0, tmp_mat->data, tmp_mat->ld, mat_in, nvec, 1.0, mat_out, nvec
            );
        }
    } else {
        int    blk_rank = -blk->ld;
        double *U_mat   = blk->data;
        double *VT_mat  = U_mat + blk->nrow * blk_rank;
        for (int i = 0; i < blk_rank; i++) tmp_idx[i] = i;
        // U: blk->nrow * blk_rank
        // V: blk_rank  * blk->ncol
        // Note: V^T instead of V is stored
        if (trans_blk == 0)
        {
            // mat_out = (U * V) * mat_in = U * (V * mat_in)
            int tmp_mat_size = blk_rank * nvec;      // tmp_v
            tmp_mat_size += blk_rank * submat_ncol;  // Sub-matrix of V
            tmp_mat_size += submat_nrow * blk_rank;  // Sub-matrix of U
            H2E_dense_mat_resize(tmp_mat, blk_rank, nvec);
            double *tmp_v = tmp_mat->data;
            double *VT_submat = tmp_v + blk_rank * nvec;
            double *U_submat  = VT_submat + blk_rank * submat_ncol;
            
            H2ERI_copy_submatrix(
                VT_mat, blk_rank, blk_rank, VT_submat, blk_rank, 
                submat_ncol, submat_col_idx, blk_rank, tmp_idx
            );
            H2ERI_copy_submatrix(
                U_mat, blk_rank, blk_rank, U_submat, blk_rank, 
                submat_nrow, submat_row_idx, blk_rank, tmp_idx
            );

            CBLAS_GEMM(
                CblasRowMajor, CblasTrans, CblasNoTrans, blk_rank, nvec, submat_ncol,
                1.0, VT_submat, blk_rank, mat_in, nvec, 0.0, tmp_v, nvec
            );
            CBLAS_GEMM(
                CblasRowMajor, CblasNoTrans, CblasNoTrans, submat_nrow, nvec, blk_rank,
                1.0, U_submat, blk_rank, tmp_v, nvec, 1.0, mat_out, nvec
            );
        } else {
            // mat_out = (U * V)^T * mat_in = V^T * (U^T * mat_in)
            int tmp_mat_size = blk_rank * nvec;      // tmp_v
            tmp_mat_size += submat_ncol * blk_rank;  // Sub-matrix of U
            tmp_mat_size += blk_rank * submat_nrow;  // Sub-matrix of V
            
            H2E_dense_mat_resize(tmp_mat, blk_rank, nvec);
            double *tmp_v = tmp_mat->data;
            double *U_submat  = tmp_v + blk_rank * nvec;
            double *VT_submat = U_submat + submat_ncol * blk_rank;

            H2ERI_copy_submatrix(
                U_mat, blk_rank, blk_rank, U_submat, blk_rank, 
                submat_ncol, submat_col_idx, blk_rank, tmp_idx
            );
            CBLAS_GEMM(
                CblasRowMajor, CblasTrans, CblasNoTrans, blk_rank, nvec, submat_ncol,
                1.0, U_submat, blk_rank, mat_in, nvec, 0.0, tmp_v, nvec
            );

            H2ERI_copy_submatrix(
                VT_mat, blk_rank, blk_rank, VT_submat, blk_rank,
                submat_nrow, submat_row_idx, blk_rank, tmp_idx
            );
            CBLAS_GEMM(
                CblasRowMajor, CblasNoTrans, CblasNoTrans, submat_nrow, nvec, blk_rank,
                1.0, VT_submat, blk_rank, tmp_v, nvec, 1.0, mat_out, nvec
            );
        }
    }
}

// Perform matmul for a B or D block blk which might be a dense block 
// or a low-rank approximation blk = U * V
static void H2ERI_BD_blk_matmul(
    const int trans_blk, H2E_dense_mat_p blk, H2E_dense_mat_p tmp_v,
    const double *mat_in, double *mat_out, const int nvec
)
{
    if (blk->ld > 0)
    {
        if (trans_blk == 0)
        {
            CBLAS_GEMM(
                CblasRowMajor, CblasNoTrans, CblasNoTrans, blk->nrow, nvec, blk->ncol,
                1.0, blk->data, blk->ld, mat_in, nvec, 1.0, mat_out, nvec
            );
        } else {
            CBLAS_GEMM(
                CblasRowMajor, CblasTrans, CblasNoTrans, blk->ncol, nvec, blk->nrow,
                1.0, blk->data, blk->ld, mat_in, nvec, 1.0, mat_out, nvec
            );
        }
    } else {
        int    blk_rank = -blk->ld;
        double *U_mat   = blk->data;
        double *VT_mat  = U_mat + blk->nrow * blk_rank;
        // U: blk->nrow * blk_rank
        // V: blk_rank  * blk->ncol
        // Note: V^T instead of V is stored
        H2E_dense_mat_resize(tmp_v, blk_rank, nvec);
        if (trans_blk == 0)
        {
            // mat_out = (U * V) * mat_in = U * (V * mat_in)
            CBLAS_GEMM(
                CblasRowMajor, CblasTrans, CblasNoTrans, blk_rank, nvec, blk->ncol,
                1.0, VT_mat, blk_rank, mat_in, nvec, 0.0, tmp_v->data, nvec
            );
            CBLAS_GEMM(
                CblasRowMajor, CblasNoTrans, CblasNoTrans, blk->nrow, nvec, blk_rank,
                1.0, U_mat, blk_rank, tmp_v->data, nvec, 1.0, mat_out, nvec
            );
        } else {
            // mat_out = (U * V)^T * mat_in = V^T * (U^T * mat_in)
            CBLAS_GEMM(
                CblasRowMajor, CblasTrans, CblasNoTrans, blk_rank, nvec, blk->nrow,
                1.0, U_mat, blk_rank, mat_in, nvec, 0.0, tmp_v->data, nvec
            );
            CBLAS_GEMM(
                CblasRowMajor, CblasNoTrans, CblasNoTrans, blk->ncol, nvec, blk_rank,
                1.0, VT_mat, blk_rank, tmp_v->data, nvec, 1.0, mat_out, nvec
            );
        }
    }
}

static void H2ERI_build_exchange_H2_matmul_partial(H2ERI_p h2eri, Kmat_workbuf_p workbuf, const int tid)
{
    int n_node         = h2eri->n_node;
    int max_child      = h2eri->max_child;
    int n_leaf_node    = h2eri->n_leaf_node;
    int n_r_adm_pair   = h2eri->n_r_adm_pair;
    int n_r_inadm_pair = h2eri->n_r_inadm_pair;
    int *children      = h2eri->children;
    int *n_child       = h2eri->n_child;
    int *level_n_node  = h2eri->level_n_node;
    int *level_nodes   = h2eri->level_nodes;
    int *node_level    = h2eri->node_level;
    int *leaf_nodes    = h2eri->height_nodes;
    int *r_adm_pairs   = h2eri->r_adm_pairs;
    int *r_inadm_pairs = h2eri->r_inadm_pairs;

    H2E_dense_mat_p  *U        = h2eri->U;
    H2E_dense_mat_p  *c_B_blks = h2eri->c_B_blks;
    H2E_dense_mat_p  *c_D_blks = h2eri->c_D_blks;
    H2E_dense_mat_p  tmp_mat   = h2eri->thread_buffs[tid]->mat0;
    H2E_int_vec_p    tmp_idx0  = h2eri->thread_buffs[tid]->idx0;
    H2E_int_vec_p    tmp_idx1  = h2eri->thread_buffs[tid]->idx1;

    int *node_adm_pairs        = h2eri->node_adm_pairs;
    int *node_adm_pairs_sidx   = h2eri->node_adm_pairs_sidx;
    int *node_inadm_pairs      = h2eri->node_inadm_pairs;
    int *node_inadm_pairs_sidx = h2eri->node_inadm_pairs_sidx;

    int    nvec                = workbuf->nvec;
    int    row_max_level       = workbuf->row_max_level;
    int    col_max_level       = workbuf->col_max_level;
    int    *row_node_flag      = workbuf->row_node_flag;
    int    *col_node_flag      = workbuf->col_node_flag;
    int    *nvi_nnz_row_idx    = workbuf->nvi_nnz_row_idx;
    int    *nvo_nnz_row_idx    = workbuf->nvo_nnz_row_idx;
    int    *nvi_nnz_row_sidx   = workbuf->nvi_nnz_row_sidx;
    int    *nvo_nnz_row_sidx   = workbuf->nvo_nnz_row_sidx;
    int    *nvi_nnz_sidx       = workbuf->nvi_nnz_sidx;
    int    *nvo_nnz_sidx       = workbuf->nvo_nnz_sidx;
    int    *y0_sidx            = workbuf->y0_sidx;
    int    *y1_sidx            = workbuf->y1_sidx;
    int    *tmp_idx2           = workbuf->tmp_arr;
    double *nvi_nnz            = workbuf->nvi_nnz;
    double *nvo_nnz            = workbuf->nvo_nnz;
    double *y0                 = workbuf->y0;
    double *y1                 = workbuf->y1;

    int *B_p2i_rowptr = h2eri->B_p2i_rowptr;
    int *B_p2i_colidx = h2eri->B_p2i_colidx;
    int *B_p2i_val    = h2eri->B_p2i_val;
    int *D_p2i_rowptr = h2eri->D_p2i_rowptr;
    int *D_p2i_colidx = h2eri->D_p2i_colidx;
    int *D_p2i_val    = h2eri->D_p2i_val;

    double st, et;
    double *timers = workbuf->timers;

    // y0, y1, vec_out have been set to zero in H2ERI_exchange_workbuf_alloc_dbl_buffer()

    // 1. Find forward & backward transformation minimal levels
    st = get_wtime_sec();
    int fwd_minlvl = 19241112, bwd_minlvl = 19241112;
    for (int node0 = 0; node0 < n_node; node0++)
    {
        if (row_node_flag[node0] == 0) continue;
        int node0_n_adm_pair = node_adm_pairs_sidx[node0 + 1] - node_adm_pairs_sidx[node0];
        int *node0_adm_pairs = node_adm_pairs + node_adm_pairs_sidx[node0];
        int level0 = node_level[node0];
        int cnt = 0;
        for (int j = 0; j < node0_n_adm_pair; j++)
        {
            int node1 = node0_adm_pairs[j];
            int level1 = node_level[node1];
            if (col_node_flag[node1] == 0) continue;
            if (level1 < fwd_minlvl) fwd_minlvl = level1;
            cnt++;
        }
        if ((cnt > 0) && (level0 < bwd_minlvl)) bwd_minlvl = level0;
    }
    if (fwd_minlvl < h2eri->min_adm_level) fwd_minlvl = h2eri->min_adm_level;
    if (bwd_minlvl < h2eri->min_adm_level) bwd_minlvl = h2eri->min_adm_level;
    int only_Dij = 0;
    if (col_max_level < fwd_minlvl) only_Dij = 1;
    if (row_max_level < bwd_minlvl) only_Dij = 1;
    et = get_wtime_sec();
    timers[BUILD_K_MM_FWD_TIMER_IDX] += et - st;

    // 2. Forward transformation
    st = get_wtime_sec();
    for (int i = col_max_level; i >= fwd_minlvl; i--)
    {
        if (only_Dij) continue;
        int *level_i_nodes = level_nodes + i * n_leaf_node;
        int level_i_n_node = level_n_node[i];
        for (int j = 0; j < level_i_n_node; j++)
        {
            int node = level_i_nodes[j];
            if (col_node_flag[node] == 0) continue;

            int child_cnt = 0;
            int U_srow = 0;
            int n_child_node = n_child[node];
            H2E_dense_mat_p U_node = U[node];
            double *y0_node = y0 + y0_sidx[node];
            int *node_children = children + node * max_child;
            for (int k = 0; k < n_child_node; k++)
            {
                int    child_k   = node_children[k];
                int    y0_k_nrow = U[child_k]->ncol;
                double *y0_k     = y0 + y0_sidx[child_k];
                double *U_node_k = U_node->data + U_srow * U_node->ld;
                if (col_node_flag[child_k])
                {
                    CBLAS_GEMM(
                        CblasRowMajor, CblasTrans, CblasNoTrans, U_node->ncol, nvec, y0_k_nrow,
                        1.0, U_node_k, U_node->ld, y0_k, nvec, 1.0, y0_node, nvec
                    );
                    child_cnt++;
                }
                U_srow += y0_k_nrow;
            }  // End of k loop
            if (child_cnt == 0)
            {
                double *B = nvi_nnz + nvi_nnz_sidx[node];
                int num_nnz_row = nvi_nnz_row_sidx[node + 1] - nvi_nnz_row_sidx[node];
                int *nnz_row = nvi_nnz_row_idx + nvi_nnz_row_sidx[node];
                H2E_dense_mat_resize(tmp_mat, num_nnz_row, U_node->ncol);
                for (int l = 0; l < num_nnz_row; l++)
                {
                    double *src = U_node->data + nnz_row[l] * U_node->ncol;
                    double *dst = tmp_mat->data + l * U_node->ncol;
                    memcpy(dst, src, sizeof(double) * U_node->ncol);
                }
                CBLAS_GEMM(
                    CblasRowMajor, CblasTrans, CblasNoTrans, tmp_mat->ncol, nvec, num_nnz_row,
                    1.0, tmp_mat->data, tmp_mat->ld, B, nvec, 1.0, y0_node, nvec
                );
            }
        }  // End of j loop
    }  // End of i loop
    et = get_wtime_sec();
    timers[BUILD_K_MM_FWD_TIMER_IDX] += et - st;

    // 3. Intermediate multiplication
    st = get_wtime_sec();
    for (int node0 = 0; node0 < n_node; node0++)
    {
        if (only_Dij) continue;
        if (row_node_flag[node0] == 0) continue;
        
        int node0_n_adm_pair = node_adm_pairs_sidx[node0 + 1] - node_adm_pairs_sidx[node0];
        int *node0_adm_pairs = node_adm_pairs + node_adm_pairs_sidx[node0];
        
        int node0_num_nnz_row  = nvo_nnz_row_sidx[node0 + 1] - nvo_nnz_row_sidx[node0];
        int *node0_nnz_row_idx = nvo_nnz_row_idx + nvo_nnz_row_sidx[node0];
        
        int y1_node0_nrow = (y1_sidx[node0 + 1] - y1_sidx[node0]) / nvec;
        H2E_int_vec_set_capacity(tmp_idx0, y1_node0_nrow);
        for (int k = 0; k < y1_node0_nrow; k++)
            tmp_idx0->data[k] = k;
        tmp_idx0->length = y1_node0_nrow;

        for (int j = 0; j < node0_n_adm_pair; j++)
        {
            int node1 = node0_adm_pairs[j];
            if (col_node_flag[node1] == 0) continue;

            int node1_num_nnz_row  = nvi_nnz_row_sidx[node1 + 1] - nvi_nnz_row_sidx[node1];
            int *node1_nnz_row_idx = nvi_nnz_row_idx + nvi_nnz_row_sidx[node1];

            int y0_node1_nrow = (y0_sidx[node1 + 1] - y0_sidx[node1]) / nvec;
            H2E_int_vec_set_capacity(tmp_idx1, y0_node1_nrow);
            for (int k = 0; k < y0_node1_nrow; k++)
                tmp_idx1->data[k] = k;
            tmp_idx1->length = y0_node1_nrow;

            int pair_idx_ij = H2E_get_int_CSR_elem(B_p2i_rowptr, B_p2i_colidx, B_p2i_val, node0, node1);
            ASSERT_PRINTF(pair_idx_ij != 0, "Cannot find Bij for i = %d, j = %d\n", node0, node1);
            int trans_blk;
            H2E_dense_mat_p Bi;
            if (pair_idx_ij > 0)
            {
                trans_blk = 0;
                Bi = c_B_blks[pair_idx_ij - 1];
            } else {
                trans_blk = 1;
                Bi = c_B_blks[-pair_idx_ij - 1];
            }

            int level0 = node_level[node0];
            int level1 = node_level[node1];

            // (1) Two nodes are of the same level, compress on both sides
            if (level0 == level1)
            {
                double *y0_node1 = y0 + y0_sidx[node1];
                double *y1_node0 = y1 + y1_sidx[node0];
                H2ERI_BD_blk_matmul(trans_blk, Bi, tmp_mat, y0_node1, y1_node0, nvec);
            }

            // (2) node1 is a leaf node and its level is higher than node0's level, 
            //     only compress on node0's side
            if (level0 > level1)
            {
                double *node1_vec_in = nvi_nnz + nvi_nnz_sidx[node1];
                double *y1_node0     = y1 + y1_sidx[node0];
                H2ERI_BD_blk_submat_matmul(
                    trans_blk, Bi, tmp_mat, tmp_idx2,
                    node1_vec_in, y1_node0, nvec,
                    tmp_idx0->length,  tmp_idx0->data,
                    node1_num_nnz_row, node1_nnz_row_idx
                );
            }

            // (3) node0 is a leaf node and its level is higher than node1's level, 
            //     only compress on node1's side
            if (level0 < level1)
            {
                double *y0_node1      = y0 + y0_sidx[node1];
                double *node0_vec_out = nvo_nnz + nvo_nnz_sidx[node0];
                H2ERI_BD_blk_submat_matmul(
                    trans_blk, Bi, tmp_mat, tmp_idx2, 
                    y0_node1, node0_vec_out, nvec,
                    node0_num_nnz_row, node0_nnz_row_idx, 
                    tmp_idx1->length,  tmp_idx1->data
                );
            }
        }  // End of j loop
    }  // End of node0 loop
    et = get_wtime_sec();
    timers[BUILD_K_MM_MID_TIMER_IDX] += et - st;

    // 4. Backward transformation
    st = get_wtime_sec();
    for (int i = bwd_minlvl; i <= row_max_level; i++)
    {
        if (only_Dij) continue;
        int *level_i_nodes = level_nodes + i * n_leaf_node;
        int level_i_n_node = level_n_node[i];
        for (int j = 0; j < level_i_n_node; j++)
        {
            int node = level_i_nodes[j];
            if (row_node_flag[node] == 0) continue;

            int child_cnt = 0;
            int U_srow = 0;
            int n_child_node = n_child[node];
            H2E_dense_mat_p U_node = U[node];
            double *y1_node = y1 + y1_sidx[node];
            int *node_children = children + node * max_child;
            for (int k = 0; k < n_child_node; k++)
            {
                int    child_k   = node_children[k];
                int    y1_k_nrow = U[child_k]->ncol;
                double *y1_k     = y1 + y1_sidx[child_k];
                double *U_node_k = U_node->data + U_srow * U_node->ld;
                if (row_node_flag[child_k])
                {
                    CBLAS_GEMM(
                        CblasRowMajor, CblasNoTrans, CblasNoTrans, y1_k_nrow, nvec, U_node->ncol,
                        1.0, U_node_k, U_node->ld, y1_node, nvec, 1.0, y1_k, nvec
                    );
                    child_cnt++;
                }
                U_srow += y1_k_nrow;
            }  // End of k loop
            if (child_cnt == 0)
            {
                double *C = nvo_nnz + nvo_nnz_sidx[node];
                int num_nnz_row = nvo_nnz_row_sidx[node + 1] - nvo_nnz_row_sidx[node];
                int *nnz_row = nvo_nnz_row_idx + nvo_nnz_row_sidx[node];
                H2E_dense_mat_resize(tmp_mat, num_nnz_row, U_node->ncol);
                for (int l = 0; l < num_nnz_row; l++)
                {
                    double *src = U_node->data + nnz_row[l] * U_node->ncol;
                    double *dst = tmp_mat->data + l * U_node->ncol;
                    memcpy(dst, src, sizeof(double) * U_node->ncol);
                }
                CBLAS_GEMM(
                    CblasRowMajor, CblasNoTrans, CblasNoTrans, num_nnz_row, nvec, tmp_mat->ncol,
                    1.0, tmp_mat->data, tmp_mat->ld, y1_node, nvec, 1.0, C, nvec
                );
            }
        }  // End of j loop
    }  // End of i loop
    et = get_wtime_sec();
    timers[BUILD_K_MM_BWD_TIMER_IDX] += et - st;

    // 5. Dense multiplication
    st = get_wtime_sec();
    for (int node0 = 0; node0 < n_node; node0++)
    {
        if (row_node_flag[node0] == 0) continue;
        double *node0_vec_out = nvo_nnz + nvo_nnz_sidx[node0];

        int node0_n_inadm_pair = node_inadm_pairs_sidx[node0 + 1] - node_inadm_pairs_sidx[node0];
        int *node0_inadm_pairs = node_inadm_pairs + node_inadm_pairs_sidx[node0];

        int node0_num_nnz_row  = nvo_nnz_row_sidx[node0 + 1] - nvo_nnz_row_sidx[node0];
        int *node0_nnz_row_idx = nvo_nnz_row_idx + nvo_nnz_row_sidx[node0];
        for (int j = 0; j < node0_n_inadm_pair; j++)
        {
            int node1 = node0_inadm_pairs[j];
            if (col_node_flag[node1] == 0) continue;
            double *node1_vec_in = nvi_nnz + nvi_nnz_sidx[node1];

            int node1_num_nnz_row  = nvi_nnz_row_sidx[node1 + 1] - nvi_nnz_row_sidx[node1];
            int *node1_nnz_row_idx = nvi_nnz_row_idx + nvi_nnz_row_sidx[node1];

            int pair_idx_ij = H2E_get_int_CSR_elem(D_p2i_rowptr, D_p2i_colidx, D_p2i_val, node0, node1);
            ASSERT_PRINTF(pair_idx_ij != 0, "Cannot find Dij for i = %d, j = %d\n", node0, node1);
            int trans_blk;
            H2E_dense_mat_p Di;
            if (pair_idx_ij > 0)
            {
                trans_blk = 0;
                Di = c_D_blks[pair_idx_ij - 1];
            } else {
                trans_blk = 1;
                Di = c_D_blks[-pair_idx_ij - 1];
            }

            H2ERI_BD_blk_submat_matmul(
                trans_blk, Di, tmp_mat, tmp_idx2,
                node1_vec_in, node0_vec_out, nvec,
                node0_num_nnz_row, node0_nnz_row_idx, 
                node1_num_nnz_row, node1_nnz_row_idx
            );
        }  // End of j loop
    }  // End of node0 loop
    et = get_wtime_sec();
    timers[BUILD_K_MM_DEN_TIMER_IDX] += et - st;
}

// Build the exchange matrix with the density matrix and H2 representation of the ERI tensor
void H2ERI_build_exchange(H2ERI_p h2eri, const double *den_mat, double *K_mat)
{
    ASSERT_PRINTF(h2eri->BD_JIT == 0, "H2ERI_build_exchange does not support BD JIT build\n");
    
    H2ERI_build_exchange_dlist(h2eri, den_mat);

    int num_bf      = h2eri->num_bf;
    int nshell      = h2eri->nshell;
    int n_thread    = h2eri->n_thread;
    int *plist      = h2eri->plist;
    int *plist_idx  = h2eri->plist_idx;
    int *plist_sidx = h2eri->plist_sidx;
    int *dlist      = h2eri->dlist;
    int *dlist_sidx = h2eri->dlist_sidx;

    BLAS_SET_NUM_THREADS(1);

    #pragma omp parallel for 
    for (int i = 0; i < num_bf * num_bf; i++) K_mat[i] = 0;

    Kmat_workbuf_p *thread_Kmat_workbuf = (Kmat_workbuf_p *) h2eri->thread_Kmat_workbuf;
    #pragma omp parallel num_threads(n_thread)
    {
        int tid = omp_get_thread_num();
        Kmat_workbuf_p workbuf = thread_Kmat_workbuf[tid];

        double st, et;
        double *timers = workbuf->timers;
        memset(timers, 0, sizeof(double) * 5);

        #pragma omp for schedule(dynamic)
        for (int N = 0; N < nshell; N++)
        {
            int num_M        = plist_sidx[N + 1] - plist_sidx[N];
            int *M_list      = plist     + plist_sidx[N];
            int *MN_pair_idx = plist_idx + plist_sidx[N];
            st = get_wtime_sec();
            H2ERI_exchange_workbuf_update_MN_list(h2eri, workbuf, N, num_M, M_list, MN_pair_idx);
            et = get_wtime_sec();
            timers[BUILD_K_AUX_TIMER_IDX] += et - st;

            int num_D   = dlist_sidx[N + 1] - dlist_sidx[N];
            int *D_list = dlist + dlist_sidx[N];
            for (int Q = 0; Q < nshell; Q++)
            {
                int num_P0       = plist_sidx[Q + 1] - plist_sidx[Q];
                int *P_list0     = plist + plist_sidx[Q];
                int *PQ_pair_idx = plist_idx + plist_sidx[Q];
                st = get_wtime_sec();
                H2ERI_exchange_workbuf_update_PQ_list(
                    h2eri, workbuf, Q, num_P0, P_list0, 
                    PQ_pair_idx, num_D, D_list
                );
                et = get_wtime_sec();
                timers[BUILD_K_AUX_TIMER_IDX] += et - st;

                st = get_wtime_sec();
                H2ERI_exchange_workbuf_alloc_dbl_buffer(h2eri, workbuf, N, Q, num_M, M_list);
                et = get_wtime_sec();
                timers[BUILD_K_AUX_TIMER_IDX] += et - st;

                st = get_wtime_sec();
                H2ERI_build_exchange_gather_vec_in(h2eri, workbuf, N, Q, den_mat);
                et = get_wtime_sec();
                timers[BUILD_K_AUX_TIMER_IDX] += et - st;

                H2ERI_build_exchange_H2_matmul_partial(h2eri, workbuf, tid);

                st = get_wtime_sec();
                H2ERI_build_exchange_scatter_vec_out(h2eri, workbuf, N, Q, num_M, M_list, K_mat);
                et = get_wtime_sec();
                timers[BUILD_K_AUX_TIMER_IDX] += et - st;
            }  // End of Q loop
        }  // End of N loop
    }  // End of "#pragma omp parallel"

    BLAS_SET_NUM_THREADS(n_thread);

    double build_K_timers[5] = {0, 0, 0, 0, 0};
    for (int i = 0; i < 5; i++)
    {
        for (int j = 0; j < n_thread; j++)
        {
            double timer_ij = thread_Kmat_workbuf[j]->timers[i];
            if (timer_ij > build_K_timers[i]) build_K_timers[i] = timer_ij;
        }
    }
    double *h2eri_timers = h2eri->timers;
    h2eri_timers[MV_VOP_TIMER_IDX] += build_K_timers[BUILD_K_AUX_TIMER_IDX];
    h2eri_timers[MV_FWD_TIMER_IDX] += build_K_timers[BUILD_K_MM_FWD_TIMER_IDX];
    h2eri_timers[MV_MID_TIMER_IDX] += build_K_timers[BUILD_K_MM_MID_TIMER_IDX];
    h2eri_timers[MV_BWD_TIMER_IDX] += build_K_timers[BUILD_K_MM_BWD_TIMER_IDX];
    h2eri_timers[MV_DEN_TIMER_IDX] += build_K_timers[BUILD_K_MM_DEN_TIMER_IDX];
    h2eri->n_matvec++;
}

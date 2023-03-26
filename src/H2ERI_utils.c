#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

#include "H2ERI_utils.h"
#include "H2ERI_aux_structs.h"

// Check if two boxes are admissible 
int H2E_check_box_admissible(const DTYPE *box0, const DTYPE *box1, const int pt_dim, const DTYPE alpha)
{
    for (int i = 0; i < pt_dim; i++)
    {
        // Radius of each box's i-th dimension
        DTYPE r0 = box0[pt_dim + i];
        DTYPE r1 = box1[pt_dim + i];
        // Center of each box's i-th dimension
        DTYPE c0 = box0[i] + 0.5 * r0;
        DTYPE c1 = box1[i] + 0.5 * r1;
        DTYPE min_r = MIN(r0, r1);
        DTYPE dist  = DABS(c0 - c1);
        if (dist >= alpha * min_r + 0.5 * (r0 + r1)) return 1;
    }
    return 0;
}

// Quick sorting an integer key-value pair array by key in ascending order
void H2E_qsort_int_kv_ascend(int *key, int *val, int l, int r)
{
    int i = l, j = r, tmp_key, tmp_val;
    int mid_key = key[(l + r) / 2];
    while (i <= j)
    {
        while (key[i] < mid_key) i++;
        while (key[j] > mid_key) j--;
        if (i <= j)
        {
            tmp_key = key[i]; key[i] = key[j]; key[j] = tmp_key;
            tmp_val = val[i]; val[i] = val[j]; val[j] = tmp_val;
            i++;  j--;
        }
    }
    if (i < r) H2E_qsort_int_kv_ascend(key, val, i, r);
    if (j > l) H2E_qsort_int_kv_ascend(key, val, l, j);
}

// Generate a random sparse matrix A for calculating y^T := A^T * x^T
void H2E_gen_rand_sparse_mat_trans(
    const int max_nnz_col, const int k, const int n, 
    H2E_dense_mat_p A_valbuf, H2E_int_vec_p A_idxbuf
)
{
    // Note: we calculate y^T := A^T * x^T. Since x/y is row-major, 
    // each of its row is a column of x^T/y^T. We can just use SpMV
    // to calculate y^T(:, i) := A^T * x^T(:, i). 

    int rand_nnz_col = (max_nnz_col <= k) ? max_nnz_col : k;
    int nnz = n * rand_nnz_col;
    H2E_dense_mat_resize(A_valbuf, 1, nnz);
    H2E_int_vec_set_capacity(A_idxbuf, (n + 1) + nnz + k);
    DTYPE *val = A_valbuf->data;
    int *row_ptr = A_idxbuf->data;
    int *col_idx = row_ptr + (n + 1);
    int *flag = col_idx + nnz; 
    memset(flag, 0, sizeof(int) * k);
    for (int i = 0; i < nnz; i++) 
        val[i] = (DTYPE) (2.0 * (rand() & 1) - 1.0);
    for (int i = 0; i <= n; i++) 
        row_ptr[i] = i * rand_nnz_col;
    for (int i = 0; i < n; i++)
    {
        int cnt = 0;
        int *row_i_cols = col_idx + i * rand_nnz_col;
        while (cnt < rand_nnz_col)
        {
            int col = rand() % k;
            if (flag[col] == 0) 
            {
                flag[col] = 1;
                row_i_cols[cnt] = col;
                cnt++;
            }
        }
        for (int j = 0; j < rand_nnz_col; j++)
            flag[row_i_cols[j]] = 0;
    }
    A_idxbuf->length = (n + 1) + nnz;
}

// Calculate y^T := A^T * x^T, where A is a sparse matrix, x and y are row-major matrices
void H2E_calc_sparse_mm_trans(
    const int m, const int n, const int k,
    H2E_dense_mat_p A_valbuf, H2E_int_vec_p A_idxbuf,
    DTYPE *x, const int ldx, DTYPE *y, const int ldy
)
{
    const DTYPE *val = A_valbuf->data;
    const int *row_ptr = A_idxbuf->data;
    const int *col_idx = row_ptr + (n + 1);  // A is k-by-n
    // Doing a naive OpenMP CSR SpMM here is good enough, using MKL SpBLAS is actually
    // slower, probably due to the cost of optimizing the storage of sparse matrix
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < m; i++)
    {
        DTYPE *x_i = x + i * ldx;
        DTYPE *y_i = y + i * ldy;
        
        for (int j = 0; j < n; j++)
        {
            DTYPE res = 0.0;
            #pragma omp simd
            for (int l = row_ptr[j]; l < row_ptr[j+1]; l++)
                res += val[l] * x_i[col_idx[l]];
            y_i[j] = res;
        }
    }
}

// Convert a integer COO matrix to a CSR matrix 
void H2E_int_COO_to_CSR(
    const int nrow, const int nnz, const int *row, const int *col, 
    const int *val, int *row_ptr, int *col_idx, int *val_
)
{
    // Get the number of non-zeros in each row
    memset(row_ptr, 0, sizeof(int) * (nrow + 1));
    for (int i = 0; i < nnz; i++) row_ptr[row[i] + 1]++;
    // Calculate the displacement of 1st non-zero in each row
    for (int i = 2; i <= nrow; i++) row_ptr[i] += row_ptr[i - 1];
    // Use row_ptr to bucket sort col[] and val[]
    for (int i = 0; i < nnz; i++)
    {
        int idx = row_ptr[row[i]];
        col_idx[idx] = col[i];
        val_[idx] = val[i];
        row_ptr[row[i]]++;
    }
    // Reset row_ptr
    for (int i = nrow; i >= 1; i--) row_ptr[i] = row_ptr[i - 1];
    row_ptr[0] = 0;
    // Sort the non-zeros in each row according to column indices
    #pragma omp parallel for
    for (int i = 0; i < nrow; i++)
        H2E_qsort_int_kv_ascend(col_idx, val_, row_ptr[i], row_ptr[i + 1] - 1);
}

// Get the value of integer CSR matrix element A(row, col)
int H2E_get_int_CSR_elem(
    const int *row_ptr, const int *col_idx, const int *val,
    const int row, const int col
)
{
    int res = 0;
    for (int i = row_ptr[row]; i < row_ptr[row + 1]; i++)
    {
        if (col_idx[i] == col) 
        {
            res = val[i];
            break;
        }
    }
    return res;
}

// Partition work units into multiple blocks s.t. each block has 
// approximately the same amount of work
void H2E_partition_workload(
    const int n_work,  const size_t *work_sizes, const size_t total_size, 
    const int n_block, H2E_int_vec_p blk_displs
)
{
    H2E_int_vec_set_capacity(blk_displs, n_block + 1);
    blk_displs->data[0] = 0;
    for (int i = 1; i < blk_displs->capacity; i++) 
        blk_displs->data[i] = n_work;
    size_t blk_size = total_size / n_block + 1;
    size_t curr_blk_size = 0;
    int idx = 1;
    for (int i = 0; i < n_work; i++)
    {
        curr_blk_size += work_sizes[i];
        if (curr_blk_size >= blk_size)
        {
            blk_displs->data[idx] = i + 1;
            curr_blk_size = 0;
            idx++;
        }
    }
    if (curr_blk_size > 0)
    {
        blk_displs->data[idx] = n_work;
        idx++;
    }
    blk_displs->length = idx;
}
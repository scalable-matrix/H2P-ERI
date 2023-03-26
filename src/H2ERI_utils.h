#ifndef __H2ERI_UTILS_H__
#define __H2ERI_UTILS_H__

#include "H2ERI_typedef.h"
#include "H2ERI_config.h"
#include "H2ERI_aux_structs.h"

#ifdef __cplusplus
extern "C" {
#endif

static inline void atomic_add_f64(volatile double *global_addr, double addend)
{
    uint64_t expected_value, new_value;
    do {
        double old_value = *global_addr;
        double tmp_value;
        #ifdef __INTEL_COMPILER
        expected_value = _castf64_u64(old_value);
        new_value      = _castf64_u64(old_value + addend);
        #else
        expected_value = *((uint64_t *) &old_value);
        tmp_value      = old_value + addend;
        new_value      = *((uint64_t *) &tmp_value);
        #endif
    } while (!__sync_bool_compare_and_swap((volatile uint64_t *) global_addr, expected_value, new_value));
}

static inline int H2ERI_gather_sum_int(const int *arr, const int nelem, const int *idx)
{
    int res = 0;
    for (int i = 0; i < nelem; i++) res += arr[idx[i]];
    return res;
}

// Check if two boxes are admissible 
// Input parameters:
//   box0, box1 : Box data, [0 : pt_dim-1] are coordinates of the box corner which is 
//                closest to the original point, [pt_dim : 2*pt_dim-1] are box length
//   pt_dim     : Dimension of point coordinate
//   alpha      : Admissible pair coefficient
// Output parameter:
//   <return>   : If two boxes are admissible 
int H2E_check_box_admissible(const DTYPE *box0, const DTYPE *box1, const int pt_dim, const DTYPE alpha);

// Quick sorting an integer key-value pair array by key in ascending order
void H2E_qsort_int_kv_ascend(int *key, int *val, int l, int r);

// Generate a random sparse matrix A for calculating y^T := A^T * x^T,
// where A is a random sparse matrix that has no more than max_nnz_col 
// random +1/-1 nonzeros in each column with random position, x and y 
// are row-major matrices. Each row of x/y is a column of x^T/y^T. We 
// can just use SpMV to calculate y^T(:, i) := A^T * x^T(:, i).
// Input parameters:
//   max_nnz_col : Maximum number of nonzeros in each column of A
//   k, n        : A is k-by-n sparse matrix
// Output parameters:
//   A_valbuf : Buffer for storing nonzeros of A^T
//   A_idxbuf : Buffer for storing CSR row_ptr and col_idx arrays of A^T. 
//              A_idxbuf->data[0 : n] stores row_ptr, A_idxbuf->data[n+1 : end]
//              stores col_idx.
void H2E_gen_rand_sparse_mat_trans(
    const int max_nnz_col, const int k, const int n, 
    H2E_dense_mat_p A_valbuf, H2E_int_vec_p A_idxbuf
);

// Calculate y^T := A^T * x^T, where A is a sparse matrix, 
// x and y are row-major matrices. Since x/y is row-major, 
// each of its row is a column of x^T/y^T. We can just use SpMV
// to calculate y^T(:, i) := A^T * x^T(:, i).
// Input parameters:
//   m, n, k  : x is m-by-k matrix, A is k-by-n sparse matrix
//   A_valbuf : Buffer for storing nonzeros of A^T
//   A_idxbuf : Buffer for storing CSR row_ptr and col_idx arrays of A^T. 
//              A_idxbuf->data[0 : n] stores row_ptr, A_idxbuf->data[n+1 : end]
//              stores col_idx.
//   x, ldx   : m-by-k row-major dense matrix, leading dimension ldx
//   ldy      : Leading dimension of y
// Output parameter:
//   y : m-by-n row-major dense matrix, leading dimension ldy
void H2E_calc_sparse_mm_trans(
    const int m, const int n, const int k,
    H2E_dense_mat_p A_valbuf, H2E_int_vec_p A_idxbuf,
    DTYPE *x, const int ldx, DTYPE *y, const int ldy
);

// Convert an integer COO matrix to a CSR matrix 
// Input parameters:
//   nrow          : Number of rows
//   nnz           : Number of nonzeros in the matrix
//   row, col, val : Size nnz, COO matrix
// Output parameters:
//   row_ptr, col_idx, val_ : Size nrow+1, nnz, nnz, CSR matrix
void H2E_int_COO_to_CSR(
    const int nrow, const int nnz, const int *row, const int *col, 
    const int *val, int *row_ptr, int *col_idx, int *val_
);

// Get the value of integer CSR matrix element A(row, col)
// Input parameters:
//   row_ptr, col_idx, val : CSR matrix array triple
//   row, col              : Target position
// Output parameter:
//   <return> : A(row, col) if exists, 0 if not
int H2E_get_int_CSR_elem(
    const int *row_ptr, const int *col_idx, const int *val,
    const int row, const int col
);

// Partition work units into multiple blocks s.t. each block has 
// approximately the same amount of work
// Input parameters:
//   n_work     : Number of work units
//   work_sizes : Size n_work, sizes of work units
//   total_size : Sum of work_sizes
//   n_block    : Number of blocks to be partitioned, the final result
//                may have fewer blocks
// Output parameter:
//   blk_displs : Indices of each block's first work unit. The actual 
//                number of work units == blk_displs->length-1 because 
//                blk_displs->data[0] == 0 and 
//                blk_displs->data[blk_displs->length-1] == total_size. 
void H2E_partition_workload(
    const int n_work,  const size_t *work_sizes, const size_t total_size, 
    const int n_block, H2E_int_vec_p blk_displs
);

#ifdef __cplusplus
}
#endif

#endif

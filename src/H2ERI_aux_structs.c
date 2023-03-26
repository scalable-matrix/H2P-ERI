#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

#include "H2ERI_aux_structs.h"
#include "utils.h"

// ========================== H2E_tree_node ========================== //

// Initialize an H2E_tree_node structure
void H2E_tree_node_init(H2E_tree_node_p *node_, const int dim)
{
    const int max_child = 1 << dim;
    H2E_tree_node_p node = (H2E_tree_node_p) malloc(sizeof(struct H2E_tree_node));
    ASSERT_PRINTF(node != NULL, "Failed to allocate H2E_tree_node structure\n");
    node->children = (void**) malloc(sizeof(H2E_tree_node_p) * max_child);
    node->enbox    = (DTYPE*) malloc(sizeof(DTYPE) * dim * 2);
    ASSERT_PRINTF(
        node->children != NULL && node->enbox != NULL,
        "Failed to allocate arrays in an H2E_tree_node structure\n"
    );
    for (int i = 0; i < max_child; i++) 
        node->children[i] = NULL;
    *node_ = node;
}

// Recursively destroy an H2E_tree_node node and its children nodes
void H2E_tree_node_destroy(H2E_tree_node_p *node_)
{
    H2E_tree_node_p node = *node_;
    if (node == NULL) return;
    for (int i = 0; i < node->n_child; i++)
    {
        H2E_tree_node_p child_i = (H2E_tree_node_p) node->children[i];
        if (child_i != NULL) H2E_tree_node_destroy(&child_i);
        free(child_i);
    }
    free(node->children);
    free(node->enbox);
    free(node);
    *node_ = NULL;
}

// ------------------------------------------------------------------- // 


// =========================== H2E_int_vec =========================== //

// Initialize an H2E_int_vec structure
void H2E_int_vec_init(H2E_int_vec_p *int_vec_, int capacity)
{
    if (capacity < 0) capacity = 128;
    H2E_int_vec_p int_vec = (H2E_int_vec_p) malloc(sizeof(struct H2E_int_vec));
    ASSERT_PRINTF(int_vec != NULL, "Failed to allocate H2E_int_vec structure\n");
    int_vec->data = (int*) malloc(sizeof(int) * capacity);
    ASSERT_PRINTF(int_vec->data != NULL, "Failed to allocate integer vector of size %d\n", capacity);
    int_vec->capacity = capacity;
    int_vec->length = 0;
    *int_vec_ = int_vec;
}

// Destroy an H2E_int_vec structure
void H2E_int_vec_destroy(H2E_int_vec_p *int_vec_)
{
    H2E_int_vec_p int_vec = *int_vec_;
    if (int_vec == NULL) return;
    free(int_vec->data);
    free(int_vec);
    *int_vec_ = NULL;
}

// Free the memory used by an H2E_int_vec structure and reset its capacity to 128
void H2E_int_vec_reset(H2E_int_vec_p int_vec)
{
    if (int_vec == NULL) return;
    free(int_vec->data);
    int_vec->capacity = 128;
    int_vec->length   = 0;
    int_vec->data = (int*) malloc(sizeof(int) * int_vec->capacity);
    ASSERT_PRINTF(int_vec->data != NULL, "Failed to allocate integer vector of size %d\n", int_vec->capacity);
}

// Concatenate values in an H2E_int_vec to another H2E_int_vec
void H2E_int_vec_concatenate(H2E_int_vec_p dst_vec, H2E_int_vec_p src_vec)
{
    int s_len = src_vec->length;
    int d_len = dst_vec->length;
    int new_length = s_len + d_len;
    if (new_length > dst_vec->capacity)
        H2E_int_vec_set_capacity(dst_vec, new_length);
    memcpy(dst_vec->data + d_len, src_vec->data, sizeof(int) * s_len);
    dst_vec->length = new_length;
}

// Gather elements in an H2E_int_vec to another H2E_int_vec
void H2E_int_vec_gather(H2E_int_vec_p src_vec, H2E_int_vec_p idx, H2E_int_vec_p dst_vec)
{
    H2E_int_vec_set_capacity(dst_vec, idx->length);
    for (int i = 0; i < idx->length; i++)
        dst_vec->data[i] = src_vec->data[idx->data[i]];
    dst_vec->length = idx->length;
}

// ------------------------------------------------------------------- // 


// ======================= H2E_partition_vars ======================== //

// Initialize an H2E_partition_vars structure
void H2E_partition_vars_init(H2E_partition_vars_p *part_vars_)
{
    H2E_partition_vars_p part_vars = (H2E_partition_vars_p) malloc(sizeof(struct H2E_partition_vars));
    H2E_int_vec_init(&part_vars->r_adm_pairs,   10240);
    H2E_int_vec_init(&part_vars->r_inadm_pairs, 10240);
    part_vars->curr_po_idx = 0;
    part_vars->max_level   = 0;
    part_vars->n_leaf_node = 0;
    *part_vars_ = part_vars;
}

// Destroy an H2E_partition_vars structure
void H2E_partition_vars_destroy(H2E_partition_vars_p *part_vars_)
{
    H2E_partition_vars_p part_vars = *part_vars_;
    if (part_vars == NULL) return;
    H2E_int_vec_destroy(&part_vars->r_adm_pairs);
    H2E_int_vec_destroy(&part_vars->r_inadm_pairs);
    free(part_vars->r_adm_pairs);
    free(part_vars->r_inadm_pairs);
    free(part_vars);
    *part_vars_ = NULL;
}

// ------------------------------------------------------------------- // 


// ========================== H2E_dense_mat ========================== //

// Initialize an H2E_dense_mat structure
void H2E_dense_mat_init(H2E_dense_mat_p *mat_, const int nrow, const int ncol)
{
    H2E_dense_mat_p mat = (H2E_dense_mat_p) malloc(sizeof(struct H2E_dense_mat));
    ASSERT_PRINTF(mat != NULL, "Failed to allocate H2E_dense_mat structure\n");
    
    mat->nrow = MAX(0, nrow);
    mat->ncol = MAX(0, ncol);
    mat->ld   = mat->ncol;
    mat->size = mat->nrow * mat->ncol;
    if (mat->size > 0)
    {
        mat->data = malloc_aligned(sizeof(DTYPE) * mat->size, 64);
        ASSERT_PRINTF(mat->data != NULL, "Failed to allocate %d * %d dense matrix\n", nrow, ncol);
    } else {
        mat->data = NULL;
    }
    
    *mat_ = mat;
}

// Destroy an H2E_dense_mat structure
void H2E_dense_mat_destroy(H2E_dense_mat_p *mat_)
{
    H2E_dense_mat_p mat = *mat_;
    if (mat == NULL) return;
    free_aligned(mat->data);
    free(mat);
    *mat_ = NULL;
}

// Reset an H2E_dense_mat structure to its default size (0-by-0) and release the memory
void H2E_dense_mat_reset(H2E_dense_mat_p mat)
{
    if (mat == NULL) return;
    free_aligned(mat->data);
    mat->data = NULL;
    mat->size = 0;
    mat->nrow = 0;
    mat->ncol = 0;
    mat->ld   = 0;
}

// Copy the data in an H2E_dense_mat structure to another H2E_dense_mat structure
void H2E_dense_mat_copy(H2E_dense_mat_p src_mat, H2E_dense_mat_p dst_mat)
{
    H2E_dense_mat_resize(dst_mat, src_mat->nrow, src_mat->ncol);
    copy_matrix_block(sizeof(DTYPE), src_mat->nrow, src_mat->ncol, src_mat->data, src_mat->ld, dst_mat->data, dst_mat->ld);
}

// Permute rows in an H2E_dense_mat structure
void H2E_dense_mat_permute_rows(H2E_dense_mat_p mat, const int *p)
{
    DTYPE *mat_dst = (DTYPE*) malloc_aligned(sizeof(DTYPE) * mat->nrow * mat->ncol, 64);
    ASSERT_PRINTF(mat_dst != NULL, "Failed to allocate buffer of size %d * %d\n", mat->nrow, mat->ncol);
    
    for (int irow = 0; irow < mat->nrow; irow++)
    {
        DTYPE *src_row = mat->data + p[irow] * mat->ld;
        DTYPE *dst_row = mat_dst + irow * mat->ncol;
        memcpy(dst_row, src_row, sizeof(DTYPE) * mat->ncol);
    }
    
    free_aligned(mat->data);
    mat->ld   = mat->ncol;
    mat->size = mat->nrow * mat->ncol;
    mat->data = mat_dst;
}

// Select rows in an H2E_dense_mat structure
void H2E_dense_mat_select_rows(H2E_dense_mat_p mat, H2E_int_vec_p row_idx)
{
    for (int irow = 0; irow < row_idx->length; irow++)
    {
        DTYPE *src = mat->data + row_idx->data[irow] * mat->ld;
        DTYPE *dst = mat->data + irow * mat->ld;
        if (src != dst) memcpy(dst, src, sizeof(DTYPE) * mat->ncol);
    }
    mat->nrow = row_idx->length;
}

// Select columns in an H2E_dense_mat structure
void H2E_dense_mat_select_columns(H2E_dense_mat_p mat, H2E_int_vec_p col_idx)
{
    for (int irow = 0; irow < mat->nrow; irow++)
    {
        DTYPE *mat_row = mat->data + irow * mat->ld;
        for (int icol = 0; icol < col_idx->length; icol++)
            mat_row[icol] = mat_row[col_idx->data[icol]];
    }
    mat->ncol = col_idx->length;
    for (int irow = 1; irow < mat->nrow; irow++)
    {
        DTYPE *src = mat->data + irow * mat->ld;
        DTYPE *dst = mat->data + irow * mat->ncol;
        memmove(dst, src, sizeof(DTYPE) * mat->ncol);
    }
    mat->ld = mat->ncol;
}

// Normalize columns in an H2E_dense_mat structure
void H2E_dense_mat_normalize_columns(H2E_dense_mat_p mat, H2E_dense_mat_p workbuf)
{
    int nrow = mat->nrow, ncol = mat->ncol;
    H2E_dense_mat_resize(workbuf, 1, ncol);
    DTYPE *inv_2norm = workbuf->data;
    
    #if 0
    #pragma omp simd
    for (int icol = 0; icol < ncol; icol++) 
        inv_2norm[icol] = mat->data[icol] * mat->data[icol];
    for (int irow = 1; irow < nrow; irow++)
    {
        DTYPE *mat_row = mat->data + irow * mat->ld;
        #pragma omp simd
        for (int icol = 0; icol < ncol; icol++) 
            inv_2norm[icol] += mat_row[icol] * mat_row[icol];
    }
    #pragma omp simd
    for (int icol = 0; icol < ncol; icol++) 
        inv_2norm[icol] = 1.0 / DSQRT(inv_2norm[icol]);
    #endif

    // Slower, but more accurate
    for (int icol = 0; icol < ncol; icol++)
        inv_2norm[icol] = 1.0 / CBLAS_NRM2(nrow, mat->data + icol, ncol);
    
    for (int irow = 0; irow < nrow; irow++)
    {
        DTYPE *mat_row = mat->data + irow * mat->ld;
        #pragma omp simd
        for (int icol = 0; icol < ncol; icol++) 
            mat_row[icol] *= inv_2norm[icol];
    }
}

// Perform GEMM C := alpha * op(A) * op(B) + beta * C
void H2E_dense_mat_gemm(
    const DTYPE alpha, const DTYPE beta, const int transA, const int transB, 
    H2E_dense_mat_p A, H2E_dense_mat_p B, H2E_dense_mat_p C
)
{
    int M, N, KA, KB;
    CBLAS_TRANSPOSE transA_, transB_;
    if (transA == 0)
    {
        transA_ = CblasNoTrans;
        M  = A->nrow;
        KA = A->ncol;
    } else {
        transA_ = CblasTrans;
        M  = A->ncol;
        KA = A->nrow;
    }
    if (transB == 0)
    {
        transB_ = CblasNoTrans;
        N  = B->ncol;
        KB = B->nrow;
    } else {
        transB_ = CblasTrans;
        N  = B->nrow;
        KB = B->ncol;
    }
    if (KA != KB)
    {
        ERROR_PRINTF("GEMM size mismatched: A[%d * %d], B[%d * %d]\n", M, KA, KB, N);
        return;
    }
    if (beta == 0.0 && (C->nrow < M || C->ncol < N)) H2E_dense_mat_resize(C, M, N);
    CBLAS_GEMM(
        CblasRowMajor, transA_, transB_, M, N, KA,
        alpha, A->data, A->ld, B->data, B->ld, 
        beta,  C->data, C->ld
    );
}

// Create a block diagonal matrix created by aligning the input matrices along the diagonal
void H2E_dense_mat_blkdiag(H2E_dense_mat_p *mats, H2E_int_vec_p idx, H2E_dense_mat_p new_mat)
{
    int nrow = 0, ncol = 0;
    for (int i = 0; i < idx->length; i++)
    {
        H2E_dense_mat_p mat_i = mats[idx->data[i]];
        nrow += mat_i->nrow;
        ncol += mat_i->ncol;
    }
    H2E_dense_mat_resize(new_mat, nrow, ncol);
    memset(new_mat->data, 0, sizeof(DTYPE) * nrow * ncol);
    nrow = 0; ncol = 0;
    for (int i = 0; i < idx->length; i++)
    {
        H2E_dense_mat_p mat_i = mats[idx->data[i]];
        int nrow_i = mat_i->nrow;
        int ncol_i = mat_i->ncol;
        DTYPE *dst = new_mat->data + nrow * new_mat->ld + ncol;
        copy_matrix_block(sizeof(DTYPE), nrow_i, ncol_i, mat_i->data, mat_i->ld, dst, new_mat->ld);
        nrow += nrow_i;
        ncol += ncol_i;
    }
}

// Vertically concatenates the input matrices
void H2E_dense_mat_vertcat(H2E_dense_mat_p *mats, H2E_int_vec_p idx, H2E_dense_mat_p new_mat)
{
    int nrow = 0, ncol = mats[idx->data[0]]->ncol;
    for (int i = 0; i < idx->length; i++)
    {
        H2E_dense_mat_p mat_i = mats[idx->data[i]];
        if (mat_i->ncol != ncol)
        {
            ERROR_PRINTF("%d-th matrix has %d columns, 1st matrix has %d columns\n", i+1, mat_i->ncol, ncol);
            return;
        }
        nrow += mat_i->nrow;
    }
    H2E_dense_mat_resize(new_mat, nrow, ncol);
    nrow = 0; ncol = 0;
    for (int i = 0; i < idx->length; i++)
    {
        H2E_dense_mat_p mat_i = mats[idx->data[i]];
        int nrow_i = mat_i->nrow;
        int ncol_i = mat_i->ncol;
        DTYPE *dst = new_mat->data + nrow * new_mat->ld;
        copy_matrix_block(sizeof(DTYPE), nrow_i, ncol_i, mat_i->data, mat_i->ld, dst, new_mat->ld);
        nrow += nrow_i;
    }
}

// Horizontally concatenates the input matrices
void H2E_dense_mat_horzcat(H2E_dense_mat_p *mats, H2E_int_vec_p idx, H2E_dense_mat_p new_mat)
{
    int nrow = mats[idx->data[0]]->nrow, ncol = 0;
    for (int i = 0; i < idx->length; i++)
    {
        H2E_dense_mat_p mat_i = mats[idx->data[i]];
        if (mat_i->nrow != nrow)
        {
            ERROR_PRINTF("%d-th matrix has %d rows, 1st matrix has %d rows\n", i+1, mat_i->nrow, nrow);
            return;
        }
        ncol += mat_i->ncol;
    }
    H2E_dense_mat_resize(new_mat, nrow, ncol);
    memset(new_mat->data, 0, sizeof(DTYPE) * nrow * ncol);
    nrow = 0; ncol = 0;
    for (int i = 0; i < idx->length; i++)
    {
        H2E_dense_mat_p mat_i = mats[idx->data[i]];
        int nrow_i = mat_i->nrow;
        int ncol_i = mat_i->ncol;
        DTYPE *dst = new_mat->data + ncol;
        copy_matrix_block(sizeof(DTYPE), nrow_i, ncol_i, mat_i->data, mat_i->ld, dst, new_mat->ld);
        ncol += mat_i->ncol;
    }
}

// Print an H2E_dense_mat structure, for debugging
void H2E_dense_mat_print(H2E_dense_mat_p mat)
{
    for (int irow = 0; irow < mat->nrow; irow++)
    {
        DTYPE *mat_row = mat->data + irow * mat->ld;
        for (int icol = 0; icol < mat->ncol; icol++) printf("% .4lf  ", mat_row[icol]);
        printf("\n");
    }
}

// ------------------------------------------------------------------- // 


// ========================== H2E_thread_buf ========================= //

void H2E_thread_buf_init(H2E_thread_buf_p *thread_buf_, const int krnl_mat_size)
{
    H2E_thread_buf_p thread_buf = (H2E_thread_buf_p) malloc(sizeof(struct H2E_thread_buf));
    ASSERT_PRINTF(thread_buf != NULL, "Failed to allocate H2E_thread_buf structure\n");
    H2E_int_vec_init(&thread_buf->idx0, 1024);
    H2E_int_vec_init(&thread_buf->idx1, 1024);
    H2E_dense_mat_init(&thread_buf->mat0, 1024, 1);
    H2E_dense_mat_init(&thread_buf->mat1, 1024, 1);
    H2E_dense_mat_init(&thread_buf->mat2, 1024, 1);
    thread_buf->y = (DTYPE*) malloc_aligned(sizeof(DTYPE) * krnl_mat_size, 64);
    ASSERT_PRINTF(thread_buf->y != NULL, "Failed to allocate y of size %d in H2E_thread_buf\n", krnl_mat_size);
    *thread_buf_ = thread_buf;
}

void H2E_thread_buf_destroy(H2E_thread_buf_p *thread_buf_)
{
    H2E_thread_buf_p thread_buf = *thread_buf_;
    if (thread_buf == NULL) return;
    H2E_int_vec_destroy(&thread_buf->idx0);
    H2E_int_vec_destroy(&thread_buf->idx1);
    H2E_dense_mat_destroy(&thread_buf->mat0);
    H2E_dense_mat_destroy(&thread_buf->mat1);
    H2E_dense_mat_destroy(&thread_buf->mat2);
    free_aligned(thread_buf->y);
    free(thread_buf);
    *thread_buf_ = NULL;
}

void H2E_thread_buf_reset(H2E_thread_buf_p thread_buf)
{
    if (thread_buf == NULL) return;
    H2E_int_vec_reset(thread_buf->idx0);
    H2E_int_vec_reset(thread_buf->idx1);
    H2E_dense_mat_reset(thread_buf->mat0);
    H2E_dense_mat_reset(thread_buf->mat1);
    H2E_dense_mat_reset(thread_buf->mat2);
}

// ------------------------------------------------------------------- // 
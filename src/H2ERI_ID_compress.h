#ifndef __H2ERI_ID_COMPRESS_H__
#define __H2ERI_ID_COMPRESS_H__

#include "H2ERI_config.h"
#include "H2ERI_aux_structs.h"

#ifdef __cplusplus
extern "C" {
#endif

// Interpolative Decomposition (ID) using partial QR over rows of a target 
// matrix. Partial pivoting QR may need to be upgraded to SRRQR later. 
// Given an m*n matrix A, an rank-k ID approximation of A is of form
//         A = U * A(J, :)
// where J is a row index subset of size k, and U is a m*k matrix (if 
// SRRQR is used, entries of U are bounded by a parameter 'f'). A(J,:) 
// and U are usually called the skeleton and projection matrix. 
// Input parameters:
//   A          : Target matrix, stored in row major
//   stop_type  : Partial QR stop criteria: QR_RANK, QR_REL_NRM, or QR_ABS_NRM
//   stop_param : Pointer to partial QR stop parameter
//   n_thread   : Number of threads used in this function
//   QR_buff    : Working buffer for partial pivoting QR. If kdim == 1, size A->nrow.
//                If kdim > 1, size (2*kdim+2)*A->ncol + (kdim+1)*A->nrow.
//   ID_buff    : Size 4 * A->nrow, working buffer for ID compression
//   kdim       : Dimension of tensor kernel's return (column block size)
// Output parameters:
//   U_ : Projection matrix, will be initialized in this function. If U_ == NULL,
//        the projection matrix will not be calculated.
//   J  : Row indices of the skeleton A
void H2E_ID_compress(
    H2E_dense_mat_p A, const int stop_type, void *stop_param, H2E_dense_mat_p *U_, 
    H2E_int_vec_p J, const int n_thread, DTYPE *QR_buff, int *ID_buff, const int kdim
);

#ifdef __cplusplus
}
#endif

#endif

#ifndef __H2ERI_MATMUL_H__
#define __H2ERI_MATMUL_H__

#include "H2ERI_typedef.h"

#ifdef __cplusplus
extern "C" {
#endif

// H2 representation for an ERI tensor multiplies a dense general column-major matrix
// Input parameters:
//   h2pack : H2Pack structure with H2 representation matrices
//   n_vec  : Number of column vectors in mat_x
//   mat_x  : Size >= n_vec * ldx, input dense matrix, the leading 
//            h2pack->krnl_mat_size-by-n_vec part of mat_x will be used
//   ldx    : Leading dimension of mat_x, >= h2pack->krnl_mat_size
//   ldy    : Leading dimension of mat_y, the same requirement as ldx
// Output parameter:
//   mat_y  : Size is the same as mat_x, output dense matrix, mat_y := A_{H2} * mat_x
void H2ERI_matmul(
    H2ERI_p h2eri, const int n_vec, 
    const double *mat_x, const int ldx, double *y, const int ldy
);

#ifdef __cplusplus
}
#endif

#endif

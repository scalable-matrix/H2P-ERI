#ifndef __H2ERI_BUILD_ACE_H__
#define __H2ERI_BUILD_ACE_H__

#include "H2ERI_typedef.h"

#ifdef __cplusplus
extern "C" {
#endif

// Build the Adaptive Compressed Exchange (ACE) matrix with the Cocc matrix 
// and H2 representation of the ERI tensor
// Input parameters:
//   h2eri    : H2ERI structure with H2 representation for ERI tensor
//   num_occ  : Number of occupied orbitals
//   Cocc_mat : Size h2eri->num_bf * num_occ, row-major matrix, Cocc_mat * Cocc_mat^T == density matrix
// Output parameters:
//   ACE_mat : ACE matrix, size h2eri->num_bf * h2eri->num_bf
void H2ERI_build_ACE(H2ERI_p h2eri, const int num_occ, const double *Cocc_mat, double *ACE_mat);

#ifdef __cplusplus
}
#endif

#endif

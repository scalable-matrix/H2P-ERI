#ifndef __H2ERI_BUILD_COULOMB_H__
#define __H2ERI_BUILD_COULOMB_H__

#include "H2ERI_typedef.h"

#ifdef __cplusplus
extern "C" {
#endif

// Build the Coulomb matrix with the density matrix and H2 representation of the ERI tensor
// Input parameters:
//   h2eri   : H2ERI structure with H2 representation for ERI tensor
//   den_mat : Symmetric density matrix, size h2eri->num_bf * h2eri->num_bf
// Output parameters:
//   J_mat : Symmetric Coulomb matrix, size h2eri->num_bf * h2eri->num_bf
void H2ERI_build_Coulomb(H2ERI_t h2eri, const double *den_mat, double *J_mat);

#ifdef __cplusplus
}
#endif

#endif

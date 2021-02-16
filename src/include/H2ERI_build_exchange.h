#ifndef __H2ERI_BUILD_EXCHANGE_H__
#define __H2ERI_BUILD_EXCHANGE_H__

#include "H2ERI_typedef.h"

#ifdef __cplusplus
extern "C" {
#endif

// Build the exchange matrix with the density matrix and H2 representation of the ERI tensor
// Input parameters:
//   h2eri   : H2ERI structure with H2 representation for ERI tensor
//   den_mat : Symmetric density matrix, size h2eri->num_bf * h2eri->num_bf
// Output parameters:
//   K_mat : Symmetric exchange matrix, size h2eri->num_bf * h2eri->num_bf
void H2ERI_build_exchange(H2ERI_p h2eri, const double *den_mat, double *K_mat);

// Initialize each thread's K mat build work buffer
void H2ERI_exchange_workbuf_init(H2ERI_p h2eri);

// Free each thread's K mat build work buffer
void H2ERI_exchange_workbuf_free(H2ERI_p h2eri);

#ifdef __cplusplus
}
#endif

#endif

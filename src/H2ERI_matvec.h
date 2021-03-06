#ifndef __H2ERI_MATVEC_H__
#define __H2ERI_MATVEC_H__

#include "H2ERI_typedef.h"

#ifdef __cplusplus
extern "C" {
#endif

// H2 representation for an ERI tensor multiplies a column vector
// Input parameters:
//   h2eri : H2ERI structure with H2 representation matrices
//   x     : Input dense vector
// Output parameter:
//   y : Output dense vector
void H2ERI_matvec(H2ERI_p h2eri, const double *x, double *y);

#ifdef __cplusplus
}
#endif

#endif

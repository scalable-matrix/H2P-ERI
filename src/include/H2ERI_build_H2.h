#ifndef __H2ERI_BUILD_H2_H__
#define __H2ERI_BUILD_H2_H__

#include "H2ERI_typedef.h"
#include "H2Pack_aux_structs.h"

#ifdef __cplusplus
extern "C" {
#endif

// Build H2 representation for ERI tensor
// Input parameter:
//   h2eri  : H2ERI structure with point partitioning info
//   BD_JIT : 0 or 1, if B and D matrices are computed just-in-time in matvec
// Output parameter:
//   h2eri : H2ERI structure with H2 representation matrices
void H2ERI_build_H2(H2ERI_t h2eri, const int BD_JIT);

#ifdef __cplusplus
}
#endif

#endif

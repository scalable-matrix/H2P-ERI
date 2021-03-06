#ifndef __H2ERI_SHELL_OPERATIONS_H__
#define __H2ERI_SHELL_OPERATIONS_H__

// Shell operations used in H2P-ERI

#include "CMS.h"

#ifdef __cplusplus
extern "C" {
#endif

// Process input shells for H2 partitioning
// Input parameters:
//   h2eri->nshell    : Number of original shells 
//   h2eri->shells    : Array, size nshell, original shells
//   h2eri->scr_tol   : Schwarz screening tolerance, typically 1e-10
//   h2eri->ext_tol   : Tolerance of shell pair extent
// Output parameters:
//   h2eri->shells    : Shells with rotated coordinates
//   h2eri->num_sp    : Number of screened shell pairs
//   h2eri->sp        : Array, size num_sp * 2, screened shell pairs
//   h2eri->sp_center : Array, size 3 * num_sp, each column is the center 
//                      coordinate of a screened shell pair
//   h2eri->sp_extent : Array, size num_sp, extents of each screened shell pair
void H2ERI_process_shells(H2ERI_p h2eri);

#ifdef __cplusplus
}
#endif

#endif

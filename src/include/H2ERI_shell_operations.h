#ifndef __H2ERI_SHELL_OPERATIONS_H__
#define __H2ERI_SHELL_OPERATIONS_H__

// Shell operations used in H2P-ERI

#include "CMS.h"

#ifdef __cplusplus
extern "C" {
#endif

// Rotate shell coordinates for better hierarchical partitioning
// Input parameters:
//   h2eri->nshell : Number of shells 
//   h2eri->shells : Shells to be rotated
// Output parameters:
//   h2eri->shells : Shells with rotated coordinates
void H2ERI_rotate_shells(H2ERI_t h2eri);

// Fully uncontract all shells into new shells that each shell has
// only 1 primitive function and screen uncontracted shell pairs
// Input parameters:
//   h2eri->nshell  : Number of original shells 
//   h2eri->shells  : Original shells
//   h2eri->scr_tol : Schwarz screening tolerance, typically 1e-10
// Output parameters:
//   h2eri->num_unc_sp : Number of uncontracted shell pairs that survives screening
//   h2eri->unc_sp     : Array, size (*num_unc_sp_) * 2, uncontracted screened shell pairs
//   h2eri->unc_center : Array, size 3 * (*num_unc_sp_), each column is the center 
//                       coordinate of a new uncontracted shell pair
void H2ERI_uncontract_shell_pairs(H2ERI_t h2eri);

// Calculate the extent (numerical support radius) of uncontract shell pairs
// Input parameters:
//   h2eri->unc_num_sp : Number of shell pairs
//   h2eri->unc_sp     : Array, size num_sp * 2, each row is a shell pair
//   h2eri->ext_tol    : Tolerance of shell pair extent
// Output parameters:
//   h2eri->unc_sp_extent : Array, size h2eri->unc_num_sp, extent of each shell pair
void H2ERI_calc_unc_sp_extents(H2ERI_t h2eri);

#ifdef __cplusplus
}
#endif

#endif

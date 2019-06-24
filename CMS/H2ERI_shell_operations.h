#ifndef __H2ERI_SHELL_OPERATIONS_H__
#define __H2ERI_SHELL_OPERATIONS_H__

// Shell operations used in H2P-ERI

#include "CMS.h"

// Rotate shell coordinates for better hierarchical partitioning
// Input parameters:
//   nshell : Number of shells 
//   shells : Shells to be rotated
// Output parameters:
//   shells : Shells with rotated coordinates
void H2ERI_rotate_shells(const int nshell, shell_t *shells);

// Fully uncontract all shells into new shells that each shell has
// only 1 primitive function and screen uncontracted shell pairs
// Input parameters:
//   nshell  : Number of original shells 
//   shells  : Original shells
//   scr_tol : Schwarz screening tolerance, typically 1e-10
// Output parameters:
//   *num_unc_sp_  : Number of uncontracted shell pairs that survives screening
//   **unc_sp_     : Array, size (*num_unc_sp_) * 2, uncontracted screened shell pairs
//   **unc_center_ : Array, size 3 * (*num_unc_sp_), each column is the center 
//                   coordinate of a new uncontracted shell pair
void H2ERI_uncontract_shell_pairs(
    const int nshell, const shell_t *shells, const double scr_tol, 
    int *num_unc_sp_, shell_t **unc_sp_, double **unc_center_
);

// Calculate the extent (numerical support radius) of each shell pair
// Input parameters:
//   num_sp  : Number of shell pairs
//   sp      : Array, size num_sp * 2, each row is a shell pair
//   ext_tol : Tolerance of shell pair extent
// Output parameters:
//   sp_extent : Array, size num_sp, extent of each shell pair
void H2ERI_calc_shell_pair_extents(
    const int num_sp, const shell_t *sp, 
    const double ext_tol, double *sp_extent
);

#endif

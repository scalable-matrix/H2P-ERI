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

#endif

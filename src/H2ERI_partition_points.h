#ifndef __H2ERI_PARTITION_POINTS_H__
#define __H2ERI_PARTITION_POINTS_H__

#include "H2ERI_typedef.h"

#ifdef __cplusplus
extern "C" {
#endif

// Hierarchical point partitioning for H2 / HSS construction
// Input parameters:
//   h2eri           : H2ERI structure
//   n_point         : Number of points for the kernel matrix
//   coord           : Matrix, size h2eri->pt_dim * n_point, each column is a point coordinate
//   max_leaf_points : Maximum point in a leaf node's box. If <= 0, will use 200 for
//                     2D points and 400 for other dimensions
//   max_leaf_size   : Maximum size of a leaf node's box. If == 0, max_leaf_points
//                     will be the only restriction.
// Output parameter:
//   h2eri : H2ERI structure with point partitioning info
void H2E_partition_points(
    H2ERI_p h2eri, const int n_point, const DTYPE *coord, 
    int max_leaf_points, DTYPE max_leaf_size
);

#ifdef __cplusplus
}
#endif

#endif

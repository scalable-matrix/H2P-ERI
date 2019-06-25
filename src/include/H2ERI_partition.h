#ifndef __H2ERI_PARTITION_H__
#define __H2ERI_PARTITION_H__

#include "H2ERI_typedef.h"

#ifdef __cplusplus
extern "C" {
#endif

// Partition uncontracted shell pair centers (as points) for H2 tree
// Input parameters:
//   h2eri->num_unc_sp    : Number of fully uncontracted shell pairs (FUSP)
//   h2eri->unc_sp_center : Array, size 3 * num_unc_sp, center of FUSP
//   max_leaf_points      : Maximum number of point in a leaf node's box. If <= 0, 
//                          will use 300.
//   max_leaf_size        : Maximum size of a leaf node's box. 
// Output parameter:
//   h2eri->h2pack : H2Pack structure with point partitioning info
void H2ERI_partition_unc_sp_centers(H2ERI_t h2eri, int max_leaf_points, double max_leaf_size);

#ifdef __cplusplus
}
#endif

#endif

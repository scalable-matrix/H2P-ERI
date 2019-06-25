#ifndef __H2ERI_PARTITION_H__
#define __H2ERI_PARTITION_H__

#include "H2ERI_typedef.h"

#ifdef __cplusplus
extern "C" {
#endif

// Partition uncontracted shell pair centers (as points) for H2 tree
// Input parameters:
//   h2eri->num_unc_sp    : Number of fully uncontracted shell pairs (FUSP)
//   h2eri->unc_sp_center : Array, size 3 * num_unc_sp, centers of FUSP
//   h2eri->unc_sp_extent : Array, size num_unc_sp, extents of FUSP
//   max_leaf_points      : Maximum number of point in a leaf node's box. If <= 0, 
//                          will use 300.
//   max_leaf_size        : Maximum size of a leaf node's box. 
// Output parameter:
//   h2eri->h2pack        : H2Pack structure with point partitioning info
//   h2eri->unc_sp_center : Array, size 3 * num_unc_sp, sorted centers of FUSP
//   h2eri->unc_sp_extent : Array, size num_unc_sp, sorted extents of FUSP
void H2ERI_partition_unc_sp_centers(H2ERI_t h2eri, int max_leaf_points, double max_leaf_size);

// Calculate the max extent of shell pairs in each H2 box
// Input parameters:
//   h2eri->h2pack        : H2 tree partitioning info
//   h2eri->num_unc_sp    : Number of fully uncontracted shell pairs (FUSP)
//   h2eri->unc_sp_center : Array, size 3 * num_unc_sp, centers of FUSP, sorted
//   h2eri->unc_sp_extent : Array, size num_unc_sp, extents of FUSP, sorted
// Output parameter:
//   h2eri->box_extent : Array, size h2pack->n_node, extent of each H2 node box
void H2ERI_calc_box_extent(H2ERI_t h2eri);

#ifdef __cplusplus
}
#endif

#endif

#ifndef __H2ERI_PARTITION_H__
#define __H2ERI_PARTITION_H__

#include "H2ERI_typedef.h"

#ifdef __cplusplus
extern "C" {
#endif

// H2 partition of uncontracted shell pair centers
// Input parameters:
//   h2eri->nshell        : Number of original shells 
//   h2eri->shells        : Array, size nshell, original shells
//   h2eri->num_unc_sp    : Number of fully uncontracted shell pairs (FUSP)
//   h2eri->unc_sp_center : Array, size 3 * num_unc_sp, centers of FUSP
//   h2eri->unc_sp_extent : Array, size num_unc_sp, extents of FUSP
// Output parameters:
//   h2eri->h2pack          : H2Pack structure with point partitioning info
//   h2eri->unc_sp_center   : Array, size 3 * num_unc_sp, sorted centers of FUSP
//   h2eri->unc_sp_extent   : Array, size num_unc_sp, sorted extents of FUSP
//   h2eri->unc_sp_nbfp     : Array, size num_unc_sp, number of basis function pairs of each FUSP
//   h2eri->unc_sp_bfp_sidx : Array, size num_unc_sp+1, indices of each FUSP first basis function pair
//   h2eri->box_extent      : Array, size h2pack->n_node, extent of each H2 node box
//   h2pack->mat_cluster    : Array, size h2pack->n_node * 2, matvec cluster for H2 nodes
//   h2eri->simint_buffs    : Array, size h2pack->n_thread, thread local Simint ERI buffers
//   h2eri->unc_sp_shells   : Array, size 2 * num_unc_sp, each column is a FUSP
//   h2eri->unc_sp          : Array, size num_unc_sp, all FUSP
void H2ERI_partition(H2ERI_t h2eri);

#ifdef __cplusplus
}
#endif

#endif

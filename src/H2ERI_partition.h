#ifndef __H2ERI_PARTITION_H__
#define __H2ERI_PARTITION_H__

#include "H2ERI_typedef.h"

#ifdef __cplusplus
extern "C" {
#endif

// H2 partition of screened shell pair centers
// Input parameters:
//   h2eri->nshell    : Number of original shells 
//   h2eri->shells    : Array, size nshell, original shells
//   h2eri->num_sp    : Number of screened shell pairs (SSP)
//   h2eri->sp_center : Array, size 3 * num_sp, centers of SSP
//   h2eri->sp_extent : Array, size num_sp, extents of SSP
// Output parameters:
//   h2eri->h2pack        : H2Pack structure with point partitioning info
//   h2eri->sp_center     : Array, size 3 * num_sp, sorted centers of SSP
//   h2eri->sp_extent     : Array, size num_sp, sorted extents of SSP
//   h2eri->shell_bf_sidx : Array, size nshell, indices of each shell's first basis function
//   h2eri->sp_nbfp       : Array, size num_sp, number of basis function pairs of each SSP
//   h2eri->sp_bfp_sidx   : Array, size num_sp+1, indices of each SSP first basis function pair
//   h2eri->box_extent    : Array, size h2pack->n_node, extent of each H2 node box
//   h2pack->mat_cluster  : Array, size h2pack->n_node * 2, matvec cluster for H2 nodes
//   h2eri->simint_buffs  : Array, size h2pack->n_thread, thread local Simint ERI buffers
//   h2eri->sp_shells     : Array, size 2 * num_sp, each column is a SSP
//   h2eri->sp            : Array, size num_sp, all screened shell pairs
void H2ERI_partition(H2ERI_p h2eri);

#ifdef __cplusplus
}
#endif

#endif

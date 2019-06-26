#ifndef __H2ERI_BUILD_H__
#define __H2ERI_BUILD_H__

#include "H2ERI_typedef.h"
#include "H2Pack_aux_structs.h"

#ifdef __cplusplus
extern "C" {
#endif

// Build H2 representation for ERI tensor
// Input parameter:
//   h2eri : H2ERI structure with point partitioning info
// Output parameter:
//   h2eri : H2ERI structure with H2 representation matrices
void H2ERI_build(H2ERI_t h2eri);

void H2ERI_generate_proxy_point_layers(
    const double r1, const double r2, const int npts_layer,
    const int nlayer, H2P_dense_mat_t pp
);

void H2ERI_calc_ovlp_ff_idx(H2ERI_t h2eri);

void H2ERI_extract_shell_pair_idx(
    const int num_target, const int *target_rows, 
    const int num_sp, const int *am1, const int *am2, 
    int *workbuf, int *num_pair_idx, int *pair_idx, int *row_idx
);

#ifdef __cplusplus
}
#endif

#endif

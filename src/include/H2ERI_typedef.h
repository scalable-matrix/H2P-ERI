#ifndef __H2ERI_TYPEDEF_H__
#define __H2ERI_TYPEDEF_H__

// Shell operations used in H2P-ERI

#include "CMS.h"
#include "H2Pack_config.h"
#include "H2Pack_typedef.h"
#include "H2Pack_aux_structs.h"

#ifdef __cplusplus
extern "C" {
#endif

struct H2ERI
{
    int     max_am;              // Maximum angular momentum in the system
    int     nshell;              // Number of contracted shells (from input file)
    int     num_unc_sp;          // Number of fully uncontracted shell pairs (FUSP)
    int     *shell_bf_sidx;      // Array, size nshell+1, indices of each shell's first basis function
    int     *unc_sp_bf_sidx;     // Array, size num_unc_sp+1, indices of each FUSP first basis function 
    double  scr_tol;             // Tolerance of Schwarz screening
    double  ext_tol;             // Tolerance of shell pair extent
    double *unc_sp_center;       // Array, size 3 * num_unc_sp, centers of FUSP
    double *unc_sp_extent;       // Array, size num_unc_sp, extents of FUSP
    double *box_extent;          // Array, size h2pack->n_node, extents of each H2 node box
    shell_t *shells;             // Array, size nshell, contracted shells
    shell_t *unc_sp;             // Array, size num_unc_sp * 2, FUSP
    H2P_int_vec_t *J_pair;       // Array, size h2pack->n_node, skeleton shell pair indices of each node
    H2P_int_vec_t *J_row;        // Array, size h2pack->n_node, skeleton row indices in each node's shell pairs
    H2P_int_vec_t *ovlp_ff_idx;  // Array, size h2pack->n_node, i-th vector contains the far field 
                                 // points whose extents are overlapped with the near field of i-th node
    simint_buff_t *simint_buffs; // Array, size h2pack->n_thread, simint_buff structures for each thread
    H2Pack_t h2pack;             // H2Pack data structure
};

typedef struct H2ERI* H2ERI_t;

#define ALPHA_SUP 1.0

// Initialize a H2ERI structure
// Input parameters:
//   scr_tol : Tolerance of Schwarz screening (typically 1e-10)
//   ext_tol : Tolerance of shell pair extent (typically 1e-10)
//   QR_tol  : Tolerance of column-pivoting QR (controls the overall accuracy)
// Output parameter:
//   h2eri_ : Initialized H2ERI structure
void H2ERI_init(H2ERI_t *h2eri_, const double scr_tol, const double ext_tol, const double QR_tol);

// Destroy a H2ERI structure
// Input parameter:
//   h2eri : H2ERI structure to be destroyed
void H2ERI_destroy(H2ERI_t h2eri);

#ifdef __cplusplus
}
#endif

#endif

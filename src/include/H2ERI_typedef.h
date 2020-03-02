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
    int    natom;                       // Number of atoms (from input file)
    int    nshell;                      // Number of contracted shells (from input file)
    int    max_am;                      // Maximum angular momentum in the system
    int    num_bf;                      // Number of basis functions in the system, == shell_bf_sidx[nshell]
    int    num_sp;                      // Number of screened shell pairs (SSP)
    int    num_sp_bfp;                  // Number of SSP basis function pairs, == sp_bfp_sidx[num_sp]
    int    pp_npts_layer;               // Number of proxy points on each layer
    int    pp_nlayer_ext;               // Number of proxy point layers on each extent
    int    *shell_bf_sidx;              // Array, size nshell, indices of each shell's first basis function
    int    *sp_nbfp;                    // Array, size num_sp, number of basis function pairs of each SSP
    int    *sp_bfp_sidx;                // Array, size num_sp+1, indices of each SSP's first basis function pair
    int    *sp_shell_idx;               // Array, size 2 * num_sp, each row is the contracted shell indices of a SSP
    int    *index_seq;                  // Array, size num_sp, [0, num_sp-1]
    double scr_tol;                     // Tolerance of Schwarz screening
    double ext_tol;                     // Tolerance of shell pair extent
    double *sp_center;                  // Array, size 3 * num_sp, each column is a SSP's center
    double *sp_extent;                  // Array, size num_sp, extents of SSP
    double *box_extent;                 // Array, size h2pack->n_node, extents of each H2 node box
    double *unc_denmat_x;               // Array, size num_sp_bfp, uncontracted density matrix as a vector
    double *H2_matvec_y;                // Array, size num_sp_bfp, H2 matvec result 
    shell_t          *shells;           // Array, size nshell, contracted shells
    shell_t          *sp_shells;        // Array, size 2 * num_sp, each column is a SSP
    multi_sp_t       *sp;               // Array, size num_sp, SSP
    H2P_int_vec_t    *J_pair;           // Array, size h2pack->n_node, skeleton shell pair indices of each node
    H2P_int_vec_t    *J_row;            // Array, size h2pack->n_node, skeleton row indices in each node's shell pairs
    H2P_int_vec_t    *ovlp_ff_idx;      // Array, size h2pack->n_node, i-th vector contains the far field 
                                        // points whose extents are overlapped with the near field of i-th node
    simint_buff_t    *simint_buffs;     // Array, size h2pack->n_thread, simint_buff structures for each thread
    eri_batch_buff_t *eri_batch_buffs;  // Array, size h2pack->n_thread, eri_batch_buff structures for each thread
    H2P_dense_mat_t  *c_B_blks;         // Array, size h2pack->n_B, compressed B blocks
    H2P_dense_mat_t  *c_D_blks;         // Array, size h2pack->n_D, compressed D blocks
    H2Pack_t         h2pack;            // H2Pack data structure
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

// Print H2ERI statistic information
// Input parameter:
//   h2eri : H2ERI structure to be destroyed
void H2ERI_print_statistic(H2ERI_t h2eri);

#ifdef __cplusplus
}
#endif

#endif

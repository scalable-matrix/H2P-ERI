#ifndef __CMS_H__
#define __CMS_H__

#include "simint/simint.h"

typedef struct simint_shell           shell_t;
typedef struct simint_multi_shellpair multi_sp_t;

struct simint_buff
{
    size_t     work_msize;  // Memory size of work_mem
    size_t     ERI_msize;   // Memory size of ERI_mem
    double     *work_mem;   // Simint work buff
    double     *ERI_mem;    // ERI results
    shell_t    NAI_shell1;  // Shell 1 for NAI calculation
    shell_t    NAI_shell2;  // Shell 2 for NAI calculation
    multi_sp_t bra_pair;    // Bra-side shell pairs for ERI calculation
    multi_sp_t ket_pair;    // Ket-side shell pairs for ERI calculation
};
typedef struct simint_buff* simint_buff_t;

#define NCART(am)  ((am>=0)?((((am)+2)*((am)+1))>>1):0)
#define MAX(a, b)  ((a) > (b) ? (a) : (b))
#define NPAIR_SIMD 16

// Read all shell information in a .mol file and normalize all these shells
// Input parameter:
//   mol_fname : .mol file path
// Output parameters:
//   *nshell_ : Total number of shells in the .mol file
//   *shells_ : Array, all shells in the .mol file, stored in Simint shell structure
void CMS_read_mol_file(const char *mol_fname, int *nshell_, shell_t **shells_);

// Destroy all Simint shells
// Input parameters:
//   nshell : Total number of shells
//   shells : Array, all shells stored in Simint shell structure
void CMS_destroy_shells(const int nshell, shell_t *shells);

// Print all shell information, for debugging
// Input parameters:
//   nshell : Total number of shells
//   shells : Array, all shells stored in Simint shell structure
void CMS_print_shells(const int nshell, shell_t *shells);

// Get the Schwarz screening value from a given set of shells
// Input parameters:
//   nshell : Total number of shells
//   shells : Array, all shells stored in Simint shell structure
// Output parameter:
//   scr_vals : Schwarz screening value of each shell pair
//   <return> : Maximum Schwarz screening value
double CMS_get_Schwarz_scrval(const int nshell, shell_t *shells, double *scr_vals);

// Initialize a Simint buffer structure
// Input parameter:
//   max_am : Maximum angular momentum in the system
// Output parameter:
//   *buff_ : Initialized Simint buffer stricture
void CMS_init_Simint_buff(const int max_am, simint_buff_t *buff_);

// Destroy a Simint buffer structure
// Input parameter:
//   buff : Simint buffer stricture to be destroyed 
void CMS_destroy_Simint_buff(simint_buff_t buff);

// Sum the number of basis function pairs of a FUSP list
// Input parameters:
//   unc_sp_shells : Array, size 2 * num_sp, each column is a FUSP
//   num_unc_sp    : Number of FUSP
//   num_sp        : Number of shell pairs (N_i M_i|
//   sp_idx        : Array, size num_sp, FUSP indices
// Output parameter:
//   <return> : Total number of basis function pairs of a FUSP list
int H2ERI_sum_sp_bfp(
    const shell_t *unc_sp_shells, const int num_unc_sp,
    const int num_sp, const int *sp_idx
);

// Calculate shell quartet pairs (N_i M_i|Q_j P_j) and unfold all ERI 
// results to form a matrix.
// The ERI result tensor of (N_i M_i|Q_j P_j) will be unfold as a 
// NCART(N_i)*NCART(M_i) rows NCART(Q_j)*NCART(P_j) columns sub-block
// and placed in the i-th row block and j-th column block of the final
// result matrix. For the moment, we use (N_i M_i|Q_j P_j) instead of
// (M_i N_i|P_j Q_j) just to follow the output of calculate_eri_pair.c
// file in simint-matlab. TODO: check if we can use (M_i N_i|P_j Q_j).
// Input parameters:
//   unc_sp     : Array, size num_unc_sp, FUSP
//   n_bra_pair : Number of bra-side shell pairs (N_i M_i|
//   n_ket_pair : Number of ket-side shell pairs |Q_j P_j)
//   bra_idx    : Array, size n_bra_pair, indices of (N_i M_i| pairs
//   ket_idx    : Array, size n_ket_pair, indices of |Q_j P_j) pairs
//   buff       : Initialized Simint buffer stricture
//   ldm        : Leading dimension of output matrix, should >= 
//                H2ERI_sum_sp_bfp(unc_sp_shells, num_unc_sp, n_ket_pair, ket_idx)
// Output parameter:
//   mat : Matrix with unfolded shell quartets ERI results, size >= ldm *
//         H2ERI_sum_sp_bfp(unc_sp_shells, num_unc_sp, n_bra_pair, bra_idx)
void H2ERI_calc_ERI_pairs_to_mat(
    const multi_sp_t *unc_sp, const int n_bra_pair, const int n_ket_pair,
    const int *bra_idx, const int *ket_idx, simint_buff_t buff, 
    double *mat, const int ldm
);

// Calculate NAI pairs (N_i M_i|[x_j, y_j, z_j]) and unfold all NAI 
// results to form a matrix.
// Each NAI result (N_i M_i|[x_j, y_j, z_j]) will be unfold as a 1-by-
// (NCART(N_i)*NCART(M_i)) row vector and place at the j-th row i-th 
// column block. For the moment, we use (N_i M_i|[x_j, y_j, z_j]) instead
// of (M_i N_i|[x_j, y_j, z_j]) just to follow the output of the file
// calculate_nai_block.m in simint-matlab. Note that the output of this 
// function is the transpose of calculate_nai_block.m's output. 
// TODO: check if we can use (M_i N_i|[x_j, y_j, z_j]) later.
// Input parameters:
//   unc_sp_shells : Array, size 2 * num_sp, each column is a FUSP
//   num_unc_sp    : Number of FUSP
//   num_sp        : Number of shell pairs (N_i M_i|
//   sp_idx        : Array, size num_sp, FUSP indices
//   n_point       : Number of point charge
//   x, y, z       : Array, size of n_point, point charge coordinates
//   ldm           : Leading dimension of output matrix, should >= 
//                   H2ERI_sum_sp_bfp(unc_sp_shells, num_unc_sp, num_sp, sp_idx)
// Output parameter:
//   mat : Matrix with unfolded NAI pairs results, size >= ldm * n_point
void H2ERI_calc_NAI_pairs_to_mat(
    const shell_t *unc_sp_shells, const int num_unc_sp,
    const int num_sp, const int *sp_idx, const int n_point,
    double *x, double *y, double *z, double *mat, const int ldm
);

#endif

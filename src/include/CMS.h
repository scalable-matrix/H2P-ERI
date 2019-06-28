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

// Sum the number of basis functions of a shell list
// Input parameters:
//   shells    : Array, size unrestricted, Simint shell structures
//   nshell    : Number of shells to sum the number of basis functions
//   shell_idx : Indices of shells to sum the number of basis functions
// Output parameters:
//   <return> : Total number of basis functions of a shell list
int CMS_sum_shell_basis_functions(const shell_t *shells, const int nshell, const int *shell_idx);

// Sum the number of basis function pairs of a shell pair list
// Input parameters:
//   shells     : Array, size unrestricted, Simint shell structures
//   num_sp     : Number of shell pairs to sum the number of basis function pairs
//   shell_idx0 : First shell indices of the shell pair list
//   shell_idx1 : Second shell indices of the shell pair list
// Output parameters:
//   <return> : Total number of basis functions pairsof a shell pair list
int CMS_sum_shell_pair_bas_func_pairs(
    const shell_t *shells, const int num_sp, 
    const int *shell_idx0, const int *shell_idx1
);

// Calculate shell quartet pairs (M_i N_i|P_j Q_j)
// Input parameters:
//   shells     : Array, Simint shell structures
//   n_bra_pair : Number of bra-side shell pairs (M_i N_i|
//   n_ket_pair : Number of ket-side shell pairs |P_j Q_j)
//   {M,N}_list : Array, size n_bra_pair, M_i and N_i values
//   {P,Q}_list : Array, size n_ket_pair, P_j and Q_j values
//   buff       : Initialized Simint buffer stricture
// Output parameters:
//   buff->ERI_mem : ERI results, storing order: (M0 N0|P0 Q0), (M0 N0|P1 Q1), ...
void CMS_calc_ERI_pairs(
    const shell_t *shells, const int n_bra_pair, const int n_ket_pair,
    int *M_list, int *N_list, int *P_list, int *Q_list, simint_buff_t buff
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
//   shells     : Array, Simint shell structures
//   n_bra_pair : Number of bra-side shell pairs (M_i N_i|
//   n_ket_pair : Number of ket-side shell pairs |P_j Q_j)
//   {M,N}_list : Array, size n_bra_pair, M_i and N_i values
//   {P,Q}_list : Array, size n_ket_pair, P_j and Q_j values
//   buff       : Initialized Simint buffer stricture
//   ldm        : Leading dimension of output matrix, should >= 
//                CMS_sum_shell_pair_bas_func_pairs(shells, n_ket_pair, P_list, Q_list)
// Output parameter:
//   mat : Matrix with unfolded shell quartets ERI results, size >= ldm *
//         CMS_sum_shell_pair_bas_func_pairs(shells, n_bra_pair, M_list, N_list) 
void CMS_calc_ERI_pairs_to_mat(
    const shell_t *shells, const int n_bra_pair, const int n_ket_pair,
    int *M_list, int *N_list, int *P_list, int *Q_list,
    simint_buff_t buff, double *mat, const int ldm
);

#endif

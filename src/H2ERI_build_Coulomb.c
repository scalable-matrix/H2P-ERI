#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>

#include <omp.h>

#include "H2ERI_typedef.h"
#include "H2ERI_build_Coulomb.h"
#include "H2ERI_matvec.h"

// "Uncontract" the density matrix according to SSP and unroll 
// the result to a column for H2 matvec.
// Input parameters:
//   den_mat              : Symmetric density matrix, size h2eri->num_bf^2
//   h2eri->num_bf        : Number of basis functions in the system
//   h2eri->num_sp        : Number of screened shell pairs (SSP)
//   h2eri->shell_bf_sidx : Array, size nshell, indices of each shell's 
//                          first basis function
//   h2eri->sp_bfp_sidx   : Array, size num_sp+1, indices of each 
//                          SSP's first basis function pair
//   h2eri->sp_shell_idx  : Array, size 2 * num_sp, each row is 
//                          the contracted shell indices of a SSP
// Output parameter:
//   h2eri->unc_denmat_x  : Array, size num_sp_bfp, uncontracted density matrix
void H2ERI_uncontract_den_mat(H2ERI_p h2eri, const double *den_mat)
{
    int num_bf = h2eri->num_bf;
    int num_sp = h2eri->num_sp;
    int *shell_bf_sidx = h2eri->shell_bf_sidx;
    int *sp_bfp_sidx   = h2eri->sp_bfp_sidx;
    int *sp_shell_idx  = h2eri->sp_shell_idx;
    double *x = h2eri->unc_denmat_x;
    
    #pragma omp parallel for schedule(dynamic, 16)
    for (int i = 0; i < num_sp; i++)
    {
        int x_spos = sp_bfp_sidx[i];
        int shell_idx0 = sp_shell_idx[i];
        int shell_idx1 = sp_shell_idx[i + num_sp];
        int srow = shell_bf_sidx[shell_idx0];
        int erow = shell_bf_sidx[shell_idx0 + 1];
        int scol = shell_bf_sidx[shell_idx1];
        int ecol = shell_bf_sidx[shell_idx1 + 1];
        int nrow = erow - srow;
        int ncol = ecol - scol;
        double sym_coef = (shell_idx0 == shell_idx1) ? 1.0 : 2.0;
        
        // Originally we need to store den_mat[srow:erow-1, scol:ecol-1]
        // column by column to x(x_spos:x_epos-1). Since den_mat is 
        // symmetric, we store den_mat[scol:ecol-1, srow:erow-1] row by 
        // row to x(x_spos:x_epos-1).
        for (int j = 0; j < ncol; j++)
        {
            const double *den_mat_ptr = den_mat + (scol + j) * num_bf + srow;
            double *x_ptr = x + x_spos + j * nrow;
            #pragma omp simd 
            for (int k = 0; k < nrow; k++)
                x_ptr[k] = sym_coef * den_mat_ptr[k];
        }
    }
}

// "Contract" the H2 matvec result according to SSP and reshape
// the result to form a symmetric Coulomb matrix
// Input parameters:
//   h2eri->num_bf        : Number of basis functions in the system
//   h2eri->num_sp        : Number of SSP
//   h2eri->shell_bf_sidx : Array, size nshell, indices of each shell's 
//                          first basis function
//   h2eri->sp_bfp_sidx   : Array, size num_sp+1, indices of each 
//                          SSP's first basis function pair
//   h2eri->sp_shell_idx  : Array, size 2 * num_sp, each row is 
//                          the contracted shell indices of a SSP
//   h2eri->H2_matvec_y   : Array, size num_sp_bfp, H2 matvec result 
// Output parameter:
//   J_mat : Symmetric Coulomb matrix, size h2eri->num_bf^2
void H2ERI_contract_H2_matvec(H2ERI_p h2eri, double *J_mat)
{
    int num_bf = h2eri->num_bf;
    int num_sp = h2eri->num_sp;
    int *shell_bf_sidx = h2eri->shell_bf_sidx;
    int *sp_bfp_sidx   = h2eri->sp_bfp_sidx;
    int *sp_shell_idx  = h2eri->sp_shell_idx;
    double *y = h2eri->H2_matvec_y;
    
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < num_bf * num_bf; i++) J_mat[i] = 0.0;
    
    #pragma omp parallel for schedule(dynamic, 16)
    for (int i = 0; i < num_sp; i++)
    {
        int y_spos = sp_bfp_sidx[i];
        int shell_idx0 = sp_shell_idx[i];
        int shell_idx1 = sp_shell_idx[i + num_sp];
        int srow = shell_bf_sidx[shell_idx0];
        int erow = shell_bf_sidx[shell_idx0 + 1];
        int scol = shell_bf_sidx[shell_idx1];
        int ecol = shell_bf_sidx[shell_idx1 + 1];
        int nrow = erow - srow;
        int ncol = ecol - scol;
        double sym_coef = (shell_idx0 == shell_idx1) ? 0.5 : 1.0;
        
        // Originally we need to reshape y(y_spos:y_epos-1) as a
        // nrow-by-ncol column-major matrix and add it to column-major
        // matrix J_mat[srow:erow-1, scol:ecol-1]. Since J_mat is 
        // symmetric, we reshape y(y_spos:y_epos-1) as a ncol-by-nrow
        // row-major matrix and add it to J_mat[scol:ecol-1, srow:erow-1].
        for (int j = 0; j < ncol; j++)
        {
            double *J_mat_ptr = J_mat + (scol + j) * num_bf + srow;
            double *y_ptr = y + y_spos + j * nrow;
            #pragma omp simd 
            for (int k = 0; k < nrow; k++) J_mat_ptr[k] += sym_coef * y_ptr[k];
        }
    }
    
    // Symmetrize the Coulomb matrix: J_mat = J_mat + J_mat^T
    #pragma omp parallel for schedule(dynamic, 16)
    for (int i = 0; i < num_bf; i++)
    {
        for (int j = 0; j < i; j++)
        {
            int idx0 = i * num_bf + j;
            int idx1 = j * num_bf + i;
            double val = J_mat[idx0] + J_mat[idx1];
            J_mat[idx0] = val;
            J_mat[idx1] = val;
        }
        int idx_ii = i * num_bf + i;
        J_mat[idx_ii] += J_mat[idx_ii];
    }
}

// Build the Coulomb matrix using the density matrix, H2 representation
// of the ERI tensor, and H2 matvec
void H2ERI_build_Coulomb(H2ERI_p h2eri, const double *den_mat, double *J_mat)
{
    if (h2eri->unc_denmat_x == NULL)
    {
        size_t vec_msize = sizeof(double) * h2eri->num_sp_bfp;
        h2eri->unc_denmat_x = (double *) malloc(vec_msize);
        h2eri->H2_matvec_y  = (double *) malloc(vec_msize);
        assert(h2eri->unc_denmat_x != NULL && h2eri->H2_matvec_y != NULL);
    }
    
    H2ERI_uncontract_den_mat(h2eri, den_mat);
    
    H2ERI_matvec(h2eri, h2eri->unc_denmat_x, h2eri->H2_matvec_y);
    
    H2ERI_contract_H2_matvec(h2eri, J_mat);
}

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>

#include "H2Pack_utils.h"
#include "H2Pack_matvec.h"
#include "H2ERI_typedef.h"


void H2ERI_uncontract_den_mat(H2ERI_t h2eri, const double *den_mat)
{
    int num_bf = h2eri->num_bf;
    int num_unc_sp = h2eri->num_unc_sp;
    int *shell_bf_sidx    = h2eri->shell_bf_sidx;
    int *unc_sp_bfp_sidx  = h2eri->unc_sp_bfp_sidx;
    int *unc_sp_shell_idx = h2eri->unc_sp_shell_idx;
    double *x = h2eri->unc_denmat_x;
    
    for (int i = 0; i < num_unc_sp; i++)
    {
        int x_spos = unc_sp_bfp_sidx[i];
        int x_epos = unc_sp_bfp_sidx[i + 1];
        int shell_idx0 = unc_sp_shell_idx[i];
        int shell_idx1 = unc_sp_shell_idx[i + num_unc_sp];
        int srow = shell_bf_sidx[shell_idx0];
        int erow = shell_bf_sidx[shell_idx0 + 1];
        int scol = shell_bf_sidx[shell_idx1];
        int ecol = shell_bf_sidx[shell_idx1 + 1];
        int nrow = erow - srow;
        int ncol = ecol - scol;
        double sym_coef = (shell_idx0 == shell_idx1) ? 1.0 : 2.0;
        
        // Originally we need to store den_mat[srow:erow-1, scol:ecol-1]
        // column by column to x(x_spos:x_epos-1). Since den_mat is 
        // symmetric,we store den_mat[scol:ecol-1, srow:erow-1] row by 
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

// Build the Coulomb matrix with the density matrix and H2 
// representation of the ERI tensor.
void H2ERI_build_Coulomb(H2ERI_t h2eri, const double *den_mat, double *J_mat)
{
    if (h2eri->unc_denmat_x == NULL)
    {
        size_t vec_msize = sizeof(double) * h2eri->num_unc_sp_bfp;
        h2eri->unc_denmat_x = (double *) malloc(vec_msize);
        h2eri->H2_matvec_y  = (double *) malloc(vec_msize);
        assert(h2eri->unc_denmat_x != NULL && h2eri->H2_matvec_y != NULL);
    }
    
    H2ERI_uncontract_den_mat(h2eri, den_mat);
}

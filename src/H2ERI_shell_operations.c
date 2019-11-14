#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

#include "mkl.h"

#include "H2ERI_typedef.h"

// Fully uncontract all shells into new shells that each shell has
// only 1 primitive function and screen uncontracted shell pairs
// Input parameters:
//   h2eri->nshell  : Number of original shells 
//   h2eri->shells  : Array, size nshell, original shells
//   h2eri->scr_tol : Schwarz screening tolerance, typically 1e-10
// Output parameters:
//   h2eri->num_unc_sp       : Number of uncontracted shell pairs that survives screening
//   h2eri->unc_sp_shells    : Array, size 2 * num_unc_sp, uncontracted screened shell pairs
//   h2eri->unc_sp_center    : Array, size 3 * num_unc_sp, each column is the center 
//                             coordinate of a new uncontracted shell pair
//   h2eri->unc_sp_shell_idx : Array, size 2 * num_unc_sp, each row is the contracted 
//                             shell indices of a FUSP
void H2ERI_uncontract_shell_pairs(H2ERI_t h2eri)
{
    int     nshell  = h2eri->nshell;
    shell_t *shells = h2eri->shells;
    double  scr_tol = h2eri->scr_tol;
    
    // 1. Uncontract all shells
    int nshell_unc = 0;
    for (int i = 0; i < nshell; i++) nshell_unc += shells[i].nprim;
    int *shells_unc_idx = (int *) malloc(sizeof(int) * nshell_unc * 2);
    shell_t *shells_unc = (shell_t *) malloc(sizeof(shell_t) * nshell_unc);
    assert(shells_unc_idx != NULL && shells_unc != NULL);
    for (int i = 0; i < nshell_unc; i++)
    {
        simint_initialize_shell(&shells_unc[i]);
        simint_allocate_shell(1, &shells_unc[i]);
    }
    int unc_idx = 0;
    for (int i = 0; i < nshell; i++)
    {
        int am = shells[i].am;
        double x = shells[i].x;
        double y = shells[i].y;
        double z = shells[i].z;
        for (int j = 0; j < shells[i].nprim; j++)
        {
            shells_unc[unc_idx].am       = am;
            shells_unc[unc_idx].nprim    = 1;
            shells_unc[unc_idx].x        = x;
            shells_unc[unc_idx].y        = y;
            shells_unc[unc_idx].z        = z;
            shells_unc[unc_idx].alpha[0] = shells[i].alpha[j];
            shells_unc[unc_idx].coef[0]  = shells[i].coef[j];
            shells_unc_idx[2*unc_idx]    = i;
            shells_unc_idx[2*unc_idx+1]  = j;
            unc_idx++;
        }
    }
    
    // 2. Construct new shell pairs with uncontracted shells
    double *scr_vals = (double *) malloc(sizeof(double) * nshell_unc * nshell_unc);
    assert(scr_vals != NULL);
    double max_scr_val = CMS_get_Schwarz_scrval(nshell_unc, shells_unc, scr_vals);
    double scr_thres = scr_tol * scr_tol / max_scr_val;
    int num_unc_sp = 0;
    for (int i = 0; i < nshell_unc; i++)
    {
        double *src_vals_row = scr_vals + i * nshell_unc;
        for (int j = i; j < nshell_unc; j++)
            if (src_vals_row[j] >= scr_thres) num_unc_sp++;
    }
    
    h2eri->num_unc_sp = num_unc_sp;
    h2eri->unc_sp_shells    = (shell_t *) malloc(sizeof(shell_t) * num_unc_sp * 2);
    h2eri->unc_sp_center    = (double *)  malloc(sizeof(double)  * num_unc_sp * 3);
    h2eri->unc_sp_shell_idx = (int *)     malloc(sizeof(int)     * num_unc_sp * 2);
    assert(h2eri->unc_sp_center != NULL && h2eri->unc_sp_shells != NULL);
    assert(h2eri->unc_sp_shell_idx != NULL);
    double  *unc_sp_center    = h2eri->unc_sp_center;
    shell_t *unc_sp_shells    = h2eri->unc_sp_shells;
    int     *unc_sp_shell_idx = h2eri->unc_sp_shell_idx;
    
    for (int i = 0; i < num_unc_sp * 2; i++)
    {
        simint_initialize_shell(&unc_sp_shells[i]);
        simint_allocate_shell(1, &unc_sp_shells[i]);
    }
    int cidx0 = 0, cidx1 = num_unc_sp, cidx2 = 2 * num_unc_sp;
    const double sqrt2 = sqrt(2.0);
    for (int i = 0; i < nshell_unc; i++)
    {
        double *src_vals_row = scr_vals + i * nshell_unc;
        double a_i = shells_unc[i].alpha[0];
        double x_i = shells_unc[i].x;
        double y_i = shells_unc[i].y;
        double z_i = shells_unc[i].z;
        for (int j = i; j < nshell_unc; j++)
        {
            if (src_vals_row[j] < scr_thres) continue;
            
            // Add a new shell pair
            double a_j = shells_unc[j].alpha[0];
            double x_j = shells_unc[j].x;
            double y_j = shells_unc[j].y;
            double z_j = shells_unc[j].z;
            double aij = a_i + a_j;
            unc_sp_center[cidx0] = (a_i * x_i + a_j * x_j) / aij;
            unc_sp_center[cidx1] = (a_i * y_i + a_j * y_j) / aij;
            unc_sp_center[cidx2] = (a_i * z_i + a_j * z_j) / aij;
            simint_copy_shell(&shells_unc[i], &unc_sp_shells[cidx0]);
            simint_copy_shell(&shells_unc[j], &unc_sp_shells[cidx1]);
            
            // If two shell_uncs come from the same contracted shell but are
            // different primitive functions, multiple a sqrt(2) for symmetry.
            // Let shell A = a1 + a2, B = b1 + b2, (AB| = \sum \sum a_i b_j.
            // If A == B, due to symmetry, we only handle (a1a2| once.
            int shell_idx_i = shells_unc_idx[2 * i];
            int prim_idx_i  = shells_unc_idx[2 * i + 1];
            int shell_idx_j = shells_unc_idx[2 * j];
            int prim_idx_j  = shells_unc_idx[2 * j + 1];
            if ((shell_idx_i == shell_idx_j) && (prim_idx_i != prim_idx_j))
            {
                unc_sp_shells[cidx0].coef[0] *= sqrt2;
                unc_sp_shells[cidx1].coef[0] *= sqrt2;
            }
            
            unc_sp_shell_idx[cidx0] = shell_idx_i;
            unc_sp_shell_idx[cidx1] = shell_idx_j;
            
            cidx0++;
            cidx1++;
            cidx2++;
        }  // End of j loop
    }  // End of i loop
    
    // 3. Free temporary arrays
    free(scr_vals);
    for (int i = 0; i < nshell_unc; i++)
        simint_free_shell(&shells_unc[i]);
    free(shells_unc);
    free(shells_unc_idx);
}

// Estimate the extents of Gaussian function 
//       coef * exp(-alpha * x^2) * x^am
// Input parameters:
//   alpha, coef, am : Gaussian function parameter
//   tol : Tolerance of extent
// Output parameter:
//   <return> : Estimated extent of given Gaussian function
#define Gaussian_tol(x) (coef * exp(-alpha * (x) * (x)) * pow((x), am) - tol)
double H2ERI_calc_Gaussian_extent(
    const double alpha, const double coef, 
    const int _am,      const double tol
)
{
    double am = (double) _am;
    double lower = 0.0, upper = 20.0;
    
    // If am > 0, the maximum of Gaussian is obtained at x = sqrt(am / 2 / alpha)
    if (_am > 0) 
    {
        lower = sqrt(am * 0.5 / alpha);
        upper = 10.0 * lower;
    }
    
    if (Gaussian_tol(lower) <= 1e-8 * tol) return 0.0;
    while (Gaussian_tol(upper) > 0) upper *= 2.0;
    
    double extent   = (upper + lower) * 0.5;
    double G_extent = Gaussian_tol(extent);
    while (fabs(G_extent) > 2e-16)
    {
        if (G_extent > 0.0) lower = extent; 
        else                upper = extent;
        extent   = (upper + lower) * 0.5;
        G_extent = Gaussian_tol(extent);
    }
    
    return extent;
}

// Calculate the extent (numerical support radius) of FUSP
// Input parameters:
//   h2eri->num_unc_sp    : Number of FUSP
//   h2eri->unc_sp_shells : Array, size 2 * num_sp, each column is a FUSP
//   h2eri->ext_tol       : Tolerance of shell pair extent
// Output parameters:
//   h2eri->unc_sp_extent : Array, size h2eri->num_unc_sp, extents of each shell pair
void H2ERI_calc_unc_sp_extents(H2ERI_t h2eri)
{
    h2eri->unc_sp_extent = (double *) malloc(sizeof(double) * h2eri->num_unc_sp);
    assert(h2eri->unc_sp_extent != NULL);
    
    int num_unc_sp = h2eri->num_unc_sp;
    shell_t *unc_sp_shells = h2eri->unc_sp_shells;
    double ext_tol = h2eri->ext_tol;
    double *unc_sp_extent = h2eri->unc_sp_extent;

    for (int i = 0; i < num_unc_sp; i++)
    {
        shell_t *shell0 = unc_sp_shells + i;
        shell_t *shell1 = unc_sp_shells + i + num_unc_sp;
        int    am12    = shell0->am + shell1->am;
        int    nprim12 = shell0->nprim * shell1->nprim;
        double dx      = shell0->x - shell1->x;
        double dy      = shell0->y - shell1->y;
        double dz      = shell0->z - shell1->z;
        double coef12  = shell0->coef[0] * shell1->coef[0];
        double alpha1  = shell0->alpha[0];
        double alpha2  = shell1->alpha[0];
        double tol_i   = ext_tol / (double) nprim12;
        double r12     = dx * dx + dy * dy + dz * dz;
        double alpha12 = alpha1 + alpha2;
        double exp_c   = (alpha1 * alpha2 / alpha12) * r12;
        double coef    = coef12 * exp(-exp_c);
        
        unc_sp_extent[i] = H2ERI_calc_Gaussian_extent(alpha12, coef, am12, tol_i);
    }
}

// Process input shells for H2 partitioning
void H2ERI_process_shells(H2ERI_t h2eri)
{
    H2ERI_uncontract_shell_pairs(h2eri);
    H2ERI_calc_unc_sp_extents(h2eri);
}


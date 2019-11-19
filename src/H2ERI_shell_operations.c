#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

#include "mkl.h"

#include "H2ERI_typedef.h"

// Schwarz screening for shell pairs
// Input parameters:
//   h2eri->nshell  : Number of original shells 
//   h2eri->shells  : Array, size nshell, original shells
//   h2eri->scr_tol : Schwarz screening tolerance, typically 1e-10
// Output parameters:
//   h2eri->num_sp       : Number of screened shell pairs (SSP)
//   h2eri->sp_shells    : Array, size 2 * num_sp, shells of the screened shell pairs 
//   h2eri->sp_center    : Array, size 3 * num_sp, each column is the center 
//                         coordinate of a screened shell pair
//   h2eri->sp_shell_idx : Array, size 2 * num_sp, each row is the contracted 
//                         shell indices of a screened shell pair
void H2ERI_screen_shell_pairs(H2ERI_t h2eri)
{
    int     nshell  = h2eri->nshell;
    shell_t *shells = h2eri->shells;
    double  scr_tol = h2eri->scr_tol;

    double *scr_vals = (double *) malloc(sizeof(double) * nshell * nshell);
    assert(scr_vals != NULL);
    double max_scr_val = CMS_get_Schwarz_scrval(nshell, shells, scr_vals);
    double scr_thres = scr_tol * scr_tol / max_scr_val;
    int num_sp = 0;
    for (int i = 0; i < nshell; i++)
    {
        double *src_vals_row = scr_vals + i * nshell;
        for (int j = i; j < nshell; j++)
            if (src_vals_row[j] >= scr_thres) num_sp++;
    }
    
    h2eri->num_sp = num_sp;
    h2eri->sp_shells    = (shell_t *) malloc(sizeof(shell_t) * num_sp * 2);
    h2eri->sp_shell_idx = (int *)     malloc(sizeof(int)     * num_sp * 2);
    assert(h2eri->sp_shells    != NULL);
    assert(h2eri->sp_shell_idx != NULL);
    shell_t *sp_shells    = h2eri->sp_shells;
    int     *sp_shell_idx = h2eri->sp_shell_idx;

    simint_initialize_shells(num_sp * 2, sp_shells);
    int cidx0 = 0, cidx1 = num_sp;
    for (int i = 0; i < nshell; i++)
    {
        double *src_vals_row = scr_vals + i * nshell;
        for (int j = i; j < nshell; j++)
        {
            if (src_vals_row[j] < scr_thres) continue;
            simint_allocate_shell(shells[i].nprim, &sp_shells[cidx0]);
            simint_allocate_shell(shells[j].nprim, &sp_shells[cidx1]);
            simint_copy_shell(&shells[i], &sp_shells[cidx0]);
            simint_copy_shell(&shells[j], &sp_shells[cidx1]);
            sp_shell_idx[cidx0] = i;
            sp_shell_idx[cidx1] = j;
            cidx0++;
            cidx1++;
        }
    }

    free(scr_vals);
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

// Calculate the extent (numerical support radius) of screened shell pairs (SSP)
// Input parameters:
//   h2eri->num_sp    : Number of SSP
//   h2eri->sp_shells : Array, size 2 * num_sp, each column is a SSP
//   h2eri->ext_tol   : Tolerance of shell pair extent
// Output parameters:
//   h2eri->sp_extent : Array, size h2eri->num_sp, extents of each SSP
void H2ERI_calc_sp_extents(H2ERI_t h2eri)
{
    h2eri->sp_extent = (double *) malloc(sizeof(double) * h2eri->num_sp);
    h2eri->sp_center = (double *) malloc(sizeof(double) * h2eri->num_sp * 3);
    assert(h2eri->sp_extent != NULL);
    assert(h2eri->sp_center != NULL);
    
    int num_sp = h2eri->num_sp;
    double ext_tol = h2eri->ext_tol;
    shell_t *sp_shells = h2eri->sp_shells;
    double  *sp_extent = h2eri->sp_extent;
    double  *sp_center = h2eri->sp_center;

    int max_nprim01 = 0;
    double *extent_i = NULL;
    double *center_i = NULL;
    double *upper_i  = NULL;
    double *lower_i  = NULL;

   
    for (int i = 0; i < num_sp; i++)
    {
        shell_t *shell0 = sp_shells + i;
        shell_t *shell1 = sp_shells + i + num_sp;
        int     nprim0  = shell0->nprim;
        int     nprim1  = shell1->nprim;
        int     nprim01 = nprim0 * nprim1;
        int     am01    = shell0->am + shell1->am;
        double *alpha0  = shell0->alpha;
        double *alpha1  = shell1->alpha;
        double *coef0   = shell0->coef;
        double *coef1   = shell1->coef;
        double dx       = shell0->x - shell1->x;
        double dy       = shell0->y - shell1->y;
        double dz       = shell0->z - shell1->z;
        double tol_i    = ext_tol / (double) nprim01;
        double r01      = dx * dx + dy * dy + dz * dz;
        double d01      = sqrt(r01);
        if (nprim01 > max_nprim01)
        {
            free(extent_i);
            free(center_i);
            free(upper_i);
            free(lower_i);
            max_nprim01 = nprim01;
            extent_i = (double*) malloc(sizeof(double) * nprim01 * 2);
            center_i = (double*) malloc(sizeof(double) * nprim01 * 3 * 2);
            upper_i  = (double*) malloc(sizeof(double) * nprim01 * 3);
            lower_i  = (double*) malloc(sizeof(double) * nprim01 * 3);
        }

        // 1. Calculate the extent of each primitive function pair
        double max_extent_i = 0.0;
        for (int j0 = 0; j0 < nprim0; j0++)
        {
            for (int j1 = 0; j1 < nprim1; j1++)
            {
                double aj0   = alpha0[j0];
                double aj1   = alpha1[j1];
                double aj0p1 = aj0 + aj1;
                double aj0m1 = aj0 * aj1;
                double cj01  = coef0[j0] * coef1[j1];
                double exp_c = aj0m1 / aj0p1 * r01;
                double coef  = cj01 * exp(-exp_c);
                int j01 = j0 * nprim1 + j1;
                extent_i[j01] = H2ERI_calc_Gaussian_extent(aj0p1, coef, am01, tol_i);
                max_extent_i = (max_extent_i < extent_i[j01]) ? extent_i[j01] : max_extent_i;
                double inv_aj0p1 = 1.0 / aj0p1;
                center_i[0 * nprim01 + j01] = (aj0 * shell0->x + aj1 * shell1->x) * inv_aj0p1;
                center_i[1 * nprim01 + j01] = (aj0 * shell0->y + aj1 * shell1->y) * inv_aj0p1;
                center_i[2 * nprim01 + j01] = (aj0 * shell0->z + aj1 * shell1->z) * inv_aj0p1;
            }
        }

        // 2. Find a large box to cover all extents
        if (d01 < 1e-16)
        {
            sp_extent[i] = max_extent_i;
            sp_center[0 * num_sp + i] = shell0->x;
            sp_center[1 * num_sp + i] = shell0->y;
            sp_center[2 * num_sp + i] = shell0->z;
        } else {
            if (max_extent_i < 1e-16)
            {
                sp_extent[i] = 0.0;
                double sx = 0.0, sy = 0.0, sz = 0.0;
                for (int j = 0; j < nprim01; j++)
                {
                    sx += center_i[0 * nprim01 + j];
                    sy += center_i[1 * nprim01 + j];
                    sz += center_i[2 * nprim01 + j];
                }
                sp_center[0 * num_sp + i] = sx / (double) nprim01;
                sp_center[1 * num_sp + i] = sy / (double) nprim01;
                sp_center[2 * num_sp + i] = sz / (double) nprim01;
                continue;
            }

            dx /= d01;  dy /= d01;  dz /= d01;
            double *extent_i_nnz = extent_i + nprim01;
            double *center_i_nnz = center_i + nprim01 * 3;
            // Filter the nonzero extents and their center
            int extent_nnz = 0;
            for (int j = 0; j < nprim01; j++)
            {
                if (extent_i[j] > 1e-16)
                {
                    extent_i_nnz[extent_nnz] = extent_i[j];
                    center_i_nnz[0 * nprim01 + extent_nnz] = center_i[0 * nprim01 + j];
                    center_i_nnz[1 * nprim01 + extent_nnz] = center_i[1 * nprim01 + j];
                    center_i_nnz[2 * nprim01 + extent_nnz] = center_i[2 * nprim01 + j];
                    extent_nnz++;
                }
            }
            // Calculate the upper and lower bound of each extent
            for (int j = 0; j < extent_nnz; j++)
            {
                double center_ij_x = center_i_nnz[0 * nprim01 + j];
                double center_ij_y = center_i_nnz[1 * nprim01 + j];
                double center_ij_z = center_i_nnz[2 * nprim01 + j];
                double extent_ij   = extent_i_nnz[j];
                upper_i[0 * extent_nnz + j] = center_ij_x + extent_ij * dx;
                upper_i[1 * extent_nnz + j] = center_ij_y + extent_ij * dy;
                upper_i[2 * extent_nnz + j] = center_ij_z + extent_ij * dz;
                lower_i[0 * extent_nnz + j] = center_ij_x - extent_ij * dx;
                lower_i[1 * extent_nnz + j] = center_ij_y - extent_ij * dy;
                lower_i[2 * extent_nnz + j] = center_ij_z - extent_ij * dz;
            }
            // j0's upper bound of extent has the largest distance to x1
            // j1's lower bound of extent has the largest distance to x0 
            int j0 = -1, j1 = -1;
            double jc0_max = 0.0, jc1_max = 0.0;
            for (int j = 0; j < extent_nnz; j++)
            {
                dx = upper_i[0 * extent_nnz + j] - shell1->x;
                dy = upper_i[1 * extent_nnz + j] - shell1->y;
                dz = upper_i[2 * extent_nnz + j] - shell1->z;
                double dist_j_c0 = dx * dx + dy * dy + dz * dz;
                if (dist_j_c0 > jc0_max)
                {
                    jc0_max = dist_j_c0;
                    j0 = j;
                }
                dx = lower_i[0 * extent_nnz + j] - shell0->x;
                dy = lower_i[1 * extent_nnz + j] - shell0->y;
                dz = lower_i[2 * extent_nnz + j] - shell0->z;
                double dist_j_c1 = dx * dx + dy * dy + dz * dz;
                if (dist_j_c1 > jc1_max)
                {
                    jc1_max = dist_j_c1;
                    j1 = j;
                }
            }
            double upper_x = upper_i[0 * extent_nnz + j0];
            double upper_y = upper_i[1 * extent_nnz + j0];
            double upper_z = upper_i[2 * extent_nnz + j0];
            double lower_x = lower_i[0 * extent_nnz + j1];
            double lower_y = lower_i[1 * extent_nnz + j1];
            double lower_z = lower_i[2 * extent_nnz + j1];
            dx = upper_x - lower_x;
            dy = upper_y - lower_y;
            dz = upper_z - lower_z;
            sp_extent[i] = 0.5 * sqrt(dx * dx + dy * dy + dz * dz);
            sp_center[0 * num_sp + i] = 0.5 * (upper_x + lower_x);
            sp_center[1 * num_sp + i] = 0.5 * (upper_y + lower_y);
            sp_center[2 * num_sp + i] = 0.5 * (upper_z + lower_z);
        }  // End of "if (d01 < 1e-16)"
    }  // End of i loop

    free(extent_i);
    free(center_i);
    free(upper_i);
    free(lower_i);
}

// Process input shells for H2 partitioning
void H2ERI_process_shells(H2ERI_t h2eri)
{
    H2ERI_screen_shell_pairs(h2eri);
    H2ERI_calc_sp_extents(h2eri);
}


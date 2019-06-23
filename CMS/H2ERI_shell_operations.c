#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

#include "mkl.h"

#include "CMS.h"

// Rotate shell coordinates for better hierarchical partitioning
void H2ERI_rotate_shells(const int nshell, shell_t *shells)
{
    size_t col_msize = sizeof(double) * nshell;
    double center[3] = {0.0, 0.0, 0.0};
    double eigval[3], eigvec[9];
    double *coord  = (double*) malloc(col_msize * 3);
    double *coord1 = (double*) malloc(col_msize * 3);
    assert(coord != NULL && coord1 != NULL);
    
    // 1. Rotate coordinates so the center is at the origin point & 
    // the minimal bounding box of center points is parallel to unit box
    for (int i = 0; i < nshell; i++)
    {
        coord[0 * nshell + i] = shells[i].x;
        coord[1 * nshell + i] = shells[i].y;
        coord[2 * nshell + i] = shells[i].z;
        center[0] += shells[i].x;
        center[1] += shells[i].y;
        center[2] += shells[i].z;
    }
    double d_nshell = (double) nshell;
    center[0] = center[0] / d_nshell;
    center[1] = center[1] / d_nshell;
    center[2] = center[2] / d_nshell;
    for (int i = 0; i < nshell; i++)
    {
        coord[0 * nshell + i] -= center[0];
        coord[1 * nshell + i] -= center[1];
        coord[2 * nshell + i] -= center[2];
    }
    cblas_dgemm(
        CblasColMajor, CblasTrans, CblasNoTrans, 3, 3, nshell,
        1.0, coord, nshell, coord, nshell, 0.0, eigvec, 3
    );
    LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'V', 'U', 3, eigvec, 3, eigval);
    cblas_dgemm(
        CblasColMajor, CblasNoTrans, CblasTrans, nshell, 3, 3, 
        1.0, coord, nshell, eigvec, 3, 0.0, coord1, nshell
    );
    
    // 2. Move the zero columns to the end
    int col_idx = 0;
    for (int i = 0; i < 3; i++)
    {
        double *col_ptr = coord1 + i * nshell;
        double max_col_val = fabs(col_ptr[0]);
        for (int j = 1; j < nshell; j++)
        {
            double val_j = fabs(col_ptr[j]);
            if (val_j > max_col_val) max_col_val = val_j;
        }
        if (max_col_val > 1e-15)
        {
            double *dst_col = coord1 + col_idx * nshell;
            if (col_idx != i) memcpy(dst_col, col_ptr, col_msize);
            col_idx++;
        }
    }
    
    // 3. Update the center coordinates of shells
    for (int i = 0; i < nshell; i++)
    {
        shells[i].x = coord1[0 * nshell + i];
        shells[i].y = coord1[1 * nshell + i];
        shells[i].z = coord1[2 * nshell + i];
    }
    
    free(coord1);
    free(coord);
}
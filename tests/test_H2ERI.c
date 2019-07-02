#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>

#include "H2ERI.h"

int main(int argc, char **argv)
{
    simint_init();
    
    H2ERI_t h2eri;
    H2ERI_init(&h2eri, 1e-10, 1e-10, 1e-6);
    
    // 1. Read molecular file
    CMS_read_mol_file(argv[1], &h2eri->nshell, &h2eri->shells);
    
    // 2. Process input shells for H2 partitioning
    H2ERI_process_shells(h2eri);
    
    // 3. H2 partition of uncontracted shell pair centers
    H2ERI_partition(h2eri);
    
    // 4. Build H2 representation for ERI tensor
    H2ERI_build(h2eri);
    
    int num_bf = h2eri->num_bf;
    double *den_mat = (double *) malloc(sizeof(double) * num_bf * num_bf);
    double *J_mat   = (double *) malloc(sizeof(double) * num_bf * num_bf);
    for (int i = 0; i < num_bf; i++)
    {
        double *dem_mat_row = den_mat + i * num_bf;
        for (int j = 0; j < num_bf; j++)
            dem_mat_row[j] = (double) (i + j);
    }
    
    H2ERI_build_Coulomb(h2eri, den_mat, J_mat);
    FILE *ouf = fopen("add_c_x.m", "w");
    fprintf(ouf, "c_x = [\n");
    for (int i = 0; i < h2eri->num_unc_sp_bfp; i++)
        fprintf(ouf, "%lf\n", h2eri->unc_denmat_x[i]);
    fprintf(ouf, "];");
    fclose(ouf);
    
    free(J_mat);
    free(den_mat);
    
    simint_finalize();
    return 0;
}
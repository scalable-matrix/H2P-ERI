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
    CMS_read_mol_file(argv[1], &h2eri->natom, &h2eri->nshell, &h2eri->shells);
    
    // 2. Process input shells for H2 partitioning
    H2ERI_process_shells(h2eri);
    
    // 3. H2 partition of uncontracted shell pair centers
    H2ERI_partition(h2eri);
    
    // 4. Build H2 representation for ERI tensor
    H2ERI_build_H2(h2eri);

    // 5. Generate a symmetric density matrix and save it to file
    int num_bf = h2eri->num_bf;
    double *den_mat = (double *) malloc(sizeof(double) * num_bf * num_bf);
    double *J_mat   = (double *) malloc(sizeof(double) * num_bf * num_bf);
    for (int i = 0; i < num_bf; i++)
    {
        double *dem_mat_row = den_mat + i * num_bf;
        for (int j = 0; j < num_bf; j++)
            dem_mat_row[j] = (double) ((i + j) % 11);
    }
    
    FILE *ouf = fopen("add_c_x.m", "w");
    fprintf(ouf, "c_x = [\n");
    for (int i = 0; i < num_bf; i++)
    {
        double *dem_mat_row = den_mat + i * num_bf;
        for (int j = 0; j < num_bf; j++)
            fprintf(ouf, "%.16lf ", dem_mat_row[j]);
        fprintf(ouf, "\n");
    }
    fprintf(ouf, "];");
    fclose(ouf);
    
    // 6. Construct the Coulomb matrix and save it to file
    H2ERI_build_Coulomb(h2eri, den_mat, J_mat);  // Warm up
    h2eri->h2pack->n_matvec = 0;
    memset(h2eri->h2pack->timers + 4, 0, sizeof(double) * 5);
    for (int k = 0; k < 10; k++)
         H2ERI_build_Coulomb(h2eri, den_mat, J_mat);
    
    ouf = fopen("add_c_y.m", "w");
    fprintf(ouf, "c_y = [\n");
    for (int i = 0; i < num_bf; i++)
    {
        double *J_mat_row = J_mat + i * num_bf;
        for (int j = 0; j < num_bf; j++)
            fprintf(ouf, "%.16lf ", J_mat_row[j]);
        fprintf(ouf, "\n");
    }
    fprintf(ouf, "];");
    fclose(ouf);
    
    H2ERI_print_statistic(h2eri);
    
    free(J_mat);
    free(den_mat);
    
    simint_finalize();
    return 0;
}
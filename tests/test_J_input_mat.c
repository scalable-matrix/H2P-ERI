#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>

#include "H2ERI.h"

int main(int argc, char **argv)
{
    simint_init();

    if (argc < 5)
    {
        printf("Usage: %s <mol file> <D mat bin file> <ref J mat bin file> <relerr>\n", argv[0]);
        return 255;
    }
    
    H2ERI_p h2eri;
    H2ERI_init(&h2eri, 1e-10, 1e-10, atof(argv[4]));
    
    // 1. Read molecular file
    CMS_read_mol_file(argv[1], &h2eri->natom, &h2eri->nshell, &h2eri->shells);
    
    // 2. Process input shells for H2 partitioning
    H2ERI_process_shells(h2eri);
    
    // 3. H2 partition of screened shell pair centers
    H2ERI_partition(h2eri);
    
    // 4. Build H2 representation for ERI tensor
    H2ERI_build_H2(h2eri, 0);

    // 5. Read the reference density and Coulomb matrix from binary file
    size_t nbf2 = h2eri->num_bf * h2eri->num_bf;
    double *D_mat = (double *) malloc(sizeof(double) * nbf2);
    double *J_mat = (double *) malloc(sizeof(double) * nbf2);
    double *J_ref = (double *) malloc(sizeof(double) * nbf2);
    FILE *D_file = fopen(argv[2], "r");
    FILE *J_file = fopen(argv[3], "r");
    fread(D_mat, nbf2, sizeof(double), D_file);
    fread(J_ref, nbf2, sizeof(double), J_file);
    fclose(D_file);
    fclose(J_file);
    
    // 6. Construct the Coulomb matrix and save it to file
    H2ERI_build_Coulomb(h2eri, D_mat, J_mat);  // Warm up
    h2eri->n_matvec = 0;
    memset(h2eri->timers + 4, 0, sizeof(double) * 5);
    for (int k = 0; k < 10; k++)
        H2ERI_build_Coulomb(h2eri, D_mat, J_mat);
    
    H2ERI_print_statistic(h2eri);

    // 7. Calculate the relative error
    double ref_l2 = 0.0, err_l2 = 0.0;
    for (int i = 0; i < nbf2; i++)
    {
        double diff = J_ref[i] - J_mat[i];
        ref_l2 += J_ref[i] * J_ref[i];
        err_l2 += diff * diff;
    }
    ref_l2 = sqrt(ref_l2);
    err_l2 = sqrt(err_l2);
    printf("||J_{H2} - J_{ref}||_2 / ||J_{ref}||_2 = %e\n", err_l2 / ref_l2);
    
    free(J_ref);
    free(J_mat);
    free(D_mat);
    
    simint_finalize();
    return 0;
}

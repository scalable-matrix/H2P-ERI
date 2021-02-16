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
        printf("Usage: %s <mol file> <D mat bin file> <ref K mat bin file> <relerr>\n", argv[0]);
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
    printf("H2ERI_build_H2 done\n");

    // 5. Read the reference density and Coulomb matrix from binary file
    size_t nbf2 = h2eri->num_bf * h2eri->num_bf;
    double *D_mat = (double *) malloc(sizeof(double) * nbf2);
    double *K_mat = (double *) malloc(sizeof(double) * nbf2);
    double *K_ref = (double *) malloc(sizeof(double) * nbf2);
    FILE *D_file = fopen(argv[2], "r");
    FILE *K_file = fopen(argv[3], "r");
    fread(D_mat, nbf2, sizeof(double), D_file);
    fread(K_ref, nbf2, sizeof(double), K_file);
    fclose(D_file);
    fclose(K_file);
    
    // 6. Construct the exchange matrix
    H2ERI_build_exchange(h2eri, D_mat, K_mat);
    
    H2ERI_print_statistic(h2eri);

    // 7. Calculate the relative error
    double ref_l2 = 0.0, err_l2 = 0.0;
    for (int i = 0; i < nbf2; i++)
    {
        double diff = K_ref[i] - K_mat[i];
        ref_l2 += K_ref[i] * K_ref[i];
        err_l2 += diff * diff;
    }
    ref_l2 = sqrt(ref_l2);
    err_l2 = sqrt(err_l2);
    printf("||J_{H2} - J_{ref}||_2 / ||J_{ref}||_2 = %e\n", err_l2 / ref_l2);
    
    free(K_ref);
    free(K_mat);
    free(D_mat);
    
    simint_finalize();
    return 0;
}

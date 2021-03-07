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
    int num_bf = h2eri->num_bf;
    size_t nbf2 = h2eri->num_bf * h2eri->num_bf;
    double *D_mat = (double *) malloc(sizeof(double) * nbf2);
    double *K_mat = (double *) malloc(sizeof(double) * nbf2);
    FILE *D_file = fopen(argv[2], "r");
    FILE *K_file = fopen(argv[3], "r");
    fread(D_mat, nbf2, sizeof(double), D_file);
    fread(K_mat, nbf2, sizeof(double), K_file);
    fclose(D_file);
    fclose(K_file);
    
    // 6. Construct the ACE matrix
    int num_occ = 0;
    double *eig_val  = (double *) malloc(sizeof(double) * num_bf);
    double *eig_vec  = (double *) malloc(sizeof(double) * nbf2);
    double *ACE_ref  = (double *) malloc(sizeof(double) * nbf2);
    double *ACE_mat  = (double *) malloc(sizeof(double) * nbf2);

    memcpy(eig_vec, D_mat, sizeof(double) * nbf2);
    LAPACKE_dsyevd(LAPACK_ROW_MAJOR, 'V', 'L', num_bf, eig_vec, num_bf, eig_val);
    for (int i = 0; i < num_bf; i++)
        if (fabs(eig_val[i]) > 1e-14) num_occ++;
    
    double *Cocc_mat = (double *) malloc(sizeof(double) * num_bf  * num_occ);
    double *KC_mat   = (double *) malloc(sizeof(double) * num_bf  * num_occ);
    double *CKC_mat  = (double *) malloc(sizeof(double) * num_occ * num_occ);
    double *tmp_mat  = (double *) malloc(sizeof(double) * num_bf  * num_occ);

    int idx = 0;
    // C0 = V(:, (nbf - nocc + 1) : nbf);
    // d0 = diag(sqrt(D0((nbf - nocc + 1) : nbf)));
    // C  = C0 * d0;
    for (int i = 0; i < num_bf; i++)
    {
        if (fabs(eig_val[i]) > 1e-14)
        {
            double sqrt_eig_val = sqrt(eig_val[i]);
            #pragma omp simd
            for (int j = 0; j < num_bf; j++) 
                Cocc_mat[j * num_occ + idx] = eig_vec[j * num_bf + i] * sqrt_eig_val;
            idx++;
        }
    }

    // KC  = Kmat * C;
    // CKC = C' * KC;
    // ACE_ref = KC * inv(CKC) * (KC)';
    CBLAS_GEMM(
        CblasRowMajor, CblasNoTrans, CblasNoTrans, num_bf, num_occ, num_bf,
        1.0, K_mat, num_bf, Cocc_mat, num_occ, 0.0, KC_mat, num_occ
    );
    CBLAS_GEMM(
        CblasRowMajor, CblasTrans, CblasNoTrans, num_occ, num_occ, num_bf,
        1.0, Cocc_mat, num_occ, KC_mat, num_occ, 0.0, CKC_mat, num_occ
    );
    int info = LAPACK_POTRF(LAPACK_ROW_MAJOR, 'L', num_occ, CKC_mat, num_occ);
    if (info != 0) ASSERT_PRINTF(info == 0, "LAPACK_POTRF returned error code %d\n", info);
    info = LAPACKE_dpotri(LAPACK_ROW_MAJOR, 'L', num_occ, CKC_mat, num_occ);
    if (info != 0) ASSERT_PRINTF(info == 0, "LAPACKE_dpotri returned error code %d\n", info);
    for (int irow = 0; irow < num_occ - 1; irow++)
    {
        for (int icol = irow + 1; icol < num_occ; icol++)
            CKC_mat[irow * num_occ + icol] = CKC_mat[icol * num_occ + irow];
    }
    CBLAS_GEMM(
        CblasRowMajor, CblasNoTrans, CblasNoTrans, num_bf, num_occ, num_occ,
        1.0, KC_mat, num_occ, CKC_mat, num_occ, 0.0, tmp_mat, num_occ
    );
    CBLAS_GEMM(
        CblasRowMajor, CblasNoTrans, CblasTrans, num_bf, num_bf, num_occ,
        1.0, tmp_mat, num_occ, KC_mat, num_occ, 0.0, ACE_ref, num_bf
    );

    H2ERI_build_ACE(h2eri, num_occ, Cocc_mat, ACE_mat);
    
    H2ERI_print_statistic(h2eri);

    // 7. Calculate the relative error
    double ref_l2 = 0.0, err_l2 = 0.0;
    for (int i = 0; i < nbf2; i++)
    {
        double diff = ACE_ref[i] - ACE_mat[i];
        ref_l2 += ACE_ref[i] * ACE_ref[i];
        err_l2 += diff * diff;
    }
    ref_l2 = sqrt(ref_l2);
    err_l2 = sqrt(err_l2);
    printf("||ACE_{H2} - ACE_{ref}||_2 / ||ACE_{ref}||_2 = %e\n", err_l2 / ref_l2);
    
    free(K_mat);
    free(D_mat);
    free(eig_val);
    free(eig_vec);
    free(ACE_ref);
    free(ACE_mat);
    free(Cocc_mat);
    free(KC_mat);
    free(CKC_mat);
    free(tmp_mat);
    
    simint_finalize();
    return 0;
}

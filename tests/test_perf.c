#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <omp.h>

#include "TinyDFT.h"
#include "H2ERI.h"

void TinyDFT_copy_shells_to_H2ERI(TinyDFT_t TinyDFT, H2ERI_t h2eri)
{
    h2eri->natom  = TinyDFT->natom;
    h2eri->nshell = TinyDFT->nshell;
    h2eri->shells = (shell_t *) malloc(sizeof(shell_t) * h2eri->nshell);
    assert(h2eri->shells != NULL);
    simint_initialize_shells(h2eri->nshell, h2eri->shells);
    
    shell_t *src_shells = (shell_t*) TinyDFT->simint->shells;
    shell_t *dst_shells = h2eri->shells;
    for (int i = 0; i < h2eri->nshell; i++)
    {
        simint_allocate_shell(src_shells[i].nprim, &dst_shells[i]);
        simint_copy_shell(&src_shells[i], &dst_shells[i]);
    }
}

int main(int argc, char **argv)
{
    if (argc < 4)
    {
        printf("Usage: %s <basis> <xyz> <QR_tol>\n", argv[0]);
        return 255;
    }
    
    printf("INFO: use H2ERI J (relerr %.2e)\n", atof(argv[3]));

    // Initialize TinyDFT
    TinyDFT_t TinyDFT;
    TinyDFT_init(&TinyDFT, argv[1], argv[2]);
    
    // Initialize H2P-ERI
    double st = get_wtime_sec();
    H2ERI_t h2eri;
    H2ERI_init(&h2eri, 1e-10, 1e-10, atof(argv[3]));
    TinyDFT_copy_shells_to_H2ERI(TinyDFT, h2eri);
    H2ERI_process_shells(h2eri);
    H2ERI_partition(h2eri);
    H2ERI_build_H2(h2eri);
    double et = get_wtime_sec();
    printf("H2ERI build H2 for J matrix done, used %.3lf (s)\n", et - st);
    
    // Compute constant matrices and get initial guess for D
    TinyDFT_build_Hcore_S_X_mat(TinyDFT, TinyDFT->Hcore_mat, TinyDFT->S_mat, TinyDFT->X_mat);
    TinyDFT_build_Dmat_SAD(TinyDFT, TinyDFT->D_mat);
    
    // Test H2ERI_build_Coulomb() performance after warm-up running
    H2ERI_build_Coulomb(h2eri, TinyDFT->D_mat, TinyDFT->J_mat);
    h2eri->h2pack->n_matvec = 0;
    memset(h2eri->h2pack->timers + 4, 0, sizeof(double) * 5);
    st = get_wtime_sec();
    int ntest = 10;
    for (int i = 0; i < ntest; i++)
        H2ERI_build_Coulomb(h2eri, TinyDFT->D_mat, TinyDFT->J_mat);
    et = get_wtime_sec();
    printf("H2ERI build J matrix average time = %.3lf (s)\n", (et - st) / (double) ntest);
    
    // Print H2P-ERI statistic info
    H2ERI_print_statistic(h2eri);
    
    // Free TinyDFT and H2P-ERI
    TinyDFT_destroy(&TinyDFT);
    H2ERI_destroy(h2eri);
    
    return 0;
}

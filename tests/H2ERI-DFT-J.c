#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <omp.h>

#include "TinyDFT.h"
#include "H2ERI.h"
#include "TinyDFT_copy_shells.h"

void H2ERI_KSDFT(TinyDFT_p TinyDFT, H2ERI_p h2eri, const int max_iter)
{
    // Start SCF iterations
    printf("KSDFT SCF iteration started...\n");
    printf("Nuclear repulsion energy = %.10lf\n", TinyDFT->E_nuc_rep);
    TinyDFT->iter = 0;
    TinyDFT->max_iter = max_iter;
    double E_prev, E_curr, E_delta = 19241112.0;
    
    int    mat_size       = TinyDFT->mat_size;
    int    xf_id          = TinyDFT->xf_id;
    int    xf_family      = TinyDFT->xf_family;
    double *Hcore_mat     = TinyDFT->Hcore_mat;
    double *S_mat         = TinyDFT->S_mat;
    double *X_mat         = TinyDFT->X_mat;
    double *J_mat         = TinyDFT->J_mat;
    double *K_mat         = TinyDFT->K_mat;
    double *XC_mat        = TinyDFT->XC_mat;
    double *F_mat         = TinyDFT->F_mat;
    double *Cocc_mat      = TinyDFT->Cocc_mat;
    double *D_mat         = TinyDFT->D_mat;
    double *E_nuc_rep     = &TinyDFT->E_nuc_rep;
    double *E_one_elec    = &TinyDFT->E_one_elec;
    double *E_two_elec    = &TinyDFT->E_two_elec;
    double *E_HF_exchange = &TinyDFT->E_HF_exchange;
    double *E_DFT_XC      = &TinyDFT->E_DFT_XC;

    double HF_x_coef;
    if (xf_id == HYB_GGA_XC_B3LYP || xf_id == HYB_GGA_XC_B3LYP5) HF_x_coef = 0.2;

    while ((TinyDFT->iter < TinyDFT->max_iter) && (fabs(E_delta) >= TinyDFT->E_tol))
    {
        printf("--------------- Iteration %d ---------------\n", TinyDFT->iter);
        
        double st0, et0, st1, et1, st2;
        double J_time, K_time, XC_time;
        st0 = get_wtime_sec();
        
        // Build the Fock matrix
        st1 = get_wtime_sec();
        H2ERI_build_Coulomb(h2eri, D_mat, J_mat);
        st2 = get_wtime_sec();
        J_time = st2 - st1;
        
        st1 = get_wtime_sec();
        if (xf_family == FAMILY_HYB_GGA)
            TinyDFT_build_JKmat(TinyDFT, D_mat, NULL, K_mat);
        st2 = get_wtime_sec();
        K_time = st2 - st1;
        
        st1 = get_wtime_sec();
        *E_DFT_XC = TinyDFT_build_XC_mat(TinyDFT, D_mat, XC_mat);
        st2 = get_wtime_sec();
        XC_time = st2 - st1;
        
        if (xf_family == FAMILY_HYB_GGA)
        {
            #pragma omp parallel for simd
            for (int i = 0; i < mat_size; i++)
                F_mat[i] = Hcore_mat[i] + 2 * J_mat[i] + XC_mat[i] - HF_x_coef * K_mat[i];
        } else {
            #pragma omp parallel for simd
            for (int i = 0; i < mat_size; i++)
                F_mat[i] = Hcore_mat[i] + 2 * J_mat[i] + XC_mat[i];
        }
        
        et1 = get_wtime_sec();
        printf("* Build Fock matrix     : %.3lf (s), J / K / XC = %.3lf, %.3lf, %.3lf (s)\n", et1 - st1, J_time, K_time, XC_time);
        
        // Calculate new system energy
        st1 = get_wtime_sec();
        if (xf_family == FAMILY_HYB_GGA)
        {
            TinyDFT_calc_HF_energy(
                mat_size, D_mat, Hcore_mat, J_mat, K_mat, 
                E_one_elec, E_two_elec, E_HF_exchange
            );
            E_curr = (*E_nuc_rep) + (*E_one_elec) + (*E_two_elec) + (*E_DFT_XC) + HF_x_coef * (*E_HF_exchange);
        } else {
            TinyDFT_calc_HF_energy(
                mat_size, D_mat, Hcore_mat, J_mat, NULL, 
                E_one_elec, E_two_elec, NULL
            );
            E_curr = (*E_nuc_rep) + (*E_one_elec) + (*E_two_elec) + (*E_DFT_XC);
        }
        et1 = get_wtime_sec();
        printf("* Calculate energy      : %.3lf (s)\n", et1 - st1);
        E_delta = E_curr - E_prev;
        E_prev = E_curr;
        
        // CDIIS acceleration (Pulay mixing)
        st1 = get_wtime_sec();
        TinyDFT_CDIIS(TinyDFT, X_mat, S_mat, D_mat, F_mat);
        et1 = get_wtime_sec();
        printf("* CDIIS procedure       : %.3lf (s)\n", et1 - st1);
        
        // Diagonalize and build the density matrix
        st1 = get_wtime_sec();
        TinyDFT_build_Dmat_eig(TinyDFT, F_mat, X_mat, D_mat, Cocc_mat);
        et1 = get_wtime_sec(); 
        printf("* Build density matrix  : %.3lf (s)\n", et1 - st1);
        
        et0 = get_wtime_sec();
        
        printf("* Iteration runtime     = %.3lf (s)\n", et0 - st0);
        printf("* Energy = %.10lf", E_curr);
        if (TinyDFT->iter > 0) 
        {
            printf(", delta = %e\n", E_delta); 
        } else {
            printf("\n");
            E_delta = 19241112.0;  // Prevent the SCF exit after 1st iteration when no SAD initial guess
        }
        
        TinyDFT->iter++;
        fflush(stdout);
    }
    printf("--------------- SCF iterations finished ---------------\n");
}

int main(int argc, char **argv)
{
    if (argc < 6)
    {
        printf("Usage: %s <basis> <xyz> <niter> <X-func ID> <C-func ID> <QR_tol>\n", argv[0]);
        return 255;
    }
    
    printf("INFO: use H2ERI J (relerr %.2e), DFT XC\n", atof(argv[6]));

    double st, et;
    
    // Initialize TinyDFT
    TinyDFT_p TinyDFT;
    TinyDFT_init(&TinyDFT, argv[1], argv[2]);
    
    // Initialize H2P-ERI
    st = get_wtime_sec();
    H2ERI_p h2eri;
    H2ERI_init(&h2eri, 1e-10, 1e-10, atof(argv[6]));
    TinyDFT_copy_shells_to_H2ERI(TinyDFT, h2eri);
    H2ERI_process_shells(h2eri);
    H2ERI_partition(h2eri);
    H2ERI_build_H2(h2eri, 0);
    et = get_wtime_sec();
    printf("H2ERI build H2 for J matrix done, used %.3lf (s)\n", et - st);
    
    // Compute constant matrices and get initial guess for D
    st = get_wtime_sec();
    TinyDFT_build_Hcore_S_X_mat(TinyDFT, TinyDFT->Hcore_mat, TinyDFT->S_mat, TinyDFT->X_mat);
    TinyDFT_build_Dmat_SAD(TinyDFT, TinyDFT->D_mat);
    et = get_wtime_sec();
    printf("TinyDFT compute Hcore, S, X matrices over, elapsed time = %.3lf (s)\n", et - st);
    
    // Set up XC numerical integral environments
    char xf_str[5] = "LDA_X";
    char cf_str[8] = "LDA_C_XA";
    st = get_wtime_sec();
    if (argc >= 6)
    {
        TinyDFT_setup_XC_integral(TinyDFT, argv[4], argv[5]);
    } else {
        TinyDFT_setup_XC_integral(TinyDFT, xf_str, cf_str);
    }
    et = get_wtime_sec();
    printf("TinyDFT set up XC integral over, nintp = %8d, elapsed time = %.3lf (s)\n", TinyDFT->nintp, et - st);
    
    // Do KSDFT calculation
    H2ERI_KSDFT(TinyDFT, h2eri, atoi(argv[3]));
    
    // Print H2P-ERI statistic info
    H2ERI_print_statistic(h2eri);
    
    // Free TinyDFT and H2P-ERI
    TinyDFT_destroy(&TinyDFT);
    H2ERI_destroy(h2eri);
    
    return 0;
}

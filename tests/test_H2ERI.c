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
    
    double *bt = h2eri->h2pack->timers;
    printf("%.3lf, %.3lf, %.3lf (s)\n", bt[1], bt[2], bt[3]);
    int UB, idx;
    FILE *ouf;
    H2P_dense_mat_t *U = h2eri->h2pack->U;
    H2P_int_vec_t *J_row  = h2eri->J_row;
    H2P_int_vec_t *J_pair = h2eri->J_pair;
    while (1)
    {
        printf("U(0) or B(1): ");
        scanf("%d", &UB);
        printf("idx: ");
        scanf("%d", &idx);
        if (UB == 0)
        {
            ouf = fopen("add_Ui.m", "w");
            fprintf(ouf, "Ui_c = [\n");
            for (int i = 0; i < U[idx]->nrow; i++)
            {
                double *U_row = U[idx]->data + i * U[idx]->ld;
                for (int j = 0; j < U[idx]->ncol; j++)
                    fprintf(ouf, "% .15lf ", U_row[j]);
                fprintf(ouf, "\n");
            }
            fprintf(ouf, "];\n");
            fclose(ouf);
            
            printf("J_pair: ");
            for (int i = 0; i < J_pair[idx]->length; i++) printf("%d, ", J_pair[idx]->data[i]+1);
            printf("\nJ_row: ");
            for (int i = 0; i < J_row[idx]->length; i++) printf("%d, ", J_row[idx]->data[i]+1);
            printf("\n");
        } else {
            ouf = fopen("add_Bi.m", "w");
            fprintf(ouf, "Bi_c = [\n");
            int Bi_nrow = h2eri->h2pack->B_nrow[idx];
            int Bi_ncol = h2eri->h2pack->B_ncol[idx];
            double *Bi = h2eri->h2pack->B_data + h2eri->h2pack->B_ptr[idx];
            for (int i = 0; i < Bi_nrow; i++)
            {
                double *B_row = Bi + i * Bi_ncol;
                for (int j = 0; j < Bi_ncol; j++)
                    fprintf(ouf, "% .15lf ", B_row[j]);
                fprintf(ouf, "\n");
            }
            fprintf(ouf, "];\n");
            fclose(ouf);
        }
    }
    
    simint_finalize();
    return 0;
}
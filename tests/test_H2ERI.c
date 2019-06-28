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
    
    int n_bra_pair, n_ket_pair;
    int *M_list, *N_list, *P_list, *Q_list;
    FILE *ouf;
    simint_buff_t buff;
    CMS_init_Simint_buff(3, &buff);
    H2P_dense_mat_t mat;
    H2P_dense_mat_init(&mat, 10, 10);
    while (1)
    {
        scanf("%d%d", &n_bra_pair, &n_ket_pair);
        printf("pairs = %d %d\n", n_bra_pair, n_ket_pair);
        M_list = (int *) malloc(sizeof(int) * n_bra_pair);
        N_list = (int *) malloc(sizeof(int) * n_bra_pair);
        P_list = (int *) malloc(sizeof(int) * n_ket_pair);
        Q_list = (int *) malloc(sizeof(int) * n_ket_pair);
        for (int i = 0; i < n_bra_pair; i++) scanf("%d", M_list + i);
        printf("M_list done\n");
        for (int i = 0; i < n_bra_pair; i++) scanf("%d", N_list + i);
        printf("N_list done\n");
        for (int i = 0; i < n_ket_pair; i++) scanf("%d", P_list + i);
        printf("P_list done\n");
        for (int i = 0; i < n_ket_pair; i++) scanf("%d", Q_list + i);
        printf("Q_list done\n");
        
        int nrow = CMS_sum_shell_pair_bas_func_pairs(h2eri->shells, n_bra_pair, M_list, N_list);
        int ncol = CMS_sum_shell_pair_bas_func_pairs(h2eri->shells, n_ket_pair, P_list, Q_list);
        H2P_dense_mat_resize(mat, nrow, ncol);

        CMS_calc_ERI_pairs_to_mat(
            h2eri->shells, n_bra_pair, n_ket_pair, 
            M_list, N_list, P_list, Q_list, buff, mat->data, mat->ld
        );
        
        ouf = fopen("add_c_eri_mat.m", "w");
        fprintf(ouf, "c_eri_mat = [\n");
        for (int i = 0; i < mat->nrow; i++)
        {
            double *mat_row = mat->data + mat->ld * i;
            for (int j = 0; j < mat->ncol; j++)
                fprintf(ouf, "% .15lf ", mat_row[j]);
            fprintf(ouf, "\n");
        }
        fprintf(ouf, "];\n");
        fclose(ouf);
        printf("Calc done\n");
        
        free(M_list);
        free(N_list);
        free(P_list);
        free(Q_list);
    }
    CMS_destroy_Simint_buff(buff);
    H2P_dense_mat_destroy(mat);
    
    simint_finalize();
    return 0;
}
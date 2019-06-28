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
    
    int num_sp, n_point;
    int *M_list, *N_list;
    double *xyz;
    FILE *ouf;
    H2P_dense_mat_t mat;
    H2P_dense_mat_init(&mat, 10, 10);
    while (1)
    {
        scanf("%d%d", &num_sp, &n_point);
        printf("num_sp = %d, n_point = %d\n", num_sp, n_point);
        M_list = (int *) malloc(sizeof(int) * num_sp);
        N_list = (int *) malloc(sizeof(int) * num_sp);
        xyz = (double *) malloc(sizeof(double) * n_point * 3);
        double *x = xyz, *y = xyz + n_point, *z = xyz + n_point * 2;
        for (int i = 0; i < num_sp; i++) scanf("%d", M_list + i);
        printf("M_list done\n");
        for (int i = 0; i < num_sp; i++) scanf("%d", N_list + i);
        printf("N_list done\n");
        for (int i = 0; i < n_point; i++) scanf("%lf%lf%lf", x + i, y + i, z + i);
        printf("coord done\n");
        
        int ncol = CMS_sum_shell_pair_bas_func_pairs(h2eri->shells, num_sp, M_list, N_list);
        H2P_dense_mat_resize(mat, n_point, ncol);

        CMS_calc_NAI_pairs_to_mat(
            h2eri->shells, num_sp, n_point, 
            M_list, N_list, x, y, z, mat->data, mat->ld
        );
        
        ouf = fopen("add_c_nai_mat.m", "w");
        fprintf(ouf, "c_nai_mat = [\n");
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
        free(xyz);
    }
    H2P_dense_mat_destroy(mat);
    
    simint_finalize();
    return 0;
}
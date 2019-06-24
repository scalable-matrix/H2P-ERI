#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>

#include "CMS.h"
#include "H2ERI_shell_operations.h"

int main(int argc, char **argv)
{
    int nshell;
    shell_t *shells;
    
    simint_init();
    
    CMS_read_mol_file(argv[1], &nshell, &shells);
    
    H2ERI_rotate_shells(nshell, shells);
    
    int num_unc_sp;
    shell_t *unc_sp;
    double *unc_sp_center;
    H2ERI_uncontract_shell_pairs(
        nshell, shells, 1e-10, 
        &num_unc_sp, &unc_sp, &unc_sp_center
    );
    
    double *sp_extent = (double *) malloc(sizeof(double) * num_unc_sp);
    assert(sp_extent != NULL);
    H2ERI_calc_shell_pair_extents(num_unc_sp, unc_sp, 1e-10, sp_extent);
    
    FILE *ouf = fopen("add_sp_extent.m", "w");
    fprintf(ouf, "sp_extent = [\n");
    for (int i = 0; i < num_unc_sp; i++) 
        fprintf(ouf, "%.15lf\n", sp_extent[i]);
    fprintf(ouf, "];\n");
    fclose(ouf);
    
    CMS_destroy_shells(nshell, shells);
    CMS_destroy_shells(num_unc_sp * 2, unc_sp);
    
    free(sp_extent);
    free(shells);
    free(unc_sp);
    free(unc_sp_center);
    
    simint_finalize();
    
    return 0;
}
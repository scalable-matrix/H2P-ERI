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
    
    CMS_read_mol_file(argv[1], &h2eri->nshell, &h2eri->shells);
    
    H2ERI_rotate_shells(h2eri);
    
    H2ERI_uncontract_shell_pairs(h2eri);
    
    H2ERI_calc_unc_sp_extents(h2eri);
    
    FILE *ouf = fopen("add_sp_extent.m", "w");
    fprintf(ouf, "sp_extent = [\n");
    for (int i = 0; i < h2eri->num_unc_sp; i++) 
        fprintf(ouf, "%.15lf\n", h2eri->unc_sp_extent[i]);
    fprintf(ouf, "];\n");
    fclose(ouf);
    
    simint_finalize();
    
    return 0;
}
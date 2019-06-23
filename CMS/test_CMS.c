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
    
    CMS_destroy_shells(nshell, shells);
    CMS_destroy_shells(num_unc_sp * 2, unc_sp);
    
    free(shells);
    free(unc_sp);
    free(unc_sp_center);
    
    simint_finalize();
    
    return 0;
}
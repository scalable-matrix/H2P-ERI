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
    
    //CMS_print_shells(nshell, shells);
    
    double scrval = CMS_get_Schwarz_scrval(nshell, shells);
    printf("scrval = %e\n", scrval);
    
    CMS_destroy_shells(nshell, shells);
    
    free(shells);
    
    simint_finalize();
    
    return 0;
}
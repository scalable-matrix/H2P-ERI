#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>

#include "CMS.h"

int main(int argc, char **argv)
{
    int nshell;
    shell_t *shells;
    
    CMS_read_mol_file(argv[1], &nshell, &shells);
    
    CMS_print_shells(nshell, shells);
    
    CMS_destroy_shells(nshell, shells);
    
    free(shells);
    
    return 0;
}
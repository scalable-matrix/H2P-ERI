#ifndef __CMS_H__
#define __CMS_H__

#include "simint/simint.h"

typedef struct simint_shell           shell_t;
typedef struct simint_multi_shellpair multi_sp_t;

// Read all shell information in a .mol file and normalize all these shells
// Input parameter:
//   mol_fname : .mol file path
// Output parameters:
//   *nshell : Total number of shells in the .mol file
//   *shells : All shells in the .mol file, stored in Simint shell structure
void CMS_read_mol_file(const char *mol_fname, int *nshell, shell_t **shells);

// Destroy all Simint shells
// Input parameter:
//   nshell : Total number of shells
//   shells : All shells stored in Simint shell structure
void CMS_destroy_shells(const int nshell, shell_t *shells);

// Print all shell information, for debugging
// Input parameter:
//   nshell : Total number of shells
//   shells : All shells stored in Simint shell structure
void CMS_print_shells(const int nshell, shell_t *shells);

#endif

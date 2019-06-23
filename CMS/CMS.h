#ifndef __CMS_H__
#define __CMS_H__

#include "simint/simint.h"

typedef struct simint_shell           shell_t;
typedef struct simint_multi_shellpair multi_sp_t;

#define NCART(am) ((am>=0)?((((am)+2)*((am)+1))>>1):0)
#define MAX(a, b) ((a)>(b)?(a):(b))

// Read all shell information in a .mol file and normalize all these shells
// Input parameter:
//   mol_fname : .mol file path
// Output parameters:
//   *nshell_ : Total number of shells in the .mol file
//   *shells_ : Array, all shells in the .mol file, stored in Simint shell structure
void CMS_read_mol_file(const char *mol_fname, int *nshell_, shell_t **shells_);

// Destroy all Simint shells
// Input parameters:
//   nshell : Total number of shells
//   shells : Array, all shells stored in Simint shell structure
void CMS_destroy_shells(const int nshell, shell_t *shells);

// Print all shell information, for debugging
// Input parameters:
//   nshell : Total number of shells
//   shells : Array, all shells stored in Simint shell structure
void CMS_print_shells(const int nshell, shell_t *shells);

// Get the Schwarz screening value from a given set of shells
// Input parameters:
//   nshell : Total number of shells
//   shells : Array, all shells stored in Simint shell structure
// Output parameter:
//   scr_vals : Schwarz screening value of each shell pair
//   <return> : Maximum Schwarz screening value
double CMS_get_Schwarz_scrval(const int nshell, shell_t *shells, double *scr_vals);

#endif

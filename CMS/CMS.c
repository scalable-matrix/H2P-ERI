#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>

#include "CMS.h"

// Read all shell information in a .mol file and normalize all these shells
void CMS_read_mol_file(const char *mol_fname, int *nshell_, shell_t **shells_)
{
    int AM_map[128];
    AM_map['S'] = 0;
    AM_map['P'] = 1;
    AM_map['D'] = 2;
    AM_map['F'] = 3;
    AM_map['G'] = 4;
    AM_map['H'] = 5;
    AM_map['I'] = 6;
    AM_map['J'] = 7;
    AM_map['K'] = 8;
    AM_map['L'] = 9;
    
    FILE *inf;
    
    inf = fopen(mol_fname, "r");
    if (inf == NULL)
    {
        printf("[FATAL] CMS cannot open mol file %s\n", mol_fname);
        assert(inf != NULL);
    }
    
    // 1. First pass, get the nshell_total
    int natom, nshell_total = 0;
    fscanf(inf, "%d", &natom);
    for (int i = 0; i < natom; i++)
    {
        char sym[8];
        int nshell, nprimall, nallprimg;
        double x, y, z;
        fscanf(inf, "%s %d %d %d", sym, &nshell, &nprimall, &nallprimg);
        fscanf(inf, "%lf %lf %lf", &x, &y, &z);
        for (int j = 0; j < nshell; j++)
        {
            char type[8];
            int nprim, ngen;
            fscanf(inf, "%s %d %d", type, &nprim, &ngen);
            nshell_total += ngen;
            for (int k = 0; k < nprim; k++)
            {
                double alpha;
                fscanf(inf, "%lf", &alpha);
                for (int l = 0; l < ngen; l++)
                {
                    double coef;
                    fscanf(inf, "%lf", &coef);
                }
            }
        }
    }
    fclose(inf);
    
    // 2. Second pass, create Simint shells
    shell_t *shells = (shell_t *) malloc(sizeof(shell_t) * nshell_total);
    assert(shells != NULL);
    int shell_idx = 0;
    inf = fopen(mol_fname, "r");
    fscanf(inf, "%d", &natom);
    for (int i = 0; i < natom; i++)
    {
        char sym[8], type[8];
        int nshell, nprimall, nallprimg;
        int nprim, ngen, sidx;
        double x, y, z;
        double alpha, coef;
        
        fscanf(inf, "%s %d %d %d", sym, &nshell, &nprimall, &nallprimg);
        fscanf(inf, "%lf %lf %lf", &x, &y, &z);
        
        for (int j = 0; j < nshell; j++)
        {
            fscanf(inf, "%s %d %d", type, &nprim, &ngen);
            
            for (int l = 0; l < ngen; l++)
            {
                sidx = shell_idx + l;
                simint_initialize_shell(&shells[sidx]);
                simint_allocate_shell(nprim, &shells[sidx]);
                shells[sidx].am    = AM_map[(char) type[l]];;
                shells[sidx].nprim = nprim;
                shells[sidx].x     = x;
                shells[sidx].y     = y;
                shells[sidx].z     = z;
            }
            
            for (int k = 0; k < nprim; k++)
            {
                fscanf(inf, "%lf", &alpha);
                for (int l = 0; l < ngen; l++)
                {
                    fscanf(inf, "%lf", &coef);
                    sidx = shell_idx + l;
                    shells[sidx].alpha[k] = alpha;
                    shells[sidx].coef[k]  = coef;
                }
            }
            
            shell_idx += ngen;
        }
    }
    fclose(inf);
    
    // 3. Normalize all shells
    simint_normalize_shells(nshell_total, shells);
    
    *nshell_ = nshell_total;
    *shells_ = shells;
}

// Destroy all Simint shells
void CMS_destroy_shells(const int nshell, shell_t *shells)
{
    for (int i = 0; i < nshell; i++)
        simint_free_shell(&shells[i]);
}

// Print all shell information, for debugging
void CMS_print_shells(const int nshell, shell_t *shells)
{
    printf("%d Shells:\n", nshell);
    for (int i = 0; i < nshell; i++)
    {
        printf(
            "%d, %2d, %.3lf, %.3lf, %.3lf, ", shells[i].am, 
            shells[i].nprim, shells[i].x, shells[i].y, shells[i].z
        );
        int nprim = shells[i].nprim;
        for (int j = 0; j < nprim; j++) printf("%.3lf, ", shells[i].alpha[j]);
        for (int j = 0; j < nprim; j++) printf("%.3lf, ", shells[i].coef[j]);
        printf("\n");
    }
}
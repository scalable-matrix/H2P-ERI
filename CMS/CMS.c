#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <omp.h>

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

// Get the Schwarz screening value from a given set of shells
double CMS_get_Schwarz_scrval(const int nshell, shell_t *shells)
{
    // 1. Calculate the size of each shell and prepare Simint buffer
    int *shell_bf_num = (int*) malloc(sizeof(int) * nshell);
    int max_am = 0, max_am_ncart = 0;
    assert(shell_bf_num != NULL);
    for (int i = 0; i < nshell; i++)
    {
        int am = shells[i].am;
        int am_ncart = NCART(am);
        max_am = MAX(max_am, am);
        max_am_ncart = MAX(max_am_ncart, am_ncart); 
        shell_bf_num[i] = am_ncart;
    }
    size_t work_msize = simint_ostei_workmem(0, max_am);
    size_t ERI_msize  = sizeof(double) * (max_am_ncart * max_am_ncart * max_am_ncart * max_am_ncart);
    ERI_msize += sizeof(double) * 8;
    
    // 2. Calculate (MN|MN) and find the Schwarz screening value
    double global_max_scrval = 0.0;
    #pragma omp parallel
	{    
        struct simint_multi_shellpair MN_pair;
        simint_initialize_multi_shellpair(&MN_pair);
        
        double *work_mem = SIMINT_ALLOC(work_msize);
        double *ERI_mem  = SIMINT_ALLOC(ERI_msize);
        assert(work_mem != NULL && ERI_mem != NULL);
        
		#pragma omp for schedule(dynamic) reduction(max:global_max_scrval)
		for (int M = 0; M < nshell; M++)
		{
			int dimM = shell_bf_num[M];
			for (int N = 0; N < nshell; N++)
			{
				int dimN = shell_bf_num[N];
				
                simint_create_multi_shellpair(1, &shells[M], 1, &shells[N], &MN_pair, 0);
                int ERI_size = simint_compute_eri(&MN_pair, &MN_pair, 0.0, work_mem, ERI_mem);
                if (ERI_size <= 0) continue;
				
                int ld_MNM_M = (dimM * dimN * dimM + dimM);
                int ld_NM_1  = (dimN * dimM + 1);
                double max_val = 0.0;
                for (int iM = 0; iM < dimM; iM++)
                {
                    for (int iN = 0; iN < dimN; iN++)
                    {
                        int    idx = iN * ld_MNM_M + iM * ld_NM_1;
                        double val = fabs(ERI_mem[idx]);
                        max_val = MAX(max_val, val);
                    }
                }
                global_max_scrval = MAX(global_max_scrval, max_val);
			}
		}
        
        SIMINT_FREE(ERI_mem);
        SIMINT_FREE(work_mem);
        simint_free_multi_shellpair(&MN_pair);
	}
    
    free(shell_bf_num);
    return global_max_scrval;
}
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <omp.h>

#include "CMS.h"

// NOTICE: Shell quartet (MN|PQ) ERI result needs NCART(M)*NCART(M)
// *NCART(P)*NCART(Q) + 8 doubles. +8 for statistic information. 
// TOTAL BUG COUNT FOR THIS: 3

// Read all shell information in a .mol file and normalize all these shells
void CMS_read_mol_file(const char *mol_fname, int *natom_, int *nshell_, shell_t **shells_)
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
    
    *natom_  = natom;
    *nshell_ = nshell_total;
    *shells_ = shells;
}

// Destroy all Simint shells
void CMS_destroy_shells(const int nshell, shell_t *shells)
{
    for (int i = 0; i < nshell; i++)
        simint_free_shell(&shells[i]);
}

// Destroy all Simint shell pairs 
void CMS_destroy_shell_pairs(const int num_sp, multi_sp_t *sp)
{
    for (int i = 0; i < num_sp; i++)
        simint_free_multi_shellpair(&sp[i]);
}

// Get the number of basis function pairs in a shell pair
int CMS_get_sp_nbfp(const multi_sp_p sp)
{
    return NCART(sp->am1) * NCART(sp->am2);
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
double CMS_get_Schwarz_scrval(const int nshell, shell_t *shells, double *scr_vals)
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
                
                simint_create_multi_shellpair(1, &shells[M], 1, &shells[N], &MN_pair, SIMINT_SCREEN_NONE);
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
                scr_vals[M * nshell + N] = max_val;
                scr_vals[N * nshell + M] = max_val;
            }
        }
        
        SIMINT_FREE(ERI_mem);
        SIMINT_FREE(work_mem);
        simint_free_multi_shellpair(&MN_pair);
    }
    
    free(shell_bf_num);
    return global_max_scrval;
}

// Initialize a Simint buffer structure
void CMS_init_Simint_buff(const int max_am, simint_buff_t *buff_)
{
    simint_buff_t buff = (simint_buff_t) malloc(sizeof(struct simint_buff));
    assert(buff != NULL);
    
    int max_ncart = NCART(max_am);
    int max_int = max_ncart * max_ncart * max_ncart * max_ncart;
    buff->work_msize = simint_ostei_workmem(0, max_am);
    buff->ERI_msize  = sizeof(double) * (max_int * NPAIR_SIMD + 8);
    buff->work_mem = SIMINT_ALLOC(buff->work_msize);
    buff->ERI_mem  = SIMINT_ALLOC(buff->ERI_msize);
    assert(buff->work_mem != NULL && buff->ERI_mem != NULL);
    
    simint_initialize_shell(&buff->NAI_shell1);
    simint_initialize_shell(&buff->NAI_shell2);
    simint_initialize_multi_shellpair(&buff->bra_pair);
    simint_initialize_multi_shellpair(&buff->ket_pair);
    
    *buff_ = buff;
}

// Destroy a Simint buffer structure
void CMS_destroy_Simint_buff(simint_buff_t buff)
{
    if (buff == NULL) return;
    buff->work_msize = 0;
    buff->ERI_msize  = 0;
    SIMINT_FREE(buff->work_mem);
    SIMINT_FREE(buff->ERI_mem);
    simint_free_shell(&buff->NAI_shell1);
    simint_free_shell(&buff->NAI_shell2);
    simint_free_multi_shellpair(&buff->bra_pair);
    simint_free_multi_shellpair(&buff->ket_pair);
}

// Initialize an ERI batch buffer structure
void CMS_init_eri_batch_buff(const int max_am, const int num_param, eri_batch_buff_t *buff_)
{
    eri_batch_buff_t buff = (eri_batch_buff_t) malloc(sizeof(struct eri_batch_buff));
    assert(buff != NULL);
    
    int num_batch = (max_am + 1) * (max_am + 1);
    buff->max_am    = max_am;
    buff->num_batch = num_batch;
    buff->num_param = num_param;

    int total_ket_pairs = num_batch * NPAIR_SIMD;
    buff->batch_cnt = (int*) malloc(sizeof(int) * num_batch);
    buff->sq_param  = (int*) malloc(sizeof(int) * total_ket_pairs * num_param);
    buff->ket_pairs = (multi_sp_p*) malloc(sizeof(multi_sp_p) * total_ket_pairs);
    assert(buff->batch_cnt != NULL);
    assert(buff->sq_param  != NULL);
    assert(buff->ket_pairs != NULL);
    memset(buff->batch_cnt, 0, sizeof(int) * num_batch);
    simint_initialize_multi_shellpair(&buff->ket_multipairs);
    
    *buff_ = buff;
}

// Destroy an ERI batch buffer structure
void CMS_destroy_eri_batch_buff(eri_batch_buff_t buff)
{
    if (buff == NULL) return;
    buff->max_am    = -1;
    buff->num_batch = 0;
    buff->num_param = 0;
    free(buff->batch_cnt);
    free(buff->sq_param);
    free(buff->ket_pairs);
    simint_free_multi_shellpair(&buff->ket_multipairs);
}

// Push a ket pair into an ERI batch
int CMS_push_ket_pair_to_eri_batch(
    eri_batch_buff_t buff, const int ket_am1, const int ket_am2, 
    const multi_sp_p ket_pair, const int *param
)
{
    int batch_id = ket_am1 * (buff->max_am + 1) + ket_am2;
    int batch_idx = buff->batch_cnt[batch_id];
    if (batch_idx >= NPAIR_SIMD) 
    {
        return 0;
    } else {
        int sq_offset = batch_id * NPAIR_SIMD + batch_idx;
        buff->ket_pairs[sq_offset] = ket_pair;
        int *sq_param_p = buff->sq_param + sq_offset * buff->num_param;
        memcpy(sq_param_p, param, sizeof(int) * buff->num_param);
        buff->batch_cnt[batch_id]++;
        return (batch_idx+1);
    }
}

// Calculate all shell quartets in an ERI batch 
void CMS_calc_ERI_batch(
    eri_batch_buff_t eri_batch_buff, simint_buff_t simint_buff, 
    const int ket_am1, const int ket_am2, int *eri_size, int **batch_param
)
{
    int batch_id = ket_am1 * (eri_batch_buff->max_am + 1) + ket_am2;
    int n_pair   = eri_batch_buff->batch_cnt[batch_id];
    multi_sp_p bra_pair         = eri_batch_buff->bra_pair;
    multi_sp_p ket_multipairs   = &eri_batch_buff->ket_multipairs;
    multi_sp_p *batch_ket_pairs = eri_batch_buff->ket_pairs + batch_id * NPAIR_SIMD;
    
    int bra_am1  = bra_pair->am1;
    int bra_am2  = bra_pair->am2;
    *eri_size    = NCART(bra_am1) * NCART(bra_am2) * NCART(ket_am1) * NCART(ket_am2);
    *batch_param = eri_batch_buff->sq_param + batch_id * NPAIR_SIMD * eri_batch_buff->num_param;
    
    ket_multipairs->nprim = 0;
    simint_cat_shellpairs(
        n_pair, (const struct simint_multi_shellpair **) batch_ket_pairs, 
        ket_multipairs, SIMINT_SCREEN_NONE
    );
    
    double prim_scrval = 0.0;
    int ret = simint_compute_eri(
        bra_pair, ket_multipairs, prim_scrval, 
        simint_buff->work_mem, simint_buff->ERI_mem
    );
    if (ret == 0)
    {
        *eri_size = 0;
        *batch_param = NULL;
    } else {
        eri_batch_buff->batch_cnt[batch_id] = 0;
    }
}

void H2ERI_copy_ERI_to_mat(
    const int num_sq, double *ERI_mem, int *batch_param,
    double *mat, const int ldm
)
{
    int ncart_MN = batch_param[0];
    int ncart_PQ = batch_param[1];
    int eri_size = ncart_MN * ncart_PQ;
    for (int i = 0; i < num_sq; i++)
    {
        int row_idx  = batch_param[4 * i + 2];
        int col_idx  = batch_param[4 * i + 3];
        double *mat_blk = mat + row_idx * ldm + col_idx;
        double *ERI_blk = ERI_mem + i * eri_size;
        for (int j = 0; j < ncart_MN; j++)
        {
            double *mat_blk_row = mat_blk + j * ldm;
            double *ERI_blk_row = ERI_blk + j * ncart_PQ;
            memcpy(mat_blk_row, ERI_blk_row, sizeof(double) * ncart_PQ);
        }
    }
}

// Calculate shell quartet pairs (N_i M_i|Q_j P_j) and unfold all ERI 
// results to form a matrix
void H2ERI_calc_ERI_pairs_to_mat(
    const multi_sp_p sp, const int n_bra_pair, const int n_ket_pair,
    const int *bra_idx, const int *ket_idx, simint_buff_t simint_buff, 
    double *mat, const int ldm, eri_batch_buff_t eri_batch_buff
)
{
    int param[4];
    int row_idx = 0;
    for (int i = 0; i < n_bra_pair; i++)
    {
        const multi_sp_p bra_pair = sp + bra_idx[i];
        int am_M = bra_pair->am1;
        int am_N = bra_pair->am2;
        int ncart_MN = NCART(am_M) * NCART(am_N);
        eri_batch_buff->bra_pair = bra_pair;
        memset(eri_batch_buff->batch_cnt, 0, sizeof(int) * eri_batch_buff->num_batch);
        param[0] = ncart_MN;
        param[2] = row_idx;
        
        int col_idx = 0;
        for (int j = 0; j < n_ket_pair; j++)
        {
            const multi_sp_p ket_pair = sp + ket_idx[j];
            int am_P = ket_pair->am1;
            int am_Q = ket_pair->am2;
            int ncart_PQ = NCART(am_P) * NCART(am_Q);
            param[1] = ncart_PQ;
            param[3] = col_idx;
            
            int num_sq = CMS_push_ket_pair_to_eri_batch(
                eri_batch_buff, am_P, am_Q, 
                ket_pair, &param[0]
            );
            if (num_sq == NPAIR_SIMD)
            {
                int eri_size, *batch_param;
                CMS_calc_ERI_batch(
                    eri_batch_buff, simint_buff, 
                    am_P, am_Q, &eri_size, &batch_param
                );
                assert(eri_size > 0);
                H2ERI_copy_ERI_to_mat(num_sq, simint_buff->ERI_mem, batch_param, mat, ldm);
            }

            col_idx += ncart_PQ;
        }
        
        for (int ibatch = 0; ibatch < eri_batch_buff->num_batch; ibatch++)
        {
            int num_sq = eri_batch_buff->batch_cnt[ibatch];
            if (num_sq == 0) continue;
            int am_P = ibatch / (eri_batch_buff->max_am + 1);
            int am_Q = ibatch % (eri_batch_buff->max_am + 1);
            int eri_size, *batch_param;
            CMS_calc_ERI_batch(
                eri_batch_buff, simint_buff, 
                am_P, am_Q, &eri_size, &batch_param
            );
            assert(eri_size > 0);
            H2ERI_copy_ERI_to_mat(num_sq, simint_buff->ERI_mem, batch_param, mat, ldm);
        }
        
        row_idx += ncart_MN;
    }
}

// Calculate NAI pairs (N_i M_i|[x_j, y_j, z_j]) and unfold all NAI 
// results to form a matrix
void H2ERI_calc_NAI_pairs_to_mat(
    const shell_t *sp_shells, const int num_sp, const int n_bra_pair, 
    const int *sp_idx, const int n_point, double *x, double *y, double *z, 
    double *mat, const int ldm, double *trans_buf
)
{
    double atomic_nums = 1.0;
    int row_idx = 0;
    for (int j = 0; j < n_bra_pair; j++)
    {
        const shell_t *M_shell = sp_shells + sp_idx[j];
        const shell_t *N_shell = sp_shells + sp_idx[j] + num_sp;
        int am_M = M_shell->am;
        int am_N = N_shell->am;
        int ncart_MN = NCART(am_M) * NCART(am_N);
        for (int i = 0; i < n_point; i++)
        {
            int ret = simint_compute_potential(
                1, &atomic_nums, x + i, y + i, z + i,
                N_shell, M_shell, trans_buf
            );
            double *mat_blk = mat + row_idx * ldm + i;
            for (int k = 0; k < ncart_MN; k++)
                mat_blk[k * ldm] = trans_buf[k];
        }
        row_idx += ncart_MN;
    }
}

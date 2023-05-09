#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <omp.h>

#include "CMS.h"
#include "H2ERI_typedef.h"
#include "H2ERI_build_exchange.h"
#include "H2ERI_aux_structs.h"

// Initialize a H2ERI structure
void H2ERI_init(H2ERI_p *h2eri_, const double scr_tol, const double ext_tol, const double QR_tol)
{
    H2ERI_p h2eri = (H2ERI_p) malloc(sizeof(struct H2ERI));
    assert(h2eri != NULL);
    memset(h2eri, 0, sizeof(struct H2ERI));
    
    h2eri->max_am  = 0;
    h2eri->scr_tol = scr_tol;
    h2eri->ext_tol = ext_tol;
    
    h2eri->pp_npts_layer = 384;
    h2eri->pp_nlayer_ext = 3;
    
    h2eri->shell_bf_sidx         = NULL;
    h2eri->sp_nbfp               = NULL;
    h2eri->sp_bfp_sidx           = NULL;
    h2eri->sp_shell_idx          = NULL;
    h2eri->index_seq             = NULL;
    h2eri->node_adm_pairs        = NULL;
    h2eri->node_adm_pairs_sidx   = NULL;
    h2eri->node_inadm_pairs      = NULL;
    h2eri->node_inadm_pairs_sidx = NULL;
    h2eri->plist                 = NULL;
    h2eri->plist_idx             = NULL;
    h2eri->plist_sidx            = NULL;
    h2eri->dlist                 = NULL;
    h2eri->dlist_sidx            = NULL;
    h2eri->thread_Kmat_workbuf   = NULL;
    h2eri->sp_center             = NULL;
    h2eri->sp_extent             = NULL;
    h2eri->box_extent            = NULL;
    h2eri->unc_denmat_x          = NULL;
    h2eri->H2_matvec_y           = NULL;
    h2eri->shells                = NULL;
    h2eri->sp_shells             = NULL;
    h2eri->sp                    = NULL;
    h2eri->J_pair                = NULL;
    h2eri->J_row                 = NULL;
    h2eri->ovlp_ff_idx           = NULL;
    h2eri->simint_buffs          = NULL;
    h2eri->eri_batch_buffs       = NULL;
    h2eri->c_B_blks              = NULL;
    h2eri->c_D_blks              = NULL;
    
    h2eri->n_thread = omp_get_max_threads();
    h2eri->pt_dim    = 3;
    h2eri->xpt_dim   = 3;
    h2eri->krnl_dim  = 1;
    h2eri->max_child = 1 << h2eri->pt_dim;
    h2eri->max_neighbor = 1;
    for (int i = 0; i < h2eri->pt_dim; i++) h2eri->max_neighbor *= 3;
    h2eri->QR_stop_type = QR_REL_NRM;
    memcpy(&h2eri->QR_stop_tol, &QR_tol, sizeof(DTYPE));

    memset(h2eri->mat_size,  0, sizeof(size_t) * 11);
    memset(h2eri->timers,    0, sizeof(double) * 11);
    H2E_int_vec_init(&h2eri->B_blk,  h2eri->n_thread * BD_NTASK_THREAD + 5);
    H2E_int_vec_init(&h2eri->D_blk0, h2eri->n_thread * BD_NTASK_THREAD + 5);
    H2E_int_vec_init(&h2eri->D_blk1, h2eri->n_thread * BD_NTASK_THREAD + 5);
    GET_ENV_INT_VAR(h2eri->print_timers,  "H2E_PRINT_TIMERS",  "print_timers",  0, 0, 1);
    GET_ENV_INT_VAR(h2eri->print_dbginfo, "H2E_PRINT_DBGINFO", "print_dbginfo", 0, 0, 1);
    if (h2eri->print_timers  == 1) INFO_PRINTF("H2ERI will print internal timers for performance analysis\n");
    if (h2eri->print_dbginfo == 1) INFO_PRINTF("H2ERI will print debug information\n");
    
    *h2eri_ = h2eri;
}

// Destroy a H2ERI structure
void H2ERI_destroy(H2ERI_p h2eri)
{
    free(h2eri->shell_bf_sidx);
    free(h2eri->sp_nbfp);
    free(h2eri->sp_bfp_sidx);
    free(h2eri->sp_shell_idx);
    free(h2eri->index_seq);
    free(h2eri->node_adm_pairs);
    free(h2eri->node_adm_pairs_sidx);
    free(h2eri->node_inadm_pairs);
    free(h2eri->node_inadm_pairs_sidx);
    free(h2eri->plist);
    free(h2eri->plist_idx);
    free(h2eri->plist_sidx);
    free(h2eri->dlist);
    free(h2eri->dlist_sidx);
    H2ERI_exchange_workbuf_free(h2eri);
    free(h2eri->sp_center);
    free(h2eri->sp_extent);
    free(h2eri->box_extent);
    free(h2eri->unc_denmat_x);
    free(h2eri->H2_matvec_y);
    CMS_destroy_shells(h2eri->nshell, h2eri->shells);
    CMS_destroy_shells(h2eri->num_sp * 2, h2eri->sp_shells);
    CMS_destroy_shell_pairs(h2eri->num_sp, h2eri->sp);
    free(h2eri->shells);
    free(h2eri->sp_shells);
    free(h2eri->sp);
    
    for (int i = 0; i < h2eri->n_node; i++)
    {
        H2E_int_vec_destroy(&h2eri->J_pair[i]);
        H2E_int_vec_destroy(&h2eri->J_row[i]);
        H2E_int_vec_destroy(&h2eri->ovlp_ff_idx[i]);
    }
    free(h2eri->J_pair);
    free(h2eri->J_row);
    free(h2eri->ovlp_ff_idx);
    
    for (int i = 0; i < h2eri->n_thread; i++)
    {
        CMS_destroy_Simint_buff(h2eri->simint_buffs[i]);
        CMS_destroy_eri_batch_buff(h2eri->eri_batch_buffs[i]);
        H2E_thread_buf_destroy(&h2eri->thread_buffs[i]);
    }
    free(h2eri->simint_buffs);
    free(h2eri->eri_batch_buffs);
    free(h2eri->thread_buffs);
    
    if (h2eri->c_B_blks != NULL)
    {
        H2E_dense_mat_p *c_B_blks = h2eri->c_B_blks;
        for (int i = 0; i < h2eri->n_B; i++)
            H2E_dense_mat_destroy(&c_B_blks[i]);
        free(c_B_blks);
    }
    
    if (h2eri->c_D_blks != NULL)
    {
        H2E_dense_mat_p *c_D_blks = h2eri->c_D_blks;
        for (int i = 0; i < h2eri->n_D; i++)
            H2E_dense_mat_destroy(&c_D_blks[i]);
        free(c_D_blks);
    }
}

// Print H2ERI statistic information
void H2ERI_print_statistic(H2ERI_p h2eri)
{
    if (h2eri == NULL) return;
    if (h2eri->n_node == 0)
    {
        printf("H2ERI has nothing to report yet.\n");
        return;
    }

    printf("================ H2ERI molecular system info ================\n");
    printf("  * Number of atoms / shells / basis functions : %d, %d, %d\n", h2eri->natom, h2eri->nshell, h2eri->num_bf);
    printf("  * Number of symm-unique screened shell pairs : %d\n", h2eri->num_sp);
    printf("==================== H2ERI H2 tree info ====================\n");
    printf("  * Number of points               : %d\n", h2eri->n_point);
    printf("  * Kernel matrix size             : %d\n", h2eri->krnl_mat_size);
    printf("  * Maximum points in a leaf node  : %d\n", h2eri->max_leaf_points);
    printf("  * Maximum leaf node box size     : %e\n", h2eri->max_leaf_size);
    printf("  * Number of levels (root at 0)   : %d\n", h2eri->max_level+1);
    printf("  * Number of nodes                : %d\n", h2eri->n_node);
    printf("  * Number of nodes on each level  : ");
    for (int i = 0; i < h2eri->max_level; i++) 
        printf("%d, ", h2eri->level_n_node[i]);
    printf("%d\n", h2eri->level_n_node[h2eri->max_level]);
    printf("  * Number of nodes on each height : ");
    for (int i = 0; i < h2eri->max_level; i++) 
        printf("%d, ", h2eri->height_n_node[i]);
    printf("%d\n", h2eri->height_n_node[h2eri->max_level]);
    printf("  * Minimum admissible pair level  : %d\n", h2eri->min_adm_level);
    printf("  * Number of reduced adm. pairs   : %d\n", h2eri->n_r_adm_pair);
    printf("  * Number of reduced inadm. pairs : %d\n", h2eri->n_r_inadm_pair);
    
    if (h2eri->U == NULL) 
    {
        printf("H2ERI H2 matrix has not been constructed yet.\n");
        return;
    }
    
    printf("==================== H2ERI storage info ====================\n");
    size_t *mat_size = h2eri->mat_size;
    double DTYPE_MB = (double) sizeof(DTYPE) / 1048576.0;
    double int_MB   = (double) sizeof(int)   / 1048576.0;
    double U_MB   = (double) mat_size[U_SIZE_IDX]      * DTYPE_MB;
    double B_MB   = (double) mat_size[B_SIZE_IDX]      * DTYPE_MB;
    double D_MB   = (double) mat_size[D_SIZE_IDX]      * DTYPE_MB;
    double fwd_MB = (double) mat_size[MV_FWD_SIZE_IDX] * DTYPE_MB;
    double mid_MB = (double) mat_size[MV_MID_SIZE_IDX] * DTYPE_MB;
    double bwd_MB = (double) mat_size[MV_BWD_SIZE_IDX] * DTYPE_MB;
    double den_MB = (double) mat_size[MV_DEN_SIZE_IDX] * DTYPE_MB;
    double vop_MB = (double) mat_size[MV_VOP_SIZE_IDX] * DTYPE_MB;
    double mv_MB  = fwd_MB + mid_MB + bwd_MB + den_MB;
    double UBD_k  = 0.0;
    UBD_k += (double) mat_size[U_SIZE_IDX];
    UBD_k += (double) mat_size[B_SIZE_IDX];
    UBD_k += (double) mat_size[D_SIZE_IDX];
    UBD_k /= (double) h2eri->krnl_mat_size;
    double matvec_MB = 0.0;
    for (int i = 0; i < h2eri->n_thread; i++)
    {
        H2E_thread_buf_p tbi = h2eri->thread_buffs[i];
        double msize0 = (double) tbi->mat0->size     + (double) tbi->mat1->size;
        double msize1 = (double) tbi->idx0->capacity + (double) tbi->idx1->capacity;
        matvec_MB += DTYPE_MB * msize0 + int_MB * msize1;
        matvec_MB += DTYPE_MB * (double) h2eri->krnl_mat_size;
    }
    if (h2eri->y0 != NULL && h2eri->y1 != NULL)
    {
        for (int i = 0; i < h2eri->n_node; i++)
        {
            H2E_dense_mat_p y0i = h2eri->y0[i];
            H2E_dense_mat_p y1i = h2eri->y1[i];
            matvec_MB += DTYPE_MB * (y0i->size + y1i->size);
        }
    }
    printf("  * Just-In-Time B & D build      : %s\n", h2eri->BD_JIT ? "Yes (B & D not allocated)" : "No");
    printf("  * H2 representation U, B, D     : %.2lf, %.2lf, %.2lf (MB) \n", U_MB, B_MB, D_MB);
    printf("  * Matvec auxiliary arrays       : %.2lf (MB) \n", matvec_MB);
    int max_node_rank = 0;
    double sum_node_rank = 0.0, non_empty_node = 0.0;
    for (int i = 0; i < h2eri->n_UJ; i++)
    {
        int rank_i = h2eri->U[i]->ncol;
        if (rank_i > 0)
        {
            sum_node_rank  += (double) rank_i;
            non_empty_node += 1.0;
            max_node_rank   = (rank_i > max_node_rank) ? rank_i : max_node_rank;
        }
    }
    printf("  * Max / Avg compressed rank     : %d, %.0lf \n", max_node_rank, sum_node_rank / non_empty_node);
    
    printf("==================== H2ERI timing info =====================\n");
    double *timers = h2eri->timers;
    double build_t = 0.0, matvec_t = 0.0;
    double d_n_matvec = (double) h2eri->n_matvec;
    build_t += timers[PT_TIMER_IDX];
    build_t += timers[U_BUILD_TIMER_IDX];
    build_t += timers[B_BUILD_TIMER_IDX];
    build_t += timers[D_BUILD_TIMER_IDX];
    printf("  * H2 construction time (sec)   = %.3lf \n", build_t);
    printf("      |----> Point partition     = %.3lf \n", timers[PT_TIMER_IDX]);
    printf("      |----> U construction      = %.3lf \n", timers[U_BUILD_TIMER_IDX]);
    printf("      |----> B construction      = %.3lf \n", timers[B_BUILD_TIMER_IDX]);
    printf("      |----> D construction      = %.3lf \n", timers[D_BUILD_TIMER_IDX]);

    if (h2eri->n_matvec == 0)
    {
        printf("H2ERI does not have matvec timings results yet.\n");
    } else {
        double fwd_t = timers[MV_FWD_TIMER_IDX] / d_n_matvec;
        double mid_t = timers[MV_MID_TIMER_IDX] / d_n_matvec;
        double bwd_t = timers[MV_BWD_TIMER_IDX] / d_n_matvec;
        double den_t = timers[MV_DEN_TIMER_IDX] / d_n_matvec;
        double vop_t = timers[MV_VOP_TIMER_IDX] / d_n_matvec;
        matvec_t = fwd_t + mid_t + bwd_t + den_t + vop_t;
        printf(
            "  * H2 matvec average time (sec) = %.3lf, %.2lf GB/s\n", 
            matvec_t, mv_MB / matvec_t / 1024.0
        );
        printf(
            "      |----> Forward transformation      = %.3lf, %.2lf GB/s\n", 
            fwd_t, fwd_MB / fwd_t / 1024.0
        );
        if (h2eri->BD_JIT == 0)
        {
            printf(
                "      |----> Intermediate multiplication = %.3lf, %.2lf GB/s\n", 
                mid_t, mid_MB / mid_t / 1024.0
            );
        } else {
            printf("      |----> Intermediate multiplication = %.3lf\n", mid_t);
        }
        printf(
            "      |----> Backward transformation     = %.3lf, %.2lf GB/s\n", 
            bwd_t, bwd_MB / bwd_t / 1024.0
        );
        if (h2eri->BD_JIT == 0)
        {
            printf(
                "      |----> Dense multiplication        = %.3lf, %.2lf GB/s\n", 
                den_t, den_MB / den_t / 1024.0
            );
        } else {
            printf("      |----> Dense multiplication        = %.3lf GFLOPS\n", den_t);
        }
        vop_MB /= d_n_matvec;
        printf(
            "      |----> OpenMP vector operations    = %.3lf, %.2lf GB/s\n", 
            vop_t, vop_MB / vop_t / 1024.0
        );
    }
    
    printf("=============================================================\n");
}

// Reset timing statistical info of an H2ERI structure
void H2E_reset_timers(H2ERI_p h2eri)
{
    h2eri->n_matvec = 0;
    h2eri->mat_size[MV_VOP_SIZE_IDX] = 0;
    h2eri->timers[MV_FWD_TIMER_IDX]  = 0.0;
    h2eri->timers[MV_MID_TIMER_IDX]  = 0.0;
    h2eri->timers[MV_BWD_TIMER_IDX]  = 0.0;
    h2eri->timers[MV_DEN_TIMER_IDX]  = 0.0;
    h2eri->timers[MV_VOP_TIMER_IDX]  = 0.0;
}

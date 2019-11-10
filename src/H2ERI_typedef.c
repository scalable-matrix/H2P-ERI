#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>

#include "CMS.h"
#include "H2Pack_utils.h"
#include "H2Pack_typedef.h"
#include "H2ERI_typedef.h"

// Initialize a H2ERI structure
void H2ERI_init(H2ERI_t *h2eri_, const double scr_tol, const double ext_tol, const double QR_tol)
{
    H2ERI_t h2eri = (H2ERI_t) malloc(sizeof(struct H2ERI));
    assert(h2eri != NULL);
    
    h2eri->max_am  = 0;
    h2eri->scr_tol = scr_tol;
    h2eri->ext_tol = ext_tol;
    
    h2eri->pp_npts_layer = 384;
    h2eri->pp_nlayer_ext = 3;
    
    h2eri->shell_bf_sidx    = NULL;
    h2eri->unc_sp_nbfp      = NULL;
    h2eri->unc_sp_bfp_sidx  = NULL;
    h2eri->unc_sp_shell_idx = NULL;
    h2eri->index_seq        = NULL;
    h2eri->unc_sp_center    = NULL;
    h2eri->unc_sp_extent    = NULL;
    h2eri->box_extent       = NULL;
    h2eri->unc_denmat_x     = NULL;
    h2eri->H2_matvec_y      = NULL;
    h2eri->shells           = NULL;
    h2eri->unc_sp_shells    = NULL;
    h2eri->unc_sp           = NULL;
    h2eri->J_pair           = NULL;
    h2eri->J_row            = NULL;
    h2eri->ovlp_ff_idx      = NULL;
    h2eri->simint_buffs     = NULL;
    
    double _QR_tol = QR_tol;
    H2P_init(&h2eri->h2pack, 3, 1, QR_REL_NRM, &_QR_tol);
    
    *h2eri_ = h2eri;
}

// Destroy a H2ERI structure
void H2ERI_destroy(H2ERI_t h2eri)
{
    free(h2eri->shell_bf_sidx);
    free(h2eri->unc_sp_nbfp);
    free(h2eri->unc_sp_bfp_sidx);
    free(h2eri->unc_sp_shell_idx);
    free(h2eri->index_seq);
    free(h2eri->unc_sp_center);
    free(h2eri->unc_sp_extent);
    free(h2eri->box_extent);
    free(h2eri->unc_denmat_x);
    free(h2eri->H2_matvec_y);
    CMS_destroy_shells(h2eri->nshell, h2eri->shells);
    CMS_destroy_shells(h2eri->num_unc_sp * 2, h2eri->unc_sp_shells);
    CMS_destroy_shell_pairs(h2eri->num_unc_sp, h2eri->unc_sp);
    free(h2eri->shells);
    free(h2eri->unc_sp_shells);
    free(h2eri->unc_sp);
    
    for (int i = 0; i < h2eri->h2pack->n_node; i++)
    {
        H2P_int_vec_destroy(h2eri->J_pair[i]);
        H2P_int_vec_destroy(h2eri->J_row[i]);
        H2P_int_vec_destroy(h2eri->ovlp_ff_idx[i]);
    }
    free(h2eri->J_pair);
    free(h2eri->J_row);
    free(h2eri->ovlp_ff_idx);
    
    for (int i = 0; i < h2eri->h2pack->n_thread; i++)
        CMS_destroy_Simint_buff(h2eri->simint_buffs[i]);
    free(h2eri->simint_buffs);
    
    H2P_destroy(h2eri->h2pack);
}

// Print H2ERI statistic information
void H2ERI_print_statistic(H2ERI_t h2eri)
{
    printf("================ H2ERI molecular system info ================\n");
    printf("  * Number of atoms                     : %d\n", h2eri->natom);
    printf("  * Number of shells                    : %d\n", h2eri->nshell);
    printf("  * Number of basis functions           : %d\n", h2eri->num_bf);
    printf("  * Number of symm-unique screened FUSP : %d\n", h2eri->num_unc_sp);
    printf("  * Number of FUSP basis function pairs : %d\n", h2eri->num_unc_sp_bfp);
    
    H2P_print_statistic(h2eri->h2pack);
}
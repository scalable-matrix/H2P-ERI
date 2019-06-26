#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>

#include "H2Pack_utils.h"
#include "H2Pack_typedef.h"
#include "H2ERI_typedef.h"

// Initialize a H2ERI structure
void H2ERI_init(H2ERI_t *h2eri_, const double scr_tol, const double ext_tol, const double QR_tol)
{
    H2ERI_t h2eri = (H2ERI_t) malloc(sizeof(struct H2ERI));
    assert(h2eri != NULL);
    
    h2eri->scr_tol = scr_tol;
    h2eri->ext_tol = ext_tol;
    
    h2eri->shell_bf_sidx  = NULL;
    h2eri->unc_sp_bf_sidx = NULL;
    h2eri->unc_sp_center  = NULL;
    h2eri->unc_sp_extent  = NULL;
    h2eri->box_extent     = NULL;
    h2eri->shells         = NULL;
    h2eri->unc_sp         = NULL;
    h2eri->J_pair         = NULL;
    h2eri->J_row          = NULL;
    h2eri->ovlp_ff_idx    = NULL;
    
    double _QR_tol = QR_tol;
    H2P_init(&h2eri->h2pack, 3, QR_REL_NRM, &_QR_tol);
    
    *h2eri_ = h2eri;
}

// Destroy a H2ERI structure
void H2ERI_destroy(H2ERI_t h2eri)
{
    free(h2eri->shell_bf_sidx);
    free(h2eri->unc_sp_bf_sidx);
    free(h2eri->unc_sp_center);
    free(h2eri->unc_sp_extent);
    free(h2eri->box_extent);
    free(h2eri->shells);
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
    
    H2P_destroy(h2eri->h2pack);
}


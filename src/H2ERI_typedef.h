#ifndef __H2ERI_TYPEDEF_H__
#define __H2ERI_TYPEDEF_H__

// Shell operations used in H2P-ERI

#include "CMS.h"
#include "H2ERI_config.h"
#include "H2ERI_aux_structs.h"

#ifdef __cplusplus
extern "C" {
#endif

struct H2ERI
{
    int    natom;                       // Number of atoms (from input file)
    int    nshell;                      // Number of contracted shells (from input file)
    int    max_am;                      // Maximum angular momentum in the system
    int    max_shell_nbf;               // Maximum basis functions in a shell, == NCART(max_am)
    int    num_bf;                      // Number of basis functions in the system, == shell_bf_sidx[nshell]
    int    num_sp;                      // Number of screened shell pairs (SSP)
    int    num_sp_bfp;                  // Number of SSP basis function pairs, == sp_bfp_sidx[num_sp]
    int    pp_npts_layer;               // Number of proxy points on each layer
    int    pp_nlayer_ext;               // Number of proxy point layers on each extent
    int    *shell_bf_sidx;              // Array, size nshell, indices of each shell's first basis function
    int    *sp_nbfp;                    // Array, size num_sp, number of basis function pairs of each SSP
    int    *sp_bfp_sidx;                // Array, size num_sp+1, indices of each SSP's first basis function pair
    int    *sp_shell_idx;               // Array, size 2 * num_sp, each row is the contracted shell indices of a SSP
    int    *index_seq;                  // Array, size num_sp, [0, num_sp-1]
    int    *node_adm_pairs;             // Array, size 2 * h2eri->n_r_adm_pair, each node's admissible node pairs
    int    *node_adm_pairs_sidx;        // Array, size h2eri->n_node+1, index of each node's first admissible node pair
    int    *node_inadm_pairs;           // Array, size 2 * h2eri->n_r_inadm_pair, each node's inadmissible node pairs
    int    *node_inadm_pairs_sidx;      // Array, size h2eri->n_node+1, index of each node's first inadmissible node pair
    int    *plist;                      // Array, size <= 2*num_sp, each shell's screened pair shells
    int    *plist_idx;                  // Array, size <= 2*num_sp, corresponding indices of each shell's screened pair shells in sp_bfp_sidx
    int    *plist_sidx;                 // Array, size nshell+1, index of each node's first item in plist & plist_idx
    int    *dlist;                      // Array, size unknown, each shell's pair shells s.t. the corresponding density matrix block is large enough
    int    *dlist_sidx;                 // Array, size nshell+1, index of each node's first item in dlist
    void   **thread_Kmat_workbuf;       // Array, size h2eri->n_thread, pointers to each thread's K mat build work buffer
    double scr_tol;                     // Tolerance of Schwarz screening
    double ext_tol;                     // Tolerance of shell pair extent
    double *sp_center;                  // Array, size 3 * num_sp, each column is a SSP's center
    double *sp_extent;                  // Array, size num_sp, extents of SSP
    double *box_extent;                 // Array, size h2eri->n_node, extents of each H2 node box
    double *unc_denmat_x;               // Array, size num_sp_bfp, uncontracted density matrix as a vector
    double *H2_matvec_y;                // Array, size num_sp_bfp, H2 matvec result 
    shell_t          *shells;           // Array, size nshell, contracted shells
    shell_t          *sp_shells;        // Array, size 2 * num_sp, each column is a SSP
    multi_sp_t       *sp;               // Array, size num_sp, SSP
    H2E_int_vec_p    *J_pair;           // Array, size h2eri->n_node, skeleton shell pair indices of each node
    H2E_int_vec_p    *J_row;            // Array, size h2eri->n_node, skeleton row indices in each node's shell pairs
    H2E_int_vec_p    *ovlp_ff_idx;      // Array, size h2eri->n_node, i-th vector contains the far field points whose extents are overlapped with the near field of i-th node
    simint_buff_p    *simint_buffs;     // Array, size h2eri->n_thread, simint_buff structures for each thread
    eri_batch_buff_p *eri_batch_buffs;  // Array, size h2eri->n_thread, eri_batch_buff structures for each thread
    H2E_dense_mat_p  *c_B_blks;         // Array, size h2eri->n_B, compressed B blocks
    H2E_dense_mat_p  *c_D_blks;         // Array, size h2eri->n_D, compressed D blocks
    H2E_thread_buf_p *thread_buffs;     // Array, size n_thread, thread buffers for each thread

    // Variables originally from H2Pack
    int    n_thread;                // Number of threads
    int    pt_dim;                  // Dimension of point coordinate
    int    xpt_dim;                 // Dimension of extended point coordinate (for RPY)
    int    krnl_dim;                // Dimension of tensor kernel's return
    int    QR_stop_type;            // Partial QR stop criteria
    int    QR_stop_rank;            // Partial QR maximum rank
    int    n_point;                 // Number of points for the kernel matrix
    int    krnl_mat_size;           // Size of the kernel matrix
    int    max_leaf_points;         // Maximum point in a leaf node's box
    int    n_node;                  // Number of nodes in this H2 tree
    int    root_idx;                // Index of the root node (== n_node - 1, save it for convenience)
    int    n_leaf_node;             // Number of leaf nodes in this H2 tree
    int    max_child;               // Maximum number of children per node, == 2^pt_dim
    int    max_neighbor;            // Maximum number of neighbor nodes per node, == 2^pt_dim
    int    max_level;               // Maximum level of this H2 tree, (root = 0, total max_level + 1 levels)
    int    min_adm_level;           // Minimum level of reduced admissible pair
    int    n_r_inadm_pair;          // Number of reduced inadmissible pairs
    int    n_r_adm_pair;            // Number of reduced admissible pairs
    int    n_UJ;                    // Number of projection matrices & skeleton row sets, == n_node
    int    n_B;                     // Number of generator matrices
    int    n_D;                     // Number of dense blocks
    int    BD_JIT;                  // If B and D matrices are computed just-in-time in matvec
    int    print_timers;            // If H2ERI prints internal timers for performance analysis
    int    print_dbginfo;           // If H2ERI prints debug information
    int    *parent;                 // Size n_node, parent index of each node
    int    *children;               // Size n_node * max_child, indices of a node's children nodes
    int    *pt_cluster;             // Size n_node * 2, start and end (included) indices of points belong to each node
    int    *mat_cluster;            // Size n_node * 2, start and end (included) indices of matvec vector elements belong to each node
    int    *n_child;                // Size n_node, number of children nodes of each node
    int    *node_level;             // Size n_node, level of each node
    int    *node_height;            // Size n_node, height of each node
    int    *level_n_node;           // Size max_level+1, number of nodes in each level
    int    *level_nodes;            // Size (max_level+1) * n_leaf_node, indices of nodes on each level
    int    *height_n_node;          // Size max_level+1, number of nodes of each height
    int    *height_nodes;           // Size (max_level+1) * n_leaf_node, indices of nodes of each height
    int    *r_inadm_pairs;          // Size unknown, reduced inadmissible pairs 
    int    *r_adm_pairs;            // Size unknown, reduced admissible pairs 
    int    *node_inadm_lists;       // Size n_node * max_neighbor, lists of each node's inadmissible nodes
    int    *node_n_r_inadm;         // Size n_node, numbers of each node's reduced inadmissible nodes
    int    *node_n_r_adm;           // Size n_node, numbers of each node's reduced admissible nodes
    int    *coord_idx;              // Size n_point, original index of each sorted point
    int    *B_p2i_rowptr;           // Size n_node+1, row_ptr array of the CSR matrix for mapping B{i, j} to a B block index
    int    *B_p2i_colidx;           // Size n_B, col_idx array of the CSR matrix for mapping B{i, j} to a B block index
    int    *B_p2i_val;              // Size n_B, val array of the CSR matrix for mapping B{i, j} to a B block index
    int    *D_p2i_rowptr;           // Size n_node+1, row_ptr array of the CSR matrix for mapping D{i, j} to a D block index
    int    *D_p2i_colidx;           // Size n_D, col_idx array of the CSR matrix for mapping D{i, j} to a D block index
    int    *D_p2i_val;              // Size n_D, val array of the CSR matrix for mapping D{i, j} to a D block index
    int    *B_nrow;                 // Size n_B, numbers of rows of generator matrices
    int    *B_ncol;                 // Size n_B, numbers of columns of generator matrices
    int    *D_nrow;                 // Size n_D, numbers of rows of dense blocks in the original matrix
    int    *D_ncol;                 // Size n_D, numbers of columns of dense blocks in the original matrix
    size_t *B_ptr;                  // Size n_B, offset of each generator matrix's data in B_data
    size_t *D_ptr;                  // Size n_D, offset of each dense block's data in D_data
    DTYPE  max_leaf_size;           // Maximum size of a leaf node's box
    DTYPE  QR_stop_tol;             // Partial QR stop column norm tolerance
    DTYPE  *coord;                  // Size n_point * xpt_dim, sorted point coordinates
    DTYPE  *coord0;                 // Size n_point * xpt_dim, original (not sorted) point coordinates
    DTYPE  *enbox;                  // Size n_node * (2*pt_dim), enclosing box data of each node
    DTYPE  *root_enbox;             // Size 2 * pt_dim, enclosing box of the root node
    DTYPE  *B_data;                 // Size unknown, data of generator matrices
    DTYPE  *D_data;                 // Size unknown, data of dense blocks in the original matrix
    H2E_int_vec_p     B_blk;        // Size BD_NTASK_THREAD * n_thread, B matrices task partitioning
    H2E_int_vec_p     D_blk0;       // Size BD_NTASK_THREAD * n_thread, diagonal blocks in D matrices task partitioning
    H2E_int_vec_p     D_blk1;       // Size BD_NTASK_THREAD * n_thread, inadmissible blocks in D matrices task partitioning
    H2E_dense_mat_p   *U;           // Size n_node, Projection matrices
    H2E_dense_mat_p   *y0;          // Size n_node, temporary arrays used in matvec
    H2E_dense_mat_p   *y1;          // Size n_node, temporary arrays used in matvec
    int    n_matvec;                // Number of performed matvec
    size_t mat_size[11];            // See below macros
    double timers[11];              // See below macros
};
typedef struct H2ERI* H2ERI_p;

// For H2ERI_t->mat_size
typedef enum 
{
    U_SIZE_IDX = 0,     // Total size of U matrices
    B_SIZE_IDX,         // Total size of B matrices
    D_SIZE_IDX,         // Total size of D matrices
    MV_FWD_SIZE_IDX,    // Total memory footprint of H2 matvec forward transformation
    MV_MID_SIZE_IDX,    // Total memory footprint of H2 matvec intermediate multiplication
    MV_BWD_SIZE_IDX,    // Total memory footprint of H2 matvec backward transformation
    MV_DEN_SIZE_IDX,    // Total memory footprint of H2 matvec dense multiplication
    MV_VOP_SIZE_IDX,    // Total memory footprint of H2 matvec OpenMP vector operations
} size_idx_t;

// For H2ERI_t->timers
typedef enum
{
    PT_TIMER_IDX = 0,   // Hierarchical partitioning
    U_BUILD_TIMER_IDX,  // U matrices construction
    B_BUILD_TIMER_IDX,  // B matrices construction
    D_BUILD_TIMER_IDX,  // D matrices construction
    MV_FWD_TIMER_IDX,   // H2 matvec forward transformation
    MV_MID_TIMER_IDX,   // H2 matvec intermediate multiplication
    MV_BWD_TIMER_IDX,   // H2 matvec backward transformation
    MV_DEN_TIMER_IDX,   // H2 matvec dense multiplication
    MV_VOP_TIMER_IDX,   // H2 matvec OpenMP vector operations
} timer_idx_t;

#define ALPHA_SUP 1.0

// Initialize a H2ERI structure
// Input parameters:
//   scr_tol : Tolerance of Schwarz screening (typically 1e-10)
//   ext_tol : Tolerance of shell pair extent (typically 1e-10)
//   QR_tol  : Tolerance of column-pivoting QR (controls the overall accuracy)
// Output parameter:
//   h2eri_ : Initialized H2ERI structure
void H2ERI_init(H2ERI_p *h2eri_, const double scr_tol, const double ext_tol, const double QR_tol);

// Destroy a H2ERI structure
// Input parameter:
//   h2eri : H2ERI structure to be destroyed
void H2ERI_destroy(H2ERI_p h2eri);

// Print H2ERI statistic information
// Input parameter:
//   h2eri : H2ERI structure to be destroyed
void H2ERI_print_statistic(H2ERI_p h2eri);

#ifdef __cplusplus
}
#endif

#endif

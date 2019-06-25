#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

#include "CMS.h"
#include "H2ERI_typedef.h"
#include "H2Pack_aux_structs.h"

// Partition the ring area (r1 < r < r2) using multiple layers of 
// box surface and generate the same number of uniformly distributed 
// proxy points on each box surface layer [-r, r]^3. 
// Input parameters:
//   r1, r2     : Radius of ring area
//   nlayer     : Number of layers
//   npts_layer : Minimum number of proxy points on each layer
// Output parameters:
//   pp : H2P_dense_mat structure, contains coordinates of proxy points
void H2ERI_generate_proxy_point_layers(
    const double r1, const double r2, const int nlayer, 
    int npts_layer, H2P_dense_mat_t pp
)
{
    // 1. Decide the number of proxy points on each layer
    int npts_face = npts_layer / 6;
    int npts_axis = (int) ceil(sqrt((double) npts_face));
    npts_layer = 6 * npts_axis * npts_axis;
    int npts_total = nlayer * npts_layer;
    H2P_dense_mat_resize(pp, 3, npts_total);
    
    // 2. Generate a layer of proxy points on a standard [-1, 1]^3 box surface
    double h = 2.0 / (double) (npts_axis + 1);
    double *x = pp->data;
    double *y = pp->data + npts_total;
    double *z = pp->data + npts_total * 2;
    int index = 0;
    for (int i = 0; i < npts_axis; i++)
    {
        double h_i = h * (i + 1) - 1.0;
        for (int j = 0; j < npts_axis; j++)
        {
            double h_j = h * (j + 1) - 1.0;
            
            x[index + 0] = h_i;
            y[index + 0] = h_j;
            z[index + 0] = -1.0;
            
            x[index + 1] = h_i;
            y[index + 1] = h_j;
            z[index + 1] = 1.0;
            
            x[index + 2] = h_i;
            y[index + 2] = -1.0;
            z[index + 2] = h_j;
            
            x[index + 3] = h_i;
            y[index + 3] = 1.0;
            z[index + 3] = h_j;
            
            x[index + 4] = -1.0;
            y[index + 4] = h_i;
            z[index + 4] = h_j;
            
            x[index + 5] = 1.0;
            y[index + 5] = h_i;
            z[index + 5] = h_j;
            
            index += 6;
        }
    }
    // Copy the proxy points on the standard [-1, 1]^3 box surface to each layer
    size_t layer_msize = sizeof(double) * npts_layer;
    for (int i = 1; i < nlayer; i++)
    {
        memcpy(x + i * npts_layer, x, layer_msize);
        memcpy(y + i * npts_layer, y, layer_msize);
        memcpy(z + i * npts_layer, z, layer_msize);
    }
    
    // 3. Scale each layer
    int nlayer1 = MAX(nlayer - 1, 1);
    double dr = ((r2 - r1) / r1) / (double) nlayer1;
    for (int i = 0; i < nlayer; i++)
    {
        double *x_i = x + i * npts_layer;
        double *y_i = y + i * npts_layer;
        double *z_i = z + i * npts_layer;
        double r = r1 * (1.0 + i * dr);
        #pragma omp simd
        for (int j = 0; j < npts_layer; j++)
        {
            x_i[j] *= r;
            y_i[j] *= r;
            z_i[j] *= r;
        }
    }
}
#ifndef GPUFIT_COMPTON_PE_CUH_INCLUDED
#define GPUFIT_COMPTON_PE_CUH_INCLUDED

#include "kn_pe.cuh"

/* Description of the calculate_gauss2d function
* ==============================================
*
* This function calculates the values of two-dimensional Compton/PE model functions
* and their partial derivatives with respect to the model parameters. 
*
* No independent variables are passed to this model function.  Hence, the 
* (X, Y) coordinate of the first data value is assumed to be (0.0, 0.0).  For
* a fit size of M x N data points, the (X, Y) coordinates of the data are
* simply the corresponding array index values of the data array, starting from
* zero.
*
* Parameters:
*
* parameters: An input vector of model parameters.
*             p[0]: Compton line integral 
*             p[1]: PE line integral
*
* n_fits: The number of fits. (not used)
*
* n_points: The number of data points per fit.
*
* value: An output vector of model function values.
*
* derivative: An output vector of model function partial derivatives.
*
* point_index: The data point index.
*
* fit_index: The fit index. (not used)
*
* chunk_index: The chunk index. (not used)
*
* user_info: An input vector containing user information. (not used)
*
* user_info_size: The size of user_info in bytes. (not used)
*
* Calling the calculate_compton_pe function
* ======================================
*
* This __device__ function can be only called from a __global__ function or an other
* __device__ function.
*
*/

__device__ void calculate_compton_pe(
    REAL const * parameters,
    int const n_fits,
    int const n_points,
    REAL * value,
    REAL * derivative,
    int const point_index,
    int const fit_index,
    int const chunk_index,
    char * user_info,
    std::size_t const user_info_size)
{
    // n_fits is the total number of sinogram pixels
    // n_points is 2, corresponding to two projections, high and low per fit
    // value is the same size of n_points
    // derivative is the same size of n_parameters
    // user_info contains the two spectra and photon count

    // parameters

    REAL const * param = parameters;

    // user info
    // layout is:
    // scc, scp, pch, pcl, n_kev_h, n_kev_l, kev_h, kev_l 
    // spctrm_h, spctrm_l, spctrm_h_ph, spctrm_l_ph, spctrm_h_kn, spctrm_l_kn

    // photon counts
    REAL * uif = (REAL *) user_info;
    REAL const scc = uif[0];
    REAL const scp = uif[1];
    REAL const pch = uif[2];
    REAL const pcl = uif[3];
    // energy levels, here we consider 10kev to 140kev
    int const n_kev_h = (int) uif[4];
    int const n_kev_l = (int) uif[5];
    int ci = 6;
    REAL * kev_h = uif + ci; ci += n_kev_h;
    REAL * kev_l = uif + ci; ci += n_kev_l;
    // normalized spectra
    REAL * spctrm_h = uif + ci; ci += n_kev_h;
    REAL * spctrm_l = uif + ci; ci += n_kev_l;
    REAL * spctrm_h_ph = uif + ci; ci += n_kev_h;
    REAL * spctrm_l_ph = uif + ci; ci += n_kev_l;
    REAL * spctrm_h_kn = uif + ci; ci += n_kev_h;
    REAL * spctrm_l_kn = uif + ci; ci += n_kev_l;

    // value 

    REAL c = param[0], p = param[1];
    REAL h = 0, l = 0;
    REAL dh_dc = 0, dh_dp = 0;
    REAL dl_dc = 0, dl_dp = 0;
    REAL tmp = 0;
    
    c /= scc; p /= scp;

    REAL * current_derivative = derivative + point_index;

    if (point_index == 0)
    {
        // for high energy projection
        for (int i = 0; i < n_kev_h; ++i)
        {
            tmp = exp(-(c * klein_nishina(kev_h[i]) + p * photoelectric(kev_h[i])));
            h += tmp * spctrm_h[i];
            dh_dc += tmp * spctrm_h_kn[i];
            dh_dp += tmp * spctrm_h_ph[i];
        }
        value[0] = -log(h * pch / pch);
        current_derivative[0 * n_points] = dh_dc / h / scc;
        current_derivative[1 * n_points] = dh_dp / h / scp; 
    } 
    else
    {
        // for low energy projection
        for (int i = 0; i < n_kev_l; ++i)
        {
            tmp = exp(-(c * klein_nishina(kev_l[i]) + p * photoelectric(kev_l[i])));
            l += tmp * spctrm_l[i];        
            dl_dc += tmp * spctrm_l_kn[i];
            dl_dp += tmp * spctrm_l_ph[i];
        }
        value[1] = -log(l * pcl / pcl);
        current_derivative[0 * n_points] = dl_dc / l / scc;
        current_derivative[1 * n_points] = dl_dp / l / scp;
    }

}

#endif

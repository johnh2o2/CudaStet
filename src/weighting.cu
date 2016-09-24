#include "weighting.h"
#include "config.h"

__device__ weight_function_t 
    		p_exp_weighting_x = exp_weighting_x;
__device__ weight_function_t 
    		p_constant_weighting = constant_weighting; 
__device__ weight_function_t
		p_exp_weighting_x_and_y = exp_weighting_x_and_y;

// EXPONENTIAL
__host__ __device__ real_type
exp_weighting_x(real_type t1, real_type t2, real_type x1, real_type x2, void *params){
    real_type *pars = (real_type *) params;
    real_type invdt = pars[0];
    real_type n     = pars[1];
    return exp(-pow(abs(t1 - t2) * invdt, n));
}

__host__ __device__ real_type
exp_weighting_x_and_y(real_type t1, real_type t2, real_type x1, real_type x2, void *params){
    real_type *pars = (real_type *) params;
    real_type invdt = pars[0];
    real_type nt     = pars[1];
    real_type invdx = pars[2];
    real_type nx    = pars[3];
    return exp(-pow(abs(t1 - t2) * invdt, nt) - pow(abs(x1 - x2) * invdx, nx));
}


// CONSTANT
__host__ __device__ real_type 
constant_weighting(real_type t1, real_type t2, real_type x1, real_type x2, void *params){
    return 1.0;
}

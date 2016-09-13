#include "weighting.h"

__device__ weight_function_t 
			p_exp_weighting = exp_weighting;
__device__ weight_function_t 
			p_constant_weighting = constant_weighting; 

// EXPONENTIAL
__host__ __device__ real_type
exp_weighting(real_type t1, real_type t2, void *params){
	real_type *pars = (real_type *) params;
	real_type invdt = pars[0];
	real_type n     = pars[1];
	return exp(-pow(abs(t1 - t2) * invdt, n));
}

// CONSTANT
__host__ __device__ real_type 
constant_weighting(real_type t1, real_type t2, void *params){
	return 1.0;
}

#include "weighting.h"

__device__ real_type
exp_weighting_gpu(real_type t1, real_type t2, void *params){
	real_type invdt = ((real_type *) params)[0];
	real_type n  = ((real_type *) params)[1];
	return exp(-pow(abs(t1 - t2) * invdt, n));
}

__device__ real_type 
constant_weighting_gpu(real_type t1, real_type t2, void *params){
	return 1.0;
}


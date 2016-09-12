#ifndef WEIGHTING_H
#define WEIGHTING_H



// WEIGHTING FUNCTIONS
typedef real_type (*weight_function_gpu_t)(real_type, real_type, void *);

__device__ real_type
exp_weighting_gpu(real_type t1, real_type t2, void *params);

__device__ real_type 
constant_weighting_gpu(real_type t1, real_type t2, void *params);


// static pointers
__device__ weight_function_gpu_t 
			p_exp_weighting_gpu = exp_weighting_gpu;
__device__ weight_function_gpu_t 
			p_constant_weighting_gpu = constant_weighting_gpu;

///////////////////////////////////

#endif
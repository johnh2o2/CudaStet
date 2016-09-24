#ifndef WEIGHTING_H
#define WEIGHTING_H
#include "config.h"

// function pointers
typedef real_type (*weight_function_t)(real_type, real_type, real_type, real_type, void *);


__host__ __device__ real_type
exp_weighting_x(real_type t1, real_type t2, real_type x1, real_type x2, void *params);

__host__ __device__ real_type 
constant_weighting(real_type t1, real_type t2, real_type x1, real_type x2, void *params);

__host__ __device__ real_type
exp_weighting_x_and_y(real_type t1, real_type t2, real_type x1, real_type x2, void *params);


// static pointers
extern __device__ weight_function_t 
    		p_exp_weighting_x; 
extern __device__ weight_function_t 
    		p_constant_weighting; 
extern __device__ weight_function_t
		p_exp_weighting_x_and_y;
///////////////////////////////////

#endif

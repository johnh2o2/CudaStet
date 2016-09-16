#ifndef WEIGHTING_H
#define WEIGHTING_H

// function pointers
typedef real_type (*weight_function_t)(real_type, real_type, void *);


__host__ __device__ real_type
exp_weighting(real_type t1, real_type t2, void *params);

__host__ __device__ real_type 
constant_weighting(real_type t1, real_type t2, void *params);


// static pointers
extern __device__ weight_function_t 
    		p_exp_weighting; 
extern __device__ weight_function_t 
    		p_constant_weighting; 

///////////////////////////////////

#endif
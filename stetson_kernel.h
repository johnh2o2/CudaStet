#ifndef STETSON_KERNEL_H
#define STETSON_KERNEL_H
#include "weighting.h"

__host__ real_type
stetson_j_kernel_cpu(real_type *delta, int N);

__global__ void
stetson_j_kernel(real_type *x, real_type *delta, real_type *J, 
	             real_type *W, weight_function_gpu_t w, 
	             void *weight_params, int N);



#endif

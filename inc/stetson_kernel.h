#ifndef STETSON_KERNEL_H
#define STETSON_KERNEL_H
#include <stdio.h>
#include "weighting.h"
#include "config.h"

// This may also be implemented somewhere in CUDA, but this ensures that it exists and we can
// customize it ourselves. Pulled this from somewhere on StackExchange, can't find the original post!!
#define CUDA_CALL(ans) { gpuAssert((ans), __FILE__, __LINE__); }

// inline function for printing cuda errors
inline void gpuAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess)
    {
        fprintf(stderr, "CUDA ERROR %-24s L[%-5d]: %s\n", 
        	            file, line, cudaGetErrorString(code));
        exit(code);
    }
}


__host__ real_type
stetson_j_kernel_cpu(real_type *x, real_type *delta,
    				 weight_function_t w, 
    	             void *weight_params, int N);
__global__ void
stetson_j_kernel(real_type *x, real_type *delta, real_type *J, 
                 real_type *W, weight_function_t w, 
                 void *weight_params, int N);

__global__ void
stetson_j_kernel_batch(real_type *x, real_type *delta, real_type *J, 
                 real_type *W, weight_function_t w, 
                 void *weight_params, size_t wparsize, int *N, int Nsample);

#endif

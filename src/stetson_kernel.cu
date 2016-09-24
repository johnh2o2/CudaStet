#include "stetson_kernel.h"


__host__ __device__ void
add_j_w(real_type x1, real_type x2, real_type d1, real_type d2,
			weight_function_t w, void *wpars, real_type *Jval, 
			real_type *Wval){

	real_type d = d1 * d2;
	if (x1 == x2) 
		d -= 1;

	real_type s = 1.0;
	if (d < 0)
		s = -1.0;

	real_type wv = (*w)(x1, x2, d1, d2, wpars);
	(*Wval) += wv;
	(*Jval) += wv * s * sqrt(s * d);

}

__host__ real_type
stetson_j_kernel_cpu(real_type *x, real_type *delta,
                     weight_function_t w, 
                     void *weight_params, int N){

    real_type J = 0, W = 0;
    for (int i = 0; i < N; i++){
        for (int j = i; j < N; j++) {
        	add_j_w(x[i], x[j], delta[i], delta[j], w, 
        		    weight_params, &J, &W);
        }
    }
    return J/W;
        
}



__global__ void
stetson_j_kernel(real_type *x, real_type *delta, real_type *J, 
                 real_type *W, weight_function_t w, 
                 void *weight_params, int N){

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        J[i] = 0;
        W[i] = 0;
        for(int j = i; j < N; j++){
        	add_j_w(x[i], x[j], delta[i], delta[j], w, 
        		    weight_params, &(J[i]), &(W[i]));
        }
    }
}

__global__ void
stetson_j_kernel_batch(real_type *x, real_type *delta, real_type *J, 
                 real_type *W, weight_function_t w, 
                 void *weight_params, size_t wparsize, int *N, int Nsample){

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int sample_index;

    int nsum = 0;
    for (sample_index = 0; sample_index < Nsample; sample_index++) {
    	nsum += N[sample_index];
    	if (i < nsum)
    		break;
    }
    
    if (i < nsum){
        J[i] = 0;
        W[i] = 0;
        void *wpars = (weight_params == NULL) ? NULL
        			    : (void *) ( ((char *)weight_params) 
        			    	             + wparsize * sample_index);

        for(int j = i; j < nsum; j++){
        	add_j_w(x[i], x[j], delta[i], delta[j], w, 
        		    wpars, &(J[i]), &(W[i]));
        }
    }
}




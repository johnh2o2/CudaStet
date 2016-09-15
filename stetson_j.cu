
#include <stdlib.h>
#include <math.h>

#include "stetson.h"
#include "stetson_kernel.h"
#include "utils.h"
#include "weighting.h"


// z-scale the data
void 
make_delta(real_type *y, real_type *err, real_type *delta, int N) {
    real_type mu = mean(y, N);
    int i;

    real_type bias = sqrt(N / ((real_type) (N - 1)));
    for(i = 0; i < N; i++) 
    	delta[i] = bias * (y[i] - mu) / err[i];
}

void 
get_exp_params(real_type *x, int N, void **params) {

	*params = malloc(2 * sizeof(real_type));

	// get array of t_i - t_{i-1}
	real_type dt[N-1];
	for(int i = 0; i < N - 1; i++) 
		dt[i] = x[i+1] - x[i];

	// param 1: inverse of mean(dt)
	((real_type *)(*params))[0] = 1./median(dt, N-1);

	// param 2: exponent n: exp(-|t2 - t1|^n / dt^n)
	((real_type *)(*params))[1] = 1;
}

void 
get_weighting_gpu(real_type *x, int N, weight_type WEIGHTING, 
    			   void **params, weight_function_t *w){

    if (WEIGHTING & EXP) {
    	CUDA_CALL(
    		cudaMemcpyFromSymbol(w, p_exp_weighting, 
    	                        sizeof(weight_function_t) )
    	);

    	void *h_params;
    	get_exp_params(x, N, &h_params);

    	CUDA_CALL(cudaMalloc(params, 2 * sizeof(real_type)));
    	//CUDA_CALL(cudaThreadSynchronize());
    	CUDA_CALL(cudaMemcpy((*params), h_params, 
    		                       2 * sizeof(real_type), 
    		                       cudaMemcpyHostToDevice ));
    	free(h_params);
    }
    else if (WEIGHTING & CONSTANT) {
    	CUDA_CALL(
    		cudaMemcpyFromSymbol(w, p_constant_weighting, 
    	                        sizeof(weight_function_t) )
    	);

    	(*params) = NULL;

    }

}

void 
get_weighting_cpu(real_type *x, int N, weight_type WEIGHTING,
    			   void **params, weight_function_t *w){

    if (WEIGHTING & EXP) {
    	(*w)      = &exp_weighting;
    	get_exp_params(x, N, params);
    }
    else if (WEIGHTING & CONSTANT) {
    	(*w)      = &constant_weighting;
    	(*params) = NULL;
    }
}

    			  
// compute Stetson J index (Stetson 1996)
real_type 
stetson_j_gpu(real_type *x, real_type *y, real_type *err, 
              weight_type WEIGHTING, int N){

    // WEIGHTING FUNCTIONS
    weight_function_t weight_func;
    void *d_params = NULL;

    get_weighting_gpu(x, N, WEIGHTING, &d_params, &weight_func);
    //////

    
    // scale y values
    real_type *delta = (real_type *) malloc(N * sizeof(real_type));
    make_delta(y, err, delta, N);
    
    // allocate GPU variables
    real_type *deltag, *Jg, *Wg, *xg;
    CUDA_CALL(cudaMalloc((void **)&deltag, N * sizeof(real_type)));
    CUDA_CALL(cudaMalloc((void **)&Jg, N * sizeof(real_type)));
    CUDA_CALL(cudaMalloc((void **)&Wg, N * sizeof(real_type)));
    CUDA_CALL(cudaMalloc((void **)&xg, N * sizeof(real_type)));
    
    // transfer to GPU
    CUDA_CALL(cudaMemcpy(deltag, delta, N * sizeof(real_type), 
    	cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(xg, x, N * sizeof(real_type), 
    	cudaMemcpyHostToDevice));

    // set block size
    int NBLOCKS = N / BLOCK_SIZE;
    if (NBLOCKS * BLOCK_SIZE < N) NBLOCKS += 1;

    // launch GPU kernel -- performs pair sums, 
    // which must be added at the end.
    stetson_j_kernel<<< NBLOCKS, BLOCK_SIZE >>>( xg, deltag, Jg, 
    	                              Wg, weight_func, d_params, N );

    real_type *J = (real_type *)malloc(N * sizeof(real_type));
    real_type *W = (real_type *)malloc(N * sizeof(real_type));
    CUDA_CALL(cudaMemcpy(J, Jg, N * sizeof(real_type), 
    	cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(W, Wg, N * sizeof(real_type), 
    	cudaMemcpyDeviceToHost));

    // Free memory
    CUDA_CALL(cudaFree(Jg));
    CUDA_CALL(cudaFree(deltag));
    CUDA_CALL(cudaFree(d_params));
    
    return sum(J, N) / sum(W, N);
}

real_type
stetson_j_cpu(real_type *x, real_type *y, real_type *err, 
              weight_type WEIGHTING, int N){

    // scale y values by mean and variance
    real_type *delta = (real_type *) malloc(N * sizeof(real_type));
    make_delta(y, err, delta, N);

    // get weighting function and parameters
    weight_function_t weight_func;
    void *params = NULL;

    get_weighting_cpu(x, N, WEIGHTING, &params, &weight_func);

    // compute J
    real_type J = stetson_j_kernel_cpu(x, delta, weight_func, 
    	                               params, N);

    // free weight function params
    if(params != NULL) free(params);

    return J;
}
/*
real_type
stetson_k(real_type *y, real_type *err, int N){


}
*/


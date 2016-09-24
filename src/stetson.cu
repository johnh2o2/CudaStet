
#include <stdlib.h>
#include <math.h>

#include "stetson.h"
#include "stetson_kernel.h"
#include "utils.h"
#include "weighting.h"
#include "stetson_mean.h"

#define STETSON_MEAN 0


void 
get_exp_params(real_type *x, const int N, void **params, size_t *paramsize) {
    *paramsize = 2 * sizeof(real_type);

	*params = malloc(*paramsize);

	// get array of t_i - t_{i-1}
	real_type dt[N-1];
	for(int i = 0; i < N - 1; i++) 
		dt[i] = abs(x[i+1] - x[i]);

	// param 1: inverse of mean(dt)
	((real_type *)(*params))[0] = 1./median(dt, N-1);

	// param 2: exponent n: exp(-|t2 - t1|^n / dt^n)
	((real_type *)(*params))[1] = 1;
}

void
get_expxy_params(real_type *x, real_type *y, const int N, void **params, size_t *paramsize) {
	void *paramx, *paramy;
	size_t xsize, ysize;
	
	// gets exponential weighting params for both x and y
	get_exp_params(x, N, &paramx, &xsize);
	get_exp_params(y, N, &paramy, &ysize);

	// now combine the two params
	(*paramsize) = xsize + ysize;
	
	// allocate
	*params = malloc(*paramsize);

	// copy x
	memcpy(*params, paramx, xsize);
	
	// hack so we can do void pointer arithmetic
	void *ptry = (void *) (((char *) *params) + xsize);
	
	// copy y
	memcpy(ptry, paramy, ysize);

	// free
	free(paramx); free(paramy);
}


void 
get_weighting_gpu_batch(real_type *x, real_type *y, int *N, const int Nsamples, 
                        const weight_type WEIGHTING, void **params, 
                        weight_function_t *w, size_t *paramsize){

    if ((WEIGHTING & EXP) || (WEIGHTING & EXPXY)) {
        // get device function pointer
	if (WEIGHTING & EXP) {
        	CUDA_CALL(
         	   cudaMemcpyFromSymbol(w, p_exp_weighting_x, 
                                sizeof(weight_function_t) )
        	);
	}
	else if (WEIGHTING & EXPXY) {
        	CUDA_CALL(
         	   cudaMemcpyFromSymbol(w, p_exp_weighting_x_and_y, 
                                sizeof(weight_function_t) )
        	);
	}

        void *h_params, *h_params_temp;

        // compute total number of observations
        int npoints = 0;
        for(int i = 0; i < Nsamples; i++) 
            npoints+=N[i];

        int npts = 0;
        // get params for each sample
        for(int i = 0; i < Nsamples; i++){
	    if (WEIGHTING & EXP)
                get_exp_params(x + npts, N[i], &h_params_temp, paramsize);

	    else if (WEIGHTING & EXPXY) 
                get_expxy_params(x + npts, y + npts, N[i], &h_params_temp, paramsize);

            if (i == 0)
                h_params = malloc(Nsamples * (*paramsize));

            npts += N[i];

            // hack to get around the fact that we cant do pointer
            // arithmetic on void pointers
            void *ptr = (void *)(((char *)h_params) + i * (*paramsize));
            memcpy(ptr, h_params_temp, *paramsize);
            free(h_params_temp);

        }

        // copy params to device
        CUDA_CALL(cudaMalloc(params, (*paramsize) * Nsamples));
        CUDA_CALL(cudaMemcpy((*params), h_params, Nsamples * (*paramsize),
                                   cudaMemcpyHostToDevice ));

        // free memory
        free(h_params);
        
    }
    else if (WEIGHTING & CONSTANT) {
        CUDA_CALL(
            cudaMemcpyFromSymbol(w, p_constant_weighting, 
                                sizeof(weight_function_t) )
        );

        (*params) = NULL;
        (*paramsize) = 0;

    }

}

void 
get_weighting_gpu(real_type *x, real_type *y, const int N, const weight_type WEIGHTING, 
                  void **params, weight_function_t *w, size_t *paramsize){
    void *h_params;

    if (WEIGHTING & EXP) {
    	CUDA_CALL(
    		cudaMemcpyFromSymbol(w, p_exp_weighting_x, 
    		                       sizeof(weight_function_t) )
    	);
	get_exp_params(x, N, &h_params, paramsize);
    }
    else if (WEIGHTING & EXPXY) {
    	CUDA_CALL(
    		cudaMemcpyFromSymbol(w, p_exp_weighting_x_and_y, 
    	                               sizeof(weight_function_t) )
    	);
	get_expxy_params(x, y, N, &h_params, paramsize);
    }
    else if (WEIGHTING & CONSTANT) {
    	CUDA_CALL(
    		cudaMemcpyFromSymbol(w, p_constant_weighting, 
    	                              sizeof(weight_function_t) )
    	);

    	(*params) = NULL;
        (*paramsize) = 0;

    }

    // clean up
    if (!(WEIGHTING & CONSTANT)) {
    	CUDA_CALL(cudaMalloc(params, *paramsize));
    	CUDA_CALL(cudaMemcpy((*params), h_params, *paramsize, 
    		                       cudaMemcpyHostToDevice ));
    	free(h_params);
    }
}

void 
get_weighting_cpu(real_type *x, real_type *y, const int N, const weight_type WEIGHTING,
                  void **params, weight_function_t *w, size_t *paramsize){

    if (WEIGHTING & EXP) {
    	(*w)      = &exp_weighting_x;
    	get_exp_params(x, N, params, paramsize);
    }
    else if (WEIGHTING & EXPXY) {
	(*w)      = &exp_weighting_x_and_y;
	get_expxy_params(x, y, N, params, paramsize);
    }
    else if (WEIGHTING & CONSTANT) {
    	(*w)      = &constant_weighting;
    	(*params) = NULL;
        (*paramsize) = 0;
    }
}
// compute Stetson J index (Stetson 1996)
real_type *
stetson_j_gpu_batch(real_type *x, real_type *y, real_type *err, 
              const weight_type WEIGHTING, int *N, const int Nsamples){

    // compute total number of observations
    int npoints = 0;
    for(int i = 0; i < Nsamples; i++) npoints += N[i];
    
    
    // scale y values
    real_type *delta = (real_type *) malloc(npoints * sizeof(real_type));
    int npts = 0;
    for (int i = 0; i < Nsamples; i++){
        real_type mu = STETSON_MEAN ? stetson_mean(y + npts, err + npts, 
                                                   APARAM, BPARAM, CRITERION, N[i])
                                    : mean(y + npts, N[i]);
        
        make_delta(y + npts, err + npts, mu, delta + npts, N[i]);
        npts += N[i];
    }

    // WEIGHTING FUNCTIONS
    weight_function_t weight_func;
    void *d_params = NULL;
    size_t psize;


    get_weighting_gpu_batch(x, delta, N, Nsamples, WEIGHTING, &d_params,
                            &weight_func, &psize);
    
    //printf("npoints = %d\n", npoints); fflush(stdout);

    // allocate GPU variables
    real_type *deltag, *Jg, *Wg, *xg;
    int *Ng;
    CUDA_CALL(cudaMalloc((void **)&deltag, npoints * sizeof(real_type)));
    CUDA_CALL(cudaMalloc((void **)&Jg, npoints * sizeof(real_type)));
    CUDA_CALL(cudaMalloc((void **)&Wg, npoints * sizeof(real_type)));
    CUDA_CALL(cudaMalloc((void **)&xg, npoints * sizeof(real_type)));
    CUDA_CALL(cudaMalloc((void **)&Ng, npoints * sizeof(int)));
    
    // transfer to GPU
    CUDA_CALL(cudaMemcpy(deltag, delta, npoints * sizeof(real_type), 
        cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(xg, x, npoints * sizeof(real_type), 
        cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(Ng, N, Nsamples * sizeof(int),
        cudaMemcpyHostToDevice));

    // set block size
    int NBLOCKS = npoints / BLOCK_SIZE;
    if (NBLOCKS * BLOCK_SIZE < npoints) NBLOCKS += 1;

    // launch GPU kernel -- performs pair sums, 
    // which must be added at the end
    stetson_j_kernel_batch<<< NBLOCKS, BLOCK_SIZE >>>( xg, deltag, Jg, 
                                      Wg, weight_func, d_params, psize, Ng, Nsamples);

    real_type *Jraw = (real_type *)malloc(npoints * sizeof(real_type));
    real_type *Wraw = (real_type *)malloc(npoints * sizeof(real_type));
    real_type *J =    (real_type *)malloc(Nsamples * sizeof(real_type));

    CUDA_CALL(cudaMemcpy(Jraw, Jg, npoints * sizeof(real_type), 
        cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(Wraw, Wg, npoints * sizeof(real_type), 
        cudaMemcpyDeviceToHost));

    // calculate J from results
    npts = 0;
    for( int i = 0; i < Nsamples; i++ ){
        J[i] = sum(Jraw + npts, N[i]) / sum(Wraw + npts, N[i]);
        npts += N[i];
    }
  
    // Free memory
    CUDA_CALL(cudaFree(deltag));
    CUDA_CALL(cudaFree(Jg));
    CUDA_CALL(cudaFree(Wg));
    CUDA_CALL(cudaFree(xg));
    CUDA_CALL(cudaFree(d_params));
    CUDA_CALL(cudaFree(Ng));

    free(delta); 
    free(Jraw);
    free(Wraw);

    return J;
}

    			  
// compute Stetson J index (Stetson 1996)
real_type 
stetson_j_gpu(real_type *x, real_type *y, real_type *err, 
              const weight_type WEIGHTING, const int N){

    // scale y values
    real_type *delta = (real_type *) malloc(N * sizeof(real_type));
    real_type mu = STETSON_MEAN ? stetson_mean(y, err, APARAM, BPARAM, CRITERION, N)
                                : mean(y, N);
    
    make_delta(y, err, mu, delta, N);

    // WEIGHTING FUNCTIONS
    weight_function_t weight_func;
    void *d_params = NULL;
    size_t psize;


    get_weighting_gpu(x, delta, N, WEIGHTING, &d_params, &weight_func, &psize);
    //////

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
    // which must be added at the end
    stetson_j_kernel<<< NBLOCKS, BLOCK_SIZE >>>( xg, deltag, Jg, 
    	                              Wg, weight_func, d_params, N);

    real_type *J = (real_type *)malloc(N * sizeof(real_type));
    real_type *W = (real_type *)malloc(N * sizeof(real_type));
    CUDA_CALL(cudaMemcpy(J, Jg, N * sizeof(real_type), 
    	cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(W, Wg, N * sizeof(real_type), 
    	cudaMemcpyDeviceToHost));

    real_type Jstet =  sum(J, N) / sum(W, N);
  
    // Free memory
    CUDA_CALL(cudaFree(deltag));
    CUDA_CALL(cudaFree(Jg));
    CUDA_CALL(cudaFree(Wg));
    CUDA_CALL(cudaFree(xg));
    CUDA_CALL(cudaFree(d_params));

    free(delta); 
    free(J);
    free(W);

    return Jstet;
}

real_type
stetson_j_cpu(real_type *x, real_type *y, real_type *err, 
              const weight_type WEIGHTING, const int N){

    // scale y values by mean and variance
    real_type *delta = (real_type *) malloc(N * sizeof(real_type));
    real_type mu = STETSON_MEAN ? stetson_mean(y, err, APARAM, BPARAM, CRITERION, N)
                                : mean(y, N);
	
    make_delta(y, err, mu, delta, N);

    // get weighting function and parameters
    weight_function_t weight_func;
    void *params = NULL;
    size_t psize = 0;

    get_weighting_cpu(x, delta, N, WEIGHTING, &params, &weight_func, &psize);

    // compute J
    real_type J = stetson_j_kernel_cpu(x, delta, weight_func, 
    	                               params, N);

    // free weight function params
    if(params != NULL) 
        free(params);

    free(delta);
    return J;
}

real_type
stetson_k(real_type *y, real_type *err, const int N){
    real_type *delta = (real_type *) malloc(N * sizeof(real_type));
    real_type mu = STETSON_MEAN ? stetson_mean(y, err, APARAM, BPARAM,  CRITERION, N)
                                : mean(y, N);
	
    make_delta(y, err, mu, delta, N);

    real_type Sabs = 0, Ssq = 0;
    for(int i = 0; i < N; i++){
	   Sabs += abs(delta[i]);
	   Ssq  += delta[i] * delta[i];
    }

    free(delta);
    return (Sabs / N) / sqrt(Ssq / N);
}

#include "stetson.h"
#include "stetson_kernel.h"
#include "utils.h"
#include "weighting.h"


#define CUDA_CALL(x) x

// z-scale the data
void 
make_delta(real_type *y, real_type *err, real_type *delta, int N) {
	real_type mu = mean(y, N);
	int i;

	real_type bias = sqrt(N / ((real_type) (N - 1)));
	for(i = 0; i < N; i++) 
		delta[i] = bias * (y[i] - mu) / err[i];
}


// compute Stetson J index (Stetson 1996)
real_type 
stetson_j_gpu(real_type *x, real_type *y, real_type *err, 
	          weight_type WEIGHTING, int N){

	// necessary?
	cudaDeviceSynchronize();

	// WEIGHTING FUNCTIONS
	// get references to weighting functions
	weight_function_gpu_t weight_func;
	void *d_params = NULL;

	if (WEIGHTING & EXP) {
		CUDA_CALL(
			cudaMemcpyFromSymbol(&weight_func, p_exp_weighting_gpu, 
		                        sizeof(weight_function_gpu_t) )
		);

		// set up parameters on host
		real_type *h_params = (real_type *) malloc(2 * sizeof(real_type));

		// get array of t_i - t_{i-1}
		real_type dt[N-1];
		for(int i = 0; i < N - 1; i++) 
			dt[i] = x[i+1] - x[i];

		// param 1: inverse of mean(dt)
		h_params[0] = 1./mean(dt, N-1);

		// param 2: exponent n: exp(-|t2 - t1|^n / dt^n)
		h_params[1] = 1;

		
		CUDA_CALL(cudaMemcpy(&d_params, h_params, 
			                       2 * sizeof(real_type), 
			                       cudaMemcpyHostToDevice ));

		free(h_params);
	}
	else if (WEIGHTING & CONSTANT) {
		CUDA_CALL(
			cudaMemcpyFromSymbol(&weight_func, p_constant_weighting_gpu, 
		                        sizeof(weight_function_gpu_t) )
		);
	}
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
stetson_j_cpu(real_type *y, real_type *err, int N) {
	real_type *delta = (real_type *) malloc(N * sizeof(real_type));
	make_delta(y, err, delta, N);
	
	return stetson_j_kernel_cpu(delta, N);
}
/*
real_type
stetson_k(real_type *y, real_type *err, int N){


}
*/
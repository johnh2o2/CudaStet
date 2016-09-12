#include "stetson_kernel.h"



__host__ real_type
stetson_j_kernel_cpu(real_type *delta, int N){
	real_type J = 0, W = 0;
	for (int i = 0; i < N; i++){
		for (int j = i; j < N; j++) {
			real_type d = delta[i] * delta[j];
			if (i == j) d -= 1;
			int s = 1;
			if (d < 0) s = -1;
			J += s * sqrt(s * d);
			W += 1;
		}
	}
	return J/W;
		
}



__global__ void
stetson_j_kernel(real_type *x, real_type *delta, real_type *J, 
	             real_type *W, weight_function_gpu_t w, 
	             void *weight_params, int N){

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
    	J[i] = 0;
    	W[i] = 0;
		for(int j = i; j < N; j++){
			real_type d = delta[j] * delta[i];
			if (i == j)
				d -= 1;

			real_type s = 1.0;
			if (d < 0)
				s = -1;
		
			J[i] += (*w)(x[i], x[j], weight_params) * s * sqrt(s * d);
			W[i] += (*w)(x[i], x[j], weight_params);
        }
    }
}


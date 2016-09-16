#include <stdlib.h>
#include <memory.h>
#include <math.h>

#include "utils.h"

// MATH UTILS
real_type
mean(real_type *x, const int N){
        real_type mu = 0;
        int i;
        for (i=0; i < N; i++) mu += x[i];
        return mu/N;
}

real_type
sum(real_type *x, const int N){
        int i;
        real_type s = 0;
        for (i = 0; i < N; i++) s += x[i];
        return s;
}

int 
comp(const void *p1, const void *p2){
	real_type v1 = *((real_type *) p1);
	real_type v2 = *((real_type *) p2);

	if (v1 < v2) return -1;
	else if (v1 > v2) return 1;
	else return 0;
}

real_type
median(real_type *x, const int N){
	// copy x
	real_type * xcopy = (real_type *) malloc(N * sizeof(real_type));
	memcpy(xcopy, x, N * sizeof(real_type));

	// sort copy
	qsort(xcopy, N, sizeof(real_type), comp);

	real_type med = xcopy[N/2];
	free(xcopy);

	// return middle element
	return med;
}


real_type
weighted_mean(real_type *x, real_type *w, 
              const int N){
        real_type mu = 0;
        for(int i = 0; i < N; i++)
                mu += w[i] * x[i];
        return mu;
}

real_type
maxval(real_type *x, const int N){
        real_type xmax = x[0];
        for(int i = 0; i < N; i++){
                if (x[i] > xmax)
                        xmax = x[i];
        }
        return xmax;
}

real_type
minval(real_type *x, const int N){
        real_type xmin = x[0];
        for(int i = 0; i < N; i++){
                if (x[i] < xmin)
                        xmin = x[i];
        }
        return xmin;
}

void
normalize(real_type *x, const int N){
        real_type S = sum(x, N);

        for(int i = 0; i < N; i++)
                x[i] /= S;
}

void
make_delta(real_type *y, real_type *err, const real_type mu, 
           real_type *delta, const int N){

        real_type bias = sqrt(N / ((real_type) (N - 1)));
        for (int i = 0; i < N; i++)
                delta[i] = bias * (y[i] - mu) / err[i];

}
/////////////////

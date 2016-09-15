#include "utils.h"

// MATH UTILS
real_type
mean(real_type *x, int N){
        real_type mu = 0;
        int i;
        for (i=0; i < N; i++) mu += x[i];
        return mu/N;
}

real_type
sum(real_type *x, int N){
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
median(real_type *x, int N){
	// copy x
	real_type * xcopy = (real_type *) malloc(N * sizeof(real_type));
	memcpy(xcopy, x, N * sizeof(real_type));

	// sort copy
	qsort(xcopy, N, sizeof(real_type), comp);

	// return middle element
	return xcopy[N/2];
}
/////////////////

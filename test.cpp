#include <stdlib.h>
#include <stdio.h>
#include <memory.h>
#include <math.h>
#include <time.h>

#include "stetson.h"

#define rmax 1000000
#define Random ((real_type) (rand() % rmax))/rmax
#define twopi (2.0 * 3.14159265358979323846)

// random normal 
real_type randnorm() {
    real_type u1 = Random;
    real_type u2 = 1.0 - u1;
    real_type r  = sqrt(-2 * log(u1));
    real_type th = twopi * u2;
    return r * cos(th);
}

// generates unequal timing array
real_type *
generateRandomTimes(int N) {
	real_type *x = (real_type *) malloc( N * sizeof(real_type));
	x[0] = 0.;
	for (int i = 1; i < N; i++)
		x[i] = x[i - 1] + Random;

	real_type xmax = x[N - 1];
	for (int i = 0; i < N; i++)
		x[i] = (x[i] / xmax) - 0.5;

	return x;
}

// generates a periodic signal
real_type *
generateSignal(real_type *x, real_type f, real_type phi, int N) {
	real_type *signal = (real_type *) malloc( N * sizeof(real_type));

	for (int i = 0; i < N; i++)
		signal[i] = cos((x[i] + 0.5) * f * twopi - phi) + randnorm();

	return signal;
}


// Do timing tests
void timing(int Nmin, int Nmax, int Ntests) {
	int      n, dN = (Nmax - Nmin) / Ntests;
	real_type    *x, *f, *err, *w;
	
	clock_t  start, dt;

	for (int i = 0; i < Ntests; i++) {
		n = Nmin + dN * i;
		
		// generate a signal.
		x = generateRandomTimes(n);
		f = generateSignal(x, 10., 0.5, n);

		err = (real_type *) malloc(n * sizeof(real_type));
		w   = (real_type *) malloc(n * sizeof(real_type));
		for (int j = 0; j < n; j++) {
			err[j] = 1.0;
			w[j]   = 1.0;
		}

		// calculate Stetson J 
		start = clock();
		//real_type J = 1.0;
		//J = StetsonJ(f, err, n);
		dt = clock() - start;
		
		// output
		printf("%-10d %-10.3e\n", n, ((real_type) dt / CLOCKS_PER_SEC));

		free(x); free(f); free(w); free(err);
	}
}

// simple test
void simple(int N, real_type f) {
	real_type *x, *y, *err;
	
	x = generateRandomTimes(N);
	y = generateSignal(x, f, 0., N);

	

	err = (real_type *) malloc(N * sizeof(real_type));
	for (int i = 0; i < N; i++) 
		err[i] = 1.0;
	

	// OUTPUT
	FILE *out;

	out = fopen("original_signal.dat", "w");
	for (int i = 0; i < N; i++)
		fprintf(out, "%e %e %e\n", x[i], y[i], err[i]);
	fclose(out);

	fprintf(stdout, "Stetson J -- CONSTANT WEIGHTING\n");
	fprintf(stdout, "CPU  : %e\n", stetson_j_cpu(x, y, err, CONSTANT, N));
	fprintf(stdout, "CUDA : %e\n", stetson_j_gpu(x, y, err, CONSTANT, N));
	
	fprintf(stdout, "Stetson J -- EXPONENTIAL WEIGHTING\n");
	fprintf(stdout, "CPU  : %e\n", stetson_j_cpu(x, y, err, EXP, N));
	fprintf(stdout, "CUDA : %e\n", stetson_j_gpu(x, y, err, EXP, N));

	
	free(x); free(y); free(err); 
}

int main(int argc, char *argv[]) {
	if (!((argc == 4 and argv[1][0] == 's') || (argc==5 and argv[1][0] == 't'))) {
		fprintf(stderr, "usage: [simple test]  (1) %s s <n>    <f>\n", argv[0]);
		fprintf(stderr, "       [timing test]  (2) %s t <nmin> <nmax> <ntests>\n", argv[0]);
		fprintf(stderr, "n      : number of data points\n");
		fprintf(stderr, "f      : angular frequency of signal\n");
		fprintf(stderr, "nmin   : Smallest data size\n");
		fprintf(stderr, "nmax   : Largest data size\n");
		fprintf(stderr, "ntests : Number of runs\n");
		exit(EXIT_FAILURE);
	}

	// initialize random number generator
	srand(time(NULL));

	if (argv[1][0] == 's')
		simple(atoi(argv[2]), atof(argv[3]));

	else if (argv[1][0] == 't')
		timing(atoi(argv[2]), atoi(argv[3]), atoi(argv[4]));

	else {
		fprintf(stderr, "What does %c mean? Should be either 's' or 't'.\n", argv[1][0]);
		exit(EXIT_FAILURE);
	}

	return EXIT_SUCCESS;
}


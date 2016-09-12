#include <stdlib.h>
#include <stdio.h>
#include <memory.h>
#include <math.h>
#include <time.h>

#include "stetson.h"

#define rmax 1000000
#define Random ((float) (rand() % rmax))/rmax
#define twopi (2.0 * 3.14159265358979323846)

// random normal 
float randnorm() {
    float u1 = Random;
    float u2 = 1.0 - u1;
    float r  = sqrt(-2 * log(u1));
    float th = twopi * u2;
    return r * cos(th);
}

// generates unequal timing array
float *
generateRandomTimes(int N) {
	float *x = (float *) malloc( N * sizeof(float));
	x[0] = 0.;
	for (int i = 1; i < N; i++)
		x[i] = x[i - 1] + Random;

	float xmax = x[N - 1];
	for (int i = 0; i < N; i++)
		x[i] = (x[i] / xmax) - 0.5;

	return x;
}

// generates a periodic signal
float *
generateSignal(float *x, float f, float phi, int N) {
	float *signal = (float *) malloc( N * sizeof(float));

	for (int i = 0; i < N; i++)
		signal[i] = cos((x[i] + 0.5) * f * twopi - phi) + randnorm();

	return signal;
}


// Do timing tests
void timing(int Nmin, int Nmax, int Ntests) {
	int      n, dN = (Nmax - Nmin) / Ntests;
	float    *x, *f, *err, *w;
	float J;
	clock_t  start, dt;

	for (int i = 0; i < Ntests; i++) {
		n = Nmin + dN * i;
		
		// generate a signal.
		x = generateRandomTimes(n);
		f = generateSignal(x, 10., 0.5, n);

		err = (float *) malloc(n * sizeof(float));
		w   = (float *) malloc(n * sizeof(float));
		for (int j = 0; j < n; j++) {
			err[j] = 1.0;
			w[j]   = 1.0;
		}

		// calculate Stetson J 
		start = clock();
		//J = StetsonJ(f, err, n);
		dt = clock() - start;
		
		// output
		printf("%-10d %-10.3e\n", n, ((float) dt / CLOCKS_PER_SEC));

		free(x); free(f); free(w); free(err);
	}
}

// simple test
void simple(int N, float f) {
	float *x, *y, *err;
	
	x = generateRandomTimes(N);
	y = generateSignal(x, f, 0., N);

	

	err = (float *) malloc(N * sizeof(float));
	for (int i = 0; i < N; i++) 
		err[i] = 1.0;
	

	// OUTPUT
	FILE *out;

	out = fopen("original_signal.dat", "w");
	for (int i = 0; i < N; i++)
		fprintf(out, "%e %e %e\n", x[i], y[i], err[i]);
	fclose(out);

	fprintf(stdout, "Stetson J (CUDA) : %e\n", stetson_j_gpu(x, y, err, 
																CONSTANT, N));
	fprintf(stdout, "Stetson J (CPU)  : %e\n", stetson_j_cpu(y, err, N));
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


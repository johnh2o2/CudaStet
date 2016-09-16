#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <memory.h>
#include "utils.h"
#include "stetson_mean.h"

void
stetson_adjust_weights(real_type *delta, real_type *win,
                       real_type *wout, const real_type a, const real_type b,
                       const int N){

        for(int i = 0; i < N; i++){
                real_type m = 1.0 / ( 1 + pow(abs(delta[i]) / a, b));
                wout[i] = win[i] * m;
        }
        normalize(wout, N);
}

int
is_stable(real_type *x0, real_type *x, const int N, 
          const real_type criterion){
        real_type *dx = (real_type *)malloc(N * sizeof(real_type));

        for(int i = 0; i < N; i++) 
            dx[i] = x[i] - x0[i];

        real_type mv = maxval(dx, N);
        free(dx);

        if (mv < criterion)
                return 1;
        else
                return 0;        
}

real_type
stetson_mean(real_type *y, real_type *yerr, const real_type a,
             const real_type b, const real_type criterion, const int N){

        real_type *weights = (real_type *)malloc(N * sizeof(real_type));
        real_type *new_weights = (real_type *)malloc(N * sizeof(real_type));
        real_type *delta = (real_type *)malloc(N * sizeof(real_type));

        // initialize weights as 1 / err (ignore points for which err = 0)
        for (int i = 0; i < N; i++)
                weights[i] = yerr[i] == 0 ? 0. : 1.0/yerr[i];

        // normalize
        normalize(weights, N);

        int done = 0;
        do {
                real_type mu = weighted_mean(y, weights, N);

                // convert to delta
                make_delta(y, yerr, mu, delta, N);

                // correct weights
                stetson_adjust_weights(delta, weights, new_weights,
                                       a, b, N);

                // test if weights or mean changed at all
                done = (is_stable(weights, new_weights, N, criterion) &&
                         abs(weighted_mean(y, new_weights, N) - mu) / mu < criterion);

                // set weights <- new_weights
                memcpy(weights, new_weights, N * sizeof(real_type));

        } while (!done);

        real_type mu = weighted_mean(y, weights, N);

        free(weights); 
        free(new_weights); 
        free(delta);

        return mu;
}


#ifndef STETSON_MEAN_H
#define STETSON_MEAN_H
#include "config.h"
// default parameters
#define APARAM 2.0
#define BPARAM 2.0
#define MAX_ITER 1000
#define CRITERION 1E-3

void
stetson_adjust_weights(real_type *delta, real_type *win,
                       real_type *wout, const real_type a, const real_type b,
                       const int N);

int
is_stable(real_type *x0, real_type *x, const int N, 
          const real_type criterion);

real_type
stetson_mean(real_type *y, real_type *yerr, const real_type a,
             const real_type b, const real_type criterion, const int N);

#endif

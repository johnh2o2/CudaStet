
#ifndef UTILS_H
#define UTILS_H
#include "config.h"

real_type 
mean(real_type *x, const int N);

real_type 
sum(real_type *x, const int N);

real_type
median(real_type *x, const int N);

real_type
weighted_mean(real_type *x, real_type *w,
              const int N);

real_type
maxval(real_type *x, const int N);

real_type
minval(real_type *x, const int N);

void
normalize(real_type *x, const int N);

void
make_delta(real_type *y, real_type *err,
           const real_type mu, real_type *delta, const int N);
#endif

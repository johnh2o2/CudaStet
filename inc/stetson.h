#ifndef STETSON_H
#define STETSON_H

#include "config.h"

enum weight_type {
  EXP = 0x01,
  CONSTANT = 0x02,
  EXPXY = 0x04
};

real_type 
stetson_j_gpu(real_type *x, real_type *y, real_type *err, 
              const weight_type WEIGHTING, const int N);

real_type *
stetson_j_gpu_batch(real_type *x, real_type *y, real_type *err, 
              const weight_type WEIGHTING, int *N, const int Nsamples);

real_type
stetson_j_cpu(real_type *x, real_type *y, real_type *err, 
              const weight_type WEIGHTING, const int N);

real_type
stetson_k(real_type *y, real_type *err, const int N);
#endif

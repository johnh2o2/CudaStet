#ifndef STETSON_H
#define STETSON_H

enum weight_type {
  EXP = 0x01,
  CONSTANT = 0x02,
};

real_type 
stetson_j_gpu(real_type *x, real_type *y, real_type *err, 
              const weight_type WEIGHTING, const int N);

real_type
stetson_j_cpu(real_type *x, real_type *y, real_type *err, 
              const weight_type WEIGHTING, const int N);

real_type
stetson_k(real_type *y, real_type *err, const int N);
#endif

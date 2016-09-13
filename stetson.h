#ifndef STETSON_H
#define STETSON_H


enum weight_type {
  EXP = 0x01,
  CONSTANT = 0x02,
};

void 
make_delta(real_type *y, real_type *err, real_type *delta, int N);

real_type 
stetson_j_gpu(real_type *x, real_type *y, real_type *err, 
              weight_type WEIGHTING, int N);

real_type
stetson_j_cpu(real_type *x, real_type *y, real_type *err, 
              weight_type WEIGHTING, int N);
#endif

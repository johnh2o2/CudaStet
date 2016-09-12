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
/////////////////

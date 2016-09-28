%module cudastet
%include typemaps.i

%{
#include "config.h"
#include "stetson.h"
#include <string.h>
%}

%include "config.h"
%include "stetson.h"


%inline %{

real_type *get_real_type_array(int n){
        real_type *x = (real_type *)malloc(n * sizeof(real_type));
        return x;
}
real_type get_val(real_type *x,int i){
        return x[i];
}
void set_val(real_type *x, int i, real_type val){
        x[i] = val;
}
void free_real_type_array(real_type *arr){
        free(arr);
}


int *get_int_array(int n){
        int *x = (int *)malloc(n * sizeof(int));
        return x;
}
int get_ival(int *x,int i){
        return x[i];
}
void set_ival(int *x, int i, int val){
        x[i] = val;
}
void free_int_array(int *n){
        free(n);
}

weight_type get_weight_type(const char *s){
        if (strcmp(s, "constant") == 0)
                return CONSTANT;
        else if (strcmp(s, "exp") == 0)
                return EXP;
        else if (strcmp(s, "expxy") == 0)
                return EXPXY;
        else{
                fprintf(stderr, "I don't know what '%s' means.\n", s);
                exit(EXIT_FAILURE);
        }
}       

%}



%pythoncode %{
from time import time


def _convert_real_type_to_c(arr):
    N = len(arr);
    carr = _cudastet.get_real_type_array(N);
    for i, val in enumerate(arr):
        _cudastet.set_val(carr, i, val)
    return carr

def _convert_int_to_c(arr):
    N = len(arr);
    carr = _cudastet.get_int_array(N);
    for i, val in enumerate(arr):
        _cudastet.set_ival(carr, i, int(val))
    return carr

def _convert_real_type_to_py(carr, N):
    return [ _cudastet.get_val(carr, i) for i in range(N) ]

def _convert_int_to_py(carr, N):
    return [ _cudastet.get_ival(carr, i) for i in range(N) ]

def _insettings(arr, settings):
    return [ v in settings for v in arr ]

def _convert_weighting_to_c(w):
    return _cudastet.get_weight_type(w)


def _convert_multiple_real_type_arrays(*args):
        carrs = [ _convert_real_type_to_c(a) for a in args ]
        return tuple(carrs)

def _free_multiple_real_type_arrays(*args):
        for arr in args:
                _cudastet.free_real_type_array(arr)

def _flatten_batch(*args):
        N = [ len(a) for a in args[0] ]
        
        flattened_arrs = []

        for a in args:
                A = []
                for av in a: 
                        A.extend(av)
                flattened_arrs.append(A)

        flattened_arrs.append(N)
        return tuple(flattened_arrs)

import sys

def stetson_j(t, x, err, weighting="constant", use_gpu=True):

        weighting = _cudastet.get_weight_type(weighting)
       
        is_batch = isinstance(t, list) and hasattr(t[0], '__getitem__')

        J = None
        
        if is_batch:
                if use_gpu:
                        T, X, ERR, N = _flatten_batch(t, x, err)
                        
                        nb = len(t)
                        _t, _x, _err = _convert_multiple_real_type_arrays(T, X, ERR)
                        _n = _convert_int_to_c(N)
                        _J = _cudastet.stetson_j_gpu_batch(_t, _x, _err,
                                          weighting, _n, nb)
                
                
                        J = _convert_real_type_to_py(_J, nb)
                        _free_multiple_real_type_arrays(_t, _x, _err, _J)
                        _cudastet.free_int_array(_n)
                else:
                        J = [ ]
                        for T, X, ERR in zip(t, x, err):
                                _t, _x, _err = _convert_multiple_real_type_arrays(T, X, ERR)
                                J.append(_cudastet.stetson_j_cpu(_t, _x, _err, weighting, len(T)))
                                _free_multiple_real_type_arrays(_t, _x, _err)

        else:                
                _t, _x, _err = _convert_multiple_real_type_arrays(t, x, err)
        
                J = _cudastet.stetson_j_gpu(_t, _x, _err, weighting, len(x)) if use_gpu\
                        else _cudastet.stetson_j_cpu(_t, _x, _err, weighting, len(x))

                _free_multiple_real_type_arrays(_t, _x, _err)

        return J 

def stetson_k(y, err):
        is_batch = isinstance(y, list) and hasattr(y[0], '__getitem__')
        
        K = None
        if is_batch:
                K = []
                for Y, ERR in zip(y, err):
                        n = len(Y)
                        _y, _err = _convert_multiple_real_type_arrays(Y, ERR)
                        K.append( _cudastet.stetson_k(_y, _err, n) )
                        
                        _free_multiple_real_type_arrays(_y, _err)
        else:
                _y = _convert_real_type_to_c(y)
                _err = _convert_real_type_to_c(err)

                K = _cudastet.stetson_k(_y, _err, len(y))

                _free_multiple_real_type_arrays(_y, _err)

        return K
%}

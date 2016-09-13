CudaStet
========


Provides both CPU and GPU functions for the Stetson J statistic ([Stetson 1996](http://adsabs.harvard.edu/abs/1996PASP..108..851S)).

Requires:
	
* [CUDA](http://www.nvidia.com/object/cuda_home_new.html)-enabled [device](https://developer.nvidia.com/cuda-gpus)
* [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) >= 7.5
* A suitable C++ compiler (e.g. `g++`)

To install:
-----------

1. Clone this repository
2. Edit the Makefile accordingly
	
	* `ARCH` -- the Compute Capability number for your device, e.g. 5.2 becomes `52`
	* `REAL_TYPE` -- you can switch between double and single precision by editing this Makefile variable to either `double` or `float`.
	* `CUDA_VERSION` -- make sure you have the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) installed; this code was developed using version 7.5
	* The `BLOCK_SIZE` is also something to play around with but is optional to change.

3. Run `make` to generate a binary that you can run for testing. This is mainly for my own debugging purposes, but you can inspect the code to get an idea of how to call the `stetson_j_cpu` and `stetson_j_gpu` functions.

To call `stetson_j_{gpu, cpu}`:
-------------------------------
Arguments (in order):
	
* `real_type *x` : values for independent variable
* `real_type *y` : values for dependent variable
* `real_type *err`	: values for uncertainty of dependent variable
* `weight_type WEIGHTING` : Must be one of
	* `CONSTANT` : all pairs of observations are weighted equally
	* `EXP` : pairs of observations are exponentially suppressed by their distance in `x` (uses `exp(-|t1 - t2| / mean(dt))`). See [Zhang _et. al._ 2003](http://adsabs.harvard.edu/abs/2003ChJAA...3..151Z) for a real-world application of this weighting scheme.
* `int N` : number of datapoints

Notes
-----

* Be careful with single precision. If you're using the exponential
  weighting scheme, the accuracy for ~50,000 datapoints is about 10^(-3) for both the CPU and GPU variants. For a constant weighting
  scheme, however, the CPU variant is very inaccurate (off by factors of 100 or more) at 50,000 datapoints, while the GPU variant remains accurate to a factor of 10^(-3).


Goals for the future
--------------------

* Python bindings with Swig
* Addition of Stetson K and L variability indices
* Adding configure script to simplify the install process

Timing 
------

Some tests run on an Ubuntu 14.04 desktop with an i7-5930K overclocked to 4.5GHz, and a 980 Ti graphics card. For these 
timing tests, `CudaStet` was compiled with single precision 
and using the `-O3` optimization flag for `g++` and 
`--use_fast_math` for `nvcc`.

Read as: 
`N`    `dt`

Where `N` is the number of data points, `dt` is the execution
time in seconds. 

```
timing : (CPU, EXPONENTIAL)
10         9.000e-06 
1009       3.612e-02 
2008       9.482e-02 
3007       1.367e-01 
4006       1.922e-01 
5005       2.986e-01 
6004       4.134e-01 
7003       4.819e-01 
8002       6.265e-01 
9001       7.888e-01 
timing : (GPU, EXPONENTIAL)
10         1.648e-01 
1009       1.351e-03 
2008       2.490e-03 
3007       3.450e-03 
4006       4.739e-03 
5005       5.822e-03 
6004       7.672e-03 
7003       8.494e-03 
8002       9.271e-03 
9001       1.066e-02 
timing : (CPU, CONSTANT)
10         1.000e-06 
1009       5.004e-03 
2008       1.877e-02 
3007       3.833e-02 
4006       6.095e-02 
5005       9.584e-02 
6004       1.385e-01 
7003       1.869e-01 
8002       2.532e-01 
9001       2.829e-01 
timing : (GPU, CONSTANT)
10         1.740e-04 
1009       9.020e-04 
2008       1.727e-03 
3007       3.200e-03 
4006       4.346e-03 
5005       4.290e-03 
6004       5.602e-03 
7003       7.210e-03 
8002       7.446e-03 
9001       8.685e-03 

```
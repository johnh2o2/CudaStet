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

Goals for the future
--------------------

* Python bindings with Swig
* Addition of Stetson K and L variability indices
* Adding configure script to simplify the install process
* 

Timing 
------

Some tests run on an Ubuntu 14.04 desktop with an i7-5930K overclocked to 4.5GHz, and a 980 Ti graphics card.

```
timing : (CPU, EXPONENTIAL)
10         1.900e-05 
2009       1.835e-01 
4008       4.209e-01 
6007       8.432e-01 
8006       1.347e+00 
10005      2.102e+00 
12004      3.018e+00 
14003      4.099e+00 
16002      5.332e+00 
18001      6.754e+00 
timing : (GPU, EXPONENTIAL)
10         1.745e-01 
2009       3.565e-03 
4008       6.760e-03 
6007       1.043e-02 
8006       1.407e-02 
10005      1.752e-02 
12004      2.087e-02 
14003      2.473e-02 
16002      2.594e-02 
18001      2.924e-02 
timing : (CPU, CONSTANT)
10         2.000e-06 
2009       4.124e-02 
4008       1.665e-01 
6007       3.707e-01 
8006       5.630e-01 
10005      8.821e-01 
12004      1.269e+00 
14003      1.717e+00 
16002      2.254e+00 
18001      2.883e+00 
timing : (GPU, CONSTANT)
10         1.760e-04 
2009       1.931e-03 
4008       3.807e-03 
6007       5.948e-03 
8006       7.720e-03 
10005      1.010e-02 
12004      1.249e-02 
14003      1.454e-02 
16002      1.603e-02 
18001      1.870e-02 
```
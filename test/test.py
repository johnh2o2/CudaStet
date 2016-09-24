import cudastet
import stetson
import numpy as np
from math import *
from time import time
import sys

N = 1000
Nlcs = 3

weight_types = [ 'constant', 'exp', 'expxy' ]

X = [ np.linspace(0, 2 * np.pi, N) for i in range(Nlcs) ]
Y = [ np.cos(x) + np.random.normal(size=len(x)) for x in X ]
ERR = [ 1 * np.ones(len(x)) for x in X ]
print ERR[0][0]


# GPU TESTS
print "simple GPU stetson test (one lightcurve)"
for weight in weight_types:
	print weight
	for x, y, err in zip(X, Y, ERR):
		t0 = time()
		J = cudastet.stetson_j(x, y, err, weighting=weight)
		print "%-20s J = %e, dt = %e seconds"%(weight, J, time() - t0)

print "batch GPU stetson test (multiple lightcurves)"
for weight in weight_types:
	print weight
	t0 = time()
	J = cudastet.stetson_j(X, Y, ERR, weighting=weight)
	dt = time() - t0
	for j in J:
		print "%-20s (b) J = %e, dt = %e seconds"%(weight, j, dt)
	

# CPU tests
print "simple CPU stetson test (one lightcurve)"
for weight in weight_types:
	print weight
	for x, y, err in zip(X, Y, ERR):
		t0 = time()
		J = cudastet.stetson_j(x, y, err, weighting=weight, use_gpu=False)
		print "%-20s J = %e dt = %e seconds"%(weight, J, time() - t0)
		if weight == 'constant':
			t0 = time()
			K = cudastet.stetson_k(y, err)
			print "%-20s K = %e dt = %e seconds"%(weight, K, time() - t0)


print "batch GPU stetson test (multiple lightcurves)"
for weight in weight_types:
	print weight
	t0 = time()
	J = cudastet.stetson_j(X, Y, ERR, weighting=weight, use_gpu=False)
	dt = time() - t0
	for j in J:
		print "%-20s (b) J = %e, dt = %e seconds"%(weight, j, dt/len(J))
	t0 = time()
	K = cudastet.stetson_k( Y, ERR )
	dt = time() - t0
	for k in K:
		print "%-20s (b) K = %e, dt = %e seconds"%(weight, k, dt/len(K))

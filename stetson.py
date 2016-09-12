import numpy as np
import sys
from math import *

def stetson_j(y, err):
	bias = sqrt(len(y) / (float(len(y)) - 1))
	delta = bias * np.multiply(y - np.mean(y),np.power( err, -1))
	
	J, W = 0., 0.
	for i in range(len(y)):
		for j in range(i, len(y)):
			Pk = delta[i] * delta[j]
			if i == j: Pk -= 1.
			W += 1.
			J += np.sign(Pk) * sqrt(abs(Pk))
	return J / W


if __name__ == '__main__':
	Lc = np.loadtxt(sys.argv[1], 
		dtype=np.dtype([ ('t', float), 
			  ('y', float), ('err', float) ]))
	
	print 'Stetson J (Pyth) : %e' %( stetson_j(Lc['y'], 
	                                          Lc['err']) )
	

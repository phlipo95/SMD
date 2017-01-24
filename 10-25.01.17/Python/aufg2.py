def aufg2():
	import numpy as np

	M = np.array([31.6,32.3,31.2,31.9,31.2,30.8,31.3])

	def altCalcChi2(x, proof):
	    Sum = np.zeros(1)
	    for i in x:
	        Sum += ((i - proof)**2) / (abs(x.mean() - proof)**2)
	    return Sum

	a = altCalcChi2(M,31.3)
	b = altCalcChi2(M,30.7)
	print(a)
	print(b)


if __name__ == '__main__':     
    aufg2()

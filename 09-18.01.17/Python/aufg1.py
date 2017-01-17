def aufg1():
	import numpy as np
	from matplotlib import pyplot as plt
	y = np.linspace(4,17.5,100)
	x=[13,8,9]
	def L(y, x):
	    return -np.log(y)*np.sum(x)+len(x)*y

	LMin=L(10,x)
	print(LMin)
	def TayL(y):
	    return LMin + 3/10*(y-10)**2
	get_ipython().magic('pinfo np.roots')
	plt.plot(y,TayL(y), label='Taylor')
	plt.plot(y,L(y,x),'k', label='Likelihood')
	plt.plot((min(y),max(y)),(LMin+1/2,LMin+1/2),'b--')
	plt.plot((min(y),max(y)),(LMin+2,LMin+2),'r--')
	plt.plot((min(y),max(y)),(LMin+9/2,LMin+9/2),'g--')
	plt.legend()
	plt.grid()
	plt.savefig('TaskA.pdf')
	plt.show()

if __name__ == '__main__':     
    aufg1()

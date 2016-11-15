def aufg1():

	#Aufgabe 1c
	def Z(E, M):
		return (1+M/2**23) * 2**(E-127)
	z = Z(0, 1)
	print(z)

	a = 1/(2*z)
	print('a =', a)
	print(a*z)

	b = 2/(3*z)
	print('b =', b)
	print(b*z)

if __name__ == '__main__':
	aufg1()

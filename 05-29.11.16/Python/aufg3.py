from matplotlib import pyplot as plt
import numpy as np

lam = 1/np.sqrt(185)*np.matrix([-8, 0, 11])
#Population
P0 = np.matrix([[2, 2, 2, 1, 3], [2, 3, 1, 2, 2], [1, 2, 2, 0, 0]])
P1 = np.matrix([[2.5, 2.5, 4, 5.5, 5.5], [2.5, 1.5, 2, 2.5, 1.5], [0, 0, 0, 0, 0]])
#Projizierte Population
P0p = lam * P0
P1p = lam * P1
#Projektionsgerade
Proj = 1/np.sqrt(185)* np.linspace(-45, 7, 10)

#Aufgabe 3 d
plt.plot(P0[0, :], P0[2, :], 'rx', label='Population 0')
plt.plot(lam[0,0]*P0p, lam[0,2]*P0p, 'ro', label='$S_{0,proj}$')
plt.plot(P1[0, :], P1[2, :], 'bx', label='Population 1')
plt.plot(lam[0,0]*P1p, lam[0,2]*P1p, 'bo', label='$S_{1,proj}$')
plt.plot(lam[0,0]*Proj, lam[0,2]*Proj, 'g-', label='Projektionsgerade')
plt.xlabel('x')
plt.ylabel('z')
plt.grid()
plt.legend(loc="best")
plt.savefig('Aufgabe3d.pdf')
plt.close()


#Aufgabe 3 e
lamcut = -12/np.sqrt(185)

plt.plot(lam[0,0]*P0p, lam[0,2]*P0p, 'ro', label='$S_{0,proj}$')
plt.plot(lam[0,0]*P1p, lam[0,2]*P1p, 'bo', label='$S_{1,proj}$')
plt.plot(lam[0,0]*Proj, lam[0,2]*Proj, 'g-', label='$\lambda$')
plt.plot(lamcut*lam[0,0], lamcut*lam[0,2], 'yo', label='$\lambda_{cut}$')
plt.xlabel('x')
plt.ylabel('z')
plt.grid()
plt.legend(loc="best")
plt.savefig('Aufgabe3e.pdf')
plt.close()

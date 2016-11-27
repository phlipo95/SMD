from matplotlib import pyplot as plt
import numpy as np

lam = 1/np.sqrt(185)*np.array([-8, 0, 11])
#P0 = P0 * lambda
P0 = 1/np.sqrt(185)* np.array([-5, 6, 6, -8, -24])
#P1 = P1 * lambda
P1 = 1/np.sqrt(185)* np.array([-12, -12, -32, -44, -44])
#Projektionsgerade
Proj = 1/np.sqrt(185)* np.linspace(-45, 7, 10)

plt.plot(lam[0]*P0, lam[2]*P0, 'ro', label='Population 0')
plt.plot(lam[0]*P1, lam[2]*P1, 'bo', label='Population 1')
plt.plot(lam[0]*Proj, lam[2]*Proj, 'g-', label='Projektionsgerade')
plt.xlabel('x')
plt.ylabel('z')
plt.grid()
plt.legend(loc="best")
plt.savefig('Aufgabe3d.pdf')
plt.close()

import ROOT
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#Aufgabe 2a
a = 1601; b = 3456; m = 10000
def randFunk(x): #Zufallsgenerator
    return (a * x + b)%m

xn = np.array([1])
for i in range(10000): #berechnet alle Zufallszahlen
    xn = np.append(xn, randFunk(xn[i]))
xn = xn/10000
print(xn[1])

#Aufagbe 2b
plt.hist(xn, bins=10)
plt.xlabel('Zufallszahl')
plt.ylabel('Anzahl')
plt.savefig('Python/Aufgabe2b.pdf')
plt.close()

#Aufgabe 2c
xn_x = xn[0:]
xn_y = np.append(xn[1:], xn[0])
xn_z = np.append(xn[2:], (xn[0], xn[1]))

plt.hist2d(xn_x, xn_y, bins=100)
plt.xlabel('$(x_i, x_{i+1})$')
plt.ylabel('$(x_{i+1}, x_{i+2})$')
plt.colorbar()
plt.savefig('Python/Aufgabe2c1.pdf')
plt.close()

plt.figure()
ax = plt.gca(projection='3d')
for i in range(10000):
    ax.scatter(xn_x[i], xn_y[i], xn_z[i], color='red')
plt.savefig('Python/Aufgabe2c2.pdf')
plt.close()

#Aufgabe 2d
for i in range(10000): #Periodizität
    if xn[i] == xn[0]:
        print(i)

#Aufgabe 2e
xn = ROOT.TRandom3()
xn = [xn.Rndm() for i in range(10000)]

xn_x = xn[0:]
xn_y = np.append(xn[1:], xn[0])
xn_z = np.append(xn[2:], (xn[0], xn[1]))

plt.hist(xn, bins=10)
plt.xlabel('Zufallszahl')
plt.ylabel('Anzahl')
plt.savefig('Python/Aufgabe2e1.pdf')
plt.close()

plt.hist2d(xn_x, xn_y, bins=1000)
plt.xlabel('$(x_i, x_{i+1})$')
plt.ylabel('$(x_{i+1}, x_{i+2})$')
plt.colorbar()
plt.savefig('Python/Aufgabe2e2.pdf')
plt.close()

plt.figure()
ax = plt.gca(projection='3d')
for i in range(10000):
    ax.scatter(xn_x[i], xn_y[i], xn_z[i], color='red')
plt.savefig('Python/Aufgabe2e3.pdf')
plt.close()

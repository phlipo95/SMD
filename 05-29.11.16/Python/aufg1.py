import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy.linalg import det

#Erwartungswerte
mu_x = 0
mu_y = 3
#Standardabweichung
sigma_x = 3.5
sigma_y = 2.6
#Korrelationskoeffizient bzw. Kovarianz
rho = 0.9
#Kovarianzmatrix
CovM = 1/(1-rho**2) * np.matrix([[1,-rho],[-rho,1]])
#Determinante der Kovarianzmatrix
detCovM = det(CovM)

#Gaußfunktion
def gauss(x, y):
	return 1/(2*np.pi*sigma_x*sigma_y*np.sqrt(1-rho**2)) * np.exp( -1/(2-2*rho**2)* (((x-mu_x)/sigma_x)**2 + ((y-mu_y)/sigma_y)**2 - (2*rho*(x-mu_x)*(y-mu_y)) / (sigma_x*sigma_y)))

#x- und y-Werte
n = 10**2 #Anzahl der Messwerte
x = np.linspace(mu_x-7, mu_x+7, n)
y = np.linspace(mu_y-7, mu_y+7, n)
#macht x und y 2D
X, Y = np.meshgrid(x, y)

#3D-Plot der Gaußfunktion
plt.figure()
ax = plt.gca(projection='3d')
ax.plot_surface(X, Y, gauss(X, Y), cmap = plt.get_cmap("jet"))
#plt.show()
plt.savefig('3dGauß.pdf')
plt.close()

#Maximalwert der Gaußfunktion durch sqrt(e)
maxGdurchE = gauss(mu_x, mu_y) / np.sqrt(np.e)

#2D-Plot der Gaußfunktion
plt.figure()
plt.pcolormesh(X, Y, gauss(X, Y))
plt.colorbar()
#plt.contour(X, Y, gauss(X, Y), np.linspace(maxGdurchE, maxGdurchE, 1), colors="k")
#plt.axvline(x=mu_x, color='r', label='$\mu_x$')
#plt.plot(mu_x+sigma_x, mu_y, 'ro', label='$\mu_x \pm \sigma_x$')
#plt.plot(mu_x-sigma_x, mu_y, 'ro')
#plt.axhline(y=mu_y, color='g', label='$\mu_y$')
#plt.plot(mu_x, mu_y+sigma_y, 'go', label='$\mu_y \pm \sigma_y$')
#plt.plot(mu_x, mu_y-sigma_y, 'go')
plt.legend(loc='best')
plt.gca().set_aspect("equal")
plt.xlabel('x', fontsize='14')
plt.ylabel('y', fontsize='14')
plt.savefig('2dGauß.pdf')
plt.close()

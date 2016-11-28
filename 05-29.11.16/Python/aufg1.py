import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#Gaußfunktion
def gauss(x, y , mu_x, mu_y, sigma_x, sigma_y, rho):
	return 1/(2*np.pi*sigma_x*sigma_y*np.sqrt(1-rho**2)) * np.exp(-1/(2-2*rho**2)* (((x-mu_x)/sigma_x)**2 + ((y-mu_y)/sigma_y)**2 - (2*rho*(x-mu_x)*(y-mu_y)) / (sigma_x*sigma_y)))

#Anzahl der Messwerte
n = 250
#Bereich für die x- und y-Werte
Bereich = 20
#Population 0 (P0)
#Erwartungswerte
mu_x0 = 0
mu_y0 = 3
#Standardabweichung
sigma_x0 = 3.5
sigma_y0 = 2.6
#Korrelationskoeffizient bzw. Kovarianz
rho0 = 0.9
#x- und y-Werte
x = np.linspace(mu_x0-Bereich, mu_x0+Bereich, n)
y = np.linspace(mu_y0-Bereich, mu_y0+Bereich, n)
#macht x und y 2D
X0, Y0 = np.meshgrid(x, y)

#3D-Plot der Gaußfunktion
plt.figure()
ax = plt.gca(projection='3d')
ax.plot_surface(X0, Y0, gauss(X0, Y0, mu_x0, mu_y0, sigma_x0, sigma_y0, rho0), cmap = plt.get_cmap("jet"))
#plt.show()
plt.savefig('P0.pdf')
plt.close()


#Population 1 (P1)
#Erwartungswerte
mu_x1 = 6
a = -0.5; b = 0.6
mu_y1 = b * mu_x1 + a #3.1
#Standardabweichung
sigma_x1 = 3.5
sigma_y1 = sigma_x1 * b #2.1
sigma_xy = 1
#Korrelationskoeffizient
rho1 = sigma_xy / (sigma_x1*sigma_y1) #0.136
#x- und y-Werte
x = np.linspace(mu_x1-Bereich, mu_x1+Bereich, n)
y = np.linspace(mu_y1-Bereich, mu_y1+Bereich, n)
#macht x und y 2D
X1, Y1 = np.meshgrid(x, y)

#3D-Plot der Gaußfunktion
plt.figure()
ax = plt.gca(projection='3d')
ax.plot_surface(X1, Y1, gauss(X1, Y1, mu_x1, mu_y1, sigma_x1, sigma_y1, rho1), cmap = plt.get_cmap("jet"))
#plt.show()
plt.savefig('P1.pdf')
plt.close()


#Scatterplot
plt.scatter(X0, Y0, gauss(X0, Y0, mu_x0, mu_y0, sigma_x0, sigma_y0, rho0)*25, marker='o', c=gauss(X0, Y0, mu_x0, mu_y0, sigma_x0, sigma_y0, rho0)*300, alpha=0.8)
plt.scatter(X1, Y1, gauss(X1, Y1, mu_x1, mu_y1, sigma_x1, sigma_y1, rho1)*25, marker='o', c=gauss(X1, Y1, mu_x1, mu_y1, sigma_x1, sigma_y1, rho1)*300, alpha=0.8)
plt.show()

plt.scatter(X0, gauss(X0, Y0, mu_x0, mu_y0, sigma_x0, sigma_y0, rho0), marker='o', color='b', alpha=0.3, label='P0')
plt.scatter(X1, gauss(X1, Y1, mu_x1, mu_y1, sigma_x1, sigma_y1, rho1), marker='o', color='r', alpha=0.3, label='P1')
plt.show()

plt.pcolormesh(X0, Y0, gauss(X0, Y0, mu_x0, mu_y0, sigma_x0, sigma_y0, rho0))
plt.colorbar()
plt.show()

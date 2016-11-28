import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from root_numpy import array2root, root2array

#Aufgabe a)
#Erwartungswerte
mu_x1 = 6
a = -0.5; b = 0.6
mu_y1 = b * mu_x1 + a #3.1
mu1 = np.array((mu_x1, mu_y1))
#Standardabweichung
sigma_x1 = 3.5
sigma_y1 = sigma_x1 * b #2.1
sigma_xy = 1
#Korrelationskoeffizient
rho1 = sigma_xy / (sigma_x1*sigma_y1) #0.136

#Gaußfunktion
def gauss(x, y , mu_x, mu_y, sigma_x, sigma_y, rho):
	return 1/(2*np.pi*sigma_x*sigma_y*np.sqrt(1-rho**2)) * np.exp(-1/(2-2*rho**2)* (((x-mu_x)/sigma_x)**2 + ((y-mu_y)/sigma_y)**2 - (2*rho*(x-mu_x)*(y-mu_y)) / (sigma_x*sigma_y)))

#x- und y-Werte
n = 500 #Anzahl der Messwerte
x = np.linspace(mu_x1-7, mu_x1+7, n)
y = np.linspace(mu_y1-7, mu_y1+7, n)
#macht x und y 2D
X1, Y1 = np.meshgrid(x, y)

#3D-Plot der Gaußfunktion
plt.figure()
ax = plt.gca(projection='3d')
ax.plot_surface(X1, Y1, gauss(X1, Y1, mu_x1, mu_y1, sigma_x1, sigma_y1, rho1), cmap = plt.get_cmap("jet"))
plt.show()



#Aufgabe b)
#Population 0 (P0)
#Erwartungswerte
mu_x0 = 0
mu_y0 = 3
mu0 = np.array((mu_x0, mu_y0))
#Standardabweichung
sigma_x0 = 3.5
sigma_y0 = 2.6
#Korrelationskoeffizient bzw. Kovarianz
rho0 = 0.9
cov0 = rho0*sigma_x0*sigma_y0
#Kovarianzmatrix
C0 = np.array(((sigma_x0**2, cov0), (cov0, sigma_y0**2)))
#Gaußverteilte Zufallszahlen
normal0 = np.random.multivariate_normal(mu0, C0, 10000)

#Population 1 (P1)
#Erwartungswerte
mu_x1 = 6
a = -0.5; b = 0.6
mu_y1 = b * mu_x1 + a #3.1
mu1 = np.array((mu_x1, mu_y1))
#Standardabweichung
sigma_x1 = 3.5
sigma_y1 = sigma_x1 * b #2.1
sigma_xy = 1
#Korrelationskoeffizient
rho1 = sigma_xy / (sigma_x1*sigma_y1) #0.136
cov1 = sigma_xy
#Kovarianzmatrix
C1 = np.array(((sigma_x1**2, cov1), (cov1, sigma_y1**2)))
#Gaußverteilte Zufallszahlen
normal1 = np.random.multivariate_normal(mu1, C1, 10000)

x = np.linspace(-15, 15, 100)
plt.scatter(normal0[:,0], normal0[:,1], 1, c='b', label='P0')
plt.plot(x+mu_x0, rho0*x+mu_y0, 'y-', label=r'$\rho = 0.9$')
plt.scatter(normal1[:,0], normal1[:,1], 1, c='r', label='P1')
plt.plot(x+mu_x1, rho1*x + mu_y1, 'g-', label=r'$\rho = 0.136$')
plt.legend(loc='best')
plt.show()

#Aufgabe d)
normal1000 = np.random.multivariate_normal(mu0, C0, 1000)
#Array mit Namen der Branches
P0_1000 = np.empty((len(normal1000)), dtype=[("x", np.float), ("y", np.float)])
#fügt dem Array die Werte hinzu
P0_1000["x"] = normal1000[:,0]
P0_1000["y"] = normal1000[:,1]
#speichert Array als .root
array2root(P0_1000, "zwei_populationen.root",
           treename="P_0_1000")

#Array mit Namen der Branches
P0_10000 = np.empty((len(normal0)), dtype=[("x", np.float), ("y", np.float)])
#fügt dem Array die Werte hinzu
P0_10000["x"] = normal0[:,0]
P0_10000["y"] = normal0[:,1]
#speichert Array als .root
array2root(P0_10000, "zwei_populationen.root",
           treename="P_0_10000")

#Array mit Namen der Branches
P1 = np.empty((len(normal1)), dtype=[("x", np.float), ("y", np.float)])
#fügt dem Array die Werte hinzu
P1["x"] = normal1[:,0]
P1["y"] = normal1[:,1]
#speichert Array als .root
array2root(P1, "zwei_populationen.root",
           treename="P1")

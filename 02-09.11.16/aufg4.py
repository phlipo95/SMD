def aufg4():
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from numpy.linalg import det

    #Aufgabe4a
    sigma_x = 3.5
    sigma_y = 1.5
    Cov = 4.2
    rho = Cov/(sigma_x*sigma_y) #Korrelationskoeffizient

    #Aufgabe4b
    #Erwartungswerte
    mu_x = 4
    mu_y = 2
    #Standardabweichung
    sigma_x = 3.5
    sigma_y = 1.5
    #x- und y-Werte
    n = 10**2 #Anzahl der Messwerte
    x = np.linspace(mu_x-7, mu_x+7, n)
    y = np.linspace(mu_y-7, mu_y+7, n)
    #Kovarianz
    Cov = 4.2
    rho = Cov/(sigma_x*sigma_y)
    #Kovarianzmatrix
    CovM = 1/(1-rho**2) * np.matrix([[1,-rho],[-rho,1]])
    #Determinante der Kovarianzmatrix
    detCovM = det(CovM)

    #2D-Gaußfunktion
    def gauss(x, y):
    	return 1/(2*np.pi*sigma_x*sigma_y) * np.e**(-0.5*(((x-mu_x)/sigma_x)**2 + ((y-mu_y)/sigma_y)**2))

    x, y = np.meshgrid(x, y)

    plt.figure()
    ax = plt.gca(projection='3d')
    ax.plot_surface(x, y, gauss(x, y), cmap = plt.get_cmap("jet"))
    plt.savefig('build/3dGauß.pdf')
    plt.close()

    #Maximalwert der Gaußfunktion durch Wurzel(e)
    maxGdurchE = gauss(mu_x, mu_y) / np.sqrt(np.e)
    #Aufgabe 4b und 4c
    plt.figure()
    plt.pcolormesh(x, y, gauss(x,y))
    plt.colorbar()
    plt.contour(x, y, gauss(x,y), np.linspace(maxGdurchE, maxGdurchE, 1), colors="k")
    plt.axvline(x=mu_x, color='r', label='$\mu_x$')
    plt.plot(mu_x+sigma_x, mu_y, 'ro', label='$\mu_x \pm \sigma_x$')
    plt.plot(mu_x-sigma_x, mu_y, 'ro')
    plt.axhline(y=mu_y, color='g', label='$\mu_y$')
    plt.plot(mu_x, mu_y+sigma_y, 'go', label='$\mu_y \pm \sigma_y$')
    plt.plot(mu_x, mu_y-sigma_y, 'go')
    plt.legend(loc='best')
    plt.gca().set_aspect("equal")
    plt.xlabel('x', fontsize='14')
    plt.ylabel('y', fontsize='14')
    plt.savefig('./build/2dGauß.pdf')
    plt.close()
if __name__ == '__main__':
    aufg4()

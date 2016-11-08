def aufg4():
    import matplotlib.pyplot as plt
    import numpy as np
    from numpy.linalg import inv, det

    #Startwerte
    myx = 4; myy = 2; sigmax = 3.5; sigmay = 1.5 
    
    #2dim Gaußfunktion
    def phi(x1,x2):
        return 1/(2*np.pi*sigmax*sigmay)*np.e**(-1/(2*sigmax**2)*(x1-myx)**2)*np.e**(-1/(2*sigmay**2)*(x2-myy)**2)

    #Maximalwert der Gaußfunktion 
    ephi = phi(4,2)/np.sqrt(np.e)
    print('1/sqrt(e) der maximalen Stelle der Gaußfunktion:', ephi)
    print(phi(1,1))
    X = np.linspace(myx-sigmax-4, myx+sigmax+4, 100)
    Y = np.linspace(myy-sigmay-4, myy+sigmay+4, 100)
    x, y = np.meshgrid(X, Y) 
    plt.figure() 
    plt.pcolormesh(x, y, phi(x,y))
    plt.colorbar()
    plt.contour(x,y,phi(x,y), np.linspace(ephi,ephi,1), colors="k")
    plt.plot(myx,myy,'bo', label='$\mu_{xy}$')
    
    #Standardabweichung eingezeichnete Punkte
    plt.plot(myx+sigmax,myy, 'ro', label='$\sigma_{x/y}$')
    plt.plot(myx-sigmax,myy, 'ro')
    plt.plot(myx,myy+sigmay, 'ro')
    plt.plot(myx,myy-sigmay, 'ro')
    plt.axvline(x=myx, color='g', label='$\mu_x$')
    plt.axhline(y=myy, label='$\mu_y$')
    plt.legend(loc='best')
    plt.gca().set_aspect("equal")
    plt.show()
if __name__ == '__main__':
    aufg4()

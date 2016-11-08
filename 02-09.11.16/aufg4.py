def aufg4():
    import matplotlib.pyplot as plt
    import numpy as np
    from numpy.linalg import inv, det

    X = np.linspace(2, 6, 200)
    Y = np.linspace(0, 4, 200)
    x, y = np.meshgrid(X, Y) 
    #Startwerte
    myx = 4; myy = 2; sigmax = 3.5; sigmay = 1.5 
    
    #2dim Gaußfunktion
    def phi(x1,x2):
        return 1/(2*np.pi*sigmax*sigmay)*np.e**(-1/(2*sigmax)*(x1-myx)**2)*np.e**(-1/(2*sigmay)*(x2-myy)**2)

    #Maximalwert der Gaußfunktion 
    ephi = phi(4,2)/np.sqrt(np.e)
    print('1/sqrt(e) der maximalen Stelle der Gaußfunktion:', ephi)

    plt.figure() 
    plt.pcolormesh(x, y, phi(x,y))
    plt.colorbar()
    plt.gca().set_aspect("equal")
    plt.show()
if __name__ == '__main__':
    aufg4()

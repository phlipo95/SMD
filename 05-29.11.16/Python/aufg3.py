def aufg3():

    from matplotlib import pyplot as plt
    import numpy as np
    from mpl_toolkits.mplot3d import Axes3D

    lam = 1/np.sqrt(185)*np.array([-8, 0, 11])
    #Population
    P0 = np.array([[2, 2, 2, 1, 3], [2, 3, 1, 2, 2], [1, 2, 2, 0, 0]])
    P1 = np.array([[2.5, 2.5, 4, 5.5, 5.5], [2.5, 1.5, 2, 2.5, 1.5], [0, 0, 0, 0, 0]])
    #Projizierte Population
    P0p = np.dot(lam,P0)
    P1p = np.dot(lam,P1)
    #Projektionsgerade
    Proj = 1/np.sqrt(185)* np.linspace(-45, 7, 10)

    #Aufgabe 3 d
    plt.plot(P0[0], P0[2], 'rx', label='Population 0')
    plt.plot(np.dot(lam[0],P0p), np.dot(lam[2],P0p), 'ro', label=r'$S_{0,proj}$')
    plt.plot(P1[0], P1[2], 'bx', label='Population 1')
    plt.plot(np.dot(lam[0],P1p), np.dot(lam[2],P1p), 'bo', label=r'$S_{1,proj}$')
    plt.plot(lam[0]*Proj, lam[2]*Proj, 'g-', label='Projektionsgerade')
    plt.xlabel('x')
    plt.ylabel('z')
    plt.grid()
    plt.legend(loc="best")
    plt.savefig('Aufgabe3d.pdf')
    #plt.show()
    plt.close()


    #Aufgabe 3 e
    lamcut = -12/np.sqrt(185)
    ProjCut = 1/np.sqrt(185)* np.linspace(-20, 15, 10)

    plt.plot(np.dot(lam[0],P0p), np.dot(lam[2],P0p), 'ro', label=r'$S_{0,proj}$')
    plt.plot(np.dot(lam[0],P1p), np.dot(lam[2],P1p), 'bo', label=r'$S_{1,proj}$')
    plt.plot(lam[0]*Proj, lam[2]*Proj, 'g-', label=r'$\lambda$')
    plt.plot(lamcut*lam[0], lamcut*lam[2], 'yo', label=r'$\lambda_{cut}$')
    plt.plot(lam[2]*ProjCut + lamcut*lam[0], lam[2]*ProjCut + lamcut*lam[2], 'y-', label=r'$\lambda_{cut}$')
    plt.xlabel('x')
    plt.ylabel('z')
    plt.grid()
    #plt.gca().set_aspect('equal')
    plt.legend(loc="best")
    plt.savefig('Aufgabe3e.pdf')
    #plt.show()
    plt.close()

if __name__ == '__main__':
    aufg3()

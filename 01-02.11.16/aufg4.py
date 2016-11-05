def aufg4():
    import matplotlib.pyplot as plt
    import numpy as np
    import math
    print('Aufgabe 4:')

    EE = 50*(10**9)
    ME = 511*(10**3)
    gamma = np.float16(EE/ME)
    beta = np.float16(np.sqrt(1 - gamma**(-2)))
    print(type(beta))
    #numerisch instabiler Wirkungsquerschnitt
    def wqs(x):
        return (2+np.sin(x)**2)/(1- beta *(np.cos(x)**2))
    #modifizierte Wirkungsquerschnitt
    def mwqs(x):
        return (3-np.cos(x)**2)/((ME/EE)**2*(np.cos(x)**2)+(np.sin(x)**2))
    x = np.linspace(-1*10**(-6)+np.pi, np.pi+1*10**(-6), 3000)
    ax4 = plt.subplot(211)
    #Plotten der Wirkungsquerschnitte im kritischen Intervall
    plt.plot(x, wqs(x))
    plt.title('Numerisch Instabil')
    plt.setp(ax4.get_xticklabels(), visible=False)
    plt.yscale('log')
    plt.xlim(-1*10**(-6)+np.pi, np.pi+1*10**(-6))
    #plt.xticks([0 , np.pi/2, np.pi],[0, r'$\frac{\pi}{2}$', r'$\pi$'])
    plt.subplot(212, sharex = ax4)
    plt.plot(x, mwqs(x))
    plt.title('Numerisch Stabil')
    plt.yscale('log')
    plt.xlim(-1*10**(-6)+np.pi, np.pi+1*10**(-6))
    plt.savefig('Aufgabe4.pdf')
    plt.close()

    print('Aufgabe 4d:')
    #Konditionszahlen
    def sKondi(x):
        return x*((2*np.sin(x)*np.cos(x))/(1-(beta*np.cos(x))**2)-((2+np.sin(x)**2)*2*(beta**2)*np.sin(x)*np.cos(x)/(1-(beta*np.cos(x))**2)**(2)))/wqs(x)
    def gKondi(x):
        return x*((2*np.sin(x)*np.cos(x))/(np.sin(x)**2+((gamma**(-2))*(np.cos(x)**2)))+(3-np.cos(x)**2)*((2*(gamma**(-2))-2)*np.sin(x)*np.cos(x))/((np.sin(x)**2+((gamma**(-2))*(np.cos(x)**2)))))/mwqs(x)

    x = np.linspace(-0.1, np.pi+0.1, 100)
    plt.plot(x, sKondi(x),'r.-', alpha =0.5, label='numerisch instabil')
    plt.plot(x, gKondi(x),'b-.', alpha=0.5, label='numerisch stabil')
    plt.xlim(0, np.pi)
    plt.legend(loc='best')
    plt.savefig('Aufgabe4e.pdf')

if __name__ == '__main__':
        aufg4()

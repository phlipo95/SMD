def aufg1():
    from scipy.integrate import quad 
    import scipy.constants as const
    import numpy as np
    
    def integrand(x, c): #Maxwell'sche Gleichung
        return np.e**(-x**2*c)*(c/np.pi)**(3/2)*4*np.pi*x**2
    
    def AbhTUM(m,T): #in Abhängigkeit der Temperatur und Masse
        a = 0
        I = [0,0]
        c = m/(2*const.k*T)
        while 0.5>I[0]:
            a += 0.01
            I = quad(integrand, 0, a, args=(c))
        print('numerisches Eregebnis', I[0], 'Intergralgrenzen = 0 bis', a)
    print(AbhTUM(10**(-21),200))

if __name__ == '__main__':
    aufg1()

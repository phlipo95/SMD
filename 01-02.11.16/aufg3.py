def aufg3():
    import matplotlib.pyplot as plt
    import numpy as np
    import math
    print('Aufgabe 3a:')
    i = 0
    x = 0
    while i < 0.01:         #Bestimmung der oberen Grenze für 1% Abweichung
        f = (x**3 + 1/3) - (x**3 - 1/3)
        i = abs((2/3)-f) / (2/3)
        x = x+10
    print('x =', x-10)
    
    i = 0
    x = 0
    while i < 0.01:         #Bestimmung der unteren Grenze für 1% Abweichung
        f = (x**3 + 1/3) - (x**3 - 1/3)
        i = abs((2/3)-f) / (2/3)
        x = x-10
    print('x =', x-10)

    i = 0
    x = 0
    while i == 0:           #Bestimmung der unteren Grenze für exakte Werte
        f = (x**3 + 1/3) - (x**3 - 1/3)
        i = abs((2/3)-f) / (2/3)
        x = x+0.0001
    print('x =', x-0.0001)

    i = 0
    x = 0
    while i == 0:           #Bestimmung der unteren Grenze für exakte Werte
        f = (x**3 + 1/3) - (x**3 - 1/3)
        i = abs((2/3)-f) / (2/3)
        x = x-0.0001
    print('x =', x+0.0001)
    print('Aufgabe 3b:')

    i = 0
    x = 1
    while i < 0.01:         #Bestimmung der unteren Grenze für 1% Abweichung
        g = ((3 + x**3/3) - (3 - x**3/3)) / x**3
        i = abs((2/3)-g) / (2/3)
        x = x/10
    print('x =', x)
    x = np.linspace(10**-6, 10**6, 10**6)
    plt.plot(x, (x**3 + 1/3) - (x**3 - 1/3), 'r-', label='Aufgabenteil a)')  #Plot von f(x)
    plt.plot(x, ((3 + x**3/3) - (3 - x**3/3)) / x**3, 'b-', label='Aufgabenteil b)')  #Plot von g(x)
    plt.xlabel(r'x')
    plt.ylabel(r'f(x)')
    plt.xscale('log')
    plt.grid()
    plt.legend(loc='best')
    plt.savefig('Aufgabe3.pdf')
    plt.close()
    
if __name__ == '__main__':
    aufg3()

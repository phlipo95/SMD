def aufg2():
    import matplotlib.pyplot as plt
    import numpy as np
    import math
    print('Aufgabe 2:')
    def f(x):       #f(x)
        return (np.sqrt(9-x)-3)/x
    x = np.ones((20, 1))
    i=1
    while i < 21:   #x = e-1 bis e-20
        x[i-1] = 10**(-i)
        i = i+1
    i=1
    while i < 21:   #f(x) ausgerechnet
        print('x =', x[i-1], 'f(x) =', f(x[i-1]))
        i = i+1

if __name__ == '__main__':
    aufg2()

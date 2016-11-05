def aufg1():
    import matplotlib.pyplot as plt
    import numpy as np
    import math
    x = np.linspace(0.999, 1.001, 1000)
    y1 = (1-x)**6
    y2 = x**6 -6*x**5 +15*x**4 -20*x**3 +15*x**2 -6*x +1 #Binomische Formel
    y3 = x*(x*(x*(x*((x-6)*x +15) -20) +15) -6) +1 #Horner Schema
    #Aufgabe 1 a
    ax1 = plt.subplot(311)
    plt.plot(x, y1, linewidth=0.5)
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.ylabel(r'f(x)')
    #Aufgabe 1 b
    ax2 = plt.subplot(312, sharex=ax1)
    plt.plot(x, y2, linewidth=0.5)
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.ylabel(r'g(x)')
    #Aufgabe 1 c
    ax3 = plt.subplot(313, sharex=ax1)
    plt.plot(x, y3, linewidth=0.5)
    plt.xlabel(r'x')
    plt.ylabel(r'h(x)')
    plt.savefig('Aufgabe1.pdf')
    plt.close()

if __name__ == '__main__':
    aufg1()


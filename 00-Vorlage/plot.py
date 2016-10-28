import matplotlib.pyplot as plt
import numpy as np
import math

#Aufgabe 1
x = np.linspace(0.999, 1.001, 1000)
y1 = (1-x)**6
y2 = x**6 -6*x**5 +15*x**4 -20*x**3 +15*x**2 -6*x +1 #Binomische Formel
y3 = x*(x*(x*(x*((x-6)*x +15) -20) +15) -6) +1 #Horner Schema

#Aufgabe 1 a
plt.subplot(5, 1, 1)
plt.plot(x, y1, linewidth=0.5)
plt.xlabel(r'x')
plt.ylabel(r'f(x)')

#Aufgabe 1 b
plt.subplot(5, 1, 3)
plt.plot(x, y2, linewidth=0.5)
plt.xlabel(r'x')
plt.ylabel(r'g(x)')

#Aufgabe 1 c
plt.subplot(5, 1, 5)
plt.plot(x, y3, linewidth=0.5)
plt.xlabel(r'x')
plt.ylabel(r'h(x)')

plt.savefig('build/Aufgabe1.pdf')
plt.close()

#Aufgabe 2
print('----------------------------------------------------')
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

print('----------------------------------------------------')
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

print('----------------------------------------------------')
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
plt.savefig('build/Aufgabe3.pdf')
plt.close()
















#

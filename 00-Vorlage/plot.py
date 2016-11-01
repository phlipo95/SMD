import matplotlib.pyplot as plt
import numpy as np
import math

#Aufgabe 1
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

print('----------------------------------------------------')
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
plt.savefig('build/Aufgabe4.pdf')
plt.close()

print('Aufgabe 4d:')

def sKondi(x):
    return x*((2*np.sin(x)*np.cos(x))/(1-(beta*np.cos(x))**2)-((2+np.sin(x)**2)*2*(beta**2)*np.sin(x)*np.cos(x)/(1-(beta*np.cos(x))**2)**(2)))/wqs(x)
def gKondi(x):
    return x*((2*np.sin(x)*np.cos(x))/(np.sin(x)**2+((gamma**(-2))*(np.cos(x)**2)))+(3-np.cos(x)**2)*((2*(gamma**(-2))-2)*np.sin(x)*np.cos(x))/((np.sin(x)**2+((gamma**(-2))*(np.cos(x)**2)))))/mwqs(x)

print(gKondi(1))
x = np.linspace(-0.1, np.pi+0.1, 100)
plt.plot(x, sKondi(x),'r.-', alpha =0.5, label='numerisch instabil')
plt.plot(x, gKondi(x),'b-.', alpha=0.5, label='numerisch stabil')
plt.xlim(0, np.pi)
plt.legend(loc='best')
plt.savefig('build/Aufgabe4e.pdf')

print('Kondi bei x=0', sKondi(0))

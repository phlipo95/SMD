import ROOT
import numpy as np
from matplotlib import pyplot as plt

#Gleichverteilte Zufallszahlen von 0 bis 1 erstellen
xn = ROOT.TRandom3()
xn = [xn.Rndm() for i in range(4000)]

#Gleichverteilung
def gV(x,xMin, xMax):
    fx =[]
    for a in range(len(xn)):
        Fx = x[a-1]*(xMax-xMin)+xMin
        fx = np.append(fx, [Fx])
    return fx
gv=gV(xn,0,100)
plt.hist(gv, 50, normed=1)
plt.savefig('aufg3a.pdf')
plt.close()

#Exponentialverteilung
def expo(x,t,n):
    fx = []
    for a in range(len(xn)):
        Fx = -t*np.log(1-x[a]/(n*t))
        fx = np.append(fx, [Fx])
    return fx

Expo=expo(xn,1,1)
plt.hist(Expo, 50, normed=1)
plt.savefig('aufg3b.pdf')
plt.close()

#Exponentialverteilung
def pot(x,xMin,n,N):
    fx = []
    for a in range(len(xn)):
        print(xn[a])
        Fx = ((-n+1)*x[a]/N+xMin**(-n+1))**(1/(-n+1))
        print(Fx)
        fx = np.append(fx, [Fx])
    return fx

Pot=pot(xn,1,3,1)
plt.hist(Pot, bins=np.linspace(0,3,50),normed=1)
plt.savefig('aufg3c.pdf')
plt.close()

#Cauchyverteilung
def cau(x):
    fx = []
    for a in range(len(xn)):
        Fx = np.tan(np.pi*(x[a]-0.5))
        fx = np.append(fx, [Fx])
    return fx

Cau=cau(xn)
plt.hist(Cau, bins=np.linspace(-20,20,50), normed=1)
plt.xlim(0,100)
plt.savefig('aufg3d.pdf')
plt.close()



#data = np.load("empirisches_histogramm.npy")
#plt.hist(data['bin_mid'], bins=np.linspace(0., 1., 50), weights=data['hist'])
#plt.show()

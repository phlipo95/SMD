import numpy as np
import numpy.random as rnd
import ROOT
from root_numpy import array2root, root2array
from matplotlib import pyplot as plt

# Aufgabenteil A)

p0 = 1
def trafo(x):
    return p0 * x ** (1 / (1 - 2.7))

# erzeugt Gleichverteilte Zufallszahlen
GvZz = np.array([])
for i in range(int(1e5)):
    GvZz = np.append(trafo(rnd.random()), GvZz)
rec = np.array(GvZz, dtype=[("Energie", np.float)])
array2root(rec, "NeutrinoMC.root", treename="Signal_MC", mode="RECREATE")

# Aufgabenteil b)
def detect(E):
    return (1 - np.exp(- E / 2)) **3

Energie = root2array("NeutrinoMC.root", "Signal_MC")
GvZz2 = rnd.random(size=len(Energie['Energie']))
mask = detect(np.array(Energie['Energie'])) >= GvZz2 
accept = np.array(Energie['Energie'][~mask],dtype=[("Energie",np.float)])

array2root(accept, "NeutrinoMC.root", treename="Signal_MC_Akzeptanz")

plt.hist(Energie['Energie'], bins=np.linspace(0,10,100), alpha=0.3, color="yellow", label="Aufgabenteil a)")
plt.hist(accept['Energie'], bins=np.linspace(0,10,100), alpha=0.3, color="red", label="Aufgabenteil b)")
plt.xlabel("Energie der Neutrinos / TeV")
plt.ylabel("Anzahl")
plt.xscale("log")
plt.yscale("log")
plt.legend(loc="best")
plt.show()


# c) Energiemessung

def polarmethode_1(E):
    a=True
    while a==True:
        v1 = 2*rnd.random()-1
        v2 = 2*rnd.random()-1
        s = v1**2 + v2**2
        if s <= 1:
            x1 = v1*np.sqrt(-(8*E/s)*np.log(s))+10*E

            if(int(x1)>0):
                a=False
    return int(x1)
hits = np.array([])


for i in Energie['Energie']:
    hits=np.append(hits,polarmethode_1(i))


plt.hist(hits,bins=100,color='yellow')
plt.xscale("log")
plt.yscale("log")
plt.show()

treffer=np.array(hits,dtype=[("Hits",np.float)])
array2root(treffer, "NeutrinoMC.root", treename="AnzahlHits")

# d) Ortsmessung
Hits = root2array("build/NeutrinoMC.root", "AnzahlHits")

x = np.array([])
y = np.array([])
for i in Hits['Hits']:
    mean = np.array([7, 3])
    sigma = 1/ (np.log10(i+1))
    cov = np.array([[sigma**2, 0], [0, sigma**2]])
    a = True
    while a == True:
        signal = rnd.multivariate_normal(mean, cov)
        if signal.any() <= 10 and signal.any() >= 0:
            a=False
        else:
            continue
    x = np.append(x, signal[0])
    y = np.append(y, signal[1])

plt.hist2d(x, y, bins=(10, 10), label="Ort")
plt.colorbar()
plt.xlim(0, 10)
plt.ylim(0, 10)
plt.xlabel("x")
plt.ylabel("y")
plt.show()

#Array mit Namen der Branches
Ort = np.empty((len(x)), dtype=[("x", np.float), ("y", np.float)])
#fügt dem Array die Werte hinzu
Ort["x"] = x
Ort["y"] = y
#speichert Array als .root
array2root(Ort, "build/NeutrinoMC.root",
           treename="Orte", mode="RECREATE")

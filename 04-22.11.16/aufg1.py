import numpy as np
import numpy.random as rnd
import ROOT
from root_numpy import array2root, root2array 
from matplotlib import pyplot as plt
#%matplotlib inline

# a) Signal MC
phi0 = 1
def trafo(x):
    return phi0 * x ** (1 / (1 - 2.7))

rnd_neutrinos = np.array([])
for i in range(int(1e3)):
    rnd_neutrinos = np.append(trafo(rnd.random()), rnd_neutrinos)

rec = np.empty((len(rnd_neutrinos)), dtype=[("Energie", np.float)])
rec = np.array(rnd_neutrinos, dtype=[("Energie", np.float)])

array2root(rec, "NeutrinoMC.root", treename="Signal_MC", mode="RECREATE")

# b) Akzeptanz
def detektionswkeit(E):
    return (1 - np.exp(- E / 2)) **3

Energie = root2array("NeutrinoMC.root", "Signal_MC")
akzeptiert=np.array([])
for i in Energie['Energie']:
    if(detektionswkeit(np.float(i))>=rnd.random()):
        akzeptiert=np.append(akzeptiert,i)
accept=np.empty((len(akzeptiert)),dtype=[("Energie",np.float)])
accept=np.array(akzeptiert,dtype=[("Energie",np.float)])


array2root(accept, "NeutrinoMC.root", treename="Signal_MC_Akzeptanz")

plt.hist(Energie['Energie'], bins=100, alpha=0.3, color="yellow", label="Aufgabenteil a)")
plt.xlabel("Energie der Neutrinos / TeV")
plt.ylabel("Anzahl")
plt.xscale("log")
plt.yscale("log")
#plt.axis("equal")
#plt.equal()
plt.xticks((1, 10, 100, 1000), ("1", "10", "100", "1000"))
plt.legend(loc="best")
#plt.xlim(1, 20)
#plt.ylim(0, 2000)
#plt.show()

plt.hist(accept['Energie'], bins=100, alpha=0.3, color="red", label="Aufgabenteil b)")
plt.xlabel("Energie der Neutrinos / TeV")
plt.ylabel("Anzahl")
plt.xscale("log")
plt.yscale("log")
#plt.axis("equal")
#plt.equal()
#plt.xticks((1, 10, 100, 1000), ("1", "10", "100", "1000"))
plt.legend(loc="best")
#plt.xlim(1, 20)
#plt.ylim(0, 2000)
plt.show()



#plt.hist(accept["Energie"], bins=100, alpha=0.3, color="yellow", label="Aufgabenteil b)")
#plt.xlabel("Detektionswahrscheinlichkeit / %")
#plt.ylabel("Anzahl")
#plt.xscale("log")
#plt.yscale("log")
#plt.xlim(2e-1, 1)
#plt.xlim(0, 1)
#plt.grid()
#plt.xticks((2e-1, 3e-1, 4e-1, 5e-1, 1e0), ("0.2", "0.3", "", "", "1"))
#plt.legend(loc="best")
#plt.show()



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

treffer=np.empty((len(hits)),dtype=[("Hits",np.float)])
treffer=np.array(hits,dtype=[("Hits",np.float)])
array2root(treffer, "NeutrinoMC.root", treename="AnzahlHits")

# d) Ortsmessung
x = np.array([])
y = np.array([])
for i in hits:
    mean = np.array([7, 3])
    sigma = 1/(np.log10(i+1))
    cov = np.array([[sigma**2, 0], [0, sigma**2]])
    a=True
    while a==True:
        signal = rnd.multivariate_normal(mean, cov)
        if signal.any() <= 10 and signal.any() >= 0:
            a=False
        else:
            continue
    x = np.append(x, signal[0])
    y = np.append(y, signal[1])

plt.hist2d(x, y, bins=(10, 10), label="Ort")
#plt.legend(loc="best")
plt.colorbar()
plt.xlim(0, 10)
plt.ylim(0, 10)
plt.xlabel("x")
plt.ylabel("y")
#plt.axis("equal")
plt.show()

x = np.array(x, dtype=[("x", np.float)])
y = np.array(x, dtype=[("y", np.float)])

array2root(x, "NeutrinoMC.root", treename="X_Cord_Hits")
array2root(y, "NeutrinoMC.root", treename="Y_Cord_Hits")

# e) Untergrund MC
def polarmethode_2():
    a=True
    while a==True:
        v1 = 2*rnd.random()-1
        v2 = 2*rnd.random()-1
        s = v1**2 + v2**2
        if s <= 1:
            x1 = v1*np.sqrt(-(2/s)*np.log(s))+2
            if(int(x1)>0):
                a=False
    return int(np.log10(x1))

hits_noise=np.array([])
n=1
while n<=10**4:
    print(n)
    hits_noise=np.append(hits_noise,polarmethode_2())
    n=n+1

plt.hist(hits_noise,bins=100,color="yellow")
plt.show()

# branches anzahlhits, x, y
mean_noise = np.array([5, 5])
sigma_noise = np.sqrt(3)
rho_noise = 0.5
cov_noise = np.array([[sigma_noise**2, rho_noise], [rho_noise, sigma_noise**2]])
x_noise = np.array([])
y_noise = np.array([])
for i in range(int(1e7)):
    signal = np.floor(rnd.multivariate_normal(mean_noise, cov_noise))
    if signal.any() <= 10 and signal.any() >= 0:
        x_noise = np.append(x_noise, signal[0])
        y_noise = np.append(y_noise, signal[1])
    else:
        continue

plt.hist2d(x_noise, y_noise)
plt.show()

# hits = np.array(hits, dtype=[("AnzahlHits", np.float)])
# x = np.array(x, dtype=[("AnzahlHits", np.float)])
# y = np.array(y, dtype=[("AnzahlHits", np.float)])

# array2root(hits, "NeutrinoMC.root", treename="Untergrund_MC")
# array2root(x, "NeutrinoMC.root", treename="Untergrund_MC")
# array2root(y, "NeutrinoMC.root", treename="Untergrund_MC")

# TO DO
# singal.any() guckt nicht jeden wert einzeln an,
# kombinierte AND abfragen funktionieren nicht!

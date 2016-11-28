import numpy as np
import numpy.random as rnd
import ROOT
from root_numpy import root2array
from matplotlib import pyplot as plt

root_file = ROOT.TFile("./zwei_populationen.root", "READ")

#Treenames
for key in root_file.GetListOfKeys():
    print(key.GetName())

#Liest Tree ein
tree = root_file.Get("P_1")
tree1 = root_file.Get("P_0_10000")

#Branchnames
branches = tree.GetListOfBranches()
branches1 = tree1.GetListOfBranches()

for branch in branches:
    print(branch.GetName())

x_vP01 = np.zeros(1, dtype=float)
y_vP01 = np.zeros(1, dtype=float)

x_vP00 = np.zeros(1, dtype=float)
y_vP00 = np.zeros(1, dtype=float)

tree.SetBranchAddress('x', x_vP01)
tree.SetBranchAddress('y', y_vP01)

tree1.SetBranchAddress('x', x_vP00)
tree1.SetBranchAddress('y', y_vP00)

nentries = tree.GetEntries()
nentries1 = tree1.GetEntries()

x_P01 = np.zeros(nentries, dtype=float)
y_P01 = np.zeros(nentries, dtype=float)
x_P00 = np.zeros(nentries1, dtype=float)
y_P00 = np.zeros(nentries1, dtype=float)

#Read in data one by one
for i in range(nentries):
    tree.GetEntry(i)
    x_P01[i] = x_vP01
    y_P01[i] = y_vP01

for i in range(nentries1):
    tree1.GetEntry(i)
    x_P00[i] = x_vP00
    y_P00[i] = y_vP00



def g1(x):
    return x-x

def g2(x):
    return -0.75*x

def g3(x):
    return -1.25*x

x = np.linspace(min(x_P00),max(x_P01),100)

vec1 = np.matrix([x_P01,y_P01])
vec2 = np.matrix([x_P00,y_P00])

G1 = np.matrix([[1],[0]])
G2 = np.matrix([[3],[-4]])
G2 = G2/np.linalg.norm(G2)
G3 = np.matrix([[5],[-4]])
G3 = G3/np.linalg.norm(G3)

def z(vec, G):
    return np.transpose(np.array(np.transpose(vec[:,:])*G))
'''

#Scatterplot
plt.figure(1)
plt.subplot(3,1,1)
plt.scatter(x_P01,y_P01,color='r',alpha=0.1)
plt.scatter(x_P00,y_P00,color='b',alpha=0.1)
plt.plot(G1[0]*z(vec1,G1)[:],G1[1]*z(vec1,G1)[:], 'go',alpha=0.2)
plt.plot(G1[0]*z(vec2,G1)[:],G1[1]*z(vec2,G1)[:], 'yo',alpha=0.2)
plt.subplot(3,1,2)
plt.scatter(x_P01,y_P01,color='r',alpha=0.2)
plt.scatter(x_P00,y_P00,color='b',alpha=0.2)
plt.plot(G2[0]*z(vec1,G2)[:],G2[1]*z(vec1,G2)[:], 'go',alpha=0.2)
plt.plot(G2[0]*z(vec2,G2)[:],G2[1]*z(vec2,G2)[:], 'yo',alpha=0.2)
plt.subplot(3,1,3)
plt.scatter(x_P01,y_P01,color='r',alpha=0.2)
plt.scatter(x_P00,y_P00,color='b',alpha=0.2)
plt.plot(G3[0]*z(vec1,G3)[:],G3[1]*z(vec1,G3)[:], 'go',alpha=0.2)
plt.plot(G3[0]*z(vec2,G3)[:],G3[1]*z(vec2,G3)[:], 'yo',alpha=0.2)
plt.savefig('Scatter.pdf')
plt.close()
'''

#Histogramme zum scatterplot
plt.figure(1)
plt.subplot(3,1,1)
H11 = np.transpose(vec1[:,:])*G1
plt.hist(H11,bins=20,alpha=0.2,label='P1',normed=1)
H12 = np.transpose(vec2[:,:])*G1
plt.hist(H12,bins=20,alpha=0.2,label='P0',normed=1)
plt.legend(loc='best')

plt.subplot(3,1,2)
H21 = np.transpose(vec1[:,:])*G2
plt.hist(H21,bins=20,alpha=0.2,label='P1',normed=1)
H22= np.transpose(vec2[:,:])*G2
plt.hist(H22,bins=20,alpha=0.2,label='P0',normed=1)
plt.legend(loc='best')

plt.subplot(3,1,3)
H31 =  np.transpose(vec1[:,:])*G3
plt.hist(H31,bins=20,alpha=0.2,label='P0',normed=1)
H32 = np.transpose(vec2[:,:])*G3
plt.hist(H32,bins=20,alpha=0.2,label='P1',normed=1)
plt.legend(loc='best')
plt.savefig('hist.pdf')
plt.close()

#Reinheit
def rein(Ha1, Ha2):
    x = np.linspace(min(Ha2).item(),max(Ha1).item(),10)
    leH1 = np.array([])
    leH2 = np.array([])
    for a in x:
        mask = Ha1 <= a
        leH1 = np.append(leH1, len(np.squeeze(np.asarray(Ha1[mask]))))
    for i in x:
        masg = Ha2 <= i
        print('Ha2',Ha2[masg])
        print(len(Ha2[masg]))

rein(H11,H12)
print('hier kommt H11',len(H11),type(H11))
print('hier kommt H12',len(H12),type(H12))





'''
Linerare Fischer Diskreminanzanalyse
#Mittelwerte der x-/y-Messwerte
my_1 = np.matrix([[np.mean(x_P01)],[np.mean(y_P01)]])
my_2 = np.matrix([[np.mean(x_P00)],[np.mean(y_P00)]])

S1 = np.array([[0,0],[0,0]])
S2 = np.array([[0,0],[0,0]])
for i in range(len(x_P01)):
    S1 = S1 + ((vec1[:,i] - my_1)*np.transpose(vec1[:,i] - my_1) )
    S2 = S2 + ((vec2[:,i] - my_1)*np.transpose(vec2[:,i] - my_1) )
print('S1 = ', S1)
print('S2 = ', S2)
SW = S1 + S2
SW1 = np.linalg.inv(SW)
#SW1 = np.matrix([[SW[1,1],-SW[0,1]],[-SW[1,0],SW[0,0]]])
#SW1 = SW1/np.linalg.det(SW)
#sB = (my_1 - my_2)*np.transpose(my_1 - my_2)
#Eigenwerte der LDA
lam = SW1*(my_1-my_2)
print(lam)
#Normierter LDA Vektor
lamN = lam/np.linalg.norm(lam)
print(lamN)

plt.hist(np.transpose(vec1[:,:])*lamN,bins=20,alpha=0.2,label='0',normed=1)
plt.hist(np.transpose(vec2[:,:])*lamN,bins=20,alpha=0.2,label='0',normed=1)
plt.show()
'''

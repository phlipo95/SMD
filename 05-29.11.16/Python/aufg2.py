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
plt.scatter(x_P01,y_P01, color='r', label='P_1')
plt.scatter(x_P00,y_P00, color='b', label='P_0_10000')
plt.plot(x,g1(x))
plt.plot(x,g2(x))
plt.plot(x,g3(x))
plt.legend(loc='best')

vec1 = np.matrix([x_P01,y_P01])
vec2 = np.matrix([x_P00,y_P00])

G1 = np.matrix([[1],[0]])
G2 = np.matrix([[-3],[4]])
G2 = G2/np.linalg.norm(G2)
G3 = np.matrix([[-5],[4]])
G3 = G3/np.linalg.norm(G3)
#plt.plot(vec1[0,:],g1(G1[0]*vec1[0,:]), 'rx')
#plt.plot(vec2[0,:],g1(G1[0]*vec2[0,:]), 'bx')
#plt.plot(vec1[0,:],g2(G2[0]*vec1[0,:]), 'rx')
#plt.plot(vec2[0,:],g2(G2[0]*vec2[0,:]), 'bx')
plt.plot(vec1[0,:],g3(G3[0]*vec1[0,:]), 'rx')
plt.plot(vec2[0,:],g3(G3[0]*vec2[0,:]), 'bx')

plt.show()
'''
Linerare Fischer Diskreminanzanalyse
'''
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
sB = (my_1 - my_2)*np.transpose(my_1 - my_2)
#Eigenwerte der LDA
lam = SW1*(my_1-my_2)
print(lam)
#Normierter LDA Vektor
lamN = lam/np.linalg.norm(lam)
print(lamN)


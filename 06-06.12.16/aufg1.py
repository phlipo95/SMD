def aufg1():
    import numpy as np
    import ROOT
    from root_numpy import root2array
    from matplotlib import pyplot as plt

    def rooteinlesen(File, Tree):
        zw = root2array(File, Tree)
        return np.array([zw['x'], zw['y']])

    #berechnet die Kovarianzmatrix der Werte der entsprechenden Mittelwerten
    def berechneKovS(xi, my):
        S = np.array([[0,0],[0,0]])
        for z in range(len(xi[0,:])):
            S = S + (xi[:,z] - my).reshape(2,1)*(xi[:,z]-my).reshape(1,2)
        return S

    #Fischer Diskreminante 
    def berechneLDA(S1,S2,my1,my2):
        SG = S1 + S2 
        print('Summierte Kovarianzmatrix = ', SG)
        SGI = np.linalg.inv(SG)
        diffmy = np.matrix((my1-my2).reshape(2,1))
        lamda =  SGI*diffmy/np.linalg.norm(SGI*diffmy)
        print(lamda)
        return lamda

    #Stelle die Population in eine 1D Histogramm da
    def plotHist(name, menge1, menge2, lam):
        plt.hist(np.transpose(menge1[:,:])*lam,normed=True,alpha=0.2)
        plt.hist(np.transpose(menge2[:,:])*lam,normed=True,alpha=0.2)
        plt.savefig(name)

    P1 = rooteinlesen("./zwei_populationen.root","P_1")
    P0 = rooteinlesen("./zwei_populationen.root","P_0_10000")
    P01 = rooteinlesen("./zwei_populationen.root","P_0_1000")
    
    myP1 = np.array([np.mean(P1[0,:]),np.mean(P01[1,:])])
    myP0 = np.array([np.mean(P0[0,:]),np.mean(P0[1,:])])
    myP01 = np.array([np.mean(P01[0,:]),np.mean(P01[1,:])])

    S_P0 = berechneKovS(P0, myP0)
    S_P1 = berechneKovS(P1, myP1)

    lam1 = berechneLDA(S_P0,S_P1,myP0,myP1)
    
    plotHist('firstHist.pdf', P0, P1,lam1)


    berechneKovS(P1, myP1)
if __name__ == '__main__':
    aufg1()

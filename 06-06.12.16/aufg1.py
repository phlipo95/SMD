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
    def plotHist(name, menge1, menge2, lam, lm1, lm2):
        plt.hist(np.transpose(menge1[:,:])*lam,normed=True,alpha=0.2,label=lm1)
        plt.hist(np.transpose(menge2[:,:])*lam,normed=True,alpha=0.2,label=lm2)
        plt.legend(loc='best')
        plt.savefig(name)
        plt.close()

    #Berchne Effizienz und Reinheit
    def berechneEffiRein(lam, menge1, menge2, plotname):
        proj1 = np.transpose(menge1[:,:])*lam
        proj2 = np.transpose(menge2[:,:])*lam
        kE = np.asscalar(min([min(proj1),min(proj2)]))
        gE = np.asscalar(max([max(proj1),max(proj2)]))
        x = np.linspace(kE,gE,10)
        pM1 = np.array([]); nM1 = np.array([]); pM2 = np.array([])
        for a in x:
            mask = menge1 >= a
            masg = menge2 >= a
            pM1 = np.append(pM1, len(menge1[mask]))
            nM1 = np.append(nM1, len(menge1[~mask]))
            pM2 = np.append(pM2, len(menge2[masg]))
        reinheit = np.array(pM2/(pM2+pM1))
        effizienz = np.array(pM2/(pM2+nM1))
        plt.plot(x, reinheit,label='Reinheit')
        plt.plot(x, effizienz,label='Effizienz')
        plt.legend(loc='best')
        plt.savefig(plotname)
        plt.close()
        return x, pM1, pM2

    def SigZuUnt(x, sig, unt, name):
        plt.figure(1)
        plt.subplot(2,1,1)
        plt.plot(x,sig/unt,label=r'Signal/Untergrund')
        plt.legend(loc='best')
        plt.subplot(2,1,2)
        plt.plot(x,sig/np.sqrt(sig + unt),label=r'Signal/Untergrund')
        plt.savefig(name.__add__('.pdf'))
        plt.close()


    P1 = rooteinlesen("./zwei_populationen.root","P_1")
    P0 = rooteinlesen("./zwei_populationen.root","P_0_10000")
    P01 = rooteinlesen("./zwei_populationen.root","P_0_1000")
    
    myP1 = np.array([np.mean(P1[0,:]),np.mean(P01[1,:])])
    myP0 = np.array([np.mean(P0[0,:]),np.mean(P0[1,:])])
    myP01 = np.array([np.mean(P01[0,:]),np.mean(P01[1,:])])

    S_P0 = berechneKovS(P0, myP0)
    S_P1 = berechneKovS(P1, myP1)

    lam1 = berechneLDA(S_P0,S_P1,myP0,myP1)

    plotHist('firstHist.pdf', P0, P1,lam1,'P0','P1')

    x1, sig1, sig2 = berechneEffiRein(lam1,P0,P1,'blabla.pdf')
    SigZuUnt(x1, sig1, sig2, 'abc')
    
    berechneKovS(P1, myP1)
if __name__ == '__main__':
    aufg1()

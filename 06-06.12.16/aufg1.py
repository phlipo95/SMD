
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
        print('Lamda = ',lamda)
        return lamda

    #Stelle die Population in eine 1D Histogramm da
    def plotHist(name, menge1, menge2, lam, lm1, lm2):
        plt.hist(np.transpose(menge1[:,:])*lam,normed=True,alpha=0.2,label=lm1)
        plt.hist(np.transpose(menge2[:,:])*lam,normed=True,alpha=0.2,label=lm2)
        plt.legend(loc='best')
        plt.savefig(name)
        plt.close()

    #Berchne Effizienz und Reinheit
    def berechneEffiRein(lam, menge2, menge1, plotname):
        menge1 = np.transpose(menge1[:,:])*lam
        menge2 = np.transpose(menge2[:,:])*lam
        kE = np.asscalar(min([min(menge1),min(menge2)]))
        gE = np.asscalar(max([max(menge1),max(menge2)])-1)
        x = np.linspace(kE,gE,100)
        pRau = np.array([]); nRau = np.array([]); pSig = np.array([])
        for a in x:
            rausch = menge1 >= a
            signal = menge2 >= a
            pRau = np.append(pRau, len(np.transpose(menge1[rausch])))
            nRau = np.append(nRau, len(np.transpose(menge1[~rausch])))
            pSig = np.append(pSig, len(np.transpose(menge2[signal])))
        reinheit = np.array(pSig/(pSig+pRau))
        effizienz = np.array(pSig/(pSig+nRau))
        plt.plot(x, reinheit,label='Reinheit')
        plt.plot(x, effizienz,label='Effizienz')
        plt.legend(loc='best')
        plt.savefig(plotname)
        plt.close()
        return x, pRau, pSig

    def SigZuUnt(x, sig, unt, name):
        plt.figure(1)
        plt.subplot(2,1,1)
        plt.plot(x,sig/unt,label=r'Signal/Untergrund')
        plt.legend(loc='best')
        plt.subplot(2,1,2)
        plt.plot(x,sig/np.sqrt(sig + unt),label=r'Signal/ $\sqrt{Signal + Untergrund}$')
        plt.legend(loc='best')
        plt.savefig(name.__add__('.pdf'))
        plt.close()


    P1 = rooteinlesen("./zwei_populationen.root","P_1")
    P0 = rooteinlesen("./zwei_populationen.root","P_0_10000")
    P01 = rooteinlesen("./zwei_populationen.root","P_0_1000")
    
    myP1 = np.array([np.mean(P1[0,:]),np.mean(P01[1,:])])
    print('Mittelwert myP1 = ',myP1)
    myP0 = np.array([np.mean(P0[0,:]),np.mean(P0[1,:])])
    print('Mittelwert myP0 = ',myP0)
    myP01 = np.array([np.mean(P01[0,:]),np.mean(P01[1,:])])
    print('Mittelwert myP01 = ',myP01)

    S_P0 = berechneKovS(P0, myP0)
    print('S_P0 = ',S_P0)
    S_P1 = berechneKovS(P1, myP1)
    print('S_P1 = ', S_P1)
    lam1 = berechneLDA(S_P0,S_P1,myP0,myP1)
    
    plotHist('./TeX/Figures/firstHist.pdf', P0, P1,lam1,'P0','P1')
    plotHist('./TeX/Figures/secondHist.pdf', P01, P1, lam1, 'P0', 'P1')

    x1, Rau1, Sig1 = berechneEffiRein(lam1,P0,P1,'./TeX/Figures/reinheit1.pdf')
    SigZuUnt(x1, Sig1, Rau1, './TeX/Figures/SigZuUnt1')
    
    x2, Rau2, Sig2 = berechneEffiRein(lam1,P01,P1,'./TeX/Figures/reinheit2.pdf')
    SigZuUnt(x2, Sig2, Rau2,'./TeX/Figures/SigZuUnt2')
    
    berechneKovS(P1, myP1)
if __name__ == '__main__':
    aufg1()


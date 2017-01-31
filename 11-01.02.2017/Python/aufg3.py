def aufg3():

    # coding: utf-8
        
    # In[1]:
    
    import numpy as np
    import matplotlib.pyplot as plt

    # # Aufgabe 3

    # In[2]:
    
    class unfold:
        def __init__(self, events, incorrAss = 0.0, deficit = 1):
            """
            Prameters:
            events: ndarray
                messaered Events
            incorrAss: float 
                incorrect Assigment 
            deficit: float
                loss of results
            Returns:
            ansM = answer matrix
            b = invert answer matrix
            """
            self.events = events
            self.deficit = deficit
            self.incorrAss = incorrAss
            self.zw = np.ones(len(events))-incorrAss  
            self.zw[1:-1] = self.zw[1:-1] -incorrAss
            self.ansM = deficit*(np.diagflat(self.zw) + np.diagflat(np.array([incorrAss]*(len(events)-1)),-1) + np.diagflat(np.array([incorrAss]*(len(events)-1)),1))
            self.B = np.matrix(np.linalg.inv(self.ansM))
          
        def calcKov(self):
            self.Kov = self.B*np.diagflat(self.events)*np.transpose(self.B)
            return self.Kov
            
        def trueEvents(self):
            self.f = self.B*np.array(self.events).reshape(len(self.events),1)
            return self.f


    # In[5]:

    f = np.array([193,485,664,783,804,805,779,736,684,626,566,508,452,400,351,308,268,233,202,173])
    task3 = unfold(f, incorrAss=0.23)
    g = np.matmul(task3.ansM,f)
    D = np.diagflat(np.sort(g, axis=None)[::-1])
    print("gemessenen Ereignisszahlen"," wahre Verteilung")
    for x in range(len(f)):
        print(g[x] , f[x])
    g_mess = np.random.poisson(np.matmul(task3.ansM,f)) 
    plt.bar(np.linspace(0,2,20), f, width=0.08, color ="red", alpha=0.3, label="wahre")
    plt.bar(np.linspace(0,2,20), g_mess, width=0.08, alpha=0.3, label="gemessene")
    plt.legend(loc="best")
    plt.savefig('gemesseneVerteilung.pdf')
    plt.show()


    # ## c)

    # In[6]:

    wert , vec = np.linalg.eig(task3.ansM)
    test = np.argsort(wert)[::-1]
    D = np.diagflat(wert[test])
    U = vec[test]


    # ## d)

    # In[7]:

    b = np.matmul(np.linalg.inv(U),f)
    c = np.matmul(np.linalg.inv(U),g)
    print(f)
    kov = np.matmul(np.linalg.inv(U), np.matmul(np.diagflat(f),np.transpose(np.linalg.inv(U))))
    b_ = np.matmul(np.linalg.inv(D), np.matmul(np.linalg.inv(U), g_mess))
    print("b = ", b_/np.diag(kov))
    plt.bar(np.linspace(0,20,20),b_/np.diag(kov))
    plt.grid()
    plt.ylim(-2,2)
    plt.savefig("bar.pdf")
    plt.show()

if __name__ == '__main__':     
        aufg3()

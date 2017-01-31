def aufg2():
	
	# coding: utf-8
	
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    
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
    
    
    # # Aufgabe 2
    
    # ## d)
    
    # In[3]:
    
    a = unfold(np.array([200,169]), deficit=0.8, incorrAss=0.1)
    print(a.calcKov())
    print("f = ",a.trueEvents())
    
    
    # ## e)
    
    # In[4]:
    
    b = unfold(np.array([200,169]), deficit=0.8, incorrAss=0.4)
    print(b.calcKov())
    print("f = ",b.trueEvents())

if __name__ == '__main__':     
        aufg2()

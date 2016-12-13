def aufg3():
    import numpy as np

    tem = np.array([29.4,26.7,28.3,21.1,20,18.3,17.8,22.2,20.6,23.9,23.9,22.2,27.2,21.7])
    wet = np.array([2,2,1,0,0,0,1,2,2,0,2,1,1,0])
    luf = np.array([85,90,78,96,80,70,65,95,70,80,70,90,75,80])
    win = np.array([False, True, False, False, False, True, True, False, False, False, True, True, False, True])
    fus = np.array([False, False, True, True, True, False, True, False, True, True, True, True, True, False])

    class Ent:
        '''Erstellt Information über das gegebene Array mit Booleanwerte (Hits Positiv, Hits Negativ, Entropie)'''
        def __init__(self, trueeth):
            self.mask = trueeth == True
            self.p = (len(trueeth[self.mask]))
            self.n = (len(trueeth[~self.mask]))
    
        def entro(self):
            return(-self.p/(self.p+self.n)*np.log2(self.p/(self.p+self.n))-self.n/(self.p+self.n)*np.log2(self.n/(self.p+self.n)))
        

    class gain(Ent):
        """Sollte die Entropie der Einzelnen Äste der verschiedenen Branches berechen. 
        Jedoch tritt wenn in einem Ast pnode bzw nnode = 0 ist das Problem auf das der Logarhytmus an dieser stelle nicht definiert ist. """
        def __init__(self, trueeth, node, gesEnt=1):
            self.branch = np.unique(node)
            self.ltr = len(trueeth)
            self.gesEnt = gesEnt
            self.speicher = 0.
            for x in self.branch:
                self.mask = node == x
                self.pnode = trueeth[self.mask]
                self.gew = len(self.pnode)/self.ltr
                self.a = Ent(self.pnode)
                print('Branch der Wurzel ',x,', ',self.a.entro())
                print('Gewichtete Entropie der ',self.gew*self.a.entro())
                self.speicher += self.gew*self.a.entro()
                print('__')
            print('Gain = ', self.gesEnt - self.speicher)

    gesEnt = Ent(fus)
    print('Die Gesamtentropie beträgt: ', gesEnt.entro())

    print('---Wind---')
    Win = gain(fus, win, gesEnt.entro())

    print('---Luftfeuchtigkeit---')
    Luf = gain(fus, luf, gesEnt.entro())

    print('---Wetter---')
    Wet = gain(fus, wet, gesEnt.entro())

    print('---Temperatur---')
    Tem = gain(fus, tem, gesEnt.entro())

if __name__ == '__main__':     
    aufg3()

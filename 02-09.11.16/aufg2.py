def aufg2():
	import ROOT
	import os
	if not os.path.exists("./build"):
	    os.makedirs("./build")
	import numpy as np
	import matplotlib.pyplot as plt
	randGen = ROOT.TRandom3()
	Ge, Gr = np.loadtxt('GroesseGewicht.txt', unpack=True)
	name=['5Bins','10Bins', '15Bins', '20Bins', '30Bins', '50Bins']
	bins = [5,10,15,20,30,50] 
	canv1 = ROOT.TCanvas("canv1", "TH2C Beispiel", 400, 800)
	canv1.Divide(2,3)
	for l in range(6):
	    canv1.cd(l+1)
	    name[l] = ROOT.TH2C(name[l], name[l], bins[l], min(Ge), max(Ge), bins[l], min(Gr),max(Gr)) 
	    for a in range(250):
	        name[l].Fill(Ge[a],Gr[a])
	    name[l].Draw("colz")
	    canv1.Update()
	canv1.Update()
	canv1.SaveAs("./build/aufg2a.pdf")	

	name =['5Bins','10Bins', '15Bins', '20Bins', '30Bins', '50Bins']
	bins = [5,10,15,20,30,50] 
	random_numbers = [randGen.Integer(100)+1 for i in range(10**5)]
	random_numbers =  np.log(random_numbers)
	canv2 = ROOT.TCanvas("canv2", "Aufgabe2c", 400,800)
	canv2.Divide(2,3)
	for l in range(6):
	    canv2.cd(l+1)
	    name[l] = ROOT.TH1F(name[l], name[l], bins[l], min(random_numbers), max(random_numbers))
	    for a in range(10**5):
	        name[l].Fill(random_numbers[a])
	    name[l].Draw()
	    canv2.Update()
	canv2.Update()
	canv2.SaveAs("./build/aufg2c.pdf")

if __name__ == '__main__':
	aufg2()

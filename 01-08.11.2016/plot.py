import ROOT
import os
if not os.path.exists("./build"):
    os.makedirs("./build")
import numpy as np
import matplotlib.pyplot as plt

root_file = ROOT.TFile("./build/Aufgabe2.root", "RECREATE")

tree = ROOT.TTree("GeGr", "GeGr")

x = np.zeros(1, dtype=float)
y = np.zeros(1, dtype=float)

tree.Branch("x", x, "x/D")
tree.Branch("y", y, "y/D")

Ge, Gr = np.loadtxt('GroesseGewicht.txt', unpack=True)
for i in range(250):
    x[0] = Ge[i]
    y[0] = Gr[i]
    tree.Fill()

root_file.Write()
root_file.Close()

root_file = ROOT.TFile("./build/Aufgabe2.root", "READ")

tree = root_file.Get("GeGr")

x_val = np.zeros(1, dtype=float)
y_val = np.zeros(1, dtype=float)

tree.SetBranchAddress("x", x_val)
tree.SetBranchAddress("y", y_val)

entries = tree.GetEntries()

x = np.zeros(entries, dtype=float)
y = np.zeros(entries, dtype=float)

for i in range(entries):
    tree.GetEntry(i)
    x[i] = x_val
    y[i] = y_val

root_file.Close()

canv1 = ROOT.TCanvas("canv1", "TH1F Beispiel", 400, 300)
hist1 = ROOT.TH1F("hist1", "hist1", 11, min(x), max(x))
for random_number in x:
    hist1.Fill(random_number)
hist1.Draw()
canv1.Update()
canv1.SaveAs("./build/THist.png")

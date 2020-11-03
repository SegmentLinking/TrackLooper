import monkeyroot
import ROOT as r
import numpy as np

h1 = r.TH1F("h1","not a regular hist",20,0,10)
h1.FillRandom("expo",100000)

print ">>> get integral between two values:",
print h1.IntegralBetween(0.,2.)

print ">>> now fill it with my own values -- super fast since loop is internal"
rands = np.random.normal(5.,1, 1000000)
h1.FillFromList(rands)
print ">>> done"
print ">>> entries: ",h1.GetEntries()

print ">>> get error objects (x +- y) for each bin",
binvalerrs = h1.GetBinValueErrors()
print map(lambda x: x.round(2),binvalerrs)
print ">>> sum them to show that you get the integral and Poisson error:",
print sum(binvalerrs).round(3)

print ">>> take a peek with imgcat"
h2 = 1.2*h1+h1-0.8*h1
h2.SetTitle("newer hist")
h2.Show("histe")
print ">>> using 'same' will augment the legend, and change the color"
h1.FillFromList(np.random.normal(2,1,100000))
h1.Show("samehiste")

"""
More details:

After importing monkeyroot, all TH1F constructors are monkeypatched.
If you do something like 
ch.Draw("blah>>h1","")
and h1 has not already been constructed,
then h1 will NOT be patched if you get it like
h1 = r.gDirectory.Get("h1").
Simply "cast" it:
h1 = r.TH1F(r.gDirectory.Get("h1"))
Done


You can also get a simple data/MC ratio in one line.
hdata, h1,h2,h3 are your data and bg hists. This gives you an SF and error. Neat.
print r.TH1F(hdata).IntegralAndErrorBetween(0.,5.)/sum([h1,h2,h3],r.TH1F()).IntegralAndErrorBetween(0.,5.)
"""

#!/bin/env python

import ROOT as r

h = r.TH1F("momentum_loss", "momentum_loss", 50, 0, 1.1)

f = open("momentum_skimmed_v2.log")

lines = f.readlines()

for line in lines:
    ls = line.split()
    momentums = [ float(x) for x in ls[1:] ]
    maxloss = 0
    for index in xrange(len(momentums)-1):
        loss = abs(momentums[index+1] - momentums[index]) / momentums[index]
        if maxloss < loss:
            maxloss = loss
    h.Fill(maxloss)
    if maxloss > 0.35 and maxloss < 0.55:
        print line

c1 = r.TCanvas()
h.Draw("hist")
c1.SetLogy()
c1.SaveAs("hist.pdf")

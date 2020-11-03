#!/bin/env python

import ROOT as r
import numpy as np

f = open("denom_pts")
denom_pts = [ float(line) for line in f.readlines()]
f = open("numer_pts")
numer_pts = [ float(line) for line in f.readlines()]

# ptbounds = [0.5, 0.52, 0.54, 0.56, 0.58, 0.6, 0.62, 0.64, 0.66, 0.68, 0.7, 0.72, 0.74, 0.76, 0.78, 0.8, 0.82, 0.84, 0.86, 0.88, 0.9, 0.92, 0.94, 0.96, 0.98, 1.0, 1.02, 1.04, 1.06, 1.08, 1.1, 1.12, 1.14, 1.16, 1.18, 1.2, 1.22, 1.24, 1.26, 1.28, 1.3, 1.32, 1.34, 1.36, 1.38, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
# ptbounds = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 2.0, 3.0, 4.0, 5.0]
ptbounds = [0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0, 3.0, 5.0, 10, 15., 25, 50]
bin_edges = np.array(ptbounds, dtype='float64')

h_denom = r.TH1F("denom_pt", "denom_pt", len(bin_edges)-1, bin_edges)
h_numer = r.TH1F("numer_pt", "numer_pt", len(bin_edges)-1, bin_edges)

h_denom.Sumw2()
h_numer.Sumw2()

for denom_pt in denom_pts: h_denom.Fill(denom_pt)
for numer_pt in numer_pts: h_numer.Fill(numer_pt)

h_eff = h_denom.Clone()
h_eff.Divide(h_numer, h_denom, 1, 1, 'b')

c1 = r.TCanvas("c1", "c1", 800, 800)

h_eff.Draw("ep")

c1.SaveAs("test.pdf")

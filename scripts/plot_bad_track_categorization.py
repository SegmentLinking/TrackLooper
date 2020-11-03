#!/bin/env python

import ROOT as r

r.gROOT.SetBatch(True)

t = r.TChain("tree")
t.Add("debug_ntuple_output/debug_20200321_0056_*.root")

# f = r.TFile("debug.root")
# t = f.Get("tree")

c1 = r.TCanvas("","",1800,300)

t.Scan("sqrt(simhit_x**2+simhit_y**2):simhit_z","is_trk_bbbbbe==1&&pt>=1&&(Sum$(mdendcap1_pass)==0)")
t.Draw("sqrt(simhit_x**2+simhit_y**2):simhit_z","")
t.SetMarkerStyle(20)
t.SetMarkerSize(0.5)
t.SetMarkerColor(2)
t.Draw("sqrt(simhit_x**2+simhit_y**2):simhit_z","is_trk_bbbbbe==1&&pt>=1&&(Sum$(mdendcap1_pass)==0)&&(Entry$==28761)", "same")
t.SetMarkerStyle(20)
t.SetMarkerSize(0.5)
t.SetMarkerColor(4)
t.Draw("sqrt(ph2_x**2+ph2_y**2):ph2_z","is_trk_bbbbbe==1&&pt>=1&&(Sum$(mdendcap1_pass)==0)&&(Entry$==28761)", "same")
# t.Scan("is_trk_bbbbbb:is_trk_bbbbbe:is_trk_bbbbee","is_trk_bbbbbe==1&&pt>=1&&(Sum$(mdendcap1_pass)==0)","")
# t.Draw("sqrt(simhit_x**2+simhit_y**2):simhit_z","is_trk_bbbbbe==1&&pt>=1&&(Sum$(mdendcap1_pass)==0)&&(Entry$==1040)","colz")

c1.SaveAs("plots_bad_track_categorization/plot.png")

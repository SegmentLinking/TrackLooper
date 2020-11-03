#!/bin/env python

import ROOT as r
import glob

froots = glob.glob("results/algo_eff/pt0p5_2p0_2020_0429_nmucut_*/fulleff_pt0p5_2p0.root")
fconns = glob.glob("data/module_connection_nmuon*.txt")

froots.sort()
fconns.sort()

effs = []
nconns = []

for froot, fconn in zip(froots, fconns):

    tf = r.TFile(froot)
    hnumer = tf.Get("Root__tc_bbbbbb_matched_track_pt_by_layer0")
    hdenom = tf.Get("Root__tc_bbbbbb_all_track_pt_by_layer0")

    eff = hdenom.Clone()
    eff.Divide(hnumer, hdenom, 1, 1, 'B')
    eff.SetDirectory(0)

    effs.append(eff)

    h = r.TH1F("nconn", "", 50, 0, 50)
    h.SetDirectory(0)

    g = open(fconn)
    lines = g.readlines()

    for line in lines:
        nconn = int(line.split()[1])
        h.Fill(nconn)
    nconns.append(h)

    print(effs)

c1 = r.TCanvas()

for index, eff in enumerate(effs):
    eff.SetLineColor(index + 1)
    eff.SetLineWidth(2)
    if index == 0:
        eff.Draw("ep")
    else:
        eff.Draw("epsame")

c1.SaveAs("effs.pdf")

for index, eff in enumerate(effs):
    eff.SetLineColor(index + 1)
    eff.SetLineWidth(2)
    if index == 0:
        eff.Draw("hist")
    else:
        eff.Divide(effs[0])
        if index == 1:
            eff.Draw("ep")
            eff.SetMaximum(1.05)
            eff.SetMinimum(0.95)
        else:
            eff.Draw("epsame")

c1.SaveAs("effs_ratios.pdf")

for index, nconn in enumerate(nconns):
    nconn.SetLineColor(index + 1)
    nconn.SetLineWidth(2)
    if index == 0:
        nconn.Draw("hist")
        nconn.SetMaximum(nconn.GetMaximum() * 1.5)
    else:
        nconn.Draw("histsame")

c1.SaveAs("nconns.pdf")

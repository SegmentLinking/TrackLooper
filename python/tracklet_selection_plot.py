#!/bin/env python

import plottery_wrapper as p
from plottery import plottery as plt
import ROOT as r
import itertools

f = r.TFile("/home/users/phchang/public_html/analysis/sdl/TrackLooper_/results/tracklet_study/mu_2020_0426_2226/fulleff_mu.root")
tree = f.Get("tree")

of = r.TFile("tracklet_hist.root", "recreate")

def draw(bound, pt, category, nPS, charge, savename):

    c1 = r.TCanvas()

    minpt = pt - 0.1 if pt < 10 else pt
    maxpt = pt + 0.1 if pt < 10 else 999999

    tree.Draw(
            "dBeta[4]>>h(50,-{},{})".format(bound, bound),
            "(matched_trk_pt>{})&&(matched_trk_pt<{})&&(category=={})&&(nPS=={})&&(matched_trk_charge{}0)".format(minpt, maxpt, category, nPS, charge)
            )

    of.cd()
    h = r.gDirectory.Get("h")
    h.SetTitle(savename)
    h.SetName(savename)
    h.Write()
    print(h)

    # c1.SaveAs("{}.pdf".format(savename))
    # c1.SaveAs("{}.png".format(savename))

bounds = ["0.15", "0.04"]
pts = [1.5, 10]
category = range(14)
nPSs = range(5)
charges = ["<", ">"]

for bound, pt, categ, n, ch in itertools.product(bounds, pts, category, nPSs, charges):
    draw("0.15", 1.5, 0, 3, ">", "dbeta_{}_pt{}_categ{}_nPS{}_{}".format(bound, pt, categ, n, "pos" if ">" in ch else "neg"))



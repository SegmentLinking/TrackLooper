#!/bin/env python

import ROOT as r

# fnew = r.TFile("/home/users/phchang/public_html/analysis/sdl/TrackLooper_/results/algo_eff/pt0p5_2p0_2020_0522_helixtracing/eff_2.root");
fnew = r.TFile("/home/users/phchang/public_html/analysis/sdl/TrackLooper_/results/algo_eff/pt0p5_2p0_2020_0522_helixtracing_w_endcap2cleaneddenommap/eff_2.root");
fold = r.TFile("/home/users/phchang/public_html/analysis/sdl/TrackLooper_/results/algo_eff/pt0p5_2p0_2020_0515/eff_2.root");

def plot(trktype, layer):

    gnew = fnew.Get("sg_eff_{}_pt_by_layer{}.pdf".format(trktype, layer));
    gold = fold.Get("sg_eff_{}_pt_by_layer{}.pdf".format(trktype, layer));

    c1 = r.TCanvas()
    c1.SetBottomMargin(0.15)
    c1.SetLeftMargin(0.15)
    c1.SetTopMargin(0.15)
    c1.SetRightMargin(0.15)
    c1.SetLogx()
    c1.SetGrid()

    def style_graph(eff):
        eff.SetMarkerStyle(19)
        eff.SetMarkerSize(1.2)
        eff.SetLineWidth(2)
        eff.GetXaxis().SetTitleSize(0.05)
        eff.GetXaxis().SetLabelSize(0.05)
        eff.GetYaxis().SetLabelSize(0.05)
        eff.GetYaxis().SetRangeUser(0.90, 1.05)

    style_graph(gnew)
    style_graph(gold)

    gnew.SetMarkerColor(2)
    gnew.SetLineColor(2)

    gnew.Draw("epa")
    gold.Draw("ep")

    c1.SaveAs("results/algo_eff_comp/pt0p5_2p0_2020_0522_helixtracing__pt0p5_2p0_2020_0515/sgeff/sg_eff_{}_pt_by_layer{}.pdf".format(trktype, layer))
    c1.SaveAs("results/algo_eff_comp/pt0p5_2p0_2020_0522_helixtracing__pt0p5_2p0_2020_0515/sgeff/sg_eff_{}_pt_by_layer{}.png".format(trktype, layer))

if __name__ == "__main__":

    trktypes = ["bbbbbb", "bbbbbe", "bbbbee", "bbbeee", "bbeeee", "beeeee"]
    layers = [0, 1, 2, 3, 4]

    for trktype in trktypes:
        for layer in layers:
            plot(trktype, layer)

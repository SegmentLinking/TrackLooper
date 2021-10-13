#!/bin/env python

import os
import sys
import glob
import ROOT as r

# Get the git hash to compare
#root_file_name = "num_den_histograms.root"
root_file_name = "efficiencies.root"
sample_name = sys.argv[1]
version_tag = sys.argv[2]

r.gROOT.SetBatch(True)

def parse_plot_name(output_name):
    if "fakerate" in output_name:
        rtnstr = ["Fake Rate of"]
    elif "duplrate" in output_name:
        rtnstr = ["Duplicate Rate of"]
    else:
        rtnstr = ["Efficiency of"]
    rtnstr.append("TC vs TE")
    types = "of all types"
    types = "of set 1 types"
    rtnstr.append(types)
    return " ".join(rtnstr)

f = r.TFile(root_file_name)
keys = [ x.GetName()[3:] for x in f.GetListOfKeys() if "Root__" not in x.GetName() and "TC_" in x.GetName()]

tc_tgraphs = {}
tce_tgraphs = {}
for key in keys:
    tc_tgraphs[key] = f.Get("TC_"+key)
    tce_tgraphs[key] = f.Get("TCE_"+key)


ms = [24, 25, 26, 27, 28, 30, 32, 42, 46, 40]
cs = [2, 3, 4, 6, 7, 8, 9, 30, 46, 38, 40]


for key in keys:

    eff = tc_tgraphs[key]
    output_name = "plots/mtv/TC_v_TCE_{}".format(key) + ".pdf"

    c1 = r.TCanvas()
    c1.SetBottomMargin(0.15)
    c1.SetLeftMargin(0.15)
    c1.SetTopMargin(0.15)
    c1.SetRightMargin(0.15)
    if "_pt" in output_name:
        c1.SetLogx()
    eff.Draw("epa")
    eff.SetMarkerStyle(19)
    eff.SetMarkerSize(1.2)
    eff.SetLineWidth(2)
    if "phi" in output_name:
        title = "#phi"
    elif "_dz" in output_name:
        title = "z [cm]"
    elif "_dxy" in output_name:
        title = "d0 [cm]"
    elif "_pt" in output_name:
        title = "p_{T} [GeV]"
    else:
        title = "#eta"
    eff.GetXaxis().SetTitle(title)
    eff.GetYaxis().SetTitle("Efficiency")
    eff.GetXaxis().SetTitleSize(0.05)
    eff.GetYaxis().SetTitleSize(0.05)
    eff.GetXaxis().SetLabelSize(0.05)
    eff.GetYaxis().SetLabelSize(0.05)
    yaxis_max = 0

    if "fakerate" in key or "duplrate" in keys:
        leg1 = r.TLegend(0.63, 0.67, 0.93, 0.87)
    else:
        leg1 = r.TLegend(0.63, 0.18, 0.93, 0.38)

    for i in xrange(0, eff.GetN()):
        if yaxis_max < eff.GetY()[i]:
            yaxis_max = eff.GetY()[i]
    # print yaxis_max
    yaxis_min = 999
    for i in xrange(0, eff.GetN()):
        if yaxis_min > eff.GetY()[i] and eff.GetY()[i] != 0:
            yaxis_min = eff.GetY()[i]
    # print yaxis_min
    if "ptzoom" in output_name:
        eff.GetYaxis().SetRangeUser(yaxis_max - 0.12, yaxis_max + 0.02)
    elif "etazoom" in output_name:
        eff.GetYaxis().SetRangeUser(yaxis_max - 0.12, yaxis_max + 0.02)
    elif "ptmaxzoom" in output_name:
        eff.GetYaxis().SetRangeUser(yaxis_max - 0.02, yaxis_max + 0.02)
    elif "etamaxzoom" in output_name:
        eff.GetYaxis().SetRangeUser(yaxis_max - 0.02, yaxis_max + 0.02)
    else:
        eff.GetYaxis().SetRangeUser(0, 1.02)

    if "eta" in output_name:
        eff.GetXaxis().SetLimits(-2.5, 2.5)

    eff.SetTitle(parse_plot_name(output_name))
    leg1.AddEntry(eff,"TC", "ep")
    # Label
    t = r.TLatex()
    t.SetTextAlign(11) # align bottom left corner of text
    t.SetTextColor(r.kBlack)
    t.SetTextSize(0.04)
    x = r.gPad.GetX1() + r.gPad.GetLeftMargin()
    y = r.gPad.GetY2() - r.gPad.GetTopMargin() + 0.01
    sample_name_label = "Sample: " + sample_name + "   Version tag:" + version_tag
    t.DrawLatexNDC(x,y,"#scale[0.9]{#font[52]{%s}}" % sample_name_label)

    tce_eff = tce_tgraphs[key]
    tce_eff.SetMarkerStyle(ms[0])
    # tce_eff.SetMarkerStyle(19)
    tce_eff.SetMarkerSize(1.2)
    tce_eff.SetLineWidth(1)
    tce_eff.SetMarkerColor(cs[0])
    tce_eff.SetLineColor(cs[0])
    tce_eff.Draw("ep")
    leg1.AddEntry(tce_eff, "TE", "ep")

    leg1.Draw()
    # Save
    c1.SetGrid()
    c1.SaveAs("{}".format(output_name))
    c1.SaveAs("{}".format(output_name.replace(".pdf", ".png")))

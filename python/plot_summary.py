#!/bin/env python

import ROOT as r
import sys
import os

def usage():
    print "Usage:"
    print ""
    print "  python {} sample"
    print ""

try:
    sample = sys.argv[1]
except:
    usage()

def set_axis(eff, xmin=0.8, variable="pt"):
    if variable == "pt":
        eff.GetXaxis().SetTitleOffset(eff.GetXaxis().GetTitleOffset() * 1.2)
        eff.GetXaxis().SetMoreLogLabels()
        eff.GetXaxis().SetRangeUser(xmin, eff.GetXaxis().GetXmax())
        print "here"
    elif variable == "ptzoom":
        eff.GetXaxis().SetTitleOffset(eff.GetXaxis().GetTitleOffset() * 1.2)
        eff.GetXaxis().SetMoreLogLabels()
        eff.GetXaxis().SetRangeUser(xmin, eff.GetXaxis().GetXmax())
        print "here"
    elif variable == "dxy":
        # eff.Set(eff.GetN() + 1)
        # eff.SetPoint(eff.GetN()-1, 20, 0)
        # eff.SetPoint(eff.GetN()-2, 20, 0)
        # xmin = eff.GetX()[0]
        # eff.Print("all")
        # print xmin
        eff.GetXaxis().SetTitleOffset(eff.GetXaxis().GetTitleOffset() * 1.2)
        eff.GetXaxis().SetLimits(-10, 10)
    elif variable == "eta":
        # eff.Print("all")
        # eff.GetXaxis().SetTitleOffset(eff.GetXaxis().GetTitleOffset() * 1.2)
        # eff.Set(eff.GetN() + 2)
        # eff.SetPoint(eff.GetN()-1, -2.5, 0)
        # eff.SetPoint(eff.GetN()-2, 2.5, 0)
        eff.GetYaxis().SetRangeUser(0, 1.05)
        eff.GetXaxis().SetRangeUser(-2.5, 2.5)
        eff.GetXaxis().SetLimits(-2.5, 2.5)

######
# Open TFiles
######
md_tfile = r.TFile("eff_1.root")
sg_tfile = r.TFile("eff_2.root")
tl_tfile = r.TFile("eff_3.root")
tc_tfile = r.TFile("eff_4.root")

######
# Constants
######
track_types = ["bbbbbb", "bbbbbe", "bbbbee", "bbbeee", "bbeeee", "beeeee"]
colors = [1, 2, 4, 1, 2, 4]
marker_types = [20, 21, 22, 24, 25, 26]

######
# Canvas
######
c1 = r.TCanvas("", "", 800, 600)
c1.SetBottomMargin(0.15)
c1.SetLeftMargin(0.15)
c1.SetTopMargin(0.10)
# c1.SetRightMargin(0.05)
c1.SetRightMargin(0.16)

def draw(tfile, nlayer, objname, objnamelong, xmin=0.8, variable="pt"):
    # Pt
    for itype, track_type in enumerate(track_types):
        effs_x = []
        if variable == "pt":
            legend = r.TLegend(0.85,0.2,1.0,0.6)
        elif variable == "dxy":
            legend = r.TLegend(0.85,0.2,1.0,0.6)
        else:
            legend = r.TLegend(0.85,0.2,1.0,0.6)
        for ilayer in range(nlayer):
            print "{}_eff_{}_{}_by_layer{}.pdf".format(objname, track_type, variable, ilayer)
            effs_x.append(tfile.Get("{}_eff_{}_{}_by_layer{}.pdf".format(objname, track_type, variable, ilayer)))
            effs_x[-1].SetMarkerColor(colors[ilayer])
            effs_x[-1].SetLineColor(colors[ilayer])
            effs_x[-1].SetLineWidth(1)
            effs_x[-1].SetMarkerStyle(marker_types[ilayer])
            effs_x[-1].SetMarkerSize(0.75)
            mult = 6-nlayer + 1
            layercombo = ""
            for i in range(ilayer, ilayer + mult):
                layercombo += "{}".format(i+1)
            legend.AddEntry(effs_x[-1], "L{}".format(layercombo), "epZ")
    
        effs_x[0].SetTitle("{} Algorithmic Efficiency for {} track".format(objnamelong, track_type))
        effs_x[0].Draw("epaZ")
        effs_x[0].GetYaxis().SetTitle("Efficiency")
        set_axis(effs_x[0], xmin, variable)
        for ilayer in range(1, nlayer):
            effs_x[ilayer].Draw("epZ")
        legend.Draw()
        if variable == "pt":
            if effs_x[0].GetXaxis().GetXmax() > 4:
                c1.SetLogx()
        elif variable == "ptzoom":
            if effs_x[0].GetXaxis().GetXmax() > 4:
                c1.SetLogx()
        else:
            c1.SetLogx(0)
        c1.SetGrid()
        c1.SaveAs("plots_{}/{}eff/summary_{}_algo_eff_{}_{}.pdf".format(sample, objname, objname, variable, track_type))
        c1.SaveAs("plots_{}/{}eff/summary_{}_algo_eff_{}_{}.png".format(sample, objname, objname, variable, track_type))

def draw_tc(tfile, objname, objnamelong, variable="pt"):
    # Pt
    effs_x = []
    if variable == "pt":
        legend = r.TLegend(0.85,0.2,1.0,0.6)
    elif variable == "dxy":
        legend = r.TLegend(0.85,0.2,1.0,0.6)
    else:
        legend = r.TLegend(0.85,0.2,1.0,0.6)
    for itype, track_type in enumerate(track_types):
        for ilayer in range(1):
            print "{}_eff_{}_{}_by_layer{}.pdf".format(objname, track_type, variable, ilayer)
            effs_x.append(tfile.Get("{}_eff_{}_{}_by_layer{}.pdf".format(objname, track_type, variable, ilayer)))
            effs_x[-1].SetMarkerColor(colors[itype])
            effs_x[-1].SetLineColor(colors[itype])
            effs_x[-1].SetLineWidth(1)
            effs_x[-1].SetMarkerStyle(marker_types[itype])
            effs_x[-1].SetMarkerSize(0.75)
            legend.AddEntry(effs_x[-1], "{}".format(track_type), "epZ")
    
    effs_x[0].SetTitle("{} Algorithmic Efficiency for Track Candidates".format(objnamelong, track_type))
    effs_x[0].Draw("epaZ")
    effs_x[0].GetYaxis().SetTitle("Efficiency")
    set_axis(effs_x[0], 0.8, variable)
    for itype in range(1, len(track_types)):
        effs_x[itype].Draw("epZ")
    legend.Draw()
    if variable == "pt":
        if effs_x[0].GetXaxis().GetXmax() > 7:
            c1.SetLogx()
    elif variable == "ptzoom":
        if effs_x[0].GetXaxis().GetXmax() > 7:
            c1.SetLogx()
    else:
        c1.SetLogx(0)
    c1.SetGrid()
    c1.SaveAs("plots_{}/{}eff/summary_{}_algo_eff_{}.pdf".format(sample, objname, objname, variable))
    c1.SaveAs("plots_{}/{}eff/summary_{}_algo_eff_{}.png".format(sample, objname, objname, variable))


######

draw(md_tfile, 6, "md", "Mini-Doublet", 0.5, "pt")
draw(sg_tfile, 5, "sg", "Segment", 0.8, "pt")
draw(tl_tfile, 3, "tl", "Tracklet", 0.8, "pt")
draw(md_tfile, 6, "md", "Mini-Doublet", 0.5, "eta")
draw(sg_tfile, 5, "sg", "Segment", 0.8, "eta")
draw(tl_tfile, 3, "tl", "Tracklet", 0.8, "eta")
draw(md_tfile, 6, "md", "Mini-Doublet", 0.5, "dxy") # 0.5 does nothing
draw(sg_tfile, 5, "sg", "Segment", 0.5, "dxy") # 0.5 does nothing
draw(tl_tfile, 3, "tl", "Tracklet", 0.5, "dxy") # 0.5 does nothing
draw_tc(tc_tfile, "tc", "Track Candidate", "pt")
draw_tc(tc_tfile, "tc", "Track Candidate", "ptzoom")
draw_tc(tc_tfile, "tc", "Track Candidate", "eta")
draw_tc(tc_tfile, "tc", "Track Candidate", "dxy")


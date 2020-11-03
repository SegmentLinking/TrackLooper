#!/bin/env python

import plottery_wrapper as p
import ROOT as r

def draw_eff(num, den, output_name):

    print num.GetNbinsX()
    num.Print("all")
    print den.GetNbinsX()
    den.Print("all")

    teff = r.TEfficiency(num, den)
    eff = teff.CreateGraph()
    #eff.Print("all")
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
    elif "_z" in output_name:
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
    eff.GetXaxis().SetLabelSize(0.05)
    eff.GetYaxis().SetLabelSize(0.05)
    yaxis_max = 0
    for i in xrange(0, eff.GetN()):
        if yaxis_max < eff.GetY()[i]:
            yaxis_max = eff.GetY()[i]
    print yaxis_max
    yaxis_min = 999
    for i in xrange(0, eff.GetN()):
        if yaxis_min > eff.GetY()[i] and eff.GetY()[i] != 0:
            yaxis_min = eff.GetY()[i]
    print yaxis_min
    if "eta" in output_name:
        eff.GetXaxis().SetLimits(-2.5, 2.5)
    if "eff_eta" in output_name and "sg_" not in output_name:
        eff.GetYaxis().SetRangeUser(0, 1.005)
    if "eff_z" in output_name and "sg_" not in output_name:
        eff.GetYaxis().SetRangeUser(0.98, 1.02)
    if "barrelflat_eta" in output_name:
        eff.GetYaxis().SetRangeUser(0.97, 1.03)
    if "eff_eta" in output_name and "sg_" in output_name:
        eff.GetYaxis().SetRangeUser(0.9, 1.005)
    if "eff_z" in output_name and "sg_" in output_name:
        eff.GetYaxis().SetRangeUser(0.98, 1.02)
    if "eff_ptzoom" in output_name:
        eff.GetYaxis().SetRangeUser(yaxis_max - 0.12, yaxis_max + 0.02)
    if "eff_etazoom" in output_name:
        eff.GetYaxis().SetRangeUser(yaxis_min - 0.02, yaxis_max + 0.02)
    if "_ptzoom" in output_name and "md_" in output_name:
        eff.GetYaxis().SetRangeUser(0.93, 1.05)
    if "_ptzoom" in output_name and "sg_" in output_name:
        eff.GetYaxis().SetRangeUser(0.90, 1.05)
    if "_ptzoom" in output_name and "tl_" in output_name:
        eff.GetYaxis().SetRangeUser(0.87, 1.05)
    if "_ptzoom" in output_name and "tc_" in output_name:
        eff.GetYaxis().SetRangeUser(0.82, 1.05)
    # if "_ptzoom" in output_name and "md_" in output_name:
    #     eff.GetYaxis().SetRangeUser(0.98, 1.02)
    # if "_ptzoom" in output_name and "tl_" in output_name:
    #     eff.GetYaxis().SetRangeUser(0.9, 1.1)
    # if "_ptzoom" in output_name and "tc_" in output_name:
    #     eff.GetYaxis().SetRangeUser(0.9, 1.1)
    if "eff_eta" in output_name and "tl_" in output_name:
        eff.GetYaxis().SetRangeUser(0.9, 1.005)
    if "_pt_by" in output_name and "tc_" in output_name:
        eff.GetYaxis().SetRangeUser(0.0, 1.1)
    if "_etazoom_by" in output_name:
        eff.GetYaxis().SetRangeUser(yaxis_min - 0.02, yaxis_max + 0.02)
    if "_pt_mtv" in output_name and "tc_" in output_name:
        eff.GetYaxis().SetRangeUser(0.0, 1.1)
    if "fr_eta" in output_name:
        eff.GetYaxis().SetRangeUser(0.0, yaxis_max + 0.02)
    if "eff_eta" in output_name and "tl_" in output_name:
        eff.GetYaxis().SetRangeUser(0.0, 1.005)
    if "_etazoom_mtv" in output_name:
        eff.GetYaxis().SetRangeUser(0.9, 1.005)
        # eff.GetXaxis().SetLimits(2, 2.5)
    if "md_eff_b" in output_name and "eta" in output_name:
        eff.GetYaxis().SetRangeUser(0.0, 1.01)
    if "tl_eff_b" in output_name and "eta" in output_name:
        eff.GetYaxis().SetRangeUser(0.0, 1.01)
    c1.SetGrid()
    c1.SaveAs("{}".format(output_name.replace(".pdf", "_eff.pdf")))
    c1.SaveAs("{}".format(output_name.replace(".pdf", "_eff.png")))
    eff.SetName(output_name.replace(".png",""))
    return eff



if __name__ == "__main__":

    p.dump_plot(
            fnames=["debug.root"],
            dirname="dev_plots"
            )

    f = r.TFile("debug.root")
    cutname = "Root__"
    pt_denom = f.Get(cutname + "pixel_pt_denom")
    pt_denom_P = f.Get(cutname + "pixel_pt_denom_P")
    pt_numer_P = f.Get(cutname + "pixel_pt_numer_P")
    pt_numer_PB1B2 = f.Get(cutname + "pixel_pt_numer_PB1B2")
    pt_numer_PB2B3 = f.Get(cutname + "pixel_pt_numer_PB2B3")
    pt_numer_PB3B4 = f.Get(cutname + "pixel_pt_numer_PB3B4")
    pt_numer_PB4B5 = f.Get(cutname + "pixel_pt_numer_PB4B5")
    pt_numer_PB5B6 = f.Get(cutname + "pixel_pt_numer_PB5B6")

    draw_eff(pt_numer_P.Clone(), pt_denom_P.Clone(), "dev_plots/pixel_tracklet_eff_pt_P.pdf")
    draw_eff(pt_numer_PB1B2.Clone(), pt_denom.Clone(), "dev_plots/pixel_tracklet_eff_pt_PB1B2.pdf")
    draw_eff(pt_numer_PB2B3.Clone(), pt_denom.Clone(), "dev_plots/pixel_tracklet_eff_pt_PB2B3.pdf")
    draw_eff(pt_numer_PB3B4.Clone(), pt_denom.Clone(), "dev_plots/pixel_tracklet_eff_pt_PB3B4.pdf")
    draw_eff(pt_numer_PB4B5.Clone(), pt_denom.Clone(), "dev_plots/pixel_tracklet_eff_pt_PB4B5.pdf")
    draw_eff(pt_numer_PB5B6.Clone(), pt_denom.Clone(), "dev_plots/pixel_tracklet_eff_pt_PB5B6.pdf")

    draw_eff(pt_numer_PB1B2.Clone(), pt_denom.Clone(), "dev_plots/pixel_tracklet_eff_ptzoom_PB1B2.pdf")
    draw_eff(pt_numer_PB2B3.Clone(), pt_denom.Clone(), "dev_plots/pixel_tracklet_eff_ptzoom_PB2B3.pdf")
    draw_eff(pt_numer_PB3B4.Clone(), pt_denom.Clone(), "dev_plots/pixel_tracklet_eff_ptzoom_PB3B4.pdf")
    draw_eff(pt_numer_PB4B5.Clone(), pt_denom.Clone(), "dev_plots/pixel_tracklet_eff_ptzoom_PB4B5.pdf")
    draw_eff(pt_numer_PB5B6.Clone(), pt_denom.Clone(), "dev_plots/pixel_tracklet_eff_ptzoom_PB5B6.pdf")

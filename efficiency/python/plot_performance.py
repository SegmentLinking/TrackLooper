#!/bin/env python

import ROOT as r
from array import array
import os
import sys
from math import sqrt

r.gROOT.SetBatch(True)

def parse_plot_name(output_name):
    if "fakerate" in output_name:
        rtnstr = ["Fake Rate of"]
    elif "duplrate" in output_name:
        rtnstr = ["Duplicate Rate of"]
    elif "inefficiency" in output_name:
        rtnstr = ["Inefficiency of"]
    else:
        rtnstr = ["Efficiency of"]
    if "MD_" in output_name:
        rtnstr.append("Mini-Doublet")
    elif "LS_" in output_name and "pLS" not in output_name:
        rtnstr.append("Line Segment")
    elif "pT4_" in output_name:
        rtnstr.append("Quadruplet w/ Pixel LS")
    elif "T4_" in output_name:
        rtnstr.append("Quadruplet w/o gap")
    elif "T4x_" in output_name:
        rtnstr.append("Quadruplet w/ gap")
    elif "pT3_" in output_name:
        rtnstr.append("Pixel Triplet")
    elif "pT5_" in output_name:
        rtnstr.append("Pixel Quintuplet")
    elif "T3_" in output_name:
        rtnstr.append("Triplet")
    elif "TC_" in output_name:
        rtnstr.append("Track Candidate")
    elif "T4s_" in output_name:
        rtnstr.append("Quadruplet w/ or w/o gap")
    elif "pLS_" in output_name:
        rtnstr.append("Pixel Line Segment")
    elif "T5_" in output_name:
        rtnstr.append("Quintuplet")
    types = "of type " + os.path.basename(output_name).split("_")[1]
    if "AllTypes" in types:
        types = "of all types"
    if "Set1Types" in types:
        types = "of set 1 types"
    rtnstr.append(types)
    return " ".join(rtnstr)


def draw_ratio(num, den, output_name, sample_name, version_tag, outputfile=None):

    if "scalar" in output_name and "ptscalar" not in output_name:
        num.Rebin(180)
        den.Rebin(180)

    if "coarse" in output_name and "ptcoarse" not in output_name:
        num.Rebin(6)
        den.Rebin(6)
    # if "eta" in output_name and "etacoarse" not in output_name:
    #     num.Rebin(2)
    #     den.Rebin(2)
    if "pt" in output_name:
        overFlowBin = num.GetBinContent(num.GetNbinsX() + 1)
        lastBin = num.GetBinContent(num.GetNbinsX())

        num.SetBinContent(num.GetNbinsX(), lastBin + overFlowBin)
        num.SetBinError(num.GetNbinsX(), sqrt(lastBin + overFlowBin))
        
        overFlowBin = den.GetBinContent(den.GetNbinsX() + 1)
        lastBin = den.GetBinContent(den.GetNbinsX())

        den.SetBinContent(den.GetNbinsX(), lastBin + overFlowBin)
        den.SetBinError(den.GetNbinsX(), sqrt(lastBin + overFlowBin))

    teff = r.TEfficiency(num, den)
    eff = teff.CreateGraph()
    c1 = r.TCanvas()
    c1.SetBottomMargin(0.15)
    c1.SetLeftMargin(0.15)
    c1.SetTopMargin(0.22)
    c1.SetRightMargin(0.15)
    if "_pt" in output_name:
        c1.SetLogx()
    eff.Draw("epa")
    eff.SetMarkerStyle(19)
    eff.SetMarkerSize(1.2)
    eff.SetLineWidth(2)

    def set_label(eff, raw_number):
        if "phi" in output_name:
            title = "#phi"
        elif "_dz" in output_name:
            title = "z [cm]"
        elif "_dxy" in output_name:
            title = "d0 [cm]"
        elif "_pt" in output_name:
            title = "p_{T} [GeV]"
        elif "_hit" in output_name:
            title = "hits"
        elif "_lay" in output_name:
            title = "layers"
        else:
            title = "#eta"
        eff.GetXaxis().SetTitle(title)
        if "fakerate" in output_name:
            eff.GetYaxis().SetTitle("Fake Rate")
        elif "duplrate" in output_name:
            eff.GetYaxis().SetTitle("Duplicate Rate")
        elif "inefficiency" in output_name:
            eff.GetYaxis().SetTitle("Inefficiency")
        else:
            eff.GetYaxis().SetTitle("Efficiency")
        if raw_number:
            eff.GetYaxis().SetTitle("# of objects of interest")
        eff.GetXaxis().SetTitleSize(0.05)
        eff.GetYaxis().SetTitleSize(0.05)
        eff.GetXaxis().SetLabelSize(0.05)
        eff.GetYaxis().SetLabelSize(0.05)
    set_label(eff, raw_number=False)

    yaxis_max = 0
    for i in xrange(0, eff.GetN()):
        if yaxis_max < eff.GetY()[i]:
            yaxis_max = eff.GetY()[i]
    # print yaxis_max
    yaxis_min = 999
    for i in xrange(0, eff.GetN()):
        if yaxis_min > eff.GetY()[i] and eff.GetY()[i] != 0:
            yaxis_min = eff.GetY()[i]
    # print yaxis_min
    # if "maxzoom" in output_name:
    #     eff.GetYaxis().SetRangeUser(yaxis_max - 0.02, yaxis_max + 0.02)
    # elif "zoom" in output_name:
    #     eff.GetYaxis().SetRangeUser(yaxis_max - 0.12, yaxis_max + 0.02)
    #if "ptzoom" in output_name:
    #    eff.GetYaxis().SetRangeUser(yaxis_max - 0.12, yaxis_max + 0.02)
    #elif "etazoom" in output_name:
    #    eff.GetYaxis().SetRangeUser(yaxis_max - 0.12, yaxis_max + 0.02)
    #elif "ptmaxzoom" in output_name:
    #    eff.GetYaxis().SetRangeUser(yaxis_max - 0.02, yaxis_max + 0.02)
    #elif "etamaxzoom" in output_name:
    #    eff.GetYaxis().SetRangeUser(yaxis_max - 0.02, yaxis_max + 0.02)
    #elif "layerszoom" in output_name:
    #    eff.GetYaxis().SetRangeUser(yaxis_max - 0.12, yaxis_max + 0.12)
    #elif "layersgapzoom" in output_name:
    #    eff.GetYaxis().SetRangeUser(yaxis_max - 0.12, yaxis_max + 0.12)
    #elif "layersmaxzoom" in output_name:
    #    eff.GetYaxis().SetRangeUser(yaxis_max - 0.02, yaxis_max + 0.02)
    #elif "layersgapmaxzoom" in output_name:
    #    eff.GetYaxis().SetRangeUser(yaxis_max - 0.02, yaxis_max + 0.02)
    # else:
    #     eff.GetYaxis().SetRangeUser(0, 1.02)

    if "zoom" not in output_name:
        eff.GetYaxis().SetRangeUser(0, 1.02)
    else:
        eff.GetYaxis().SetRangeUser(0.6, 1.02)

    if "eta" in output_name:
        eff.GetXaxis().SetLimits(-4.5, 4.5)

    eff.SetTitle(parse_plot_name(output_name))

    def draw_label():
        # Label
        t = r.TLatex()
        t.SetTextAlign(11) # align bottom left corner of text
        t.SetTextColor(r.kBlack)
        t.SetTextSize(0.04)
        x = r.gPad.GetX1() + r.gPad.GetLeftMargin()
        y = r.gPad.GetY2() - r.gPad.GetTopMargin() + 0.09 + 0.03
        sample_name_label = "Sample: " + sample_name + "   Version tag:" + version_tag
        t.DrawLatexNDC(x,y,"#scale[0.9]{#font[42]{%s}}" % sample_name_label)
        x = r.gPad.GetX1() + r.gPad.GetLeftMargin()
        y = r.gPad.GetY2() - r.gPad.GetTopMargin() + 0.045 + 0.03
        if "_pt" in output_name:
            fiducial_label = "|#eta| < 4.5, |Vtx_{z}| < 30 cm, |Vtx_{xy}| < 2.5 cm"
        elif "_eta" in output_name:
            fiducial_label = "p_{T} > 0.9 GeV, |Vtx_{z}| < 30 cm, |Vtx_{xy}| < 2.5 cm"
        elif "_dz" in output_name:
            fiducial_label = "|#eta| < 4.5, p_{T} > 0.9 GeV, |Vtx_{xy}| < 2.5 cm"
        elif "_dxy" in output_name:
            fiducial_label = "|#eta| < 4.5, p_{T} > 0.9 GeV, |Vtx_{z}| < 30 cm"
        #elif "_lay" in output_name:
        #    fiducial_label = "|#eta| < 4.5, p_{T} > 0.9 GeV, |Vtx_{z}| < 30 cm, |Vtx_{xy}| < 2.5 cm"
        else:
            fiducial_label = "|#eta| < 4.5, p_{T} > 0.9 GeV, |Vtx_{z}| < 30 cm, |Vtx_{xy}| < 2.5 cm"
        if "fakerate" in output_name or "duplrate" in output_name:
            if "_pt" in output_name:
                fiducial_label = "|#eta| < 4.5"
            elif "_eta" in output_name:
                fiducial_label = "p_{T} > 0.9 GeV"
            else:
                fiducial_label = "|#eta| < 4.5, p_{T} > 0.9 GeV"
        t.DrawLatexNDC(x,y,"#scale[0.9]{#font[42]{%s}}" % fiducial_label)
        cms_label = "Simulation"
        x = r.gPad.GetX1() + r.gPad.GetLeftMargin()
        y = r.gPad.GetY2() - r.gPad.GetTopMargin() + 0.005
        t.DrawLatexNDC(x,y,"#scale[1.25]{#font[61]{CMS}} #scale[1.1]{#font[52]{%s}}" % cms_label)
    draw_label()
    # Save
    c1.SetGrid()
    c1.SaveAs("{}".format(output_name.replace(".pdf", "_eff.pdf")))
    c1.SaveAs("{}".format(output_name.replace(".pdf", "_eff.png")))
    eff.SetName(output_name.replace(".png",""))
    if outputfile:
        outputfile.cd()
        basename = os.path.basename(output_name)
        outputname = basename.replace(".pdf","")
        print(outputname)
        eff.SetName(outputname)
        eff.Write()
        eff_num = r.TGraphAsymmErrors(num)
        eff_den = r.TGraphAsymmErrors(den)
        eff_num.SetName(outputname+"_num")
        eff_den.SetName(outputname+"_den")
        eff_num.Write()
        eff_den.Write()
        outputfile.ls()

    set_label(num, raw_number=True)
    num.Draw("hist")
    draw_label()
    c1.SaveAs("{}".format(output_name.replace(".pdf", "_num.pdf")))
    c1.SaveAs("{}".format(output_name.replace(".pdf", "_num.png")))

    set_label(den, raw_number=True)
    den.Draw("hist")
    draw_label()
    c1.SaveAs("{}".format(output_name.replace(".pdf", "_den.pdf")))
    c1.SaveAs("{}".format(output_name.replace(".pdf", "_den.png")))

    return eff

if __name__ == "__main__":

    root_file_name = "num_den_histograms.root"
    sample_name = sys.argv[1]
    version_tag = sys.argv[2]

    f = r.TFile(root_file_name)
    of = r.TFile("efficiencies.root", "RECREATE")

    num_den_pairs = []
    for key in f.GetListOfKeys():
        if "denom" in key.GetName():
            continue
        # if "Set4" not in key.GetName():
        #     continue
        if "TC_All" not in key.GetName() and "T4s_All" not in key.GetName() and "T3_All" not in key.GetName() and "pLS" not in key.GetName() and "T5" not in key.GetName() and "pT4_All" not in key.GetName() and "pT3_All" not in key.GetName() and "pT5_All" not in key.GetName():
            continue
        # if "pLS_P" not in key.GetName():
        #     continue
        # if "pix_P" not in key.GetName():
        #     continue
        numer_name = key.GetName()
        print(numer_name)
        denom_name = numer_name.replace("numer", "denom")
        nice_name = numer_name.replace("Root__", "")
        nice_name = nice_name.replace("h_numer", "")
        num_den_pairs.append((numer_name, denom_name, nice_name))

    for numer_histname, denom_histname, nice_name in num_den_pairs:
        numer = f.Get(numer_histname)
        denom = f.Get(denom_histname)
        draw_ratio(numer.Clone(), denom.Clone(), "plots/mtv/{}.pdf".format(nice_name), sample_name, version_tag, of)
        draw_ratio(numer.Clone(), denom.Clone(), "plots/mtv/{}zoom.pdf".format(nice_name), sample_name, version_tag, of)
        draw_ratio(numer.Clone(), denom.Clone(), "plots/mtv/{}zoomcoarse.pdf".format(nice_name), sample_name, version_tag, of)
        draw_ratio(numer.Clone(), denom.Clone(), "plots/mtv/{}maxzoom.pdf".format(nice_name), sample_name, version_tag, of)
        draw_ratio(numer.Clone(), denom.Clone(), "plots/mtv/{}maxzoomcoarse.pdf".format(nice_name), sample_name, version_tag, of)
        draw_ratio(numer.Clone(), denom.Clone(), "plots/mtv/{}scalar.pdf".format(nice_name), sample_name, version_tag, of)
        draw_ratio(numer.Clone(), denom.Clone(), "plots/mtv/{}coarse.pdf".format(nice_name), sample_name, version_tag, of)

    of.Write()
    of.Close()


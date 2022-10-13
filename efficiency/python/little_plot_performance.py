#!/bin/env python

import argparse 
import ROOT as r
from array import array
import os
import sys
from math import sqrt

# ptcut = 1.5
# etacut = 2.4
ptcut = 0.9
etacut = 4.5

r.gROOT.SetBatch(True)

def parse_plot_name(output_name):
    if "fake" in output_name:
        rtnstr = ["Fake Rate of"]
    elif "dup" in output_name:
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
    elif "TCE_" in output_name:
        rtnstr.append("Extended Track")
    elif "T3T3_" in output_name:
        rtnstr.append("T3T3 Extensions")
    elif "pureTCE_" in output_name:
        rtnstr.append("Pure Extensions")
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


#def draw_label():
#    # Label
#    t = r.TLatex()
#    t.SetTextAlign(11) # align bottom left corner of text
#    t.SetTextColor(r.kBlack)
#    t.SetTextSize(0.04)
#    x = r.gPad.GetX1() + r.gPad.GetLeftMargin()
#    y = r.gPad.GetY2() - r.gPad.GetTopMargin() + 0.09 + 0.03
#    sample_name_label = "Sample: " + sample_name + "   Version tag:" + version_tag
#    t.DrawLatexNDC(x,y,"#scale[0.9]{#font[42]{%s}}" % sample_name_label)
#    x = r.gPad.GetX1() + r.gPad.GetLeftMargin()
#    y = r.gPad.GetY2() - r.gPad.GetTopMargin() + 0.045 + 0.03
#    if "_pt" in output_name:
#        fiducial_label = "|#eta| < {eta}, |Vtx_{{z}}| < 30 cm, |Vtx_{{xy}}| < 2.5 cm".format(eta=etacut)
#    elif "_eta" in output_name:
#        fiducial_label = "p_{{T}} > {pt} GeV, |Vtx_{{z}}| < 30 cm, |Vtx_{{xy}}| < 2.5 cm".format(pt=ptcut)
#    elif "_dz" in output_name:
#        fiducial_label = "|#eta| < {eta}, p_{{T}} > {pt} GeV, |Vtx_{{xy}}| < 2.5 cm".format(pt=ptcut, eta=etacut)
#    elif "_dxy" in output_name:
#        fiducial_label = "|#eta| < {eta}, p_{{T}} > {pt} GeV, |Vtx_{{z}}| < 30 cm".format(pt=ptcut, eta=etacut)
#    #elif "_lay" in output_name:
#    #    fiducial_label = "|#eta| < {eta}, p_{{T}} > {pt} GeV, |Vtx_{{z}}| < 30 cm, |Vtx_{{xy}}| < 2.5 cm".format(pt=ptcut, eta=etacut)
#    else:
#        fiducial_label = "|#eta| < {eta}, p_{{T}} > {pt} GeV, |Vtx_{{z}}| < 30 cm, |Vtx_{{xy}}| < 2.5 cm".format(pt=ptcut, eta=etacut)
#    if "fakerate" in output_name or "duplrate" in output_name:
#        if "_pt" in output_name:
#            fiducial_label = "|#eta| < {eta}".format(eta=etacut)
#        elif "_eta" in output_name:
#            fiducial_label = "p_{{T}} > {pt} GeV".format(pt=ptcut)
#        else:
#            fiducial_label = "|#eta| < {eta}, p_{{T}} > {pt} GeV".format(pt=ptcut, eta=etacut)
#    t.DrawLatexNDC(x,y,"#scale[0.9]{#font[42]{%s}}" % fiducial_label)
#    cms_label = "Simulation"
#    x = r.gPad.GetX1() + r.gPad.GetLeftMargin()
#    y = r.gPad.GetY2() - r.gPad.GetTopMargin() + 0.005
#    t.DrawLatexNDC(x,y,"#scale[1.25]{#font[61]{CMS}} #scale[1.1]{#font[52]{%s}}" % cms_label)
#def set_label(eff, raw_number):
#    if "phi" in output_name:
#        title = "#phi"
#    elif "_dz" in output_name:
#        title = "z [cm]"
#    elif "_dxy" in output_name:
#        title = "d0 [cm]"
#    elif "_pt" in output_name:
#        title = "p_{T} [GeV]"
#    elif "_hit" in output_name:
#        title = "hits"
#    elif "_lay" in output_name:
#        title = "layers"
#    else:
#        title = "#eta"
#    eff.GetXaxis().SetTitle(title)
#    if "fakerate" in output_name:
#        eff.GetYaxis().SetTitle("Fake Rate")
#    elif "duplrate" in output_name:
#        eff.GetYaxis().SetTitle("Duplicate Rate")
#    elif "inefficiency" in output_name:
#        eff.GetYaxis().SetTitle("Inefficiency")
#    else:
#        eff.GetYaxis().SetTitle("Efficiency")
#    if raw_number:
#        eff.GetYaxis().SetTitle("# of objects of interest")
#    eff.GetXaxis().SetTitleSize(0.05)
#    eff.GetYaxis().SetTitleSize(0.05)
#    eff.GetXaxis().SetLabelSize(0.05)
#    eff.GetYaxis().SetLabelSize(0.05)
def draw_stack(nums, den, output_name, sample_name, version_tag, outputfile=None):

    if "scalar" in output_name and "ptscalar" not in output_name:
        for i in range(len(nums)):
            nums[i].Rebin(180)
        den.Rebin(180)

    if "coarse" in output_name and "ptcoarse" not in output_name:
        for i in range(len(nums)):
            nums[i].Rebin(6)
        den.Rebin(6)
    # if "eta" in output_name and "etacoarse" not in output_name:
    #     num.Rebin(2)
    #     den.Rebin(2)
    if "pt" in output_name:
        for i in range(len(nums)):
            overFlowBin = nums[i].GetBinContent(nums[i].GetNbinsX() + 1)
            lastBin = nums[i].GetBinContent(nums[i].GetNbinsX())
            nums[i].SetBinContent(nums[i].GetNbinsX(), lastBin + overFlowBin)
            nums[i].SetBinError(nums[i].GetNbinsX(), sqrt(lastBin + overFlowBin))

        overFlowBin = den.GetBinContent(den.GetNbinsX() + 1)
        lastBin = den.GetBinContent(den.GetNbinsX())

        den.SetBinContent(den.GetNbinsX(), lastBin + overFlowBin)
        den.SetBinError(den.GetNbinsX(), sqrt(lastBin + overFlowBin))

    num1, num2, num3, num4, num5 = nums
    teff = r.TEfficiency(num1, den)
    teff2 = r.TEfficiency(num2, den)
    teff3 = r.TEfficiency(num3, den)
    teff4 = r.TEfficiency(num4, den)
    teff5 = r.TEfficiency(num5, den)
    eff = teff.CreateGraph()
    eff2 = teff2.CreateGraph()
    eff3 = teff3.CreateGraph()
    eff4 = teff4.CreateGraph()
    eff5 = teff5.CreateGraph()
    c1 = r.TCanvas()
    c1.SetBottomMargin(0.15)
    c1.SetLeftMargin(0.15)
    c1.SetTopMargin(0.22)
    c1.SetRightMargin(0.15)
    if "_pt" in output_name:
        c1.SetLogx()
    eff.Draw("epa")
    eff2.Draw("epsame")
    eff3.Draw("epsame")
    eff4.Draw("epsame")
    eff5.Draw("epsame")
    eff.SetMarkerColor(1)
    eff.SetLineColor(1)
    eff.SetMarkerStyle(19)
    eff.SetMarkerSize(1.2)
    eff.SetLineWidth(2)
    eff2.SetMarkerStyle(26)
    eff2.SetMarkerSize(1.2)
    eff2.SetLineWidth(2)
    eff2.SetMarkerColor(2)
    eff2.SetLineColor(2)
    eff3.SetMarkerStyle(28)
    eff3.SetMarkerSize(1.2)
    eff3.SetLineWidth(2)
    eff3.SetMarkerColor(3)
    eff3.SetLineColor(3)
    eff4.SetMarkerStyle(24)
    eff4.SetMarkerSize(1.2)
    eff4.SetLineWidth(2)
    eff4.SetMarkerColor(4)
    eff4.SetLineColor(4)
    eff5.SetMarkerStyle(27)
    eff5.SetMarkerSize(1.2)
    eff5.SetLineWidth(2)
    eff5.SetMarkerColor(6)
    eff5.SetLineColor(6)

    legend = r.TLegend(0.15,0.55,0.25,0.75)
    legend.AddEntry(eff,"TC")
    legend.AddEntry(eff2,"pT5")
    legend.AddEntry(eff3,"pT3")
    legend.AddEntry(eff4,"T5")
    legend.AddEntry(eff5,"pLS")
    legend.Draw("same")

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
        if "fake" in output_name:
            eff.GetYaxis().SetTitle("Fake Rate")
        elif "dup" in output_name:
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

    if "zoom" not in output_name:
        eff.GetYaxis().SetRangeUser(0, 1.02)
    else:
        if "fakerate" in output_name:
            eff.GetYaxis().SetRangeUser(0.0, yaxis_max * 1.1)
        elif "duplrate" in output_name:
            eff.GetYaxis().SetRangeUser(0.0, yaxis_max * 1.1)
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
            fiducial_label = "|#eta| < {eta}, |Vtx_{{z}}| < 30 cm, |Vtx_{{xy}}| < 2.5 cm".format(eta=etacut)
        elif "_eta" in output_name:
            fiducial_label = "p_{{T}} > {pt} GeV, |Vtx_{{z}}| < 30 cm, |Vtx_{{xy}}| < 2.5 cm".format(pt=ptcut)
        elif "_dz" in output_name:
            fiducial_label = "|#eta| < {eta}, p_{{T}} > {pt} GeV, |Vtx_{{xy}}| < 2.5 cm".format(pt=ptcut, eta=etacut)
        elif "_dxy" in output_name:
            fiducial_label = "|#eta| < {eta}, p_{{T}} > {pt} GeV, |Vtx_{{z}}| < 30 cm".format(pt=ptcut, eta=etacut)
        #elif "_lay" in output_name:
        #    fiducial_label = "|#eta| < {eta}, p_{{T}} > {pt} GeV, |Vtx_{{z}}| < 30 cm, |Vtx_{{xy}}| < 2.5 cm".format(pt=ptcut, eta=etacut)
        else:
            fiducial_label = "|#eta| < {eta}, p_{{T}} > {pt} GeV, |Vtx_{{z}}| < 30 cm, |Vtx_{{xy}}| < 2.5 cm".format(pt=ptcut, eta=etacut)
        if "fakerate" in output_name or "duplrate" in output_name:
            if "_pt" in output_name:
                fiducial_label = "|#eta| < {eta}".format(eta=etacut)
            elif "_eta" in output_name:
                fiducial_label = "p_{{T}} > {pt} GeV".format(pt=ptcut)
            else:
                fiducial_label = "|#eta| < {eta}, p_{{T}} > {pt} GeV".format(pt=ptcut, eta=etacut)
        t.DrawLatexNDC(x,y,"#scale[0.9]{#font[42]{%s}}" % fiducial_label)
        cms_label = "Simulation"
        x = r.gPad.GetX1() + r.gPad.GetLeftMargin()
        y = r.gPad.GetY2() - r.gPad.GetTopMargin() + 0.005
        t.DrawLatexNDC(x,y,"#scale[1.25]{#font[61]{CMS}} #scale[1.1]{#font[52]{%s}}" % cms_label)
    draw_label()
    # Save
    c1.SetGrid()
    c1.SaveAs("{}".format(output_name.replace("/mtv/", "/mtv/var/")))
    c1.SaveAs("{}".format(output_name.replace("/mtv/", "/mtv/var/").replace(".pdf", ".png")))
    eff.SetName(output_name.replace(".png",""))
    if outputfile:
        outputfile.cd()
        basename = os.path.basename(output_name)
        outputname = basename.replace(".pdf","")
        # print(outputname)
        eff.SetName(outputname)
        eff.Write()
        eff_num = r.TGraphAsymmErrors(num1)
        eff_den = r.TGraphAsymmErrors(den)
        eff_num.SetName(outputname+"_num")
        eff_den.SetName(outputname+"_den")
        eff_num.Write()
        eff_den.Write()
        # outputfile.ls()

    set_label(num1, raw_number=True)
    num1.Draw("hist")
    draw_label()
    c1.SaveAs("{}".format(output_name.replace("/mtv/", "/mtv/num/").replace(".pdf", "_num.pdf")))
    c1.SaveAs("{}".format(output_name.replace("/mtv/", "/mtv/num/").replace(".pdf", "_num.png")))

    set_label(den, raw_number=True)
    den.Draw("hist")
    draw_label()
    c1.SaveAs("{}".format(output_name.replace("/mtv/", "/mtv/den/").replace(".pdf", "_den.pdf")))
    c1.SaveAs("{}".format(output_name.replace("/mtv/", "/mtv/den/").replace(".pdf", "_den.png")))

    return eff
def draw_ratio(num, den, output_name, sample_name, version_tag, outputfile=None):

    # num.Rebin(6)
    # den.Rebin(6)
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
        if "fakerate" in output_name:
            eff.GetYaxis().SetRangeUser(0.0, yaxis_max * 1.1)
        elif "duplrate" in output_name:
            eff.GetYaxis().SetRangeUser(0.0, yaxis_max * 1.1)
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
            fiducial_label = "|#eta| < {eta}, |Vtx_{{z}}| < 30 cm, |Vtx_{{xy}}| < 2.5 cm".format(eta=etacut)
        elif "_eta" in output_name:
            fiducial_label = "p_{{T}} > {pt} GeV, |Vtx_{{z}}| < 30 cm, |Vtx_{{xy}}| < 2.5 cm".format(pt=ptcut)
        elif "_dz" in output_name:
            fiducial_label = "|#eta| < {eta}, p_{{T}} > {pt} GeV, |Vtx_{{xy}}| < 2.5 cm".format(pt=ptcut, eta=etacut)
        elif "_dxy" in output_name:
            fiducial_label = "|#eta| < {eta}, p_{{T}} > {pt} GeV, |Vtx_{{z}}| < 30 cm".format(pt=ptcut, eta=etacut)
        #elif "_lay" in output_name:
        #    fiducial_label = "|#eta| < {eta}, p_{{T}} > {pt} GeV, |Vtx_{{z}}| < 30 cm, |Vtx_{{xy}}| < 2.5 cm".format(pt=ptcut, eta=etacut)
        else:
            fiducial_label = "|#eta| < {eta}, p_{{T}} > {pt} GeV, |Vtx_{{z}}| < 30 cm, |Vtx_{{xy}}| < 2.5 cm".format(pt=ptcut, eta=etacut)
        if "fakerate" in output_name or "duplrate" in output_name:
            if "_pt" in output_name:
                fiducial_label = "|#eta| < {eta}".format(eta=etacut)
            elif "_eta" in output_name:
                fiducial_label = "p_{{T}} > {pt} GeV".format(pt=ptcut)
            else:
                fiducial_label = "|#eta| < {eta}, p_{{T}} > {pt} GeV".format(pt=ptcut, eta=etacut)
        t.DrawLatexNDC(x,y,"#scale[0.9]{#font[42]{%s}}" % fiducial_label)
        cms_label = "Simulation"
        x = r.gPad.GetX1() + r.gPad.GetLeftMargin()
        y = r.gPad.GetY2() - r.gPad.GetTopMargin() + 0.005
        t.DrawLatexNDC(x,y,"#scale[1.25]{#font[61]{CMS}} #scale[1.1]{#font[52]{%s}}" % cms_label)
    draw_label()
    # Save
    c1.SetGrid()
    c1.SaveAs("{}".format(output_name.replace("/mtv/", "/mtv/var/")))
    c1.SaveAs("{}".format(output_name.replace("/mtv/", "/mtv/var/").replace(".pdf", ".png")))
    eff.SetName(output_name.replace(".png",""))
    if outputfile:
        outputfile.cd()
        basename = os.path.basename(output_name)
        outputname = basename.replace(".pdf","")
        # print(outputname)
        eff.SetName(outputname)
        eff.Write()
        eff_num = r.TGraphAsymmErrors(num)
        eff_den = r.TGraphAsymmErrors(den)
        eff_num.SetName(outputname+"_num")
        eff_den.SetName(outputname+"_den")
        eff_num.Write()
        eff_den.Write()
        # outputfile.ls()

    set_label(num, raw_number=True)
    num.Draw("hist")
    draw_label()
    c1.SaveAs("{}".format(output_name.replace("/mtv/", "/mtv/num/").replace(".pdf", "_num.pdf")))
    c1.SaveAs("{}".format(output_name.replace("/mtv/", "/mtv/num/").replace(".pdf", "_num.png")))

    set_label(den, raw_number=True)
    den.Draw("hist")
    draw_label()
    c1.SaveAs("{}".format(output_name.replace("/mtv/", "/mtv/den/").replace(".pdf", "_den.pdf")))
    c1.SaveAs("{}".format(output_name.replace("/mtv/", "/mtv/den/").replace(".pdf", "_den.png")))

    return eff

def plot_standard_performance_plots():

    # Create output directory
    os.system("mkdir -p plots/mtv/var")
    os.system("mkdir -p plots/mtv/num")
    os.system("mkdir -p plots/mtv/den")

    # Efficiency plots
    metricsuffixs = ["ef_", "fr_", "dr_"]
    ybins = ["", "zoom"]
    variables = {
            "ef_": ["pt", "eta", "phi", "dxy", "dz"],
            "fr_": ["pt", "eta", "phi"],
            "dr_": ["pt", "eta", "phi"],
            }
    xbins = {
            "pt": [""],
            "eta": ["", "coarse"],
            "phi": ["", "coarse"],
            "dxy": ["", "coarse"],
            "dz": ["", "coarse"],
            }
    types = ["TC", "pT5", "pT3", "T5", "pLS"]

    for metricsuffix in metricsuffixs:
        for variable in variables[metricsuffix]:
            for ybin in ybins:
                for xbin in xbins[variable]:
                    for typ in types:
                        if typ == "TC":
                            plot(variable, ybin, xbin, typ, metricsuffix, True)
                            plot(variable, ybin, xbin, typ, metricsuffix, False)
                        else:
                            plot(variable, ybin, xbin, typ, metricsuffix, False)

    DIR = os.path.realpath(os.path.dirname(__file__))
    os.system("cp -r {}/../misc/summary plots/".format(DIR))

def plot(variable, ybinning, xbinning, objecttype, metricsuffix, is_stack):

    metric = "eff"
    if metricsuffix == "fr_":
        metric = "fakerate"
    if metricsuffix == "dr_":
        metric = "duplrate"

    # Get denominator histogram
    denom_histname = "Root__{objecttype}_{metricsuffix}denom_{variable}".format(objecttype=objecttype, metricsuffix=metricsuffix, variable=variable)
    try:
        denom = f.Get(denom_histname).Clone()
    except:
        print(denom_histname)
        sys.exit("ERROR: Did not find denominator histogram = {}".format(denom_histname))

    # Get numerator histograms
    numer_histname = "Root__{objecttype}_{metricsuffix}numer_{variable}".format(objecttype=objecttype, metricsuffix=metricsuffix, variable=variable)
    numer = f.Get(numer_histname).Clone()
    try:
        numer = f.Get(numer_histname).Clone()
    except:
        print(numer_histname)
        sys.exit("ERROR: Did not find numerator histogram = {}".format(numer_histname))

    stack_hist_types = ["pT5", "pT3", "T5", "pLS"]
    stack_hists = []
    if is_stack:
        for stack_hist_type in stack_hist_types:
            stack_histname = numer_histname.replace("TC", stack_hist_type)
            hist = f.Get(stack_histname)
            stack_hists.append(hist.Clone())

    output_plot_name = "{objecttype}_{metric}{stackvar}_{variable}".format(objecttype=objecttype, metric=metric, stackvar="_stack" if is_stack else "", variable=variable)
    if xbinning == "coarse":
        output_plot_name += "coarse"
    if ybinning == "zoom":
        output_plot_name += "zoom"

    if is_stack:
        draw_stack(
                [numer] + stack_hists, # numerator histograms
                denom, # denominator histogram
                "plots/mtv/{0}.pdf".format(output_plot_name), # output plot name
                sample_name, # sample type
                git_hash, # version tag
                # of # TGraph output rootfile
                )
    else:

        draw_ratio(
                numer, # numerator histogram
                denom, # denominator histogram
                "plots/mtv/{0}.pdf".format(output_plot_name), # output plot name
                sample_name, # sample type
                git_hash, # version tag
                # of # TGraph output rootfile
                )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="What are we wanting to graph?")
    parser.add_argument('--input' , '-i' , dest='input' , type=str , default='num_den_hist.root' , help='input file name, -- default is input')
    parser.add_argument('--variable' , '-v' , dest='variable' , type=str , default='pt' , help='pt, eta, phi, dxy, dz, layers, hits, layersgap, -- default is pt')
    parser.add_argument('--yrange' , '-y' , dest='yrange' , type=str , default='normal' , help='yrange - default is normal')
    parser.add_argument('--xbinning' , '-x' , dest='xbinning' , type=str , default='normal' , help='xbinning - default is normal')
    parser.add_argument('--objecttype' , '-ot' , dest='objecttype' , type=str , default='TC' , help='TC, TCE, T3, pT3, T5, pT5, pureTCE, pT4, pT3, pLS -- deafult is TC')
    parser.add_argument('--metric' , '-m' , dest='metric' , type=str , default='eff' , help='metric - default is eff (e.g. eff, duplrate, fakerate)')
    parser.add_argument('--sample_name' , '-sn' , dest='sample_name' , type=str , default='DEFAULT' , help='sample name')
    parser.add_argument('--git_hash' , '-gh' , dest='git_hash' , type=str , default='12345' , help='git hash')
    parser.add_argument('--is_stack' , '-is_t' , dest='is_stack' , action="store_true", help='is stack - default is True')
    parser.add_argument('--standard_perf_plots' , '-std' , dest='std' , action="store_true", help='plot a full set of standard plots - default is True')

    args = parser.parse_args()

    #############
    variable = args.variable
    yrange = args.yrange
    xbinning = args.xbinning
    objecttype = args.objecttype
    metric = args.metric
    sample_name = args.sample_name
    git_hash = args.git_hash
    is_stack = args.is_stack
    std = args.std
    #############

    root_file_name = args.input
    f = r.TFile(root_file_name)

    if std:
        plot_standard_performance_plots()
        sys.exit()

    if is_stack:
        print("Warning! objecttype is set to \"TC\" because is_stack is True!")
        objecttype = "TC"

    # SPECIAL CARE on how we deal with different metric
    if metric == "eff": metricsuffix = "ef_"
    if metric == "duplrate": metricsuffix = "dr_"
    if metric == "fakerate": metricsuffix = "fr_"
    if metric == "eff": stackmetricsuffix = ""
    if metric == "duplrate": stackmetricsuffix = "dup"
    if metric == "fakerate": stackmetricsuffix = "fake"

    # Create output directory
    os.system("mkdir -p plots/mtv/var")
    os.system("mkdir -p plots/mtv/num")
    os.system("mkdir -p plots/mtv/den")

    # Get denominator histogram
    denom_histname = "Root__{objecttype}_{metricsuffix}denom_{variable}".format(objecttype=objecttype, metricsuffix=metricsuffix, variable=variable)
    print(denom_histname)
    denom = f.Get(denom_histname).Clone()

    # Get numerator histograms
    numer_histname = "Root__{objecttype}_{metricsuffix}numer_{variable}".format(objecttype=objecttype, metricsuffix=metricsuffix, variable=variable)
    numer = f.Get(numer_histname).Clone()

    stack_hist_types = ["pT5", "pT3", "T5", "pLS"]
    stack_hists = []
    if is_stack:
        for stack_hist_type in stack_hist_types:
            stack_histname = numer_histname.replace("TC", stack_hist_type)
            print(stack_histname)
            hist = f.Get(stack_histname)
            stack_hists.append(hist.Clone())

    output_plot_name = "{objecttype}_{metric}{stackvar}_{variable}".format(objecttype=objecttype, metric=metric, stackvar="_stack" if is_stack else "", variable=variable)
    if xbinning == "coarse":
        output_plot_name += "coarse"

    if is_stack:
        draw_stack(
                [numer] + stack_hists, # numerator histograms
                denom, # denominator histogram
                "plots/mtv/{0}.pdf".format(output_plot_name), # output plot name
                sample_name, # sample type
                git_hash, # version tag
                # of # TGraph output rootfile
                )
    else:

        draw_ratio(
                numer, # numerator histogram
                denom, # denominator histogram
                "plots/mtv/{0}.pdf".format(output_plot_name), # output plot name
                sample_name, # sample type
                git_hash, # version tag
                # of # TGraph output rootfile
                )

    DIR = os.path.realpath(os.path.dirname(__file__))
    os.system("cp -r {}/../misc/summary plots/".format(DIR))

#!/bin/env python

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
    for i in range(0, eff.GetN()):
        if yaxis_max < eff.GetY()[i]:
            yaxis_max = eff.GetY()[i]
    # print yaxis_max
    yaxis_min = 999
    for i in range(0, eff.GetN()):
        if yaxis_min > eff.GetY()[i] and eff.GetY()[i] != 0:
            yaxis_min = eff.GetY()[i]

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
    c1.SaveAs("{}".format(output_name.replace(".pdf", "_effstack.pdf")))
    c1.SaveAs("{}".format(output_name.replace(".pdf", "_effstack.png")))
    eff.SetName(output_name.replace(".png",""))
    if outputfile:
        outputfile.cd()
        basename = os.path.basename(output_name)
        outputname = basename.replace(".pdf","")
        print(outputname)
        eff.SetName(outputname)
        eff.Write()
        eff_num = r.TGraphAsymmErrors(num1)
        eff_den = r.TGraphAsymmErrors(den)
        eff_num.SetName(outputname+"_num")
        eff_den.SetName(outputname+"_den")
        eff_num.Write()
        eff_den.Write()
        outputfile.ls()

    set_label(num1, raw_number=True)
    num1.Draw("hist")
    draw_label()
    c1.SaveAs("{}".format(output_name.replace(".pdf", "_numstack.pdf")))
    c1.SaveAs("{}".format(output_name.replace(".pdf", "_numstack.png")))

    set_label(den, raw_number=True)
    den.Draw("hist")
    draw_label()
    c1.SaveAs("{}".format(output_name.replace(".pdf", "_denstack.pdf")))
    c1.SaveAs("{}".format(output_name.replace(".pdf", "_denstrack.png")))

    return eff
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
    for i in range(0, eff.GetN()):
        if yaxis_max < eff.GetY()[i]:
            yaxis_max = eff.GetY()[i]
    # print yaxis_max
    yaxis_min = 999
    for i in range(0, eff.GetN()):
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
        if "TC_All" not in key.GetName() and "T4s_All" not in key.GetName() and "T3_All" not in key.GetName() and "pLS" not in key.GetName() and "T5" not in key.GetName() and "pT4_All" not in key.GetName() and "pT3_All" not in key.GetName() and "pT5_All" not in key.GetName() and "TCE_All" not in key.GetName():
            continue
        # if "pLS_P" not in key.GetName():
        #     continue
        # if "pix_P" not in key.GetName():
        #     continue
        if "stack" in key.GetName():
            continue
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

    for key in f.GetListOfKeys():
        if "TC_All" in key.GetName() and "stack" in key.GetName():
          print("xxx:",key.GetName())
    for kin in ["pt","eta","phi", "dz", "dxy"]:
      hist0 = f.Get("Root__TC_AllTypes_h_numer_%s"%kin)
      hist1 = f.Get("Root__TC_AllTypes_stackpT5_numer_%s"%kin)
      hist2 = f.Get("Root__TC_AllTypes_stackpT3_numer_%s"%kin)
      hist3 = f.Get("Root__TC_AllTypes_stackT5_numer_%s"%kin)
      hist4 = f.Get("Root__TC_AllTypes_stackpLS_numer_%s"%kin)
      denom = f.Get("Root__TC_AllTypes_h_denom_%s"%kin)
      nice_name = "TC_AllTypes_stack_%s"%kin
      draw_stack([hist0.Clone(),hist1.Clone(),hist2.Clone(),hist3.Clone(),hist4.Clone()],denom.Clone(),"plots/mtv/{}coarse.pdf".format(nice_name), sample_name, version_tag, of)
      if kin == "dz" or kin == "dxy":
          continue
      hist0 = f.Get("Root__TC_AllTypes_h_fakerate_numer_%s"%kin)
      hist1 = f.Get("Root__TC_AllTypes_fakestackpT5_numer_%s"%kin)
      hist2 = f.Get("Root__TC_AllTypes_fakestackpT3_numer_%s"%kin)
      hist3 = f.Get("Root__TC_AllTypes_fakestackT5_numer_%s"%kin)
      hist4 = f.Get("Root__TC_AllTypes_fakestackpLS_numer_%s"%kin)
      denom = f.Get("Root__TC_AllTypes_h_fakerate_denom_%s"%kin)
      nice_name = "TC_AllTypes_fakestack_%s"%kin
      draw_stack([hist0.Clone(),hist1.Clone(),hist2.Clone(),hist3.Clone(),hist4.Clone()],denom.Clone(),"plots/mtv/{}coarse.pdf".format(nice_name), sample_name, version_tag, of)
      hist0 = f.Get("Root__TC_AllTypes_h_duplrate_numer_%s"%kin)
      hist1 = f.Get("Root__TC_AllTypes_dupstackpT5_numer_%s"%kin)
      hist2 = f.Get("Root__TC_AllTypes_dupstackpT3_numer_%s"%kin)
      hist3 = f.Get("Root__TC_AllTypes_dupstackT5_numer_%s"%kin)
      hist4 = f.Get("Root__TC_AllTypes_dupstackpLS_numer_%s"%kin)
      denom = f.Get("Root__TC_AllTypes_h_duplrate_denom_%s"%kin)
      nice_name = "TC_AllTypes_dupstack_%s"%kin
      draw_stack([hist0.Clone(),hist1.Clone(),hist2.Clone(),hist3.Clone(),hist4.Clone()],denom.Clone(),"plots/mtv/{}coarse.pdf".format(nice_name), sample_name, version_tag, of)
      print("HERE 3 !!!!!!!!!!!!")
    of.Write()

    of.Close()


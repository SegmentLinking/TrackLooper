#!/bin/env python

import ROOT as r
r.gROOT.SetBatch(True)
r.gStyle.SetOptStat(0)

def plot(opt):

    ## PT
    histname = opt["histname"]
    YaxisLabel = opt["YaxisLabel"]
    XaxisLabel = opt["XaxisLabel"]
    Selection = opt["Selection"]
    plotname = opt["plotname"]
    RangeMax = opt["RangeMax"]
    SetLogX = opt["SetLogX"]
    ChangeXaxisRange = opt["ChangeXaxisRange"]

    ## Base
    f = r.TFile("DPNote/harvestedFiles/plots_patatrack.root")
    d = f.Get("DQMData").Get("Run 1").Get("Tracking").Get("Run summary").Get("Track").Get("general_trackingParticleGeneralAssociation")
    eff = d.Get(histname)

    c1 = r.TCanvas("", "", 800, 800)
    c1.SetRightMargin(0.02)
    c1.SetTopMargin(0.02)
    c1.SetBottomMargin(0.14)
    c1.SetLeftMargin(0.14)
    c1.SetLogx(SetLogX)
    if "r_" in plotname:
        c1.SetLogy(1)

    r.gPad.SetTickx(1)
    r.gPad.SetTicky(1)

    # c1.SetGrid()
    # r.gStyle.SetPadGridX(True)
    # r.gStyle.SetPadGridY(True)
    # r.gStyle.SetGridStyle(1)
    # color = r.TColor(9482, 0, 0, 0, "", 0.1);
    # r.gStyle.SetGridColor(9482)

    eff.SetTitle("")

    eff.SetMarkerStyle(20)
    #eff.SetMarkerColor(4)
    #eff.SetLineColor(4)
    eff.SetMarkerColor(r.TColor.GetColor("#5790FC"))
    eff.SetLineColor(r.TColor.GetColor("#5790FC"))
    eff.SetLineWidth(2)

    eff.GetXaxis().SetTitleSize(0.045)
    eff.GetYaxis().SetTitleSize(0.050)
    eff.GetXaxis().SetTitleOffset(1.4)
    eff.GetYaxis().SetTitleOffset(1.4)
    eff.GetXaxis().SetLabelSize(0.045)
    eff.GetYaxis().SetLabelSize(0.045)
    eff.GetXaxis().SetTitle(XaxisLabel)
    eff.GetYaxis().SetTitle(YaxisLabel)

    eff.SetMaximum(RangeMax)
    if "eff_eta" in plotname:
        eff.SetMinimum(0.0)
    if "fr_pt" in plotname:
        eff.SetMaximum(1000)
        eff.SetMinimum(0.005)
    if "r_eta" in plotname:
        eff.SetMaximum(1000)
        eff.SetMinimum(0.0001)
    if "dr_pt" in plotname:
        eff.SetMaximum(1000)
        eff.SetMinimum(0.0001)

    eff.Draw("ep")

    ################ Extra

    f_LSTandCKFLegacy3 = r.TFile("DPNote/harvestedFiles/plots_patatrack_LSTSplitpTTCandpLSTCandT5TC_defaultTrkIdBuiltTracks_CKFRecovery_crossCleanpLST5At0p005.root")
    d_LSTandCKFLegacy3 = f_LSTandCKFLegacy3.Get("DQMData").Get("Run 1").Get("Tracking").Get("Run summary").Get("Track").Get("general_trackingParticleGeneralAssociation")
    eff_LSTandCKFLegacy3 = d_LSTandCKFLegacy3.Get(histname)
    eff_LSTandCKFLegacy3.SetMarkerStyle(24)
    #eff_LSTandCKFLegacy3.SetMarkerColor(2)
    #eff_LSTandCKFLegacy3.SetLineColor(2)
    eff_LSTandCKFLegacy3.SetMarkerColor(r.TColor.GetColor("#E42536"))
    eff_LSTandCKFLegacy3.SetLineColor(r.TColor.GetColor("#E42536"))
    eff_LSTandCKFLegacy3.SetLineWidth(2)
    eff_LSTandCKFLegacy3.Draw("epsame")

    f_LSTandCKFLST4 = r.TFile("DPNote/harvestedFiles/plots_patatrack_LSTSplitpTTCandT5TCandpLSTC_defaultTrkIdBuiltTracksDefaultTrkId.root")
    d_LSTandCKFLST4 = f_LSTandCKFLST4.Get("DQMData").Get("Run 1").Get("Tracking").Get("Run summary").Get("Track").Get("general_trackingParticleGeneralAssociation")
    eff_LSTandCKFLST4 = d_LSTandCKFLST4.Get(histname)
    eff_LSTandCKFLST4.SetMarkerStyle(21)
    #eff_LSTandCKFLST4.SetMarkerColor(3)
    #eff_LSTandCKFLST4.SetLineColor(3)
    eff_LSTandCKFLST4.SetMarkerColor(r.TColor.GetColor("#F89C20"))
    eff_LSTandCKFLST4.SetLineColor(r.TColor.GetColor("#F89C20"))
    eff_LSTandCKFLST4.SetLineWidth(2)
    eff_LSTandCKFLST4.Draw("epsame")

    f_LSTandCKFLST43 = r.TFile("DPNote/harvestedFiles/plots_patatrack_LSTNopLSDupCleanTripletpLSSplitpTTCandT5TCandpLSTC_defaultTrkIdBuiltTracksDefaultTrkId.root")
    d_LSTandCKFLST43 = f_LSTandCKFLST43.Get("DQMData").Get("Run 1").Get("Tracking").Get("Run summary").Get("Track").Get("general_trackingParticleGeneralAssociation")
    eff_LSTandCKFLST43 = d_LSTandCKFLST43.Get(histname)
    eff_LSTandCKFLST43.SetMarkerStyle(25)
    #eff_LSTandCKFLST43.SetMarkerColor(6)
    #eff_LSTandCKFLST43.SetLineColor(6)
    eff_LSTandCKFLST43.SetMarkerColor(r.TColor.GetColor("#964A8B"))
    eff_LSTandCKFLST43.SetLineColor(r.TColor.GetColor("#964A8B"))
    eff_LSTandCKFLST43.SetLineWidth(2)
    eff_LSTandCKFLST43.Draw("epsame")

    if ChangeXaxisRange:
        if plotname == "eff_pt" or plotname == "dr_pt" or plotname == "fr_pt":
            #eff.GetXaxis().SetRangeUser(0.1, 500)
            eff.GetXaxis().SetRangeUser(0.9, 500)
            eff_LSTandCKFLegacy3.GetXaxis().SetRangeUser(0.9, 500)
            eff_LSTandCKFLST4.GetXaxis().SetRangeUser(0.9, 500)
            eff_LSTandCKFLST43.GetXaxis().SetRangeUser(0.9, 500)
        if plotname == "eff_eta" or plotname == "dr_eta" or plotname == "fr_eta":
            eff.GetXaxis().SetRangeUser(-4.0, 4.0)
            eff_LSTandCKFLegacy3.GetXaxis().SetRangeUser(-4.0, 4.0)
            eff_LSTandCKFLST4.GetXaxis().SetRangeUser(-4.0, 4.0)
            eff_LSTandCKFLST43.GetXaxis().SetRangeUser(-4.0, 4.0)
        if plotname == "eff_vxy":
            eff.GetXaxis().SetRangeUser(0.2, 200)
            eff_LSTandCKFLegacy3.GetXaxis().SetRangeUser(0.2, 200)
            eff_LSTandCKFLST4.GetXaxis().SetRangeUser(0.2, 200)
            eff_LSTandCKFLST43.GetXaxis().SetRangeUser(0.2, 200)


    box = r.TPad("", "", 0.14, 0.65, 0.979, 0.979)
    box.SetFillColor(0)
    box.Draw()

    t = r.TLatex()
    t.SetTextAlign(11)
    t.SetTextFont(42)
    t.SetTextColor(r.kBlack)
    t.SetTextSize(0.045)
    ts = t.GetTextSize()
    cms_label = "Simulation Preliminary"
    t.DrawLatexNDC(0.17, 0.91, "#scale[1.375]{#font[61]{CMS}} #scale[1.21]{#font[52]{%s}}" % cms_label)
    t.SetTextSize(0.042)
    t.DrawLatexNDC(0.17, 0.91 - 1.4*ts, "#sqrt{s} = 14 TeV PU200 t#bar{t}")
    t.DrawLatexNDC(0.17, 0.91 - 2.7*ts, Selection)

    ################ TLegend

    leg = r.TLegend(0.15, 0.65, 0.98, 0.75)
    leg.SetNColumns(2)
    leg.SetFillStyle(0)
    leg.SetBorderSize(0)
    leg.SetColumnSeparation(-0.03)
    leg.SetMargin(0.15)
    leg.AddEntry(eff, "Base CKF", "lp")
    leg.AddEntry(eff_LSTandCKFLegacy3, "LST w/ CKF on Legacy Triplets", "lp")
    leg.AddEntry(eff_LSTandCKFLST4, "LST w/ CKF on LST Quads", "lp")
    leg.AddEntry(eff_LSTandCKFLST43, "LST w/ CKF on LST Quads+Triplets", "lp")
    leg.Draw()

    r.gPad.RedrawAxis()

    c1.SaveAs("~/public_html/SDL/LSTinHLT_alpaka/DPNote/plots/{}.pdf".format(plotname))
    c1.SaveAs("~/public_html/SDL/LSTinHLT_alpaka/DPNote/plots/{}.png".format(plotname))

opt = {}

## Eta
opt["histname"] = "effic"
opt["YaxisLabel"] = "Tracking efficiency"
opt["XaxisLabel"] = "Simulated track #eta"
opt["Selection"] = "p_{T} > 0.9 GeV, |z_{vertex}| < 30 cm, r_{vertex} < 2.5 cm"
opt["plotname"] = "eff_eta"
opt["RangeMax"] = 1.7
opt["SetLogX"] = False
opt["ChangeXaxisRange"] = True
plot(opt)

## PT
opt["histname"] = "efficPt"
opt["YaxisLabel"] = "Tracking efficiency"
opt["XaxisLabel"] = "Simulated track p_{T} [GeV]"
opt["Selection"] = "|#eta| < 4.5, |z_{vertex}| < 30 cm, r_{vertex} < 2.5 cm"
opt["plotname"] = "eff_pt"
opt["RangeMax"] = 1.7
opt["SetLogX"] = True
opt["ChangeXaxisRange"] = True
plot(opt)

## Dxy
opt["histname"] = "effic_vs_vertpos"
opt["YaxisLabel"] = "Tracking efficiency"
opt["XaxisLabel"] = "Simulated track r_{vertex} [cm]"
opt["Selection"] = "p_{T} > 0.9 GeV, |#eta| < 4.5, |z_{vertex}| < 30 cm"
opt["plotname"] = "eff_vxy"
opt["RangeMax"] = 1.7
opt["SetLogX"] = True
opt["ChangeXaxisRange"] = True
plot(opt)

## Eta
opt["histname"] = "fakerate"
opt["YaxisLabel"] = "Fake rate"
opt["XaxisLabel"] = "Track #eta"
opt["Selection"] = "p_{T} > 0.9 GeV"
opt["plotname"] = "fr_eta"
opt["RangeMax"] = 1.7
opt["SetLogX"] = False
opt["ChangeXaxisRange"] = True
plot(opt)

## PT
opt["histname"] = "fakeratePt"
opt["YaxisLabel"] = "Fake rate"
opt["XaxisLabel"] = "Track p_{T} [GeV]"
opt["Selection"] = "|#eta| < 4.5"
opt["plotname"] = "fr_pt"
opt["RangeMax"] = 1.7
opt["SetLogX"] = True
opt["ChangeXaxisRange"] = True
plot(opt)

## Eta
opt["histname"] = "duplicatesRate"
opt["YaxisLabel"] = "Duplicate rate"
opt["XaxisLabel"] = "Track #eta"
opt["Selection"] = "p_{T} > 0.9 GeV"
opt["plotname"] = "dr_eta"
opt["RangeMax"] = 0.249
opt["SetLogX"] = False
opt["ChangeXaxisRange"] = True
plot(opt)

## PT
opt["histname"] = "duplicatesRate_Pt"
opt["YaxisLabel"] = "Duplicate rate"
opt["XaxisLabel"] = "Track p_{T} [GeV]"
opt["Selection"] = "|#eta| < 4.5"
opt["plotname"] = "dr_pt"
opt["RangeMax"] = 0.249
opt["SetLogX"] = True
opt["ChangeXaxisRange"] = True
plot(opt)


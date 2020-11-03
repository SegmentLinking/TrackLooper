#!/bin/env python

import ROOT as r
from array import array

r.gROOT.SetBatch(True)

# f = r.TFile("debug.root")
# t = f.Get("tree")

t = r.TChain("tree")
t.Add("debug_ntuple_output/debug_202003242242_*.root")

c1 = r.TCanvas()

# t.Draw("Max$(mdendcap1_minicut - abs(mdendcap1_dphichange))", "is_trk_beeeee == 1 && abs(dxy) < 3.5 && (Sum$(mdendcap1_pass)) > 0 && pt > 0.85 && pt < 0.95")

# c1.SaveAs("beeeee_md_dphichange_endcap1.pdf")

def pt_eff():
    binbounds = [0.5, 0.52, 0.54, 0.56, 0.58, 0.6, 0.62, 0.64, 0.66, 0.68, 0.7, 0.72, 0.74, 0.76, 0.78, 0.8, 0.82, 0.84, 0.86, 0.88, 0.9, 0.92, 0.94, 0.96, 0.98, 1.0, 1.02, 1.04, 1.06, 1.08, 1.1, 1.12, 1.14, 1.16, 1.18, 1.2, 1.22, 1.24, 1.26, 1.28, 1.3, 1.32, 1.34, 1.36, 1.38, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
    root_binbounds = array('d', [0.5, 0.52, 0.54, 0.56, 0.58, 0.6, 0.62, 0.64, 0.66, 0.68, 0.7, 0.72, 0.74, 0.76, 0.78, 0.8, 0.82, 0.84, 0.86, 0.88, 0.9, 0.92, 0.94, 0.96, 0.98, 1.0, 1.02, 1.04, 1.06, 1.08, 1.1, 1.12, 1.14, 1.16, 1.18, 1.2, 1.22, 1.24, 1.26, 1.28, 1.3, 1.32, 1.34, 1.36, 1.38, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0])
    hdenom = r.TH1F("hdenom", "", len(binbounds) - 1, root_binbounds)
    hnumer = r.TH1F("hnumer", "", len(binbounds) - 1, root_binbounds)
    t.Draw("pt>>hdenom","(is_trk_bbbbee == 1) && (is_trk_bbbbbe == 0) && (is_trk_bbbbbb == 0) && abs(dxy) < 3.5", "goffe")
    hdenom.Draw()
    # t.Draw("pt>>hnumer","(is_trk_bbbbee == 1) && (is_trk_bbbbbe == 0) && (is_trk_bbbbbb == 0) && abs(dxy) < 3.5 && (Sum$(tl3_pass * (tl3_innerSg_innerMd_lower_hit_subdet==5 && tl3_innerSg_outerMd_lower_hit_subdet==5 && tl3_outerSg_innerMd_lower_hit_subdet==4 && tl3_outerSg_outerMd_lower_hit_subdet==4))) > 0", "goffe")
    t.Draw("pt>>hnumer","(pt>1.4) && (abs(dxy) < 3.5) && (is_trk_bbbbee == 1) && (is_trk_bbbbbe == 0) && (is_trk_bbbbbb == 0) && abs(dxy) < 3.5 && (Sum$(tl3_pass * (tl3_innerSg_innerMd_lower_hit_subdet==5 && tl3_innerSg_outerMd_lower_hit_subdet==5 && tl3_outerSg_innerMd_lower_hit_subdet==4 && tl3_outerSg_outerMd_lower_hit_subdet==4))) == 0", "goffe")
    hnumer.Draw()
    c1.SaveAs("hnumer.pdf")
    hnumer.Print("all")
    hdenom.Print("all")
    eff = hnumer.Clone()
    eff.Sumw2()
    eff.Divide(hnumer, hdenom, 1, 1, 'B')
    eff.Draw("ep")
    c1.SetLogx()
    c1.SaveAs("eff.pdf")

def eta_eff():
    hdenom = r.TH1F("hdenom", "", 50, 0.9, 1.3)
    hnumer = r.TH1F("hnumer", "", 50, 0.9, 1.3)
    t.Draw("abs(eta)>>hdenom","is_trk_bbbbbe == 1 && abs(dxy) < 3.5 && pt > 1.4", "goffe")
    hdenom.Draw()
    t.Draw("abs(eta)>>hnumer","is_trk_bbbbbe == 1 && abs(dxy) < 3.5 && pt > 1.4 && (Sum$(tl3_pass)) > 0", "goffe")
    hnumer.Draw()
    c1.SaveAs("hnumer.pdf")
    hnumer.Print("all")
    hdenom.Print("all")
    eff = hnumer.Clone()
    eff.Sumw2()
    eff.Divide(hnumer, hdenom, 1, 1, 'B')
    eff.Draw("ep")
    c1.SaveAs("eff.pdf")

def bad_event():
    c1 = r.TCanvas("c1","",1800,300)
    t.Scan("is_trk_bbbbbb:is_trk_bbbbbe:is_trk_bbbbee:pt:eta","is_trk_bbbbbe==1&&abs(dxy)<3.5&&(Sum$(tl3_pass)==0)","")
    event = raw_input("event : ")
    t.SetMarkerStyle(0)
    t.SetMarkerColor(1)
    t.Draw("sqrt(simhit_x**2+simhit_y**2):simhit_z","Entry$<5000")
    t.SetMarkerStyle(20)
    t.SetMarkerSize(0.5)
    t.SetMarkerColor(2)
    t.Draw("sqrt(simhit_x**2+simhit_y**2):simhit_z","(Entry$=={})".format(event), "same")
    t.SetMarkerStyle(20)
    t.SetMarkerSize(0.5)
    t.SetMarkerColor(4)
    t.Draw("sqrt(ph2_x**2+ph2_y**2):ph2_z","(Entry$=={})".format(event), "same")
    c1.SaveAs("plots_bad_track_categorization/plot_rz.pdf")

    c2 = r.TCanvas("c2","",600,600)
    t.SetMarkerStyle(0)
    t.SetMarkerColor(1)
    t.Draw("simhit_x:simhit_y","Entry$<5000")
    t.SetMarkerStyle(20)
    t.SetMarkerSize(0.5)
    t.SetMarkerColor(2)
    t.Draw("simhit_x:simhit_y","(Entry$=={})".format(event), "same")
    t.SetMarkerStyle(20)
    t.SetMarkerSize(0.5)
    t.SetMarkerColor(4)
    t.Draw("simhit_x:simhit_y","(Entry$=={})".format(event), "same")
    c2.SaveAs("plots_bad_track_categorization/plot_xy.pdf")

def save_bulk():
    c1 = r.TCanvas("bulk","bulk",1800,300)
    t.Scan("is_trk_bbbbbb:is_trk_bbbbbe:is_trk_bbbbee","is_trk_bbbbbe==1&&abs(dxy)<3.5&&pt>1.4&&(Sum$(tl3_pass)==0)","")
    event = raw_input("event : ")
    t.Draw("sqrt(simhit_x**2+simhit_y**2):simhit_z>>h","")
    h = r.gDirectory.Get("h")
    f = r.TFile("bulk.root","recreate")
    c1.Write()

def draw_on_bulk():
    f_bulk = r.TFile("bulk.root")
    c1_bulk = f_bulk.Get("bulk")
    t.Scan("is_trk_bbbbbb:is_trk_bbbbbe:is_trk_bbbbee","is_trk_bbbbbe==1&&abs(dxy)<3.5&&pt>1.4&&(Sum$(tl3_pass)==0)","")
    event = raw_input("event : ")
    t.SetMarkerStyle(20)
    t.SetMarkerSize(0.5)
    t.SetMarkerColor(2)
    t.Draw("sqrt(simhit_x**2+simhit_y**2):simhit_z","is_trk_bbbbbe==1&&pt>1&&(Sum$(tl3_pass)==0)&&(Entry$=={})".format(event), "same")
    t.SetMarkerStyle(20)
    t.SetMarkerSize(0.5)
    t.SetMarkerColor(4)
    t.Draw("sqrt(ph2_x**2+ph2_y**2):ph2_z","is_trk_bbbbbe==1&&pt>1&&(Sum$(tl3_pass)==0)&&(Entry$=={})".format(event), "same")
    c1_bulk.SaveAs("plots_bad_track_categorization/plot.png")

def draw_debug():
    c1 = r.TCanvas()
    # t.Draw("TMath::Log2(tl3_passbits+1)>>(8,0,8)", "(is_trk_bbbbee==1)&&(is_trk_bbbbbe==0)&&(is_trk_bbbbbb==0)&&abs(dxy)<3.5&&pt>1.4&&(Sum$(tl3_pass)==0)");
    # t.Draw("(TMath::Log2(tl3_passbits+1)*(tl3_pass * (tl3_innerSg_innerMd_lower_hit_subdet==5 && tl3_innerSg_outerMd_lower_hit_subdet==5 && tl3_outerSg_innerMd_lower_hit_subdet==4 && tl3_outerSg_outerMd_lower_hit_subdet==4)) + (-999)*(!(tl3_pass * (tl3_innerSg_innerMd_lower_hit_subdet==5 && tl3_innerSg_outerMd_lower_hit_subdet==5 && tl3_outerSg_innerMd_lower_hit_subdet==4 && tl3_outerSg_outerMd_lower_hit_subdet==4))))>>(10,-2,8)", "(pt>1.4) && (abs(dxy) < 3.5) && (is_trk_bbbbee == 1) && (is_trk_bbbbbe == 0) && (is_trk_bbbbbb == 0) && abs(dxy) < 3.5 && (Sum$(tl3_pass * (tl3_innerSg_innerMd_lower_hit_subdet==5 && tl3_innerSg_outerMd_lower_hit_subdet==5 && tl3_outerSg_innerMd_lower_hit_subdet==4 && tl3_outerSg_outerMd_lower_hit_subdet==4))) == 0");
    t.Draw("TMath::Log2(tl3_passbits+1) * (tl3_innerSg_innerMd_lower_hit_subdet==5 && tl3_innerSg_outerMd_lower_hit_subdet==5 && tl3_outerSg_innerMd_lower_hit_subdet==4 && tl3_outerSg_outerMd_lower_hit_subdet==4)>>(10,-2,8)", "(pt>1.4) && (abs(dxy) < 3.5) && (is_trk_bbbbee == 1) && (is_trk_bbbbee == 1) && (is_trk_bbbbbe == 0) && (is_trk_bbbbbb == 0) && abs(dxy) < 3.5 && (Sum$(tl3_pass * (tl3_innerSg_innerMd_lower_hit_subdet==5 && tl3_innerSg_outerMd_lower_hit_subdet==5 && tl3_outerSg_innerMd_lower_hit_subdet==4 && tl3_outerSg_outerMd_lower_hit_subdet==4))) == 0");
    # t.Draw("(tl3_dBeta*tl3_dBeta)-tl3_dBetaCut2", "is_trk_bbbbbe==1&&abs(dxy)<3.5&&pt>1.4&&(Sum$(tl3_pass)==0)&&(tl3_passbits>0)");
    # t.Draw("tl3_dBetaCut2-(tl3_dBeta*tl3_dBeta)", "is_trk_bbbbbe==1&&abs(dxy)<3.5&&pt>1.4&&(tl3_passbits>62)");
    # t.Draw("tl3_dBetaCut2-(tl3_dBeta*tl3_dBeta)", "is_trk_bbbbbe==1&&abs(dxy)<3.5&&pt>1.4&&(tl3_passbits>62)");
    # t.Draw("tl3_outerSg_outerMd_lower_hit_subdet", "is_trk_bbbbbe==1&&abs(dxy)<3.5&&pt>1.4&&(tl3_passbits>62)");
    # t.Draw("tl3_betacormode", "is_trk_bbbbbe==1&&abs(dxy)<3.5&&pt>1.4&&(tl3_passbits>62)");
    c1.SaveAs("debug.pdf")

def draw_deltaBeta():
    c1 = r.TCanvas()
    t.Draw("tl1_dBeta", "tl1_passbits>=63")
    c1.SaveAs("debug.pdf")

if __name__ == "__main__":

    # eta_eff()
    # bad_event()
    # save_bulk()
    # draw_on_bulk()
    # pt_eff()
    draw_debug()
    # draw_deltaBeta()

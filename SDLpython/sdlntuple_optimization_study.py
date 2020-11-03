#!/bin/env python

import ROOT as r
from array import array

def draw_eff(num, den, output_name):

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


    # f = r.TFile("/home/users/phchang/public_html/analysis/sdl/TrackLooper_/results/write_sdl_ntuple/pt0p5_2p0_20200913_Test_v2//fulleff_pt0p5_2p0.root")
    # f = r.TFile("/home/users/phchang/public_html/analysis/sdl/TrackLooper_/results/write_sdl_ntuple/pu200_w_truth_pdgid211_20200919_Test_v1_w_gap//fulleff_pu200_w_truth_pdgid211.root")
    # f = r.TFile("/home/users/phchang/public_html/analysis/sdl/TrackLooper_/results/write_sdl_ntuple/pu200_w_truth_pdgid211_20200919_Test_v2_w_gap//fulleff_pu200_w_truth_pdgid211.root")
#    f = r.TFile("/home/users/phchang/public_html/analysis/sdl/TrackLooper_/results/write_sdl_ntuple/pion_20200925_Test_v2//fulleff_pion.root")
    f= r.TFile("debug.root")
    t = f.Get("tree")

    eta_denom = r.TH1F("eta_denom", "eta_denom", 30, -2.5, 2.5)
    eta_numer = r.TH1F("eta_numer", "eta_numer", 30, -2.5, 2.5)
    # pt_boundaries = [0.5, 0.52, 0.54, 0.56, 0.58, 0.6, 0.62, 0.64, 0.66, 0.68, 0.7, 0.72, 0.74, 0.76, 0.78, 0.8, 0.82, 0.84, 0.86, 0.88, 0.9, 0.92, 0.94, 0.96, 0.98, 1.0, 1.02, 1.04, 1.06, 1.08, 1.1, 1.12, 1.14, 1.16, 1.18, 1.2, 1.22, 1.24, 1.26, 1.28, 1.3, 1.32, 1.34, 1.36, 1.38, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
    pt_boundaries = [0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0, 3.0, 5.0, 10, 15., 25, 50]
    pt_denom = r.TH1F("pt_denom", "pt_denom", len(pt_boundaries)-1, array('d', pt_boundaries))
    pt_numer = r.TH1F("pt_numer", "pt_numer", len(pt_boundaries)-1, array('d', pt_boundaries))

    n_layers = 6
    n_eta_sections = 10
    eta_section_length = 0.2

    variables = ["dphiChange", "miniCut"]
    # ana.tx->createBranch<vector<float>>("md_type");
    # ana.tx->createBranch<vector<float>>("md_dz");
    # ana.tx->createBranch<vector<float>>("md_dzCut");
    # ana.tx->createBranch<vector<float>>("md_drt");
    # ana.tx->createBranch<vector<float>>("md_drtCut");
    # ana.tx->createBranch<vector<float>>("md_miniCut");
    # ana.tx->createBranch<vector<float>>("md_dphi");
    # ana.tx->createBranch<vector<float>>("md_dphiChange");

    mini_doublet_variables = {}

    # initialize the ntuples
    for i_var in variables:
        mini_doublet_variables[i_var] = {}
        for i_layer in range(n_layers):
            mini_doublet_variables[i_var][i_layer] = {}
            for i_eta_section in range(n_eta_sections):
                mini_doublet_variables[i_var][i_layer][i_eta_section] = []

    for ievent, event in enumerate(t):

        for isim, _ in enumerate(event.sim_qpIdx):

            pt = event.sim_pt[isim]
            eta = event.sim_eta[isim]
            dz = event.sim_pca_dz[isim]
            dxy = event.sim_pca_dxy[isim]
            pdgid = event.sim_pdgId[isim]
            bunch = event.sim_bunchCrossing[isim]
            n_mtv_tc = len(list(event.sim_qpIdx_isMTVmatch[isim]))

            for qp_idx in event.sim_qpIdx_isMTVmatch[isim]:

                all_hit_idxs = list(event.qp_hitIdx[qp_idx])
                lower_hit_idxs = all_hit_idxs[::2]
                upper_hit_idxs = all_hit_idxs[1::2]

            if abs(pdgid) != 13:
                continue

            if bunch != 0:
                continue

            if abs(dz) > 30 or abs(dxy) > 2.5:
                continue;
            if pt > 1.5:
                eta_denom.Fill(eta)
            if abs(eta) < 2.4:
                pt_denom.Fill(pt)
            if n_mtv_tc > 0:
                if pt > 1.5:
                    eta_numer.Fill(eta)
                if abs(eta) < 2.4:
                    pt_numer.Fill(pt)

        # break

    eta_denom.Print("all")
    eta_numer.Print("all")
    pt_denom.Print("all")
    pt_numer.Print("all")
                
    draw_eff(pt_numer, pt_denom, "mtv_optimization_study_pt.pdf")
    draw_eff(eta_numer, eta_denom, "mtv_optimization_study_eta.pdf")


#!/bin/env python

import ROOT as r

r.gROOT.SetBatch(True)


def double_ratio(ilayer, numfile, denfile):

    eff_absolute_file = r.TFile("{}".format(denfile))
    eff_nominal_file = r.TFile("{}".format(numfile))

    den = eff_absolute_file.Get("sg_eff_barrelbarrel_pt_by_layer{}.pdf".format(ilayer))
    num = eff_nominal_file.Get("sg_eff_barrelbarrel_pt_by_layer{}.pdf".format(ilayer))

    c1 = r.TCanvas()

    num.Draw("ap")
    den.Draw("psame")

    num.SetLineColor(2)
    num.SetMarkerColor(2)

    c1.SaveAs("plots_double_ratio/numden_by_layer_{}.pdf".format(ilayer))
    c1.SaveAs("plots_double_ratio/numden_by_layer_{}.png".format(ilayer))

    denhist_file = denfile.replace("/eff_","/")
    numhist_file = numfile.replace("/eff_","/")
    absolute_file = r.TFile("{}".format(denhist_file))
    nominal_file = r.TFile("{}".format(numhist_file))

    hist_den = absolute_file.Get("Root__sg_barrelbarrel_matched_track_pt_by_layer{}".format(ilayer))
    hist_num = nominal_file.Get("Root__sg_barrelbarrel_matched_track_pt_by_layer{}".format(ilayer))

    teff = r.TEfficiency(hist_num, hist_den)
    eff = teff.CreateGraph()

    eff.GetYaxis().SetRangeUser(0.9, 1.1)

    eff.Draw("epa")
    eff.SetMarkerStyle(19)
    eff.SetMarkerSize(1.2)
    eff.SetLineWidth(2)
    eff.GetXaxis().SetTitle("p_{T} [GeV]")
    eff.GetXaxis().SetTitleSize(0.05)
    eff.GetXaxis().SetLabelSize(0.05)
    eff.GetYaxis().SetLabelSize(0.05)

    c1.SaveAs("plots_double_ratio/ratio_by_layer_{}.pdf".format(ilayer))
    c1.SaveAs("plots_double_ratio/ratio_by_layer_{}.png".format(ilayer))


if __name__ == "__main__":

    double_ratio(0, "results/20191028_segment_mapping_check/eff_mu_pt0p5_1p5_eff_seg__finalmap.root", "results/20191028_segment_mapping_check/eff_mu_pt0p5_1p5_eff_seg__nomap.root")
    double_ratio(1, "results/20191028_segment_mapping_check/eff_mu_pt0p5_1p5_eff_seg__finalmap.root", "results/20191028_segment_mapping_check/eff_mu_pt0p5_1p5_eff_seg__nomap.root")
    double_ratio(2, "results/20191028_segment_mapping_check/eff_mu_pt0p5_1p5_eff_seg__finalmap.root", "results/20191028_segment_mapping_check/eff_mu_pt0p5_1p5_eff_seg__nomap.root")
    double_ratio(3, "results/20191028_segment_mapping_check/eff_mu_pt0p5_1p5_eff_seg__finalmap.root", "results/20191028_segment_mapping_check/eff_mu_pt0p5_1p5_eff_seg__nomap.root")
    double_ratio(4, "results/20191028_segment_mapping_check/eff_mu_pt0p5_1p5_eff_seg__finalmap.root", "results/20191028_segment_mapping_check/eff_mu_pt0p5_1p5_eff_seg__nomap.root")

    # double_ratio(0, "results/20191029_segment_mapping_check/eff_mu_pt0p5_1p5_eff_seg__finalmap.root", "results/20191028_segment_mapping_check/eff_mu_pt0p5_1p5_eff_seg__nomap.root")
    # double_ratio(1, "results/20191029_segment_mapping_check/eff_mu_pt0p5_1p5_eff_seg__finalmap.root", "results/20191028_segment_mapping_check/eff_mu_pt0p5_1p5_eff_seg__nomap.root")
    # double_ratio(2, "results/20191029_segment_mapping_check/eff_mu_pt0p5_1p5_eff_seg__finalmap.root", "results/20191028_segment_mapping_check/eff_mu_pt0p5_1p5_eff_seg__nomap.root")
    # double_ratio(3, "results/20191029_segment_mapping_check/eff_mu_pt0p5_1p5_eff_seg__finalmap.root", "results/20191028_segment_mapping_check/eff_mu_pt0p5_1p5_eff_seg__nomap.root")
    # double_ratio(4, "results/20191029_segment_mapping_check/eff_mu_pt0p5_1p5_eff_seg__finalmap.root", "results/20191028_segment_mapping_check/eff_mu_pt0p5_1p5_eff_seg__nomap.root")

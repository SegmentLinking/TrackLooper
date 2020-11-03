#!/bin/env python

import plottery_wrapper as p
import ROOT as r

# f = r.TFile("results/pt0p5_2p0_20200317_1550/fulleff_pt0p5_2p0.root") # Very first denom (March 6th? ish)
f = r.TFile("results/pt0p5_2p0_20200318_0843/fulleff_pt0p5_2p0.root") # Older Denom (March 18 morning)
# f = r.TFile("results/pt0p5_2p0_20200319_1008/fulleff_pt0p5_2p0.root") # Newer Denom (March 19 morning)
# f = r.TFile("results/pt0p5_2p0_20200319_1023/fulleff_pt0p5_2p0.root") # Newer Denom with no pdgId matching between simhits and simtrack
# f = r.TFile("results/pt0p5_2p0_20200319_1143/fulleff_pt0p5_2p0.root") # Newer Denom with no pdgId matching between simhits and simtrack and some priority scheme
# f = r.TFile("results/pt0p5_2p0_20200319_1153/fulleff_pt0p5_2p0.root") # Newer Denom with yes pdgId matching between simhits and simtrack and some priority scheme

bbbbbb_denom = f.Get("Root__tc_bbbbbb_all_track_pt_by_layer0")
bbbbbe_denom = f.Get("Root__tc_bbbbbe_all_track_pt_by_layer0")
bbbbee_denom = f.Get("Root__tc_bbbbee_all_track_pt_by_layer0")
bbbeee_denom = f.Get("Root__tc_bbbeee_all_track_pt_by_layer0")
bbeeee_denom = f.Get("Root__tc_bbeeee_all_track_pt_by_layer0")
beeeee_denom = f.Get("Root__tc_beeeee_all_track_pt_by_layer0")

p.plot_hist(bgs=[bbbbbb_denom, bbbbbe_denom, bbbbee_denom, bbbeee_denom, bbeeee_denom, beeeee_denom],
        options={
            "output_name":"plots_denom/denom_pt.pdf",
            "bkg_sort_method":"unsorted",
            }
        )

bbbbbb_denom = f.Get("Root__tc_bbbbbb_all_track_eta_by_layer0")
bbbbbe_denom = f.Get("Root__tc_bbbbbe_all_track_eta_by_layer0")
bbbbee_denom = f.Get("Root__tc_bbbbee_all_track_eta_by_layer0")
bbbeee_denom = f.Get("Root__tc_bbbeee_all_track_eta_by_layer0")
bbeeee_denom = f.Get("Root__tc_bbeeee_all_track_eta_by_layer0")
beeeee_denom = f.Get("Root__tc_beeeee_all_track_eta_by_layer0")

p.plot_hist(bgs=[bbbbbb_denom, bbbbbe_denom, bbbbee_denom, bbbeee_denom, bbeeee_denom, beeeee_denom],
        options={
            "output_name":"plots_denom/denom_eta.pdf",
            "bkg_sort_method":"unsorted",
            }
        )

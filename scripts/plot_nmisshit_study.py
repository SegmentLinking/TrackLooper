#!/bin/env python

import plottery_wrapper as p
import ROOT as r
import sys

f = r.TFile("results/pt0p5_2p0_20200209/fulleff_pt0p5_2p0_mtv.root")


h_plot = f.Get("Root__pt_0p95_1p05_hit_miss_study")

p.plot_hist(
        bgs=[h_plot],
        # data=h_all,
        legend_labels=[],
        options={"output_name":"plots_missing_hits/summary.pdf", "print_yield":True},
        )

prop_phi_2Slayer0 = f.Get("Root__pt_0p95_1p05_nmiss2_prop_phi_2Slayer0").Clone()
p.plot_hist(
        bgs=[prop_phi_2Slayer0],
        legend_labels=["prop #phi layer4"],
        options={ "output_name":"plots_missing_hits/hit_nmiss2_phi_prop_pt_0p95_1p05_2Slayer0.pdf", "ratio_range":[0., 0.4], "print_yield":True}
        )

prop_phi_2Slayer1 = f.Get("Root__pt_0p95_1p05_nmiss2_prop_phi_2Slayer1").Clone()
p.plot_hist(
        bgs=[prop_phi_2Slayer1],
        legend_labels=["prop #phi layer5"],
        options={ "output_name":"plots_missing_hits/hit_nmiss2_phi_prop_pt_0p95_1p05_2Slayer1.pdf", "ratio_range":[0., 0.4], "print_yield":True}
        )

prop_phi_2Slayer2 = f.Get("Root__pt_0p95_1p05_nmiss2_prop_phi_2Slayer2").Clone()
p.plot_hist(
        bgs=[prop_phi_2Slayer2],
        legend_labels=["prop #phi layer5"],
        options={ "output_name":"plots_missing_hits/hit_nmiss2_phi_prop_pt_0p95_1p05_2Slayer2.pdf", "ratio_range":[0., 0.4], "print_yield":True}
        )


##################
# Below are standard ones from previous script

prop_phi_2Slayer2 = f.Get("Root__prop_phi_2Slayer2").Clone()
p.plot_hist(
        bgs=[prop_phi_2Slayer2],
        legend_labels=["prop #phi layer6"],
        options={ "output_name":"plots_missing_hits/hit_nmiss2_phi_prop.pdf", "ratio_range":[0., 0.4], "print_yield":True}
        )

prop_phi_2Slayer2 = f.Get("Root__prop_phi_2Slayer2_zoom_m02_p02").Clone()
p.plot_hist(
        bgs=[prop_phi_2Slayer2],
        legend_labels=["prop #phi layer6"],
        options={ "output_name":"plots_missing_hits/hit_nmiss2_phi_prop_m02_p02.pdf", "ratio_range":[0., 0.4], "remove_overflow":True, "remove_underflow":True , "print_yield":True}
        )

prop_phi_2Slayer1 = f.Get("Root__prop_phi_2Slayer1").Clone()
p.plot_hist(
        bgs=[prop_phi_2Slayer1],
        legend_labels=["prop #phi layer5"],
        options={ "output_name":"plots_missing_hits/hit_nmiss2_phi_prop_2Slayer1.pdf", "ratio_range":[0., 0.4], "print_yield":True}
        )

prop_phi_2Slayer1 = f.Get("Root__prop_phi_2Slayer1_zoom_m02_p02").Clone()
p.plot_hist(
        bgs=[prop_phi_2Slayer1],
        legend_labels=["prop #phi layer5"],
        options={ "output_name":"plots_missing_hits/hit_nmiss2_phi_prop_m02_p02_2Slayer1.pdf", "ratio_range":[0., 0.4], "remove_overflow":True, "remove_underflow":True ,"print_yield":True}
        )

prop_phi_2Slayer0 = f.Get("Root__prop_phi_2Slayer0").Clone()
p.plot_hist(
        bgs=[prop_phi_2Slayer0],
        legend_labels=["prop #phi layer4"],
        options={ "output_name":"plots_missing_hits/hit_nmiss2_phi_prop_2Slayer0.pdf", "ratio_range":[0., 0.4], "print_yield":True}
        )

prop_phi_2Slayer0 = f.Get("Root__prop_phi_2Slayer0_zoom_m02_p02").Clone()
p.plot_hist(
        bgs=[prop_phi_2Slayer0],
        legend_labels=["prop #phi layer4"],
        options={ "output_name":"plots_missing_hits/hit_nmiss2_phi_prop_m02_p02_2Slayer0.pdf", "ratio_range":[0., 0.4], "remove_overflow":True, "remove_underflow":True ,"print_yield":True}
        )

prop_phi_2Slayer2 = f.Get("Root__prop_phi_2Slayer2").Clone()
p.plot_hist(
        bgs=[prop_phi_2Slayer2],
        legend_labels=["prop #phi layer6"],
        options={ "output_name":"plots_missing_hits/hit_nmiss2_phi_prop_nbin1.pdf", "ratio_range":[0., 0.4], "print_yield":True, "nbins":1}
        )

prop_phi_2Slayer1 = f.Get("Root__prop_phi_2Slayer1").Clone()
p.plot_hist(
        bgs=[prop_phi_2Slayer1],
        legend_labels=["prop #phi layer5"],
        options={ "output_name":"plots_missing_hits/hit_nmiss2_phi_prop_2Slayer1_nbin1.pdf", "ratio_range":[0., 0.4], "print_yield":True, "nbins":1}
        )

prop_phi_2Slayer0 = f.Get("Root__prop_phi_2Slayer0").Clone()
p.plot_hist(
        bgs=[prop_phi_2Slayer0],
        legend_labels=["prop #phi layer4"],
        options={ "output_name":"plots_missing_hits/hit_nmiss2_phi_prop_2Slayer0_nbin1.pdf", "ratio_range":[0., 0.4], "print_yield":True, "nbins":1}
        )

h_nmiss_0 = f.Get("Root__pt_w_simhit_miss0").Clone()
h_nmiss_1 = f.Get("Root__pt_w_simhit_miss1").Clone()
h_nmiss_2 = f.Get("Root__pt_w_simhit_miss2").Clone()
h_nmiss_3 = f.Get("Root__pt_w_simhit_miss3").Clone()
h_nmiss_4 = f.Get("Root__pt_w_simhit_miss4").Clone()
h_nmiss_5 = f.Get("Root__pt_w_simhit_miss5").Clone()
h_nmiss_6 = f.Get("Root__pt_w_simhit_miss6").Clone()
h_nmiss_7 = f.Get("Root__pt_w_simhit_miss7").Clone()
h_nmiss_8 = f.Get("Root__pt_w_simhit_miss8").Clone()
h_nmiss_9 = f.Get("Root__pt_w_simhit_miss9").Clone()
h_nmiss_10 = f.Get("Root__pt_w_simhit_miss10").Clone()
h_nmiss_11 = f.Get("Root__pt_w_simhit_miss11").Clone()
h_nmiss_12 = f.Get("Root__pt_w_simhit_miss12").Clone()
# h_all     = f.Get("Root__pt_all_w_last_layer").Clone()
h_all     = f.Get("Root__pt_all").Clone()

# h_nmiss_0.SetBinContent(10, 0)
# h_nmiss_1.SetBinContent(10, 0)
# h_nmiss_2.SetBinContent(10, 0)
# h_nmiss_3.SetBinContent(10, 0)
# h_nmiss_4.SetBinContent(10, 0)
# h_nmiss_5.SetBinContent(10, 0)
# h_nmiss_6.SetBinContent(10, 0)
# h_nmiss_7.SetBinContent(10, 0)
# h_nmiss_8.SetBinContent(10, 0)
# h_nmiss_9.SetBinContent(10, 0)
# h_nmiss_10.SetBinContent(10, 0)
# h_nmiss_11.SetBinContent(10, 0)
# h_nmiss_12.SetBinContent(10, 0)
# # h_all     = f.Get("Root__pt_all_w_last_layer").Clone()
# h_all.SetBinContent(10, 0)

p.plot_hist(
        bgs=[h_nmiss_0, h_nmiss_1, h_nmiss_2, h_nmiss_3, h_nmiss_4, h_nmiss_5, h_nmiss_6, h_nmiss_7, h_nmiss_8, h_nmiss_9, h_nmiss_10, h_nmiss_11, h_nmiss_12],
        data=h_all,
        legend_labels=["Nmiss0", "Nmiss1", "Nmiss2", "Nmiss3", "Nmiss4", "Nmiss5", "Nmiss6", "Nmiss7", "Nmiss8", "Nmiss9", "Nmiss10", "Nmiss11", "Nmiss12"],
        options={"output_name":"plots_conditional/hit_efficiency.pdf", "remove_overflow":True, "print_yield":True}
        )


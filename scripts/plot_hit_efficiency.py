#!/bin/env python

import plottery_wrapper as p
import ROOT as r
import sys

# f = r.TFile("results/pt0p5_2p0_20200120/fulleff_pt0p5_2p0_mtv.root")
# f = r.TFile("results/e200_20200120/fulleff_e200_mtv.root")
# f = r.TFile("results/e200_20200121/fulleff_e200_mtv.root")
# f = r.TFile("results/pt0p5_2p0_20200121/fulleff_pt0p5_2p0_mtv.root")
# f = r.TFile("results/pt0p5_2p0_20200127/fulleff_pt0p5_2p0_mtv.root")
# f = r.TFile("results/pt0p5_2p0_conditional/fulleff_pt0p5_2p0_mtv.root")
# f = r.TFile("results/pt0p5_2p0_conditional_pt1p0/fulleff_pt0p5_2p0_mtv.root")
# f = r.TFile("results/pt0p5_2p0_20200131_v1/fulleff_pt0p5_2p0_mtv.root")
f = r.TFile("results/e200_20200204/fulleff_e200_mtv.root")

prop_phi_2Slayer2 = f.Get("Root__prop_phi_2Slayer2").Clone()
p.plot_hist(
        bgs=[prop_phi_2Slayer2],
        legend_labels=["prop #phi layer6"],
        options={ "output_name":"plots_conditional/hit_nmiss2_phi_prop.pdf", "ratio_range":[0., 0.4], }
        )

prop_phi_2Slayer2 = f.Get("Root__prop_phi_2Slayer2_zoom_m02_p02").Clone()
p.plot_hist(
        bgs=[prop_phi_2Slayer2],
        legend_labels=["prop #phi layer6"],
        options={ "output_name":"plots_conditional/hit_nmiss2_phi_prop_m02_p02.pdf", "ratio_range":[0., 0.4], "remove_overflow":True, "remove_underflow":True }
        )

prop_phi_2Slayer1 = f.Get("Root__prop_phi_2Slayer1").Clone()
p.plot_hist(
        bgs=[prop_phi_2Slayer1],
        legend_labels=["prop #phi layer6"],
        options={ "output_name":"plots_conditional/hit_nmiss2_phi_prop_2Slayer1.pdf", "ratio_range":[0., 0.4], }
        )

prop_phi_2Slayer1 = f.Get("Root__prop_phi_2Slayer1_zoom_m02_p02").Clone()
p.plot_hist(
        bgs=[prop_phi_2Slayer1],
        legend_labels=["prop #phi layer6"],
        options={ "output_name":"plots_conditional/hit_nmiss2_phi_prop_m02_p02_2Slayer1.pdf", "ratio_range":[0., 0.4], "remove_overflow":True, "remove_underflow":True }
        )

prop_phi_2Slayer0 = f.Get("Root__prop_phi_2Slayer0").Clone()
p.plot_hist(
        bgs=[prop_phi_2Slayer0],
        legend_labels=["prop #phi layer6"],
        options={ "output_name":"plots_conditional/hit_nmiss2_phi_prop_2Slayer0.pdf", "ratio_range":[0., 0.4], }
        )

prop_phi_2Slayer0 = f.Get("Root__prop_phi_2Slayer0_zoom_m02_p02").Clone()
p.plot_hist(
        bgs=[prop_phi_2Slayer0],
        legend_labels=["prop #phi layer6"],
        options={ "output_name":"plots_conditional/hit_nmiss2_phi_prop_m02_p02_2Slayer0.pdf", "ratio_range":[0., 0.4], "remove_overflow":True, "remove_underflow":True }
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

h_nmiss_0 = f.Get("Root__eta_w_simhit_miss0").Clone()
h_nmiss_1 = f.Get("Root__eta_w_simhit_miss1").Clone()
h_nmiss_2 = f.Get("Root__eta_w_simhit_miss2").Clone()
h_nmiss_3 = f.Get("Root__eta_w_simhit_miss3").Clone()
h_nmiss_4 = f.Get("Root__eta_w_simhit_miss4").Clone()
h_nmiss_5 = f.Get("Root__eta_w_simhit_miss5").Clone()
h_nmiss_6 = f.Get("Root__eta_w_simhit_miss6").Clone()
h_nmiss_7 = f.Get("Root__eta_w_simhit_miss7").Clone()
h_nmiss_8 = f.Get("Root__eta_w_simhit_miss8").Clone()
h_nmiss_9 = f.Get("Root__eta_w_simhit_miss9").Clone()
h_nmiss_10 = f.Get("Root__eta_w_simhit_miss10").Clone()
h_nmiss_11 = f.Get("Root__eta_w_simhit_miss11").Clone()
h_nmiss_12 = f.Get("Root__eta_w_simhit_miss12").Clone()
# h_all     = f.Get("Root__eta_all_w_last_layer").Clone()
h_all     = f.Get("Root__eta_all").Clone()

p.plot_hist(
        bgs=[h_nmiss_0, h_nmiss_1, h_nmiss_2, h_nmiss_3, h_nmiss_4, h_nmiss_5, h_nmiss_6, h_nmiss_7, h_nmiss_8, h_nmiss_9, h_nmiss_10, h_nmiss_11, h_nmiss_12],
        data=h_all,
        legend_labels=["Nmiss0", "Nmiss1", "Nmiss2", "Nmiss3", "Nmiss4", "Nmiss5", "Nmiss6", "Nmiss7", "Nmiss8", "Nmiss9", "Nmiss10", "Nmiss11", "Nmiss12"],
        options={"output_name":"plots_conditional/hit_eta_efficiency.pdf"}
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

p.plot_hist(
        bgs=[h_all],
        data=h_nmiss_0,
        legend_labels=["All #mu tracks"],
        options={
            "output_name":"plots_conditional/hit_all_efficiency.pdf",
            "ratio_range":[0., 1.05],
            },
        )

h_nmiss_0 = f.Get("Root__eta_w_simhit_miss0").Clone()
h_nmiss_1 = f.Get("Root__eta_w_simhit_miss1").Clone()
h_nmiss_2 = f.Get("Root__eta_w_simhit_miss2").Clone()
h_nmiss_3 = f.Get("Root__eta_w_simhit_miss3").Clone()
h_nmiss_4 = f.Get("Root__eta_w_simhit_miss4").Clone()
h_nmiss_5 = f.Get("Root__eta_w_simhit_miss5").Clone()
h_nmiss_6 = f.Get("Root__eta_w_simhit_miss6").Clone()
h_nmiss_7 = f.Get("Root__eta_w_simhit_miss7").Clone()
h_nmiss_8 = f.Get("Root__eta_w_simhit_miss8").Clone()
h_nmiss_9 = f.Get("Root__eta_w_simhit_miss9").Clone()
h_nmiss_10 = f.Get("Root__eta_w_simhit_miss10").Clone()
h_nmiss_11 = f.Get("Root__eta_w_simhit_miss11").Clone()
h_nmiss_12 = f.Get("Root__eta_w_simhit_miss12").Clone()
# h_all     = f.Get("Root__eta_all_w_last_layer").Clone()
h_all     = f.Get("Root__eta_all").Clone()

p.plot_hist(
        bgs=[h_all],
        data=h_nmiss_0,
        legend_labels=["Nmiss0"],
        options={
            "output_name":"plots_conditional/hit_all_eta_efficiency.pdf",
            "ratio_range":[0., 1.05],
            },
        )

h_all     = f.Get("Root__eta_all").Clone()
p.plot_hist(
        bgs=[h_all],
        data=h_nmiss_2,
        legend_labels=["Nmiss2"],
        options={
            "output_name":"plots_conditional/hit_miss2_eta_efficiency.pdf",
            "ratio_range":[0., 0.3 ],
            },
        )

h_nmiss_miss_w_layer0 = f.Get("Root__pt_w_nmiss2_miss_layer0").Clone()
h_nmiss_miss_w_layer1 = f.Get("Root__pt_w_nmiss2_miss_layer1").Clone()
h_nmiss_miss_w_layer2 = f.Get("Root__pt_w_nmiss2_miss_layer2").Clone()
h_nmiss_miss_w_layer3 = f.Get("Root__pt_w_nmiss2_miss_layer3").Clone()
h_nmiss_miss_w_layer4 = f.Get("Root__pt_w_nmiss2_miss_layer4").Clone()
h_nmiss_miss_w_layer5 = f.Get("Root__pt_w_nmiss2_miss_layer5").Clone()
h_nmiss_miss_w_layer6 = f.Get("Root__pt_w_nmiss2_miss_layer6").Clone()
h_nmiss_miss_w_layer7 = f.Get("Root__pt_w_nmiss2_miss_layer7").Clone()
h_nmiss_miss_w_layer8 = f.Get("Root__pt_w_nmiss2_miss_layer8").Clone()
h_nmiss_miss_w_layer9 = f.Get("Root__pt_w_nmiss2_miss_layer9").Clone()
h_nmiss_miss_w_layer10 = f.Get("Root__pt_w_nmiss2_miss_layer10").Clone()
h_nmiss_miss_w_layer11 = f.Get("Root__pt_w_nmiss2_miss_layer11").Clone()

h_all = f.Get("Root__pt_w_simhit_miss2").Clone()
h_all.Scale(2)

p.plot_hist(
        bgs=[h_nmiss_miss_w_layer0, h_nmiss_miss_w_layer1 , h_nmiss_miss_w_layer2 , h_nmiss_miss_w_layer3 , h_nmiss_miss_w_layer4 , h_nmiss_miss_w_layer5 , h_nmiss_miss_w_layer6 , h_nmiss_miss_w_layer7 , h_nmiss_miss_w_layer8 , h_nmiss_miss_w_layer9 , h_nmiss_miss_w_layer10 , h_nmiss_miss_w_layer11],
        data=h_all,
        legend_labels=[
            "layer0",
            "layer1",
            "layer2",
            "layer3",
            "layer4",
            "layer5",
            "layer6",
            "layer7",
            "layer8",
            "layer9",
            "layer10",
            "layer11",
            ],
        options={
            "output_name":"plots_conditional/hit_nmiss2_pt_efficiency.pdf",
            "ratio_range":[0., 1.05],
            },
        )

p.plot_hist( bgs=[h_all.Clone()], data=h_nmiss_miss_w_layer0.Clone(), legend_labels=[ "nmiss2", ], options={ "output_name":"plots_conditional/hit_nmiss2_pt_efficiency_layer0.pdf", "ratio_range":[0., 0.4], },)
p.plot_hist( bgs=[h_all.Clone()], data=h_nmiss_miss_w_layer1.Clone(), legend_labels=[ "nmiss2", ], options={ "output_name":"plots_conditional/hit_nmiss2_pt_efficiency_layer1.pdf", "ratio_range":[0., 0.4], },)
p.plot_hist( bgs=[h_all.Clone()], data=h_nmiss_miss_w_layer2.Clone(), legend_labels=[ "nmiss2", ], options={ "output_name":"plots_conditional/hit_nmiss2_pt_efficiency_layer2.pdf", "ratio_range":[0., 0.4], },)
p.plot_hist( bgs=[h_all.Clone()], data=h_nmiss_miss_w_layer3.Clone(), legend_labels=[ "nmiss2", ], options={ "output_name":"plots_conditional/hit_nmiss2_pt_efficiency_layer3.pdf", "ratio_range":[0., 0.4], },)
p.plot_hist( bgs=[h_all.Clone()], data=h_nmiss_miss_w_layer4.Clone(), legend_labels=[ "nmiss2", ], options={ "output_name":"plots_conditional/hit_nmiss2_pt_efficiency_layer4.pdf", "ratio_range":[0., 0.4], },)
p.plot_hist( bgs=[h_all.Clone()], data=h_nmiss_miss_w_layer5.Clone(), legend_labels=[ "nmiss2", ], options={ "output_name":"plots_conditional/hit_nmiss2_pt_efficiency_layer5.pdf", "ratio_range":[0., 0.4], },)
p.plot_hist( bgs=[h_all.Clone()], data=h_nmiss_miss_w_layer6.Clone(), legend_labels=[ "nmiss2", ], options={ "output_name":"plots_conditional/hit_nmiss2_pt_efficiency_layer6.pdf", "ratio_range":[0., 0.4], },)
p.plot_hist( bgs=[h_all.Clone()], data=h_nmiss_miss_w_layer7.Clone(), legend_labels=[ "nmiss2", ], options={ "output_name":"plots_conditional/hit_nmiss2_pt_efficiency_layer7.pdf", "ratio_range":[0., 0.4], },)
p.plot_hist( bgs=[h_all.Clone()], data=h_nmiss_miss_w_layer8.Clone(), legend_labels=[ "nmiss2", ], options={ "output_name":"plots_conditional/hit_nmiss2_pt_efficiency_layer8.pdf", "ratio_range":[0., 0.4], },)
p.plot_hist( bgs=[h_all.Clone()], data=h_nmiss_miss_w_layer9.Clone(), legend_labels=[ "nmiss2", ], options={ "output_name":"plots_conditional/hit_nmiss2_pt_efficiency_layer9.pdf", "ratio_range":[0., 0.4], },)
p.plot_hist( bgs=[h_all.Clone()], data=h_nmiss_miss_w_layer10.Clone(), legend_labels=[ "nmiss2", ], options={ "output_name":"plots_conditional/hit_nmiss2_pt_efficiency_layer10.pdf", "ratio_range":[0., 0.4], },)
p.plot_hist( bgs=[h_all.Clone()], data=h_nmiss_miss_w_layer11.Clone(), legend_labels=[ "nmiss2", ], options={ "output_name":"plots_conditional/hit_nmiss2_pt_efficiency_layer11.pdf", "ratio_range":[0., 0.4], },)

h_nmiss_miss_w_layer0 = f.Get("Root__eta_w_nmiss2_miss_layer0").Clone()
h_nmiss_miss_w_layer1 = f.Get("Root__eta_w_nmiss2_miss_layer1").Clone()
h_nmiss_miss_w_layer2 = f.Get("Root__eta_w_nmiss2_miss_layer2").Clone()
h_nmiss_miss_w_layer3 = f.Get("Root__eta_w_nmiss2_miss_layer3").Clone()
h_nmiss_miss_w_layer4 = f.Get("Root__eta_w_nmiss2_miss_layer4").Clone()
h_nmiss_miss_w_layer5 = f.Get("Root__eta_w_nmiss2_miss_layer5").Clone()
h_nmiss_miss_w_layer6 = f.Get("Root__eta_w_nmiss2_miss_layer6").Clone()
h_nmiss_miss_w_layer7 = f.Get("Root__eta_w_nmiss2_miss_layer7").Clone()
h_nmiss_miss_w_layer8 = f.Get("Root__eta_w_nmiss2_miss_layer8").Clone()
h_nmiss_miss_w_layer9 = f.Get("Root__eta_w_nmiss2_miss_layer9").Clone()
h_nmiss_miss_w_layer10 = f.Get("Root__eta_w_nmiss2_miss_layer10").Clone()
h_nmiss_miss_w_layer11 = f.Get("Root__eta_w_nmiss2_miss_layer11").Clone()

h_all = f.Get("Root__eta_w_simhit_miss2").Clone()
h_all.Scale(2)

p.plot_hist( bgs=[h_all.Clone()], data=h_nmiss_miss_w_layer0.Clone(), legend_labels=[ "nmiss2", ], options={ "output_name":"plots_conditional/hit_nmiss2_eta_efficiency_layer0.pdf", "ratio_range":[0., 0.4], },)
p.plot_hist( bgs=[h_all.Clone()], data=h_nmiss_miss_w_layer1.Clone(), legend_labels=[ "nmiss2", ], options={ "output_name":"plots_conditional/hit_nmiss2_eta_efficiency_layer1.pdf", "ratio_range":[0., 0.4], },)
p.plot_hist( bgs=[h_all.Clone()], data=h_nmiss_miss_w_layer2.Clone(), legend_labels=[ "nmiss2", ], options={ "output_name":"plots_conditional/hit_nmiss2_eta_efficiency_layer2.pdf", "ratio_range":[0., 0.4], },)
p.plot_hist( bgs=[h_all.Clone()], data=h_nmiss_miss_w_layer3.Clone(), legend_labels=[ "nmiss2", ], options={ "output_name":"plots_conditional/hit_nmiss2_eta_efficiency_layer3.pdf", "ratio_range":[0., 0.4], },)
p.plot_hist( bgs=[h_all.Clone()], data=h_nmiss_miss_w_layer4.Clone(), legend_labels=[ "nmiss2", ], options={ "output_name":"plots_conditional/hit_nmiss2_eta_efficiency_layer4.pdf", "ratio_range":[0., 0.4], },)
p.plot_hist( bgs=[h_all.Clone()], data=h_nmiss_miss_w_layer5.Clone(), legend_labels=[ "nmiss2", ], options={ "output_name":"plots_conditional/hit_nmiss2_eta_efficiency_layer5.pdf", "ratio_range":[0., 0.4], },)
p.plot_hist( bgs=[h_all.Clone()], data=h_nmiss_miss_w_layer6.Clone(), legend_labels=[ "nmiss2", ], options={ "output_name":"plots_conditional/hit_nmiss2_eta_efficiency_layer6.pdf", "ratio_range":[0., 0.4], },)
p.plot_hist( bgs=[h_all.Clone()], data=h_nmiss_miss_w_layer7.Clone(), legend_labels=[ "nmiss2", ], options={ "output_name":"plots_conditional/hit_nmiss2_eta_efficiency_layer7.pdf", "ratio_range":[0., 0.4], },)
p.plot_hist( bgs=[h_all.Clone()], data=h_nmiss_miss_w_layer8.Clone(), legend_labels=[ "nmiss2", ], options={ "output_name":"plots_conditional/hit_nmiss2_eta_efficiency_layer8.pdf", "ratio_range":[0., 0.4], },)
p.plot_hist( bgs=[h_all.Clone()], data=h_nmiss_miss_w_layer9.Clone(), legend_labels=[ "nmiss2", ], options={ "output_name":"plots_conditional/hit_nmiss2_eta_efficiency_layer9.pdf", "ratio_range":[0., 0.4], },)
p.plot_hist( bgs=[h_all.Clone()], data=h_nmiss_miss_w_layer10.Clone(), legend_labels=[ "nmiss2", ], options={ "output_name":"plots_conditional/hit_nmiss2_eta_efficiency_layer10.pdf", "ratio_range":[0., 0.4], },)
p.plot_hist( bgs=[h_all.Clone()], data=h_nmiss_miss_w_layer11.Clone(), legend_labels=[ "nmiss2", ], options={ "output_name":"plots_conditional/hit_nmiss2_eta_efficiency_layer11.pdf", "ratio_range":[0., 0.4], },)

h_nmiss_0 = f.Get("Root__phi_w_simhit_miss0").Clone()
h_nmiss_1 = f.Get("Root__phi_w_simhit_miss1").Clone()
h_nmiss_2 = f.Get("Root__phi_w_simhit_miss2").Clone()
h_nmiss_3 = f.Get("Root__phi_w_simhit_miss3").Clone()
h_nmiss_4 = f.Get("Root__phi_w_simhit_miss4").Clone()
h_nmiss_5 = f.Get("Root__phi_w_simhit_miss5").Clone()
h_nmiss_6 = f.Get("Root__phi_w_simhit_miss6").Clone()
h_nmiss_7 = f.Get("Root__phi_w_simhit_miss7").Clone()
h_nmiss_8 = f.Get("Root__phi_w_simhit_miss8").Clone()
h_nmiss_9 = f.Get("Root__phi_w_simhit_miss9").Clone()
h_nmiss_10 = f.Get("Root__phi_w_simhit_miss10").Clone()
h_nmiss_11 = f.Get("Root__phi_w_simhit_miss11").Clone()
h_nmiss_12 = f.Get("Root__phi_w_simhit_miss12").Clone()
# h_all     = f.Get("Root__phi_all_w_last_layer").Clone()
h_all     = f.Get("Root__phi_all").Clone()

p.plot_hist(
        bgs=[h_nmiss_0, h_nmiss_1, h_nmiss_2, h_nmiss_3, h_nmiss_4, h_nmiss_5, h_nmiss_6, h_nmiss_7, h_nmiss_8, h_nmiss_9, h_nmiss_10, h_nmiss_11, h_nmiss_12],
        data=h_all,
        legend_labels=["Nmiss0", "Nmiss1", "Nmiss2", "Nmiss3", "Nmiss4", "Nmiss5", "Nmiss6", "Nmiss7", "Nmiss8", "Nmiss9", "Nmiss10", "Nmiss11", "Nmiss12"],
        options={"output_name":"plots_conditional/hit_phi_efficiency.pdf", "remove_overflow":True}
        )

h_nmiss_miss_w_layer0 = f.Get("Root__phi_w_nmiss2_miss_layer0").Clone()
h_nmiss_miss_w_layer1 = f.Get("Root__phi_w_nmiss2_miss_layer1").Clone()
h_nmiss_miss_w_layer2 = f.Get("Root__phi_w_nmiss2_miss_layer2").Clone()
h_nmiss_miss_w_layer3 = f.Get("Root__phi_w_nmiss2_miss_layer3").Clone()
h_nmiss_miss_w_layer4 = f.Get("Root__phi_w_nmiss2_miss_layer4").Clone()
h_nmiss_miss_w_layer5 = f.Get("Root__phi_w_nmiss2_miss_layer5").Clone()
h_nmiss_miss_w_layer6 = f.Get("Root__phi_w_nmiss2_miss_layer6").Clone()
h_nmiss_miss_w_layer7 = f.Get("Root__phi_w_nmiss2_miss_layer7").Clone()
h_nmiss_miss_w_layer8 = f.Get("Root__phi_w_nmiss2_miss_layer8").Clone()
h_nmiss_miss_w_layer9 = f.Get("Root__phi_w_nmiss2_miss_layer9").Clone()
h_nmiss_miss_w_layer10 = f.Get("Root__phi_w_nmiss2_miss_layer10").Clone()
h_nmiss_miss_w_layer11 = f.Get("Root__phi_w_nmiss2_miss_layer11").Clone()

h_all = f.Get("Root__phi_w_simhit_miss2").Clone()
h_all.Scale(2)

p.plot_hist( bgs=[h_all.Clone()], data=h_nmiss_miss_w_layer0.Clone(), legend_labels=[ "nmiss2", ], options={ "output_name":"plots_conditional/hit_nmiss2_phi_efficiency_layer0.pdf", "ratio_range":[0., 0.4], },)
p.plot_hist( bgs=[h_all.Clone()], data=h_nmiss_miss_w_layer1.Clone(), legend_labels=[ "nmiss2", ], options={ "output_name":"plots_conditional/hit_nmiss2_phi_efficiency_layer1.pdf", "ratio_range":[0., 0.4], },)
p.plot_hist( bgs=[h_all.Clone()], data=h_nmiss_miss_w_layer2.Clone(), legend_labels=[ "nmiss2", ], options={ "output_name":"plots_conditional/hit_nmiss2_phi_efficiency_layer2.pdf", "ratio_range":[0., 0.4], },)
p.plot_hist( bgs=[h_all.Clone()], data=h_nmiss_miss_w_layer3.Clone(), legend_labels=[ "nmiss2", ], options={ "output_name":"plots_conditional/hit_nmiss2_phi_efficiency_layer3.pdf", "ratio_range":[0., 0.4], },)
p.plot_hist( bgs=[h_all.Clone()], data=h_nmiss_miss_w_layer4.Clone(), legend_labels=[ "nmiss2", ], options={ "output_name":"plots_conditional/hit_nmiss2_phi_efficiency_layer4.pdf", "ratio_range":[0., 0.4], },)
p.plot_hist( bgs=[h_all.Clone()], data=h_nmiss_miss_w_layer5.Clone(), legend_labels=[ "nmiss2", ], options={ "output_name":"plots_conditional/hit_nmiss2_phi_efficiency_layer5.pdf", "ratio_range":[0., 0.4], },)
p.plot_hist( bgs=[h_all.Clone()], data=h_nmiss_miss_w_layer6.Clone(), legend_labels=[ "nmiss2", ], options={ "output_name":"plots_conditional/hit_nmiss2_phi_efficiency_layer6.pdf", "ratio_range":[0., 0.4], },)
p.plot_hist( bgs=[h_all.Clone()], data=h_nmiss_miss_w_layer7.Clone(), legend_labels=[ "nmiss2", ], options={ "output_name":"plots_conditional/hit_nmiss2_phi_efficiency_layer7.pdf", "ratio_range":[0., 0.4], },)
p.plot_hist( bgs=[h_all.Clone()], data=h_nmiss_miss_w_layer8.Clone(), legend_labels=[ "nmiss2", ], options={ "output_name":"plots_conditional/hit_nmiss2_phi_efficiency_layer8.pdf", "ratio_range":[0., 0.4], },)
p.plot_hist( bgs=[h_all.Clone()], data=h_nmiss_miss_w_layer9.Clone(), legend_labels=[ "nmiss2", ], options={ "output_name":"plots_conditional/hit_nmiss2_phi_efficiency_layer9.pdf", "ratio_range":[0., 0.4], },)
p.plot_hist( bgs=[h_all.Clone()], data=h_nmiss_miss_w_layer10.Clone(), legend_labels=[ "nmiss2", ], options={ "output_name":"plots_conditional/hit_nmiss2_phi_efficiency_layer10.pdf", "ratio_range":[0., 0.4], },)
p.plot_hist( bgs=[h_all.Clone()], data=h_nmiss_miss_w_layer11.Clone(), legend_labels=[ "nmiss2", ], options={ "output_name":"plots_conditional/hit_nmiss2_phi_efficiency_layer11.pdf", "ratio_range":[0., 0.4], },)

def plot_v1():

    h_nmiss_0 = f.Get("Root__pt_w_hit_miss0").Clone()
    h_nmiss_1 = f.Get("Root__pt_w_hit_miss1").Clone()
    h_nmiss_2 = f.Get("Root__pt_w_hit_miss2").Clone()
    h_nmiss_3 = f.Get("Root__pt_w_hit_miss3").Clone()
    h_nmiss_4 = f.Get("Root__pt_w_hit_miss4").Clone()
    h_nmiss_5 = f.Get("Root__pt_w_hit_miss5").Clone()
    h_nmiss_6 = f.Get("Root__pt_w_hit_miss6").Clone()
    h_nmiss_7 = f.Get("Root__pt_w_hit_miss7").Clone()
    h_nmiss_8 = f.Get("Root__pt_w_hit_miss8").Clone()
    h_nmiss_9 = f.Get("Root__pt_w_hit_miss9").Clone()
    h_nmiss_10 = f.Get("Root__pt_w_hit_miss10").Clone()
    # h_all     = f.Get("Root__pt_all_w_last_layer").Clone()
    h_all     = f.Get("Root__pt_all").Clone()
    
    p.plot_hist(
            bgs=[h_nmiss_0, h_nmiss_1, h_nmiss_2, h_nmiss_3, h_nmiss_4, h_nmiss_5],
            data=h_all,
            legend_labels=["Nmiss0", "Nmiss1", "Nmiss2", "Nmiss3", "Nmiss4", "Nmiss5"],
            options={"output_name":"plots_conditional/hit_efficiency.pdf"}
            )

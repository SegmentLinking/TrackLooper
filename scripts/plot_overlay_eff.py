#!/bin/env python

import ROOT as r

mdeff_file = r.TFile("results/pt0p5_2p0_20200210_copied_from_20200131_v1/mdeff.root");
sgeff_file = r.TFile("results/pt0p5_2p0_20200210_copied_from_20200131_v1/sgeff.root");
tleff_file = r.TFile("results/pt0p5_2p0_20200210_copied_from_20200131_v1/tleff.root");
tceff_file = r.TFile("results/pt0p5_2p0_20200210_copied_from_20200131_v1/tceff.root");

md_eff_barrel_pt_by_layer0 = mdeff_file.Get("md_eff_barrel_pt_by_layer0.pdf")
md_eff_barrel_pt_by_layer1 = mdeff_file.Get("md_eff_barrel_pt_by_layer1.pdf")
md_eff_barrel_pt_by_layer2 = mdeff_file.Get("md_eff_barrel_pt_by_layer2.pdf")
md_eff_barrel_pt_by_layer3 = mdeff_file.Get("md_eff_barrel_pt_by_layer3.pdf")
md_eff_barrel_pt_by_layer4 = mdeff_file.Get("md_eff_barrel_pt_by_layer4.pdf")
md_eff_barrel_pt_by_layer5 = mdeff_file.Get("md_eff_barrel_pt_by_layer5.pdf")

sg_eff_bb_pt_by_layer0 = sgeff_file.Get("sg_eff_bb_pt_by_layer0.pdf")
sg_eff_bb_pt_by_layer1 = sgeff_file.Get("sg_eff_bb_pt_by_layer1.pdf")
sg_eff_bb_pt_by_layer2 = sgeff_file.Get("sg_eff_bb_pt_by_layer2.pdf")
sg_eff_bb_pt_by_layer3 = sgeff_file.Get("sg_eff_bb_pt_by_layer3.pdf")
sg_eff_bb_pt_by_layer4 = sgeff_file.Get("sg_eff_bb_pt_by_layer4.pdf")

tl_eff_bbbb_pt_by_layer0 = tleff_file.Get("tl_eff_bbbb_pt_by_layer0.pdf")
tl_eff_bbbb_pt_by_layer1 = tleff_file.Get("tl_eff_bbbb_pt_by_layer1.pdf")
tl_eff_bbbb_pt_by_layer2 = tleff_file.Get("tl_eff_bbbb_pt_by_layer2.pdf")

tc_eff_bbbbbb_pt_by_layer0 = tceff_file.Get("tc_eff_bbbbbb_pt_by_layer0.pdf")

c1 = r.TCanvas()


def draw(blackline, redline, blueline, outputname):
    # TC and TL comparison
    blackline.Draw("epa")
    blackline.SetMarkerSize(0.5)
    blackline.SetMarkerColor(r.kBlack)
    blackline.SetLineColor(r.kBlack)
    blackline.GetYaxis().SetRangeUser(0.7, 1.05)
    blackline.GetXaxis().SetRangeUser(0.8, 1.5)
    # md_eff_barrel_pt_by_layer0.Draw("epsame")
    redline.SetMarkerColor(r.kRed)
    blueline.SetMarkerColor(r.kBlue)
    redline.SetMarkerSize(1)
    blueline.SetMarkerSize(1)
    redline.SetLineColor(r.kRed)
    blueline.SetLineColor(r.kBlue)
    redline.Draw("epsame")
    blueline.Draw("epsame")
    c1.SetLogx()
    c1.SaveAs(outputname)

draw(tc_eff_bbbbbb_pt_by_layer0, tl_eff_bbbb_pt_by_layer0, tl_eff_bbbb_pt_by_layer2, "tc_tl02.pdf")
draw(tl_eff_bbbb_pt_by_layer0, sg_eff_bb_pt_by_layer0, sg_eff_bb_pt_by_layer2, "tl0_sg02.pdf")
draw(sg_eff_bb_pt_by_layer0, md_eff_barrel_pt_by_layer0, md_eff_barrel_pt_by_layer1, "sg0_md01.pdf")

draw(md_eff_barrel_pt_by_layer0, md_eff_barrel_pt_by_layer1, md_eff_barrel_pt_by_layer2, "PS_mds.pdf")
draw(md_eff_barrel_pt_by_layer3, md_eff_barrel_pt_by_layer4, md_eff_barrel_pt_by_layer5, "2S_mds.pdf")

draw(tc_eff_bbbbbb_pt_by_layer0, md_eff_barrel_pt_by_layer0, md_eff_barrel_pt_by_layer1, "tc_md01_mds.pdf")
draw(tc_eff_bbbbbb_pt_by_layer0, md_eff_barrel_pt_by_layer2, md_eff_barrel_pt_by_layer3, "tc_md23_mds.pdf")
draw(tc_eff_bbbbbb_pt_by_layer0, md_eff_barrel_pt_by_layer4, md_eff_barrel_pt_by_layer5, "tc_md45_mds.pdf")

draw(tc_eff_bbbbbb_pt_by_layer0, md_eff_barrel_pt_by_layer1, md_eff_barrel_pt_by_layer5, "tc_md15_mds.pdf")

# md_eff_barrel_pt_by_layer0.Setj
# md_eff_barrel_pt_by_layer1 = mdeff_file.Get("md_eff_barrel_pt_by_layer1.pdf")
# md_eff_barrel_pt_by_layer2 = mdeff_file.Get("md_eff_barrel_pt_by_layer2.pdf")
# md_eff_barrel_pt_by_layer3 = mdeff_file.Get("md_eff_barrel_pt_by_layer3.pdf")
# md_eff_barrel_pt_by_layer4 = mdeff_file.Get("md_eff_barrel_pt_by_layer4.pdf")
# md_eff_barrel_pt_by_layer5 = mdeff_file.Get("md_eff_barrel_pt_by_layer5.pdf")

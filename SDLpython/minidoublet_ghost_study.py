#!/bin/env python

import ROOT as r
import sys

########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################
def studying_how_many_minidoublets_in_track_candidates_have_doubles():
    f = r.TFile("/home/users/phchang/public_html/analysis/sdl/TrackLooper_/results/write_sdl_ntuple/pt0p5_2p0_20200913_Test_v2//fulleff_pt0p5_2p0.root")
    t = f.Get("tree")
    n_mds_per_hit_in_tc = []
    h = r.TH1F("n_mds_per_hit_in_tc", "", 5, 0, 5)
    for ievent, event in enumerate(t):
        print ievent
        for tc_hitIdx in event.tc_hitIdx:
            for hitIdx in tc_hitIdx:
                h.Fill(len(list(event.ph2_mdIdx[hitIdx])))
        if ievent > 9:
            break
    c1 = r.TCanvas()
    h.Draw()
    c1.SaveAs("n_mds_per_hit_in_tc.pdf")

########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################
def studying_how_many_minidoublets_have_doubles():
    f = r.TFile("/home/users/phchang/public_html/analysis/sdl/TrackLooper_/results/write_sdl_ntuple/pt0p5_2p0_20200913_Test_v2//fulleff_pt0p5_2p0.root")
    t = f.Get("tree")
    h = r.TH1F("n_mds_per_hit", "", 5, 0, 5)
    for ievent, event in enumerate(t):
        print ievent
        if ievent > 100:
            break
        for ihit, _ in enumerate(event.ph2_mdIdx):
            mdIdx = list(event.ph2_mdIdx[ihit])
            if len(mdIdx) == 0:
                continue
            h.Fill(len(mdIdx))
    c1 = r.TCanvas()
    h.Draw()
    c1.SaveAs("n_mds_per_hit.pdf")

########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################
def studying_purity_of_minidoublets_by_whether_or_not_md_has_doubles():
    f = r.TFile("/home/users/phchang/public_html/analysis/sdl/TrackLooper_/results/write_sdl_ntuple/pt0p5_2p0_20200913_Test_v2//fulleff_pt0p5_2p0.root")
    t = f.Get("tree")
    h1 = r.TH1F("purity_md_mult_1", "", 2, 0, 2)
    h2 = r.TH1F("purity_md_mult_2", "", 2, 0, 2)
    h3 = r.TH1F("purity_md_mult_3", "", 2, 0, 2)
    h5 = r.TH1F("more_than_2", "", 5, 0, 5)
    for ievent, event in enumerate(t):
        print ievent
        if ievent > 100:
            break
        for ihit, _ in enumerate(event.ph2_mdIdx):
            mdIdx = list(event.ph2_mdIdx[ihit])
            if len(mdIdx) == 0:
                continue
            if len(mdIdx) == 1:
                imd = mdIdx[0]
                if len(list(event.md_simTrkIdx[imd])) > 0:
                    h1.Fill(1)
                else:
                    h1.Fill(0)
            if len(mdIdx) == 2:
                # first one
                imd = mdIdx[0]
                if len(list(event.md_simTrkIdx[imd])) > 0:
                    h2.Fill(1)
                else:
                    h2.Fill(0)
                # second one
                imd = mdIdx[1]
                if len(list(event.md_simTrkIdx[imd])) > 0:
                    h2.Fill(1)
                else:
                    h2.Fill(0)
            if len(mdIdx) == 3:
                # first one
                imd = mdIdx[0]
                if len(list(event.md_simTrkIdx[imd])) > 0:
                    h3.Fill(1)
                else:
                    h3.Fill(0)
                # second one
                imd = mdIdx[1]
                if len(list(event.md_simTrkIdx[imd])) > 0:
                    h3.Fill(1)
                else:
                    h3.Fill(0)
                # third one
                imd = mdIdx[2]
                if len(list(event.md_simTrkIdx[imd])) > 0:
                    h3.Fill(1)
                else:
                    h3.Fill(0)
            if len(mdIdx) > 2:
                h3.Fill(len(mdIdx))
    c1 = r.TCanvas()
    h1.Draw()
    c1.SaveAs("purity_md_mult_1.pdf")
    h2.Draw()
    c1.SaveAs("purity_md_mult_2.pdf")
    h3.Draw()
    c1.SaveAs("purity_md_mult_3.pdf")
    h5.Draw()
    c1.SaveAs("more_than_2.pdf")

########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################
def studying_correct_and_incorrect_md_differences():
    f = r.TFile("/home/users/phchang/public_html/analysis/sdl/TrackLooper_/results/write_sdl_ntuple/pt0p5_2p0_20200913_Test_v2//fulleff_pt0p5_2p0.root")
    t = f.Get("tree")
    h_dphiChange_good = r.TH1F("dphiChange_good", "", 50, -1, 1)
    h_dphiChange_bad = r.TH1F("dphiChange_bad", "", 50, -1, 1)
    for ievent, event in enumerate(t):
        print ievent
        if ievent > 500:
            break
        for ihit, _ in enumerate(event.ph2_mdIdx):
            mdIdx = list(event.ph2_mdIdx[ihit])
            if len(mdIdx) > 1:
                for imd in mdIdx:
                    is_good_md = len(list(event.md_simTrkIdx[imd])) > 0
                    if is_good_md:
                        # h_dphiChange_good.Fill(abs(event.md_dphiChange[imd])-event.md_miniCut[imd])
                        h_dphiChange_good.Fill(event.md_dphiChange[imd])
                    else:
                        # h_dphiChange_bad.Fill(abs(event.md_dphiChange[imd])-event.md_miniCut[imd])
                        h_dphiChange_bad.Fill(event.md_dphiChange[imd])
    c1 = r.TCanvas()
    h_dphiChange_good.Draw()
    c1.SaveAs("dphiChange_good.pdf")
    h_dphiChange_bad.Draw()
    c1.SaveAs("dphiChange_bad.pdf")

if __name__ == "__main__":
    studying_correct_and_incorrect_md_differences()

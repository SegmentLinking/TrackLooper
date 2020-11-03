#!/bin/env python

import ROOT as r
from array import array

# f = r.TFile("/home/users/phchang/public_html/analysis/sdl/TrackLooper_/results/write_sdl_ntuple/pu200_w_truth_pdgid211_20200919_Test_v2_w_gap//fulleff_pu200_w_truth_pdgid211.root")
# f = r.TFile("/home/users/phchang/public_html/analysis/sdl/TrackLooper_/results/write_sdl_ntuple/pu200_w_truth_pdgid211_20200919_Test_v2_w_gap/fulleff_pu200_w_truth_pdgid211_0.root")
f = r.TFile("/nfs-7/userdata/bsathian/SDL_trackingNtuple/ttbar_highPU/trackingNtuple_with_PUinfo_500_evts.root")
t = f.Get("trackingNtuple/tree")

pt_boundaries = [0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0, 3.0, 5.0, 10, 15., 25, 50]
pt_denom = r.TH1F("pt_denom", "pt_denom", len(pt_boundaries)-1, array('d', pt_boundaries))

for ievent, event in enumerate(t):

    # print "len(event.md_hitIdx)", len(event.md_hitIdx)
    # print "len(event.sg_hitIdx)", len(event.sg_hitIdx)
    # print "len(event.tl_hitIdx)", len(event.tl_hitIdx)
    # print "len(event.tp_hitIdx)", len(event.tp_hitIdx)
    # print "len(event.tc_hitIdx)", len(event.tc_hitIdx)

    n_denom_pions = 0

    for isim, _ in enumerate(event.sim_pt):

        pt = event.sim_pt[isim]
        eta = event.sim_eta[isim]
        dz = event.sim_pca_dz[isim]
        dxy = event.sim_pca_dxy[isim]
        pdgid = event.sim_pdgId[isim]
        bunch = event.sim_bunchCrossing[isim]
        # n_mtv_tc = len(list(event.sim_tcIdx_isMTVmatch[isim]))

        if abs(eta) < 2.4 and abs(dz) < 30. and abs(dxy) < 2.5 and abs(pdgid) == 211 and abs(bunch) == 0:

            n_denom_pions += 1

            pt_denom.Fill(pt)

    print n_denom_pions

    break

pt_denom.Print("all")

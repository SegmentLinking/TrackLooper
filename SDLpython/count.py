#!/usr/bin/env python

import ROOT as r
import numpy as np
from Module import Module
from sdlmath import getCenterFromThreePoints
import math
from collections import Counter

# f = r.TFile("/home/users/phchang/public_html/analysis/sdl/TrackLooper_/results/write_sdl_ntuple/pu200_w_truthinfo_charged_2020_0723_starbucksBundang/fulleff_pu200_w_truthinfo_charged.root")
# f = r.TFile("/home/users/phchang/public_html/analysis/sdl/TrackLooper_/results/write_sdl_ntuple/pu200_w_truthinfo_charged_2020_0730_ICHEP/fulleff_pu200_w_truthinfo_charged.root")
# f = r.TFile("/home/users/phchang/public_html/analysis/sdl/TrackLooper_/results/write_sdl_ntuple/pu200_w_truthinfo_charged_2020_0801_Chuncheon//fulleff_pu200_w_truthinfo_charged.root")
# f = r.TFile("/home/users/phchang/public_html/analysis/sdl/TrackLooper_/results/write_sdl_ntuple/pu200_w_truthinfo_charged_2020_0805_Danyang/fulleff_pu200_w_truthinfo_charged.root")
# f = r.TFile("/home/users/phchang/public_html/analysis/sdl/TrackLooper_/results/write_sdl_ntuple/pu200_w_truthinfo_charged_2020_0805_Danyang_tlgt4/fulleff_pu200_w_truthinfo_charged.root")
# f = r.TFile("/home/users/phchang/public_html/analysis/sdl/TrackLooper_/results/write_sdl_ntuple/pu200_w_truthinfo_charged_2020_0819_SanDiego//fulleff_pu200_w_truthinfo_charged.root") # tracklet match > 4
# f = r.TFile("/home/users/phchang/public_html/analysis/sdl/TrackLooper_/results/write_sdl_ntuple/pu200_w_truthinfo_charged_2020_0819_SanDiego_tlgt6//fulleff_pu200_w_truthinfo_charged.root")
# f = r.TFile("/home/users/phchang/public_html/analysis/sdl/TrackLooper_/results/write_sdl_ntuple/pu200_w_truthinfo_charged_20200825_Test//fulleff_pu200_w_truthinfo_charged.root")
# f = r.TFile("/home/users/phchang/public_html/analysis/sdl/TrackLooper_/results/write_sdl_ntuple/pu200_w_truthinfo_charged_20200826_Triplet_Test1//fulleff_pu200_w_truthinfo_charged_0.root")
# f = r.TFile("/home/users/phchang/public_html/analysis/sdl/TrackLooper_/results/write_sdl_ntuple/pu200_w_truthinfo_charged_20200826_Triplet_Test2//fulleff_pu200_w_truthinfo_charged_0.root")
# f = r.TFile("/home/users/phchang/public_html/analysis/sdl/TrackLooper_/results/write_sdl_ntuple/pu200_w_truthinfo_charged_20200918_Test_v2//fulleff_pu200_w_truthinfo_charged.root")
f = r.TFile("/home/users/phchang/public_html/analysis/sdl/TrackLooper_/results/write_sdl_ntuple/pu200_w_truthinfo_charged_20200919_ReproducingOldNumber//fulleff_pu200_w_truthinfo_charged.root")
# f = r.TFile("debug.root")
tree = f.Get("tree")

segment_layer_type = [
        "1-2",
        "2-3",
        "3-4",
        "4-5",
        "5-6",
        "1-7",
        "2-7",
        "3-7",
        "4-7",
        "5-7",
        "7-8",
        "8-9",
        "9-10",
        "10-11",
        ]

tracklet_layer_type = {
        24   : "1-2-3-4",
        42   : "1-2-3-7",
        112  : "1-2-7-8",
        504  : "1-7-8-9",
        120  : "2-3-4-5",
        168  : "2-3-4-7",
        336  : "2-3-7-8",
        1008 : "2-7-8-9",
        360  : "3-4-5-6",
        420  : "3-4-5-7",
        672  : "3-4-7-8",
        1512 : "3-7-8-9",
        1120 : "4-5-7-8",
        2016 : "4-7-8-9",
        5040 : "7-8-9-10",
        7920 : "8-9-10-11"
        }

tracklet_layer_type_keys = [
        24   ,
        42   ,
        112  ,
        504  ,
        120  ,
        168  ,
        336  ,
        1008 ,
        360  ,
        420  ,
        672  ,
        1512 ,
        1120 ,
        2016 ,
        5040 ,
        7920 ,
        ]

triplet_layer_type = {
        6:  "1-2-3",
        14: "1-2-7",
        56: "1-7-8",
        24: "2-3-4",
        42: "2-3-7",
        112:"2-7-8",
        60: "3-4-5",
        84: "3-4-7",
        168:"3-7-8",
        120:"4-5-6",
        140:"4-5-7",
        224:"4-7-8",
        280:"5-7-8",
        504:"7-8-9",
        720:"8-9-10",
        990:"9-10-11",
        }

triplet_layer_type_keys = [
        6,
        14,
        56,
        24,
        42,
        112,
        60,
        84,
        168,
        120,
        140,
        224,
        280,
        504,
        720,
        990,
        ]

# for ievent, event in enumerate(tree):
#     print ievent, "events processed"
#     for simTrkIdx in event.simhit_simTrkIdx:
#         if simTrkIdx < 0:
#             print "there is a simhit without any match to simtrack"
# import sys
# sys.exit()


for ievent, event in enumerate(tree):

    print ievent, "processing"

    # Count Hits of interest
    ph2_pt = []
    ph2_eta = []
    ph2_match = []
    ph2_layer = []
    ph2_simtype = []
    ph2_hasMD = []
    ph2_anchor = []
    for index, (ph2_simHitIdx, ph2_detId, ph2_simType, ph2_mdIdx, ph2_anchorLayer) in enumerate(zip(event.ph2_simHitIdx, event.ph2_detId, event.ph2_simType, event.ph2_mdIdx, event.ph2_anchorLayer)):
        module = Module(ph2_detId)
        if not (module.subdet() == 4 or module.subdet() == 5):
            continue
        ph2_simtype.append(ph2_simType)
        if len(ph2_mdIdx) > 0:
            ph2_hasMD.append(1)
        else:
            ph2_hasMD.append(0)
        ph2_anchor.append(ph2_anchorLayer)
        if len(ph2_simHitIdx) > 0:
            maxpt = 0
            maxeta = 0
            for isimhit in ph2_simHitIdx:
                tmppt = event.sim_pt[event.simhit_simTrkIdx[isimhit]]
                tmpeta = event.sim_eta[event.simhit_simTrkIdx[isimhit]]
                if tmppt > maxpt:
                    maxpt = tmppt
                    maxeta = tmpeta
            ph2_pt.append(maxpt)
            ph2_eta.append(maxeta)
            ph2_match.append(1)
            ph2_layer.append(module.logicalLayer())
        else:
            ph2_pt.append(-999)
            ph2_eta.append(-999)
            ph2_match.append(0)
            ph2_layer.append(module.logicalLayer())

    # Count Hits of interest
    # Count Mini-Doublet of interest
    md_pt = []
    md_eta = []
    md_match = []
    md_reco_pt = []
    md_layer = []
    md_goodsimtypes = []
    for index, (md_simTrkIdx, md_hitIdx) in enumerate(zip(event.md_simTrkIdx, event.md_hitIdx)):
        center = getCenterFromThreePoints([0, 0], [event.ph2_x[md_hitIdx[0]], event.ph2_y[md_hitIdx[0]]], [event.ph2_x[md_hitIdx[1]], event.ph2_y[md_hitIdx[1]]])
        radius = math.sqrt(center[0]**2 + center[1]**2)
        reco_pt = 2.99792458e-3 * 3.8 * radius;
        detid = event.ph2_detId[md_hitIdx[0]]
        module = Module(detid)
        md_reco_pt.append(reco_pt)
        noise_0 = (event.ph2_simType[md_hitIdx[0]] == 3)
        noise_1 = (event.ph2_simType[md_hitIdx[1]] == 3)
        is_noise = noise_0 + noise_1
        md_goodsimtypes.append(is_noise)
        if len(md_simTrkIdx) > 0:
            maxpt = 0
            maxeta = 0
            for isimtrk in md_simTrkIdx:
                tmppt = event.sim_pt[isimtrk]
                tmpeta = event.sim_eta[isimtrk]
                if tmppt > maxpt:
                    maxpt = tmppt
                    maxeta = tmpeta
            md_pt.append(maxpt)
            md_eta.append(maxeta)
            md_match.append(1)
            md_layer.append(module.logicalLayer())
        else:
            md_pt.append(-999)
            md_eta.append(-999)
            md_match.append(0)
            md_layer.append(module.logicalLayer())

    # Count Segment of interest
    sg_pt = []
    sg_eta = []
    sg_match = []
    sg_reco_pt = []
    sg_layer = []
    sg_goodsimtypes = []
    for index, (sg_simTrkIdx, sg_hitIdx, sg_simTrkIdx_anchorMatching) in enumerate(zip(event.sg_simTrkIdx, event.sg_hitIdx, event.sg_simTrkIdx_anchorMatching)):
        center = getCenterFromThreePoints([0, 0], [event.ph2_x[sg_hitIdx[0]], event.ph2_y[sg_hitIdx[0]]], [event.ph2_x[sg_hitIdx[3]], event.ph2_y[sg_hitIdx[3]]])
        radius = math.sqrt(center[0]**2 + center[1]**2)
        reco_pt = 2.99792458e-3 * 3.8 * radius;
        sg_reco_pt.append(reco_pt)
        detid_lower = event.ph2_detId[sg_hitIdx[0]]
        detid_upper = event.ph2_detId[sg_hitIdx[3]]
        module_lower = Module(detid_lower)
        module_upper = Module(detid_upper)

        segment_layer = 0

        if module_lower.logicalLayer() == 1 and module_upper.logicalLayer() == 2: segment_layer = 1
        if module_lower.logicalLayer() == 2 and module_upper.logicalLayer() == 3: segment_layer = 2
        if module_lower.logicalLayer() == 3 and module_upper.logicalLayer() == 4: segment_layer = 3
        if module_lower.logicalLayer() == 4 and module_upper.logicalLayer() == 5: segment_layer = 4
        if module_lower.logicalLayer() == 5 and module_upper.logicalLayer() == 6: segment_layer = 5

        if module_lower.logicalLayer() == 1 and module_upper.logicalLayer() == 7: segment_layer = 6
        if module_lower.logicalLayer() == 2 and module_upper.logicalLayer() == 7: segment_layer = 7
        if module_lower.logicalLayer() == 3 and module_upper.logicalLayer() == 7: segment_layer = 8
        if module_lower.logicalLayer() == 4 and module_upper.logicalLayer() == 7: segment_layer = 9
        if module_lower.logicalLayer() == 5 and module_upper.logicalLayer() == 7: segment_layer = 10

        if module_lower.logicalLayer() == 7  and module_upper.logicalLayer() == 8:  segment_layer = 11
        if module_lower.logicalLayer() == 8  and module_upper.logicalLayer() == 9:  segment_layer = 12
        if module_lower.logicalLayer() == 9  and module_upper.logicalLayer() == 10: segment_layer = 13
        if module_lower.logicalLayer() == 10 and module_upper.logicalLayer() == 11: segment_layer = 14

        sg_layer.append(segment_layer)

        noise_0 = (event.ph2_simType[sg_hitIdx[0]] == 3)
        noise_1 = (event.ph2_simType[sg_hitIdx[1]] == 3)
        noise_2 = (event.ph2_simType[sg_hitIdx[2]] == 3)
        noise_3 = (event.ph2_simType[sg_hitIdx[3]] == 3)
        is_noise = noise_0 + noise_1 + noise_2 + noise_3
        sg_goodsimtypes.append(is_noise)

        if len(sg_simTrkIdx_anchorMatching) > 0:
            maxpt = 0
            maxeta = 0
            for isimtrk in sg_simTrkIdx_anchorMatching:
                tmppt = event.sim_pt[isimtrk]
                tmpeta = event.sim_eta[isimtrk]
                if tmppt > maxpt:
                    maxpt = tmppt
                    maxeta = tmpeta
            sg_pt.append(maxpt)
            sg_eta.append(maxeta)
            sg_match.append(1)
        else:
            sg_pt.append(-999)
            sg_eta.append(-999)
            sg_match.append(0)

    # Count Triplet of interest
    tp_pt = []
    tp_eta = []
    tp_match = []
    tp_reco_pt = []
    tp_layer = []
    tp_goodsimtypes = []
    for index, (tp_simTrkIdx, tp_hitIdx) in enumerate(zip(event.tp_simTrkIdx, event.tp_hitIdx)):
        center = getCenterFromThreePoints([0, 0], [event.ph2_x[tp_hitIdx[0]], event.ph2_y[tp_hitIdx[0]]], [event.ph2_x[tp_hitIdx[5]], event.ph2_y[tp_hitIdx[5]]])
        radius = math.sqrt(center[0]**2 + center[1]**2)
        reco_pt = 2.99792458e-3 * 3.8 * radius;
        tp_reco_pt.append(reco_pt)
        detid_1 = event.ph2_detId[tp_hitIdx[0]]
        detid_2 = event.ph2_detId[tp_hitIdx[2]]
        detid_3 = event.ph2_detId[tp_hitIdx[4]]
        module_1 = Module(detid_1)
        module_2 = Module(detid_2)
        module_3 = Module(detid_3)
        layer_1 = module_1.logicalLayer()
        layer_2 = module_2.logicalLayer()
        layer_3 = module_3.logicalLayer()
        layer_combo = layer_1 * layer_2 * layer_3
        tp_layer.append(layer_combo)

        noise_0 = (event.ph2_simType[tp_hitIdx[0]] == 3)
        noise_1 = (event.ph2_simType[tp_hitIdx[1]] == 3)
        noise_2 = (event.ph2_simType[tp_hitIdx[2]] == 3)
        noise_3 = (event.ph2_simType[tp_hitIdx[3]] == 3)
        noise_4 = (event.ph2_simType[tp_hitIdx[4]] == 3)
        noise_5 = (event.ph2_simType[tp_hitIdx[5]] == 3)
        is_noise = noise_0 + noise_1 + noise_2 + noise_3 + noise_4 + noise_5
        tp_goodsimtypes.append(is_noise)

        if len(tp_simTrkIdx) > 0:
            maxpt = 0
            maxeta = 0
            for isimtrk in tp_simTrkIdx:
                tmppt = event.sim_pt[isimtrk]
                tmpeta = event.sim_eta[isimtrk]
                if tmppt > maxpt:
                    maxpt = tmppt
                    maxeta = tmpeta
                break
            tp_pt.append(maxpt)
            tp_eta.append(maxeta)
            tp_match.append(1)
        else:
            tp_pt.append(-999)
            tp_eta.append(-999)
            tp_match.append(0)

    # Count Tracklet of interest
    tl_pt = []
    tl_eta = []
    tl_match = []
    tl_reco_pt = []
    tl_layer = []
    tl_goodsimtypes = []
    for index, (tl_simTrkIdx, tl_hitIdx) in enumerate(zip(event.tl_simTrkIdx, event.tl_hitIdx)):
        center = getCenterFromThreePoints([0, 0], [event.ph2_x[tl_hitIdx[0]], event.ph2_y[tl_hitIdx[0]]], [event.ph2_x[tl_hitIdx[7]], event.ph2_y[tl_hitIdx[7]]])
        radius = math.sqrt(center[0]**2 + center[1]**2)
        reco_pt = 2.99792458e-3 * 3.8 * radius;
        tl_reco_pt.append(reco_pt)
        detid_1 = event.ph2_detId[tl_hitIdx[0]]
        detid_2 = event.ph2_detId[tl_hitIdx[2]]
        detid_3 = event.ph2_detId[tl_hitIdx[4]]
        detid_4 = event.ph2_detId[tl_hitIdx[6]]
        module_1 = Module(detid_1)
        module_2 = Module(detid_2)
        module_3 = Module(detid_3)
        module_4 = Module(detid_4)
        layer_1 = module_1.logicalLayer()
        layer_2 = module_2.logicalLayer()
        layer_3 = module_3.logicalLayer()
        layer_4 = module_4.logicalLayer()
        layer_combo = layer_1 * layer_2 * layer_3 * layer_4
        tl_layer.append(layer_combo)

        noise_0 = (event.ph2_simType[tl_hitIdx[0]] == 3)
        noise_1 = (event.ph2_simType[tl_hitIdx[1]] == 3)
        noise_2 = (event.ph2_simType[tl_hitIdx[2]] == 3)
        noise_3 = (event.ph2_simType[tl_hitIdx[3]] == 3)
        noise_4 = (event.ph2_simType[tl_hitIdx[4]] == 3)
        noise_5 = (event.ph2_simType[tl_hitIdx[5]] == 3)
        noise_6 = (event.ph2_simType[tl_hitIdx[6]] == 3)
        noise_7 = (event.ph2_simType[tl_hitIdx[7]] == 3)
        is_noise = noise_0 + noise_1 + noise_2 + noise_3 + noise_4 + noise_5 + noise_6 + noise_7
        tl_goodsimtypes.append(is_noise)

        if len(tl_simTrkIdx) > 0:
            maxpt = 0
            maxeta = 0
            for isimtrk in tl_simTrkIdx:
                tmppt = event.sim_pt[isimtrk]
                tmpeta = event.sim_eta[isimtrk]
                if tmppt > maxpt:
                    maxpt = tmppt
                    maxeta = tmpeta
                break
            tl_pt.append(maxpt)
            tl_eta.append(maxeta)
            tl_match.append(1)
        else:
            tl_pt.append(-999)
            tl_eta.append(-999)
            tl_match.append(0)

    # Count Tracklet of interest
    sim_pt = []
    sim_eta = []
    sim_layer = []
    for index, (pt, eta, pdgid, q, simHitIdx, simhitboth, drfrac) in enumerate(zip(event.sim_pt, event.sim_eta, event.sim_pdgId, event.sim_q, event.sim_simHitIdx, event.sim_simHitBoth, event.sim_simHitDrFracWithHelix)):
        if pt < 1 or abs(eta) > 2.4:
            continue
        if q == 0:
            continue
        sim_pt.append(pt)
        sim_eta.append(eta)
        layers = []
        for isimhit, both, drf in zip(simHitIdx, simhitboth, drfrac):
            if not (event.simhit_subdet[isimhit] == 4 or event.simhit_subdet[isimhit] == 5):
                continue
            if drf > 0.05:
                continue
            if not both:
                continue
            if q == 0:
                continue
            module = Module(event.simhit_detId[isimhit])
            layers.append(module.logicalLayer())
        layers = list(set(layers))
        layers.sort()
        # if len(layers) == 0:
        #     print "layers == 0"
        #     print index
        #     print simHitIdx
        #     print simhitboth
        #     print drfrac
        #     print [ event.simhit_subdet[isimhit] for isimhit in simHitIdx ]

        sim_layer.append(tuple(layers))

    counter_info =  Counter(sim_layer)
    for trk_type in counter_info:
        print trk_type, counter_info[trk_type]

    sim_layer = list(set(sim_layer))
    sim_layer.sort()
    print sim_layer

    # print sim_layer

    # print list(set(sim_layer))

    ## Print summary of the event

    print "{:40s} {:10s} {:10s} {:10s} {:10s} {:10s}".format("Category", ">1 GeV", ">0 GeV", "Total", ">1 GeV %", ">0 GeV %")

    ph2_pt_array = np.array(ph2_pt)
    print "{:40s} {:10d} {:10d} {:10d} {:10f} {:10f}".format("Hits Total:", np.sum(ph2_pt_array > 1), np.sum(ph2_pt_array > 0), np.sum(ph2_pt_array != 0), float(np.sum(ph2_pt_array > 1)) / np.sum(ph2_pt_array != 0), float(np.sum(ph2_pt_array > 0)) / np.sum(ph2_pt_array != 0))
    for i in xrange(1, 12):
        ph2_pt_array = np.array(ph2_pt) * (np.array(ph2_layer) == i)
        print "{:40s} {:10d} {:10d} {:10d} {:10f} {:10f}".format("Hits L{}:".format(i), np.sum(ph2_pt_array > 1), np.sum(ph2_pt_array > 0), np.sum(ph2_pt_array != 0), float(np.sum(ph2_pt_array > 1)) / np.sum(ph2_pt_array != 0), float(np.sum(ph2_pt_array > 0)) / np.sum(ph2_pt_array != 0))
        # print "Hits L{}:      {} (>1 GeV) {} (>0 GeV) out of {}".format(i, np.sum(ph2_pt_array > 1), np.sum(ph2_pt_array > 0), np.sum(ph2_pt_array != 0))

    for isimtype in range(4):
        ph2_pt_array = np.array(ph2_pt) * (np.array(ph2_simtype) == isimtype)
        print "{:40s} {:10d} {:10d} {:10d} {:10f} {:10f}".format("Hits (st={}) Total:".format(isimtype), np.sum(ph2_pt_array > 1), np.sum(ph2_pt_array > 0), np.sum(ph2_pt_array != 0), float(np.sum(ph2_pt_array > 1)) / np.sum(ph2_pt_array != 0), float(np.sum(ph2_pt_array > 0)) / np.sum(ph2_pt_array != 0))
        for i in xrange(1, 12):
            ph2_pt_array = np.array(ph2_pt) * (np.array(ph2_layer) == i) * (np.array(ph2_simtype) == isimtype)
            print "{:40s} {:10d} {:10d} {:10d} {:10f} {:10f}".format("Hits (st={}) L{}:".format(isimtype, i), np.sum(ph2_pt_array > 1), np.sum(ph2_pt_array > 0), np.sum(ph2_pt_array != 0), float(np.sum(ph2_pt_array > 1)) / np.sum(ph2_pt_array != 0), float(np.sum(ph2_pt_array > 0)) / np.sum(ph2_pt_array != 0))
            # print "Hits L{}:      {} (>1 GeV) {} (>0 GeV) out of {}".format(i, np.sum(ph2_pt_array > 1), np.sum(ph2_pt_array > 0), np.sum(ph2_pt_array != 0))

    thr = 0.5
    ph2_pt_array = np.array(ph2_pt) * (np.array(ph2_anchor) == 1)
    print "{:40s} {:10d} {:10d} {:10d} {:10f} {:10f}".format("Hits Total Anchor Layer:", np.sum(ph2_pt_array > thr), np.sum(ph2_pt_array > 0), np.sum(ph2_pt_array != 0), float(np.sum(ph2_pt_array > thr)) / np.sum(ph2_pt_array != 0), float(np.sum(ph2_pt_array > 0)) / np.sum(ph2_pt_array != 0))
    for i in xrange(1, 12):
        ph2_pt_array = np.array(ph2_pt) * (np.array(ph2_anchor) == 1) * (np.array(ph2_layer) == i)
        print "{:40s} {:10d} {:10d} {:10d} {:10f} {:10f}".format("Hits Anchor Layer L{}:".format(i), np.sum(ph2_pt_array > thr), np.sum(ph2_pt_array > 0), np.sum(ph2_pt_array != 0), float(np.sum(ph2_pt_array > thr)) / np.sum(ph2_pt_array != 0), float(np.sum(ph2_pt_array > 0)) / np.sum(ph2_pt_array != 0))
        # print "Hits L{}:      {} (>1 GeV) {} (>0 GeV) out of {}".format(i, np.sum(ph2_pt_array > thr), np.sum(ph2_pt_array > 0), np.sum(ph2_pt_array != 0))

    ph2_pt_array = np.array(ph2_pt) * (np.array(ph2_anchor) == 1) * (np.array(ph2_hasMD) == 1)
    print "{:40s} {:10d} {:10d} {:10d} {:10f} {:10f}".format("Hits Total Anchor hasMD:", np.sum(ph2_pt_array > thr), np.sum(ph2_pt_array > 0), np.sum(ph2_pt_array != 0), float(np.sum(ph2_pt_array > thr)) / np.sum(ph2_pt_array != 0), float(np.sum(ph2_pt_array > 0)) / np.sum(ph2_pt_array != 0))
    for i in xrange(1, 12):
        ph2_pt_array = np.array(ph2_pt) * (np.array(ph2_anchor) == 1) * (np.array(ph2_hasMD) == 1) * (np.array(ph2_layer) == i)
        print "{:40s} {:10d} {:10d} {:10d} {:10f} {:10f}".format("Hits Anchor hasMD L{}:".format(i), np.sum(ph2_pt_array > thr), np.sum(ph2_pt_array > 0), np.sum(ph2_pt_array != 0), float(np.sum(ph2_pt_array > thr)) / np.sum(ph2_pt_array != 0), float(np.sum(ph2_pt_array > 0)) / np.sum(ph2_pt_array != 0))
        # print "Hits L{}:      {} (>1 GeV) {} (>0 GeV) out of {}".format(i, np.sum(ph2_pt_array > thr), np.sum(ph2_pt_array > 0), np.sum(ph2_pt_array != 0))

    md_pt_array = np.array(md_pt)
    print "{:40s} {:10d} {:10d} {:10d} {:10f} {:10f}".format("MiniDoublets Total:", np.sum(md_pt_array > 1), np.sum(md_pt_array > 0), np.sum(md_pt_array != 0), float(np.sum(md_pt_array > 1)) / np.sum(md_pt_array != 0), float(np.sum(md_pt_array > 0)) / np.sum(md_pt_array != 0))
    for i in xrange(1, 12):
        md_pt_array = np.array(md_pt) * (np.array(md_layer) == i)
        print "{:40s} {:10d} {:10d} {:10d} {:10f} {:10f}".format("MiniDoublets L{}:".format(i), np.sum(md_pt_array > 1), np.sum(md_pt_array > 0), np.sum(md_pt_array != 0), float(np.sum(md_pt_array > 1)) / np.sum(md_pt_array != 0), float(np.sum(md_pt_array > 0)) / np.sum(md_pt_array != 0))

    md_pt_array = np.array(md_pt) * (np.array(md_goodsimtypes) == 0)
    print "{:40s} {:10d} {:10d} {:10d} {:10f} {:10f}".format("MiniDoublets (no noise) Total:", np.sum(md_pt_array > 1), np.sum(md_pt_array > 0), np.sum(md_pt_array != 0), float(np.sum(md_pt_array > 1)) / np.sum(md_pt_array != 0), float(np.sum(md_pt_array > 0)) / np.sum(md_pt_array != 0))
    for i in xrange(1, 12):
        md_pt_array = np.array(md_pt) * (np.array(md_layer) == i) * (np.array(md_goodsimtypes) == 0)
        print "{:40s} {:10d} {:10d} {:10d} {:10f} {:10f}".format("MiniDoublets L{}:".format(i), np.sum(md_pt_array > 1), np.sum(md_pt_array > 0), np.sum(md_pt_array != 0), float(np.sum(md_pt_array > 1)) / np.sum(md_pt_array != 0), float(np.sum(md_pt_array > 0)) / np.sum(md_pt_array != 0))

    sg_pt_array = np.array(sg_pt)
    print "{:40s} {:10d} {:10d} {:10d} {:10f} {:10f}".format("Segments Total:", np.sum(sg_pt_array > 1), np.sum(sg_pt_array > 0), np.sum(sg_pt_array != 0), float(np.sum(sg_pt_array > 1)) / np.sum(sg_pt_array != 0), float(np.sum(sg_pt_array > 0)) / np.sum(sg_pt_array != 0))
    for i in xrange(1, 15):
        sg_pt_array = np.array(sg_pt) * (np.array(sg_layer) == i)
        print "{:40s} {:10d} {:10d} {:10d} {:10f} {:10f}".format("Segments L{}:".format(segment_layer_type[i-1]), np.sum(sg_pt_array > 1), np.sum(sg_pt_array > 0), np.sum(sg_pt_array != 0), float(np.sum(sg_pt_array > 1)) / np.sum(sg_pt_array != 0), float(np.sum(sg_pt_array > 0)) / np.sum(sg_pt_array != 0))

    sg_pt_array = np.array(sg_pt) * (np.array(sg_goodsimtypes) == 0)
    print "{:40s} {:10d} {:10d} {:10d} {:10f} {:10f}".format("Segments (no noise) Total:", np.sum(sg_pt_array > 1), np.sum(sg_pt_array > 0), np.sum(sg_pt_array != 0), float(np.sum(sg_pt_array > 1)) / np.sum(sg_pt_array != 0), float(np.sum(sg_pt_array > 0)) / np.sum(sg_pt_array != 0))
    for i in xrange(1, 15):
        sg_pt_array = np.array(sg_pt) * (np.array(sg_layer) == i) * (np.array(sg_goodsimtypes) == 0)
        print "{:40s} {:10d} {:10d} {:10d} {:10f} {:10f}".format("Segments L{}:".format(segment_layer_type[i-1]), np.sum(sg_pt_array > 1), np.sum(sg_pt_array > 0), np.sum(sg_pt_array != 0), float(np.sum(sg_pt_array > 1)) / np.sum(sg_pt_array != 0), float(np.sum(sg_pt_array > 0)) / np.sum(sg_pt_array != 0))

    tp_pt_array = np.array(tp_pt)
    print "{:40s} {:10d} {:10d} {:10d} {:10f} {:10f}".format("Triplet Total:", np.sum(tp_pt_array > 1), np.sum(tp_pt_array > 0), np.sum(tp_pt_array != 0), float(np.sum(tp_pt_array > 1)) / np.sum(tp_pt_array != 0), float(np.sum(tp_pt_array > 0)) / np.sum(tp_pt_array != 0))
    for layer_combo_product in triplet_layer_type_keys:
        tp_pt_array = np.array(tp_pt) * (np.array(tp_layer) == layer_combo_product)
        print "{:40s} {:10d} {:10d} {:10d} {:10f} {:10f}".format("Triplet L{}:".format(triplet_layer_type[layer_combo_product]), np.sum(tp_pt_array > 1), np.sum(tp_pt_array > 0), np.sum(tp_pt_array != 0), float(np.sum(tp_pt_array > 1)) / np.sum(tp_pt_array != 0), float(np.sum(tp_pt_array > 0)) / np.sum(tp_pt_array != 0))

    tp_pt_array = np.array(tp_pt) * (np.array(tp_goodsimtypes) == 0)
    print "{:40s} {:10d} {:10d} {:10d} {:10f} {:10f}".format("Triplet (no noise) Total:", np.sum(tp_pt_array > 1), np.sum(tp_pt_array > 0), np.sum(tp_pt_array != 0), float(np.sum(tp_pt_array > 1)) / np.sum(tp_pt_array != 0), float(np.sum(tp_pt_array > 0)) / np.sum(tp_pt_array != 0))
    for layer_combo_product in triplet_layer_type_keys:
        tp_pt_array = np.array(tp_pt) * (np.array(tp_layer) == layer_combo_product) * (np.array(tp_goodsimtypes) == 0)
        print "{:40s} {:10d} {:10d} {:10d} {:10f} {:10f}".format("Triplet L{}:".format(triplet_layer_type[layer_combo_product]), np.sum(tp_pt_array > 1), np.sum(tp_pt_array > 0), np.sum(tp_pt_array != 0), float(np.sum(tp_pt_array > 1)) / np.sum(tp_pt_array != 0), float(np.sum(tp_pt_array > 0)) / np.sum(tp_pt_array != 0))

    tl_pt_array = np.array(tl_pt)
    print "{:40s} {:10d} {:10d} {:10d} {:10f} {:10f}".format("Tracklet Total:", np.sum(tl_pt_array > 1), np.sum(tl_pt_array > 0), np.sum(tl_pt_array != 0), float(np.sum(tl_pt_array > 1)) / np.sum(tl_pt_array != 0), float(np.sum(tl_pt_array > 0)) / np.sum(tl_pt_array != 0))
    for layer_combo_product in tracklet_layer_type_keys:
        tl_pt_array = np.array(tl_pt) * (np.array(tl_layer) == layer_combo_product)
        print "{:40s} {:10d} {:10d} {:10d} {:10f} {:10f}".format("Tracklet L{}:".format(tracklet_layer_type[layer_combo_product]), np.sum(tl_pt_array > 1), np.sum(tl_pt_array > 0), np.sum(tl_pt_array != 0), float(np.sum(tl_pt_array > 1)) / np.sum(tl_pt_array != 0), float(np.sum(tl_pt_array > 0)) / np.sum(tl_pt_array != 0))

    tl_pt_array = np.array(tl_pt) * (np.array(tl_goodsimtypes) == 0)
    print "{:40s} {:10d} {:10d} {:10d} {:10f} {:10f}".format("Tracklet (no noise) Total:", np.sum(tl_pt_array > 1), np.sum(tl_pt_array > 0), np.sum(tl_pt_array != 0), float(np.sum(tl_pt_array > 1)) / np.sum(tl_pt_array != 0), float(np.sum(tl_pt_array > 0)) / np.sum(tl_pt_array != 0))
    for layer_combo_product in tracklet_layer_type_keys:
        tl_pt_array = np.array(tl_pt) * (np.array(tl_layer) == layer_combo_product) * (np.array(tl_goodsimtypes) == 0)
        print "{:40s} {:10d} {:10d} {:10d} {:10f} {:10f}".format("Tracklet L{}:".format(tracklet_layer_type[layer_combo_product]), np.sum(tl_pt_array > 1), np.sum(tl_pt_array > 0), np.sum(tl_pt_array != 0), float(np.sum(tl_pt_array > 1)) / np.sum(tl_pt_array != 0), float(np.sum(tl_pt_array > 0)) / np.sum(tl_pt_array != 0))

    sim_pt_array = np.array(sim_pt) * (np.abs(np.array(sim_eta)) < 2.4)
    print "{:40s} {:10d} {:10d} {:10d} {:10f} {:10f}".format("SimTrack Total:", np.sum(sim_pt_array > 1), np.sum(sim_pt_array > 0), np.sum(sim_pt_array != 0), float(np.sum(sim_pt_array > 1)) / np.sum(sim_pt_array != 0), float(np.sum(sim_pt_array > 0)) / np.sum(sim_pt_array != 0))
    # for layer_combo_product in tracklet_layer_type_keys:
    #     sim_pt_array = np.array(sim_pt) * (np.array(sim_layer) == layer_combo_product)
    #     print "{:40s} {:10d} {:10d} {:10d} {:10f} {:10f}".format("Tracklet L{}:".format(tracklet_layer_type[layer_combo_product]), np.sum(sim_pt_array > 1), np.sum(sim_pt_array > 0), np.sum(sim_pt_array != 0), float(np.sum(sim_pt_array > 1)) / np.sum(sim_pt_array != 0), float(np.sum(sim_pt_array > 0)) / np.sum(sim_pt_array != 0))

    break

    # if ievent > 0:
    #     break

# for i in xrange(len(list_md_pt)):

#     list_md_pt = np.array(list_md_pt[i])
#     print np.sum(list_md_pt > 1), np.sum(list_md_pt > 0), np.sum(list_md_pt != 0)

#     list_sg_pt = np.array(list_sg_pt[i])
#     print np.sum(list_sg_pt > 1), np.sum(list_sg_pt > 0), np.sum(list_sg_pt != 0)


#!/usr/bin/env python

import ROOT as r
import numpy as np
from Module import Module
from sdlmath import getCenterFromThreePoints
import math

# f = r.TFile("/home/users/phchang/public_html/analysis/sdl/TrackLooper_/results/write_sdl_ntuple/pt0p5_2p0_2020_0710//fulleff_pt0p5_2p0.root")
# f = r.TFile("/home/users/phchang/public_html/analysis/sdl/TrackLooper_/results/write_sdl_ntuple/pt0p5_2p0_2020_0723_starbucksBundang/fulleff_pt0p5_2p0.root")
# f = r.TFile("results/write_sdl_ntuple/pu200_w_truthinfo_charged_2020_0723_starbucksBundang/fulleff_pu200_w_truthinfo_charged_4.root")
# f = r.TFile("results/write_sdl_ntuple/pu200_w_truthinfo_charged_2020_0723_starbucksBundang/fulleff_pu200_w_truthinfo_charged.root")
# f = r.TFile("/home/users/phchang/public_html/analysis/sdl/TrackLooper_/results/write_sdl_ntuple/pu200_w_truthinfo_charged_2020_0723_starbucksBundang/fulleff_pu200_w_truthinfo_charged_0.root")
# f = r.TFile("debug.root")
f = r.TFile("/home/users/phchang/public_html/analysis/sdl/TrackLooper_/results/write_sdl_ntuple/pu200_w_truthinfo_charged_2020_0723_starbucksBundang/fulleff_pu200_w_truthinfo_charged.root")
tree = f.Get("tree")

list_ph2_pt = [] # list of pt of matched sim track with highest pt to the hits
list_ph2_eta = [] # list of pt of matched sim track with highest pt to the hits
list_ph2_match = [] # list of pt of matched sim track with highest pt to the hits

list_md_pt = [] # list of pt of matched sim track with highest pt to the minidoublets
list_md_eta = [] # list of pt of matched sim track with highest pt to the minidoublets
list_md_match = [] # list of pt of matched sim track with highest pt to the minidoublets
list_md_reco_pt = [] # list of pt of matched sim track with highest pt to the minidoublets

list_sg_pt = [] # list of pt of matched sim track with highest pt to the segments
list_sg_eta = [] # list of pt of matched sim track with highest pt to the segments
list_sg_match = [] # list of pt of matched sim track with highest pt to the segments
list_sg_reco_pt = [] # list of pt of matched sim track with highest pt to the segments

list_tl_pt = [] # list of pt of matched sim track with highest pt to the tracklets
list_tl_eta = [] # list of pt of matched sim track with highest pt to the tracklets
list_tl_match = [] # list of pt of matched sim track with highest pt to the tracklets
list_tl_reco_pt = [] # list of pt of matched sim track with highest pt to the tracklets

list_tc_pt = [] # list of pt of matched sim track with highest pt to the trackcandidates
list_tc_eta = [] # list of pt of matched sim track with highest pt to the trackcandidates
list_tc_match = [] # list of pt of matched sim track with highest pt to the trackcandidates
list_tc_reco_pt = [] # list of pt of matched sim track with highest pt to the trackcandidates

for ievent, event in enumerate(tree):

    # Count Hits of interest
    ph2_pt = []
    ph2_eta = []
    ph2_match = []
    for index, (ph2_simHitIdx, ph2_detId) in enumerate(zip(event.ph2_simHitIdx, event.ph2_detId)):
        if not (Module(ph2_detId).subdet() == 4 or Module(ph2_detId).subdet() == 5):
            continue
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
        else:
            ph2_pt.append(-999)
            ph2_eta.append(-999)
            ph2_match.append(0)
    list_ph2_pt.append(ph2_pt)
    list_ph2_eta.append(ph2_eta)
    list_ph2_match.append(ph2_match)

    # Count Mini-Doublet of interest
    md_pt = []
    md_eta = []
    md_match = []
    md_reco_pt = []
    for index, (md_simTrkIdx, md_hitIdx) in enumerate(zip(event.md_simTrkIdx, event.md_hitIdx)):
        center = getCenterFromThreePoints([0, 0], [event.ph2_x[md_hitIdx[0]], event.ph2_y[md_hitIdx[0]]], [event.ph2_x[md_hitIdx[1]], event.ph2_y[md_hitIdx[1]]])
        radius = math.sqrt(center[0]**2 + center[1]**2)
        reco_pt = 2.99792458e-3 * 3.8 * radius;
        md_reco_pt.append(reco_pt)
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
        else:
            md_pt.append(-999)
            md_eta.append(-999)
            md_match.append(0)
    list_md_pt.append(md_pt)
    list_md_eta.append(md_eta)
    list_md_match.append(md_match)
    list_md_reco_pt.append(md_reco_pt)

    # Count Segment of interest
    sg_pt = []
    sg_eta = []
    sg_match = []
    sg_reco_pt = []
    for index, (sg_simTrkIdx, sg_hitIdx) in enumerate(zip(event.sg_simTrkIdx, event.sg_hitIdx)):
        center = getCenterFromThreePoints([0, 0], [event.ph2_x[sg_hitIdx[0]], event.ph2_y[sg_hitIdx[0]]], [event.ph2_x[sg_hitIdx[3]], event.ph2_y[sg_hitIdx[3]]])
        radius = math.sqrt(center[0]**2 + center[1]**2)
        reco_pt = 2.99792458e-3 * 3.8 * radius;
        sg_reco_pt.append(reco_pt)
        if len(sg_simTrkIdx) > 0:
            maxpt = 0
            maxeta = 0
            for isimtrk in sg_simTrkIdx:
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
    list_sg_pt.append(sg_pt)
    list_sg_eta.append(sg_eta)
    list_sg_match.append(sg_match)
    list_sg_reco_pt.append(sg_reco_pt)

    # Count Tracklet of interest
    tl_pt = []
    tl_eta = []
    tl_match = []
    tl_reco_pt = []
    for index, (tl_simTrkIdx, tl_hitIdx) in enumerate(zip(event.tl_simTrkIdx, event.tl_hitIdx)):
        center = getCenterFromThreePoints([0, 0], [event.ph2_x[tl_hitIdx[0]], event.ph2_y[tl_hitIdx[0]]], [event.ph2_x[tl_hitIdx[7]], event.ph2_y[tl_hitIdx[7]]])
        radius = math.sqrt(center[0]**2 + center[1]**2)
        reco_pt = 2.99792458e-3 * 3.8 * radius;
        tl_reco_pt.append(reco_pt)
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
    list_tl_pt.append(tl_pt)
    list_tl_eta.append(tl_eta)
    list_tl_match.append(tl_match)
    list_tl_reco_pt.append(tl_reco_pt)

    # Count TrackCandidate of interest
    tc_pt = []
    tc_eta = []
    tc_match = []
    tc_reco_pt = []
    for index, (tc_simTrkIdx, tc_hitIdx) in enumerate(zip(event.tc_simTrkIdx, event.tc_hitIdx)):
        center = getCenterFromThreePoints([0, 0], [event.ph2_x[tc_hitIdx[0]], event.ph2_y[tc_hitIdx[0]]], [event.ph2_x[tc_hitIdx[11]], event.ph2_y[tc_hitIdx[11]]])
        radius = math.sqrt(center[0]**2 + center[1]**2)
        reco_pt = 2.99792458e-3 * 3.8 * radius;
        tc_reco_pt.append(reco_pt)
        if len(tc_simTrkIdx) > 0:
            maxpt = 0
            maxeta = 0
            for isimtrk in tc_simTrkIdx:
                tmppt = event.sim_pt[isimtrk]
                tmpeta = event.sim_eta[isimtrk]
                if tmppt > maxpt:
                    maxpt = tmppt
                    maxeta = tmpeta
                break
            tc_pt.append(maxpt)
            tc_eta.append(maxeta)
            tc_match.append(1)
        else:
            tc_pt.append(-999)
            tc_eta.append(-999)
            tc_match.append(0)
    list_tc_pt.append(tc_pt)
    list_tc_eta.append(tc_eta)
    list_tc_match.append(tc_match)
    list_tc_reco_pt.append(tc_reco_pt)

    if ievent > 0:
        break

for i in xrange(len(list_ph2_pt)):

    # list_ph2_pt = np.array(list_ph2_pt)
    # print np.sum(list_ph2_pt > 1), np.sum(list_ph2_pt > 0), np.sum(list_ph2_pt != 0)

    list_md_pt = np.array(list_md_pt[i])
    print np.sum(list_md_pt > 1), np.sum(list_md_pt > 0), np.sum(list_md_pt != 0)

    # list_md_reco_pt = np.array(list_md_reco_pt)
    # list_md_reco_pt *= (list_md_pt > 0)
    # print np.sum(list_md_reco_pt > 1), np.sum(list_md_reco_pt > 0), np.sum(list_md_reco_pt != 0)

    list_sg_pt = np.array(list_sg_pt[i])
    print np.sum(list_sg_pt > 1), np.sum(list_sg_pt > 0), np.sum(list_sg_pt != 0)

    # list_sg_reco_pt = np.array(list_sg_reco_pt)
    # list_sg_reco_pt *= (list_sg_pt > 0)
    # print np.sum(list_sg_reco_pt > 1), np.sum(list_sg_reco_pt > 0), np.sum(list_sg_reco_pt != 0)

    # list_tl_pt = np.array(list_tl_pt)
    # print np.sum(list_tl_pt > 1), np.sum(list_tl_pt > 0), np.sum(list_tl_pt != 0)

    # list_tl_reco_pt = np.array(list_tl_reco_pt)
    # list_tl_reco_pt *= (list_tl_pt > 0)
    # print np.sum(list_tl_reco_pt > 1), np.sum(list_tl_reco_pt > 0), np.sum(list_tl_reco_pt != 0)

    # list_tc_pt = np.array(list_tc_pt)
    # print np.sum(list_tc_pt > 1), np.sum(list_tc_pt > 0), np.sum(list_tc_pt != 0)

    # list_tc_reco_pt = np.array(list_tc_reco_pt)
    # list_tc_reco_pt *= (list_tc_pt > 0)
    # print np.sum(list_tc_reco_pt > 1), np.sum(list_tc_reco_pt > 0), np.sum(list_tc_reco_pt != 0)

    # # Count MiniDoublet of interest
    # md_match = 0
    # md_nomatch = 0
    # for index, (md_hitIdx, md_simTrkIdx, simhit_simTrkIdx) in enumerate(zip(event.md_hitIdx, event.md_simTrkIdx, event.simhit_simTrkIdx)):
    #     if len(md_simTrkIdx) > 0:
    #         if event.sim_pt[md_simTrkIdx[0]] > 1:
    #             md_match += 1
    #         else:
    #             md_nomatch += 1
    #     else:
    #         md_nomatch += 1
    # list_md_match.append(md_match)
    # list_md_nomatch.append(md_nomatch)

    # sg_match = 0
    # sg_nomatch = 0
    # for index, (sg_hitIdx, sg_simTrkIdx, simhit_simTrkIdx) in enumerate(zip(event.sg_hitIdx, event.sg_simTrkIdx, event.simhit_simTrkIdx)):
    #     if len(sg_simTrkIdx) > 0:
    #         if event.sim_pt[sg_simTrkIdx[0]] > 1:
    #             sg_match += 1
    #         else:
    #             sg_nomatch += 1
    #     else:
    #         sg_nomatch += 1

    # print sg_match, sg_nomatch, len(event.sg_hitIdx)

    # tl_match = 0
    # tl_nomatch = 0
    # for index, (tl_hitIdx, tl_simTrkIdx, simhit_simTrkIdx) in enumerate(zip(event.tl_hitIdx, event.tl_simTrkIdx, event.simhit_simTrkIdx)):
    #     if len(tl_simTrkIdx) > 0:
    #         tl_match += 1
    #     else:
    #         tl_nomatch += 1

    # print tl_match, tl_nomatch, len(event.tl_hitIdx)

    # # for index, (md_hitIdx, md_simTrkIdx, simhit_simTrkIdx) in enumerate(zip(event.md_hitIdx, event.md_simTrkIdx, event.simhit_simTrkIdx)):
    # #     if len(md_simTrkIdx) == 0:
    # #         print len(md_hitIdx)
    # #         print md_hitIdx
    # #         print md_hitIdx[0]
    # #         print md_hitIdx[1]
    # #         print event.ph2_x[md_hitIdx[0]], event.ph2_y[md_hitIdx[0]], event.ph2_z[md_hitIdx[0]]
    # #         print event.ph2_x[md_hitIdx[1]], event.ph2_y[md_hitIdx[1]], event.ph2_z[md_hitIdx[1]]
    # #         print event.ph2_simHitIdx[md_hitIdx[0]]
    # #         print event.ph2_simHitIdx[md_hitIdx[1]]
    # #         if len(event.ph2_simHitIdx[md_hitIdx[0]]) > 0:
    # #             print "here", event.simhit_simTrkIdx[event.ph2_simHitIdx[md_hitIdx[0]][0]]
    # #         if len(event.ph2_simHitIdx[md_hitIdx[1]]) > 0:
    # #             print "here", event.simhit_simTrkIdx[event.ph2_simHitIdx[md_hitIdx[1]][0]]
    # #         print md_simTrkIdx


#!/bin/env python

# Tracking Ntuple single track visualization
# Philip Chang

import ROOT as r
r.gROOT.SetBatch(True)

r.gROOT.ProcessLine(".L SDL/sdl.so")

r.gROOT.ProcessLine(".L SDL/Algo.h")
r.gROOT.ProcessLine(".L SDL/Constants.h")
r.gROOT.ProcessLine(".L SDL/EndcapGeometry.h")
r.gROOT.ProcessLine(".L SDL/Event.h")
r.gROOT.ProcessLine(".L SDL/GeometryUtil.h")
r.gROOT.ProcessLine(".L SDL/Hit.h")
r.gROOT.ProcessLine(".L SDL/Layer.h")
r.gROOT.ProcessLine(".L SDL/MathUtil.h")
r.gROOT.ProcessLine(".L SDL/MiniDoublet.h")
r.gROOT.ProcessLine(".L SDL/Module.h")
r.gROOT.ProcessLine(".L SDL/ModuleConnectionMap.h")
r.gROOT.ProcessLine(".L SDL/PrintUtil.h")
r.gROOT.ProcessLine(".L SDL/Segment.h")
r.gROOT.ProcessLine(".L SDL/TiltedGeometry.h")
r.gROOT.ProcessLine(".L SDL/TrackCandidate.h")
r.gROOT.ProcessLine(".L SDL/Tracklet.h")
r.gROOT.ProcessLine(".L SDL/TrackletBase.h")
r.gROOT.ProcessLine(".L SDL/Triplet.h")

print(r.SDL.Hit())

import sys
import os
import math

import trkcore

def usage():

    print "Usage:"
    print ""
    print "   python {} TRACKINGNTUPLE EVENTINDEX SIMTRKINDEX".format(sys.argv[0])
    print ""
    print "     if SIMTRKINDEX < 0, then the user will be asked to input the simtrkidx"
    print ""
    print ""
    sys.exit()

try:
    tkntuple = sys.argv[1]
    evt_idx_to_plot = int(sys.argv[2])
except:
    usage()

# Open the input data
t = r.TChain("trackingNtuple/tree")
t.Add(tkntuple)

# Now select the track and aggregate simhit information
# simhit_selected = []
# clean_ph2hit_selected = []
# dirty_ph2hit_selected = []
# pt_of_track = 0
# eta_of_track = 0

ph2_selected = []
ph2_selected_by_layer = {
        1: [],
        2: [],
        3: [],
        4: [],
        5: [],
        6: [],
        }

sim_denoms = []
good_ph2_selected = []
good_ph2_selected_by_layer = {
        1: [],
        2: [],
        3: [],
        4: [],
        5: [],
        6: [],
        }

sim_denoms_have_all_hits = []
good_ph2_selected_have_all_hits = []
good_ph2_selected_by_layer_have_all_hits = {
        1: [],
        2: [],
        3: [],
        4: [],
        5: [],
        6: [],
        }

ph2_special_module = []

sdlEvent = r.SDL.Event()
r.SDL.endcapGeometry.load("/home/users/phchang/public_html/analysis/sdl/TrackLooper_/scripts/endcap_orientation_data_v2.txt");
r.SDL.tiltedGeometry.load("/home/users/phchang/public_html/analysis/sdl/TrackLooper_/scripts/tilted_orientation_data.txt");
r.SDL.moduleConnectionMap.load("/home/users/phchang/public_html/analysis/sdl/TrackLooper_/scripts/module_connection_map_data_10_e0_200_100_pt0p8_2p0_400_pt0p8_2p0_nolossers_dxy35cm_endcaplayer2.txt");

hit_occupancy = r.TH1F("hit_occupancy", "", 100, 0, 100)
md_occupancy = r.TH1F("md_occupancy", "", 100, 0, 100)
hit_o_md_occupancy = r.TH1F("hit_o_md_occupancy", "", 100, 0, 3)
hit_v_md_occupancy = r.TH2F("hit_v_md_occupancy", "", 40, 0, 40, 40, 0, 40)

def add_all_reco_hits():

    # Loop over reco hits
    for index, (x, y, z, subdet, layer, detid) in enumerate(zip(t.ph2_x, t.ph2_y, t.ph2_z, t.ph2_subdet, t.ph2_layer, t.ph2_detId)):

        # Then it is in outer tracker
        if subdet == 5:

            ph2_selected.append((x, y, z, layer, index))
            ph2_selected_by_layer[layer].append((x, y, z, index))

            # if r.SDL.Module(t.ph2_detId[index]).isLower():

            sdlEvent.addHitToModule(r.SDL.Hit(x, y, z, index), t.ph2_detId[index])

            # if x > 80 and x < 87 and y > -5 and y < 5:
            # if layer == 5:
                # print(detid)
                # sdlEvent.addHitToModule(r.SDL.Hit(x, y, z, index), t.ph2_detId[index])

def add_denom_reco_hits():

    for index, (pt, eta, phi, q, dz, dxy, pdgid, bx) in enumerate(zip(t.sim_pt, t.sim_eta, t.sim_phi, t.sim_q, t.sim_pca_dz, t.sim_pca_dxy, t.sim_pdgId, t.sim_bunchCrossing)):

        if bx != 0:
            continue

        if pt > 1.0 and abs(eta) < 0.8:

            # sim_denoms.append((pt, eta, phi, q, dz, dxy, pdgid, index))

            hit_layers = []

            for simhitidx in t.sim_simHitIdx[index]:

                if t.simhit_subdet[simhitidx] == 5 and t.simhit_particle[simhitidx] == t.sim_pdgId[index]:

                    for hitidx in t.simhit_hitIdx[simhitidx]:

                        if t.ph2_subdet[hitidx] == 5:

                            hit_layers.append(t.ph2_layer[hitidx])

                            # good_ph2_selected.append((t.ph2_x[hitidx], t.ph2_y[hitidx], t.ph2_z[hitidx], t.ph2_layer[hitidx], hitidx))
                            # good_ph2_selected_by_layer[t.ph2_layer[hitidx]].append((t.ph2_x[hitidx], t.ph2_y[hitidx], t.ph2_z[hitidx], hitidx))

            if len(list(set(hit_layers))) == 6:

                # sim_denoms_have_all_hits.append((pt, eta, phi, q, dz, dxy, pdgid, index))

                for simhitidx in t.sim_simHitIdx[index]:

                    if t.simhit_subdet[simhitidx] == 5 and t.simhit_particle[simhitidx] == t.sim_pdgId[index]:

                        for hitidx in t.simhit_hitIdx[simhitidx]:

                            if t.ph2_subdet[hitidx] == 5:

                                # good_ph2_selected_have_all_hits.append((t.ph2_x[hitidx], t.ph2_y[hitidx], t.ph2_z[hitidx], t.ph2_layer[hitidx], hitidx))
                                # good_ph2_selected_by_layer_have_all_hits[t.ph2_layer[hitidx]].append((t.ph2_x[hitidx], t.ph2_y[hitidx], t.ph2_z[hitidx], hitidx))

                                sdlEvent.addHitToModule(r.SDL.Hit(t.ph2_x[hitidx], t.ph2_y[hitidx], t.ph2_z[hitidx], hitidx), t.ph2_detId[hitidx])

f = r.TFile("output.root", "recreate")

all_sim_in_bx = []
all_sim_in_abseta25 = []
all_sim_in_abseta08 = []
all_sim_in_recoable = []
all_sim_in_0misshit = []

all_sim_pt = r.TH1F("all_sim_pt", "all_sim_pt", 40, 0, 4)
all_sim_in_abseta25_pt = r.TH1F("all_sim_in_abseta25_pt", "all_sim_in_abseta25_pt", 40, 0, 4)
all_sim_in_abseta08_pt = r.TH1F("all_sim_in_abseta08_pt", "all_sim_in_abseta08_pt", 40, 0, 4)
all_sim_in_recoable_pt = r.TH1F("all_sim_in_recoable_pt", "all_sim_in_recoable_pt", 40, 0, 4)
all_sim_in_0misshit_pt = r.TH1F("all_sim_in_0misshit_pt", "all_sim_in_0misshit_pt", 40, 0, 4)

def count_denom():
    for index, (pt, eta, phi, q, dz, dxy, pdgid, bx) in enumerate(zip(t.sim_pt, t.sim_eta, t.sim_phi, t.sim_q, t.sim_pca_dz, t.sim_pca_dxy, t.sim_pdgId, t.sim_bunchCrossing)):
        all_sim_in_bx.append((pt, eta, phi, q, dz, dxy, pdgid, index))
        if abs(eta) < 2.5:
            all_sim_in_abseta25.append((pt, eta, phi, q, dz, dxy, pdgid, index))
            if abs(eta) < 0.8:
                all_sim_in_abseta08.append((pt, eta, phi, q, dz, dxy, pdgid, index))

                simhit_layers = []
                for simhitidx in t.sim_simHitIdx[index]:
                    if t.simhit_subdet[simhitidx] == 5 and t.simhit_particle[simhitidx] == t.sim_pdgId[index]:
                        simhit_layers.append(t.simhit_layer[simhitidx])
                if len(list(set(simhit_layers))) == 6:
                    all_sim_in_recoable.append((pt, eta, phi, q, dz, dxy, pdgid, index))
                    hit_layers = []
                    for simhitidx in t.sim_simHitIdx[index]:
                        for hitidx in t.simhit_hitIdx[simhitidx]:
                            if t.ph2_subdet[hitidx] == 5:
                                hit_layers.append(t.ph2_layer[hitidx])
                    if len(list(set(hit_layers))) == 6:
                        all_sim_in_0misshit.append((pt, eta, phi, q, dz, dxy, pdgid, index))
    for trk in all_sim_in_bx: all_sim_pt.Fill(trk[0])
    for trk in all_sim_in_abseta25: all_sim_in_abseta25_pt.Fill(trk[0])
    for trk in all_sim_in_abseta08: all_sim_in_abseta08_pt.Fill(trk[0])
    for trk in all_sim_in_recoable: all_sim_in_recoable_pt.Fill(trk[0])
    for trk in all_sim_in_0misshit: all_sim_in_0misshit_pt.Fill(trk[0])

    all_sim_pt.Write()
    all_sim_in_abseta25_pt.Write()
    all_sim_in_abseta08_pt.Write()
    all_sim_in_recoable_pt.Write()
    all_sim_in_0misshit_pt.Write()

hit_data = open("hit_data.txt", "w")
md_data = open("md_data.txt", "w")
sg_data = open("sg_data.txt", "w")
tl_data = open("tl_data.txt", "w")

# Loop over to get the simhit, and ph2hit idxs
for index, event in enumerate(t):

    # If the event is the one you want
    # I don't know how to access the specific event directly in PyROOT efficiently
    # Let me know if you know
    if index == evt_idx_to_plot:

        # # Loop over reco hits
        # for index, (x, y, z, subdet, layer, detid) in enumerate(zip(t.ph2_x, t.ph2_y, t.ph2_z, t.ph2_subdet, t.ph2_layer, t.ph2_detId)):

        #     # Then it is in outer tracker
        #     if subdet == 5:

        #         ph2_selected.append((x, y, z, layer, index))
        #         ph2_selected_by_layer[layer].append((x, y, z, index))

        #         if r.SDL.Module(t.ph2_detId[index]).isLower():

        #             sdlEvent.addHitToModule(r.SDL.Hit(x, y, z, index), t.ph2_detId[index])

        #         # if x > 80 and x < 87 and y > -5 and y < 5:
        #         # if layer == 5:
        #             # print(detid)
        #             # sdlEvent.addHitToModule(r.SDL.Hit(x, y, z, index), t.ph2_detId[index])

        # count_denom()

        # sys.exit()

        add_all_reco_hits()
        # add_denom_reco_hits()

        # for module in sdlEvent.getLowerModulePtrs():
        #     sdlEvent.addHitToModule(r.SDL.Hit(0, 0, 0, 0), module.partnerDetId())

        print("# of Hits : {}" .format( sdlEvent.getNumberOfHits()))
        print("# of Hits in layer 1: {}" .format( sdlEvent.getNumberOfHitsByLayerBarrel(0)))
        print("# of Hits in layer 2: {}" .format( sdlEvent.getNumberOfHitsByLayerBarrel(1)))
        print("# of Hits in layer 3: {}" .format( sdlEvent.getNumberOfHitsByLayerBarrel(2)))
        print("# of Hits in layer 4: {}" .format( sdlEvent.getNumberOfHitsByLayerBarrel(3)))
        print("# of Hits in layer 5: {}" .format( sdlEvent.getNumberOfHitsByLayerBarrel(4)))
        print("# of Hits in layer 6: {}" .format( sdlEvent.getNumberOfHitsByLayerBarrel(5)))
        print("# of Hits Upper Module in layer 1: {}" .format( sdlEvent.getNumberOfHitsByLayerBarrelUpperModule(0)))
        print("# of Hits Upper Module in layer 2: {}" .format( sdlEvent.getNumberOfHitsByLayerBarrelUpperModule(1)))
        print("# of Hits Upper Module in layer 3: {}" .format( sdlEvent.getNumberOfHitsByLayerBarrelUpperModule(2)))
        print("# of Hits Upper Module in layer 4: {}" .format( sdlEvent.getNumberOfHitsByLayerBarrelUpperModule(3)))
        print("# of Hits Upper Module in layer 5: {}" .format( sdlEvent.getNumberOfHitsByLayerBarrelUpperModule(4)))
        print("# of Hits Upper Module in layer 6: {}" .format( sdlEvent.getNumberOfHitsByLayerBarrelUpperModule(5)))

        # Run minidoublet
        # sdlEvent.createMiniDoublets(r.SDL.AllComb_MDAlgo);
        sdlEvent.createMiniDoublets();
        print("# of Mini-doublets produced: {}" .format( sdlEvent.getNumberOfMiniDoublets()))
        print("# of Mini-doublets produced layer 1: {}" .format( sdlEvent.getNumberOfMiniDoubletsByLayerBarrel(0)))
        print("# of Mini-doublets produced layer 2: {}" .format( sdlEvent.getNumberOfMiniDoubletsByLayerBarrel(1)))
        print("# of Mini-doublets produced layer 3: {}" .format( sdlEvent.getNumberOfMiniDoubletsByLayerBarrel(2)))
        print("# of Mini-doublets produced layer 4: {}" .format( sdlEvent.getNumberOfMiniDoubletsByLayerBarrel(3)))
        print("# of Mini-doublets produced layer 5: {}" .format( sdlEvent.getNumberOfMiniDoubletsByLayerBarrel(4)))
        print("# of Mini-doublets produced layer 6: {}" .format( sdlEvent.getNumberOfMiniDoubletsByLayerBarrel(5)))
        print("# of Mini-doublets considered: {}" .format( sdlEvent.getNumberOfMiniDoubletCandidates()))
        print("# of Mini-doublets considered layer 1: {}" .format( sdlEvent.getNumberOfMiniDoubletCandidatesByLayerBarrel(0)))
        print("# of Mini-doublets considered layer 2: {}" .format( sdlEvent.getNumberOfMiniDoubletCandidatesByLayerBarrel(1)))
        print("# of Mini-doublets considered layer 3: {}" .format( sdlEvent.getNumberOfMiniDoubletCandidatesByLayerBarrel(2)))
        print("# of Mini-doublets considered layer 4: {}" .format( sdlEvent.getNumberOfMiniDoubletCandidatesByLayerBarrel(3)))
        print("# of Mini-doublets considered layer 5: {}" .format( sdlEvent.getNumberOfMiniDoubletCandidatesByLayerBarrel(4)))
        print("# of Mini-doublets considered layer 6: {}" .format( sdlEvent.getNumberOfMiniDoubletCandidatesByLayerBarrel(5)))

        for module in sdlEvent.getLowerModulePtrs():
            # print("{} -> {}".format(len(module.getHitPtrs()), len(module.getMiniDoubletPtrs())))
            hit_v_md_occupancy.Fill(len(module.getHitPtrs()), len(module.getMiniDoubletPtrs()))
            hit_o_md_occupancy.Fill(float(len(module.getMiniDoubletPtrs())) / float(len(module.getHitPtrs())))

        c2 = r.TCanvas("","",2000,2000)
        # hit_o_md_occupancy.Draw("hist")
        # c2.SaveAs("test.pdf")
        # sys.exit()

        # # Drawing baseline background
        # t.Draw("ph2_x:ph2_y","(ph2_subdet==5)&&(Entry$<10)".format(evt_idx_to_plot))
        # print("Done plotting base")

        # t.SetMarkerStyle(20)
        # t.SetMarkerSize(0.5)
        # # t.SetMarkerColorAlpha(2, 0.5)
        # t.SetMarkerColor(2)
        # i = 0
        # for module in sdlEvent.getModulePtrs():
        #     # if module.layer() != 3: continue
        #     for hit in module.getHitPtrs():
        #         t.Draw("{}:{}".format(hit.x(), hit.y()),"Entry$==0", "same")
        #         if i % 100 == 0:
        #             print(i)
        #         i += 1
        # print("Done plotting all hits")
        # c2.SaveAs("test1.png")

        # t.SetMarkerColor(1)
        # # t.Draw("ph2_x:ph2_y","(ph2_subdet==5)&&(Entry$=={})&&(ph2_x>0)&&(ph2_y>0)".format(evt_idx_to_plot))
        # t.Draw("ph2_x:ph2_y","(ph2_subdet==5)&&(Entry$<1)".format(evt_idx_to_plot))
        # t.SetMarkerStyle(20)
        # t.SetMarkerSize(0.5)
        # # t.SetMarkerColorAlpha(2, 0.5)
        # t.SetMarkerColor(2)
        # i = 0
        # for module in sdlEvent.getLowerModulePtrs():
        #     # if module.layer() != 3: continue
        #     for md in module.getMiniDoubletPtrs():
        #         t.Draw("{}:{}".format(md.lowerHitPtr().x(), md.lowerHitPtr().y()),"Entry$==0", "same")
        #         if i % 100 == 0:
        #             print(i)
        #         i += 1
        # print("Done plotting all mds")

        # c2.SaveAs("test2.png")

        # sys.exit()

        sdlEvent.createSegmentsWithModuleMap();
        # sdlEvent.createSegmentsWithModuleMap();
        print("# of Segments produced: {}" .format( sdlEvent.getNumberOfSegments()))
        print("# of Segments produced layer 1: {}" .format( sdlEvent.getNumberOfSegmentsByLayerBarrel(0)))
        print("# of Segments produced layer 2: {}" .format( sdlEvent.getNumberOfSegmentsByLayerBarrel(1)))
        print("# of Segments produced layer 3: {}" .format( sdlEvent.getNumberOfSegmentsByLayerBarrel(2)))
        print("# of Segments produced layer 4: {}" .format( sdlEvent.getNumberOfSegmentsByLayerBarrel(3)))
        print("# of Segments produced layer 5: {}" .format( sdlEvent.getNumberOfSegmentsByLayerBarrel(4)))
        print("# of Segments produced layer 6: {}" .format( sdlEvent.getNumberOfSegmentsByLayerBarrel(5)))
        print("# of Segments considered: {}" .format( sdlEvent.getNumberOfSegmentCandidates()))
        print("# of Segments considered layer 1: {}" .format( sdlEvent.getNumberOfSegmentCandidatesByLayerBarrel(0)))
        print("# of Segments considered layer 2: {}" .format( sdlEvent.getNumberOfSegmentCandidatesByLayerBarrel(1)))
        print("# of Segments considered layer 3: {}" .format( sdlEvent.getNumberOfSegmentCandidatesByLayerBarrel(2)))
        print("# of Segments considered layer 4: {}" .format( sdlEvent.getNumberOfSegmentCandidatesByLayerBarrel(3)))
        print("# of Segments considered layer 5: {}" .format( sdlEvent.getNumberOfSegmentCandidatesByLayerBarrel(4)))
        print("# of Segments considered layer 6: {}" .format( sdlEvent.getNumberOfSegmentCandidatesByLayerBarrel(5)))

        for module in sdlEvent.getModulePtrs():
            for hit in module.getHitPtrs():
                hit_x = hit.x()
                hit_y = hit.y()
                hit_z = hit.z()
                hit_data.write("{} {} {} {}\n".format(hit_x, hit_y, hit_z, module.layer()))

        for module in sdlEvent.getLowerModulePtrs():
            for md in module.getMiniDoubletPtrs():
                md_x = md.anchorHitPtr().x()
                md_y = md.anchorHitPtr().y()
                md_z = md.anchorHitPtr().z()
                md_data.write("{} {} {} {}\n".format(md_x, md_y, md_z, module.layer()))

        for module in sdlEvent.getLowerModulePtrs():
            for sg in module.getSegmentPtrs():
                iMD_x = sg.innerMiniDoubletPtr().anchorHitPtr().x()
                iMD_y = sg.innerMiniDoubletPtr().anchorHitPtr().y()
                iMD_z = sg.innerMiniDoubletPtr().anchorHitPtr().z()
                oMD_x = sg.outerMiniDoubletPtr().anchorHitPtr().x()
                oMD_y = sg.outerMiniDoubletPtr().anchorHitPtr().y()
                oMD_z = sg.outerMiniDoubletPtr().anchorHitPtr().z()

                sg_data.write("{} {} {} {} {} {} {} {}\n".format(iMD_x, iMD_y, iMD_z, oMD_x, oMD_y, oMD_z, module.layer(), trkcore.isTrueSegment(sg, t)))

        # sdlEvent.createTrackletsWithModuleMap();
        sdlEvent.createTrackletsViaNavigation();
        print("# of Tracklets produced: {}" .format( sdlEvent.getNumberOfTracklets()))
        print("# of Tracklets produced layer 1: {}" .format( sdlEvent.getNumberOfTrackletsByLayerBarrel(0)))
        print("# of Tracklets produced layer 2: {}" .format( sdlEvent.getNumberOfTrackletsByLayerBarrel(1)))
        print("# of Tracklets produced layer 3: {}" .format( sdlEvent.getNumberOfTrackletsByLayerBarrel(2)))
        print("# of Tracklets produced layer 4: {}" .format( sdlEvent.getNumberOfTrackletsByLayerBarrel(3)))
        print("# of Tracklets produced layer 5: {}" .format( sdlEvent.getNumberOfTrackletsByLayerBarrel(4)))
        print("# of Tracklets produced layer 6: {}" .format( sdlEvent.getNumberOfTrackletsByLayerBarrel(5)))
        print("# of Tracklets considered: {}" .format( sdlEvent.getNumberOfTrackletCandidates()))
        print("# of Tracklets considered layer 1: {}" .format( sdlEvent.getNumberOfTrackletCandidatesByLayerBarrel(0)))
        print("# of Tracklets considered layer 2: {}" .format( sdlEvent.getNumberOfTrackletCandidatesByLayerBarrel(1)))
        print("# of Tracklets considered layer 3: {}" .format( sdlEvent.getNumberOfTrackletCandidatesByLayerBarrel(2)))
        print("# of Tracklets considered layer 4: {}" .format( sdlEvent.getNumberOfTrackletCandidatesByLayerBarrel(3)))
        print("# of Tracklets considered layer 5: {}" .format( sdlEvent.getNumberOfTrackletCandidatesByLayerBarrel(4)))
        print("# of Tracklets considered layer 6: {}" .format( sdlEvent.getNumberOfTrackletCandidatesByLayerBarrel(5)))

        for module in sdlEvent.getLowerModulePtrs():
            for tl in module.getTrackletPtrs():
                iSG_iMD_x = tl.innerSegmentPtr().innerMiniDoubletPtr().anchorHitPtr().x()
                iSG_iMD_y = tl.innerSegmentPtr().innerMiniDoubletPtr().anchorHitPtr().y()
                iSG_iMD_z = tl.innerSegmentPtr().innerMiniDoubletPtr().anchorHitPtr().z()
                iSG_oMD_x = tl.innerSegmentPtr().outerMiniDoubletPtr().anchorHitPtr().x()
                iSG_oMD_y = tl.innerSegmentPtr().outerMiniDoubletPtr().anchorHitPtr().y()
                iSG_oMD_z = tl.innerSegmentPtr().outerMiniDoubletPtr().anchorHitPtr().z()
                oSG_iMD_x = tl.outerSegmentPtr().innerMiniDoubletPtr().anchorHitPtr().x()
                oSG_iMD_y = tl.outerSegmentPtr().innerMiniDoubletPtr().anchorHitPtr().y()
                oSG_iMD_z = tl.outerSegmentPtr().innerMiniDoubletPtr().anchorHitPtr().z()
                oSG_oMD_x = tl.outerSegmentPtr().outerMiniDoubletPtr().anchorHitPtr().x()
                oSG_oMD_y = tl.outerSegmentPtr().outerMiniDoubletPtr().anchorHitPtr().y()
                oSG_oMD_z = tl.outerSegmentPtr().outerMiniDoubletPtr().anchorHitPtr().z()
                tl_data.write("{} {} {} {} {} {} {} {} {} {} {} {} {}\n".format(iSG_iMD_x, iSG_iMD_y, iSG_iMD_z, iSG_oMD_x, iSG_oMD_y, iSG_oMD_z, oSG_iMD_x, oSG_iMD_y, oSG_iMD_z, oSG_oMD_x, oSG_oMD_y, oSG_oMD_z, module.layer()))

        # sdlEvent.createTrackletsWithModuleMap(r.SDL.AllComb_SGAlgo);
        # sdlEvent.createTrackCandidatesFromTracklets();
        print("# of TrackCandidates produced: {}" .format( sdlEvent.getNumberOfTrackCandidates()))
        print("# of TrackCandidates produced layer 1: {}" .format( sdlEvent.getNumberOfTrackCandidatesByLayerBarrel(0)))
        print("# of TrackCandidates produced layer 2: {}" .format( sdlEvent.getNumberOfTrackCandidatesByLayerBarrel(1)))
        print("# of TrackCandidates produced layer 3: {}" .format( sdlEvent.getNumberOfTrackCandidatesByLayerBarrel(2)))
        print("# of TrackCandidates produced layer 4: {}" .format( sdlEvent.getNumberOfTrackCandidatesByLayerBarrel(3)))
        print("# of TrackCandidates produced layer 5: {}" .format( sdlEvent.getNumberOfTrackCandidatesByLayerBarrel(4)))
        print("# of TrackCandidates produced layer 6: {}" .format( sdlEvent.getNumberOfTrackCandidatesByLayerBarrel(5)))
        print("# of TrackCandidates considered: {}" .format( sdlEvent.getNumberOfTrackCandidateCandidates()))
        print("# of TrackCandidates considered layer 1: {}" .format( sdlEvent.getNumberOfTrackCandidateCandidatesByLayerBarrel(0)))
        print("# of TrackCandidates considered layer 2: {}" .format( sdlEvent.getNumberOfTrackCandidateCandidatesByLayerBarrel(1)))
        print("# of TrackCandidates considered layer 3: {}" .format( sdlEvent.getNumberOfTrackCandidateCandidatesByLayerBarrel(2)))
        print("# of TrackCandidates considered layer 4: {}" .format( sdlEvent.getNumberOfTrackCandidateCandidatesByLayerBarrel(3)))
        print("# of TrackCandidates considered layer 5: {}" .format( sdlEvent.getNumberOfTrackCandidateCandidatesByLayerBarrel(4)))
        print("# of TrackCandidates considered layer 6: {}" .format( sdlEvent.getNumberOfTrackCandidateCandidatesByLayerBarrel(5)))

        # sys.exit()

        for index, (pt, eta, phi, q, dz, dxy, pdgid, bx) in enumerate(zip(t.sim_pt, t.sim_eta, t.sim_phi, t.sim_q, t.sim_pca_dz, t.sim_pca_dxy, t.sim_pdgId, t.sim_bunchCrossing)):

            if bx != 0:
                continue

            if pt > 1.0 and abs(eta) < 0.8:

                sim_denoms.append((pt, eta, phi, q, dz, dxy, pdgid, index))

                hit_layers = []

                for simhitidx in t.sim_simHitIdx[index]:

                    if t.simhit_subdet[simhitidx] == 5 and t.simhit_particle[simhitidx] == t.sim_pdgId[index]:

                        for hitidx in t.simhit_hitIdx[simhitidx]:

                            if t.ph2_subdet[hitidx] == 5:

                                hit_layers.append(t.ph2_layer[hitidx])

                                good_ph2_selected.append((t.ph2_x[hitidx], t.ph2_y[hitidx], t.ph2_z[hitidx], t.ph2_layer[hitidx], hitidx))
                                good_ph2_selected_by_layer[t.ph2_layer[hitidx]].append((t.ph2_x[hitidx], t.ph2_y[hitidx], t.ph2_z[hitidx], hitidx))

                if len(list(set(hit_layers))) == 6:

                    sim_denoms_have_all_hits.append((pt, eta, phi, q, dz, dxy, pdgid, index))

                    for simhitidx in t.sim_simHitIdx[index]:

                        if t.simhit_subdet[simhitidx] == 5 and t.simhit_particle[simhitidx] == t.sim_pdgId[index]:

                            for hitidx in t.simhit_hitIdx[simhitidx]:

                                if t.ph2_subdet[hitidx] == 5:

                                    good_ph2_selected_have_all_hits.append((t.ph2_x[hitidx], t.ph2_y[hitidx], t.ph2_z[hitidx], t.ph2_layer[hitidx], hitidx))
                                    good_ph2_selected_by_layer_have_all_hits[t.ph2_layer[hitidx]].append((t.ph2_x[hitidx], t.ph2_y[hitidx], t.ph2_z[hitidx], hitidx))


        break

print(len(ph2_selected))
print(len(ph2_selected_by_layer[1]))
print(len(ph2_selected_by_layer[2]))
print(len(ph2_selected_by_layer[3]))
print(len(ph2_selected_by_layer[4]))
print(len(ph2_selected_by_layer[5]))
print(len(ph2_selected_by_layer[6]))

print("Sim denom")
print(len(sim_denoms))
print(len(good_ph2_selected))
print(len(good_ph2_selected_by_layer[1]))
print(len(good_ph2_selected_by_layer[2]))
print(len(good_ph2_selected_by_layer[3]))
print(len(good_ph2_selected_by_layer[4]))
print(len(good_ph2_selected_by_layer[5]))
print(len(good_ph2_selected_by_layer[6]))
print(len(sim_denoms_have_all_hits))
print(len(good_ph2_selected_have_all_hits))
print(len(good_ph2_selected_by_layer_have_all_hits[1]))
print(len(good_ph2_selected_by_layer_have_all_hits[2]))
print(len(good_ph2_selected_by_layer_have_all_hits[3]))
print(len(good_ph2_selected_by_layer_have_all_hits[4]))
print(len(good_ph2_selected_by_layer_have_all_hits[5]))
print(len(good_ph2_selected_by_layer_have_all_hits[6]))

# Canvas
c1 = r.TCanvas("","",1700,1700)

h_dxy = r.TH1F("dxy_denom", "dxy_denom", 360, -25, 25)
for sim_denom in sim_denoms:
    h_dxy.Fill(sim_denom[4])
h_dxy.Draw("hist")

c1.SaveAs("test.pdf")

# # Drawing baseline background
# t.Draw("ph2_x:ph2_y","(ph2_subdet==5)&&(Entry$=={})".format(evt_idx_to_plot))
# print("Done plotting base")

# t.SetMarkerStyle(20)
# t.SetMarkerSize(0.5)
# # t.SetMarkerColorAlpha(2, 0.5)
# t.SetMarkerColor(2)
# for hit in good_ph2_selected:
#     t.Draw("{}:{}".format(hit[0], hit[1]),"Entry$==0", "same")

# t.SetMarkerStyle(20)
# t.SetMarkerSize(0.5)
# # t.SetMarkerColorAlpha(2, 0.5)
# t.SetMarkerColor(4)
# for hit in good_ph2_selected_have_all_hits:
#     t.Draw("{}:{}".format(hit[0], hit[1]),"Entry$==0", "same")

# c1.SaveAs("test.pdf")

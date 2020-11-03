#!/bin/env import

import ROOT as r
r.gROOT.SetBatch(True)

# SDL package
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
r.SDL.endcapGeometry.load("/home/users/phchang/public_html/analysis/sdl/TrackLooper_/scripts/endcap_orientation_data_v2.txt");
r.SDL.tiltedGeometry.load("/home/users/phchang/public_html/analysis/sdl/TrackLooper_/scripts/tilted_orientation_data.txt");
r.SDL.moduleConnectionMap.load("/home/users/phchang/public_html/analysis/sdl/TrackLooper_/scripts/module_connection_map_data_10_e0_200_100_pt0p8_2p0_400_pt0p8_2p0_nolossers_dxy35cm_endcaplayer2.txt");

# # RooUtil package
# r.gROOT.ProcessLine(".L rooutil/rooutil.so")
# r.gROOT.ProcessLine(".L rooutil/rooutil.h")

##############################################################
#
# Useful objects
#
##############################################################

#_____________________________________________________________
# SimTrack object
class SimTrack():
    def __init__(self, t, simtrk_idx):
        self.simtrk_idx = simtrk_idx
        self.pt = t.sim_pt[simtrk_idx]
        self.eta = t.sim_eta[simtrk_idx]
        self.phi = t.sim_phi[simtrk_idx]
        self.pdgId = t.sim_pdgId[simtrk_idx]
        self.q = t.sim_q[simtrk_idx]
        self.dz = t.sim_pca_dz[simtrk_idx]
        self.dxy = t.sim_pca_dxy[simtrk_idx]

##############################################################
#
# Steering SDL
#
##############################################################

#_____________________________________________________________
# Run the main SDL algorithm
def getSDLEvent(t, verbose=0):

    # Create SDL Event
    sdlEvent = r.SDL.Event()

    # Loop over reco hits
    for hit_idx, (x, y, z, subdet, layer, detid) in enumerate(zip(t.ph2_x, t.ph2_y, t.ph2_z, t.ph2_subdet, t.ph2_layer, t.ph2_detId)):

        # Then it is in outer tracker
        # if subdet == 5 or subdet == 4:
        if subdet == 5:

            # Add the hit to the SDL::Event
            sdlEvent.addHitToModule(r.SDL.Hit(x, y, z, hit_idx), t.ph2_detId[hit_idx])

    if verbose:
        # Print hit information
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
    sdlEvent.createMiniDoublets();
    if verbose:
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

    # Run segment
    sdlEvent.createSegmentsWithModuleMap();
    # sdlEvent.createSegmentsWithModuleMap();
    if verbose:
        print("# of Segments produced: {}" .format( sdlEvent.getNumberOfSegments()))
        print("# of Segments produced layer 1: {}" .format( sdlEvent.getNumberOfSegmentsByLayerBarrel(0)))
        print("# of Segments produced layer 2: {}" .format( sdlEvent.getNumberOfSegmentsByLayerBarrel(1)))
        print("# of Segments produced layer 3: {}" .format( sdlEvent.getNumberOfSegmentsByLayerBarrel(2)))
        print("# of Segments produced layer 4: {}" .format( sdlEvent.getNumberOfSegmentsByLayerBarrel(3)))
        print("# of Segments produced layer 5: {}" .format( sdlEvent.getNumberOfSegmentsByLayerBarrel(4)))
        print("# of Segments considered: {}" .format( sdlEvent.getNumberOfSegmentCandidates()))
        print("# of Segments considered layer 1: {}" .format( sdlEvent.getNumberOfSegmentCandidatesByLayerBarrel(0)))
        print("# of Segments considered layer 2: {}" .format( sdlEvent.getNumberOfSegmentCandidatesByLayerBarrel(1)))
        print("# of Segments considered layer 3: {}" .format( sdlEvent.getNumberOfSegmentCandidatesByLayerBarrel(2)))
        print("# of Segments considered layer 4: {}" .format( sdlEvent.getNumberOfSegmentCandidatesByLayerBarrel(3)))
        print("# of Segments considered layer 5: {}" .format( sdlEvent.getNumberOfSegmentCandidatesByLayerBarrel(4)))

    # Run tracklets
    sdlEvent.createTrackletsWithModuleMap();
    # sdlEvent.createTrackletsViaNavigation();
    if verbose:
        print("# of Tracklets produced: {}" .format( sdlEvent.getNumberOfTracklets()))
        print("# of Tracklets produced layer 1: {}" .format( sdlEvent.getNumberOfTrackletsByLayerBarrel(0)))
        print("# of Tracklets produced layer 2: {}" .format( sdlEvent.getNumberOfTrackletsByLayerBarrel(1)))
        print("# of Tracklets produced layer 3: {}" .format( sdlEvent.getNumberOfTrackletsByLayerBarrel(2)))
        print("# of Tracklets considered: {}" .format( sdlEvent.getNumberOfTrackletCandidates()))
        print("# of Tracklets considered layer 1: {}" .format( sdlEvent.getNumberOfTrackletCandidatesByLayerBarrel(0)))
        print("# of Tracklets considered layer 2: {}" .format( sdlEvent.getNumberOfTrackletCandidatesByLayerBarrel(1)))
        print("# of Tracklets considered layer 3: {}" .format( sdlEvent.getNumberOfTrackletCandidatesByLayerBarrel(2)))

    # Run track candidates
    sdlEvent.createTrackCandidatesFromTracklets();
    if verbose:
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

    return sdlEvent

##############################################################
#
# Doing some calculation with tracking ntuples
#
##############################################################

#_____________________________________________________________
# Good barrel tracks is where at least one sim hit with correct pdgid land on each layer
# Does not require pair of hits land on each layer.
# Does not require that the paired hits land on module pairs.
# Does not care whether a single layer has 4 hits
# Only one sim hit with correct pdgid is needed per layer to pass the requirement
# Input: TTree event, and sim trk index
def goodBarrelTracks(t, simtrk_idx, pdgid=0):

    # List of layer index with the simhit with correct pdgid
    # Check this later to get the list
    layer_idx_with_hits = []

    # Loop over the sim hit index
    for simhitidx in t.sim_simHitIdx[simtrk_idx]:

        # If not a correct sim hit skip
        if t.simhit_particle[simhitidx] != abs(t.sim_pdgId[simtrk_idx]):
            continue

        # Check it is barrel
        if t.simhit_subdet[simhitidx] != 5:
            continue

        # If pdgId condition is called require the pdgid
        if pdgid:
            if t.sim_pdgId[simtrk_idx] != abs(pdgid):
                continue

        # Add the layer index
        layer_idx_with_hits.append(t.simhit_layer[simhitidx])

    if sorted(list(set(layer_idx_with_hits))) == [1, 2, 3, 4, 5, 6]:
        return True
    else:
        return False

#_____________________________________________________________
# Get list of goodBarrelTracks
def listOfGoodBarrelTracks(t, pdgid=0):

    list_of_good_barrel_tracks = []

    # Loop over sim tracks
    for simtrk_idx, pt in enumerate(t.sim_pt):

        # Ask whether this is a good denominator
        if goodBarrelTracks(t, simtrk_idx, pdgid):

            list_of_good_barrel_tracks.append(SimTrack(t, simtrk_idx))

    return list_of_good_barrel_tracks

#_____________________________________________________________
# matched sim track indices of Track Candidate
# It could potentially return more than one
def matchedSimTrkIdxs(trkcand, t):
    # Aggregate 12 hit idxs
    hitidxs = []
    hitidxs.append(trkcand.innerTrackletPtr().innerSegmentPtr().innerMiniDoubletPtr().lowerHitPtr().idx())
    hitidxs.append(trkcand.innerTrackletPtr().innerSegmentPtr().innerMiniDoubletPtr().upperHitPtr().idx())
    hitidxs.append(trkcand.innerTrackletPtr().innerSegmentPtr().outerMiniDoubletPtr().lowerHitPtr().idx())
    hitidxs.append(trkcand.innerTrackletPtr().innerSegmentPtr().outerMiniDoubletPtr().upperHitPtr().idx())
    hitidxs.append(trkcand.outerTrackletPtr().innerSegmentPtr().innerMiniDoubletPtr().lowerHitPtr().idx())
    hitidxs.append(trkcand.outerTrackletPtr().innerSegmentPtr().innerMiniDoubletPtr().upperHitPtr().idx())
    hitidxs.append(trkcand.outerTrackletPtr().innerSegmentPtr().outerMiniDoubletPtr().lowerHitPtr().idx())
    hitidxs.append(trkcand.outerTrackletPtr().innerSegmentPtr().outerMiniDoubletPtr().upperHitPtr().idx())
    hitidxs.append(trkcand.outerTrackletPtr().outerSegmentPtr().innerMiniDoubletPtr().lowerHitPtr().idx())
    hitidxs.append(trkcand.outerTrackletPtr().outerSegmentPtr().innerMiniDoubletPtr().upperHitPtr().idx())
    hitidxs.append(trkcand.outerTrackletPtr().outerSegmentPtr().outerMiniDoubletPtr().lowerHitPtr().idx())
    hitidxs.append(trkcand.outerTrackletPtr().outerSegmentPtr().outerMiniDoubletPtr().upperHitPtr().idx())

    # Get sim trk idx
    # if no match add -1
    simtrk_idxs = []
    for hitidx in hitidxs:
        simtrk_idxs_per_hit = []
        for simhit_idx in t.ph2_simHitIdx[hitidx]:
            simtrk_idxs_per_hit.append(t.simhit_simTrkIdx[simhit_idx])
        if len(simtrk_idxs_per_hit) == 0:
            simtrk_idxs_per_hit.append(-1)
        simtrk_idxs.append(simtrk_idxs_per_hit)

    # Obtain all permutatin
    # [ [ 12, [ 12], [13], [12], [12, ... ]
    #     13],                    14],

    # using list comprehension  
    # to compute all possible permutations 
    perm = [[i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12]
            for i1  in simtrk_idxs[0]
            for i2  in simtrk_idxs[1]
            for i3  in simtrk_idxs[2]
            for i4  in simtrk_idxs[3]
            for i5  in simtrk_idxs[4]
            for i6  in simtrk_idxs[5]
            for i7  in simtrk_idxs[6]
            for i8  in simtrk_idxs[7]
            for i9  in simtrk_idxs[8]
            for i10 in simtrk_idxs[9]
            for i11 in simtrk_idxs[10]
            for i12 in simtrk_idxs[11]
          ] 

    matched_sim_trk_idxs = []

    # For each combination get the highest occurrence
    # Check that the highest occurence is not -1 (no match)
    # And check that the highest occurence is more than 9 (75% of 12)
    for elem in perm:
        print elem
        idx = max(elem,key=elem.count)
        if idx < 0:
            continue
        count = elem.count(idx)
        if count > 9:
            matched_sim_trk_idxs.append(idx)

    return matched_sim_trk_idxs

##############################################################
#
# Misc. or old
#
##############################################################

#_____________________________________________________________
# Checks whether a segment is from a true track or not
# Checks the associated simhit's simtrk idx
# and check whether there is a common trkidx associated
# with the anchor hits
# TODO if it is a 2S module, check if one of the two match
# For now, it will use lower one only
def isTrueSegment(sg, t):

    # Get inner anchorHit idx
    iMD_hitidx = sg.innerMiniDoubletPtr().anchorHitPtr().idx()

    # Get list of trk idx associated with the anchorHit
    innerMD_trk_idx = []
    for simhitidx in t.ph2_simHitIdx[iMD_hitidx]:
        t.simhit_particle
        innerMD_trk_idx.append(t.simhit_simTrkIdx[simhitidx])

    # Get list of trk idx associated with the anchorHit
    oMD_hitidx = sg.outerMiniDoubletPtr().anchorHitPtr().idx()
    outerMD_trk_idx = []
    for simhitidx in t.ph2_simHitIdx[oMD_hitidx]:
        outerMD_trk_idx.append(t.simhit_simTrkIdx[simhitidx])

    # Intersection of two list
    common_trk_idx = [value for value in innerMD_trk_idx if value in outerMD_trk_idx]

    return len(common_trk_idx) > 0


#_____________________________________________________________
def barrelTruthTracks(t, layer_requirement=[1,1,1,1,1,1]):

    # Return object
    rtn_obj = []

    # loop over tracks
    for index, (pt, eta, phi, q, dz, dxy, pdgid, bx) in enumerate(zip(t.sim_pt, t.sim_eta, t.sim_phi, t.sim_q, t.sim_pca_dz, t.sim_pca_dxy, t.sim_pdgId, t.sim_bunchCrossing)):

        # If not barrel eta continue
        if abs(eta) > 0.8:
            continue

        # Check simhit layers
        simhit_layers = []
        for simhitidx in t.sim_simHitIdx[index]:
            if t.simhit_subdet[simhitidx] == 5 and t.simhit_particle[simhitidx] == t.sim_pdgId[index]:
                simhit_layers.append(t.simhit_layer[simhitidx])

        # pass simhit_layer requirement
        is_good = True
        for index, req in enumerate(layer_requirement):
            if req:
                if index + 1 not in simhit_layers:
                    is_good = False

        if not is_good:
            continue

        hit_layers = []
        for simhitidx in t.sim_simHitIdx[index]:
            if t.simhit_subdet[simhitidx] == 5 and t.simhit_particle[simhitidx] == t.sim_pdgId[index]:
                for hitidx in t.simhit_hitIdx[simhitidx]:
                    if t.ph2_subdet[hitidx] == 5:
                        hit_layers.append(t.ph2_layer[hitidx])
        if len(list(set(hit_layers))) == 6:
            all_sim_in_0misshit.append((pt, eta, phi, q, dz, dxy, pdgid, index))

        # pass hit_layer requirement
        is_good = True
        for index, req in enumerate(layer_requirement):
            if req:
                if index + 1 not in hit_layers:
                    is_good = False

        if not is_good:
            continue

        rtn_obj.append((pt, eta, phi, q, dz, dxy, pdgid, index))

    return rtn_obj


#!/bin/env python

# Tracking Ntuple MTV efficiency code
# Philip Chang

import ROOT as r
import sys
import os
import math
import trkcore
import numpy as np

# Open the input data
t = r.TChain("trackingNtuple/tree")
t.Add("/home/users/phchang/public_html/analysis/sdl/trackingNtuple/CMSSW_10_4_0/src/trackingNtuple_100_pt0p5_2p0.root")
# t.Add("/home/users/phchang/public_html/analysis/sdl/trackingNtuple/CMSSW_10_4_0/src/trackingNtuple_10_pt0p5_5p0.root")
# t.Add("/nfs-7/userdata/bsathian/SDL_trackingNtuple/ttbar_highPU/trackingNtuple_with_PUinfo_500_evts.root")

# Set verbose mode
verbose = 0

denom_tracks = []
numer_tracks = []

# Loop over events
for index, event in enumerate(t):

    if index % 100 == 0:
        print(index)

    # Retrieve good barrel track
    list_of_good_barrel_tracks = trkcore.listOfGoodBarrelTracks(event)

    # Get an SDL event and run the reco on them
    sdlEvent = trkcore.getSDLEvent(event, verbose)

    denom_tracks += list_of_good_barrel_tracks

    numer_track_idxs = []
    for trkcand in sdlEvent.getLayer(1, r.SDL.Layer.Barrel).getTrackCandidatePtrs():
        for matchedSimTrkIdx in trkcore.matchedSimTrkIdxs(trkcand, event):
            if trkcore.goodBarrelTracks(event, matchedSimTrkIdx):
                numer_track_idxs.append(matchedSimTrkIdx)

    print("test")
    print(len(sdlEvent.getLayer(1, r.SDL.Layer.Barrel).getTrackCandidatePtrs()))

    for denom_track in denom_tracks:
        print(denom_track.simtrk_idx)

    for idx in list(set(numer_track_idxs)):
        numer_tracks.append(trkcore.SimTrack(event, idx))

    break

    if index > 10:
        break

denom_pts = [ denom.pt for denom in denom_tracks if abs(denom.eta) < 0.8 ]
numer_pts = [ numer.pt for numer in numer_tracks if abs(numer.eta) < 0.8 ]

f = open("denom_pts", "w")
for denom_pt in denom_pts: f.write("{}\n".format(denom_pt))

f = open("numer_pts", "w")
for numer_pt in numer_pts: f.write("{}\n".format(numer_pt))

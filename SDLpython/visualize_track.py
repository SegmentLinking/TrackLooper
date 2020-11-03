#!/bin/env python

import ROOT as r
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import pickle
import numpy as np
import math
import sdlmath
import SDLDisplay
from tqdm import tqdm

print "Opening root files ..."

# f = r.TFile("/home/users/phchang/public_html/analysis/sdl/trackingNtuple/CMSSW_10_4_0/src/trackingNtuple_100_pt0p5_3p0.root")
# f = r.TFile("/home/users/phchang/public_html/analysis/sdl/trackingNtuple/CMSSW_10_4_0/src/trackingNtuple_100_pt0p5_2p0.root")
# t = f.Get("trackingNtuple/tree")

import sys

# filepath = "/home/users/phchang/public_html/analysis/sdl/TrackLooper_/results/write_sdl_ntuple/pt0p5_2p0_2020_0710//fulleff_pt0p5_2p0.root"
filepath = "/home/users/phchang/public_html/analysis/sdl/TrackLooper_/results/write_sdl_ntuple/pu200_w_truthinfo_charged_2020_0801_Chuncheon//fulleff_pu200_w_truthinfo_charged.root"
treepath = "tree"
pick_ievent = 0
pick_itrack = 161717



## Pion samples
filepath = "/home/users/phchang/public_html/analysis/sdl/TrackLooper_/results/write_sdl_ntuple/pion_20200925_Test_v2//fulleff_pion.root"
treepath = "tree"
# pick_ievent = 1 
# pick_itrack = 14
# pick_ievent = 1 
# pick_itrack = 83
# pick_ievent = 2 
# pick_itrack = 65
# pick_ievent = 2 
# pick_itrack = 69
# pick_ievent = 2 
# pick_itrack = 230
# pick_ievent = 2 
# pick_itrack = 240
# pick_ievent = 2 
# pick_itrack = 241
# pick_ievent = 3 
# pick_itrack = 27
# pick_ievent = 3 
# pick_itrack = 44
# pick_ievent = 3 
# pick_itrack = 94
# pick_ievent = 4 
# pick_itrack = 11
# pick_ievent = 4 
# pick_itrack = 61
# pick_ievent = 4 
# pick_itrack = 343
# pick_ievent = 5 
# pick_itrack = 40
# pick_ievent = 5 
# pick_itrack = 67
# pick_ievent = 5 
# pick_itrack = 101
# pick_ievent = 5 
# pick_itrack = 218
# pick_ievent = 5 
# pick_itrack = 286
# pick_ievent = 6 
# pick_itrack = 20
# pick_ievent = 6 
# pick_itrack = 54
# pick_ievent = 6 
# pick_itrack = 94
pick_ievent = 6 
pick_itrack = 227
# pick_ievent = 7 
# pick_itrack = 64
# pick_ievent = 7 
# pick_itrack = 139
# pick_ievent = 8 
# pick_itrack = 16
# pick_ievent = 8 
# pick_itrack = 34
# pick_ievent = 8 
# pick_itrack = 148
# pick_ievent = 8 
# pick_itrack = 256
# pick_ievent = 9 
# pick_itrack = 19
# pick_ievent = 9 
# pick_itrack = 49
# pick_ievent = 9 
# pick_itrack = 85
# pick_ievent = 9 
# pick_itrack = 94

# 1 14
# Bad  25.64  0.24 x x x x x x
# 1 83
# Bad  41.57  0.22 o o x x x x
# 2 65
# Bad  47.70 -0.08 o o o x x x
# 2 69
# Bad  10.87 -0.07 x x x x x x
# 2 230
# Bad   2.22  0.13 o o o o x o
# 2 240
# Bad  14.33 -0.10 x x x o o o
# 2 241
# Bad  30.76 -0.08 x x x o o o
# 3 27
# Bad   1.79 -0.24 x x x x x x
# 3 44
# Bad  29.73 -0.79 o o o x x x
# 3 94
# Bad  44.35  0.04 o o o o o o
# 4 11
# Bad  15.74  0.39 o o x x x x
# 4 61
# Bad  46.40 -0.56 o o o o x x
# 4 343
# Bad   8.93 -0.44 x x x x x x
# 5 40
# Bad  35.30 -0.06 o o x x x x
# 5 67
# Bad  32.30 -0.75 o x x x x x
# 5 101
# Bad  15.30  0.71 x x x x x x
# 5 218
# Bad   7.75 -0.33 x x x x x o
# 5 286
# Bad   6.36  0.31 x x x x x x
# 6 20
# Bad  49.95 -0.46 o o x x x x
# 6 54
# Bad  18.16  0.06 o o o x x x
# 6 94
# Bad  18.34 -0.48 o o o x x x
# 6 227
# Bad  18.03  0.05 x x x o o o
# 7 64
# Bad  40.55  0.47 o o o o x x
# 7 139
# Bad  18.73  0.47 x x o o o o
# 8 16
# Bad   3.65  0.04 o o x x x x
# 8 34
# Bad   9.47 -0.41 o o o o o o
# THIS
# 8 148
# Bad  14.99  0.25 x x x x x x
# 8 256
# Bad   2.06 -0.26 x o o o o x
# 9 19
# Bad  14.83 -0.19 o o x x x x
# 9 49
# Bad  24.34  0.07 o o x x x x
# 9 85
# Bad  39.80 -0.58 o o o x x x
# 9 94
# Bad  30.09 -0.62 x x x x x x



try:
    filepath = sys.argv[1]
    treepath = sys.argv[2]
    pick_ievent = sys.argv[3]
    pick_itrack = sys.argv[4]
except:
    pass


f = r.TFile(filepath)
t = f.Get(treepath)

print "Opened root files ..."

t.GetEntry(pick_ievent)

print "Statrting ..."

for itrack, pt in enumerate(t.sim_pt):
    if itrack == pick_itrack:

        print "Found the", pick_itrack, "-th track in event", pick_ievent
        pt = t.sim_pt[itrack]
        eta = t.sim_eta[itrack]
        phi = t.sim_phi[itrack]
        charge = t.sim_q[itrack]
        vx = t.simvtx_x[0]
        vy = t.simvtx_y[0]
        vz = t.simvtx_z[0]

        print pt, eta, phi, charge, vx, vy, vz

        simhit_xs = []
        simhit_ys = []
        simhit_zs = []
        simhit_subdets = []
        simhit_pdgids  = []
        simhit_detids  = []
        for _, isimhit in  enumerate(t.sim_simHitIdx[itrack]):
            if t.simhit_subdet[isimhit] == 4 or t.simhit_subdet[isimhit] == 5:
                simhit_xs.append(t.simhit_x[isimhit])
                simhit_ys.append(t.simhit_y[isimhit])
                simhit_zs.append(t.simhit_z[isimhit])
                simhit_subdets.append(t.simhit_subdet[isimhit])
                simhit_pdgids.append(t.simhit_particle[isimhit])
                simhit_detids.append(int(t.simhit_detId[isimhit]))
        simhit_xs = np.array(simhit_xs)
        simhit_ys = np.array(simhit_ys)
        simhit_zs = np.array(simhit_zs)
        simhit_rs = np.sqrt(simhit_xs**2 + simhit_ys**2)

        print simhit_xs
        print simhit_ys
        print simhit_zs
        print simhit_rs
        print simhit_subdets
        print simhit_pdgids
        print simhit_detids

        break

detids = simhit_detids

sdlDisplay = SDLDisplay.getDefaultSDLDisplay()
fullSDLDisplay = SDLDisplay.getDefaultSDLDisplay()

sdlDisplay.set_detector_rz_collection(detids)
sdlDisplay.set_detector_xy_collection(detids)
sdlDisplay.set_detector_etaphi_collection(detids)

# ax = pickle.load(file('/nfs-7/userdata/phchang/detector_layout_matplotlib_pickle/detrz.pickle'))
ax = pickle.load(file('detrz.pickle'))
fullSDLDisplay.display_detector_rz(ax)
sdlDisplay.display_detector_rz(ax, color=(1,0,0))
sdlmath.draw_track_rz(ax, pt, eta, phi, vx, vy, vz, charge)
# lc = LineCollection(segments_rz, colors=(1,0,0), linewidth=0.5, alpha=0.4)
# ax.add_collection(lc)
plt.scatter(simhit_zs, simhit_rs, s=0.5)
plt.savefig("pdfs/detrz_track_visualization.pdf")

# ax = pickle.load(file('/nfs-7/userdata/phchang/detector_layout_matplotlib_pickle/detxy.pickle'))
ax = pickle.load(file('detxy.pickle'))
fullSDLDisplay.display_detector_xy(ax)
sdlDisplay.display_detector_xy(ax, color=(0,0,1))
sdlmath.draw_track_xy(ax, pt, eta, phi, vx, vy, vz, charge)
# lc = LineCollection(segments_xy, colors=(1,0,0), linewidth=0.5, alpha=0.4)
# ax.add_collection(lc)
plt.scatter(simhit_xs, simhit_ys, s=0.5)
plt.savefig("pdfs/detxy_track_visualization.pdf")

fig, ax = plt.subplots(figsize=(4. * 2,2.*math.pi))
sdlDisplay.display_detector_etaphi(ax, color=(0,0,1))
fig.savefig("pdfs/detetaphi_track_visualization.pdf")


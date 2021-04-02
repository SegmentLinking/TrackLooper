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

# f = r.TFile("/home/users/phchang/public_html/analysis/sdl/trackingNtuple/CMSSW_10_4_0/src/trackingNtuple_100_pt0p5_3p0.root")
f = r.TFile("/home/users/phchang/public_html/analysis/sdl/trackingNtuple/CMSSW_10_4_0/src/trackingNtuple_100_pt0p5_2p0.root")
# f = r.TFile("/home/users/phchang/public_html/analysis/sdl/trackingNtuple/CMSSW_10_4_0/src/trackingNtuple_10_pt0p5_5p0.root")
t = f.Get("trackingNtuple/tree")
pick_ievent = 9806
select_itracks = []
select_irecohits = []
segment_idxs = []

select_itracks = [32, 28]
# select_itracks = [85, 1, 59]
# select_itracks = [36, 52, 37, 53]

# select_irecohits = [24,
# 26,
# 35,
# 36,
# 38,
# 39,
# 40,
# 41,
# 42,
# 43,
# 87,
# 88,
# 89,
# 90,
# 130,
# 131,
# 132,
# 133,
# 134,
# 135,
# 136,
# 137,
# 232,
# 233,
# 234,
# 235,
# 286,
# 287,
# 288,
# 289,
# 331,
# 332,
# 333,
# 334,
# 335,
# 336,
# 454,
# 455,
# 456,
# 457,
# 458,
# 459,
# 464,
# 465,
# 467,
# 468,
# 507,
# 508,
# 511,
# 512,
# 622,
# 623,
# 624,
# 625,
# 628,
# 631,
# 654,
# 655,
# 656,
# 657,
# 737,
# 738,
# 739,
# 740,
# 768,
# 769,
# 770,
# 771,
# 920,
# 923,
# 924,
# 925,]

# segment_idxs = [
# [26, 87],
# [26, 88],
# [35, 87],
# [36, 87],
# [42, 87],
# [43, 87],
# [87, 130],
# [87, 131],
# [87, 136],
# [87, 137],
# [88, 130],
# [88, 131],
# [88, 136],
# [88, 137],
# [286, 332],
# [286, 333],
# [287, 332],
# [287, 333],
# [287, 334],
# [454, 622],
# [454, 623],
# [454, 628],
# [458, 622],
# [458, 623],
# [458, 628],
# [459, 622],
# [459, 623],
# [459, 628],
# [464, 622],
# [464, 623],
# [464, 628],
# [465, 622],
# [465, 623],
# [465, 628],
# [507, 657],
# [508, 656],
# [739, 26],
# [739, 35],
# [739, 36],
# [739, 42],
# [739, 43],
# [739, 920],
# [739, 923],
# [740, 26],
# [740, 35],
# [740, 36],
# [740, 42],
# [740, 43],
# [740, 920],
# [740, 923],
# [768, 235],
# [769, 234],
# [769, 235],
# ]

t.GetEntry(pick_ievent)

sdlDisplay = SDLDisplay.getDefaultSDLDisplay()
fullSDLDisplay = SDLDisplay.getDefaultSDLDisplay()
# ax = pickle.load(file('/nfs-7/userdata/phchang/detector_layout_matplotlib_pickle/detrz.pickle'))
ax_rz = pickle.load(file('detrz.pickle'))
# ax = pickle.load(file('/nfs-7/userdata/phchang/detector_layout_matplotlib_pickle/detxy.pickle'))
ax_xy = pickle.load(file('detxy.pickle'))

simhit_xs = []
simhit_ys = []
simhit_zs = []
simhit_subdets = []
simhit_pdgids  = []
simhit_detids  = []

recohit_xs = []
recohit_ys = []
recohit_zs = []

for itrack, pt in enumerate(t.sim_pt):

    if len(select_itracks) > 0:
        if itrack not in select_itracks:
            continue

    # if itrack >= 50:
    #     break

    pt = t.sim_pt[itrack]
    eta = t.sim_eta[itrack]
    phi = t.sim_phi[itrack]
    charge = t.sim_q[itrack]
    vx = t.simvtx_x[0]
    vy = t.simvtx_y[0]
    vz = t.simvtx_z[0]

    if abs(eta) > 2.4:
        continue

    print pt, eta, phi, charge, vx, vy, vz
    sdlmath.draw_track_rz(ax_rz, pt, eta, phi, vx, vy, vz, charge)
    sdlmath.draw_track_xy(ax_xy, pt, eta, phi, vx, vy, vz, charge)

    for _, isimhit in  enumerate(t.sim_simHitIdx[itrack]):
        if t.simhit_subdet[isimhit] == 4 or t.simhit_subdet[isimhit] == 5:
            simhit_xs.append(t.simhit_x[isimhit])
            simhit_ys.append(t.simhit_y[isimhit])
            simhit_zs.append(t.simhit_z[isimhit])
            simhit_subdets.append(t.simhit_subdet[isimhit])
            simhit_pdgids.append(t.simhit_particle[isimhit])
            simhit_detids.append(int(t.simhit_detId[isimhit]))

for idx, _ in enumerate(t.ph2_x):

    if len(select_irecohits) > 0:
        if idx not in select_irecohits:
            continue

    if t.ph2_subdet[idx] == 4 or t.ph2_subdet[idx] == 5:
        recohit_xs.append(t.ph2_x[idx])
        recohit_ys.append(t.ph2_y[idx])
        recohit_zs.append(t.ph2_z[idx])

rz_segments = []
xy_segments = []
for segment_idx in segment_idxs:
    rz_segment = [(t.ph2_z[segment_idx[0]], math.sqrt(t.ph2_x[segment_idx[0]]**2 + t.ph2_y[segment_idx[0]]**2 )), (t.ph2_z[segment_idx[1]], math.sqrt(t.ph2_x[segment_idx[1]]**2 + t.ph2_y[segment_idx[1]]**2 ))]
    rz_segments.append(rz_segment)
    xy_segment = [(t.ph2_x[segment_idx[0]], t.ph2_y[segment_idx[0]]), (t.ph2_x[segment_idx[1]], t.ph2_y[segment_idx[1]])]
    xy_segments.append(xy_segment)

print xy_segments

simhit_xs = np.array(simhit_xs)
simhit_ys = np.array(simhit_ys)
simhit_zs = np.array(simhit_zs)
simhit_rs = np.sqrt(simhit_xs**2 + simhit_ys**2)

recohit_xs = np.array(recohit_xs)
recohit_ys = np.array(recohit_ys)
recohit_zs = np.array(recohit_zs)
recohit_rs = np.sqrt(recohit_xs**2 + recohit_ys**2)

# ax_rz.scatter(simhit_zs, simhit_rs, s=0.1)
# ax_xy.scatter(simhit_xs, simhit_ys, s=0.1)
ax_rz.scatter(recohit_zs, recohit_rs, s=0.1)
ax_xy.scatter(recohit_xs, recohit_ys, s=0.1)

ax_xy.add_collection(LineCollection(xy_segments, colors=(0,0,1), linewidth=0.5, alpha=0.9))
ax_rz.add_collection(LineCollection(rz_segments, colors=(0,0,1), linewidth=0.5, alpha=0.9))

plt.sca(ax_rz)
plt.savefig("detrz_event_visualization.pdf")
plt.sca(ax_xy)
plt.savefig("detxy_event_visualization.pdf")


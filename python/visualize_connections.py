#!/bin/env python

import SDLDisplay
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle
from Centroid import Centroid
import math
from Module import Module
import numpy as np
import sys
from tqdm import tqdm
import pickle

# Centroid database
centroidDB = Centroid("data/centroid_2020_0428.txt")

# f = open("data/module_connection_2020_0429.txt")
# f = open("data/module_connection_tracing.txt")
# f = open("data/module_connection_tracing_2020_0514_ray.txt")
# f = open("data/module_connection_tracing_2020_0518_helix.txt")
f = open("data/module_connection_combined_2020_0520_helixray.txt")
# f = open("data/module_connection_combined_2020_0518_helixray.txt")
# f = open("data/module_connection_combined.txt")
# f = open("data/module_connection_nmuon10.txt")
# f = open("data/module_connection_nmuon5.txt")
# f = open("/home/users/phchang/public_html/analysis/sdl/TrackLooper_/scripts/module_connection_map_data_10_e0_200_100_pt0p8_2p0_400_pt0p8_2p0_nolossers_dxy35cm_endcaplayer2.txt")

lines = f.readlines()

n_outliers = 0

outlier_ref_detid = []
outlier_tar_detids = []

detids = []
segments_rz = []
segments_xy = []

nconns_barrel_per_layer = {}
nconns_endcap_per_layer = {}
nconns_barrel_per_layer[1] = []
nconns_barrel_per_layer[2] = []
nconns_barrel_per_layer[3] = []
nconns_barrel_per_layer[4] = []
nconns_barrel_per_layer[5] = []
nconns_barrel_per_layer[6] = []
nconns_endcap_per_layer[1] = []
nconns_endcap_per_layer[2] = []
nconns_endcap_per_layer[3] = []
nconns_endcap_per_layer[4] = []
nconns_endcap_per_layer[5] = []

for line in tqdm(lines, desc="Looping over connection map"):
    ls = line.split()
    ref_detid = int(ls[0])
    nconn = int(ls[1])
    target_detids = [ int(x) for x in ls[2:] ]

    module = Module(ref_detid)
    layer = module.layer()
    subdet = module.subdet()

    if subdet == 5:
        nconns_barrel_per_layer[layer].append(nconn)
    else:
        nconns_endcap_per_layer[layer].append(nconn)

    if nconn >= 20:
        n_outliers += 1
        detids += target_detids
        detids.append(ref_detid)
        centroid_ref = centroidDB.getCentroid(ref_detid)
        outlier_ref_detid.append(ref_detid)
        outlier_tar_detids.append(target_detids)
        ref_x = centroid_ref[0]
        ref_y = centroid_ref[1]
        ref_z = centroid_ref[2]
        ref_rt = math.sqrt(ref_x**2 + ref_y**2)
        for target_detid in target_detids:
            centroid_target = centroidDB.getCentroid(target_detid)
            target_x = centroid_target[0]
            target_y = centroid_target[1]
            target_z = centroid_target[2]
            target_rt = math.sqrt(target_x**2 + target_y**2)
            segments_rz.append([(ref_z, ref_rt), (target_z, target_rt)])
            segments_xy.append([(ref_x, ref_y), (target_x, target_y)])

print(n_outliers)

nconns_barrel = [nconns_barrel_per_layer[layer] for layer in range(1,6)] + [nconns_endcap_per_layer[layer] for layer in range(1,5)]
labels = ["Barrel Layer{}".format(layer) for layer in range(1,6)] + ["Endcap Layer{}".format(layer) for layer in range(1,5)]
plt.hist(nconns_barrel, bins=range(30), alpha=1.0, stacked=True, label=labels)
plt.legend()
plt.ylabel("# of modules")
plt.xlabel("# of connections")
plt.title("# of connections")
plt.show()
plt.savefig("nconn.pdf")

sdlDisplay = SDLDisplay.getDefaultSDLDisplay()
fullSDLDisplay = SDLDisplay.getDefaultSDLDisplay()

sdlDisplay.set_detector_rz_collection(detids)
sdlDisplay.set_detector_xy_collection(detids)

# ax = pickle.load(file('/nfs-7/userdata/phchang/detector_layout_matplotlib_pickle/detrz.pickle'))
ax = pickle.load(file('detrz.pickle'))
fullSDLDisplay.display_detector_rz(ax)
sdlDisplay.display_detector_rz(ax, color=(1,0,0))
lc = LineCollection(segments_rz, colors=(1,0,0), linewidth=0.5, alpha=0.4)
ax.add_collection(lc)
plt.savefig("detrz_outlier_connections.pdf")

# ax = pickle.load(file('/nfs-7/userdata/phchang/detector_layout_matplotlib_pickle/detxy.pickle'))
ax = pickle.load(file('detxy.pickle'))
fullSDLDisplay.display_detector_xy(ax)
sdlDisplay.display_detector_xy(ax, color=(0,0,1))
lc = LineCollection(segments_xy, colors=(1,0,0), linewidth=0.5, alpha=0.4)
ax.add_collection(lc)
plt.savefig("detxy_outlier_connections.pdf")

import os
try:
    os.mkdir("outlier_connections/")
except:
    pass

for ref_detid, tar_detids in zip(outlier_ref_detid, outlier_tar_detids):

    fig, ax = plt.subplots(figsize=(4. * 2,2.*math.pi))

    sdlDisplay.set_detector_etaphi_collection([ref_detid])
    sdlDisplay.display_detector_etaphi(ax, color=(1,0,0))

    sdlDisplay.set_detector_etaphi_collection(tar_detids)
    sdlDisplay.display_detector_etaphi(ax, color=(0,0,1))

    fig.savefig("outlier_connections/det_etaphi_outlier_connections_refdetid{}.pdf".format(ref_detid))
    fig.savefig("outlier_connections/det_etaphi_outlier_connections_refdetid{}.png".format(ref_detid))

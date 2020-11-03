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

rows = []
for line in sys.stdin:
    line = line.strip()
    try: parts = map(int, line.split())
    except: continue
    rows.append(parts)

# flatten the list
rows = [item for sublist in rows for item in sublist]

# Found trk not matched
# 437787702
# 437787705
# 438838350
# 438838353
# 439890034
# 419730554
# 420000930
# 420000930
# 420000925
# 420254893
# 420525282

# detids=[437787702
# ,437787705
# ,438838350
# ,438838353
# ,439890034
# ,419730554
# ,420000930
# ,420000930
# ,420000925
# ,420254893
# ,420525282]


# detids=[437524529,
# 438572102,
# 438571078,
# 438570054,
# 438570054,
# 439617634,
# 439617634,
# 411350153,
# 411346038,
# 411346038,
# 411616417,
# 411870382,
# 411874497,
# 412140753,
# 412140753]

print rows

detids = rows

sdlDisplay = SDLDisplay.getDefaultSDLDisplay()
fullSDLDisplay = SDLDisplay.getDefaultSDLDisplay()

sdlDisplay.set_detector_rz_collection(detids)
sdlDisplay.set_detector_xy_collection(detids)
sdlDisplay.set_detector_etaphi_collection(detids)

# ax = pickle.load(file('/nfs-7/userdata/phchang/detector_layout_matplotlib_pickle/detrz.pickle'))
ax = pickle.load(file('detrz.pickle'))
fullSDLDisplay.display_detector_rz(ax)
sdlDisplay.display_detector_rz(ax, color=(1,0,0))
# lc = LineCollection(segments_rz, colors=(1,0,0), linewidth=0.5, alpha=0.4)
# ax.add_collection(lc)
plt.savefig("detrz_outlier_connections.pdf")

# ax = pickle.load(file('/nfs-7/userdata/phchang/detector_layout_matplotlib_pickle/detxy.pickle'))
ax = pickle.load(file('detxy.pickle'))
fullSDLDisplay.display_detector_xy(ax)
sdlDisplay.display_detector_xy(ax, color=(0,0,1))
# lc = LineCollection(segments_xy, colors=(1,0,0), linewidth=0.5, alpha=0.4)
# ax.add_collection(lc)
plt.savefig("detxy_outlier_connections.pdf")

fig, ax = plt.subplots(figsize=(4. * 2,2.*math.pi))
sdlDisplay.display_detector_etaphi(ax, color=(0,0,1))
fig.savefig("detetaphi_outlier_connections.pdf")


# import os
# try:
#     os.mkdir("outlier_connections/")
# except:
#     pass

# for ref_detid, tar_detids in zip(outlier_ref_detid, outlier_tar_detids):

#     fig, ax = plt.subplots(figsize=(4. * 2,2.*math.pi))

#     sdlDisplay.set_detector_etaphi_collection([ref_detid])
#     sdlDisplay.display_detector_etaphi(ax, color=(1,0,0))

#     sdlDisplay.set_detector_etaphi_collection(tar_detids)
#     sdlDisplay.display_detector_etaphi(ax, color=(0,0,1))

#     fig.savefig("outlier_connections/det_etaphi_outlier_connections_refdetid{}.pdf".format(ref_detid))
#     fig.savefig("outlier_connections/det_etaphi_outlier_connections_refdetid{}.png".format(ref_detid))

#!/bin/env python

import ROOT as r
import math

# This import registers the 3D projection, but is otherwise unused.
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

# Segment data
f = open("sg_data.txt")
_start_xs = {1: [], 2: [], 3: [], 4: [], 5: []}
_start_ys = {1: [], 2: [], 3: [], 4: [], 5: []}
_start_zs = {1: [], 2: [], 3: [], 4: [], 5: []}
_end_xs   = {1: [], 2: [], 3: [], 4: [], 5: []}
_end_ys   = {1: [], 2: [], 3: [], 4: [], 5: []}
_end_zs   = {1: [], 2: [], 3: [], 4: [], 5: []}
lines = f.readlines()
for line in lines:
    _start_xs[int(line.split()[6])].append(float(line.split()[0]))
    _start_ys[int(line.split()[6])].append(float(line.split()[1]))
    _start_zs[int(line.split()[6])].append(float(line.split()[2]))
    _end_xs  [int(line.split()[6])].append(float(line.split()[3]))
    _end_ys  [int(line.split()[6])].append(float(line.split()[4]))
    _end_zs  [int(line.split()[6])].append(float(line.split()[5]))

# True segments
_true_start_xs = []
_true_start_ys = []
_true_start_zs = []
_true_end_xs = []
_true_end_ys = []
_true_end_zs = []
_fake_start_xs = []
_fake_start_ys = []
_fake_start_zs = []
_fake_end_xs = []
_fake_end_ys = []
_fake_end_zs = []
for line in lines:
    if line.split()[7] == "True":
        _true_start_xs.append(float(line.split()[0]))
        _true_start_ys.append(float(line.split()[1]))
        _true_start_zs.append(float(line.split()[2]))
        _true_end_xs  .append(float(line.split()[3]))
        _true_end_ys  .append(float(line.split()[4]))
        _true_end_zs  .append(float(line.split()[5]))
    else:
        _fake_start_xs.append(float(line.split()[0]))
        _fake_start_ys.append(float(line.split()[1]))
        _fake_start_zs.append(float(line.split()[2]))
        _fake_end_xs  .append(float(line.split()[3]))
        _fake_end_ys  .append(float(line.split()[4]))
        _fake_end_zs  .append(float(line.split()[5]))

# MiniDoublet data
md = open("md_data.txt")
_md_xs = {1: [], 2: [], 3: [], 4: [], 5: [], 6:[]}
_md_ys = {1: [], 2: [], 3: [], 4: [], 5: [], 6:[]}
_md_zs = {1: [], 2: [], 3: [], 4: [], 5: [], 6:[]}
lines = md.readlines()
for line in lines:
    _md_xs[int(line.split()[3])].append(float(line.split()[0]))
    _md_ys[int(line.split()[3])].append(float(line.split()[1]))
    _md_zs[int(line.split()[3])].append(float(line.split()[2]))

# # Tracklet data
# md = open("tl_data.txt")
# _tracklet_xs = {1: [], 2: [], 3: []}
# _tracklet_ys = {1: [], 2: [], 3: []}
# _tracklet_zs = {1: [], 2: [], 3: []}
# lines = md.readlines()
# for line in lines:
#     _md_xs[int(line.split()[3])].append(float(line.split()[0]))
#     _md_ys[int(line.split()[3])].append(float(line.split()[1]))
#     _md_zs[int(line.split()[3])].append(float(line.split()[2]))

fig = plt.figure(figsize=(10,10), dpi=800)
# fig = plt.figure(figsize=(17,6), dpi=800)
ax = fig.gca()
print ax

print len(_start_xs)

colors = {1: (0, 0.5, 0.5), 2: (1, 0, 1), 3: (1, 0, 0), 4: (0, 1, 0), 5: (0, 0, 1)}
colors = {1: (0, 0.5, 0.5), 2: (1, 0, 1), 3: (1, 0, 0), 4: (0, 1, 0), 5: (0, 0, 1), 6:(0.5, 0.5, 0)}

segments = []
for i in xrange(len(_true_start_xs)):
    if i % 1000 == 0:
        print i
    segments.append([(_true_start_xs[i], _true_start_ys[i]), (_true_end_xs[i], _true_end_ys[i])])
    # segments.append([(_true_start_zs[i], math.sqrt(_true_start_xs[i]**2+_true_start_ys[i]**2)), (_true_end_zs[i], math.sqrt(_true_end_xs[i]**2+_true_end_ys[i]**2))])

# lc = LineCollection(segments, colors=(0.5,0.5,0.5), linewidth=0.1)
lc = LineCollection(segments, colors=(1,0,0), linewidth=1)
ax.autoscale()
ax.add_collection(lc)
# ax.set_ylim(20, 120)
ax.set_ylim(-150, 150)
ax.set_xlim(-150, 150)

# for l in _start_xs:
#     segments = []
#     for i in xrange(len(_start_xs[l])):
#         if i % 1000 == 0:
#             print i
#         # segments.append([(_start_zs[l][i],_end_zs[l][i]), (math.sqrt(_start_xs[l][i]**2+_start_ys[l][i]**2), math.sqrt(_end_xs[l][i]**2+_end_ys[l][i]**2))])
#         segments.append([(_start_zs[l][i], math.sqrt(_start_xs[l][i]**2+_start_ys[l][i]**2)), (_end_zs[l][i], math.sqrt(_end_xs[l][i]**2+_end_ys[l][i]**2))])

#     lc = LineCollection(segments, colors=colors[l], linewidth=0.1)
#     ax.autoscale()
#     ax.add_collection(lc)
#     ax.set_ylim(0, 120)

# for l in _md_xs:
#     mds_x = []
#     mds_y = []
#     for i in xrange(len(_md_xs[l])):
#         if i % 1000 == 0:
#             print i
#         mds_x.append(_md_zs[l][i])
#         mds_y.append(math.sqrt(_md_xs[l][i]**2+_md_ys[l][i]**2))
#         # ax.plot(_md_zs[l][i], math.sqrt(_md_xs[l][i]**2+_md_ys[l][i]**2), color=colors[l], marker="o", markersize=2.0)
#     ax.scatter(mds_x, mds_y, color=colors[l])

plt.show()

# fig.savefig('xy.pdf')
fig.savefig('xy.png')
# fig.savefig('rz.png')

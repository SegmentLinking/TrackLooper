#!/bin/env python

from Module import Module
from Centroid import Centroid

import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from SDLDisplay import SDLDisplay
from DetectorGeometry import DetectorGeometry

import os

# Centroid database
centroidDB = Centroid("data/centroid_2020_0428.txt")

nmod = 0
for key in centroidDB.data:
    if Module(key).logicalLayer() == 6 or Module(key).logicalLayer() == 11 or Module(key).isLower() == 0:
        continue
    if Module(key).ring() == 15 and Module(key).logicalLayer() == 7:
        continue
    if Module(key).ring() == 15 and Module(key).logicalLayer() == 8:
        continue
    if Module(key).ring() == 12 and Module(key).logicalLayer() == 9:
        continue
    if Module(key).ring() == 12 and Module(key).logicalLayer() ==10:
        continue
    nmod += 1
print(nmod)

# f = open("data/module_connection.txt")
# detids = [ int(x.split()[0]) for x in f.readlines() ]

# dirpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# det_geom = DetectorGeometry("{}/data/phase2_2020_0428.txt".format(dirpath))
# sdlDisplay = SDLDisplay(det_geom)
# sdlDisplay.set_detector_xy_collection(detids)
# sdlDisplay.set_detector_rz_collection(detids)

# fig, ax = plt.subplots(figsize=(10,4))
# sdlDisplay.display_detector_rz(ax)
# fig.savefig("detrz.pdf")

#!/bin/env python

import numpy as np
import os
import math
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
# import mpl_toolkits.mplot3d as a3
import pylab as pl
from DetectorGeometry import DetectorGeometry
from Module import Module
import sdlmath
from Centroid import Centroid
from tqdm import tqdm
import pickle
from matplotlib.collections import LineCollection
import multiprocessing

# Setting up detector geometry (centroids and boundaries)
centroidDB = Centroid("data/centroid_2020_0428.txt")
dirpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
det_geom = DetectorGeometry("{}/data/phase2_2020_0428.txt".format(dirpath))
det_geom.buildByLayer()

# for barrel layers compute the average R
for ilayer in range(1, 7):
    tar_detids_in_layer = []
    tar_detids_in_layer += det_geom.getBarrelLayerDetIds(ilayer)

    # Loop over the modules in the layer
    radii = []
    for tar_detid in tar_detids_in_layer:
        centroid = centroidDB.getCentroid(tar_detid)
        radii.append(math.sqrt(centroid[0]**2 + centroid[1]**2))
    radii = np.array(radii)
    radii_sum = radii.sum()
    avg_radius = radii_sum / len(tar_detids_in_layer)

    print avg_radius

# for barrel layers compute the average Z
for ilayer in range(1, 6):
    tar_detids_in_layer = []
    tar_detids_in_layer += det_geom.getEndcapLayerDetIds(ilayer)

    # Loop over the modules in the layer
    zs = []
    for tar_detid in tar_detids_in_layer:
        centroid = centroidDB.getCentroid(tar_detid)
        zs.append(abs(centroid[2]))
    zs = np.array(zs)
    zs = zs.sum()
    avg_radius = zs / len(tar_detids_in_layer)

    print avg_radius

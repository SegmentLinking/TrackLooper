#!/bin/env python

import LSTDisplay
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

####################################
#
# Usage
#
#  $ python visualize_modules.py DETID1 DETID2 DETID3 ...
#
#
####################################

# Get the input of modules via stdin
rows = []
for line in sys.stdin:
    line = line.strip()
    try: parts = map(int, line.split())
    except: continue
    rows.append(parts)

# flatten the list
detids = [item for sublist in rows for item in sublist]

# Print the list of modules to plot that was provided by the user
print detids

# Visualize the detector via drawing xy or rz projections
sdlDisplay = LSTDisplay.getDefaultLSTDisplay()
fullLSTDisplay = LSTDisplay.getDefaultLSTDisplay()

# Set the displayer detids to turn on
sdlDisplay.set_detector_rz_collection(detids)
sdlDisplay.set_detector_xy_collection(detids)
sdlDisplay.set_detector_etaphi_collection(detids)

# Load the detector module elements
ax = pickle.load(file('/data2/segmentlinking/detrz.pickle'))
fullLSTDisplay.display_detector_rz(ax)
sdlDisplay.display_detector_rz(ax, color=(1,0,0))
plt.savefig("visualization_module_detrz.pdf")

# Load the detector module elements
ax = pickle.load(file('/data2/segmentlinking/detxy.pickle'))
fullLSTDisplay.display_detector_xy(ax)
sdlDisplay.display_detector_xy(ax, color=(0,0,1))
plt.savefig("visualization_module_detxy.pdf")

fig, ax = plt.subplots(figsize=(4. * 2,2.*math.pi))
sdlDisplay.display_detector_etaphi(ax, color=(0,0,1))
fig.savefig("visualization_module_detetaphi.pdf")

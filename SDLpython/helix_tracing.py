#!/bin/env python

import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import pickle
import numpy as np
import math
import sdlmath
from scipy import optimize
from DetectorGeometry import DetectorGeometry
from Centroid import Centroid
from Module import Module
import SDLDisplay
import os

if __name__ == "__main__":

    dirpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    det_geom = DetectorGeometry("{}/data/phase2_2020_0428.txt".format(dirpath))
    centroidDB = Centroid("data/centroid_2020_0428.txt")

    list_of_detids_xy = det_geom.getDetIds(lambda x: Module(x[0]).subdet() == 5 and Module(x[0]).side() == 3 and Module(x[0]).module() == 4 and Module(x[0]).layer() == 1 and Module(x[0]).isLower() == 1 and Module(x[0]).rod() == 1)

    print(list_of_detids_xy)

    for detid in list_of_detids_xy:

        # Printing module information
        module = Module(detid)

        # Obtaining 4 boundary points
        boundary_points = np.array(det_geom.getData()[detid])

        # Obtaining the centroid of the module
        centroid = np.array(centroidDB.getCentroid(detid))

        # Get a canvas with detector laid out in xy
        ax_xy = pickle.load(file('/nfs-7/userdata/phchang/detector_layout_matplotlib_pickle/detxy.pickle'))
        ax_rz = pickle.load(file('/nfs-7/userdata/phchang/detector_layout_matplotlib_pickle/detrz.pickle'))

        # "Display"-er with the module we want only
        sdlDisplay = SDLDisplay.getDefaultSDLDisplay() # Get an object to handle reading of centroids, and detector geometry etc. etc.
        sdlDisplay.set_detector_xy_collection([detid]) # Turn on only the given modules in a list
        sdlDisplay.set_detector_rz_collection([detid]) # Turn on only the given modules in a list
        sdlDisplay.display_detector_xy(ax_xy, (1,0,0))
        sdlDisplay.display_detector_rz(ax_rz, (1,0,0))

        # Draw a track passing through one of the boundary
        pt = 1.
        sdlmath.draw_track_xy_from_points(ax_xy, pt, 0., 0., 0., boundary_points[0][1], boundary_points[0][2], boundary_points[0][0], -1)
        sdlmath.draw_track_xy_from_points(ax_xy, pt, 0., 0., 0., boundary_points[1][1], boundary_points[1][2], boundary_points[1][0], -1)
        sdlmath.draw_track_xy_from_points(ax_xy, pt, 0., 0., 0., boundary_points[2][1], boundary_points[2][2], boundary_points[2][0], -1)
        sdlmath.draw_track_xy_from_points(ax_xy, pt, 0., 0., 0., boundary_points[3][1], boundary_points[3][2], boundary_points[3][0], -1)
        sdlmath.draw_track_rz_from_points(ax_rz, pt, 0., 0., 0., boundary_points[0][1], boundary_points[0][2], boundary_points[0][0], -1)
        sdlmath.draw_track_rz_from_points(ax_rz, pt, 0., 0., 0., boundary_points[1][1], boundary_points[1][2], boundary_points[1][0], -1)
        sdlmath.draw_track_rz_from_points(ax_rz, pt, 0., 0., 0., boundary_points[2][1], boundary_points[2][2], boundary_points[2][0], -1)
        sdlmath.draw_track_rz_from_points(ax_rz, pt, 0., 0., 0., boundary_points[3][1], boundary_points[3][2], boundary_points[3][0], -1)
        sdlmath.draw_track_xy_from_points(ax_xy, pt, 0., 0., 0., boundary_points[0][1], boundary_points[0][2], boundary_points[0][0],  1)
        sdlmath.draw_track_xy_from_points(ax_xy, pt, 0., 0., 0., boundary_points[1][1], boundary_points[1][2], boundary_points[1][0],  1)
        sdlmath.draw_track_xy_from_points(ax_xy, pt, 0., 0., 0., boundary_points[2][1], boundary_points[2][2], boundary_points[2][0],  1)
        sdlmath.draw_track_xy_from_points(ax_xy, pt, 0., 0., 0., boundary_points[3][1], boundary_points[3][2], boundary_points[3][0],  1)
        sdlmath.draw_track_rz_from_points(ax_rz, pt, 0., 0., 0., boundary_points[0][1], boundary_points[0][2], boundary_points[0][0],  1)
        sdlmath.draw_track_rz_from_points(ax_rz, pt, 0., 0., 0., boundary_points[1][1], boundary_points[1][2], boundary_points[1][0],  1)
        sdlmath.draw_track_rz_from_points(ax_rz, pt, 0., 0., 0., boundary_points[2][1], boundary_points[2][2], boundary_points[2][0],  1)
        sdlmath.draw_track_rz_from_points(ax_rz, pt, 0., 0., 0., boundary_points[3][1], boundary_points[3][2], boundary_points[3][0],  1)
        sdlmath.draw_track_xy_from_points(ax_xy, pt, 0., 0., 0., centroid[0], centroid[1], centroid[2],  1)
        sdlmath.draw_track_xy_from_points(ax_xy, pt, 0., 0., 0., centroid[0], centroid[1], centroid[2], -1)
        sdlmath.draw_track_rz_from_points(ax_rz, pt, 0., 0., 0., centroid[0], centroid[1], centroid[2],  1)
        sdlmath.draw_track_rz_from_points(ax_rz, pt, 0., 0., 0., centroid[0], centroid[1], centroid[2], -1)

        print(module.rod(), module.module())

        plt.sca(ax_xy)
        plt.savefig("temp/detxy_rod{}_module{}.pdf".format(module.rod(), module.module()))
        plt.sca(ax_rz)
        plt.savefig("temp/detrz_rod{}_module{}.pdf".format(module.rod(), module.module()))



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

    # pt = 1.1325262784957886
    # eta = -1.224777340888977
    # phi = 1.841907024383545
    # vx = -0.000797238782979548
    # vy = -0.0006373987998813391
    # vz = 3.2855265140533447)
    # charge = -1
    # Above track passes through the followig four modules
    # [437526553, 439620661, 438574117, 441206794]

    # Testing first module
    # detid = 437526553
    # Printing module information
    module = Module(437526553)
    # print(module) # detid=437526553 logicalLayer=1 side=1 moduleType=0 ring=0

    # Obtaining 4 boundary points
    boundary_points = np.array(det_geom.getData()[437526553])
    # print(boundary_points)
    # [[-37.264586548075904, -4.818972663825464, 27.865702430922006],
    # [-37.2642637125735, -14.216115063345738, 24.446095053032607],
    # [-39.764781970840225, -12.735559826177374, 20.377266643063948],
    # [-39.76510480634263, -3.3384174266571005, 23.796874020953346]]

    # Obtaining the centroid of the module
    centroid = np.array(centroidDB.getCentroid(437526553))
    # print(centroid) # [-8.77723, 24.1215, -38.5147] 

    # Normal vector obtain via cross vector between two vectors from centroid to two end points on a boundary
    vec1_on_module = centroid - np.array([boundary_points[0][1], boundary_points[0][2], boundary_points[0][0]])
    vec2_on_module = centroid - np.array([boundary_points[1][1], boundary_points[1][2], boundary_points[1][0]])
    norm_vec = np.cross(vec1_on_module, vec2_on_module)
    norm_vec = norm_vec / np.linalg.norm(norm_vec)

    def h(t):
        return sdlmath.get_track_point(1.1325262784957886, -1.224777340888977, 1.841907024383545, -0.000797238782979548, -0.0006373987998813391, 3.2855265140533447, -1, t)

    # # Debugging track drawing
    # points = np.array([ h(np.pi / 100. * i) for i in range(100) ])
    # ax_xy = pickle.load(file('/nfs-7/userdata/phchang/detector_layout_matplotlib_pickle/detxy.pickle'))
    # ax_xy.scatter(points[0:100,0], points[0:100,1])
    # plt.savefig("detxy.pdf")

    # Real points (approximately should be on the helix, barring *ASDF*(@(#$*()
    true_hits =  np.array(
    [[ -10.48864269,   24.35284233,  -38.05117416],    # rT = 26.51551537
     [ -28.04890823,   45.70554352,  -80.96523285],    # rT = 53.62590756
     [ -17.19555473,   34.00582504,  -56.25342941],    # rT = 38.10620999
     [ -38.91120148,   54.34824753, -102.4031601 ]])   # rT = 66.84170562

    n = norm_vec
    p0 = centroid

    print(true_hits[0])
    print(p0)
    print(n)

    print("(true hit-centroid).norm_vec:", np.dot(true_hits[0] - p0, n))

    res = optimize.minimize_scalar(lambda t: abs(np.dot(h(t) - p0, n)))

    # Obtain the intersection point
    intersection_point = h(res.x)
    print("intersection_point:", intersection_point)
    print("object to be minimized evaluated at (t_min):", abs(np.dot(h(res.x) - p0, n)))

    # "Display"-er with the module we want only
    sdlDisplay = SDLDisplay.getDefaultSDLDisplay() # Get an object to handle reading of centroids, and detector geometry etc. etc.
    sdlDisplay.set_detector_xy_collection([437526553]) # Turn on only the given modules in a list
    sdlDisplay.set_detector_rz_collection([437526553]) # Turn on only the given modules in a list

    # Create a figure and axes
    ax_xy = pickle.load(file('/nfs-7/userdata/phchang/detector_layout_matplotlib_pickle/detxy.pickle'))

    # Draw the given detector in xy-plane
    sdlDisplay.display_detector_xy(ax_xy)

    # Draw the intersection point
    ax_xy.scatter(true_hits[0:4,0], true_hits[0:4,1], s=0.2)
    ax_xy.scatter(intersection_point[0], intersection_point[1], s=0.2)

    # sdlmath.draw_track_xy(ax_xy, 1.1325262784957886, -1.224777340888977, 1.841907024383545, -0.000797238782979548, -0.0006373987998813391, 3.2855265140533447, -1)
    sdlmath.draw_track_xy_from_points(ax_xy, 1.1325262784957886, -0.000797238782979548, -0.0006373987998813391, 3.2855265140533447, -10.48864269, 24.35284233, -38.05117416, -1)

    print("saving figure")
    plt.savefig("detxy.pdf")

    # Load the axis in standard detector layout on rz
    ax_rz = pickle.load(file('/nfs-7/userdata/phchang/detector_layout_matplotlib_pickle/detrz.pickle'))

    # Draw the given detector in xy-plane (probaly won't change much)
    sdlDisplay.display_detector_rz(ax_rz, color=1)

    # Draw the intersection point
    ax_rz.scatter(true_hits[0:4,2], np.sqrt(true_hits[0:4,0]**2 + true_hits[0:4,1]**2), s=0.2)
    ax_rz.scatter(intersection_point[2], math.sqrt(intersection_point[0]**2 + intersection_point[1]**2), s=0.2)

    # sdlmath.draw_track_rz(ax_rz, 1.1325262784957886, -1.224777340888977, 1.841907024383545, -0.000797238782979548, -0.0006373987998813391, 3.2855265140533447, -1)
    sdlmath.draw_track_rz_from_points(ax_rz, 1.1325262784957886, -0.000797238782979548, -0.0006373987998813391, 3.2855265140533447, -10.48864269, 24.35284233, -38.05117416, -1)

    print("saving figure")
    plt.savefig("detrz.pdf")

    # fig_xy = pickle.load(file('/nfs-7/userdata/phchang/detector_layout_matplotlib_pickle/figxy.pickle'))
    # ax_xy = pickle.load(file('/nfs-7/userdata/phchang/detector_layout_matplotlib_pickle/detxy.pickle'))
    # fig_rz = pickle.load(file('/nfs-7/userdata/phchang/detector_layout_matplotlib_pickle/figrz.pickle'))
    # ax_rz = pickle.load(file('/nfs-7/userdata/phchang/detector_layout_matplotlib_pickle/detrz.pickle'))

    # rays = []
    # nslice = 60
    # for i in range(nslice):
    #     for j in range(nslice):
    #         phi = i * (2. * math.pi/nslice)
    #         eta = j * (2. * 0.5 / nslice) - 0.5
    #         sdlmath.draw_track_xy(ax_xy, 1000, eta, phi, 0, 0, 0, 1)
    #         sdlmath.draw_track_rz(ax_rz, 1000, eta, phi, 0, 0, 0, 1)

    # plt.sca(ax_xy)
    # plt.savefig("detxy.pdf")
    # plt.sca(ax_rz)
    # plt.savefig("detrz.pdf")



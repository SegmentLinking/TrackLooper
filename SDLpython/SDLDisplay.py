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

import sdlmath
from DetectorGeometry import DetectorGeometry
from Module import Module
from Centroid import Centroid

# Displayer

class SDLDisplay:

    def __init__(self, det_geom):
        self.det_geom = det_geom
        self.centroidDB = Centroid("data/centroid_2020_0428.txt")

    # def display_detector_xyz(self, ax, color=None):

    #     p = a3.art3d.Poly3DCollection(self.patches_xyz) #, cmap=matplotlib.cm.jet, alpha=0.4, facecolors=color)

    #     # # if color:
    #     # #     colors = np.ones(len(self.patches_xy)) * color
    #     # #     p.set_array(np.array(colors))

    #     ax.add_collection(p)
    #     # # ax.autoscale()
    #     # ax.set_ylim(-150, 150)
    #     # ax.set_xlim(-150, 150)

    def display_detector_etaphi(self, ax, color=None):

        p = PatchCollection(self.patches_etaphi, cmap=matplotlib.cm.jet, alpha=0.15, facecolors=color)

        ax.add_collection(p)
        # ax.autoscale()
        ax.set_xlim(-2.6, 2.6)
        # ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(-math.pi, math.pi)

    def display_detector_xy(self, ax, color=None):

        p = PatchCollection(self.patches_xy, cmap=matplotlib.cm.jet, alpha=0.4, facecolors=color)

        # if color:
        #     colors = np.ones(len(self.patches_xy)) * color
        #     p.set_array(np.array(colors))

        ax.add_collection(p)
        # ax.autoscale()
        ax.set_ylim(-150, 150)
        ax.set_xlim(-150, 150)

    def display_detector_rz(self, ax, color=None):

        p = PatchCollection(self.patches_rz, cmap=matplotlib.cm.jet, alpha=0.4, facecolors=color)

        # if color:
        #     colors = np.ones(len(self.patches_rz)) * color
        #     p.set_array(np.array(colors))

        ax.add_collection(p)
        # ax.autoscale()
        ax.set_ylim(0, 150)
        ax.set_xlim(-300, 300)

    def display_centroid_xy(self):

        f = open("data/centroid_2020_0421.txt")
        lines = f.readlines()

        # figure
        fig, ax = plt.subplots(figsize=(4,4))
        xs = []
        ys = []
        for line in lines:
            ls = line.split(",")
            if int(ls[5]) != 6:
                continue
            xs.append(ls[0])
            ys.append(ls[1])
        ax.scatter(xs, ys)

        ax.set_ylim(-150, 150)
        ax.set_xlim(-150, 150)

        fig.savefig("test.pdf")

    def get_six_faces(self, upper_module_points, lower_module_points):

        #    3----2
        #   /|   /|
        #  0----1 |
        #  | 3--|-2
        #  |/   |/
        #  0----1

        # One ordering should be correct

        min_uib = -1 # beginning index
        min_dist_sum = 999999999
        for i in range(4):
            temp_sum = 0
            for j in range(4):
                # print(j, i)
                # print(np.array(upper_module_points[j]))
                # print(np.array(lower_module_points[(j+i)%4]))
                # print(np.linalg.norm(np.array(upper_module_points[j]) - np.array(lower_module_points[(j+i)%4])))
                temp_sum += np.linalg.norm(np.array(upper_module_points[j]) - np.array(lower_module_points[(j+i)%4]))
            # print(temp_sum)
            if temp_sum < min_dist_sum:
                min_uib = i
                min_dist_sum = temp_sum
        # print(min_uib)
        # print(min_dist_sum)

        six_faces = [
            [upper_module_points[(0+min_uib)%4], lower_module_points[0], lower_module_points[1], upper_module_points[(1+min_uib)%4]],
            [upper_module_points[(1+min_uib)%4], lower_module_points[1], lower_module_points[2], upper_module_points[(2+min_uib)%4]],
            [upper_module_points[(2+min_uib)%4], lower_module_points[2], lower_module_points[3], upper_module_points[(3+min_uib)%4]],
            [upper_module_points[(3+min_uib)%4], lower_module_points[3], lower_module_points[0], upper_module_points[(0+min_uib)%4]],
            [upper_module_points[0], upper_module_points[1], upper_module_points[2], upper_module_points[3]],
            [lower_module_points[0], lower_module_points[1], lower_module_points[2], lower_module_points[3]],
            ]

        # print(six_faces)

        return six_faces

    # def set_detector_xyz_collection(self, list_of_detids):

    #     # Create detector patches
    #     self.patches_xyz = []
    #     n_failed = 0
    #     for detid in list_of_detids:

    #         module = Module(detid) 
    #         partner_detid = Module(detid).partnerDetId()

    #         if partner_detid not in self.det_geom.getData().keys():
    #             # print("{} not found from {} (isLower = {}, isInverted = {})".format(partner_detid, detid, module.isLower(), module.isInverted()))
    #             n_failed += 1
    #             continue
    #         else:
    #             # print("{}     found from {} (isLower = {}, isInverted = {})".format(partner_detid, detid, module.isLower(), module.isInverted()))
    #             pass

    #         six_faces = self.get_six_faces(self.det_geom.getData()[detid], self.det_geom.getData()[partner_detid])

    #         for face in six_faces:
    #             points = [ [x[0], x[1], x[2]] for x in face ]
    #             # polygon = a3.art3d.Poly3DCollection([np.array(points)])
    #             self.patches_xyz.append(np.array(points))

    #     # print(len(list_of_detids))
    #     # print(n_failed)

    def set_detector_etaphi_collection(self, list_of_detids):

        # Create detector patches
        self.patches_etaphi = []
        n_failed = 0
        for detid in list_of_detids:

            module = Module(detid) 

            bound_points = self.det_geom.getData()[detid]
            centroid = self.centroidDB.getCentroid(detid)

            points = []
            for bp in bound_points:
                x = bp[1]
                y = bp[2]
                z = bp[0] # The index is weird because that's how it is saved in det_geom
                refphi = math.atan2(centroid[1], centroid[0])
                eta, phi = sdlmath.get_etaphi([x, y, z], refphi)
                points.append([eta, phi+refphi])
                # eta, phi = sdlmath.get_etaphi([x, y, z])
                # points.append([eta, phi])
                # phi = math.atan2(y, x)
                # # print(x, y, phi)
                # # print(x, y, z)
                # eta = math.copysign(-math.log(math.tan(math.atan(math.sqrt(y**2+x**2) / abs(z)) / 2.)), z)

            polygon = Polygon(np.array(points), True)
            self.patches_etaphi.append(polygon)

        # print(len(list_of_detids))
        # print(n_failed)


    def set_detector_xy_collection(self, list_of_detids):

        # Create detector patches
        self.patches_xy = []
        n_failed = 0
        for detid in list_of_detids:

            module = Module(detid) 
            partner_detid = Module(detid).partnerDetId()

            if partner_detid not in self.det_geom.getData().keys():
                # print("{} not found from {} (isLower = {}, isInverted = {})".format(partner_detid, detid, module.isLower(), module.isInverted()))
                n_failed += 1
                continue
            else:
                # print("{}     found from {} (isLower = {}, isInverted = {})".format(partner_detid, detid, module.isLower(), module.isInverted()))
                pass

            six_faces = self.get_six_faces(self.det_geom.getData()[detid], self.det_geom.getData()[partner_detid])

            for face in six_faces:
                points = [ [x[1], x[2]] for x in face ]
                polygon = Polygon(np.array(points), True)
                self.patches_xy.append(polygon)

        # print(len(list_of_detids))
        # print(n_failed)

    def set_detector_rz_collection(self, list_of_detids):

        # Create detector patches
        self.patches_rz = []
        n_failed = 0
        for detid in list_of_detids:

            module = Module(detid) 
            partner_detid = Module(detid).partnerDetId()

            if partner_detid not in self.det_geom.getData().keys():
                # print("{} not found from {} (isLower = {}, isInverted = {})".format(partner_detid, detid, module.isLower(), module.isInverted()))
                n_failed += 1
                continue
            else:
                # print("{}     found from {} (isLower = {}, isInverted = {})".format(partner_detid, detid, module.isLower(), module.isInverted()))
                pass

            six_faces = self.get_six_faces(self.det_geom.getData()[detid], self.det_geom.getData()[partner_detid])

            for face in six_faces:
                points = [ [x[0], math.sqrt(x[1]**2+x[2]**2)] for x in face ]
                polygon = Polygon(np.array(points), True)
                self.patches_rz.append(polygon)

        # print(len(list_of_detids))
        # print(n_failed)

def getDefaultSDLDisplay():
    dirpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    det_geom = DetectorGeometry("{}/data/phase2_2020_0428.txt".format(dirpath))
    sdlDisplay = SDLDisplay(det_geom)
    list_of_detids = det_geom.getDetIds(lambda x: Module(x[0]).subdet() == 5 and Module(x[0]).side() == 3 and Module(x[0]).module() <= 2)
    list_of_detids.sort()
    sdlDisplay.set_detector_xy_collection(list_of_detids)
    list_of_detids = det_geom.getDetIds(lambda x: (Module(x[0]).subdet() == 5 and Module(x[0]).side() == 3 and Module(x[0]).rod() <= 2) or (Module(x[0]).subdet() == 5 and Module(x[0]).side() != 3 and Module(x[0]).module() <= 2) or (Module(x[0]).subdet() == 4 and Module(x[0]).module() <= 2))
    list_of_detids.sort()
    sdlDisplay.set_detector_rz_collection(list_of_detids)
    return sdlDisplay

##################################################################################################3
def test1():

    # figure
    fig, ax = plt.subplots(figsize=(4,4))

    sdlDisplay = getDefaultSDLDisplay()

    sdlDisplay.display_detector_xy(ax)

    fig.savefig("test1.pdf")

def test2():

    return

    # import mpl_toolkits.mplot3d as a3

    # ax = a3.Axes3D(pl.figure())

    # sdlDisplay = getDefaultSDLDisplay()
    # list_of_detids = det_geom.getDetIds(lambda x: Module(x[0]).subdet() == 5 and Module(x[0]).side() == 3 and Module(x[0]).module() <= 2)
    # list_of_detids.sort()
    # sdlDisplay.set_detector_xyz_collection(list_of_detids)

    # sdlDisplay.display_detector_xyz(ax)

    # plt.savefig("test2.pdf")

def test3():

    from Centroid import Centroid
    centroidDB = Centroid("data/centroid_2020_0428.txt")

    # figure
    # fig, ax = plt.subplots(figsize=(5.2,2.*math.pi))
    fig, ax = plt.subplots(figsize=(4. * 2,2.*math.pi))
    dirpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    det_geom = DetectorGeometry("{}/data/phase2_2020_0428.txt".format(dirpath))
    sdlDisplay = SDLDisplay(det_geom)
    # list_of_detids_etaphi = det_geom.getDetIds(lambda x: Module(x[0]).subdet() == 5 and Module(x[0]).side() == 3 and Module(x[0]).module() == 7 and Module(x[0]).layer() == 1 and Module(x[0]).isLower() == 1 and Module(x[0]).rod() == 1)
    layer = 1
    def get_etaphi(point):
        x, y, z = point
        phi = math.atan2(y, x)
        eta = math.copysign(-math.log(math.tan(math.atan(math.sqrt(y**2+x**2) / abs(z)) / 2.)), z)
        return (eta, phi)
    def deltaR(point1, point2):
        return np.linalg.norm(np.array(point1) - np.array(point2))
    list_of_detids_etaphi_layer1 = det_geom.getDetIds(lambda x: Module(x[0]).subdet() == 5 and Module(x[0]).side() == 3 and Module(x[0]).layer() == 1 and deltaR(get_etaphi(centroidDB.getCentroid(Module(x[0]).detId())), (0, 0)) < 0.1)
    list_of_detids_etaphi_layer2 = det_geom.getDetIds(lambda x: Module(x[0]).subdet() == 5 and Module(x[0]).side() == 3 and Module(x[0]).layer() == 2 and deltaR(get_etaphi(centroidDB.getCentroid(Module(x[0]).detId())), (0, 0)) < 0.1)
    list_of_detids_etaphi_layer3 = det_geom.getDetIds(lambda x: Module(x[0]).subdet() == 5 and Module(x[0]).side() == 3 and Module(x[0]).layer() == 3 and deltaR(get_etaphi(centroidDB.getCentroid(Module(x[0]).detId())), (0, 0)) < 0.1)
    list_of_detids_etaphi_layer4 = det_geom.getDetIds(lambda x: Module(x[0]).subdet() == 5 and Module(x[0]).side() == 3 and Module(x[0]).layer() == 4 and deltaR(get_etaphi(centroidDB.getCentroid(Module(x[0]).detId())), (0, 0)) < 0.1)
    list_of_detids_etaphi_layer5 = det_geom.getDetIds(lambda x: Module(x[0]).subdet() == 5 and Module(x[0]).side() == 3 and Module(x[0]).layer() == 5 and deltaR(get_etaphi(centroidDB.getCentroid(Module(x[0]).detId())), (0, 0)) < 0.1)
    list_of_detids_etaphi_layer6 = det_geom.getDetIds(lambda x: Module(x[0]).subdet() == 5 and Module(x[0]).side() == 3 and Module(x[0]).layer() == 6 and deltaR(get_etaphi(centroidDB.getCentroid(Module(x[0]).detId())), (0, 0)) < 0.1)
    sdlDisplay.set_detector_etaphi_collection(list_of_detids_etaphi_layer1)
    sdlDisplay.display_detector_etaphi(ax, color=(1,0,0))
    sdlDisplay.set_detector_etaphi_collection(list_of_detids_etaphi_layer2)
    sdlDisplay.display_detector_etaphi(ax, color=(1,1,0))
    sdlDisplay.set_detector_etaphi_collection(list_of_detids_etaphi_layer3)
    sdlDisplay.display_detector_etaphi(ax, color=(1,0,1))
    sdlDisplay.set_detector_etaphi_collection(list_of_detids_etaphi_layer4)
    sdlDisplay.display_detector_etaphi(ax, color=(0,1,1))
    sdlDisplay.set_detector_etaphi_collection(list_of_detids_etaphi_layer5)
    sdlDisplay.display_detector_etaphi(ax, color=(0,0,1))
    sdlDisplay.set_detector_etaphi_collection(list_of_detids_etaphi_layer6)
    sdlDisplay.display_detector_etaphi(ax, color=(0,0,0))
    fig.savefig("test3.pdf")

if __name__ == "__main__":

    test3()



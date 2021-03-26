#!/bin/env python

import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import pickle
import numpy as np
import math
from scipy import optimize
from shapely.geometry import Polygon

class Helix:
    def __init__(self, center, radius, phi, lam, charge):
        self.center_ = center
        self.radius_ = radius
        self.phi_ = self.Phi_mpi_pi(phi)
        self.lam_ = lam
        self.charge_ = charge
    def center(self): return self.center_
    def radius(self): return self.radius_
    def phi(self): return self.phi_
    def lam(self): return self.lam_
    def charge(self): return self.charge_
    def __str__(self):
        rtnstr = [
        "Helix():",
        "  center = {}".format(self.center()),
        "  radius = {}".format(self.radius()),
        "  phi    = {}".format(self.phi()),
        "  lam    = {}".format(self.lam()),
        "  charge = {}".format(self.charge()),
        ]
        return "\n".join(rtnstr)
    def Phi_mpi_pi(self, phi):
        f = phi
        while f >= math.pi: f -= 2. * math.pi;
        while f < -math.pi: f += 2. * math.pi;
        return f

def Phi_mpi_pi(phi):
    f = phi
    while f >= math.pi: f -= 2. * math.pi;
    while f < -math.pi: f += 2. * math.pi;
    return f

def get_helix_point(helix, t):
    x = helix.center()[0] - helix.charge() * helix.radius() * np.sin(helix.phi() - (helix.charge()) * t)
    y = helix.center()[1] + helix.charge() * helix.radius() * np.cos(helix.phi() - (helix.charge()) * t)
    z = helix.center()[2] +                  helix.radius() * np.tan(helix.lam()) * t
    r = np.sqrt(x**2 + y**2)
    return (x, y, z, r)

def get_helix_point_from_radius(helix, r):
    def h(t):
        x = helix.center()[0] - helix.charge() * helix.radius() * np.sin(helix.phi() - (helix.charge()) * t)
        y = helix.center()[1] + helix.charge() * helix.radius() * np.cos(helix.phi() - (helix.charge()) * t)
        return math.sqrt(x**2 + y**2)
    res = optimize.minimize_scalar(lambda t: abs(h(t) - r), bounds=(0, math.pi), method='bounded')
    t = res.x
    x = helix.center()[0] - helix.charge() * helix.radius() * np.sin(helix.phi() - (helix.charge()) * t)
    y = helix.center()[1] + helix.charge() * helix.radius() * np.cos(helix.phi() - (helix.charge()) * t)
    z = helix.center()[2] +                  helix.radius() * np.tan(helix.lam()) * t
    r = np.sqrt(x**2 + y**2)
    return (x, y, z, r)

def get_helix_point_from_z(helix, z):
    def h(t):
        z_ = helix.center()[2] +                  helix.radius() * np.tan(helix.lam()) * t
        return z_
    res = optimize.minimize_scalar(lambda t: abs(h(t) - z), bounds=(0, math.pi), method='bounded')
    t = res.x
    x = helix.center()[0] - helix.charge() * helix.radius() * np.sin(helix.phi() - (helix.charge()) * t)
    y = helix.center()[1] + helix.charge() * helix.radius() * np.cos(helix.phi() - (helix.charge()) * t)
    z = helix.center()[2] +                  helix.radius() * np.tan(helix.lam()) * t
    r = np.sqrt(x**2 + y**2)
    return (x, y, z, r)

def get_helix_points(helix):
    xs = []
    ys = []
    zs = []
    rs = []
    for t in np.linspace(0, 2.*np.pi, 1000):
        x, y, z, r = get_helix_point(helix, t)
        if r > 120:
            break
        xs.append(x)
        ys.append(y)
        zs.append(z)
        rs.append(r)
    xs = np.array(xs)
    ys = np.array(ys)
    zs = np.array(zs)
    rs = np.array(rs)
    return xs, ys, zs, rs

def get_square_point_at_radius_from_helices(helices, r, t1, t2):
    ''' t1 and t2 are the two parameters to define the point on square. Assumes
    helices are length 4 and also in order to create a square going from
    0->1->2->3->0'''
    point1 = get_helix_point_from_radius(helices[0], r)
    point2 = get_helix_point_from_radius(helices[1], r)
    point3 = get_helix_point_from_radius(helices[2], r)
    point4 = get_helix_point_from_radius(helices[3], r)


def construct_helix_from_kinematics(pt, eta, phi, vx, vy, vz, charge):

    print(pt, eta, phi, vx, vy, vz, charge)

    # Radius based on pt
    radius = pt / (2.99792458e-3 * 3.8)

    # reference point vector which for sim track is the vertex point
    ref_vec = np.array([vx, vy, vz]) # reference point vector

    # The reference to center vector
    inward_radial_vec = charge * radius * np.array([math.sin(phi), -math.cos(phi), 0]) # reference point to center vector

    # The center point
    center_vec = ref_vec + inward_radial_vec # center of the helix

    # The lambda
    lam = math.copysign(math.pi/2. - 2. * math.atan(math.exp(-abs(eta))), eta) # lambda

    return Helix(center_vec, radius, phi, lam, charge)

def construct_helix_from_points(pt, vx, vy, vz, mx, my, mz, charge):
    '''Clarification : phi was derived assuming a negatively charged particle would start
    at the first quadrant. However the way signs are set up in the get_track_point function
    implies the particle actually starts out in the fourth quadrant, and phi is measured from
    the y axis as opposed to x axis in the expression provided in this function. Hence I tucked
    in an extra pi/2 to account for these effects'''
    # print(pt,vx,vy,vz,mx,my,mz,charge)

    radius = pt / (2.99792458e-3 * 3.8)
    R = abs(radius) #For geometrical calculations

    t = 2 * np.arcsin(np.sqrt( (vx - mx) **2 + (vy - my) **2 )/(2*R))
    phi = np.pi/2 + np.arctan((vy-my)/(vx-mx)) + ((vy-my)/(vx-mx) < 0) * (np.pi) +charge *  t/2 + (my-vy < 0) * (np.pi/2) - (my-vy > 0) * (np.pi/2)
    cx = vx + charge *  radius * np.sin(phi)
    cy = vy - charge *  radius * np.cos(phi)
    cz = vz
    lam = np.arctan((mz - vz)/( radius * t))

    return Helix(np.array([cx,cy,cz]), radius, phi, lam, charge)

def get_etaphi(point, refphi=0):
    x, y, z = point
    if refphi != 0:
        xnew = x * math.cos(-refphi) - y * math.sin(-refphi)
        ynew = x * math.sin(-refphi) + y * math.cos(-refphi)
        x = xnew
        y = ynew
    phi = math.atan2(y, x)
    eta = math.copysign(-math.log(math.tan(math.atan(math.sqrt(y**2+x**2) / abs(z)) / 2.)), z)
    return (eta, phi)

def deltaR(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

# # Deprecated
# def etaphi_point_within_boundary(point, tar_mod_boundaries): # inputs are meant to be in eta, phi space
#     vec = [np.array(tar_mod_boundaries[0]) - np.array(point),
#            np.array(tar_mod_boundaries[1]) - np.array(point),
#            np.array(tar_mod_boundaries[2]) - np.array(point),
#            np.array(tar_mod_boundaries[3]) - np.array(point)]
#     vec_signs = [ (int(x[0] > 0) - int(x[0] < 0), int(x[1] > 0) - int(x[1] < 0)) for x in vec ]
#     summed = np.array(vec_signs).sum(0)
#     if summed[0] == 0 and summed[1] == 0:
#         return True
#     else:
#         return False

# Deprecated
def etaphi_point_within_boundary(point, tar_mod_boundaries): # inputs are meant to be in eta, phi space
    vec = [np.array(tar_mod_boundaries[0]) - np.array(point),
           np.array(tar_mod_boundaries[1]) - np.array(point),
           np.array(tar_mod_boundaries[2]) - np.array(point),
           np.array(tar_mod_boundaries[3]) - np.array(point)]
    vec_signs = [ (int(x[0] > 0) - int(x[0] < 0), int(x[1] > 0) - int(x[1] < 0)) for x in vec ]
    summed = np.array(vec_signs).sum(0)
    if summed[0] == 0 and summed[1] == 0:
        return True
    else:
        return False

def module_etaphi_within_boundary_etaphi(ref_mod_boundaries, tar_mod_boundaries): # inputs are meant to be in 3d in the format of [x, y, z]
    for point in ref_mod_boundaries:
        if etaphi_point_within_boundary(point, tar_mod_boundaries):
            return True
    return False

def get_etaphi_polygon(mod_boundaries, refphi, zshift=0):
    mod_boundaries = np.array([ get_etaphi([x[1], x[2], x[0]+zshift], refphi) for x in mod_boundaries ])
    return Polygon(mod_boundaries)

def module_overlaps_in_eta_phi(ref_mod_boundaries, tar_mod_boundaries, refphi=0, zshift=0, verbose=False):
    ref_center = np.array(ref_mod_boundaries).sum(0) / 4
    tar_center = np.array(tar_mod_boundaries).sum(0) / 4
    ref_center_phi = math.atan2(ref_center[2], ref_center[1])
    tar_center_phi = math.atan2(tar_center[2], tar_center[1])
    if abs(Phi_mpi_pi(ref_center_phi-tar_center_phi)) > math.pi / 2:
        return False
    # Turn it into eta phi
    ref_mod_boundaries = np.array([ get_etaphi([x[1], x[2], x[0]+zshift], refphi) for x in ref_mod_boundaries ])
    tar_mod_boundaries = np.array([ get_etaphi([x[1], x[2], x[0]+zshift], refphi) for x in tar_mod_boundaries ])
    # quick cut
    diff = ref_mod_boundaries[0] - tar_mod_boundaries[0]
    if abs(diff[0]) > 0.5:
        return False
    if abs(Phi_mpi_pi(diff[1])) > 1:
        return False
    p1 = Polygon(ref_mod_boundaries)
    p2 = Polygon(tar_mod_boundaries)
    if verbose:
        print(p1.intersects(p2))
    return p1.intersects(p2)

def module_etaphi_within_boundary_zxy(ref_mod_boundaries, tar_mod_boundaries, refphi=0, verbose=False): # inputs are meant to be in 3d in the format of [z, x, y] DetGeom is in this order
    ref_mod_boundaries = [ get_etaphi([x[1], x[2], x[0]], refphi) for x in ref_mod_boundaries ]
    tar_mod_boundaries = [ get_etaphi([x[1], x[2], x[0]], refphi) for x in tar_mod_boundaries ]
    for point in ref_mod_boundaries:
        if verbose:
            print(point)
            print(tar_mod_boundaries)
        if etaphi_point_within_boundary(point, tar_mod_boundaries):
            if verbose:
                print("found")
            return True
    if verbose:
        print("not found")
    return False

def point_on_square(point, square_boundaries): # inputs are meant to be in 3d in the format of [z, x, y] DetGeom is in this order
    bounds = [ [x[1], x[2], x[0]] for x in square_boundaries ] # in principle i shouldn't be doing this here in this function
    bounds = np.array(bounds)
    edge_vector_1 = np.array(bounds[0] - bounds[1])
    edge_vector_2 = np.array(bounds[2] - bounds[1])
    point_relative_vector = np.array(point - bounds[1])
    edge_norm_1 = np.linalg.norm(edge_vector_1)
    edge_norm_2 = np.linalg.norm(edge_vector_2)
    edge_vector_1 = edge_vector_1 / edge_norm_1
    edge_vector_2 = edge_vector_2 / edge_norm_2
    point_dot_edge_1 = np.dot(point_relative_vector, edge_vector_1)
    point_dot_edge_2 = np.dot(point_relative_vector, edge_vector_2)
    print point_relative_vector
    print edge_vector_1
    print edge_vector_2
    print point_dot_edge_1
    print point_dot_edge_2
    if point_dot_edge_1 >= 0 and point_dot_edge_1 <= edge_norm_1 and point_dot_edge_2 >= 0 and point_dot_edge_2 <= edge_norm_2:
        return True
    else:
        return False

def point_helix_intersection(helix, norm_vec, centroid):

    def h(t):
        return np.array(get_helix_point(helix, t)[0:3])

    n = np.array(norm_vec)
    p0 = np.array(centroid)

    # print(true_hits[0])
    # print(p0)
    # print(n)

    # print("(true hit-centroid).norm_vec:", np.dot(true_hits[0] - p0, n))

    res = optimize.minimize_scalar(lambda t: abs(np.dot(h(t) - p0, n)))

    # Obtain the intersection point
    intersection_point = h(res.x)

    return intersection_point

def helix_intersects_module(helix, square_boundaries, centroid):
    bounds = [ [x[1], x[2], x[0]] for x in square_boundaries ] # in principle i shouldn't be doing this here in this function
    bounds = np.array(bounds)
    edge_vector_1 = np.array(bounds[0] - bounds[1])
    edge_vector_2 = np.array(bounds[2] - bounds[1])
    norm_vec = np.cross(edge_vector_1, edge_vector_2)
    norm_vec = norm_vec / np.linalg.norm(norm_vec)
    print "computing intersection"
    print norm_vec
    print centroid
    point = point_helix_intersection(helix, norm_vec, centroid)
    print point
    print bounds
    if point_on_square(point, square_boundaries):
        return (True, point)
    else:
        return (False, np.zeros(3))

def draw_track_xy(ax, pt, eta, phi, vx, vy, vz, charge, verbose=False):
    if verbose:
        print("draw_track_xy: pt, eta, phi, vx, vy, vz, charge = ", pt, eta, phi, vx, vy, vz, charge)
    helix = construct_helix_from_kinematics(pt, eta, phi, vx, vy, vz, charge)
    print(helix)
    xs, ys, zs, rs = get_helix_points(helix)
    # ax.scatter(helix.center()[0], helix.center()[1])
    ax.plot(xs, ys, linewidth=0.2, color=(1,0,0))

def draw_track_rz(ax, pt, eta, phi, vx, vy, vz, charge, verbose=False):
    if verbose:
        print("draw_track_rz: pt, eta, phi, vx, vy, vz, charge = ", pt, eta, phi, vx, vy, vz, charge)
    helix = construct_helix_from_kinematics(pt, eta, phi, vx, vy, vz, charge)
    print(helix)
    xs, ys, zs, rs = get_helix_points(helix)
    ax.plot(zs, rs, linewidth=0.2, color=(1,0,0))

def draw_track_xy_from_points(ax, pt, vx, vy, vz, mx, my, mz, charge, verbose=False):
    if verbose:
        print("draw_track_xy_from_points: pt, vx, vy, vz, mx, my, mz, charge = ", pt, vx, vy, vz, mx, my, mz, charge)
    helix = construct_helix_from_points(pt, vx, vy, vz, mx, my, mz, charge)
    print(helix)
    xs, ys, zs, rs = get_helix_points(helix)
    # ax.scatter(helix.center()[0], helix.center()[1], s=0.4)
    ax.plot(xs, ys, linewidth=0.2, color=(1,0,0))

def draw_track_rz_from_points(ax, pt, vx, vy, vz, mx, my, mz, charge, verbose=False):
    if verbose:
        print("draw_track_rz_from_points: pt, vx, vy, vz, mx, my, mz, charge = ", pt, vx, vy, vz, mx, my, mz, charge)
    helix = construct_helix_from_points(pt, vx, vy, vz, mx, my, mz, charge)
    print(helix)
    xs, ys, zs, rs = get_helix_points(helix)
    ax.plot(zs, rs, linewidth=0.2, color=(1,0,0))

#_____________________________________________________________
# Good barrel tracks is where at least one sim hit with correct pdgid land on each layer
# Does not require pair of hits land on each layer.
# Does not require that the paired hits land on module pairs.
# Does not care whether a single layer has 4 hits
# Only one sim hit with correct pdgid is needed per layer to pass the requirement
# Input: TTree event, and sim trk index
def goodBarrelTracks(t, simtrk_idx, pdgid=0):

    # List of layer index with the simhit with correct pdgid
    # Check this later to get the list
    layer_idx_with_hits = []

    # Loop over the sim hit index
    for simhitidx in t.sim_simHitIdx[simtrk_idx]:

        # If not a correct sim hit skip
        if t.simhit_particle[simhitidx] != t.sim_pdgId[simtrk_idx]:
            continue

        # Check it is barrel
        if t.simhit_subdet[simhitidx] != 5:
            continue

        # If pdgId condition is called require the pdgid
        if pdgid:
            if abs(t.sim_pdgId[simtrk_idx]) != abs(pdgid):
                continue

        # Add the layer index
        layer_idx_with_hits.append(t.simhit_layer[simhitidx])

    if sorted(list(set(layer_idx_with_hits))) == [1, 2, 3, 4, 5, 6]:
        return True
    else:
        return False

def getCenterFromThreePoints(hitA, hitB, hitC):

    # //       C
    # //
    # //
    # //
    # //    B           d <-- find this point that makes the arc that goes throw a b c
    # //
    # //
    # //     A

    # // Steps:
    # // 1. Calculate mid-points of lines AB and BC
    # // 2. Find slopes of line AB and BC
    # // 3. construct a perpendicular line between AB and BC
    # // 4. set the two equations equal to each other and solve to find intersection

    xA = hitA[0]
    yA = hitA[1]
    xB = hitB[0]
    yB = hitB[1]
    xC = hitC[0]
    yC = hitC[1]

    x_mid_AB = (xA + xB) / 2.
    y_mid_AB = (yA + yB) / 2.

    x_mid_BC = (xB + xC) / 2.
    y_mid_BC = (yB + yC) / 2.

    slope_AB_inf = (xB - xA) == 0
    slope_BC_inf = (xC - xB) == 0

    slope_AB_zero = (yB - yA) == 0
    slope_BC_zero = (yC - yB) == 0

    slope_AB = 0. if slope_AB_inf else (yB - yA) / (xB - xA)
    slope_BC = 0. if slope_BC_inf else (yC - yB) / (xC - xB)

    slope_perp_AB = 0. if (slope_AB_inf or slope_AB_zero) else -1. / (slope_AB)
    slope_perp_BC = 0. if (slope_BC_inf or slope_BC_zero) else -1. / (slope_BC)

    # if ((slope_AB - slope_BC) == 0):
    #     std::cout <<  " slope_AB_zero: " << slope_AB_zero <<  std::endl;
    #     std::cout <<  " slope_BC_zero: " << slope_BC_zero <<  std::endl;
    #     std::cout <<  " slope_AB_inf: " << slope_AB_inf <<  std::endl;
    #     std::cout <<  " slope_BC_inf: " << slope_BC_inf <<  std::endl;
    #     std::cout <<  " slope_AB: " << slope_AB <<  std::endl;
    #     std::cout <<  " slope_BC: " << slope_BC <<  std::endl;
    #     std::cout << hitA << std::endl;
    #     std::cout << hitB << std::endl;
    #     std::cout << hitC << std::endl;
    #     std::cout << "SDL::MathUtil::getCenterFromThreePoints() function the three points are in straight line!" << std::endl;
    #     return SDL::Hit();

    x = (slope_AB * slope_BC * (yA - yC) + slope_BC * (xA + xB) - slope_AB * (xB + xC)) / (2. * (slope_BC - slope_AB));
    y = slope_perp_AB * (x - x_mid_AB) + y_mid_AB;

    return [x, y, 0]

if __name__ == "__main__":

    import ROOT as r
    import sys

    f = r.TFile("/home/users/phchang/public_html/analysis/sdl/trackingNtuple/CMSSW_10_4_0/src/trackingNtuple_100_pt0p5_2p0.root")
    # f = r.TFile("/nfs-7/userdata/phchang/trackingNtuple/trackingNtuple_10_pt0p5_50_50cm_cube.root")
    tree = f.Get("trackingNtuple").Get("tree")

    ntrk = 0
    itrk_sel = int(sys.argv[1])
    for index, event in enumerate(tree):
        for itrk, dxy in enumerate(event.sim_pca_dxy):

            if not (index == 0 and itrk == itrk_sel):
                continue

            ntrk += 1
            pt = event.sim_pt[itrk]
            eta = event.sim_eta[itrk]
            phi = event.sim_phi[itrk]
            dxy = event.sim_pca_dxy[itrk]
            dz = event.sim_pca_dz[itrk]
            charge = event.sim_q[itrk]
            vx = event.simvtx_x[0]
            vy = event.simvtx_y[0]
            vz = event.simvtx_z[0]
            xs = []
            ys = []
            zs = []
            for isimhit in event.sim_simHitIdx[itrk]:
                if event.simhit_subdet[isimhit] != 4 and event.simhit_subdet[isimhit] != 5:
                    continue
                if abs(event.simhit_particle[isimhit]) != 13:
                    continue
                xs.append(event.simhit_x[isimhit])
                ys.append(event.simhit_y[isimhit])
                zs.append(event.simhit_z[isimhit])

            print(vx, vy, vz)

            print("Track info read from the TTree:")
            print(index, itrk)

            ax_xy = pickle.load(file('/nfs-7/userdata/phchang/detector_layout_matplotlib_pickle/detxy.pickle'))
            draw_track_xy(ax_xy, pt, eta, phi, vx, vy, vz, charge, verbose=True)
            draw_track_xy_from_points(ax_xy, pt, vx, vy, vz, xs[0], ys[0], zs[0], charge, verbose=True)
            plt.scatter(xs, ys, s=0.1)
            helix = construct_helix_from_kinematics(pt, eta, phi, vx, vy, vz, charge)
            for r in [20, 30, 50, 80]:
                point_on_radius = get_helix_point_from_radius(helix, r)
                plt.scatter(point_on_radius[0], point_on_radius[1])
            plt.savefig("detxy.pdf")

            # ax_rz = pickle.load(file('/nfs-7/userdata/phchang/detector_layout_matplotlib_pickle/detrz.pickle'))
            # draw_track_rz(ax_rz, pt, eta, phi, vx, vy, vz, charge, verbose=True)
            # # draw_track_rz_from_points(ax_rz, pt, vx, vy, vz, xs[0], ys[0], zs[0], charge, verbose=True)
            # plt.scatter(zs, np.sqrt(np.array(xs)**2+np.array(ys)**2), s=0.1)
            # plt.savefig("detrz.pdf")

            if index == 0 and itrk == itrk_sel:
                sys.exit()


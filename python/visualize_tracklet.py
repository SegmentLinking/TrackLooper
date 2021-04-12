#!/bin/env python

import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import pickle
import numpy as np
import math
import sdlmath
import SDLDisplay
from tqdm import tqdm

def get_circle(hits):

    # https://stackoverflow.com/questions/28910718/give-3-points-and-a-plot-circle
    # x, y, z = 0+1j, 1+0j, 0-1j
    # w = z-x
    # w /= y-x
    # c = (x-y)*(w-abs(w)**2)/2j/w.imag-x
    # print '(x%+.3f)^2+(y%+.3f)^2 = %.3f^2' % (c.real, c.imag, abs(c+x))

    x, y, z = hits[0][0] + hits[0][1] * 1j, hits[1][0] + hits[1][1] * 1j, hits[2][0] + hits[2][1] * 1j
    w = z-x
    w /= y-x
    c = (x-y)*(w-abs(w)**2)/2j/w.imag-x
    return c.real, c.imag, abs(c+x)

def draw_tracklet_xy(ax_xy, hits, useRecoStyle, drawChords=True):
    hits_xs = hits[0:4,0]
    hits_ys = hits[0:4,1]
    hits_zs = hits[0:4,2]
    hits_rs = hits[0:4,3]
    xy_segments = [[ (hits[0][0:2]), (hits[2][0:2]) ], [ (hits[1][0:2]), (hits[3][0:2]) ]]
    rz_segments = [[ (hits[0][2:4]), (hits[2][2:4]) ], [ (hits[1][2:4]), (hits[3][2:4]) ]]
    chord_xy_segments = [[ (hits[0][0:2]), (hits[3][0:2]) ]]
    chord_rz_segments = [[ (hits[0][0:2]), (hits[3][0:2]) ]]
    if useRecoStyle:
        ax_xy.add_collection(LineCollection(xy_segments, colors=(0,0,1), linewidth=0.5, alpha=0.4))
        if drawChords:
            ax_xy.add_collection(LineCollection(chord_xy_segments, colors=(0,0,0), linewidth=0.5, alpha=0.4, linestyle=':'))
        ax_xy.scatter(hits_xs, hits_ys, s=2, marker='o', c=(1/255., 132/255., 72/255.))
    else:
        ax_xy.add_collection(LineCollection(xy_segments, colors=(1,0,0), linewidth=0.5, alpha=0.4))
        ax_xy.add_collection(LineCollection(chord_xy_segments, colors=(0,0,0), linewidth=0.5, alpha=0.4))
        ax_xy.scatter(hits_xs, hits_ys, s=2, marker='o', c=(1,0,0))
    window = abs(hits_xs[0] + hits_xs[3]) if abs(hits_xs[0] + hits_xs[3]) > abs(hits_ys[0] + hits_ys[3]) else abs(hits_ys[0] + hits_ys[3])
    ax_xy.set_xlim((hits_xs[0] + hits_xs[3])/2. - 0.55 * window, (hits_xs[0] + hits_xs[3])/2. + 0.55 * window)
    ax_xy.set_ylim((hits_ys[0] + hits_ys[3])/2. - 0.55 * window, (hits_ys[0] + hits_ys[3])/2. + 0.55 * window)

def draw_tracklet_rz(ax_rz, hits, useRecoStyle):
    hits_xs = hits[0:4,0]
    hits_ys = hits[0:4,1]
    hits_zs = hits[0:4,2]
    hits_rs = hits[0:4,3]
    rz_segments = [[ (hits[0][0:2]), (hits[2][0:2]) ], [ (hits[1][0:2]), (hits[3][0:2]) ]]
    rz_segments = [[ (hits[0][2:4]), (hits[2][2:4]) ], [ (hits[1][2:4]), (hits[3][2:4]) ]]
    chord_rz_segments = [[ (hits[0][0:2]), (hits[3][0:2]) ]]
    chord_rz_segments = [[ (hits[0][0:2]), (hits[3][0:2]) ]]
    if useRecoStyle:
        ax_rz.add_collection(LineCollection(rz_segments, colors=(0,0,1), linewidth=0.5, alpha=0.4))
        ax_rz.scatter(hits_zs, hits_rs, s=2, marker='o', c=(1/255., 132/255., 72/255.))
    else:
        ax_rz.add_collection(LineCollection(rz_segments, colors=(1,0,0), linewidth=0.5, alpha=0.4))
        ax_rz.scatter(hits_zs, hits_rs, s=2, marker='o', c=(1,0,0))
    window = abs(hits_rs[0] + hits_rs[3]) if abs(hits_rs[0] + hits_rs[3]) > abs(hits_zs[0] + hits_zs[3]) else abs(hits_zs[0] + hits_zs[3])
    ax_rz.set_xlim((hits_zs[0] + hits_zs[3])/2. - 0.55 * window * 2.5, (hits_zs[0] + hits_zs[3])/2. + 0.55 * window * 2.5)
    ax_rz.set_ylim((hits_rs[0] + hits_rs[3])/2. - 0.55 * window *   1, (hits_rs[0] + hits_rs[3])/2. + 0.55 * window *   1)

if __name__ == "__main__":

    import ROOT as r

    # f = r.TFile("/home/users/phchang/public_html/analysis/sdl/TrackLooper_/results/tracklet_study/pt0p5_2p0_2020_0430_1449/fulleff_pt0p5_2p0.root")
    f = r.TFile("/home/users/phchang/public_html/analysis/sdl/TrackLooper_/results/tracklet_study/pu200_2020_0501_1123/fulleff_pu200_0.root")
    t = f.Get("tree")

    # ax_xy = pickle.load(file('/nfs-7/userdata/phchang/detector_layout_matplotlib_pickle/detxy.pickle'))
    # sdlmath.draw_track_xy(ax_xy, pt, eta, phi, vx, vy, vz, charge)
    # sdlDisplay.display_detector_xy(ax_xy)
    # if len(true_hits):
    #     draw_tracklet_xy(ax_xy, true_hits, useRecoStyle=False)
    # draw_tracklet_xy(ax_xy, reco_hits, useRecoStyle=True)

    # plt.savefig("detxy.pdf")

    # ax_xy = pickle.load(file('/nfs-7/userdata/phchang/detector_layout_matplotlib_pickle/detxy.pickle'))
    # ax_rz = pickle.load(file('/nfs-7/userdata/phchang/detector_layout_matplotlib_pickle/detrz.pickle'))
    # sdlmath.draw_track_rz(ax_rz, pt, eta, phi, vx, vy, vz, charge)
    # sdlDisplay.display_detector_rz(ax_rz)
    # if len(true_hits):
    #     draw_tracklet_rz(ax_rz, true_hits, useRecoStyle=False)
    # draw_tracklet_rz(ax_rz, reco_hits, useRecoStyle=True)

    # plt.savefig("detrz.pdf")

    # ax_xy = pickle.load(file('/nfs-7/userdata/phchang/detector_layout_matplotlib_pickle/detxy.pickle'))

    # for index, event in enumerate(tqdm(t)):

    #     if event.category != 0:
    #         continue

    #     if index > 10000:
    #         break

    #     reco_hits = np.array([ [x, y, z, math.sqrt(x**2 + y**2)] for x, y, z in zip(event.x, event.y, event.z) ])

    #     draw_tracklet_xy(ax_xy, reco_hits, useRecoStyle=True, drawChords=False)

    # ax_xy.set_xlim(-150, 150)
    # ax_xy.set_ylim(-150, 150)

    # plt.savefig("detxy.pdf")

    for event in t:
        if event.category != 0:
            continue
        # if event.matched_trk_pt < 0:
        if abs(event.matched_trk_pt - 1.1) > 0.05:
            continue
        # if event.nPS != 0:
        #     continue

        # if event.betaIn[0] * event.betaOut[0] > 0:
        #     continue

        if len(event.true_x) < 4:
            continue
        # if len(event.true_x) >= 4:
        #     continue

        reco_hits = np.array([ [x, y, z, math.sqrt(x**2 + y**2)] for x, y, z in zip(event.x, event.y, event.z) ])

        true_hits = []
        if len(event.true_x) >= 4:
            true_hits = np.array([ [x, y, z, math.sqrt(x**2 + y**2)] for x, y, z in zip(event.true_x, event.true_y, event.true_z) ])

        detids = [ int(x) for x in event.module_detId ]

        pt = event.matched_trk_pt
        phi = event.matched_trk_phi
        eta = event.matched_trk_eta
        vx = event.simvtx_x
        vy = event.simvtx_y
        vz = event.simvtx_z
        charge = event.matched_trk_charge

        print(pt, phi, eta, vx, vy, vz, charge)
        print(pt, eta, phi, vx, vy, vz, charge)
        print(detids)
        print(true_hits)

        print(event.betaIn)
        print(event.betaOut)
        print(event.dBeta)

        print(pt)
        print(np.linalg.norm(reco_hits[3][0:2] - reco_hits[0][0:2]) * 2.99792458e-3 * 3.8 / 2. / np.sin(event.betaIn))
        print(np.linalg.norm(reco_hits[3][0:2] - reco_hits[0][0:2]) * 2.99792458e-3 * 3.8 / 2. / np.sin(event.betaOut))

        h, k, radius = get_circle([reco_hits[0], reco_hits[2], reco_hits[3]])
        print(h, k)
        print(2.99792458e-3 * 3.8 * radius)
        h, k, radius = get_circle([reco_hits[0], reco_hits[1], reco_hits[3]])
        print(h, k)
        print(2.99792458e-3 * 3.8 * radius)

        sdlDisplay = SDLDisplay.getDefaultSDLDisplay()

        sdlDisplay.set_detector_xy_collection(detids)
        sdlDisplay.set_detector_rz_collection(detids)

        ax_xy = pickle.load(file('/nfs-7/userdata/phchang/detector_layout_matplotlib_pickle/detxy.pickle'))
        sdlmath.draw_track_xy(ax_xy, pt, eta, phi, vx, vy, vz, charge)
        sdlDisplay.display_detector_xy(ax_xy)
        if len(true_hits):
            draw_tracklet_xy(ax_xy, true_hits, useRecoStyle=False)
        draw_tracklet_xy(ax_xy, reco_hits, useRecoStyle=True)

        plt.savefig("detxy.pdf")

        ax_rz = pickle.load(file('/nfs-7/userdata/phchang/detector_layout_matplotlib_pickle/detrz.pickle'))
        sdlmath.draw_track_rz(ax_rz, pt, eta, phi, vx, vy, vz, charge)
        sdlDisplay.display_detector_rz(ax_rz)
        if len(true_hits):
            draw_tracklet_rz(ax_rz, true_hits, useRecoStyle=False)
        draw_tracklet_rz(ax_rz, reco_hits, useRecoStyle=True)

        plt.savefig("detrz.pdf")

        break

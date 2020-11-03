#!/bin/env python

from data_new_v12 import data
import sys
import math
import ROOT as r

with open('scripts/module_center_data.txt','r') as inf:
    dict_from_file = eval(inf.read())

import math

def dotproduct(v1, v2):
  return sum((a*b) for a, b in zip(v1, v2))

def length(v):
  return math.sqrt(dotproduct(v, v))

def angle(v1, v2):
  return math.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))

start_xs = []
start_ys = []
start_zs = []
start_rs = []
end_xs = []
end_ys = []
end_zs = []
end_rs = []
dist = []
distxy = []
distz = []
dphis = []
nconns = []
nconns_perconn = []

orig_layer_perconn = []
orig_side_perconn = []
targ_layer_perconn = []
targ_side_perconn = []

# bylayer_start_xs = {}
# bylayer_start_ys = {}
# bylayer_start_zs = {}
# bylayer_start_rs = {}
# bylayer_end_xs = {}
# bylayer_end_ys = {}
# bylayer_end_zs = {}
# bylayer_end_rs = {}
# bylayer_dist = {}
# bylayer_distxy = {}

# for nodetype in nodetypes:
#     bylayer_start_xs[nodetype] = []
#     bylayer_start_ys[nodetype] = []
#     bylayer_start_zs[nodetype] = []
#     bylayer_start_rs[nodetype] = []
#     bylayer_end_xs[nodetype] = []
#     bylayer_end_ys[nodetype] = []
#     bylayer_end_zs[nodetype] = []
#     bylayer_end_rs[nodetype] = []
#     bylayer_dist[nodetype] = []
#     bylayer_distxy[nodetype] = []

for index, i in enumerate(data):

    # if index != 200:
    #     continue

    # if len(data[i]) < 40 or len(data[i]) > 60:
    #     continue

    nconns.append(len(data[i]))

    # has_large_dphi = False

    # for j in data[i]:

    #     orig = dict_from_file[i[0]]
    #     targ = dict_from_file[j[0]]

    #     ax = [orig[0], orig[1]]
    #     bx = [targ[0]-orig[0], targ[1]-orig[1]]
    #     ang = angle(ax, bx)

    #     if ang > 1.:
    #         has_large_dphi = True
    #         break

    # if not has_large_dphi:
    #     continue

    for j in data[i]:

        # print i[0]
        # print j[0]

        orig = dict_from_file[i[0]]
        targ = dict_from_file[j[0]]

        # orig = dict_from_file[i[0]]
        # targ = dict_from_file[mod[0]]

        d = math.sqrt((targ[0]-orig[0])**2 + (targ[1]-orig[1])**2 + (targ[2]-orig[2])**2)

        ax = [orig[0], orig[1]]
        bx = [targ[0]-orig[0], targ[1]-orig[1]]
        ang = angle(ax, bx)

        # if i[3] != j[3]:
        #     continue

        # if ang < 1.:
        #     continue

        start_xs.append(orig[0])
        start_ys.append(orig[1])
        start_zs.append(orig[2])
        end_xs.append(targ[0])
        end_ys.append(targ[1])
        end_zs.append(targ[2])
        start_rs.append(math.sqrt(orig[0]**2 + orig[1]**2))
        end_rs.append(math.sqrt(targ[0]**2 + targ[1]**2))
        dist.append(d)
        distxy.append(math.sqrt((targ[0]-orig[0])**2 + (targ[1]-orig[1])**2))
        distz.append(abs(targ[2]-orig[2]))

        orig_layer_perconn.append(i[2])
        orig_side_perconn.append(i[3])
        targ_layer_perconn.append(j[2])
        targ_side_perconn.append(j[3])

        dphis.append(ang)

        nconns_perconn.append(len(data[i]))

hist_nconn = r.TH1F("nconn", "nconn", 150, 0, 150)
hist = r.TH1F("moddist", "moddist", 180, 0, 200)
histz = r.TH1F("moddistz", "moddistz", 180, 0, 200)
histdphi = r.TH1F("moddistdphi", "moddistdphi", 180, 0, 3.1416)
histdphi_z = r.TH2F("moddistdphi_z", "moddistdphi_z", 180, 0, 3.1416, 180, 0, 200)
histdphi_d = r.TH2F("moddistdphi_d", "moddistdphi_d", 180, 0, 3.1416, 180, 0, 200)
histdphi_dxy = r.TH2F("moddistdphi_dxy", "moddistdphi_dxy", 180, 0, 3.1416, 180, 0, 200)
histnconn_z = r.TH2F("nconn_z", "nconn_z", 150, 0, 150, 180, -200, 200)
histdphi_layer = r.TH2F("moddistdphi_layer", "moddistdphi_layer", 180, 0, 3.1416, 7, 0, 7)
histdphi_targlayer = r.TH2F("moddistdphi_targlayer", "moddistdphi_targlayer", 180, 0, 3.1416, 7, 0, 7)
histdphi_side = r.TH2F("moddistdphi_side", "moddistdphi_side", 180, 0, 3.1416, 7, 0, 7)
histdphi_targside = r.TH2F("moddistdphi_targside", "moddistdphi_targside", 180, 0, 3.1416, 7, 0, 7)

conntypes = [
        (1,5),
        (2,5),
        (3,5),
        (4,5),
        (5,5),
        (1,4),
        (2,4),
        (3,4),
        (4,4),
        ]

hists_dphi_v_d = []

for conntype in conntypes:
    hists_dphi_v_d.append(r.TH2F("moddist_dphi_v_d_{}_{}".format(conntype[0], conntype[1]), "", 180, 0, 3.1416, 180, 0, 200))

for nconn in nconns:
    hist_nconn.Fill(nconn)

for d in dist:
    hist.Fill(d)

for dz in distz:
    histz.Fill(dz)

for dphi in dphis:
    histdphi.Fill(dphi)

for z, dphi in zip(distz, dphis):
    histdphi_z.Fill(dphi, z)

for d, dphi in zip(dist, dphis):
    histdphi_d.Fill(dphi, d)

for dxy, dphi in zip(distxy, dphis):
    histdphi_dxy.Fill(dphi, dxy)

for nconn, startz in zip(nconns_perconn, start_zs):
    histnconn_z.Fill(nconn, startz)

for layer, dphi in zip(orig_layer_perconn, dphis):
    histdphi_layer.Fill(dphi, layer)

for layer, dphi in zip(targ_layer_perconn, dphis):
    histdphi_targlayer.Fill(dphi, layer)

for side, dphi in zip(orig_side_perconn, dphis):
    histdphi_side.Fill(dphi, side)

for side, dphi in zip(targ_side_perconn, dphis):
    histdphi_targside.Fill(dphi, side)

for layer, side, d, dphi in zip(orig_layer_perconn, orig_side_perconn, dist, dphis):

    for index, conntype in enumerate(conntypes):

        if conntype[0] == layer and conntype[1] == side:

            hists_dphi_v_d[index].Fill(dphi, d)

c1 = r.TCanvas()
hist.Draw("hist")
c1.SaveAs("plots_connections/moddist.pdf")
c1.SaveAs("plots_connections/moddist.png")
histz.Draw("hist")
c1.SaveAs("plots_connections/moddistz.pdf")
c1.SaveAs("plots_connections/moddistz.png")
histdphi.Draw("hist")
c1.SaveAs("plots_connections/moddistdphi.pdf")
c1.SaveAs("plots_connections/moddistdphi.png")
histdphi_z.Draw("colz")
c1.SaveAs("plots_connections/moddistdphi_z.pdf")
c1.SaveAs("plots_connections/moddistdphi_z.png")
histdphi_d.Draw("colz")
c1.SaveAs("plots_connections/moddistdphi_d.pdf")
c1.SaveAs("plots_connections/moddistdphi_d.png")
histdphi_dxy.Draw("colz")
c1.SaveAs("plots_connections/moddistdphi_dxy.pdf")
c1.SaveAs("plots_connections/moddistdphi_dxy.png")
hist_nconn.Draw("hist")
c1.SaveAs("plots_connections/nconn.pdf")
c1.SaveAs("plots_connections/nconn.png")
histnconn_z.Draw("colz")
c1.SaveAs("plots_connections/nconn_z.pdf")
c1.SaveAs("plots_connections/nconn_z.png")
histdphi_layer.Draw("colz")
c1.SaveAs("plots_connections/moddistdphi_layer.pdf")
c1.SaveAs("plots_connections/moddistdphi_layer.png")
histdphi_targlayer.Draw("colz")
c1.SaveAs("plots_connections/moddistdphi_targlayer.pdf")
c1.SaveAs("plots_connections/moddistdphi_targlayer.png")
histdphi_side.Draw("colz")
c1.SaveAs("plots_connections/moddistdphi_side.pdf")
c1.SaveAs("plots_connections/moddistdphi_side.png")
histdphi_targside.Draw("colz")
c1.SaveAs("plots_connections/moddistdphi_targside.pdf")
c1.SaveAs("plots_connections/moddistdphi_targside.png")

for index, conntype in enumerate(conntypes):
    hists_dphi_v_d[index].Draw("colz")
    c1.SaveAs("plots_connections/moddistdphi_d_{}_{}.pdf".format(conntype[0], conntype[1]))
    c1.SaveAs("plots_connections/moddistdphi_d_{}_{}.png".format(conntype[0], conntype[1]))

# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def plot_connections(_start_xs, _start_ys, _start_zs, _start_rs, _end_xs, _end_ys, _end_zs, _end_rs, _distxy, _dist, outputname):

    print outputname

    fig = plt.figure()
    # ax = fig.gca(projection='3d')
    ax = fig.gca()
    print ax
    
    print len(_start_xs)
    
    # sort by distance
    _start_xs = [ x for _,x in sorted(zip(_distxy, _start_xs)) ]
    _start_ys = [ x for _,x in sorted(zip(_distxy, _start_ys)) ]
    _start_zs = [ x for _,x in sorted(zip(_distxy, _start_zs)) ]
    _start_rs = [ x for _,x in sorted(zip(_distxy, _start_rs)) ]
    _end_xs = [ x for _,x in sorted(zip(_distxy, _end_xs)) ]
    _end_ys = [ x for _,x in sorted(zip(_distxy, _end_ys)) ]
    _end_zs = [ x for _,x in sorted(zip(_distxy, _end_zs)) ]
    _end_rs = [ x for _,x in sorted(zip(_distxy, _end_rs)) ]
    _distxy = [ x for _,x in sorted(zip(_distxy, _distxy)) ]
    mx = max(_distxy)
    for i in xrange(len(_start_xs)):
    # for i in xrange(100):
        if i % 1000 == 0:
            print i
        ax.plot([_start_xs[i], _end_xs[i]], [_start_ys[i],_end_ys[i]], color=(1,1-_distxy[i] / mx,1-_distxy[i] / mx))
    plt.show()
    fig.savefig('plots_connections/{}xy.pdf'.format(outputname))
    fig.savefig('plots_connections/{}xy.png'.format(outputname))
    
    fig = plt.figure()
    # ax = fig.gca(projection='3d')
    ax = fig.gca()
    print ax
    # sort by distance
    _start_xs = [ x for _,x in sorted(zip(_dist, _start_xs)) ]
    _start_ys = [ x for _,x in sorted(zip(_dist, _start_ys)) ]
    _start_zs = [ x for _,x in sorted(zip(_dist, _start_zs)) ]
    _start_rs = [ x for _,x in sorted(zip(_dist, _start_rs)) ]
    _end_xs = [ x for _,x in sorted(zip(_dist, _end_xs)) ]
    _end_ys = [ x for _,x in sorted(zip(_dist, _end_ys)) ]
    _end_zs = [ x for _,x in sorted(zip(_dist, _end_zs)) ]
    _end_rs = [ x for _,x in sorted(zip(_dist, _end_rs)) ]
    _dist = [ x for _,x in sorted(zip(_dist, _dist)) ]
    mx = max(_dist)
    for i in xrange(len(_start_xs)):
    # for i in xrange(100):
        if i % 1000 == 0:
            print i
        ax.plot([_start_zs[i], _end_zs[i]], [_start_rs[i],_end_rs[i]], color=(1,1-_dist[i] / mx,1-_dist[i] / mx))
    plt.show()
    fig.savefig('plots_connections/{}rz.pdf'.format(outputname))
    fig.savefig('plots_connections/{}rz.png'.format(outputname))

plot_connections(start_xs, start_ys, start_zs, start_rs, end_xs, end_ys, end_zs, end_rs, distxy, dist, "all")

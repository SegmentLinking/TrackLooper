#!/bin/env python

import collections

# (1, 5) Counter({(2, 5): 47903, (1, 4): 10260, (3, 5): 6, (2, 4): 1})
# (2, 5) Counter({(3, 5): 39690, (1, 4): 8193, (4, 5): 17, (2, 4): 11})
# (3, 5) Counter({(4, 5): 32431, (1, 4): 7225, (2, 4): 27, (5, 5): 13, (6, 5): 1})
# (4, 5) Counter({(5, 5): 27733, (1, 4): 4692, (6, 5): 11, (2, 4): 10})
# (5, 5) Counter({(6, 5): 22729, (1, 4): 3085, (5, 4): 1})
# (6, 5) Counter({(5, 4): 2, (4, 4): 1, (3, 4): 1})
# (1, 4) Counter({(2, 4): 31952, (3, 4): 17})
# (2, 4) Counter({(3, 4): 28279, (4, 4): 3936, (5, 4): 11})
# (3, 4) Counter({(4, 4): 24359, (5, 4): 19})
# (4, 4) Counter({(5, 4): 24383})
# (5, 4) Counter()

with open('scripts/module_center_data.txt','r') as inf:
     dict_from_file = eval(inf.read())

# print dict_from_file


# What i think are OK connections
def isOKConn(key1, key2):
    if key1 == (1, 5) and (key2 == (2, 5) or key2 == (1, 4)): return True
    if key1 == (2, 5) and (key2 == (3, 5) or key2 == (1, 4)): return True
    if key1 == (3, 5) and (key2 == (4, 5) or key2 == (1, 4)): return True
    if key1 == (4, 5) and (key2 == (5, 5) or key2 == (1, 4)): return True
    if key1 == (5, 5) and (key2 == (6, 5) or key2 == (1, 4)): return True
    if key1 == (1, 4) and (key2 == (2, 4)): return True
    if key1 == (2, 4) and (key2 == (3, 4) or key2 == (4, 4) or key2 == (5, 4)): return True
    if key1 == (3, 4) and (key2 == (4, 4)): return True
    if key1 == (4, 4) and (key2 == (5, 4)): return True

    # if key1 == (1, 5) and (key2 == (2, 5)): return True
    # if key1 == (2, 5) and (key2 == (3, 5)): return True
    # if key1 == (3, 5) and (key2 == (4, 5)): return True
    # if key1 == (4, 5) and (key2 == (5, 5)): return True
    # if key1 == (5, 5) and (key2 == (6, 5)): return True
    # if key1 == (1, 4) and (key2 == (2, 4)): return True
    # if key1 == (2, 4) and (key2 == (3, 4) or key2 == (4, 4)): return True
    # if key1 == (3, 4) and (key2 == (4, 4)): return True
    # if key1 == (4, 4) and (key2 == (5, 4)): return True

    return False


f = open("scripts/moduleconnection_data.txt")

def f7(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

all_seqs = []
all_connections = {}

for index, line in enumerate(f.readlines()):

    # Strip end of line
    ls = line.strip()

    items = [ x.strip() for x in ls.split(":")[1].split(";") if len(x) > 0 ]

    if len(items) < 2:
        continue

    seq = []
    layerkeys = []
    detid_db = {}

    for item in items:
        item = item.replace("(","")
        item = item.replace(")","")
        keys = item.split(',')
        seq.append((int(keys[0]), int(keys[1])))

        layerkey = (int(keys[0]), int(keys[1]))
        detid = int(keys[2])
        partnerdetid = int(keys[3])
        if layerkey not in detid_db:
            layerkeys.append(layerkey)
            detid_db[layerkey] = []
            detid_db[layerkey].append((detid,partnerdetid,layerkey[0],layerkey[1]))
        else:
            detid_db[layerkey].append((detid,partnerdetid,layerkey[0],layerkey[1]))

    seq = f7(seq)

    for index in xrange(len(seq)-1):
        curr_layerkey = seq[index]
        next_layerkey = seq[index+1]

        if not isOKConn(curr_layerkey, next_layerkey):
            continue
        # if isOKConn(curr_layerkey, next_layerkey):
        #     continue

        for ii in detid_db[curr_layerkey]:
            for jj in detid_db[next_layerkey]:
                if ii not in all_connections:
                    all_connections[ii] = []
                    all_connections[ii].append(jj)
                else:
                    all_connections[ii].append(jj)

    # for i in range(len(seq)-1):
    #     if seq[i]==(6,5) and seq[i+1]==(4,4):
    #         print line

    all_seqs.append(seq)

nodetypes = [(1,5), (2,5), (3,5), (4,5), (5,5), (6,5), (1,4), (2,4), (3,4), (4,4), (5,4)]
connections = {}
for nodetype in nodetypes:
    connections[nodetype] = []

for seq in all_seqs:

    for index in xrange(len(seq)-1):

        connections[seq[index]].append(seq[index+1])

for key in connections:
    # connections[key] = list(set(connections[key]))
    connections[key] = collections.Counter(connections[key])

# for nodetype in nodetypes:
#     print nodetype, connections[nodetype]

import ROOT as r
import math

h_r3dist = r.TH1F("r3dist","",200,0,200)
h_nconn = r.TH1F("nconn","",20,0,20)

h_r3dist_dict = {}
h_nconn_dict = {}
for nodetype in nodetypes:
    h_r3dist_dict[nodetype] = r.TH1F("r3dist_{}_{}".format(nodetype[0], nodetype[1]), "", 200, 0, 200)
    h_nconn_dict[nodetype] = r.TH1F("nconn_{}_{}".format(nodetype[0], nodetype[1]), "", 20, 0, 20)

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

bylayer_start_xs = {}
bylayer_start_ys = {}
bylayer_start_zs = {}
bylayer_start_rs = {}
bylayer_end_xs = {}
bylayer_end_ys = {}
bylayer_end_zs = {}
bylayer_end_rs = {}
bylayer_dist = {}
bylayer_distxy = {}

for nodetype in nodetypes:
    bylayer_start_xs[nodetype] = []
    bylayer_start_ys[nodetype] = []
    bylayer_start_zs[nodetype] = []
    bylayer_start_rs[nodetype] = []
    bylayer_end_xs[nodetype] = []
    bylayer_end_ys[nodetype] = []
    bylayer_end_zs[nodetype] = []
    bylayer_end_rs[nodetype] = []
    bylayer_dist[nodetype] = []
    bylayer_distxy[nodetype] = []


sumnconn = 0

for i in all_connections:
    modules = all_connections[i]
    multiplicities = collections.Counter(modules)
    modules_in_order = [ x[0] for x in multiplicities.most_common(len(multiplicities)) ]
    # print i, modules_in_order
    print i[0], len(modules_in_order), " ".join([ str(x[0]) for x in modules_in_order ])
    sumnconn += len(modules_in_order)
    # print dict_from_file[i[0]]
    for index, mod in enumerate(modules_in_order):
        orig = dict_from_file[i[0]]
        targ = dict_from_file[mod[0]]
        d = math.sqrt((targ[0]-orig[0])**2 + (targ[1]-orig[1])**2 + (targ[2]-orig[2])**2)
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
        bylayer_start_xs[(i[2],i[3])].append(orig[0])
        bylayer_start_ys[(i[2],i[3])].append(orig[1])
        bylayer_start_zs[(i[2],i[3])].append(orig[2])
        bylayer_end_xs[(i[2],i[3])].append(targ[0])
        bylayer_end_ys[(i[2],i[3])].append(targ[1])
        bylayer_end_zs[(i[2],i[3])].append(targ[2])
        bylayer_start_rs[(i[2],i[3])].append(math.sqrt(orig[0]**2 + orig[1]**2))
        bylayer_end_rs[(i[2],i[3])].append(math.sqrt(targ[0]**2 + targ[1]**2))
        bylayer_dist[(i[2],i[3])].append(d)
        bylayer_distxy[(i[2],i[3])].append(math.sqrt((targ[0]-orig[0])**2 + (targ[1]-orig[1])**2))
        h_r3dist.Fill(d)
        h_r3dist_dict[(i[2],i[3])].Fill(d)
    h_nconn.Fill(len(modules_in_order))
    h_nconn_dict[(i[2],i[3])].Fill(len(modules_in_order))

import sys
sys.exit()

print sumnconn, len(all_connections)

c = r.TCanvas()
h_r3dist.Draw("hist")
c.SetLogy()
c.SaveAs("plots_connections/moddist.pdf")
c.SaveAs("plots_connections/moddist.png")
h_nconn.Draw("hist")
c.SetLogy(0)
c.SaveAs("plots_connections/nconn.pdf")
c.SaveAs("plots_connections/nconn.png")

for nodetype in nodetypes:
    h_r3dist_dict[nodetype].Draw("hist")
    c.SetLogy()
    c.SaveAs("plots_connections/moddist_{}_{}.pdf".format(nodetype[0],nodetype[1]))
    c.SaveAs("plots_connections/moddist_{}_{}.png".format(nodetype[0],nodetype[1]))
    h_nconn_dict[nodetype].Draw("hist")
    c.SetLogy(0)
    c.SaveAs("plots_connections/nconn_{}_{}.pdf".format(nodetype[0],nodetype[1]))
    c.SaveAs("plots_connections/nconn_{}_{}.png".format(nodetype[0],nodetype[1]))

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
for nodetype in nodetypes:
    if nodetype == (6, 5):
        continue
    plot_connections(bylayer_start_xs[nodetype], bylayer_start_ys[nodetype], bylayer_start_zs[nodetype], bylayer_start_rs[nodetype], bylayer_end_xs[nodetype], bylayer_end_ys[nodetype], bylayer_end_zs[nodetype], bylayer_end_rs[nodetype], bylayer_distxy[nodetype], bylayer_dist[nodetype], "{}_{}".format(nodetype[0],nodetype[1]))

#!/bin/env python

from Module import Module
from Centroid import Centroid
import numpy as np
import math

from operator import itemgetter
from itertools import imap, groupby, product

import collections
from tqdm import tqdm

import pickle
import os

# Centroid database
centroidDB = Centroid("data/centroid_2020_0428.txt")

def unique_justseen(iterable, key=None):
    "List unique elements, preserving order. Remember only the element just seen."
    # unique_justseen('AAAABBBCCDAABBB') --> A B C D A B
    # unique_justseen('ABBCcAD', str.lower) --> A B C A D
    return imap(next, imap(itemgetter(1), groupby(iterable, key)))

# What i think are OK connections
def isOKConn(detid1, detid2):
    mod1 = Module(detid1)
    mod2 = Module(detid2)
    layer1 = mod1.logicalLayer()
    layer2 = mod2.logicalLayer()
    centroid_vec1 = np.array(centroidDB.getCentroid(detid1))
    centroid_vec2 = np.array(centroidDB.getCentroid(detid2))
    centroid_rt1 = math.sqrt(centroid_vec1[0]**2 + centroid_vec1[1]**2)
    centroid_rt2 = math.sqrt(centroid_vec2[0]**2 + centroid_vec2[1]**2)

    if centroid_rt1 - 15 > centroid_rt2:
        return False

    # Barrel Barrel
    if layer1 ==  1 and layer2 ==  2: return True
    if layer1 ==  2 and layer2 ==  3: return True
    if layer1 ==  3 and layer2 ==  4: return True
    if layer1 ==  4 and layer2 ==  5: return True
    if layer1 ==  5 and layer2 ==  6: return True

    # Barrel Endcap
    if layer1 ==  1 and layer2 ==  7: return True
    if layer1 ==  2 and layer2 ==  7: return True
    if layer1 ==  3 and layer2 ==  7: return True
    if layer1 ==  4 and layer2 ==  7: return True
    if layer1 ==  5 and layer2 ==  7: return True

    # Endcap Endcap
    if layer1 ==  7 and layer2 ==  8: return True
    if layer1 ==  8 and layer2 ==  9: return True
    if layer1 ==  9 and layer2 == 10: return True
    if layer1 == 10 and layer2 == 11: return True

    # Endcap Endcap special for disk 2
    if layer1 ==  8 and mod1.ring() <= 3 and layer2 ==  9: return True
    if layer1 ==  8 and mod1.ring() <= 2 and layer2 == 10: return True
    if layer1 ==  8 and mod1.ring() <= 2 and layer2 == 11: return True

    return False

connections = {}

if not os.path.exists("data/module_connection.pickle"):

    # Looping over the raw module connection data output from ./bin/sdl
    raw_data = open("data/conn_2020_0429.txt")
    lines = raw_data.readlines()
    module_connection_raw_data = []
    for index, line in enumerate(tqdm(lines[1:-1])):
    
        command = "{}]".format(line[:-5])
    
        mod_conn_data = eval(command)
    
        # # Track index
        # print(index)
    
        # Parsed data
        layer_ordered = []
        detids = {}
    
        # Looping over the hits and the module data
        for detid, x, y, z in mod_conn_data:
    
            # Parsing some information
            simhit_vec = np.array([x, y, z])
            centroid_vec = np.array(centroidDB.getCentroid(detid))
            module = Module(detid)
            layer = module.layer() + (6 if module.subdet() == 4 else 0)
    
            layer_ordered.append(layer)
    
            if layer not in detids:
                detids[layer] = []
    
            detids[layer].append(detid)
    
            # # Debug printing
            # print(detid, layer, module.side(), module.ring(), module.rod(), module.module(), simhit_vec, centroid_vec, np.linalg.norm(simhit_vec-centroid_vec))
    
        # Debug printing
        module_detids_by_layer = []
        for layer in unique_justseen(layer_ordered):
            # print(detids[layer])
            module_detids_by_layer.append(detids[layer])
    
        for index in range(len(module_detids_by_layer) - 1):
            for combo in product(module_detids_by_layer[index], module_detids_by_layer[index + 1]):
                reference = combo[0]
                target = combo[1]
    
                if reference not in connections:
                    connections[reference] = []
    
                connections[reference].append(target)
    
        # if index > 100:
        #     break
    
    # print(connections)
    
    for key in connections:
        # connections[key] = list(set(connections[key]))
        connections[key] = collections.Counter(connections[key])
        # print(connections[key])
    
    # print(connections)
    
    pickle.dump(connections, file("data/module_connection.pickle", "w"))

else:

    connections = pickle.load(file("data/module_connection.pickle"))

# g = open("data/conn_muon_multiplicity.txt", "w")
# # Checking what are "not OK" connections
# for reference in connections:
#     for target in connections[reference]:
#         g.write("{} {} {}\n".format(reference, target, connections[reference][target]))
#         if not isOKConn(reference, target):
#             print(reference, target)
#             print(Module(reference).__str__(), Module(target).__str__())

import sys
nconncut = int(sys.argv[1])

f = open("data/module_connection_nmuon{}.txt".format(nconncut), "w")

for reference in connections:
    nconn = len(connections[reference])
    targets = []
    for target in connections[reference]:
        if isOKConn(reference, target) and int(connections[reference][target]) >= nconncut:
            targets.append(str(target))
    f.write("{} {} {}\n".format(reference, len(targets), " ".join(targets)))


#!/bin/env python

import ROOT as r
import itertools
import array

tf = r.TFile("/hadoop/cms/store/user/slava77/CMSSW_10_4_0_patch1-tkNtuple/pass-e072c1a/27411.0_TenMuExtendedE_0_200/trackingNtuple.root")
ttree = tf.Get("trackingNtuple/tree")

branches = [
# "ph2_isBarrel",
"ph2_layer",
#"ph2_isUpper",
# "ph2_isStack",
"ph2_order",
"ph2_ring",
"ph2_rod",
"ph2_subdet",
"ph2_side",
"ph2_module",
#"ph2_simType",
"ph2_isLower",
"ph2_detId",
]

# To hold the N-tuple of ids
l = []
module_id_map = {}
for event in ttree:

    # To hold the contents read
    vectors = []

    # Reading IDs
    for br in branches:
        vector = list(getattr(event, br))
        vectors.append(vector)

    # Get the IDs
    for idx in zip(*vectors):
        l.append(idx)

# Form unique IDs
index_l = list(set(l))

index_l = sorted(index_l, key=lambda x: x[-1])

detIds = []
seqIds = []
detId_to_seqId = {}
seqId_to_detId = {}

for ii, idx in enumerate(index_l):

    seqId = idx[:-1]
    detId = idx[-1]

    if detId in detId_to_seqId:
        print "Error detId already included", idx

    if seqId in seqId_to_detId:
        print "Error seqId already included", idx

    detId_to_seqId[idx[-1]] = idx[:-1]
    seqId_to_detId[idx[:-1]] = idx[-1]

    detIds.append(detId)
    seqIds.append(seqId)

f = open("detId_to_seqId.txt", "w")

f.write("detId ")
f.write(",".join(branches[:-1]))
f.write("\n")

for detId in detIds:
    f.write("{} {}\n".format(str(detId), ",".join(map(str, detId_to_seqId[detId]))))

# c1 = r.TCanvas()

# for detId in detId_to_seqId:

#     ttree.Draw("ph2_y:ph2_x", "ph2_detId=={}".format(detId), "colzgoff")
#     c1.SaveAs("det_maps/yx_{}.pdf".format(detId))

#     ttree.Draw("sqrt(ph2_y**2+ph2_x**2):ph2_z", "ph2_detId=={}".format(detId), "colzgoff")
#     c1.SaveAs("det_maps/rz_{}.pdf".format(detId))

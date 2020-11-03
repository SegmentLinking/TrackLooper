#!/bin/env python

import ROOT as r

f = r.TFile("/home/users/phchang/public_html/analysis/sdl/TrackLooper_/results/write_sdl_ntuple/e200_20200910_Test_v2//fulleff_e200.root")
t = f.Get("tree")

tc_types = []

for ievent, event in enumerate(t):

    print ievent, len(event.tc_layer)

    for tc_layer in event.tc_layer:
        layers = list(set(list(tc_layer)))
        layers.sort()
        tc_types.append(tuple(layers))

tc_types = [ " ".join([str(i) for i in x]) for x in tc_types ]
tc_types = list(set(tc_types))
tc_types.sort()

for i in tc_types:
    print i

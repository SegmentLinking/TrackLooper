#!/bin/env python

import ROOT as r
import plottery_wrapper as p

# f = open("scripts/12hit_data.log")
# f = open("scripts/12hit_data_e0_200.log")
f = open("scripts/12hit_data_e0_200_nosimhit13requirement.log")

lines = f.readlines()

h_layers_pt_denom = r.TH1F("layers_denom", "", 5, 0, 5)
h_layers_pt_numer = r.TH1F("layers_numer", "", 5, 0, 5)

h_layers_pt_denom_pt1p2_1p5 = r.TH1F("layers_denom_pt1p2_1p5", "", 5, 0, 5)
h_layers_pt_numer_pt1p2_1p5 = r.TH1F("layers_numer_pt1p2_1p5", "", 5, 0, 5)

ntrk_12hits = 0

for line in lines:

    ls = line.split()

    both_layer_has_hits = [ int(x) for i, x in enumerate(ls[:12]) if i % 2 == 1 ]
    pt = float(ls[13])
    eta = float(ls[15])

    all_layers_have_hits = sum(both_layer_has_hits) == 6

    if all_layers_have_hits and pt > 1.2 and pt < 1.5:
        ntrk_12hits += 1

    # print both_layer_has_hits, ls[25], ls[27]

    for ilayer, i in enumerate(both_layer_has_hits):
        if ilayer == 0:
            continue
        if i == 1:
            h_layers_pt_denom.Fill(ilayer - 1)
            if both_layer_has_hits[ilayer - 1]:
                h_layers_pt_numer.Fill(ilayer - 1)

    if pt > 1.2 and pt < 1.5:
        for ilayer, i in enumerate(both_layer_has_hits):
            if ilayer == 0:
                continue
            if i == 1:
                h_layers_pt_denom_pt1p2_1p5.Fill(ilayer - 1)
                if both_layer_has_hits[ilayer - 1]:
                    h_layers_pt_numer_pt1p2_1p5.Fill(ilayer - 1)

print ntrk_12hits

p.plot_hist(bgs=[h_layers_pt_denom], data=h_layers_pt_numer, options={"output_name": "plots_hit_eff/layers_eff.pdf", "print_yield":True})
p.plot_hist(bgs=[h_layers_pt_denom_pt1p2_1p5], data=h_layers_pt_numer_pt1p2_1p5, options={"output_name": "plots_hit_eff/layers_eff_pt1p2_1p5.pdf", "print_yield":True})

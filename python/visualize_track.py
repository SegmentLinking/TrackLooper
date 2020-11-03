#!/bin/env python

# Tracking Ntuple single track visualization
# Philip Chang

import ROOT as r
r.gROOT.SetBatch(True)

import sys
import os
import math

def usage():

    print "Usage:"
    print ""
    print "   python {} TRACKINGNTUPLE EVENTINDEX SIMTRKINDEX".format(sys.argv[0])
    print ""
    print "     if SIMTRKINDEX < 0, then the user will be asked to input the simtrkidx"
    print ""
    print ""
    sys.exit()

try:
    tkntuple = sys.argv[1]
    evt_idx_to_plot = int(sys.argv[2])
    trk_idx_to_plot = int(sys.argv[3])
except:
    usage()

# Open the input data
t = r.TChain("trackingNtuple/tree")
t.Add(tkntuple)

# Open the reference input data that is useful for laying out basic tracker
# Some ntuples contain lots of events and takes eons to map it out
t_ref = r.TChain("trackingNtuple/tree")
t_ref.Add("/hadoop/cms/store/user/slava77/CMSSW_10_4_0_patch1-tkNtuple/pass-e072c1a/27411.0_TenMuExtendedE_0_200/trackingNtuple.root")

# Set the trk index if -1 is provided
if trk_idx_to_plot < 0:
    t.Scan("sim_pt:sim_eta:sim_phi:sim_q", "Entry$=={}".format(evt_idx_to_plot))
    trk_idx_to_plot = int(raw_input("trkidx : "))

# Now select the track and aggregate simhit information
simhit_selected = []
clean_ph2hit_selected = []
dirty_ph2hit_selected = []
pt_of_track = 0
eta_of_track = 0

# Loop over to get the simhit, and ph2hit idxs
for index, event in enumerate(t):

    # If the event is the one you want
    # I don't know how to access the specific event directly in PyROOT efficiently
    # Let me know if you know
    if index == evt_idx_to_plot:

        # Get the true pt, eta information
        pt_of_track = event.sim_pt[trk_idx_to_plot]
        eta_of_track = event.sim_eta[trk_idx_to_plot]

        # Aggregate the simhits in the outer tracker
        for simhitidx in event.sim_simHitIdx[trk_idx_to_plot]:
            if event.simhit_subdet[simhitidx] == 4 or event.simhit_subdet[simhitidx] == 5:
                simhit_selected.append((event.simhit_x[simhitidx], event.simhit_y[simhitidx], event.simhit_z[simhitidx]))

                # Aggregate ph2 hits only matched to pdgid
                for hitidx in event.simhit_hitIdx[simhitidx]:

                    # Clean ph2 hits where the simhits are matched to parent pdgid
                    if event.simhit_particle[simhitidx] == event.sim_pdgId[trk_idx_to_plot]:
                        clean_ph2hit_selected.append((event.ph2_x[hitidx], event.ph2_y[hitidx], event.ph2_z[hitidx]))
                    # Otherwise, "dirty" (most likely secondary hits)
                    else:
                        dirty_ph2hit_selected.append((event.ph2_x[hitidx], event.ph2_y[hitidx], event.ph2_z[hitidx]))

        # Done so exit the event loop
        break

print("Now plotting...")

def draw():

    # Drawing x-y
    c1 = r.TCanvas("","",1700,2210)

    xy_pad = r.TPad("xy_pad", "xy_pad", 0.0, 0.23, 1.0, 1.0 )
    rz_pad = r.TPad("rz_pad", "rz_pad", 0.0, 0.0 , 1.0, 0.23)

    xy_pad.Draw()
    rz_pad.Draw()

    # Drawing baseline background
    xy_pad.cd()
    t_ref.Draw("simhit_x:simhit_y","(simhit_subdet==5)")
    h2 = r.gPad.GetPrimitive("htemp");
    h2.SetTitle("p_{{T}}={:.1f}, #eta={:.2f}, event={}, itrk={}".format(pt_of_track, eta_of_track, evt_idx_to_plot, trk_idx_to_plot))
    rz_pad.cd()
    t_ref.Draw("sqrt(simhit_x**2+simhit_y**2):simhit_z","(simhit_subdet==5)||(simhit_subdet==4)")
    h2 = r.gPad.GetPrimitive("htemp");
    h2.SetTitle("")
    h2.GetYaxis().SetRangeUser(0., 120.)
    print("Done plotting base")

    xy_pad.cd()
    # Drawing the center
    t.SetMarkerStyle(20)
    t.SetMarkerSize(1.5)
    t.SetMarkerColor(1)
    t.Draw("0:0","", "same")
    print("Done plotting center")
    # rz_pad.cd()
    # Drawing the center
    # t.SetMarkerStyle(20)
    # t.SetMarkerSize(1.5)
    # t.SetMarkerColor(1)
    # t.Draw("0:0","", "same")
    # t.Draw("-10:0","", "same")
    # t.Draw("10:0","", "same")
    # print("Done plotting luminoous")

    # Draw sim hits
    t.SetMarkerStyle(20)
    t.SetMarkerSize(0.7)
    # t.SetMarkerColorAlpha(2, 0.5)
    t.SetMarkerColor(2)
    print("To plot {} simhits".format(len(simhit_selected)))
    xy_pad.cd()
    for simhit in simhit_selected:
        t.Draw("{}:{}".format(simhit[0], simhit[1]),"Entry$==0", "same")
    rz_pad.cd()
    for simhit in simhit_selected:
        t.Draw("{}:{}".format(math.sqrt(simhit[0]**2+simhit[1]**2), simhit[2]),"Entry$==0", "same")
    print("Done plotting simhits")

    # Draw clean ph2 hits
    t.SetMarkerStyle(20)
    t.SetMarkerSize(0.5)
    # t.SetMarkerColorAlpha(4, 0.5)
    t.SetMarkerColor(4)
    print("To plot {} clean rec hits".format(len(clean_ph2hit_selected)))
    xy_pad.cd()
    for hit in clean_ph2hit_selected:
        t.Draw("{}:{}".format(hit[0], hit[1]),"Entry$==0", "same")
    rz_pad.cd()
    for hit in clean_ph2hit_selected:
        t.Draw("{}:{}".format(math.sqrt(hit[0]**2+hit[1]**2), hit[2]),"Entry$==0", "same")
    print("Done plotting clean rec hits")

    # Draw secondary ph2 hits
    t.SetMarkerStyle(20)
    t.SetMarkerSize(0.5)
    # t.SetMarkerColorAlpha(4, 0.5)
    t.SetMarkerColor(4)
    print("To plot {} dirty rec hits".format(len(dirty_ph2hit_selected)))
    xy_pad.cd()
    for hit in dirty_ph2hit_selected:
        t.Draw("{}:{}".format(hit[0], hit[1]),"Entry$==0", "same")
    rz_pad.cd()
    for hit in dirty_ph2hit_selected:
        t.Draw("{}:{}".format(math.sqrt(hit[0]**2+hit[1]**2), hit[2]),"Entry$==0", "same")
    print("Done plotting dirty rec hits")

    os.system("mkdir -p plots/track_visualization/")

    print("Creating png")
    c1.SaveAs("plots/track_visualization/plot.png")

draw()

#!/bin/env python

import ROOT as r
import sys

r.gROOT.SetBatch(True)

tf = r.TFile("/home/users/phchang/public_html/analysis/sdl/TrackLooper/trackingNtuple.root")
ttree = tf.Get("trackingNtuple/tree")

c1 = r.TCanvas()
print "here"

event = sys.argv[1]
isimtrk = sys.argv[2]
mods = [ int(mod) for mod in sys.argv[3].split(',') ]

def rz_specific_track_and_module():
    ttree.SetMarkerColor(16)
    ttree.Draw("sqrt(ph2_x**2+ph2_y**2):ph2_z", "")
    cuts = []
    for mod in mods:
        cuts.append("ph2_detId=={}".format(mod))
    cut = "||".join(cuts)
    ttree.SetMarkerColor(4)
    ttree.Draw("sqrt(ph2_x**2+ph2_y**2):ph2_z", cut, "same")
    ttree.SetMarkerColor(2)
    ttree.SetMarkerSize(20)
    ttree.Draw("sqrt(simhit_x**2+simhit_y**2):simhit_z", "(event=={})&&(simhit_simTrkIdx=={})&&(abs(simhit_particle)==13)&&(simhit_subdet==4||simhit_subdet==5)".format(event, isimtrk), "same")
    c1.SaveAs("plots/lin/rz_specific_track_and_module.pdf")
    c1.SaveAs("plots/lin/rz_specific_track_and_module.png")

def xy_specific_track_and_module():
    ttree.SetMarkerColor(16)
    ttree.Draw("ph2_y:ph2_x", "")
    cuts = []
    for mod in mods:
        cuts.append("ph2_detId=={}".format(mod))
    cut = "||".join(cuts)
    ttree.SetMarkerColor(4)
    ttree.Draw("ph2_y:ph2_x", cut, "same")
    ttree.SetMarkerColor(2)
    ttree.SetMarkerSize(20)
    ttree.Draw("simhit_y:simhit_x", "(event=={})&&(simhit_simTrkIdx=={})&&(abs(simhit_particle)==13)&&(simhit_subdet==4||simhit_subdet==5)".format(event, isimtrk), "same")
    c1.SaveAs("plots/lin/xy_specific_track_and_module.pdf")
    c1.SaveAs("plots/lin/xy_specific_track_and_module.png")

if __name__ == "__main__":

    rz_specific_track_and_module()
    xy_specific_track_and_module()

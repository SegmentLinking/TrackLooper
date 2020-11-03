#!/bin/env python

import ROOT as r
import sys

r.gROOT.SetBatch(True)

tf = r.TFile("/home/users/phchang/public_html/analysis/sdl/TrackLooper/trackingNtuple.root")
ttree = tf.Get("trackingNtuple/tree")

c1 = r.TCanvas()
print "here"

def plot(cuts, color, same):
    ttree.Draw("ph2_y:ph2_x", cuts, same)
    ttree.Draw("sqrt(ph2_x**2+ph2_y**2):ph2_z", cuts, same)


def rz_all_side():
    # rz all with color codes on ph2_side
    ttree.SetMarkerColor(1)
    ttree.Draw("sqrt(ph2_x**2+ph2_y**2):ph2_z", "")
    ttree.SetMarkerColor(4)
    ttree.Draw("sqrt(ph2_x**2+ph2_y**2):ph2_z", "ph2_side==1", "same")
    ttree.SetMarkerColor(2)
    ttree.Draw("sqrt(ph2_x**2+ph2_y**2):ph2_z", "ph2_side==2", "same")
    ttree.SetMarkerColor(1)
    ttree.Draw("sqrt(ph2_x**2+ph2_y**2):ph2_z", "ph2_side==3", "same")
    c1.SaveAs("plots/lin/rz_side.pdf")
    c1.SaveAs("plots/lin/rz_side.png")

def rz_side3_layer1_order0():
    # rz layer 1 barrel
    ttree.SetMarkerColor(1)
    ttree.Draw("sqrt(ph2_x**2+ph2_y**2):ph2_z", "ph2_side==3&&ph2_order==0&&ph2_layer==1")
    c1.SaveAs("plots/lin/rz_side3_layer1_order0.pdf")
    c1.SaveAs("plots/lin/rz_side3_layer1_order0.png")

def xy_side3_layer1_order0():
    # xy layer 1 barrel
    ttree.SetMarkerColor(1)
    ttree.Draw("ph2_y:ph2_x", "ph2_side==3&&ph2_order==0&&ph2_layer==1")
    c1.SaveAs("plots/lin/xy_side3_layer1_order0.pdf")
    c1.SaveAs("plots/lin/xy_side3_layer1_order0.png")

def rz_side1_layer1_order0():
    # rz layer 1 barrel
    ttree.SetMarkerColor(1)
    ttree.Draw("sqrt(ph2_x**2+ph2_y**2):ph2_z", "ph2_side==1&&ph2_order==0&&ph2_layer==1")
    c1.SaveAs("plots/lin/rz_side1_layer1_order0.pdf")
    c1.SaveAs("plots/lin/rz_side1_layer1_order0.png")

def rz_side1_layer1_order1():
    # rz layer 1 barrel
    ttree.SetMarkerColor(1)
    ttree.Draw("sqrt(ph2_x**2+ph2_y**2):ph2_z", "ph2_side==1&&ph2_order==1&&ph2_layer==1")
    c1.SaveAs("plots/lin/rz_side1_layer1_order1.pdf")
    c1.SaveAs("plots/lin/rz_side1_layer1_order1.png")

def rz_side3_order0():
    # rz layer 1 barrel
    ttree.SetMarkerColor(1)
    ttree.Draw("sqrt(ph2_x**2+ph2_y**2):ph2_z", "ph2_side==3&&ph2_order==0")
    c1.SaveAs("plots/lin/rz_side3_order0.pdf")
    c1.SaveAs("plots/lin/rz_side3_order0.png")

def xy_side3_order0():
    # xy layer 1 barrel
    ttree.SetMarkerColor(1)
    ttree.Draw("ph2_y:ph2_x", "ph2_side==3&&ph2_order==0")
    c1.SaveAs("plots/lin/xy_side3_order0.pdf")
    c1.SaveAs("plots/lin/xy_side3_order0.png")

def xy_side3_order0_rodcolors():
    colors = [1, 2]
    ttree.Draw("ph2_y:ph2_x", "ph2_side==3&&ph2_order==0")
    # xy layer 1 barrel
    for i in xrange(1, 77):
        ttree.SetMarkerColor(colors[i%2])
        ttree.Draw("ph2_y:ph2_x", "ph2_side==3&&ph2_order==0&&ph2_rod=={}".format(i), "same")
    c1.SaveAs("plots/lin/xy_side3_order0_rodcolor.pdf")
    c1.SaveAs("plots/lin/xy_side3_order0_rodcolor.png")

def rz_side3_order0_rodcolors():
    colors = [1, 2]
    ttree.Draw("sqrt(ph2_x**2+ph2_y**2):ph2_z", "ph2_side==3&&ph2_order==0")
    # rz layer 1 barrel
    for i in xrange(1, 79):
        ttree.SetMarkerColor(colors[i%2])
        ttree.Draw("sqrt(ph2_x**2+ph2_y**2):ph2_z", "ph2_side==3&&ph2_order==0&&ph2_rod=={}".format(i), "same")
    c1.SaveAs("plots/lin/rz_side3_order0_rodcolor.pdf")
    c1.SaveAs("plots/lin/rz_side3_order0_rodcolor.png")

def rz_side3_order0_rodcolors_modcolors():
    colors = [1, 2, 4, 8]
    ttree.Draw("sqrt(ph2_x**2+ph2_y**2):ph2_z", "ph2_side==3&&ph2_order==0")

    cuts = [
            "ph2_rod%2==1&&ph2_module%2==1",
            "ph2_rod%2==0&&ph2_module%2==1",
            "ph2_rod%2==1&&ph2_module%2==0",
            "ph2_rod%2==0&&ph2_module%2==0",
            ]
    # rz layer 1 barrel
    for i, cut in enumerate(cuts):
        ttree.SetMarkerColor(colors[i])
        ttree.Draw("sqrt(ph2_x**2+ph2_y**2):ph2_z", "ph2_side==3&&ph2_order==0&&{}".format(cut), "same")
    c1.SaveAs("plots/lin/rz_side3_order0_rodcolor_modcolor.pdf")
    c1.SaveAs("plots/lin/rz_side3_order0_rodcolor_modcolor.png")

def xy_side3_order0_rodall():
    for i in xrange(1, 79):
        ttree.SetMarkerColor(1)
        ttree.Draw("ph2_y:ph2_x", "ph2_side==3&&ph2_order==0")
        ttree.SetMarkerColor(2)
        ttree.Draw("ph2_y:ph2_x", "ph2_side==3&&ph2_order==0&&ph2_rod=={}".format(i), "same")
        c1.SaveAs("plots/lin/xy_side3_order0_rod{}.pdf".format(i))
        c1.SaveAs("plots/lin/xy_side3_order0_rod{}.png".format(i))

def rz_subdet4():
    # rz layer 1 barrel
    ttree.Draw("sqrt(ph2_x**2+ph2_y**2):ph2_z", "")
    ttree.SetMarkerColor(1)
    ttree.Draw("sqrt(ph2_x**2+ph2_y**2):ph2_z", "ph2_subdet==4")
    c1.SaveAs("plots/lin/rz_subdet4.pdf")
    c1.SaveAs("plots/lin/rz_subdet4.png")

def xy_side1_layer1_ringall():
    for i in xrange(1, 16):
        ttree.SetMarkerColor(1)
        ttree.Draw("ph2_y:ph2_x", "ph2_side==1&&ph2_layer==1")
        ttree.SetMarkerColor(2)
        ttree.Draw("ph2_y:ph2_x", "ph2_side==1&&ph2_layer==1&&ph2_ring=={}".format(i))
        c1.SaveAs("plots/lin/xy_side1_layer1_ring{}.pdf".format(i))
        c1.SaveAs("plots/lin/xy_side1_layer1_ring{}.png".format(i))

def xy_side1_layer1_ring2_moduleall():
    for i in xrange(1, 16):
        ttree.SetMarkerColor(1)
        ttree.Draw("ph2_y:ph2_x", "ph2_side==1&&ph2_layer==1&&ph2_ring==2")
        ttree.SetMarkerColor(2)
        ttree.Draw("ph2_y:ph2_x", "ph2_side==1&&ph2_layer==1&&ph2_ring==2&&ph2_module=={}".format(i), "same")
        c1.SaveAs("plots/lin/xy_side1_layer1_ring2_module{}.pdf".format(i))
        c1.SaveAs("plots/lin/xy_side1_layer1_ring2_module{}.png".format(i))

def xy_side1_layer1_moduleall():
    for i in xrange(1, 16):
        ttree.SetMarkerColor(1)
        ttree.Draw("ph2_y:ph2_x", "ph2_side==1&&ph2_layer==1")
        ttree.SetMarkerColor(2)
        ttree.Draw("ph2_y:ph2_x", "ph2_side==1&&ph2_layer==1&&ph2_module=={}".format(i), "same")
        c1.SaveAs("plots/lin/xy_side1_layer1_module{}.pdf".format(i))
        c1.SaveAs("plots/lin/xy_side1_layer1_module{}.png".format(i))

def rz_side1_layer1_moduleall():
    for i in xrange(1, 16):
        ttree.SetMarkerColor(1)
        ttree.Draw("sqrt(ph2_x**2+ph2_y**2):ph2_z", "ph2_side==1&&ph2_layer==1")
        ttree.SetMarkerColor(2)
        ttree.Draw("sqrt(ph2_x**2+ph2_y**2):ph2_z", "ph2_side==1&&ph2_layer==1&&ph2_module=={}".format(i), "same")
        c1.SaveAs("plots/lin/rz_side1_layer1_module{}.pdf".format(i))
        c1.SaveAs("plots/lin/rz_side1_layer1_module{}.png".format(i))

def rz_side1_layer1_order1_moduleall():
    for i in xrange(1, 16):
        ttree.SetMarkerColor(1)
        ttree.Draw("sqrt(ph2_x**2+ph2_y**2):ph2_z", "ph2_side==1&&ph2_layer==1&&ph2_order==1")
        ttree.SetMarkerColor(2)
        ttree.Draw("sqrt(ph2_x**2+ph2_y**2):ph2_z", "ph2_side==1&&ph2_layer==1&&ph2_order==1&&ph2_module=={}".format(i), "same")
        c1.SaveAs("plots/lin/rz_side1_layer1_order1_module{}.pdf".format(i))
        c1.SaveAs("plots/lin/rz_side1_layer1_order1_module{}.png".format(i))

def xy_side1_layer1_order1_moduleall():
    for i in xrange(1, 16):
        ttree.SetMarkerColor(1)
        ttree.Draw("ph2_y:ph2_x", "ph2_side==1&&ph2_layer==1&&ph2_order==1")
        ttree.SetMarkerColor(2)
        ttree.Draw("ph2_y:ph2_x", "ph2_side==1&&ph2_layer==1&&ph2_order==1&&ph2_module=={}".format(i), "same")
        c1.SaveAs("plots/lin/xy_side1_layer1_order1_module{}.pdf".format(i))
        c1.SaveAs("plots/lin/xy_side1_layer1_order1_module{}.png".format(i))

def rz_side1_layer1_order0_moduleall():
    for i in xrange(1, 16):
        ttree.SetMarkerColor(1)
        ttree.Draw("sqrt(ph2_x**2+ph2_y**2):ph2_z", "ph2_side==1&&ph2_layer==1&&ph2_order==0")
        ttree.SetMarkerColor(2)
        ttree.Draw("sqrt(ph2_x**2+ph2_y**2):ph2_z", "ph2_side==1&&ph2_layer==1&&ph2_order==0&&ph2_module=={}".format(i), "same")
        c1.SaveAs("plots/lin/rz_side1_layer1_order0_module{}.pdf".format(i))
        c1.SaveAs("plots/lin/rz_side1_layer1_order0_module{}.png".format(i))

def xy_side1_layer1_order0_moduleall():
    for i in xrange(1, 16):
        ttree.SetMarkerColor(1)
        ttree.Draw("ph2_y:ph2_x", "ph2_side==1&&ph2_layer==1&&ph2_order==0")
        ttree.SetMarkerColor(2)
        ttree.Draw("ph2_y:ph2_x", "ph2_side==1&&ph2_layer==1&&ph2_order==0&&ph2_module=={}".format(i), "same")
        c1.SaveAs("plots/lin/xy_side1_layer1_order0_module{}.pdf".format(i))
        c1.SaveAs("plots/lin/xy_side1_layer1_order0_module{}.png".format(i))

def xy_side1_order0_moduleall():
    for i in xrange(1, 37):
        ttree.SetMarkerColor(1)
        ttree.Draw("ph2_y:ph2_x", "ph2_side==1&&ph2_order==0")
        ttree.SetMarkerColor(2)
        ttree.Draw("ph2_y:ph2_x", "ph2_side==1&&ph2_order==0&&ph2_module=={}".format(i), "same")
        c1.SaveAs("plots/lin/xy_side1_layer1_order0_module{}.pdf".format(i))
        c1.SaveAs("plots/lin/xy_side1_layer1_order0_module{}.png".format(i))

def rz_isLowerUpper():
    # rz layer 1 barrel
    ttree.SetMarkerColor(1)
    ttree.Draw("sqrt(ph2_x**2+ph2_y**2):ph2_z", "")
    ttree.SetMarkerColor(2)
    ttree.Draw("sqrt(ph2_x**2+ph2_y**2):ph2_z", "ph2_isLower==1", "same")
    c1.SaveAs("plots/lin/rz_isLowerUpper.pdf")
    c1.SaveAs("plots/lin/rz_isLowerUpper.png")

def rz_layerleq3_side3_isLowerUpper():
    # rz layer 1 barrel
    ttree.SetMarkerColor(1)
    ttree.Draw("sqrt(ph2_x**2+ph2_y**2):ph2_z", "ph2_layer<=3&&ph2_side==3")
    ttree.SetMarkerColor(2)
    ttree.Draw("sqrt(ph2_x**2+ph2_y**2):ph2_z", "ph2_layer<=3&&ph2_side==3&&ph2_isLower==1", "same")
    c1.SaveAs("plots/lin/rz_layerleq3_side3_isLowerUpper.pdf")
    c1.SaveAs("plots/lin/rz_layerleq3_side3_isLowerUpper.png")

def rz_layerleq3_side1_isLowerUpper():
    # rz layer 1 barrel
    ttree.SetMarkerColor(1)
    ttree.Draw("sqrt(ph2_x**2+ph2_y**2):ph2_z", "ph2_layer<=3&&ph2_side==1")
    ttree.SetMarkerColor(2)
    ttree.Draw("sqrt(ph2_x**2+ph2_y**2):ph2_z", "ph2_layer<=3&&ph2_side==1&&ph2_isLower==1", "same")
    c1.SaveAs("plots/lin/rz_layerleq3_side1_isLowerUpper.pdf")
    c1.SaveAs("plots/lin/rz_layerleq3_side1_isLowerUpper.png")

def rz_layergeq4_side3_isLowerUpper():
    # rz layer 1 barrel
    ttree.SetMarkerColor(1)
    ttree.Draw("sqrt(ph2_x**2+ph2_y**2):ph2_z", "ph2_layer>=4&&ph2_side==3")
    ttree.SetMarkerColor(2)
    ttree.Draw("sqrt(ph2_x**2+ph2_y**2):ph2_z", "ph2_layer>=4&&ph2_side==3&&ph2_isLower==1", "same")
    c1.SaveAs("plots/lin/rz_layergeq4_side3_isLowerUpper.pdf")
    c1.SaveAs("plots/lin/rz_layergeq4_side3_isLowerUpper.png")

def rz_layergeq4_side1_isLowerUpper():
    # rz layer 1 barrel
    ttree.SetMarkerColor(1)
    ttree.Draw("sqrt(ph2_x**2+ph2_y**2):ph2_z", "ph2_layer>=4&&ph2_side==1")
    ttree.SetMarkerColor(2)
    ttree.Draw("sqrt(ph2_x**2+ph2_y**2):ph2_z", "ph2_layer>=4&&ph2_side==1&&ph2_isLower==1", "same")
    c1.SaveAs("plots/lin/rz_layergeq4_side1_isLowerUpper.pdf")
    c1.SaveAs("plots/lin/rz_layergeq4_side1_isLowerUpper.png")

def rz_layerleq3_notside3_rodall():
    # rz layer 1 barrel
    colors = [1, 2, 4, 8]
    for i in xrange(1, 13):
        ttree.SetMarkerColor(1)
        ttree.Draw("sqrt(ph2_x**2+ph2_y**2):ph2_z", "ph2_layer<=3&&ph2_side!=3&&ph2_rod>=1")
        ttree.SetMarkerColor(2)
        ttree.Draw("sqrt(ph2_x**2+ph2_y**2):ph2_z", "ph2_layer<=3&&ph2_side!=3&&ph2_rod=={}".format(i), "same")
        c1.SaveAs("plots/lin/rz_layerleq3_notside3_rod{}.pdf".format(i))
        c1.SaveAs("plots/lin/rz_layerleq3_notside3_rod{}.png".format(i))

def rz_layerleq3_notside3_moduleall():
    # rz layer 1 barrel
    colors = [1, 2, 4, 8]
    for i in xrange(1, 13):
        ttree.SetMarkerColor(1)
        ttree.Draw("sqrt(ph2_x**2+ph2_y**2):ph2_z", "ph2_layer<=3&&ph2_side!=3&&ph2_module>=1")
        ttree.SetMarkerColor(2)
        ttree.Draw("sqrt(ph2_x**2+ph2_y**2):ph2_z", "ph2_layer<=3&&ph2_side!=3&&ph2_module=={}".format(i), "same")
        c1.SaveAs("plots/lin/rz_layerleq3_notside3_module{}.pdf".format(i))
        c1.SaveAs("plots/lin/rz_layerleq3_notside3_module{}.png".format(i))

def xy_layerleq3_notside3_moduleall():
    # xy layer 1 barrel
    colors = [1, 2, 4, 8]
    for i in xrange(1, 13):
        ttree.SetMarkerColor(1)
        ttree.Draw("ph2_y:ph2_x", "ph2_layer<=3&&ph2_side!=3&&ph2_module>=1")
        ttree.SetMarkerColor(2)
        ttree.Draw("ph2_y:ph2_x", "ph2_layer<=3&&ph2_side!=3&&ph2_module=={}".format(i), "same")
        c1.SaveAs("plots/lin/xy_layerleq3_notside3_module{}.pdf".format(i))
        c1.SaveAs("plots/lin/xy_layerleq3_notside3_module{}.png".format(i))

def xy_layerleq3_side1_moduleall():
    # xy layer 1 barrel
    colors = [1, 2, 4, 8]
    for i in xrange(1, 13):
        ttree.SetMarkerColor(1)
        ttree.Draw("ph2_y:ph2_x", "ph2_layer<=3&&ph2_side==1&&ph2_module>=1")
        ttree.SetMarkerColor(2)
        ttree.Draw("ph2_y:ph2_x", "ph2_layer<=3&&ph2_side==1&&ph2_module=={}".format(i), "same")
        c1.SaveAs("plots/lin/xy_layerleq3_side1_module{}.pdf".format(i))
        c1.SaveAs("plots/lin/xy_layerleq3_side1_module{}.png".format(i))

def xy_layerleq3_side1_module1_rodall():
    # xy layer 1 barrel
    colors = [1, 2, 4, 8]
    for i in xrange(1, 13):
        ttree.SetMarkerColor(1)
        ttree.Draw("ph2_y:ph2_x", "ph2_layer<=3&&ph2_side==1&&ph2_module==1&&ph2_rod>=1")
        ttree.SetMarkerColor(2)
        ttree.Draw("ph2_y:ph2_x", "ph2_layer<=3&&ph2_side==1&&ph2_module==1&&ph2_rod=={}".format(i), "same")
        c1.SaveAs("plots/lin/xy_layerleq3_side1_module1_rod{}.pdf".format(i))
        c1.SaveAs("plots/lin/xy_layerleq3_side1_module1_rod{}.png".format(i))

def xy_layerleq3_notside3_module1_rodall():
    # xy layer 1 barrel
    colors = [1, 2, 4, 8]
    for i in xrange(1, 13):
        ttree.SetMarkerColor(1)
        ttree.Draw("ph2_y:ph2_x", "ph2_layer<=3&&ph2_side!=3&&ph2_module==1&&ph2_rod>=1")
        ttree.SetMarkerColor(2)
        ttree.Draw("ph2_y:ph2_x", "ph2_layer<=3&&ph2_side!=3&&ph2_module==1&&ph2_rod=={}".format(i), "same")
        c1.SaveAs("plots/lin/xy_layerleq3_notside3_module1_rod{}.pdf".format(i))
        c1.SaveAs("plots/lin/xy_layerleq3_notside3_module1_rod{}.png".format(i))

def xy_layerleq3_side1_order0_moduleall():
    # xy layer 1 barrel
    colors = [1, 2, 4, 8]
    for i in xrange(1, 13):
        ttree.SetMarkerColor(1)
        ttree.Draw("ph2_y:ph2_x", "ph2_layer<=3&&ph2_side==1&&ph2_order==0&&ph2_module>=1")
        ttree.SetMarkerColor(2)
        ttree.Draw("ph2_y:ph2_x", "ph2_layer<=3&&ph2_side==1&&ph2_order==0&&ph2_module=={}".format(i), "same")
        c1.SaveAs("plots/lin/xy_layerleq3_side1_order0_module{}.pdf".format(i))
        c1.SaveAs("plots/lin/xy_layerleq3_side1_order0_module{}.png".format(i))

def xy_layerleq3_side2_order0_moduleall():
    # xy layer 1 barrel
    colors = [1, 2, 4, 8]
    for i in xrange(1, 13):
        ttree.SetMarkerColor(1)
        ttree.Draw("ph2_y:ph2_x", "ph2_layer<=3&&ph2_side==2&&ph2_order==0&&ph2_module>=1")
        ttree.SetMarkerColor(2)
        ttree.Draw("ph2_y:ph2_x", "ph2_layer<=3&&ph2_side==2&&ph2_order==0&&ph2_module=={}".format(i), "same")
        c1.SaveAs("plots/lin/xy_layerleq3_side2_order0_module{}.pdf".format(i))
        c1.SaveAs("plots/lin/xy_layerleq3_side2_order0_module{}.png".format(i))

def xy_side1_order1_moduleall():
    print "here"
    # xy layer 1 barrel
    colors = [1, 2, 4, 8]
    for i in xrange(1, 13):
        ttree.SetMarkerColor(1)
        ttree.Draw("ph2_y:ph2_x", "ph2_order==1&&ph2_side==1&&ph2_module>=1")
        ttree.SetMarkerColor(2)
        ttree.Draw("ph2_y:ph2_x", "ph2_order==1&&ph2_side==1&&ph2_module=={}".format(i), "same")
        c1.SaveAs("plots/lin/xy_layerleq3_side1_order1_module{}.pdf".format(i))
        c1.SaveAs("plots/lin/xy_layerleq3_side1_order1_module{}.png".format(i))

def xy_layer5_side1_order1_moduleall():
    # xy layer 1 barrel
    colors = [1, 2, 4, 8]
    for i in xrange(1, 13):
        ttree.SetMarkerColor(1)
        # ttree.Draw("ph2_y:ph2_x", "ph2_ring==12&&ph2_layer==5&&ph2_order==1&&ph2_side==1&&ph2_module>=1")
        ttree.Draw("ph2_y:ph2_x", "ph2_ring==12&&ph2_layer==5&&ph2_order==1&&ph2_side==1&&ph2_module=={}".format(i))
        ttree.SetMarkerColor(2)
        ttree.Draw("ph2_y:ph2_x", "ph2_ring==12&&ph2_layer==5&&ph2_order==1&&ph2_side==1&&ph2_module=={}".format(i), "same")
        c1.SaveAs("plots/lin/xy_layer5_side1_order1_module{}.pdf".format(i))
        c1.SaveAs("plots/lin/xy_layer5_side1_order1_module{}.png".format(i))

def rz_layer5_side1_order1_moduleall():
    # rz layer 1 barrel
    colors = [1, 2, 4, 8]
    for i in xrange(1, 13):
        ttree.SetMarkerColor(1)
        ttree.Draw("sqrt(ph2_x**2+ph2_y**2):ph2_z", "ph2_ring==12&&ph2_layer==5&&ph2_order==1&&ph2_side==1&&ph2_module>=1")
        ttree.SetMarkerColor(2)
        ttree.Draw("sqrt(ph2_x**2+ph2_y**2):ph2_z", "ph2_ring==12&&ph2_layer==5&&ph2_order==1&&ph2_side==1&&ph2_module=={}".format(i), "same")
        c1.SaveAs("plots/lin/rz_layer5_side1_order1_module{}.pdf".format(i))
        c1.SaveAs("plots/lin/rz_layer5_side1_order1_module{}.png".format(i))

def rz_oddmodules():
    ttree.SetMarkerColor(1)
    ttree.Draw("sqrt(ph2_x**2+ph2_y**2):ph2_z", "")
    ttree.SetMarkerColor(2)
    ttree.Draw("sqrt(ph2_x**2+ph2_y**2):ph2_z", "ph2_module%2==1", "same")
    c1.SaveAs("plots/lin/rz_oddmodule.pdf")
    c1.SaveAs("plots/lin/rz_oddmodule.png")

def rz_evenmodules():
    ttree.SetMarkerColor(1)
    ttree.Draw("sqrt(ph2_x**2+ph2_y**2):ph2_z", "")
    ttree.SetMarkerColor(2)
    ttree.Draw("sqrt(ph2_x**2+ph2_y**2):ph2_z", "ph2_module%2==0", "same")
    c1.SaveAs("plots/lin/rz_evenmodule.pdf")
    c1.SaveAs("plots/lin/rz_evenmodule.png")

def rz_layerleq3_oddmodules():
    ttree.SetMarkerColor(1)
    ttree.Draw("sqrt(ph2_x**2+ph2_y**2):ph2_z", "ph2_layer<=3&&ph2_order==0")
    ttree.SetMarkerColor(2)
    ttree.Draw("sqrt(ph2_x**2+ph2_y**2):ph2_z", "ph2_layer<=3&&ph2_order==0&&ph2_module%2==1", "same")
    c1.SaveAs("plots/lin/rz_layerleq3_oddmodule.pdf")
    c1.SaveAs("plots/lin/rz_layerleq3_oddmodule.png")

def rz_layerleq3_evenmodules():
    ttree.SetMarkerColor(1)
    ttree.Draw("sqrt(ph2_x**2+ph2_y**2):ph2_z", "ph2_layer<=3&&ph2_order==0")
    ttree.SetMarkerColor(2)
    ttree.Draw("sqrt(ph2_x**2+ph2_y**2):ph2_z", "ph2_layer<=3&&ph2_order==0&&ph2_module%2==0", "same")
    c1.SaveAs("plots/lin/rz_layerleq3_evenmodule.pdf")
    c1.SaveAs("plots/lin/rz_layerleq3_evenmodule.png")

def rz_layergeq4_oddmodules():
    ttree.SetMarkerColor(1)
    ttree.Draw("sqrt(ph2_x**2+ph2_y**2):ph2_z", "ph2_layer>=4&&ph2_order==0")
    ttree.SetMarkerColor(2)
    ttree.Draw("sqrt(ph2_x**2+ph2_y**2):ph2_z", "ph2_layer>=4&&ph2_order==0&&ph2_module%2==1", "same")
    c1.SaveAs("plots/lin/rz_layergeq4_oddmodule.pdf")
    c1.SaveAs("plots/lin/rz_layergeq4_oddmodule.png")

def rz_layergeq4_evenmodules():
    ttree.SetMarkerColor(1)
    ttree.Draw("sqrt(ph2_x**2+ph2_y**2):ph2_z", "ph2_layer>=4&&ph2_order==0")
    ttree.SetMarkerColor(2)
    ttree.Draw("sqrt(ph2_x**2+ph2_y**2):ph2_z", "ph2_layer>=4&&ph2_order==0&&ph2_module%2==0", "same")
    c1.SaveAs("plots/lin/rz_layergeq4_evenmodule.pdf")
    c1.SaveAs("plots/lin/rz_layergeq4_evenmodule.png")

def xy_layergeq4_evenmodules():
    ttree.SetMarkerColor(1)
    ttree.Draw("ph2_y:ph2_x", "ph2_layer>=4&&ph2_order==0")
    ttree.SetMarkerColor(2)
    ttree.Draw("ph2_y:ph2_x", "ph2_layer>=4&&ph2_order==0&&ph2_module%2==0", "same")
    c1.SaveAs("plots/lin/xy_layergeq4_evenmodule.pdf")
    c1.SaveAs("plots/lin/xy_layergeq4_evenmodule.png")

def rz_layer1_side1_evenmodules():
    ttree.SetMarkerColor(1)
    ttree.Draw("sqrt(ph2_x**2+ph2_y**2):ph2_z", "ph2_layer==1&&ph2_side==1&&ph2_order==1")
    ttree.SetMarkerColor(2)
    ttree.Draw("sqrt(ph2_x**2+ph2_y**2):ph2_z", "ph2_layer==1&&ph2_side==1&&ph2_order==1&&ph2_module%2==0", "same")
    c1.SaveAs("plots/lin/rz_layer1_side1_evenmodule.pdf")
    c1.SaveAs("plots/lin/rz_layer1_side1_evenmodule.png")

def rz_lower():
    ttree.SetMarkerColor(1)
    ttree.Draw("sqrt(ph2_x**2+ph2_y**2):ph2_z", "")
    ttree.SetMarkerColor(2)
    ttree.Draw("sqrt(ph2_x**2+ph2_y**2):ph2_z", "ph2_isLower==1", "same")
    c1.SaveAs("plots/lin/rz_lower.pdf")
    c1.SaveAs("plots/lin/rz_lower.png")

def rz_upper():
    ttree.SetMarkerColor(1)
    ttree.Draw("sqrt(ph2_x**2+ph2_y**2):ph2_z", "")
    ttree.SetMarkerColor(2)
    ttree.Draw("sqrt(ph2_x**2+ph2_y**2):ph2_z", "ph2_isUpper==1", "same")
    c1.SaveAs("plots/lin/rz_upper.pdf")
    c1.SaveAs("plots/lin/rz_upper.png")

def rz_oddmodule_isLower():
    ttree.SetMarkerColor(1)
    ttree.Draw("sqrt(ph2_x**2+ph2_y**2):ph2_z", "")
    ttree.SetMarkerColor(2)
    ttree.Draw("sqrt(ph2_x**2+ph2_y**2):ph2_z", "ph2_isLower==1&&ph2_module%2==1", "same")
    c1.SaveAs("plots/lin/rz_oddmodule_isLower.pdf")
    c1.SaveAs("plots/lin/rz_oddmodule_isLower.png")

def rz_evenmodule_isLower():
    ttree.SetMarkerColor(1)
    ttree.Draw("sqrt(ph2_x**2+ph2_y**2):ph2_z", "")
    ttree.SetMarkerColor(2)
    ttree.Draw("sqrt(ph2_x**2+ph2_y**2):ph2_z", "ph2_isLower==1&&ph2_module%2==0", "same")
    c1.SaveAs("plots/lin/rz_evenmodule_isLower.pdf")
    c1.SaveAs("plots/lin/rz_evenmodule_isLower.png")

def rz_side3_evenmodule_isLower():
    ttree.SetMarkerColor(1)
    ttree.Draw("sqrt(ph2_x**2+ph2_y**2):ph2_z", "ph2_side==3")
    ttree.SetMarkerColor(2)
    ttree.Draw("sqrt(ph2_x**2+ph2_y**2):ph2_z", "ph2_isLower==1&&ph2_module%2==0&&ph2_side==3", "same")
    c1.SaveAs("plots/lin/rz_side3_evenmodule_isLower.pdf")
    c1.SaveAs("plots/lin/rz_side3_evenmodule_isLower.png")

def rz_layerleq3_side3_evenmodule_isLower():
    ttree.SetMarkerColor(1)
    ttree.Draw("sqrt(ph2_x**2+ph2_y**2):ph2_z", "ph2_side==3&&ph2_layer<=3")
    ttree.SetMarkerColor(2)
    ttree.Draw("sqrt(ph2_x**2+ph2_y**2):ph2_z", "ph2_isLower==1&&ph2_module%2==0&&ph2_side==3&&ph2_layer<=3", "same")
    c1.SaveAs("plots/lin/rz_layerleq3_side3_evenmodule_isLower.pdf")
    c1.SaveAs("plots/lin/rz_layerleq3_side3_evenmodule_isLower.png")

def rz_layergeq4_side3_evenmodule_isLower():
    ttree.SetMarkerColor(1)
    ttree.Draw("sqrt(ph2_x**2+ph2_y**2):ph2_z", "ph2_side==3&&ph2_layer>=4")
    ttree.SetMarkerColor(2)
    ttree.Draw("sqrt(ph2_x**2+ph2_y**2):ph2_z", "ph2_isLower==1&&ph2_module%2==0&&ph2_side==3&&ph2_layer>=4", "same")
    c1.SaveAs("plots/lin/rz_layergeq4_side3_evenmodule_isLower.pdf")
    c1.SaveAs("plots/lin/rz_layergeq4_side3_evenmodule_isLower.png")

def rz_layerleq3_side1_evenmodule_isLower():
    ttree.SetMarkerColor(1)
    ttree.Draw("sqrt(ph2_x**2+ph2_y**2):ph2_z", "ph2_side==1&&ph2_layer<=3")
    ttree.SetMarkerColor(2)
    ttree.Draw("sqrt(ph2_x**2+ph2_y**2):ph2_z", "ph2_isLower==1&&ph2_module%2==0&&ph2_side==1&&ph2_layer<=3", "same")
    c1.SaveAs("plots/lin/rz_layerleq3_side1_evenmodule_isLower.pdf")
    c1.SaveAs("plots/lin/rz_layerleq3_side1_evenmodule_isLower.png")

def rz_side1_subdet4_evenmodule_isLower():
    ttree.SetMarkerColor(1)
    ttree.Draw("sqrt(ph2_x**2+ph2_y**2):ph2_z", "ph2_side==1&&ph2_subdet==4")
    ttree.SetMarkerColor(2)
    ttree.Draw("sqrt(ph2_x**2+ph2_y**2):ph2_z", "ph2_isLower==1&&ph2_subdet==4&&ph2_module%2==0&&ph2_side==1", "same")
    c1.SaveAs("plots/lin/rz_side1_subdet4_evenmodule_isLower.pdf")
    c1.SaveAs("plots/lin/rz_side1_subdet4_evenmodule_isLower.png")

def rz_side2_subdet4_evenmodule_isLower():
    ttree.SetMarkerColor(1)
    ttree.Draw("sqrt(ph2_x**2+ph2_y**2):ph2_z", "ph2_side==2&&ph2_subdet==4")
    ttree.SetMarkerColor(2)
    ttree.Draw("sqrt(ph2_x**2+ph2_y**2):ph2_z", "ph2_isLower==1&&ph2_subdet==4&&ph2_module%2==0&&ph2_side==2", "same")
    c1.SaveAs("plots/lin/rz_side2_subdet4_evenmodule_isLower.pdf")
    c1.SaveAs("plots/lin/rz_side2_subdet4_evenmodule_isLower.png")

def rz_side3_specific_detId():
    ttree.SetMarkerColor(1)
    ttree.Draw("sqrt(ph2_x**2+ph2_y**2):ph2_z", "ph2_subdet==5")
    ttree.SetMarkerColor(2)
    ttree.Draw("sqrt(ph2_x**2+ph2_y**2):ph2_z", "ph2_detId==437530645", "same")
    c1.SaveAs("plots/lin/rz_side3_specific_detId.pdf")
    c1.SaveAs("plots/lin/rz_side3_specific_detId.png")

def rz_simhit():
    ttree.SetMarkerColor(1)
    ttree.Draw("sqrt(simhit_x**2+simhit_y**2):simhit_z", "simhit_subdet==5")
    ttree.SetMarkerColor(2)
    ttree.Draw("sqrt(simhit_x**2+simhit_y**2):simhit_z", "simhit_subdet==5", "same")
    c1.SaveAs("plots/lin/rz_simhit.pdf")
    c1.SaveAs("plots/lin/rz_simhit.png")

def rz_side3_pixel():
    ttree.SetMarkerColor(1)
    ttree.Draw("sqrt(ph2_x**2+ph2_y**2):ph2_z", "")
    ttree.SetMarkerColor(2)
    ttree.Draw("sqrt(ph2_x**2+ph2_y**2):ph2_z", "ph2_isLower==0", "same")
    c1.SaveAs("plots/lin/rz_side3_pixel.pdf")
    c1.SaveAs("plots/lin/rz_side3_pixel.png")

def xy_side1_order1_moduleall_individ():
    print "here"
    # xy layer 1 barrel
    colors = [1, 2, 4, 8]
    for i in xrange(1, 13):
        ttree.SetMarkerColor(1)
        ttree.Draw("ph2_y:ph2_x", "ph2_order==1&&ph2_side==1&&ph2_module=={}".format(i))
        c1.SaveAs("plots/lin/xy_layerleq3_side1_order1_module{}.pdf".format(i))
        c1.SaveAs("plots/lin/xy_layerleq3_side1_order1_module{}.png".format(i))

def rz_side1_subdet1_even_detId_even_module():
    ttree.SetMarkerColor(1)
    ttree.Draw("sqrt(ph2_x**2+ph2_y**2):ph2_z", "ph2_subdet==5&&ph2_side==1")
    ttree.SetMarkerColor(2)
    ttree.Draw("sqrt(ph2_x**2+ph2_y**2):ph2_z", "ph2_subdet==5&&ph2_side==1&&ph2_detId%2==0&&ph2_module%2==0", "same")
    c1.SaveAs("plots/lin/rz_side1_subdet1_even_detId_even_module.pdf")
    c1.SaveAs("plots/lin/rz_side1_subdet1_even_detId_even_module.png")

def rz_side1_subdet1_even_detId_even_module():
    ttree.SetMarkerColor(1)
    ttree.Draw("sqrt(ph2_x**2+ph2_y**2):ph2_z", "ph2_subdet==5&&ph2_side==1")
    ttree.SetMarkerColor(2)
    ttree.Draw("sqrt(ph2_x**2+ph2_y**2):ph2_z", "ph2_subdet==5&&ph2_side==1&&ph2_detId%2==0&&ph2_module%2==0", "same")
    c1.SaveAs("plots/lin/rz_side1_subdet1_even_detId_even_module.pdf")
    c1.SaveAs("plots/lin/rz_side1_subdet1_even_detId_even_module.png")

def rz_side1_subdet1_even_detId_even_module_rod11():
    ttree.SetMarkerColor(1)
    ttree.Draw("sqrt(ph2_x**2+ph2_y**2):ph2_z", "ph2_subdet==5&&ph2_side==1&&ph2_rod==11")
    ttree.SetMarkerColor(2)
    ttree.Draw("sqrt(ph2_x**2+ph2_y**2):ph2_z", "ph2_subdet==5&&ph2_side==1&&ph2_detId%2==0&&ph2_module%2==0&&ph2_rod==11", "same")
    c1.SaveAs("plots/lin/rz_side1_subdet1_even_detId_even_module_rod11.pdf")
    c1.SaveAs("plots/lin/rz_side1_subdet1_even_detId_even_module_rod11.png")

def rz_side1_subdet1_even_detId_even_module_rod12():
    ttree.SetMarkerColor(1)
    ttree.Draw("sqrt(ph2_x**2+ph2_y**2):ph2_z", "ph2_subdet==5&&ph2_side==1&&ph2_rod==12")
    ttree.SetMarkerColor(2)
    ttree.Draw("sqrt(ph2_x**2+ph2_y**2):ph2_z", "ph2_subdet==5&&ph2_side==1&&ph2_detId%2==0&&ph2_module%2==0&&ph2_rod==12", "same")
    c1.SaveAs("plots/lin/rz_side1_subdet1_even_detId_even_module_rod12.pdf")
    c1.SaveAs("plots/lin/rz_side1_subdet1_even_detId_even_module_rod12.png")

def xy_side1_subdet1_even_detId_even_module_rod11():
    ttree.SetMarkerColor(1)
    ttree.Draw("ph2_y:ph2_x", "ph2_subdet==5&&ph2_side==1&&ph2_rod==11")
    ttree.SetMarkerColor(2)
    ttree.Draw("ph2_y:ph2_x", "ph2_subdet==5&&ph2_side==1&&ph2_detId%2==0&&ph2_module%2==0&&ph2_rod==11", "same")
    c1.SaveAs("plots/lin/xy_side1_subdet1_even_detId_even_module_rod11.pdf")
    c1.SaveAs("plots/lin/xy_side1_subdet1_even_detId_even_module_rod11.png")

def xy_side1_subdet1_even_detId_even_module_rod12():
    ttree.SetMarkerColor(1)
    ttree.Draw("ph2_y:ph2_x", "ph2_subdet==5&&ph2_side==1&&ph2_rod==12")
    ttree.SetMarkerColor(2)
    ttree.Draw("ph2_y:ph2_x", "ph2_subdet==5&&ph2_side==1&&ph2_detId%2==0&&ph2_module%2==0&&ph2_rod==12", "same")
    c1.SaveAs("plots/lin/xy_side1_subdet1_even_detId_even_module_rod12.pdf")
    c1.SaveAs("plots/lin/xy_side1_subdet1_even_detId_even_module_rod12.png")

def xy_specific():
    ttree.SetMarkerColor(1)
    mods = [443360282,
     443360281,
     443360285,
     443360286,
     437530646,
     437530645,
     437529622,
     437529621,
    ]
    mods = [
    437523494,
    438570038,
    439616586,
    439616589,
    411341905,
    411337809,
    411612262,
    411866222,
    412136585,
    ]
    cuts = []
    for mod in mods:
        cuts.append("ph2_detId=={}".format(mod))
    cut = "||".join(cuts)
    ttree.Draw("ph2_y:ph2_x", "")
    ttree.SetMarkerColor(2)
    ttree.Draw("ph2_y:ph2_x", cut, "same")
    c1.SaveAs("plots/lin/xy_specific.pdf")
    c1.SaveAs("plots/lin/xy_specific.png")

def rz_specific():
    ttree.SetMarkerColor(1)
    mods = [443360282,
     443360281,
     443360285,
     443360286,
     437530646,
     437530645,
     437529622,
     437529621,
    ]
    mods = [
    437523494,
    438570038,
    439616586,
    439616589,
    411341905,
    411337809,
    411612262,
    411866222,
    412136585,
    ]
    cuts = []
    for mod in mods:
        cuts.append("ph2_detId=={}".format(mod))
    cut = "||".join(cuts)
    ttree.Draw("sqrt(ph2_x**2+ph2_y**2):ph2_z", "")
    ttree.SetMarkerColor(2)
    ttree.Draw("sqrt(ph2_x**2+ph2_y**2):ph2_z", cut, "same")
    c1.SaveAs("plots/lin/rz_specific.pdf")
    c1.SaveAs("plots/lin/rz_specific.png")

def rz_connection_specific():
    # source = 437789750
    # target = [438840393, 438840398, 438841422, 438841417, 438839369, 438839374, 439110689, 419959874, 438837321, 438839366, 438835270, 439109665, 439103534, 438831177, 438841430, 419701845]
    source = 437782542
    target = [438831121, 438831126, 438832145, 438830097, 438831118, 438830094, 438830102, 438832150, 438832142, 438833169, 439094318, 438833166]
    ttree.SetMarkerColor(1)
    ttree.Draw("sqrt(ph2_x**2+ph2_y**2):ph2_z", "")
    cuts = []
    for mod in target:
        cuts.append("ph2_detId=={}".format(mod))
    cut = "||".join(cuts)
    ttree.SetMarkerColor(2)
    ttree.Draw("sqrt(ph2_x**2+ph2_y**2):ph2_z", cut, "same")
    cuts = []
    for mod in [source]:
        cuts.append("ph2_detId=={}".format(mod))
    cut = "||".join(cuts)
    ttree.SetMarkerColor(4)
    ttree.Draw("sqrt(ph2_x**2+ph2_y**2):ph2_z", cut, "same")
    c1.SaveAs("plots/lin/rz_connection_specific.pdf")
    c1.SaveAs("plots/lin/rz_connection_specific.png")

def rz_specific_track():
    ttree.SetMarkerColor(1)
    ttree.Draw("sqrt(ph2_x**2+ph2_y**2):ph2_z", "")
    ttree.SetMarkerColor(2)
    ttree.Draw("sqrt(simhit_x**2+simhit_y**2):simhit_z", "(Entry$==2)&&(simhit_simTrkIdx==4)&&(abs(simhit_particle)==13)&&(simhit_subdet==4||simhit_subdet==5)", "same")
    c1.SaveAs("plots/lin/rz_specific_track.pdf")
    c1.SaveAs("plots/lin/rz_specific_track.png")

def rz_specific_track_and_module():
    ttree.SetMarkerColor(16)
    ttree.Draw("sqrt(ph2_x**2+ph2_y**2):ph2_z", "ph2_z>0")
    mods = [
    437523494,
    438570038,
    439616586,
    439616589,
    411341905,
    411337809,
    411612262,
    411866222,
    412136585,
    ]
    mods = [
    438046750,
    438046745,
    439097385,
    440148025,
    441198658,
    441197634,
    442248261,
    443296845,
    420258821,
    ]
    cuts = []
    for mod in mods:
        cuts.append("ph2_detId=={}".format(mod))
    cut = "||".join(cuts)
    ttree.SetMarkerColor(4)
    ttree.Draw("sqrt(ph2_x**2+ph2_y**2):ph2_z", cut, "same")
    ttree.SetMarkerColor(2)
    ttree.SetMarkerSize(20)
    ttree.Draw("sqrt(simhit_x**2+simhit_y**2):simhit_z", "(event==7399)&&(simhit_simTrkIdx==4)&&(abs(simhit_particle)==13)&&(simhit_subdet==4||simhit_subdet==5)", "same")
    c1.SaveAs("plots/lin/rz_specific_track_and_module.pdf")
    c1.SaveAs("plots/lin/rz_specific_track_and_module.png")

def xy_specific_track_and_module():
    ttree.SetMarkerColor(16)
    ttree.Draw("ph2_y:ph2_x", "")
    mods = [
    437523494,
    438570038,
    439616586,
    439616589,
    411341905,
    411337809,
    411612262,
    411866222,
    412136585,
    ]
    mods = [
    438046750,
    438046745,
    439097385,
    440148025,
    441198658,
    441197634,
    442248261,
    443296845,
    420258821,
    ]
    cuts = []
    for mod in mods:
        cuts.append("ph2_detId=={}".format(mod))
    cut = "||".join(cuts)
    ttree.SetMarkerColor(4)
    ttree.Draw("ph2_y:ph2_x", cut, "same")
    ttree.SetMarkerColor(2)
    ttree.SetMarkerSize(20)
    ttree.Draw("simhit_y:simhit_x", "(event==7399)&&(simhit_simTrkIdx==4)&&(abs(simhit_particle)==13)&&(simhit_subdet==4||simhit_subdet==5)", "same")
    c1.SaveAs("plots/lin/xy_specific_track_and_module.pdf")
    c1.SaveAs("plots/lin/xy_specific_track_and_module.png")

if __name__ == "__main__":

    # rz_side1_layer1_order0_moduleall()
    # xy_side1_layer1_order0_moduleall()
    # xy_side1_order0_moduleall()
    # rz_isLowerUpper()
    # rz_side3_isLowerUpper()
    # rz_layerleq3_side3_isLowerUpper()
    # rz_layergeq4_side1_isLowerUpper()
    # rz_layergeq4_side3_isLowerUpper()
    # rz_layerleq3_notside3_rodall()
    # rz_layerleq3_notside3_moduleall()
    # xy_layerleq3_notside3_moduleall()
    # xy_layerleq3_side1_moduleall()
    # xy_layerleq3_side1_module1_rodall()
    # xy_layerleq3_notside3_module1_rodall()
    # xy_layerleq3_side1_order0_moduleall()
    # xy_layerleq3_side2_order0_moduleall()
    # xy_side1_order1_moduleall()
    # xy_layer5_side1_order1_moduleall()
    # rz_layer5_side1_order1_moduleall()
    # rz_subdet4()
    # rz_oddmodules()
    # rz_evenmodules()
    # rz_layerleq3_oddmodules()
    # rz_layerleq3_evenmodules()
    # rz_layergeq4_oddmodules()
    # rz_layergeq4_evenmodules()
    # xy_layergeq4_evenmodules()
    # rz_layer1_side1_evenmodules()
    # rz_lower()
    # rz_upper()
    # rz_oddmodule_isLower()
    # rz_evenmodule_isLower()
    # rz_side3_evenmodule_isLower()
    # rz_layerleq3_side3_evenmodule_isLower()
    # rz_layergeq4_side3_evenmodule_isLower()
    # rz_layerleq3_side1_evenmodule_isLower()
    # rz_side1_evenmodule_isLower()
    # rz_side1_subdet4_evenmodule_isLower()
    # rz_side2_subdet4_evenmodule_isLower()
    # rz_side3_specific_detId()
    # rz_simhit()
    # rz_side3_pixel()
    # xy_side1_order1_moduleall_individ()
    # rz_side1_subdet1_even_detId_even_module()
    # rz_side1_subdet1_even_detId_even_module_rod11()
    # rz_side1_subdet1_even_detId_even_module_rod12()
    # xy_side1_subdet1_even_detId_even_module_rod11()
    # xy_side1_subdet1_even_detId_even_module_rod12()
    # xy_specific()
    # rz_specific()
    # rz_connection_specific()
    # rz_specific_track()
    rz_specific_track_and_module()
    xy_specific_track_and_module()

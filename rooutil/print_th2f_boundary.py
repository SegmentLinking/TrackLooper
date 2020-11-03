#!/bin/env python

import ROOT as r
import sys

def usage():
    print "Usage:"
    print ""
    print "  python {} ROOTPATH:HistName (e.g. /nfs-7/userdata/phchang/scalefactors/EGM2D_BtoH_GT20GeV_RecoSF_Legacy2016.root:EGamma_SF2D)".format(sys.argv[0])
    print ""
    print ""
    print ""
    sys.exit()
    return

if len(sys.argv) <= 1:
    usage()

histfullpath = sys.argv[1]
rootfilepath = histfullpath.split(":")[0]
histname = histfullpath.split(":")[1]
tf = r.TFile(rootfilepath)
th2 = tf.Get(histname)

print "=========="
print rootfilepath
print histname
print "=========="

print "X-axis range"
# for i in xrange(1, th2.GetNbinsX()+1):
#     print th2.GetXaxis().GetBinLowEdge(i)
print "({}, {})".format(th2.GetXaxis().GetBinLowEdge(1), th2.GetXaxis().GetBinUpEdge(th2.GetNbinsX()))

print "Y-axis range"
# for i in xrange(1, th2.GetNbinsY()+1):
#     print th2.GetYaxis().GetBinLowEdge(i)
print "({}, {})".format(th2.GetYaxis().GetBinLowEdge(1), th2.GetYaxis().GetBinUpEdge(th2.GetNbinsY()))

print
print

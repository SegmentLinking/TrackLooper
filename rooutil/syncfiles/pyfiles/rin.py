#!/usr/bin/env python

"""
Given rootfile(s), this script prints out the most useful treename, 
nevents, scale1fb, kfactor. Optionally can be told to only spit out
nevents. This works on CMS3, SS babies, most other root files, and
even centrally hosted root files via xrootd (must begin with
'/store/' to be recognized as one of these).
"""

import sys
import argparse
import os
import glob

# from ROOT import TChain, TFile, TTree
import ROOT as r

def get_treename_from_file(tfile):
    """
    Given a tfile, this returns the name of the (only) ttree in the file, or
    "Events" in the case of MINIAOD
    """
    keys = tfile.GetListOfKeys()
    treenames = [key.GetName() for key in keys if key.ReadObj().InheritsFrom(r.TTree.Class())]
    if len(treenames) > 0 and "Events" in treenames: treename = "Events"
    else: treename = treenames[0]
    return treename

def main(fpatt, nevents_only):

    if fpatt.startswith("/store/"):
        fpatt = "root://cmsxrootd.fnal.gov//" + fpatt

    fname = fpatt
    if "*" in fpatt or "[" in fpatt or "]" in fpatt:
        fname = glob.glob(fpatt)[0]

    f1 = r.TFile.Open(fname)
    treename = get_treename_from_file(f1)

    ch = r.TChain(treename)
    ch.Add(fpatt)
    nevts = ch.GetEntries()

    scale1fb = -1.0
    kfactor = -1.0
    xsec = -1.0
    if not nevents_only:
        for evt in ch:
            try:
                scale1fb = evt.evt_scale1fb
                kfactor = evt.evt_kfactor
                xsec = evt.evt_xsec_incl
            except:
                scale1fb = evt.scale1fb
                kfactor = evt.kfactor
                xsec = evt.xsec
            break

    if not nevents_only:
        print "treename: %s" % treename
        print "nevts: %i" % nevts
        print "scale1fb: %f" % scale1fb
        print "kfactor: %f" % kfactor
        print "xsec: %f" % xsec
    else:
        print nevts

def test():
    main("/hadoop/cms/store/group/snt/run2_data/Run2016H_DoubleEG_MINIAOD_PromptReco-v3/merged/V08-00-15/merged_ntuple_1.root", True)
    main("/nfs-7/userdata/ss2015/ssBabies/v8.07/DataDoubleEG.root", True)
    main("/store/user/namin/tth_m350_MINIAOD/MINIAOD_18.root", True)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("files", help="input file(s), quoted if there are wildcards")
    parser.add_argument("-n", "--nevents", help="show nevents only", action="store_true")
    args = parser.parse_args()

    main(args.files, args.nevents)


#!/bin/env python

import ROOT as r
import math
import sys

f = open("scripts/tilted_hits_sorted.txt")
lines = f.readlines()
hits = {}

# Parse and store hits
for line in lines:
    ls = line.split()
    detid = ls[0]
    hit = (float(ls[2]), float(ls[3]), float(ls[1])) # NOTE in txt file we have z, y, x coordinates 
    if detid not in hits:
        hits[detid] = []
    hits[detid].append(hit)

# Computing two groups of hits
print "# detid drdz xy-slope"
for detid in hits:

    # Number of events
    n = len(hits[detid])

    # There are two groups of hits

    azs = []
    for ii, hit in enumerate(hits[detid]):
        azs.append(abs(hit[2]))
    azs = list(set(azs))
    azs.sort()

    # Loop again and group them into two groups
    hl = [] # low-z hits
    hh = [] # high-z hits
    xls = []
    xhs = []
    for ii, hit in enumerate(hits[detid]):
        az = abs(hit[2])
        # put them into two buckets
        if az == azs[0]:
            hl.append(hit)
            xls.append(hit[0])
        else:
            hh.append(hit)
            xhs.append(hit[0])

    # Create two TGraph's for fit
    gl = r.TGraph(len(hl))
    gh = r.TGraph(len(hh))

    for ii, hit in enumerate(hl):
        gl.SetPoint(ii, hit[0], hit[1])

    # if lying 90 degrees in x-y plane the fit will fail with infinite slope
    # so take care of it as a special case

    if len(list(set(xls))) != 1:

        rl = gl.Fit("pol1", "q") # Result of low hits fit
        yl = r.gROOT.FindObject("pol1").GetParameter(0)
        sl = r.gROOT.FindObject("pol1").GetParameter(1)

        for ii, hit in enumerate(hh):
            gh.SetPoint(ii, hit[0], hit[1])

        rh = gh.Fit("pol1", "q") # Result of high hits fit
        yh = r.gROOT.FindObject("pol1").GetParameter(0)
        sh = r.gROOT.FindObject("pol1").GetParameter(1)

        if abs(sl - sh) > 0.005:
            print "ERROR"

        if abs(yh-yl)/math.sqrt(sl**2+1) == 0:
            print ""
            for h in hl:
                print h
            print ""
            for h in hh:
                print h
            rl = gl.Fit("pol0", "q") # Result of low hits fit
            yl = r.gROOT.FindObject("pol0").GetParameter(0)
            print yl
            sys.exit()

        print detid, abs(yh-yl)/math.sqrt(sl**2+1)/abs(azs[0]-azs[1]), sl #, abs(yh-yl)/math.sqrt(sl**2+1), abs(azs[0] - azs[1])

    else:

        print detid, abs(list(set(xls))[0]-list(set(xhs))[0])/ abs(azs[0] - azs[1]), 123456789 #, abs(list(set(xls))[0]-list(set(xhs))[0]), abs(azs[0] - azs[1])


#!/bin/env python

import ROOT as r

f = open("scripts/endcap_hits_sorted.txt")
lines = f.readlines()
hits = {}

# Parse and store hits
for line in lines:
    ls = line.split()
    detid = ls[3]
    hit = (float(ls[5]), float(ls[7]))
    if detid not in hits:
        hits[detid] = []
    hits[detid].append(hit)

# Computing two groups of hits
print "# detid average_r2s y_intercept_low_hits slope_low_hits y_intercept_high_hits slope_high_hits"
for detid in hits:

    # Number of events
    n = len(hits[detid])

    # There are two groups of hits
    # The average value of r^2's will be used to divide the hits into two groups

    # Trying to group into two hits
    r2s = []

    # compute r^2 = x^2 + y^2
    sumr2s = 0
    for ii, hit in enumerate(hits[detid]):
        r2 = hit[0]**2 + hit[1]**2
        r2s.append(r2)
        sumr2s += r2

    # The average value
    avgr2s = sumr2s / n

    # Loop again and group them into two groups
    hl = [] # low hits
    hh = [] # high hits
    for ii, hit in enumerate(hits[detid]):
        r2 = hit[0]**2 + hit[1]**2
        # put them into two buckets
        if r2 < avgr2s:
            hl.append(hit)
        else:
            hh.append(hit)

    # Create two TGraph's for fit
    gl = r.TGraph(len(hl))
    gh = r.TGraph(len(hh))

    for ii, hit in enumerate(hl):
        gl.SetPoint(ii, hit[0], hit[1])

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

    print detid, avgr2s, yl, sl, yh, sh


#!/bin/env python

import ROOT as r
import sys

def help():
    print "Usage:"
    print "  python print_cms4_eventlist.py ROOTFILEPATH"
    print ""
    sys.exit()

try:
    fname = sys.argv[1]
except:
    help()

f = r.TFile(fname)
t = f.Get("t")

eventlist = {}

for event in t:
    if str(event.CMS4path) not in eventlist:
        eventlist[str(event.CMS4path)] = []
    eventlist[str(event.CMS4path)].append(event.CMS4index)

for cms4path in eventlist:
    print cms4path, len(eventlist[cms4path]), " ".join([ str(x) for x in eventlist[cms4path] ])


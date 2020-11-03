#!/usr/bin/env python

import math, sys, os
import argparse
OKGREEN = '\033[92m'
FAIL = '\033[91m'
ENDC = '\033[0m'

def parse_bin_settings(bin_settings, ndim=1):
    d = {"nbinsx": 25, "xlow": -1, "xhigh": -2, "nbinsy": 25, "ylow": -1, "yhigh": -2}
    if not bin_settings: return d

    bs = map(float,bin_settings.split(","))
    if len(bs) == 1 and ndim == 1:
        d["nbinsx"] = int(bs[0])
    elif len(bs) == 2 and ndim == 2:
        d["nbinsx"] = int(bs[0])
        d["nbinsy"] = int(bs[1])
    elif len(bs) == 3 and ndim == 1:
        d["nbinsx"] = int(bs[0])
        d["xlow"] = bs[1]
        d["xhigh"] = bs[2]
    elif len(bs) == 3 and ndim == 2:
        d["nbinsx"] = int(bs[0])
        d["xlow"] = bs[1]
        d["xhigh"] = bs[2]
        d["nbinsy"] = int(bs[0])
        d["ylow"] = bs[1]
        d["yhigh"] = bs[2]
    elif len(bs) == 6 and ndim == 2:
        d["nbinsx"] = int(bs[0])
        d["xlow"] = bs[1]
        d["xhigh"] = bs[2]
        d["nbinsy"] = int(bs[3])
        d["ylow"] = bs[4]
        d["yhigh"] = bs[5]
    else:
        print "not sure how to interpret %s" % bin_settings

    return d


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--bin_settings", help="nbins *OR* nbinsx,nbinsy *OR* nbins,xlow,xhigh *OR* nbinsx,xlow,xhigh,nbinsy,ylow,yhigh")
    parser.add_argument("-n", "--name", help="histogram name and filename")
    parser.add_argument("-d", "--drawopt", help="draw options, like 'TEXTE'")
    parser.add_argument("-s", "--nostatbox", help="don't show statbox", action="store_true")
    parser.add_argument("-l", "--log", help="logscale", action="store_true")
    parser.add_argument("-ll", "--loglog", help="loglogscale", action="store_true")
    parser.add_argument("-x", "--profx", help="x profile of 2d hist", action="store_true")
    parser.add_argument("-y", "--profy", help="y profile of 2d hist", action="store_true")
    args = parser.parse_args()

    name = args.name or "hist"
    outname = "%s.pdf" % name.replace("_","").replace(" ","_").replace("^","")
    drawopt = args.drawopt or ""

    rows = []
    for line in sys.stdin:
        line = line.strip()
        try: parts = map(float, line.split())
        except: continue
        rows.append(parts)

    # get most common value for number of columns and ensure all values have that many cols
    sizes = [len(row) for row in rows]
    best_size = max(set(sizes), key=sizes.count)
    rows = [row for row in rows if len(row) == best_size]

    dbin = parse_bin_settings(args.bin_settings, best_size)
    print "Making %iD histogram (%s) with settings:" % (best_size, name)
    print "\tnbinsx,xlow,xhigh = %i,%.1f,%.1f" % (dbin["nbinsx"],dbin["xlow"],dbin["xhigh"])
    if best_size > 1:
        print "\tnbinsy,ylow,yhigh = %i,%.1f,%.1f" % (dbin["nbinsy"],dbin["ylow"],dbin["yhigh"])
    print "\tstatbox: %s, draw options: %s" % (str(not args.nostatbox), drawopt)

    import ROOT as r
    c1 = r.TCanvas("c1")
    if args.nostatbox: r.gStyle.SetOptStat(0)

    if best_size == 1:
        h1 = r.TH1F("vals", name, dbin["nbinsx"],dbin["xlow"],dbin["xhigh"])
        for row in rows:
            h1.Fill(row[0])

        if args.log: c1.SetLogy()

        h1.Draw(drawopt)
        c1.SaveAs(outname)

    elif best_size == 2:
        h2 = r.TH2F("vals", name, dbin["nbinsx"],dbin["xlow"],dbin["xhigh"],dbin["nbinsy"],dbin["ylow"],dbin["yhigh"])
        for row in rows:
            h2.Fill(row[0], row[1])

        if args.log: c1.SetLogz()
        if args.loglog:
            c1.SetLogx()
            c1.SetLogy()
        
        if args.profx:
            prof = h2.ProfileX()
            prof.Draw(drawopt)
        elif args.profy:
            prof = h2.ProfileY()
            prof.Draw(drawopt)
        else:
            h2.Draw(drawopt)
        c1.SaveAs(outname)

    else:

        print "I don't know how to handle %i columns" % best_size
        sys.exit()

    os.system("cat %s | ic" % outname)

    os.system("cp %s ~/public_html/dump/" % outname)
    print "%s\nOutput: uaf-6.t2.ucsd.edu/~namin/dump/%s.pdf %s" % (OKGREEN, name, ENDC)


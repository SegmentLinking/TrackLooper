#!/usr/bin/env python

import argparse
import datetime
import glob
import os
import sys
import time
import commands
import math

def linfit(xs,ys):
    n = len(xs)
    sumx, sumy = sum(xs), sum(ys)
    sumxx, sumyy = sum([x*x for x in xs]), sum([y*y for y in ys])
    sumxy = sum([xs[i]*ys[i] for i in range(len(xs))])
    avgx, avgy = 1.0*sumx/n, 1.0*sumy/n
    ssxx, ssyy = sumxx-n*avgx*avgx, sumyy-n*avgy*avgy
    ssxy = sumxy-n*avgx*avgy
    m = 1.0*ssxy/ssxx
    b = avgy-m*avgx
    try:
        s = math.sqrt((ssyy-m*ssxy)/(n-2))
        errorm = s/math.sqrt(ssxx)
        errorb = s/math.sqrt(1.0/n+avgx*avgx/ssxx)
        if(n == 2): errorm, errorb = 0.0, 0.0
        return m,b, errorm, errorb
    except Exception as e:
        print "ERROR:",m,b,ssxx,ssyy,ssxy,avgx,n
        return m,b,-1,-1

def get_info(cid):
    stat, out = commands.getstatusoutput("condor_tail {0} -maxbytes 100000 -stderr".format(cid))

    lines = out.splitlines()

    processingtuples = []
    for line in lines:
        if line.startswith("Begin processing the"):
            record = float("".join([b for b in line.split("record")[0].split("the")[-1] if b in "1234567890"]))
            to_parse = " ".join(line.split()[-3:-1])
            dtobj = datetime.datetime.strptime( to_parse, "%d-%b-%Y %H:%M:%S.%f" )
            ts = time.mktime(dtobj.timetuple())+(dtobj.microsecond/1.e6)
            processingtuples.append([record,dtobj,ts])

    xs = [pp[0] for pp in processingtuples]
    ys = [pp[2] for pp in processingtuples]
    mint = min(ys)
    ys = map(lambda y: y-mint, ys)
    m, b, merr, berr = linfit(xs,ys)

    minago = (datetime.datetime.now()-processingtuples[-1][1]+datetime.timedelta(hours=7)).total_seconds() / 60
    return {
            "last_update_mins_ago": minago,
            "last_nevents": processingtuples[-1][0],
            "avg_event_rate": 1.0/m,
            "last_lines": lines[-20:],
            }

def print_info(d):
    print "Last 20 lines ---->"
    print "\t"+"\n\t".join(d["last_lines"])
    print "<----"
    print "Last update was {:.0f} minutes ago with {:.0f} events".format(d["last_update_mins_ago"],d["last_nevents"])
    print "Average event rate: {0:.1f}Hz".format(d["avg_event_rate"])


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("condorid", help="condor job id")
    args = parser.parse_args()

    d = get_info(args.condorid)
    print_info(d)

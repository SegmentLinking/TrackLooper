#!/bin/env python

import argparse
parser = argparse.ArgumentParser(description="duplicate event removal and create a new set of TTrees")
parser.add_argument('--output', '-o', dest='output', default='output.root', help='output root file path')
parser.add_argument('--treename', '-t', dest='treename', default='t', help='treename')
parser.add_argument('--size', '-s', dest='size', help='max tree size in MB')
parser.add_argument('files', metavar='FILE.root', type=str, nargs='+', help='input files')

args = parser.parse_args()

import ROOT as r
import os
r.gROOT.SetBatch(True)
thispypathdir = os.path.dirname(os.path.realpath(__file__))
r.gSystem.Load(os.path.join(thispypathdir, "rooutil.so"))
r.gROOT.ProcessLine('.L {}'.format(os.path.join(thispypathdir, "scripts.h")))

chain = r.TChain(args.treename)
for rfile in args.files:
    chain.Add(rfile)

print chain.GetEntries()

if not args.size:
    args.size = 500*1048576

r.RooUtil.split_files(chain, args.output, int(args.size))

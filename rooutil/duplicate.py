#!/bin/env python

import argparse
parser = argparse.ArgumentParser(description="duplicate event removal and create a new set of TTrees")
parser.add_argument('--output', '-o', dest='output', default='output.root', help='output root file path')
parser.add_argument('--size', '-s', dest='size', help='max tree size')
parser.add_argument('--treename', '-t', dest='treename', default='t', help='treename')
parser.add_argument('--evt', '-e', dest='evt', default='evt', help='event branch name')
parser.add_argument('--run', '-r', dest='run', default='run', help='run branch name')
parser.add_argument('--lumi', '-l', dest='lumi', default='lumi', help='lumi branch name')
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
    args.size = 0

r.RooUtil.remove_duplicate(chain, args.output, args.run, args.lumi, args.evt, args.size)

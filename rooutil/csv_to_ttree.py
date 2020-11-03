#!/bin/env python

import argparse
parser = argparse.ArgumentParser(description="From series of CSVs create a TTree")
parser.add_argument('--output', '-o', dest='output', default='output.root', help='output root file path')
parser.add_argument('--treename', '-t', dest='treename', default='t', help='treename')
parser.add_argument('--delim', '-d', dest='delim', default=',', help='delimiter used in the file')
parser.add_argument('--qevtlist', '-q', dest='qevtlist', action='store_true', default=False, help='the format is from the qframework')
parser.add_argument('--branchdesc', '-b', dest='branchdesc', default='', help='branch descriptor (See TTree::ReadFile)')
parser.add_argument('files', metavar='FILE.csv', type=str, nargs='+', help='input files')

args = parser.parse_args()

import ROOT as r

print args.branchdesc
print args.files

ofile = r.TFile.Open(args.output, 'recreate')

if args.qevtlist:

    ttree = r.TTree(args.treename, args.treename)
    for index, csvfile in enumerate(args.files):
        f = open("/tmp/csv_to_ttree.txt", "w")
        lines = open(csvfile).readlines()
        for iline, line in enumerate(lines):
            if iline == 0:
                f.write(':'.join(line.split())+'\n')
            else:
                if line.find("---") != -1:
                    continue
                else:
                    f.write(line)
        f.close()
        ttree.ReadFile("/tmp/csv_to_ttree.txt")
    
    ttree.Write()

else:

    ttree = r.TTree(args.treename, args.treename)
    for index, csvfile in enumerate(args.files):
        if index == 0:
            ttree.ReadFile(csvfile, args.branchdesc, args.delim)
        else:
            ttree.ReadFile(csvfile)
    
    ttree.Write()


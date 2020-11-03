#!/bin/env python

#
# python get_lumi.py CMS4DATA.root
#

# To create lumi.dat

#{
#    TFile *_file0 = TFile::Open("/hadoop/cms/store/group/snt/run2_data2017/DoubleMuon_Run2017B-31Mar2018-v1_MINIAOD_CMS4_V09-04-12/merged_ntuple_1.root");
#    ((TTreePlayer*)(Events->GetPlayer()))->SetScanRedirect(true); 
#    ((TTreePlayer*)(Events->GetPlayer()))->SetScanFileName("lumi.dat"); 
#    Events->Scan("evt_run:evt_lumiBlock","","");
#    // See here for more options to "TTree::Scan" https://root.cern.ch/doc/master/classTTreePlayer.html#aa0149b416e4b812a8762ec1e389ba2db
#    // For instance, the column width and format can be modified by doing something like: tree->Scan("a:b:c","","colsize=30 precision=3 col=::20.10:#x:5ld");
#}

import ROOT as r
import sys

lumimap = {}
lumis_file = open("/home/users/namin/public_html/dump/lumis.csv")
lumilines = lumis_file.readlines()
for line in lumilines:
    if line.find("#") != -1:
        continue
    if line.find(",") == -1:
        continue
    run = int(line.split(",")[0].split(":")[0])
    lumi = int(line.split(",")[1].split(":")[0])
    intlumi = float(line.split(",")[6])
    lumimap["{} {}".format(run, lumi)] = intlumi
#297050:5839,1:1,06/16/17 20:51:29,STABLE BEAMS,6500,0.160,0.126,31.7,BCM1F

fname = sys.argv[1]
tfile = r.TFile(fname)
ttree = tfile.Get("Events")
ttree.GetPlayer().SetScanRedirect(True)
ttree.GetPlayer().SetScanFileName("lumi.dat")
ttree.Scan("evt_run:evt_lumiBlock","","")
f = open("lumi.dat")
lines = f.readlines()
run_lumi = []
for index, line in enumerate(lines):
    if index < 3:
        continue
    if index == len(lines) - 1:
        continue
    run = int(line.split("*")[2].strip())
    lumi = int(line.split("*")[3].strip())
    run_lumi.append("{} {}".format(run, lumi))
run_lumi = list(set(run_lumi))
run_lumi.sort()
sumlumi = 0
for item in run_lumi:
    print item, lumimap[item]
    sumlumi += lumimap[item]
print sumlumi

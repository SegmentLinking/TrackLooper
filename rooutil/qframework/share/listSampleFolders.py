#!/usr/bin/env python
import os
if 'xterm' in os.environ['TERM']:
    os.environ['TERM'] = 'vt100'
import sys
    
from ROOT import TString,TFile,TKey
from sys import argv

for idx in range(1,len(argv)):
    fname = argv[idx]
    f = TFile.Open(fname,"READ");
    if not f or not f.IsOpen():
        quit

    for k in f.GetListOfKeys():
        if not k.GetClassName() == "TQSampleFolder":
            continue
        name = k.GetName();
        if not "-" in name:
            print(name)

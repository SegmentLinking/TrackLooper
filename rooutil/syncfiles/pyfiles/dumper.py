#!/usr/bin/env python

import argparse
import warnings

"""
can do
python dumper.py ntuple.root -n 5 -c "evt_event>=1090375608" | grep hcalnoise_eventTrackEnergy
or
python dumper.py ntuple.root > dump.txt
"""

def ptetaphi(p4):
    return (p4.Pt(), p4.Eta(), p4.Phi())

def get_vals(bobj, class_name):
    is_wrapper = "edm::Wrapper<" in class_name
    is_vector = "vector<" in class_name
    is_lorentz = "Lorentz" in class_name
    is_int = "int" in class_name
    is_string = "TString" in class_name
    is_bool = "bool" in class_name

    if is_wrapper: 
        class_name = class_name[:-1].replace("edm::Wrapper<", "")
        bobj = bobj.product()

        if not is_vector and not is_lorentz: bobj = bobj[0]

    if is_vector: bobj = list(bobj)

    if is_string:
        if is_vector: bobj = map(str, bobj)
        else: bobj = str(bobj)

    if is_lorentz:
        if is_vector: bobj = [ptetaphi(obj) for obj in bobj]
        else: bobj = ptetaphi(bobj)

    if is_int:
        if is_vector: bobj = [int(obj) for obj in bobj]
        else: bobj = int(bobj)

    if is_bool:
        if is_vector: bobj = map(lambda x: int(bool(x)), bobj)

    # if class_name == "TBits":
    #     bobj = map(int, [bobj.TestBitNumber(i) for i in range(bobj.GetNbits())])

    return bobj


def dump(fname_in, treename="Events", max_nevents=1, cut=""):
    # max_nevents = 5
    # cut = "dRq1b<1.54495632648"

    f = TFile(fname_in)
    tree = f.Get(treename)
    try:
        aliases = tree.GetListOfAliases()
    except:
        aliases = None
    branches = tree.GetListOfBranches()

    d_bname_to_info = {}


    # cuts = ["pfcandsisGlobalMuon"] # FIXME
    # cuts = ["filtcscBeamHalo2015","evtevent","evtlumiBlock","evtbsp4","hltprescales","hltbits","hlttrigNames","musp4","evtpfmet","muschi2"] # FIXME
    for branch in branches:
        bname = branch.GetName()
        cname = branch.GetClassName()
        if "TBits" in cname: continue
        if "detectorStatus" in bname: continue
        if "pvAssociationQuality" in bname: continue
        if "fromPV" in bname: continue
        if "vector<vector<" in cname: continue # yeah, right. like I care about these

        if bname in ["EventSelections", "BranchListIndexes", "EventAuxiliary", "EventProductProvenance"]: continue
        # if not any([cut in bname for cut in cuts]): continue # FIXME
        d_bname_to_info[bname] = {
                "class": cname,
                "alias": bname,
                }

    if aliases:
        for ialias, alias in enumerate(aliases):
            aliasname = alias.GetName()
            branch = tree.GetBranch(tree.GetAlias(aliasname))
            branchname = branch.GetName().replace("obj","")
            if branchname not in d_bname_to_info: continue
            d_bname_to_info[branchname]["alias"] = aliasname

    ievents = range(max_nevents)

    if cut:
        tree.Draw(">>+elist",cut,"goff")
        elist = gDirectory.Get("elist")
        ievent_list = []
        for j in range(min(elist.GetN(), max_nevents)):
            ievent_list.append(elist.GetEntry(j))
        ievents = ievent_list

    for i in ievents:
        tree.GetEntry(i)
        print "=== Entry %i ===" % i
        for bname in d_bname_to_info:

            if bname in ["EventAuxiliary"]: continue

            alias = d_bname_to_info[bname]["alias"]
            class_name = d_bname_to_info[bname]["class"]
            bobj = tree.__getattr__(bname)


            try:
                vals = get_vals(bobj, class_name)
            except:
                continue

            print "--> %s:" % alias,

            if type(vals) == list:
                print ",".join(map(str,vals))
            else:
                print vals

            # if list of strings or list of tuples, then print each element on a new line
            # if type(vals) == list and len(vals) > 0 and type(vals[0]) in [str, tuple]:
            #     for val in vals:
            #         print "  ", val
            # elif type(vals) == list:
            #     print "  ", ",".join(map(str,vals))
            # else:
            #     print "   ", vals
            # print

        print

    f.Close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="root file")
    parser.add_argument("-t", "--treename", help="name of TTree")
    parser.add_argument("-n", "--nevents", help="max number of events to dump out")
    parser.add_argument("-c", "--cut", help="specify a cut. e.g.: evt_event==12345&&run=123")
    args = parser.parse_args()

    from ROOT import *

    nevents = 1
    cut = ""
    treename = "Events"
    if args.nevents:
        nevents = int(args.nevents)
    if args.cut:
        cut = str(args.cut)
    if args.treename:
        treename = args.treename


    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        dump(args.filename, treename=treename, max_nevents=nevents, cut=cut)

    # dump("data.root")
    # dump("ntuple.root")
    # dump("TTbar_madgraph_25.root", "t")

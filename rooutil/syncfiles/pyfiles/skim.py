#!/usr/bin/env python
import os
import argparse
import time


"""
This script makes skimmed root files by retaining only specified branches
and, optionally, can be paired with a TCut to further slim ntuples.
If no branches are provided, but a cut is provided, then all branches will be kept.

CLI usage can be found with `python skim.py -h`

Of course, a python API is provided:
>>> import skim
>>> help(skim.skim_tree)
>>> skim.skim_tree(<list_of_filename_patterns>, <list_of_branches_to_keep>, treename="t", fname_out="skim.root", cut_str="")

Also note that an easy way to get branches in a .so file is:
```
nm -gC ScanChain_C.so | grep ss:: | cut -d ':' -f3 | cut -d '(' -f1 | tr '\n' ','
```
You will need to change the grep string to work for your analysis.
"""

def readable_size(size):
    if size>1e9: return "%iGB" % (size//1e9)
    elif size>1e6: return "%iMB" % (size//1e6)
    elif size>1e3: return "%iKB" % (size//1e3)
    else: return "%iB" % size

def get_filesizes(fnames): return sum(map(os.path.getsize, fnames))

    
def skim_tree(fname_patts, branches_to_keep, treename="t", fname_out="skim.root", cut_str="", flip_branches=False):


    ch = TChain(treename)
    for patt in fname_patts:
        ch.Add(patt)
    nevents = ch.GetEntries()
    branches_to_keep = [b for b in branches_to_keep if b] # remove empty strings

    # https://root-forum.cern.ch/t/pyroot-crashes-when-in-arguments/25379/3 -- figured out solution to below lines
    # # cut_str = "Sum$(abs(genps_id_mother)==24 && genps_isLastCopy && (abs(genps_id)==11 || abs(genps_id)==13))==2" # 2 leps
    # # cut_str = "abs(Sum$(genps_id*(abs(genps_id_mother)==24 && genps_isLastCopy && (abs(genps_id)==11 || abs(genps_id)==13))))>20" # SS
    # # apparently when using a $ sign in the cut_str, root hijacks the arguments to the script and crashes. Sigh.
    # cut_str = cut_str.replace("SSS","$")

    if len(cut_str) > 0:
        print ">>> [!] You specified a cut string of: %s" % cut_str
        print ">>> [!] Make sure that you are opting to keep all branches used in that cut string."


    filenames = [f.GetTitle() for f in ch.GetListOfFiles()]


    f1 = TFile(filenames[0])
    tree = f1.Get(treename)
    tree.SetMakeClass(1);
    branches = [b.GetName() for b in tree.GetListOfBranches()]


    # see if the dummy user specified any branches to keep that aren't in the chain
    # and subtract them out to avoid segfaulttttt
    if not flip_branches:
        branches_not_in_chain = set(branches_to_keep)-set(branches)
        if len(branches_not_in_chain) > 0 and len(branches_to_keep) > 0:
            print ">>> [!] You dummy! I am going to neglect these branches which are not even in the TTree: %s" % ",".join(list(branches_not_in_chain))
        branches_to_keep = list(set(branches_to_keep)-branches_not_in_chain)

    if len(branches_to_keep) == 0 and not flip_branches:
        if len(cut_str) == 0:
            print ">>> [!] You dummy! You want me to skim 0 branches without any cut? That's pointless."
            return
        else:
            print ">>> [!] You specified 0 branches to keep, but you gave me a cut string, so keeping ALL branches."
            branches_to_keep = branches[:]
    else:

        # whitelist the ones to copy
        # or reverse if we have flip_branches
        if not flip_branches:
            ch.SetBranchStatus("*",0)
            for bname in branches_to_keep:
                ch.SetBranchStatus(bname,1)
        else:
            ch.SetBranchStatus("*",1)
            for bname in branches_to_keep:
                ch.SetBranchStatus(bname,0)

        # need this to actually copy over any 4vectors. WTF.
        # https://root.cern.ch/phpBB3/viewtopic.php?t=10725
        ch.SetBranchStatus("fCoordinates*",1);

    # actually do the skim and save the file
    t0 = time.time()
    new_file = TFile(fname_out,"RECREATE") 

    # copy over all the histograms too - note that this only takes the first file (TODO is to actually add multiples, but this is not a use case for me right now)
    for key in f1.GetListOfKeys():
        if key.ReadObj().InheritsFrom(TTree.Class()): continue
        name = key.GetName()
        f1.Get(name).Write()

    print ">>> Started skimming tree %s with %i events: %i --> %i branches" % (treename, nevents, len(branches), len(branches_to_keep))
    ch_new = ch.CopyTree(cut_str)
    print ">>> Finished skim in %.2f seconds" % (time.time()-t0)
    ch_new.GetCurrentFile().Write() 
    ch_new.GetCurrentFile().Close()

    # wow the user with incredible reduction stats
    size_before = get_filesizes(filenames)
    size_after = get_filesizes([fname_out])
    print ">>> Size reduction: %s --> %s (factor of %.1f)" % (readable_size(size_before),readable_size(size_after),1.0*size_before/size_after)
    print ">>> Your output file is %s." % fname_out

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("files", help="E.g., \"stuff/WJets*.root\". Use quotes to suppress wildcard expansion", nargs="*")
    parser.add_argument("-t", "--treename", help="Name of TTree", default="t")
    parser.add_argument("-o", "--output", help="Name of output skim file", default="skim.root")
    parser.add_argument("-b", "--branches", help="Comma-delimited list of branches to keep", default="")
    parser.add_argument("-c", "--cut", help="Specify a TCut. e.g.: evt_event==12345&&run=123", default="")
    parser.add_argument("-f", "--flipbranches", help="If branches to keep are specified, drop them instead", action="store_true")
    args = parser.parse_args()

    args.branches = map(lambda x: x.strip(), args.branches.split(","))

    print "-"*30
    print "Args:"
    print " "*4, "File patterns:", args.files
    print " "*4, "Treename:", args.treename
    print " "*4, "Output file:", args.output
    print " "*4, "Branches to keep:", args.branches
    print " "*4, "Cut string:", args.cut
    print "-"*30

    # This stuff is super necessary or else we all die
    from ROOT import TChain, TFile, gSystem, gROOT, TTree
    import ROOT as r
    r.PyConfig.IgnoreCommandLineOptions = True # https://root-forum.cern.ch/t/pyroot-crashes-when-in-arguments/25379/3
    r.v5.TFormula.SetMaxima(5000000)

    gSystem.Load("libFWCoreFWLite.so") 
    gSystem.Load("libDataFormatsFWLite.so");
    gROOT.ProcessLine("FWLiteEnabler::enable()")

    skim_tree(args.files, args.branches, args.treename, args.output, args.cut, args.flipbranches)

    # fname_patts = "/nfs-7/userdata/ss2015/ssBabies/v8.04_trigsafe_v4/WJets*.root"
    # treename = "t"
    # fname_out = "skim.root"
    # # cut_str = ""
    # cut_str = "lep1_p4.pt()>300"
    # branches_to_keep = ['is_real_data', 'scale1fb', 'lep1_id', 'lep1_p4', 'ht']
    # skim_tree(fname_patts, branches_to_keep, treename, fname_out, cut_str)


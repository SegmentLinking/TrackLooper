from ROOT import *

def get_chunks(v,n=50): return [ v[i:i+n] for i in range(0, len(v), n) ]

def merge_files(fnames_in, fname_out=None, treename="Events"):
    """
    # takes list of root files (or single string with wildcard). merges the entries in the given tree.

    basedir = "/hadoop/cms/store/user/namin/T2bW_575_2875_1_step3/crab_T2bW_575_2875_1_step3_apatters-T2bW_mSTOP575_mNLSP2875_mLSP1_seed42484-eb69b0448a13fda070ca35fd76ab/160327_061942/0000/"
    fnames = ["ntuple_1*.root", "ntuple_2*.root"]
    n_evts = merge_files(fnames)
    print "%i events merged" % n_evts
    """

    if not fname_out: fname_out = "merged.root"

    ch = TChain(treename)

    if type(fnames_in) in [list, tuple]:
        for fname in fnames_in:
            ch.Add(fname)
            print "Adding %s" % fname
    else:
        ch.Add(fnames_in)

    n_entries = ch.GetEntries()
    ch.Merge(fname_out)

    return n_entries

def copy_events(fname_in, runlumievts, fname_out=None, treename="Events"):
    # takes list of 3-tuples (run,lumi,evt), input root file (or wildcard), output name (if None, construct it)
    """
    rles = [(1,1,evt) for evt in [5,9,17,20,29,32,33,26,38,40,42]]
    n_evts = copy_events("/hadoop/cms/store/user/namin/T2bW_575_2875_1_step3/crab_T2bW_575_2875_1_step3_apatters-T2bW_mSTOP575_mNLSP2875_mLSP1_seed42484-eb69b0448a13fda070ca35fd76ab/160327_061942/0000/ntuple_1.root", rles)
    print "%i events selected" % n_evts
    """

    if not fname_out: 
        if "*" not in fname_in: fname_out = fname_in.split("/")[-1].replace(".root", "_skim.root")
        else: fname_out = "skim.root"

    d_rle = set()
    for rle in runlumievts:
        rle = map(int, list(rle)) # make sure all ints
        d_rle.add( tuple(rle) )

    ch = TChain(treename)
    ch.Add(fname_in)
    n_entries = ch.GetEntries()

    ch.SetBranchStatus("*",0)
    ch.SetBranchStatus("*_eventMaker_*",1)

    for chunk in get_chunks(runlumievts, n=50):
        cut_str = " || ".join(["(evt_run==%i && evt_lumiBlock==%i && evt_event==%i)" % tuple(rle) for rle in chunk])
        ch.Draw(">>+elist", cut_str, "goff")
    elist = gDirectory.Get("elist")

    ch.SetBranchStatus("*",1)

    new_file = TFile(fname_out,"RECREATE") 
    ch.SetEventList(elist)
    ch_new = ch.CopyTree("")
 
    ch_new.GetCurrentFile().Write() 
    ch_new.GetCurrentFile().Close()

    return ch_new.GetEntries()

def get_branch_info(tree):
    """
    Returns dictionary where keys are the branch names in the ttree and values are dictionaries
    with the type, class, and alias
    """


    aliases = tree.GetListOfAliases()
    branches = tree.GetListOfBranches()

    d_bname_to_info = {}

    # cuts = ["filtcscBeamHalo2015","evtevent","evtlumiBlock","evtbsp4","hltprescales","hltbits","hlttrigNames","musp4","evtpfmet","muschi2","ak8jets_pfcandIndicies","hlt_prescales"]
    # cuts = ["lep1_p4","lep2_p4"]
    isCMS3 = False
    have_aliases = False
    classname_to_type = lambda cname: "const " + cname.strip()
    for branch in branches:
        bname = branch.GetName()
        cname = branch.GetClassName()
        btitle = branch.GetTitle()

        if bname in ["EventSelections", "BranchListIndexes", "EventAuxiliary", "EventProductProvenance"]: continue
        # if not any([cut in bname for cut in cuts]): continue

        # sometimes root is stupid and gives no class name, so must infer it from btitle (stuff like "btag_up/F")
        if not cname:
            if btitle.endswith("/i"): cname = "unsigned int"
            elif btitle.endswith("/l"): cname = "unsigned long long"
            elif btitle.endswith("/F"): cname = "float"
            elif btitle.endswith("/I"): cname = "int"
            elif btitle.endswith("/O"): cname = "bool"
            elif btitle.endswith("/D"): cname = "double"

        typ = cname[:]

        if "edm::TriggerResults" in cname:
            continue

        if "edm::Wrapper" in cname:
            isCMS3 = True
            typ = cname.replace("edm::Wrapper<","")[:-1]
        typ = classname_to_type(typ)

        d_bname_to_info[bname] = {
                "class": cname,
                "alias": bname.replace(".",""),
                "type": typ,
                }

    if aliases:
        have_aliases = True
        for ialias, alias in enumerate(aliases):
            aliasname = alias.GetName()
            branch = tree.GetBranch(tree.GetAlias(aliasname))
            branchname = branch.GetName().replace("obj","")
            if branchname not in d_bname_to_info: continue
            d_bname_to_info[branchname]["alias"] = aliasname.replace(".","")

    return d_bname_to_info

def get_treename_from_file(tfile):
    """
    Given a tfile, this returns the name of the (only) ttree in the file, or
    "Events" in the case of MINIAOD
    """
    keys = tfile.GetListOfKeys()
    treenames = [key.GetName() for key in keys if key.ReadObj().InheritsFrom(TTree.Class())]
    if len(treenames) > 0 and "Events" in treenames: treename = "Events"
    else: treename = treenames[0]
    return treename




if __name__ == "__main__":

    # f = TFile("ntuple.root")
    # t = f.Get("Events")
    # print get_branch_info(t)
    # f = TFile("WJets.root")
    # t = f.Get("t")
    # print get_branch_info(t)
    f = TFile("ntuple.root")
    print get_treename_from_file(f)

#     basedir = "/hadoop/cms/store/user/namin/T2bW_575_2875_1_step3/crab_T2bW_575_2875_1_step3_apatters-T2bW_mSTOP575_mNLSP2875_mLSP1_seed42484-eb69b0448a13fda070ca35fd76ab/160327_061942/0000/"
#     fnames = [basedir+fname for fname in ["ntuple_11*.root", "ntuple_21*.root"]]
#     n_evts = merge_files(fnames)
#     print "%i events merged" % n_evts

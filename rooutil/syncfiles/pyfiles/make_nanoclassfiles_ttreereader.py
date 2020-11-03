#!/usr/bin/env python

import ROOT as r
import argparse
import sys
import os
import itertools


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="filename to make classfile on")
    parser.add_argument("-t", "--tree", help="treename (default: Events)", default="Events")
    parser.add_argument("-n", "--namespace", help="namespace (default: tas)", default="tas")
    parser.add_argument("-o", "--objectname", help="objectname (default: nt)", default="nt")
    parser.add_argument("-c", "--classname", help="classname (default: Nano)", default="Nano")
    parser.add_argument("-l", "--looper", help="make a looper as well", default=False, action="store_true")
    parser.add_argument("-b", "--branches", help="use only these specified (comma separated) branches", default="")
    args = parser.parse_args()

    fname = os.path.abspath(args.filename.strip())
    treename = args.tree
    classname = args.classname
    objectname = args.objectname
    namespace = args.namespace
    make_looper = args.looper
    filter_branches = args.branches.strip().split(",") if args.branches.strip() else []

    ginfo = {
            "treename": treename,
            "classname": classname,
            "namespace": namespace,
            "objectname": objectname,
            "filename": os.path.abspath(fname),
            "args": " ".join(sys.argv[1:]).replace(args.filename.strip(),fname),
            }

    f = r.TFile(fname)
    t = f.Get(treename)

    d_branch_info = {}

    branches = list(t.GetListOfBranches())
    for branch in branches:
        title = branch.GetTitle()
        name = branch.GetName()

        if filter_branches and name not in filter_branches: continue

        leaf = branch.GetLeaf(branch.GetName())
        leaf_title = leaf.GetTitle()
        ndata = leaf.GetNdata()
        typename = leaf.GetTypeName()
        tmap = {
                "Float_t": "float",
                "Int_t": "int",
                "Bool_t": "bool",
                }
        typename = tmap.get(typename,typename)
        typename_novec = typename[:]
        is_array = "[" in leaf_title
        if is_array:
            leaf_title = "{}[{}]".format(leaf_title.split("[")[0],ndata)
            typename = "vector<{}>".format(typename)

        d_branch_info[name] = {
                "desc": title,
                "name": name,
                "is_array": is_array,
                "typename": typename,
                "typename_novec": typename_novec,
                "ndata": ndata,
                "leaf_title": leaf_title,
                }

    names = d_branch_info.keys()
    for_p4s = []
    for prefix,sub in itertools.groupby(sorted(names), key=lambda x:x.split("_")[0]):
        if len(set([prefix+y for y in ["_pt", "_eta", "_phi", "_mass"]]) & set(sub)) == 4:
            bi = dict(d_branch_info[prefix+"_pt"])
            bi["typename"] = bi["typename"].replace("float","LorentzVector")
            bi["typename_novec"] = bi["typename_novec"].replace("float","LorentzVector")
            bi["name"] = bi["name"].replace("_pt","_p4")
            bi["leaf_title"] = bi["leaf_title"].replace("_pt","_p4")
            bi["desc"] = "from {}_pt,eta,phi,mass".format(prefix)
            for_p4s.append(bi)

    vals = d_branch_info.values() + for_p4s
    binfo = sorted(vals, key=lambda x: x["name"])

    def get_cc_top(ginfo,binfo):
        yield "#include \"{classname}.h\"".format(**ginfo)
        yield "{classname} {objectname};".format(**ginfo)

    def get_cc_init(ginfo,binfo):
        yield "void {classname}::Init(TTree *tree) {{".format(**ginfo)
        yield "    fReader.SetTree(tree);"
        # for bi in binfo:
        #     if "LorentzVector" in bi["typename"]: continue
        #     yield """    {name}_ = {{fReader, "{name}"}};""".format(**bi)
        yield "}"

    def get_cc_getentry(ginfo,binfo):
        yield "void {classname}::GetEntry(unsigned int idx) {{".format(**ginfo)
        yield "    index = idx;"
        yield "    fReader.SetEntry(idx);"
        for bi in binfo:
            yield """    loaded_{name}_ = false;""".format(**bi)
        yield "}"

    def get_cc_getfunctions(ginfo,binfo):
        for bi in binfo:
            p = { "name": bi["name"], "type": bi["typename"], "classname": ginfo["classname"] }

            yield "const {type} &{classname}::{name}() {{".format(**p)
            yield "    if (!loaded_{name}_) {{".format(**p)
            if "LorentzVector" in bi["typename"]:
                if bi["is_array"]:
                    yield "        value_{name}_.clear();".format(**p)
                    yield "        vector<float> pts = {classname}::{name}();".format(classname=p["classname"],name=p["name"].replace("_p4","_pt"))
                    yield "        vector<float> etas = {classname}::{name}();".format(classname=p["classname"],name=p["name"].replace("_p4","_eta"))
                    yield "        vector<float> phis = {classname}::{name}();".format(classname=p["classname"],name=p["name"].replace("_p4","_phi"))
                    yield "        vector<float> masses = {classname}::{name}();".format(classname=p["classname"],name=p["name"].replace("_p4","_mass"))
                    yield "        for (unsigned int i=0; i < pts.size(); i++) {"
                    yield "            value_{name}_.push_back(LorentzVector(pts[i],etas[i],phis[i],masses[i]));".format(**p)
                    yield "        }"
                else:
                    yield "        value_{name}_.clear();".format(**p)
                    yield "        value_{name}_ = LorentzVector(*{name_pt}_,*{name_eta}_,*{name_phi}_,*{name_mass}_);".format(
                            name = p["name"],
                            name_pt = p["name"].replace("_p4","_pt"),
                            name_eta = p["name"].replace("_p4","_eta"),
                            name_phi = p["name"].replace("_p4","_phi"),
                            name_mass = p["name"].replace("_p4","_mass"),
                            )
            else:
                if bi["is_array"]:
                    yield "        value_{name}_ = {type}({name}_.begin(),{name}_.end());".format(**p)
                else:
                    yield "        value_{name}_ = *{name}_;".format(**p)
            yield "        loaded_{name}_ = true;".format(**p)
            yield "    }"
            yield "    return value_{name}_;".format(**p)
            yield "}"

    def get_cc_tas(ginfo,binfo,extra=None):
        yield "namespace {namespace} {{".format(**ginfo)
        for bi in binfo:
            p = { "name": bi["name"], "type": bi["typename"], "objectname": ginfo["objectname"] }
            yield "    const {type} &{name}() {{ return {objectname}.{name}(); }}".format(**p)
        if extra:
            for x in extra(): yield x
        yield "}"

    def get_h_top(ginfo,binfo):
        yield "// -*- C++ -*-"
        yield "#ifndef {classname}_H".format(**ginfo)
        yield "#define {classname}_H".format(**ginfo)
        for include in [
                "Math/LorentzVector.h",
                "Math/GenVector/PtEtaPhiM4D.h",
                "Math/Point3D.h",
                "TMath.h",
                "TBranch.h",
                "TTree.h",
                "TH1F.h",
                "TFile.h",
                "TBits.h",
                "vector",
                "unistd.h",
                "chrono",
                "ctime",
                "numeric",
                "TTreeReader.h",
                "TTreeReaderValue.h",
                "TTreeReaderArray.h",
                ]:
            yield "#include \"{}\"".format(include)
        yield ""
        yield "typedef ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<float> > LorentzVector;"
        yield "#define makeP4(Collection, idx) LorentzVector(Collection##_pt[idx],Collection##_eta[idx],Collection##_phi[idx],Collection##_mass[idx]);"
        yield "#define RANGE(a, b) unsigned a=0; a<b; a++"
        yield ""
        yield "// Generated with args: {args}".format(**ginfo)
        yield ""
        yield "using namespace std;"

    def get_h_class(ginfo,binfo):
        yield "class {classname} {{".format(**ginfo)
        yield "private:"
        yield "protected:"
        yield "    unsigned int index;"
        yield "    TTreeReader fReader;"
        for bi in binfo:
            if "LorentzVector" not in bi["typename"]:
                if bi["is_array"]:
                    yield """    TTreeReaderArray<{typename_novec}> {name}_ = {{fReader, "{name}"}};""".format(**bi)
                else:
                    yield """    TTreeReaderValue<{typename_novec}> {name}_ = {{fReader, "{name}"}};""".format(**bi)
            yield """    {typename} value_{name}_; // {leaf_title} - {desc} """.format(**bi)
            yield """    bool loaded_{name}_;""".format(**bi)
        yield "public:"
        yield "    void Init(TTree *tree);"
        yield "    void GetEntry(unsigned int idx);"
        for bi in binfo:
            yield "    const {typename} &{name}();".format(**bi)
        yield "};"
        yield ""
        yield "#ifndef __CINT__"
        yield "extern {classname} {objectname};".format(**ginfo)
        yield "#endif"

    def get_h_tas(ginfo,binfo,extra=None):
        yield "namespace {namespace} {{".format(**ginfo)
        for bi in binfo:
            yield "    const {typename} &{name}();".format(**bi)
        if extra:
            for x in extra(): yield x
        yield "}"
        yield "#endif"

    def get_looper_ScanChain(ginfo):
        yield """#pragma GCC diagnostic ignored "-Wsign-compare"
#include "TFile.h"
#include "TTree.h"
#include "TChain.h"

#include "{classname}.h"
#include "tqdm.h"

using namespace std;

int ScanChain(TChain *ch) {{

    TH1F * h1 = new TH1F("met", "met", 50, 0, 300);

    int nEventsTotal = 0;
    int nEventsChain = ch->GetEntries();
    TFile *currentFile = 0;
    TObjArray *listOfFiles = ch->GetListOfFiles();
    TIter fileIter(listOfFiles);
    tqdm bar;

    while ( (currentFile = (TFile*)fileIter.Next()) ) {{
        TFile *file = new TFile( currentFile->GetTitle() );
        TTree *tree = (TTree*)file->Get("{treename}");
        TString filename(currentFile->GetTitle());
        {objectname}.Init(tree);

        for( unsigned int event = 0; event < tree->GetEntriesFast(); ++event) {{
            {objectname}.GetEntry(event);
            nEventsTotal++;
            bar.progress(nEventsTotal, nEventsChain);

            // Analysis code

            // auto jets = {namespace}::GetVLV("Jet_p4"); if (jets.size() > 0) h1->Fill(jets[0].pt());
            // auto jets = {namespace}::Jet_p4(); if (jets.size() > 0) h1->Fill(jets[0].pt());
            // auto jetpts = {namespace}::Jet_pt(); if (jetpts.size() > 0) h1->Fill(jetpts[0]);
            // for (auto& s : {{"MET_pt", "CaloMET_pt", "GenMET_pt", "RawMET_pt", "TkMET_pt"}}) h1->Fill({namespace}::GetF(s));
            h1->Fill({namespace}::MET_pt());

        }} // Event loop
        delete file;
    }} // File loop
    bar.finish();

    TFile* f1 = new TFile("output.root", "RECREATE");
    std::cout <<  "Mean of h1: " << h1->GetMean() <<  std::endl;
    h1->Write();
    f1->Write();
    f1->Close();
    return 0;
}}
""".format(**ginfo)


    def get_looper_doAll(ginfo):
        yield "{"
        yield "    gROOT->ProcessLine(\".L {classname}.cc+\");".format(**ginfo)
        yield "    gROOT->ProcessLine(\".L ScanChain.C+\");".format(**ginfo)
        yield "    TChain *ch = new TChain(\"{treename}\");".format(**ginfo)
        yield "    ch->Add(\"{filename}\");".format(**ginfo)
        yield "    ScanChain(ch);"
        yield "}"

    types = set(b["typename"] for b in binfo if "U" not in b["typename"])  # drop things like Uints
    short_map = {
            "vector<float>": "VF",
            "vector<int>": "VI",
            "vector<LorentzVector>": "VLV",
            "vector<bool>": "VB",
            "float": "F",
            "int": "I",
            "LorentzVector": "LV",
            "bool": "B",
            }

    def extra_h_tas():
        for t in types:
            yield "    {typename} Get{shortname}(const string &name);".format(typename=t, shortname=short_map[t])

    def extra_cc_tas():
        for t in types:
            yield """    {t} Get{short}(const string &name) {{""".format(t=t,short=short_map[t])
            bsame = [x for x in binfo if x["typename"] == t]
            for ibi,bi in enumerate(bsame):
                p = { "name": bi["name"], "type": bi["typename"], "objectname": ginfo["objectname"] }
                if ibi == 0:
                    yield "        if (name == \"{name}\") return {objectname}.{name}();".format(**p)
                elif ibi == len(bsame)-1:
                    yield "        return {t}();".format(t=t)
                else:
                    yield "        else if (name == \"{name}\") return {objectname}.{name}();".format(**p)
            if len(bsame) < 2:
                yield "        return {t}();".format(t=t)
            yield "    }"

    with open("{}.cc".format(ginfo["classname"]), "w") as fhout:
        fhout.write("\n".join(get_cc_top(ginfo,binfo)))
        fhout.write("\n\n")
        fhout.write("\n".join(get_cc_init(ginfo,binfo)))
        fhout.write("\n\n")
        fhout.write("\n".join(get_cc_getentry(ginfo,binfo)))
        fhout.write("\n\n")
        fhout.write("\n".join(get_cc_getfunctions(ginfo,binfo)))
        fhout.write("\n\n")
        fhout.write("\n".join(get_cc_tas(ginfo,binfo,extra=extra_cc_tas)))

    with open("{}.h".format(ginfo["classname"]), "w") as fhout:
        fhout.write("\n".join(get_h_top(ginfo,binfo)))
        fhout.write("\n\n")
        fhout.write("\n".join(get_h_class(ginfo,binfo)))
        fhout.write("\n\n")
        fhout.write("\n".join(get_h_tas(ginfo,binfo,extra=extra_h_tas)))

    if os.path.isfile("ScanChain.C") or os.path.isfile("doAll.C"):
        print(">>> Hey, you already have a looper here! Not overwriting the looper. Delete to regenerate.")
        make_looper = False

    if make_looper:

        with open("ScanChain.C", "w") as fhout:
            fhout.write("\n".join(get_looper_ScanChain(ginfo)))

        with open("doAll.C", "w") as fhout:
            fhout.write("\n".join(get_looper_doAll(ginfo)))


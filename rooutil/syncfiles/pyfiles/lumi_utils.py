#!/usr/bin/env python

# http://stackoverflow.com/questions/15760712/python-readline-module-prints-escape-character-during-import
import os, sys
if 'xterm' in os.environ.get('TERM',""): os.environ['TERM'] = 'vt100'
redirect = not sys.stdout.isatty()

import commands
from itertools import groupby
import glob
import json
# from multiprocessing.dummy import Pool as ThreadPool 
from multiprocessing import Pool as ThreadPool 
import glob
try:
    from tqdm import tqdm
except:
    redirect = True
import argparse

class RunLumis():
    def __init__(self, rls={}):
        if type(rls) == type(self): # constructor should be idempotent
            self.rls = rls.rls.copy()
        elif type(rls) == file: # handle snt input (as file handle)
            lines = fhin.readlines()
            d_runlumis = {}
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 3:
                    run, lumiLower, lumiUpper = map(int, parts)
                    if run not in d_runlumis: d_runlumis[run] = []
                    d_runlumis[run].append([lumiLower, lumiUpper])
            for run in d_runlumis: d_runlumis[run] = self.rangesToSet(d_runlumis[run])
            self.rls = d_runlumis
        elif type(rls) == dict:
            try: isJson = type(rls[rls.keys()[0]][0]) == list # handle json object
            except: isJson = False

            clean_rls = {}
            if not isJson:
                for run in rls: clean_rls[run] = set(rls[run]) # handle regular dict
            else:
                for run in rls: clean_rls[int(run)] = self.rangesToSet(rls[run])

            self.rls = clean_rls

    def keys(self): return self.rls.keys()

    def __getitem__(self, run): return self.rls[run]

    def __setitem__(self, run, value): self.rls[run] = value

    def __add__(self, other): return self.pairsToRunLumis(self.getRunLumiPairs() | other.getRunLumiPairs())

    def __sub__(self, other): return self.pairsToRunLumis(self.getRunLumiPairs() - other.getRunLumiPairs())

    def __eq__(self, other): return (self in other) and (other in self)

    def __ne__(self, other): return not(self.__eq__(other))

    def __str__(self): 
        # convert keys to strings
        js = self.getJson()
        return json.dumps(js)
        # newjs = {}
        # for key in js:
        #     newjs[str(key)] = js[key]
        # return str(newjs)

    def __contains__(self, obj):
        if type(obj) == int: return obj in self.rls
        elif type(obj) == type(self):
            objPairs = obj.getRunLumiPairs()
            selfPairs = self.getRunLumiPairs()
            return (objPairs < selfPairs) or (objPairs == selfPairs)
        return False

    def pairsToRunLumis(self, pairs):
        """
        takes set( (275103,1), (275103,2), (275999, 1) ) and makes RunLumis object
        """
        d_runlumis = {}
        for run, lumi in pairs:
            if run not in d_runlumis: d_runlumis[run] = []
            d_runlumis[run].append(lumi)
        return RunLumis(d_runlumis)

    def getRunLumiPairs(self):
        # returns stuff like: set( (275103,1), (275103,2), (275999, 1) )
        pairs = set([])
        for run,lumis in self.rls.items():
            for lumi in lumis:
                pairs.add((run,lumi))
        return pairs

    def listToRanges(self, a):
        # turns [1,2,4,5,9] into [[1,2],[4,5],[9]]
        ranges = []
        for k, iterable in groupby(enumerate(sorted(a)), lambda x: x[1]-x[0]):
             rng = list(iterable)
             if len(rng) == 1: first, second = rng[0][1], rng[0][1]
             else: first, second = rng[0][1], rng[-1][1]
             ranges.append([first,second])
        return ranges

    def rangesToSet(self, a):
        # turns [[1,2],[4,5],[9]] into set([1,2,4,5,9])
        s = set([])
        for rng in a:
            first,second = rng
            s.update(range(first,second+1))
        return s

    def getJson(self):
        # returns json format for dict of runs with values being sets of lumis
        new_dict = {}
        for run in self.rls:
            new_dict[run] = self.listToRanges(self.rls[run])
        return new_dict

    def getIntLumi(self, typ="recorded", first_run=None, last_run=None):
        intlumi = 0.0
        if typ == "recorded":
            if not d_brilcalc: makeBrilcalcMap(delivered=False)
        else:
            if not d_brilcalc_delivered: makeBrilcalcMap(delivered=True)
        # print d_brilcalc
        for pair in self.getRunLumiPairs():
            if first_run and pair[0] < first_run: continue
            if last_run and pair[0] > last_run: continue
            if typ == "recorded":
                intlumi += d_brilcalc.get(pair, 0.0)
            else:
                intlumi += d_brilcalc_delivered.get(pair, 0.0)
        return intlumi

    def getBrilcalcMap(self, delivered=False):
        if delivered:
            if not d_brilcalc_delivered: makeBrilcalcMap(delivered)
            return d_brilcalc_delivered
        else:
            if not d_brilcalc: makeBrilcalcMap()
            return d_brilcalc

    def getSNT(self):
        buff = ""
        js = self.getJson()
        for run in js:
            for lumiFirst, lumiLast in js[run]:
                buff += "%i %i %i\n" % (run, lumiFirst, lumiLast)
        return buff

    def writeToFile(self, fname):
        with open(fname,"w") as fhout:
            js = self.getJson()
            json.dump(js, fhout)
            print "Wrote JSON to file %s" % fname

    def writeToFileSNT(self, fname):
        with open(fname,"w") as fhout:
            fhout.write(self.getSNT())
            print "Wrote snt format to file %s" % fname

# BRILCALC_FILE = "/home/users/namin/dataTuple/2016D/NtupleTools/dataTuple/lumis/lumis_skim.csv"
# BRILCALC_FILES = ["/home/users/namin/luminosity/fetcher/lumis_skim.csv", "/home/users/namin/luminosity/fetcher/lumis_skim_2016.csv"]
BRILCALC_FILES = ["/home/users/namin/luminosity/fetcher/lumis_skim.csv"]
d_brilcalc = {}
d_brilcalc_delivered = {}
def makeBrilcalcMap(delivered=False):
    dLumiMap =  {}
    for bfn in BRILCALC_FILES:
        with open(bfn, "r") as fhin:
            for line in fhin:
                line = line.strip()
                try:
                    run,ls,ts,deliv,recorded = line.split(",")
                    run = int(run)
                    ls = int(ls)
                    if delivered:
                        deliveredPB = float(deliv)
                        d_brilcalc_delivered[(run,ls)] = deliveredPB
                    else:
                        recordedPB = float(recorded)
                        d_brilcalc[(run,ls)] = recordedPB
                except: pass

def getChunks(v,n=3): return [ v[i:i+n] for i in range(0, len(v), n) ]

def getRunLumis(fnames, treename="Events"):
    # print "fnames", fnames
    # returns dict where keys are runs and values are sets of lumi sections
    if type(fnames) == list:
        fname = fnames[0]
    else:
        fname = fnames

    import ROOT as r
    f1 = r.TFile(fname)
    treenames = [obj.GetName() for obj in f1.GetListOfKeys()]
    treename = treenames[0]
    if len(treenames) > 1 and "Events" in treenames:
        treename = "Events"

    if type(fnames) == list:
        tree = r.TChain(treename)
        for fname in fnames:
            tree.Add(fname)
    else:
        tree = f1.Get(treename)

    isLeptonTree = ("Lepton" in tree.GetTitle()) or ("Lepton" in f1.GetListOfKeys()[0].GetTitle())
    isStopTree = ("Stop" in f1.GetListOfKeys()[0].GetTitle())
    isCMS3style = (treename == "Events")  or isLeptonTree
    N = tree.GetEntries()

    # print "running on %i entries for %s" % (N, fname)
    if isCMS3style:
        tree.SetBranchStatus("*",0)
        if isLeptonTree:
            tree.SetBranchStatus("*evt_run*",1)
            tree.SetBranchStatus("*evt_lumiBlock*",1)
        else:
            tree.SetBranchStatus("*evtrun*",1)
            tree.SetBranchStatus("*evtlumiBlock*",1)
        tree.SetEstimate(N);
        tree.Draw("evt_run:evt_lumiBlock","","goff")
    else:

        if isStopTree:
            tree.SetBranchStatus("*",0)
            tree.SetBranchStatus("run",1)
            tree.SetBranchStatus("ls",1)
            tree.SetEstimate(N);
            tree.Draw("run:ls","","goff")
        else:
            # SS
            tree.SetBranchStatus("*",0)
            tree.SetBranchStatus("*run*",1)
            tree.SetBranchStatus("*lumi*",1)
            tree.SetBranchStatus("*fired_trigger*",1)
            tree.SetEstimate(N);
            # tree.Draw("run:lumi","","goff")
            tree.Draw("run:lumi","fired_trigger","goff")


    runs = tree.GetV1()
    lumis = tree.GetV2()

    d_rl = { }
    for i in range(N):
        run, ls = int(runs[i]), int(lumis[i])
        if run not in d_rl: d_rl[run] = set([])
        d_rl[run].add(ls)

    if type(fnames) != list: f1.Close()
    return RunLumis(d_rl)

def test():
    j0 = RunLumis({})
    j1 = RunLumis({273290: [4, 5, 6, 7, 8, 9, 10], 275603: [1, 2, 3, 4, 5, 8]})
    j2 = RunLumis({273290: [4, 5, 6, 7, 11, 12, 13]})
    j3 = RunLumis({273290: [4, 5, 6], 275603: [1,5]}) # subset of j1

    assert(273290 in j2)
    assert((273291 in j2) == False)
    assert(j3 in j1)
    assert(j2+j1 == j1+j2)
    assert((j2 in j1+j3) == False)
    assert(j1-j2 in j1)
    assert(j0+j3 == j3)
    assert((j0+j3 != j3) == False)
    assert(j1 == RunLumis(j1.getJson()))
    assert(j1 == RunLumis(j1))

    print "passed all tests!"

if __name__ == '__main__':

    # test()

    parser = argparse.ArgumentParser()

    parser.add_argument("files", help="input file(s), quoted if there are wildcards")
    parser.add_argument("-j", "--json", help="show total json", action="store_true")
    parser.add_argument("-s", "--snt", help="show snt format json", action="store_true")
    parser.add_argument("-l", "--lumi", help="show int lumi", action="store_true")
    args = parser.parse_args()

    fname_patt = args.files
    doJson = args.json
    doLumi = args.lumi
    doSNT = args.snt

    if not doJson and not doLumi and not doSNT:
        doJson, doLumi = True, True

    fnames = glob.glob(fname_patt)
    isRootFile = ".root" in fnames[0]
    isJsonTextFile = ".json" in fnames[0] or ".txt" in fnames[0]

    # fname_patt = "/hadoop/cms/store/group/snt/run2_data/Run2016C_MET_MINIAOD_PromptReco-v2/merged/V08-00-07/merged_ntuple_10.root"
    # fname_patt = "/nfs-7/userdata/leptonTree/v1.09FR_80X/2p6ifb/2016DoubleMuon.root"
    # fname_patt = "/nfs-7/userdata/ss2015/ssBabies/v8.02/Data*.root"
    # fname_patt = "/nfs-7/userdata/dataTuple/nick/json_lists/Run2016B_MET_MINIAOD_PromptReco-v2/*.txt"
    # fname_patt = "/home/users/namin/2016/ss/master/SSAnalysis/goodRunList/*.txt"
    if isRootFile:
        pool = ThreadPool(10)
        vals = []
        chunks = getChunks(fnames, 6)
        if redirect:
            for result in pool.imap_unordered(getRunLumis, chunks):
                vals.append(result)
        else:
            for result in tqdm(pool.imap_unordered(getRunLumis, chunks),total=len(chunks)):
                vals.append(result)
        pool.close()
        pool.join()
        allRunLumis = sum(vals, RunLumis({}))

    elif isJsonTextFile:
        isJson = False
        allRunLumis = RunLumis()
        if len(fnames) > 10 and not redirect: fnames = tqdm(fnames)
        for fname in fnames:
            with open(fname, "r") as fhin:
                try:
                    js = json.load(fhin)
                    isJson = True
                except: pass

                if isJson: allRunLumis += RunLumis(js)
                else: 
                    fhin.seek(0,0)
                    allRunLumis += RunLumis(fhin)

    if doJson:
        print allRunLumis
    if doLumi:
        print "total integrated luminosity (/pb): %.2f" % (allRunLumis.getIntLumi())
    if doSNT:
        print allRunLumis.getSNT()


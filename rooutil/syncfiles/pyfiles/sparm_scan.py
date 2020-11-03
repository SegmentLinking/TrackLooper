from ROOT import *
import os
import time
import glob
from tqdm import tqdm
from multiprocessing.dummy import Pool as ThreadPool 

if __name__ == "__main__":

    name = "T1tttt"
    basename = "points_%s" % name

    # fpatt = "/hadoop/cms/store/group/snt/run2_25ns_80MiniAODv2_fastsim/SMS-T1tttt_TuneCUETP8M1_13TeV-madgraphMLM-pythia8_RunIISpring16MiniAODv2-PUSpring16Fast_80X_mcRun2_asymptotic_2016_miniAODv2_v0-v1/V08-00-05/merged_ntuple_[0-2][0-9].root"
    fpatt = "/hadoop/cms/store/group/snt/run2_25ns_80MiniAODv2_fastsim/SMS-T1tttt_TuneCUETP8M1_13TeV-madgraphMLM-pythia8_RunIISpring16MiniAODv2-PUSpring16Fast_80X_mcRun2_asymptotic_2016_miniAODv2_v0-v1/V08-00-05/merged_ntuple_*.root"
    fnames = glob.glob(fpatt)
    print len(fnames)


    tfout = TFile("%s.root" % basename, "RECREATE")
    
    # book histos
    hists = []
    bins = "100,0,2500,100,0,2500"
    for ifname, fname in enumerate(fnames):
        hists.append( TH2F("h"+str(ifname),"h"+str(ifname),100,0,2500,100,0,2500) )

    def fill_histo(ifname, fname):
        f1 = TFile.Open(fname)
        histname = "h"+str(ifname)
        tree = f1.Get("Events")
        tree.Draw("sparm_values[0]:sparm_values[1]>>%s(%s)" % (histname, bins))
        hists[ifname].Add( gDirectory.Get(histname) )
        f1.Close()

    t0 = time.time()
    for ifname, fname in tqdm(list(enumerate(fnames))):
        fill_histo(ifname, fname)
    t1 = time.time()
    print "elapsed time: %.3f" % (t1-t0)

    # get first histogram and add the rest to it
    hall = hists[0]
    for hist in hists[1:]:
        hall.Add(hist)
    tfout.Write()

    cnt = 0
    with open("%s.txt" % basename, "w") as fhout:
        for ix in range(hall.GetNbinsX()):
            for iy in range(hall.GetNbinsY()):
                n = int(hall.GetBinContent(ix,iy))
                if n == 0: continue
                x,y = int(hall.GetXaxis().GetBinLowEdge(ix)), int(hall.GetYaxis().GetBinLowEdge(iy))
                if x == 0: x = 1 # LSP is mass 1 in pythia :(
                if y == 0: y = 1 # LSP is mass 1 in pythia :(
                # print y,x,n
                fhout.write("    if (mSp == %i && mSp2 == %i) return %i;\n" % (y,x,n))

    gStyle.SetOptStat(0)
    c1 = TCanvas()
    hall.Draw("colz")
    hall.SetTitle("%s [%i events]" % (name, hall.GetEntries()))
    # c1.SetLogz(1)
    hall.GetXaxis().SetTitle("mSp2")
    hall.GetYaxis().SetTitle("mSp1")

    c1.SaveAs("%s.pdf" % basename)
    os.system("web %s.pdf" % basename)

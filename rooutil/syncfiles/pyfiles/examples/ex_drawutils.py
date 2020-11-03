import ROOT as r
import time

"""
0. Get a ROOT environment (CMSSW 8+ works for me)
1. python ex_drawutils.py
2. ???
3. Profit

Note that the Entries$ variable is available within the 
draw statement *ONLY AFTER IT HAS BEEN COMPUTED*. I.e.,
that means the ch.GetEntries() preceding it is not just
for show!

is_duplicate(run,event) returns 1 if the pair has been
seen before, otherwise 0.

progress(Entry$,Entries$) returns 1, but shows a nice
progress bar with a processing rate!
"""

r.gROOT.ProcessLine(".L ../../miscfiles/draw_utils.cc")

ch = r.TChain("t")
ch.Add("/nfs-7/userdata/namin/tupler_babies/merged/FT/v0.10_data/output/*Hv3.root")

t0 = time.time()
print ch.GetEntries()
ch.Draw("event:run:lumi:progress(Entry$,Entries$)")
print "time taken reading branches: {:.1f}".format(time.time()-t0)

t0 = time.time()
print ch.Draw("is_duplicate(run,event)", "progress(Entry$,Entries$)", "goff")
print "time taken calculating duplicates: {:.1f}".format(time.time()-t0)



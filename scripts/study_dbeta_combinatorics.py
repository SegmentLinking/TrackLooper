#!/bin/env python

import ROOT as r

r.gROOT.SetBatch(True)

f = open("dbeta_data.txt")

h_pdgids_multiplicity_passed = r.TH1F("h_pdgids_multiplicity_passed", "", 4, 0, 4)
h_trkidx_multiplicity_passed = r.TH1F("h_trkidx_multiplicity_passed", "", 4, 0, 4)

h_pdgids_multiplicity_failed = r.TH1F("h_pdgids_multiplicity_failed", "", 4, 0, 4)
h_trkidx_multiplicity_failed = r.TH1F("h_trkidx_multiplicity_failed", "", 4, 0, 4)

lines = f.readlines()

for index, line in enumerate(lines):
    ls = line.split()

    passed = True if ls[1] == "passed" else False
    pdgids = [ls[3], ls[5], ls[7], ls[9], ls[11], ls[13], ls[15], ls[17]]
    trkidx = [ls[19], ls[21], ls[23], ls[25], ls[27], ls[29], ls[31], ls[33]]

    pdgids = [ abs(int(x)) for x in pdgids ]
    trkidx = [ abs(int(x)) for x in trkidx ]

    pdgids_multiplicity = len(list(set(pdgids)))
    trkidx_multiplicity = len(list(set(trkidx)))


    if passed:
        h_pdgids_multiplicity_passed.Fill(pdgids_multiplicity)
        h_trkidx_multiplicity_passed.Fill(trkidx_multiplicity)
    else:
        h_pdgids_multiplicity_failed.Fill(pdgids_multiplicity)
        h_trkidx_multiplicity_failed.Fill(trkidx_multiplicity)


c1 = r.TCanvas()
h_pdgids_multiplicity_passed.Draw("histtext")
c1.SaveAs("plots_combinatorics/h_pdgids_multiplicity_passed.pdf")
c1.SaveAs("plots_combinatorics/h_pdgids_multiplicity_passed.png")
h_trkidx_multiplicity_passed.Draw("histtext")
c1.SaveAs("plots_combinatorics/h_trkidx_multiplicity_passed.pdf")
c1.SaveAs("plots_combinatorics/h_trkidx_multiplicity_passed.png")
h_pdgids_multiplicity_failed.Draw("histtext")
c1.SaveAs("plots_combinatorics/h_pdgids_multiplicity_failed.pdf")
c1.SaveAs("plots_combinatorics/h_pdgids_multiplicity_failed.png")
h_trkidx_multiplicity_failed.Draw("histtext")
c1.SaveAs("plots_combinatorics/h_trkidx_multiplicity_failed.pdf")
c1.SaveAs("plots_combinatorics/h_trkidx_multiplicity_failed.png")

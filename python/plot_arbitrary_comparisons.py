import ROOT as r
from array import array
import os
import sys
from math import sqrt

r.gROOT.SetBatch(True)

# Get the files to be compared

eff_files_1 = sys.argv[1]
eff_files_2 = sys.argv[2]
histName_1 = sys.argv[3]
histName_2 = sys.argv[4]
gitHash_1 = sys.argv[5]
gitHash_2 = sys.argv[6]
legendName_1 = sys.argv[7]
legendName_2 = sys.argv[8]
title = sys.argv[9]
output_name = sys.argv[10]

# assert that the bin sizes are same!

f1 = r.TFile(eff_files_1)
f2 = r.TFile(eff_files_2)

keys1 = [x.GetName() for x in f1.GetListOfKeys() if "Root__" not in x.GetName()]
keys2 = [x.GetName() for x in f2.GetListOfKeys() if "Root__" not in x.GetName()]

assert(histName_1 in keys1)
assert(histName_2 in keys2)

ms = [24, 25, 26, 27, 28, 30, 32, 42, 46, 40]
cs = [2, 3, 4, 6, 7, 8, 9, 30, 46, 38, 40]

eff = f1.Get(histName_1)
eff2 = f2.Get(histName_2)

c1 = r.TCanvas()
c1.SetBottomMargin(0.15)
c1.SetLeftMargin(0.15)
c1.SetTopMargin(0.15)
c1.SetRightMargin(0.15)
if "_pt" in output_name:
    c1.SetLogx()
eff.Draw("epa")
eff.SetMarkerStyle(19)
eff.SetMarkerSize(1.2)
eff.SetLineWidth(2)
if "phi" in output_name:
    title = "#phi"
elif "_dz" in output_name:
    title = "z [cm]"
elif "_dxy" in output_name:
    title = "d0 [cm]"
elif "_pt" in output_name:
    title = "p_{T} [GeV]"
else:
    title = "#eta"
eff.GetXaxis().SetTitle(title)
eff.GetYaxis().SetTitle("Efficiency")
eff.GetXaxis().SetTitleSize(0.05)
eff.GetYaxis().SetTitleSize(0.05)
eff.GetXaxis().SetLabelSize(0.05)
eff.GetYaxis().SetLabelSize(0.05)
yaxis_max = 0

# leg1 = r.TLegend(0.63, 0.67, 0.93, 0.87)
leg1 = r.TLegend(0.63, 0.18, 0.93, 0.38)

for i in range(0, eff.GetN()):
    if yaxis_max < eff.GetY()[i]:
        yaxis_max = eff.GetY()[i]
# print yaxis_max
yaxis_min = 999
for i in range(0, eff.GetN()):
    if yaxis_min > eff.GetY()[i] and eff.GetY()[i] != 0:
        yaxis_min = eff.GetY()[i]
# print yaxis_min
if "ptzoom" in output_name:
    eff.GetYaxis().SetRangeUser(yaxis_max - 0.12, yaxis_max + 0.02)
elif "etazoom" in output_name:
    eff.GetYaxis().SetRangeUser(yaxis_max - 0.12, yaxis_max + 0.02)
elif "ptmaxzoom" in output_name:
    eff.GetYaxis().SetRangeUser(yaxis_max - 0.02, yaxis_max + 0.02)
elif "etamaxzoom" in output_name:
    eff.GetYaxis().SetRangeUser(yaxis_max - 0.02, yaxis_max + 0.02)
else:
    eff.GetYaxis().SetRangeUser(0, 1.02)

if "eta" in output_name:
    eff.GetXaxis().SetLimits(-2.5, 2.5)

eff.SetTitle(title)
if len(sys.argv) > 5:
    leg1.AddEntry(eff, legendName_1, "ep")
# Label
t = r.TLatex()
t.SetTextAlign(11)  # align bottom left corner of text
t.SetTextColor(r.kBlack)
t.SetTextSize(0.04)
x = r.gPad.GetX1() + r.gPad.GetLeftMargin()
y = r.gPad.GetY2() - r.gPad.GetTopMargin() + 0.01
sample_name_label = "  Version tag 1:" + gitHash_1 + "   Version tag 2: " + gitHash_2
t.DrawLatexNDC(x, y, "#scale[0.9]{#font[52]{%s}}" % sample_name_label)

eff2.SetMarkerStyle(ms[0])
eff2.SetMarkerSize(1.2)
eff2.SetLineWidth(1)
eff2.SetMarkerColor(cs[0])
eff2.SetLineColor(cs[0])
eff2.Draw("ep")
leg1.AddEntry(eff2, legendName_2, "ep")

leg1.Draw()
# Save
c1.SetGrid()
c1.SaveAs("{}".format(output_name))
c1.SaveAs("{}".format(output_name.replace(".pdf", ".png")))

#!/bin/env python

import os
import sys
import glob
import ROOT as r

# Get the git hash to compare
githash = sys.argv[1]
refgithash = sys.argv[2]
sample = sys.argv[3]
refsample = sys.argv[4]
if len(sys.argv) > 5:
    runType = sys.argv[5]
else:
    runType = "unified"
if len(sys.argv) > 6:
    refRunType = sys.argv[6]
else:
    refRunType = "unified"
pathname = ""
refpathname = ""
if len(sys.argv) > 9:
    pathname = sys.argv[9]
    refpathname = sys.argv[10]
desc = ""
refdesc = ""
if len(sys.argv) > 11:
    desc = sys.argv[11]
    refdesc = sys.argv[12]

r.gROOT.SetBatch(True)

def parse_plot_name(output_name):
    if "fakerate" in output_name:
        rtnstr = ["Fake Rate of"]
    elif "duplrate" in output_name:
        rtnstr = ["Duplicate Rate of"]
    else:
        rtnstr = ["Efficiency of"]
    if "MD_" in output_name:
        rtnstr.append("Mini-Doublet")
    elif "LS_" in output_name and "pLS" not in output_name:
        rtnstr.append("Line Segment")
    elif "pT4_" in output_name:
        rtnstr.append("Quadruplet w/ Pixel LS")
    elif "T4_" in output_name:
        rtnstr.append("Quadruplet w/o gap")
    elif "T4x_" in output_name:
        rtnstr.append("Quadruplet w/ gap")
    elif "pT3_" in output_name:
        rtnstr.append("Pixel Triplet")
    elif "pT5_" in output_name:
        rtnstr.append("Pixel Quintuplet")
    elif "T3_" in output_name:
        rtnstr.append("Triplet")
    elif "TC_" in output_name:
        rtnstr.append("Track Candidate")
    elif "T4s_" in output_name:
        rtnstr.append("Quadruplet w/ or w/o gap")
    elif "pLS_" in output_name:
        rtnstr.append("Pixel Line Segment")
    elif "T5_" in output_name:
        rtnstr.append("Quintuplet")
    elif "TCE_" in output_name:
        rtnstr.append("Track Extension")
    types = "of type " + os.path.basename(output_name).split("_")[1]
    if "AllTypes" in types:
        types = "of all types"
    if "Set1Types" in types:
        types = "of set 1 types"
    rtnstr.append(types)
    return " ".join(rtnstr)


# Get the files to be compared
if len(sys.argv) <= 4:
    eff_files_cpu = glob.glob("efficiencies/eff_plots__GPU_*{}_{}/efficiencies.root".format(refgithash, refsample))
else:
    eff_file_cpu = glob.glob("efficiencies/{}/eff_plots__GPU_{}_{}_{}{}/efficiencies.root".format(refpathname, refRunType, refgithash, refsample, refdesc))

eff_files_gpu = glob.glob("efficiencies/{}/eff_plots__GPU_{}_{}_{}{}/efficiencies.root".format(pathname, runType, githash, sample, desc))

# Get cpu efficiency graph files
cpu_file = r.TFile(eff_file_cpu[0])

# Obtain the list of TGraphs to compare
keys = [ x.GetName() for x in cpu_file.GetListOfKeys() if "Root__" not in x.GetName() ]
## Print for debugging
#for key in keys:
#    print key

# List to hold the tgraphs of the CPU histograms
cpu_tgraphs = {}
for key in keys:
    cpu_tgraphs[key] = cpu_file.Get(key)

## Printing CPU tgraphs for debugging
#for cpu_tgraph in cpu_tgraphs:
#    cpu_tgraph.Print()

ms = [24, 25, 26, 27, 28, 30, 32, 42, 46, 40]
cs = [2, 3, 4, 6, 7, 8, 9, 30, 46, 38, 40]

gpu_tgraphs = {}
configurations = []
for eff_file_gpu in eff_files_gpu:
    configuration = os.path.basename(os.path.dirname(eff_file_gpu)).split("GPU_")[1].split("_{}".format(githash))[0]
    configurations.append(configuration)
    gpu_file = r.TFile(eff_file_gpu)
    gpu_tgraphs[configuration] = {}
    for key in keys:
        gpu_tgraphs[configuration][key] = gpu_file.Get(key)
        if "GPU_unified" in eff_files_gpu and "TC_AllTypes__pt" in key:
            print(gpu_tgraphs[configuration][key].Print())

for key in keys:


    eff = cpu_tgraphs[key]
    output_name = "efficiencies/eff_comparison_plots__{}_{}_{}{}_{}_{}_{}{}/mtv/{}".format(sample, githash, runType, desc, refsample, refgithash, refRunType, refdesc, key) + ".pdf"

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

    if "fakerate" in key or "duplrate" in key or "pT3" in key:
        leg1 = r.TLegend(0.63, 0.67, 0.93, 0.87)
    else:
        leg1 = r.TLegend(0.63, 0.18, 0.93, 0.38)

    for i in xrange(0, eff.GetN()):
        if yaxis_max < eff.GetY()[i]:
            yaxis_max = eff.GetY()[i]
    # print yaxis_max
    yaxis_min = 999
    for i in xrange(0, eff.GetN()):
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
        eff.GetXaxis().SetLimits(-4.5, 4.5)

    eff.SetTitle(parse_plot_name(output_name))
    if len(sys.argv) > 6:
        leg1.AddEntry(eff,sys.argv[8], "ep")
    # Label
    t = r.TLatex()
    t.SetTextAlign(11) # align bottom left corner of text
    t.SetTextColor(r.kBlack)
    t.SetTextSize(0.04)
    x = r.gPad.GetX1() + r.gPad.GetLeftMargin()
    y = r.gPad.GetY2() - r.gPad.GetTopMargin() + 0.01
    sample_name_label = "Sample: " + sample + "   Reference sample: " + refsample + "   Version tag:" + githash + "   Reference version tag: " + refgithash
    t.DrawLatexNDC(x,y,"#scale[0.9]{#font[52]{%s}}" % sample_name_label)

    gpu_graphs = []
    for ii, config in enumerate(configurations):
        gpu_graphs.append(gpu_tgraphs[config][key])
        gpu_graphs[-1].SetMarkerStyle(ms[ii])
        # gpu_graphs[-1].SetMarkerStyle(19)
        gpu_graphs[-1].SetMarkerSize(1.2)
        gpu_graphs[-1].SetLineWidth(1)
        gpu_graphs[-1].SetMarkerColor(cs[ii])
        gpu_graphs[-1].SetLineColor(cs[ii])
        gpu_graphs[-1].Draw("ep")
        leg1.AddEntry(gpu_graphs[-1], sys.argv[7], "ep")

    leg1.Draw()
    # Save
    c1.SetGrid()
    c1.SaveAs("{}".format(output_name))
    c1.SaveAs("{}".format(output_name.replace(".pdf", ".png")))

    leg = r.TLegend(0.15, 0.15, 0.5, 0.4)
    gpu_ratio = []
    good_comparisons = {}
    for ii, config in enumerate(configurations):
        gpu_ratio.append(gpu_graphs[ii].Clone())
        good_comparison = True
        for ix in xrange(gpu_graphs[ii].GetN()):
            gpu_val = gpu_graphs[ii].GetPointY(ix)
            cpu_val = cpu_tgraphs[key].GetPointY(ix)
            ratio = 1
            if gpu_val != 0:
                ratio = cpu_val / gpu_val
            else:
                if cpu_val == 0:
                    ratio = 1
                else:
                    ratio = 0
            if ratio != 1:
                good_comparison = False
            gpu_ratio[-1].SetPointY(ix, ratio)
            gpu_ratio[-1].SetPointEYhigh(ix, 0)
            gpu_ratio[-1].SetPointEYlow(ix, 0)
        if not good_comparison:
            print("VALIDATION FAILED! :: Config = {} for output_name = {} failed to agree!".format(config, output_name))
        good_comparisons[config] = good_comparison
        gpu_ratio[-1].SetMarkerStyle(ms[ii])
        gpu_ratio[-1].SetMarkerSize(1.2)
        gpu_ratio[-1].SetLineWidth(1)
        gpu_ratio[-1].SetMarkerColor(cs[ii])
        gpu_ratio[-1].SetLineColor(cs[ii])
        if ii == 0:
            gpu_ratio[-1].SetTitle(str(gpu_ratio[-1].GetTitle()).replace("Efficiency", "Efficiency ratio"))
            gpu_ratio[-1].GetYaxis().SetTitle("Efficiency CPU / GPU")
            gpu_ratio[-1].GetYaxis().SetRangeUser(0.8, 1.2)
            gpu_ratio[-1].Draw("epa")
            leg.AddEntry(gpu_ratio[-1], config, "ep")
        else:
            gpu_ratio[-1].Draw("ep")
            leg.AddEntry(gpu_ratio[-1], config, "ep")

    leg.Draw()

    # Save
    c1.SetGrid()
    c1.SaveAs("{}".format(output_name.replace(".pdf", "_ratio.pdf")))
    c1.SaveAs("{}".format(output_name.replace(".pdf", "_ratio.png")))



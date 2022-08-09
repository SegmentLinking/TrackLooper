#!/bin/env python

import os
import sys
import glob
import ROOT as r
from plot_performance import parse_plot_name

# Get the git hash to compare
githash = sys.argv[1]
sample = sys.argv[2]

# Get the files to be compared
cpu_benchmark_file_path = ""
if sample == "muonGun":
    cpu_benchmark_file_path = os.getenv("LATEST_CPU_BENCHMARK_EFF_MUONGUN")
elif sample == "pionGun":
    cpu_benchmark_file_path = os.getenv("LATEST_CPU_BENCHMARK_EFF_PIONGUN")
elif sample == "PU200":
    cpu_benchmark_file_path = os.getenv("LATEST_CPU_BENCHMARK_EFF_PU200")
else:
    sys.exit("ERROR: Sample = {} does not have a corresponding CPU benchmark yet!".format(sample))
eff_file_cpu = glob.glob(cpu_benchmark_file_path) # Benchmark
eff_files_gpu = glob.glob("efficiencies/*_GPU_*{}_{}/efficiencies.root".format(githash, sample))

# Get cpu efficiency graph files
cpu_file = r.TFile(eff_file_cpu[0])

# Obtain the list of TGraphs to compare
keys = [ x.GetName() for x in cpu_file.GetListOfKeys() if "Root__" not in x.GetName() ]
# # Print for debugging
# for key in keys:
#     print key

# List to hold the tgraphs of the CPU histograms
cpu_tgraphs = {}
for key in keys:
    cpu_tgraphs[key] = cpu_file.Get(key)

# # Printing CPU tgraphs for debugging
# for cpu_tgraph in cpu_tgraphs:
#     cpu_tgraph.Print()

ms = [24, 25, 26, 27, 28, 30, 32, 42, 46, 40]
cs = [2, 3, 4, 6, 7, 8, 9, 30, 46, 38, 40]

gpu_tgraphs = {}
configurations = []
for eff_file_gpu in eff_files_gpu:
    configuration = os.path.basename(os.path.dirname(eff_file_gpu)).split("GPU_")[1].split("_{}".format(githash))[0]
    tempf = r.TFile(eff_file_gpu)
    if tempf.Get(keys[0]):
        configurations.append(configuration)
    else:
        continue
    gpu_file = r.TFile(eff_file_gpu)
    gpu_tgraphs[configuration] = {}
    for key in keys:
        gpu_tgraphs[configuration][key] = gpu_file.Get(key)
        if "GPU_unified" in eff_files_gpu and "TC_AllTypes__pt" in key:
            print(gpu_tgraphs[configuration][key].Print())

for key in keys:


    eff = cpu_tgraphs[key]
    output_name = "plots/{}".format(key) + ".pdf"
    sample_name = sample

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
    for i in xrange(0, eff.GetN()):
        if yaxis_max < eff.GetY()[i]:
            yaxis_max = eff.GetY()[i]
    # print yaxis_max
    yaxis_min = 999
    for i in xrange(0, eff.GetN()):
        if yaxis_min > eff.GetY()[i] and eff.GetY()[i] != 0:
            yaxis_min = eff.GetY()[i]
    # print yaxis_min
    if "eta" in output_name:
        eff.GetXaxis().SetLimits(-4.5, 4.5)
    if "ptzoom" in output_name:
        eff.GetYaxis().SetRangeUser(yaxis_max - 0.12, yaxis_max + 0.02)
    if "etazoom" in output_name:
        eff.GetYaxis().SetRangeUser(yaxis_max - 0.12, yaxis_max + 0.02)
    if "ptmaxzoom" in output_name:
        eff.GetYaxis().SetRangeUser(yaxis_max - 0.02, yaxis_max + 0.02)
    if "etamaxzoom" in output_name:
        eff.GetYaxis().SetRangeUser(yaxis_max - 0.02, yaxis_max + 0.02)
    eff.SetTitle(parse_plot_name(output_name))
    # Label
    t = r.TLatex()
    t.SetTextAlign(11) # align bottom left corner of text
    t.SetTextColor(r.kBlack)
    t.SetTextSize(0.04)
    x = r.gPad.GetX1() + r.gPad.GetLeftMargin()
    y = r.gPad.GetY2() - r.gPad.GetTopMargin() + 0.01
    sample_name_label = "Sample: " + sample_name + "   Version tag:" + githash
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



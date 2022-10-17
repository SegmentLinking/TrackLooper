#!/bin/env python

import argparse
import ROOT as r
from array import array
import os
import sys
from math import sqrt

# ptcut = 1.5
# etacut = 2.4
ptcut = 0.9
etacut = 4.5

r.gROOT.SetBatch(True)

#______________________________________________________________________________________________________
def parse_plot_name(output_name):
    if "fake" in output_name:
        rtnstr = ["Fake Rate of"]
    elif "dup" in output_name:
        rtnstr = ["Duplicate Rate of"]
    elif "inefficiency" in output_name:
        rtnstr = ["Inefficiency of"]
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
    elif "TCE_" in output_name:
        rtnstr.append("Extended Track")
    elif "T3T3_" in output_name:
        rtnstr.append("T3T3 Extensions")
    elif "pureTCE_" in output_name:
        rtnstr.append("Pure Extensions")
    elif "TC_" in output_name:
        rtnstr.append("Track Candidate")
    elif "T4s_" in output_name:
        rtnstr.append("Quadruplet w/ or w/o gap")
    elif "pLS_" in output_name:
        rtnstr.append("Pixel Line Segment")
    elif "T5_" in output_name:
        rtnstr.append("Quintuplet")
    return " ".join(rtnstr)

#______________________________________________________________________________________________________
def get_pdgidstr(pdgid):
    if abs(pdgid) == 0: return "All"
    elif abs(pdgid) == 11: return "Electron"
    elif abs(pdgid) == 13: return "Muon"
    elif abs(pdgid) == 211: return "Pion"

#______________________________________________________________________________________________________
def draw_ratio(nums, den, legend_labels, params):

    output_name = params["output_img_name"] 

    if "scalar" in output_name and "ptscalar" not in output_name:
        for num in nums:
            num.Rebin(180)
        den.Rebin(180)

    if "coarse" in output_name and "ptcoarse" not in output_name:
        for num in nums:
            num.Rebin(6)
        den.Rebin(6)

    if "pt" in output_name:
        for num in nums:
            overFlowBin = num.GetBinContent(num.GetNbinsX() + 1)
            lastBin = num.GetBinContent(num.GetNbinsX())
            num.SetBinContent(num.GetNbinsX(), lastBin + overFlowBin)
            num.SetBinError(num.GetNbinsX(), sqrt(lastBin + overFlowBin))
        overFlowBin = den.GetBinContent(den.GetNbinsX() + 1)
        lastBin = den.GetBinContent(den.GetNbinsX())
        den.SetBinContent(den.GetNbinsX(), lastBin + overFlowBin)
        den.SetBinError(den.GetNbinsX(), sqrt(lastBin + overFlowBin))

    teffs = []
    effs = []
    for num in nums:
        teff = r.TEfficiency(num, den)
        eff = teff.CreateGraph()
        teffs.append(teff)
        effs.append(eff)

    hist_name_suffix = ""
    if params["xcoarse"]:
        hist_name_suffix += "coarse"
    if params["yzoom"]:
        hist_name_suffix += "zoom"

    if params["output_file"]:
        params["output_file"].cd()
        basename = os.path.basename(output_name)
        outputname = basename.replace(".pdf","")
        den.Write(den.GetName() + hist_name_suffix, r.TObject.kOverwrite)
        eff_den = r.TGraphAsymmErrors(den)
        eff_den.SetName(outputname+"_den")
        eff_den.Write("", r.TObject.kOverwrite)
        for num in nums:
            num.Write(num.GetName() + hist_name_suffix, r.TObject.kOverwrite)
            eff_num = r.TGraphAsymmErrors(num)
            eff_num.SetName(outputname+"_num")
            eff_num.Write("", r.TObject.kOverwrite)
        for eff in effs:
            eff.SetName(outputname)
            eff.Write("", r.TObject.kOverwrite)

    output_name = params["output_img_name"]
    sample_name = params["sample_name"]
    version_tag = params["git_hash"]
    pdgidstr = params["pdgidstr"]
    draw_plot(effs, nums, den, legend_labels, output_name, sample_name, version_tag, pdgidstr)

#______________________________________________________________________________________________________
def set_label(eff, output_name, raw_number):
    if "phi" in output_name:
        title = "#phi"
    elif "_dz" in output_name:
        title = "z [cm]"
    elif "_dxy" in output_name:
        title = "d0 [cm]"
    elif "_pt" in output_name:
        title = "p_{T} [GeV]"
    elif "_hit" in output_name:
        title = "hits"
    elif "_lay" in output_name:
        title = "layers"
    else:
        title = "#eta"
    eff.GetXaxis().SetTitle(title)
    if "fakerate" in output_name:
        eff.GetYaxis().SetTitle("Fake Rate")
    elif "duplrate" in output_name:
        eff.GetYaxis().SetTitle("Duplicate Rate")
    elif "inefficiency" in output_name:
        eff.GetYaxis().SetTitle("Inefficiency")
    else:
        eff.GetYaxis().SetTitle("Efficiency")
    if raw_number:
        eff.GetYaxis().SetTitle("# of objects of interest")
    eff.GetXaxis().SetTitleSize(0.05)
    eff.GetYaxis().SetTitleSize(0.05)
    eff.GetXaxis().SetLabelSize(0.05)
    eff.GetYaxis().SetLabelSize(0.05)

#______________________________________________________________________________________________________
def draw_label(version_tag, sample_name, pdgidstr, output_name):
    # Label
    t = r.TLatex()

    # Draw information about sample, git version, and types of particles
    t.SetTextAlign(11) # align bottom left corner of text
    t.SetTextColor(r.kBlack)
    t.SetTextSize(0.04)
    x = r.gPad.GetX1() + r.gPad.GetLeftMargin()
    y = r.gPad.GetY2() - r.gPad.GetTopMargin() + 0.09 + 0.03
    sample_name_label = "Sample:" + sample_name
    sample_name_label += "   Version tag:" + version_tag
    if n_events_processed:
        sample_name_label +="  N_{evt}:" + n_events_processed
    t.DrawLatexNDC(x,y,"#scale[0.9]{#font[42]{%s}}" % sample_name_label)

    x = r.gPad.GetX1() + r.gPad.GetLeftMargin()
    y = r.gPad.GetY2() - r.gPad.GetTopMargin() + 0.045 + 0.03

    # If efficiency plots follow the following fiducial label rule
    ptcut = 0.9
    etacut = 4.5
    if "eff" in output_name:
        if "_pt" in output_name:
            fiducial_label = "|#eta| < {eta}, |Vtx_{{z}}| < 30 cm, |Vtx_{{xy}}| < 2.5 cm".format(eta=etacut)
        elif "_eta" in output_name:
            fiducial_label = "p_{{T}} > {pt} GeV, |Vtx_{{z}}| < 30 cm, |Vtx_{{xy}}| < 2.5 cm".format(pt=ptcut)
        elif "_dz" in output_name:
            fiducial_label = "|#eta| < {eta}, p_{{T}} > {pt} GeV, |Vtx_{{xy}}| < 2.5 cm".format(pt=ptcut, eta=etacut)
        elif "_dxy" in output_name:
            fiducial_label = "|#eta| < {eta}, p_{{T}} > {pt} GeV, |Vtx_{{z}}| < 30 cm".format(pt=ptcut, eta=etacut)
        else:
            fiducial_label = "|#eta| < {eta}, p_{{T}} > {pt} GeV, |Vtx_{{z}}| < 30 cm, |Vtx_{{xy}}| < 2.5 cm".format(pt=ptcut, eta=etacut)
        particleselection = ((", Particle:" + pdgidstr) if pdgidstr else "" )
        fiducial_label += particleselection
    # If fake rate or duplicate rate plot follow the following fiducial label rule
    elif "fakerate" in output_name or "duplrate" in output_name:
        if "_pt" in output_name:
            fiducial_label = "|#eta| < {eta}".format(eta=etacut)
        elif "_eta" in output_name:
            fiducial_label = "p_{{T}} > {pt} GeV".format(pt=ptcut)
        else:
            fiducial_label = "|#eta| < {eta}, p_{{T}} > {pt} GeV".format(pt=ptcut, eta=etacut)
    t.DrawLatexNDC(x,y,"#scale[0.9]{#font[42]{%s}}" % fiducial_label)

    # Draw CMS label
    cms_label = "Simulation"
    x = r.gPad.GetX1() + r.gPad.GetLeftMargin()
    y = r.gPad.GetY2() - r.gPad.GetTopMargin() + 0.005
    t.DrawLatexNDC(x,y,"#scale[1.25]{#font[61]{CMS}} #scale[1.1]{#font[52]{%s}}" % cms_label)

#______________________________________________________________________________________________________
def draw_plot(effs, nums, den, legend_labels, output_name, sample_name, version_tag, pdgidstr):

    # Get Canvas
    c1 = r.TCanvas()
    c1.SetBottomMargin(0.15)
    c1.SetLeftMargin(0.15)
    c1.SetTopMargin(0.22)
    c1.SetRightMargin(0.15)

    # Set logx
    if "_pt" in output_name:
        c1.SetLogx()

    # Set title
    print(output_name)
    print(parse_plot_name(output_name))
    effs[0].SetTitle(parse_plot_name(output_name))

    # Draw the efficiency graphs
    colors = [1, 2, 3, 4, 6]
    markerstyles = [19, 26, 28, 24, 27]
    markersize = 1.2
    linewidth = 2
    for i, eff in enumerate(effs):
        if i == 0:
            eff.Draw("epa")
        else:
            eff.Draw("epsame")
        eff.SetMarkerStyle(markerstyles[i])
        eff.SetMarkerSize(markersize)
        eff.SetLineWidth(linewidth)
        eff.SetMarkerColor(colors[i])
        eff.SetLineColor(colors[i])
        set_label(eff, output_name, raw_number=False)

    nleg = len(legend_labels)
    legend = r.TLegend(0.15,0.75-nleg*0.04,0.25,0.75)
    for i, label in enumerate(legend_labels):
        legend.AddEntry(effs[i], label)
    legend.Draw("same")

    # Compute the yaxis_max
    yaxis_max = 0
    for i in xrange(0, eff.GetN()):
        if yaxis_max < eff.GetY()[i]:
            yaxis_max = eff.GetY()[i]

    # Compute the yaxis_min
    yaxis_min = 999
    for i in xrange(0, eff.GetN()):
        if yaxis_min > eff.GetY()[i] and eff.GetY()[i] != 0:
            yaxis_min = eff.GetY()[i]

    # Set Yaxis range
    if "zoom" not in output_name:
        eff.GetYaxis().SetRangeUser(0, 1.02)
    else:
        if "fakerate" in output_name:
            eff.GetYaxis().SetRangeUser(0.0, yaxis_max * 1.1)
        elif "duplrate" in output_name:
            eff.GetYaxis().SetRangeUser(0.0, yaxis_max * 1.1)
        else:
            eff.GetYaxis().SetRangeUser(0.6, 1.02)

    # Set xaxis range
    if "eta" in output_name:
        eff.GetXaxis().SetLimits(-4.5, 4.5)

    # Draw label
    draw_label(version_tag, sample_name, pdgidstr, output_name)

    # Save
    c1.SetGrid()
    c1.SaveAs("{}".format(output_name.replace("/mtv/", "/mtv/var/")))
    c1.SaveAs("{}".format(output_name.replace("/mtv/", "/mtv/var/").replace(".pdf", ".png")))
    eff.SetName(output_name.replace(".png",""))

    for i, num in enumerate(nums):
        set_label(num, output_name, raw_number=True)
        num.Draw("hist")
        c1.SaveAs("{}".format(output_name.replace("/mtv/", "/mtv/num/").replace(".pdf", "_num{}.pdf".format(i))))
        c1.SaveAs("{}".format(output_name.replace("/mtv/", "/mtv/num/").replace(".pdf", "_num{}.png".format(i))))

    set_label(den, output_name, raw_number=True)
    den.Draw("hist")
    c1.SaveAs("{}".format(output_name.replace("/mtv/", "/mtv/den/").replace(".pdf", "_den.pdf")))
    c1.SaveAs("{}".format(output_name.replace("/mtv/", "/mtv/den/").replace(".pdf", "_den.png")))

    # Double ratio if more than one nums are provided
    # Take the first num as the base
    if len(nums) > 1:
        base = nums[0].Clone()
        base.Divide(nums[0], den, 1, 1, "B") #Binomial
        others = []
        for num in nums[1:]:
            other = num.Clone()
            other.Divide(other, den, 1, 1, "B")
            others.append(other)

        # Take double ratio
        for other in others:
            other.Divide(base)

        for i, other in enumerate(others):
            other.Draw("ep")
            other.GetYaxis().SetTitle("{} / {}".format(legend_labels[i+1], legend_labels[i]))
            other.SetMarkerStyle(markerstyles[i+1])
            other.SetMarkerSize(markersize)
            other.SetMarkerColor(colors[i+1])
            other.SetLineWidth(linewidth)
            other.SetLineColor(colors[i+1])
            c1.SaveAs("{}".format(output_name.replace("/mtv/", "/mtv/ratio/").replace(".pdf", "_ratio{}.pdf".format(i))))
            c1.SaveAs("{}".format(output_name.replace("/mtv/", "/mtv/ratio/").replace(".pdf", "_ratio{}.png".format(i))))

#______________________________________________________________________________________________________
def plot_standard_performance_plots(params):

    # Output file
    params["output_file"] = r.TFile("{output_dir}/efficiency.root".format(**params), "recreate")

    # git version hash
    params["input_file"].Get("githash").Write()

    # sample name
    params["input_file"].Get("input").Write()

    # Efficiency plots
    metricsuffixs = ["ef_", "fr_", "dr_"]
    ybins = ["", "zoom"]
    variables = {
            "ef_": ["pt", "ptlow", "ptmtv", "eta", "phi", "dxy", "dz"],
            "fr_": ["pt", "ptlow", "ptmtv", "eta", "phi"],
            "dr_": ["pt", "ptlow", "ptmtv", "eta", "phi"],
            }
    xbins = {
            "pt": [""],
            "ptlow": [""],
            "ptmtv": [""],
            "eta": ["", "coarse"],
            "phi": ["", "coarse"],
            "dxy": ["", "coarse"],
            "dz": ["", "coarse"],
            }
    types = ["TC", "pT5", "pT3", "T5", "pLS"]
    pdgids = [0, 11, 13, 211]

    for metricsuffix in metricsuffixs:
        if metricsuffix == "ef_":
            metric = "eff"
        if metricsuffix == "fr_":
            metric = "fakerate"
        if metricsuffix == "dr_":
            metric = "duplrate"
        for variable in variables[metricsuffix]:
            for ybin in ybins:
                for xbin in xbins[variable]:
                    for typ in types:
                        isstacks = [False]
                        if typ == "TC":
                            isstacks = [True, False]
                        for isstack in isstacks:
                            for pdgid in pdgids:
                                # print(metricsuffix, variable, ybin, xbin, typ, isstack, pdgid)
                                params["xcoarse"] = xbin == "coarse"
                                params["yzoom"] = ybin == "zoom"
                                params["variable"] = variable
                                params["metricsuffix"] = metricsuffix
                                params["metric"] = metric
                                params["is_stack"] = isstack
                                params["stackvar"] = "_stack" if isstack else ""
                                params["objecttype"] = typ
                                params["pdgid"] = pdgid
                                params["pdgidstr"] = get_pdgidstr(pdgid)
                                plot(params)

#______________________________________________________________________________________________________
def plot(params):

    variable = params["variable"]
    yzoom = params["yzoom"]
    xcoarse = params["xcoarse"]
    objecttype = params["objecttype"]
    metricsuffix = params["metricsuffix"]
    is_stack = params["is_stack"]
    pdgidstr = params["pdgidstr"]

    params["objecttypesuffix"] = "_{}".format(pdgid) if metricsuffix == "ef_" else ""
    params["pdgidstr"] = get_pdgidstr(pdgid)
    params["output_plot_name"] = "{objecttype}{objecttypesuffix}_{metric}{stackvar}_{variable}".format(**params)
    if params["xcoarse"]:
        params["output_plot_name"] += "coarse"
    if params["yzoom"]:
        params["output_plot_name"] += "zoom"
    params["output_img_name"] = "{output_dir}/mtv/{output_plot_name}.pdf".format(**params)

    # print("Drawing")
    # for key in params:
    #     print(key, params[key])

    # Get histogram names
    denom_histname = "Root__{objecttype}{objecttypesuffix}_{metricsuffix}denom_{variable}".format(**params)
    numer_histname = "Root__{objecttype}{objecttypesuffix}_{metricsuffix}numer_{variable}".format(**params)

    # Denom histogram
    denom = f.Get(denom_histname).Clone()

    # Numerator histograms
    numer = f.Get(numer_histname).Clone()

    stack_hist_types = ["pT5", "pT3", "T5", "pLS"]
    stack_hists = []
    if is_stack:
        for stack_hist_type in stack_hist_types:
            stack_histname = numer_histname.replace("TC", stack_hist_type)
            hist = f.Get(stack_histname)
            stack_hists.append(hist.Clone())

    legend_labels = ["TC" ,"pT5" ,"pT3" ,"T5" ,"pLS"]

    if is_stack:
        draw_ratio(
                [numer] + stack_hists, # numerator histogram
                denom, # denominator histogram
                legend_labels, # legend_labels
                params,
                )
    else:
        draw_ratio(
                [numer], # numerator histogram
                denom, # denominator histogram
                legend_labels[:1], # legend_labels
                params,
                )

#______________________________________________________________________________________________________
def compare(file1, file2, legend0, legend1):

    f1 = r.TFile(file1)
    f2 = r.TFile(file2)

    h1 = f1.Get("githash").GetTitle()
    h2 = f2.Get("githash").GetTitle()

    i1 = f1.Get("input").GetTitle()
    i2 = f2.Get("input").GetTitle()

    params = {}
    params["output_dir"] = "comparison/{}_v_{}".format(h1, h2)

    # Create output directory
    os.system("mkdir -p {output_dir}/mtv/var".format(**params))
    os.system("mkdir -p {output_dir}/mtv/num".format(**params))
    os.system("mkdir -p {output_dir}/mtv/den".format(**params))
    os.system("mkdir -p {output_dir}/mtv/ratio".format(**params))

    ratios = []
    numers = []
    denoms = []
    for key in f1.GetListOfKeys():
        if key.GetName() == "githash":
            continue
        if key.GetName() == "input":
            continue
        if "_num" not in key.GetName() and "_den" not in key.GetName() and "stack" not in key.GetName():
            ratio_name = key.GetName()
            hist_suffix = ""
            if "eff" in ratio_name:
                metric = "eff"
                metric_suffix = "ef"
            if "fakerate" in ratio_name:
                metric = "fakerate"
                metric_suffix = "fr"
            if "duplrate" in ratio_name:
                metric = "duplrate"
                metric_suffix = "dr"
            if f2.Get(ratio_name):
                ratios.append(ratio_name)
                numers.append("Root__" + ratio_name.replace(metric, metric_suffix + "_numer"))
                denoms.append("Root__" + ratio_name.replace(metric, metric_suffix + "_denom"))

    legend_labels = []
    legend_labels.append("{}".format(legend0))
    legend_labels.append("{}".format(legend1))
    for eff, num, den in zip(ratios, numers, denoms):
        effs = [f1.Get(eff), f2.Get(eff)]
        nums = [f1.Get(num), f2.Get(num)]
        hden = f1.Get(den)
        params["histname"] = effs[0].GetName()
        params["pdgidstr"] = params["histname"].split("_")[1]
        params["output_file_name"] = "{output_dir}/mtv/{histname}.pdf".format(**params)
        draw_plot(effs, nums, hden, legend_labels, params["output_file_name"], "{}, {}".format(i1, i2), "{}, {}".format(h1, h2), params["pdgidstr"])

    DIR = os.path.realpath(os.path.dirname(__file__))
    os.system("cp -r {}/../misc/compare {}/".format(DIR, params["output_dir"]))

if __name__ == "__main__":

    # Argument parsing
    parser = argparse.ArgumentParser(description="What are we wanting to graph?")
    # parser.add_argument('--input'               , '-i'    , dest='input'       , type=str , default='num_den_hist.root' , help='input file name [DEFAULT=num_den_hist.root]')
    parser.add_argument('--compare_legend'      , '-l'    , dest='comp_leg'    , type=str , default=''                  , help='comma separated lables for comparison')
    parser.add_argument('--variable'            , '-v'    , dest='variable'    , type=str , default='pt'                , help='pt, eta, phi, dxy, dz [DEFAULT=pt]')
    parser.add_argument('--objecttype'          , '-o'    , dest='objecttype'  , type=str , default='TC'                , help='TC, pT3, T5, pT5, pLS [DEFAULT=TC]')
    parser.add_argument('--metric'              , '-m'    , dest='metric'      , type=str , default='eff'               , help='eff, duplrate, fakerate [DEFAULT=eff]')
    parser.add_argument('--sample_name'         , '-s'    , dest='sample_name' , type=str , default='DEFAULT'           , help='sample name [DEFAULT=sampleName]')
    parser.add_argument('--dirname'             , '-d'    , dest='dirname'     , type=str , default='performance'       , help='main performance output dir name [DEFAULT=performance]')
    parser.add_argument('--tag'                 , '-t'    , dest='tag'         , type=str , default='v0'                , help='tag of the run [DEFAULT=v0]')
    parser.add_argument('--pdgid'               , '-p'    , dest='pdgid'       , type=int , default=0                   , help='pdgid (efficiency plots only) [DEFAULT=0]')
    parser.add_argument('--is_stack'            , '-T'    , dest='is_stack'               , action="store_true"         , help='is stack')
    parser.add_argument('--yzoom'               , '-y'    , dest='yzoom'                  , action="store_true"         , help='zoom in y')
    parser.add_argument('--xcoarse'             , '-x'    , dest='xcoarse'                , action="store_true"         , help='coarse in x')
    parser.add_argument('--single_plot'         , '-1'    , dest='single_plot'            , action="store_true"         , help='plot only one plot')
    parser.add_argument('inputs', nargs='+', help='input num_den_hist.root files')

    args = parser.parse_args()

    if len(args.inputs) > 1:
        print("Running on compare mode!")
        # TODO This is not the right way to handle!
        n_events_processed = None
        if not args.comp_leg:
            print("Need to provide --compare_legend! (e.g. --compare_legend mywork1,currentmaster")
            parser.print_help(sys.stderr)
            sys.exit(1)
        legend0 = args.comp_leg.split(",")[0]
        legend1 = args.comp_leg.split(",")[1]
        compare(args.inputs[0], args.inputs[1], legend0, legend1)
        sys.exit()

    #############
    variable   = args.variable
    yzoom      = args.yzoom
    xcoarse    = args.xcoarse
    objecttype = args.objecttype
    metric     = args.metric
    is_stack   = args.is_stack
    dirname    = args.dirname
    tag        = args.tag
    pdgid      = args.pdgid
    #############

    # Parse from input file
    root_file_name = args.inputs[0]
    f = r.TFile(root_file_name)

    # git version hash
    git_hash = f.Get("githash").GetTitle()

    # sample name
    sample_name = f.Get("input").GetTitle()
    if args.sample_name:
        sample_name = args.sample_name

    n_events_processed = str(int(f.Get("nevts").GetBinContent(1)))

    # parse metric suffix
    if metric == "eff": metricsuffix = "ef_"
    elif metric == "duplrate": metricsuffix = "dr_"
    elif metric == "fakerate": metricsuffix = "fr_"
    else:
        parser.print_help(sys.stderr)
        sys.exit(1)

    # If is_stack it must be object type of TC
    if is_stack:
        print("Warning! objecttype is set to \"TC\" because is_stack is True!")
        objecttype = "TC"

    # Building output dirname
    output_dir = "{dirname}/{tag}_{git_hash}".format(dirname=dirname, tag=tag, git_hash=git_hash)

    # Create output directory
    os.system("mkdir -p {output_dir}/mtv/var".format(output_dir=output_dir))
    os.system("mkdir -p {output_dir}/mtv/num".format(output_dir=output_dir))
    os.system("mkdir -p {output_dir}/mtv/den".format(output_dir=output_dir))
    os.system("mkdir -p {output_dir}/mtv/ratio".format(output_dir=output_dir))

    params = {
            "input_file"   : f,
            "sample_name"  : sample_name,
            "git_hash"     : git_hash,
            "variable"     : variable,
            "yzoom"        : yzoom,
            "variable"     : variable,
            "yzoom"        : yzoom,
            "xcoarse"      : xcoarse,
            "objecttype"   : objecttype,
            "metric"       : metric,
            "metricsuffix" : metricsuffix,
            "is_stack"     : is_stack,
            "stackvar"     : "_stack" if is_stack else "",
            "dirname"      : dirname,
            "output_dir"   : output_dir,
            "tag"          : tag,
            "pdgid"        : pdgid,
            "pdgidstr"     : get_pdgidstr(pdgid),
            "output_file"  : None,
            "nevts"        : n_events_processed,
            }

    # Output file
    params["output_file"] = r.TFile("{output_dir}/efficiency.root".format(**params), "update")

    # git version hash
    params["input_file"].Get("githash").Write("", r.TObject.kOverwrite)

    # sample name
    params["input_file"].Get("input").Write("", r.TObject.kOverwrite)

    # Draw all standard particle
    if args.single_plot:
        plot(params)
    else:
        plot_standard_performance_plots(params)

    DIR = os.path.realpath(os.path.dirname(__file__))
    os.system("cp -r {}/../misc/summary {}/".format(DIR, output_dir))

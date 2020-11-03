#!/bin/env python

import plottery_wrapper as p
from plottery import plottery as plt
import ROOT as r

import sys

filename = "debug.root"
if len(sys.argv) > 2:
    filename = sys.argv[2]

option = 1
if len(sys.argv) > 1:
    option = int(sys.argv[1])

tag = ""
if len(sys.argv) > 3:
    tag = sys.argv[3]


drawMDplots = False
drawSGplots =False
drawSGSelPlots = False
drawSGTruthSelPlots = False
drawTLplots = False
drawTLSelPlots = False
drawTCplots = False
drawTCSelPlots = False
drawTPSelPlots = False
drawMTVplots = False

if option == 1:
    drawMDplots = True

if option == 2:
    drawSGplots = True

if option == 3:
    drawTLplots = True

if option == 4:
    drawTCplots = True

if option == 5:
    drawTLSelPlots = True

if option == 6:
    drawTCSelPlots = True

if option == 7:
    drawTPSelPlots = True

if option == 8:
    drawMTVplots = True

eff_file = r.TFile("eff_{}.root".format(option), "recreate")

def plot_eff(num_name, den_name, output_name, dirname="lin", tag=""):
    f = r.TFile(filename)
    num = f.Get(num_name)
    den = f.Get(den_name)

    # if "_eta" in output_name:
    #     num.Rebin(2)
    #     den.Rebin(2)

    if "_phi" in output_name:
        num.Rebin(5)
        den.Rebin(5)

    suffix = ""
    if tag != "":
        suffix = "_" + tag

    if "_pt" in output_name:
        den.SetBinContent(1, 0)
        num.SetBinContent(1, 0)
        den.SetBinError(1, 0)
        num.SetBinError(1, 0)

    yaxis_log = False
    if "_dxy_" in output_name:
        yaxis_log = True
    if "_etalog_" in output_name:
        yaxis_log = True

    p.plot_hist(bgs=[den.Clone()],
            data=num.Clone(),
            options={
                "yaxis_log":yaxis_log,
                "legend_smart":False,
                "print_yield":False,
                "output_name":"plots{}/{}/{}".format(suffix, dirname, output_name.replace(".pdf","_numden.pdf")),
                # "remove_underflow": True if "_pt_mtv_numden" in output_name else False,
                # "remove_overflow":True,
                # "yaxis_range": [0.95, 1.05] if "eta" in output_name else [],
                "yaxis_range": [0.1, den.GetMaximum() * 1000] if "_dxy_" in output_name else [],
                # "no_ratio":False,
                "draw_points":False,
                "do_stack":False,
                # "print_yield":True,
                "yield_prec":4,
                # "xaxis_log":False if "eta" in output_name else True,
                # "hist_disable_xerrors": True if "eta" in output_name else False,
                # "hist_black_line": True,
                "show_bkg_errors": True,
                "hist_line_black": True,
                "ratio_range": [0., 1.05],
                # "divide_by_bin_width":True,
                "print_yield":True,
                "xaxis_log":True if "_pt" in output_name else False,
                # "xaxis_log":False,
                # "no_ratio":True,
                "ratio_binomial_errors":True,
                # "remove_overflow": True,
                },
        )

    teff = r.TEfficiency(num, den)
    eff = teff.CreateGraph()
    #eff.Print("all")
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
    if "_eta" in output_name:
        title = "#eta"
    elif "phi" in output_name:
        title = "#phi"
    elif "_z" in output_name:
        title = "z [cm]"
    elif "_dxy" in output_name:
        title = "d0 [cm]"
    else:
        title = "p_{T} [GeV]"
    eff.GetXaxis().SetTitle(title)
    eff.GetXaxis().SetTitleSize(0.05)
    eff.GetXaxis().SetLabelSize(0.05)
    eff.GetYaxis().SetLabelSize(0.05)
    yaxis_max = 0
    for i in xrange(0, eff.GetN()):
        if yaxis_max < eff.GetY()[i]:
            yaxis_max = eff.GetY()[i]
    print yaxis_max
    yaxis_min = 999
    for i in xrange(0, eff.GetN()):
        if yaxis_min > eff.GetY()[i] and eff.GetY()[i] != 0:
            yaxis_min = eff.GetY()[i]
    print yaxis_min
    if "eff_eta" in output_name and "sg_" not in output_name:
        eff.GetYaxis().SetRangeUser(0, 1.005)
    if "eff_z" in output_name and "sg_" not in output_name:
        eff.GetYaxis().SetRangeUser(0.98, 1.02)
    if "barrelflat_eta" in output_name:
        eff.GetYaxis().SetRangeUser(0.97, 1.03)
    if "eff_eta" in output_name and "sg_" in output_name:
        eff.GetYaxis().SetRangeUser(0.9, 1.005)
    if "eff_z" in output_name and "sg_" in output_name:
        eff.GetYaxis().SetRangeUser(0.98, 1.02)
    if "eff_ptzoom" in output_name:
        eff.GetYaxis().SetRangeUser(yaxis_max - 0.12, yaxis_max + 0.02)
    if "eff_etazoom" in output_name:
        eff.GetYaxis().SetRangeUser(yaxis_min - 0.02, yaxis_max + 0.02)
    if "_ptzoom" in output_name and "md_" in output_name:
        eff.GetYaxis().SetRangeUser(0.93, 1.05)
    if "_ptzoom" in output_name and "sg_" in output_name:
        eff.GetYaxis().SetRangeUser(0.90, 1.05)
    if "_ptzoom" in output_name and "tl_" in output_name:
        eff.GetYaxis().SetRangeUser(0.87, 1.05)
    if "_ptzoom" in output_name and "tc_" in output_name:
        eff.GetYaxis().SetRangeUser(0.82, 1.05)
    # if "_ptzoom" in output_name and "md_" in output_name:
    #     eff.GetYaxis().SetRangeUser(0.98, 1.02)
    # if "_ptzoom" in output_name and "tl_" in output_name:
    #     eff.GetYaxis().SetRangeUser(0.9, 1.1)
    # if "_ptzoom" in output_name and "tc_" in output_name:
    #     eff.GetYaxis().SetRangeUser(0.9, 1.1)
    if "eff_eta" in output_name and "tl_" in output_name:
        eff.GetYaxis().SetRangeUser(0.9, 1.005)
    if "_pt_by" in output_name and "tc_" in output_name:
        eff.GetYaxis().SetRangeUser(0.0, 1.1)
    if "_etazoom_by" in output_name:
        eff.GetYaxis().SetRangeUser(yaxis_min - 0.02, yaxis_max + 0.02)
    c1.SetGrid()
    c1.SaveAs("plots{}/{}/{}".format(suffix, dirname, output_name.replace(".pdf", "_eff.pdf")))
    c1.SaveAs("plots{}/{}/{}".format(suffix, dirname, output_name.replace(".pdf", "_eff.png")))
    eff_file.cd()
    eff.SetName(output_name.replace(".png",""))
    eff.Write()

def pos_neg_tracklet_comparison(filename, tlcombo, ptrange, suffix, do_und_ov_flow, plot_range="_maxzoom", iteration="_4thCorr"):

    tfile = r.TFile(filename)
    hist_neg = tfile.Get("Root__tl_matched_neg_{}_track_{}_deltaBeta{}{}".format(ptrange, tlcombo, iteration, plot_range))
    hist_pos = tfile.Get("Root__tl_matched_pos_{}_track_{}_deltaBeta{}{}".format(ptrange, tlcombo, iteration, plot_range))
    output_name = "Root__tl_matched_posnegcomp_{}_track_{}_deltaBeta{}{}".format(ptrange, tlcombo, iteration, plot_range)

    try:
        p.plot_hist(bgs=[hist_neg.Clone()],
                data=hist_pos.Clone(),
                options={
                    "output_name": "plots/tracklet{}/{}".format(suffix, output_name+".pdf"),
                    "nbins":45,
                    "yaxis_log":False, "legend_smart":False, "print_yield":True, "remove_overflow":do_und_ov_flow, "remove_underflow":do_und_ov_flow,
                    "ratio_range":[0., 3.],
                    },
                )
    except:
        print(ptrange, tlcombo, filename)

if drawMDplots:

    # mdcombos = ["barrel"]
    mdcombos = ["bbbbbb", "bbbbbe", "bbbbee", "bbbeee", "bbeeee", "beeeee"]

    for mdcombo in mdcombos:

        for i in xrange(6):
            plot_eff("Root__md_{}_matched_track_pt_by_layer{}".format(mdcombo, i), "Root__md_{}_all_track_pt_by_layer{}".format(mdcombo, i), "md_eff_{}_pt_by_layer{}.pdf".format(mdcombo, i), "mdeff", tag)
            plot_eff("Root__md_{}_matched_track_pt_by_layer{}".format(mdcombo, i), "Root__md_{}_all_track_pt_by_layer{}".format(mdcombo, i), "md_eff_{}_ptzoom_by_layer{}.pdf".format(mdcombo, i), "mdeff", tag)
            plot_eff("Root__md_{}_matched_track_eta_by_layer{}".format(mdcombo, i), "Root__md_{}_all_track_eta_by_layer{}".format(mdcombo, i), "md_eff_{}_eta_by_layer{}.pdf".format(mdcombo, i), "mdeff", tag)
            plot_eff("Root__md_{}_matched_track_eta_by_layer{}".format(mdcombo, i), "Root__md_{}_all_track_eta_by_layer{}".format(mdcombo, i), "md_eff_{}_etazoom_by_layer{}.pdf".format(mdcombo, i), "mdeff", tag)
            plot_eff("Root__md_{}_matched_track_eta_by_layer{}".format(mdcombo, i), "Root__md_{}_all_track_eta_by_layer{}".format(mdcombo, i), "md_eff_{}_etalog_by_layer{}.pdf".format(mdcombo, i), "mdeff", tag)
            plot_eff("Root__md_{}_matched_track_dxy_by_layer{}".format(mdcombo, i), "Root__md_{}_all_track_dxy_by_layer{}".format(mdcombo, i), "md_eff_{}_dxy_by_layer{}.pdf".format(mdcombo, i), "mdeff", tag)
            plot_eff("Root__md_{}_matched_track_dz_by_layer{}".format(mdcombo, i), "Root__md_{}_all_track_dz_by_layer{}".format(mdcombo, i), "md_eff_{}_dz_by_layer{}.pdf".format(mdcombo, i), "mdeff", tag)

if drawSGplots:

    # sgcombos = ["bbbbbb"]
    sgcombos = ["bbbbbb", "bbbbbe", "bbbbee", "bbbeee", "bbeeee", "beeeee"]

    for sgcombo in sgcombos:
        for i in xrange(5):
            plot_eff("Root__sg_{}_matched_track_pt_by_layer{}".format(sgcombo, i), "Root__sg_{}_all_track_pt_by_layer{}".format(sgcombo, i), "sg_eff_{}_pt_by_layer{}.pdf".format(sgcombo, i), "sgeff", tag)
            plot_eff("Root__sg_{}_matched_track_pt_by_layer{}".format(sgcombo, i), "Root__sg_{}_all_track_pt_by_layer{}".format(sgcombo, i), "sg_eff_{}_ptzoom_by_layer{}.pdf".format(sgcombo, i), "sgeff", tag)
            plot_eff("Root__sg_{}_matched_track_eta_by_layer{}".format(sgcombo, i), "Root__sg_{}_all_track_eta_by_layer{}".format(sgcombo, i), "sg_eff_{}_eta_by_layer{}.pdf".format(sgcombo, i), "sgeff", tag)
            plot_eff("Root__sg_{}_matched_track_eta_by_layer{}".format(sgcombo, i), "Root__sg_{}_all_track_eta_by_layer{}".format(sgcombo, i), "sg_eff_{}_etazoom_by_layer{}.pdf".format(sgcombo, i), "sgeff", tag)
            plot_eff("Root__sg_{}_matched_track_dxy_by_layer{}".format(sgcombo, i), "Root__sg_{}_all_track_dxy_by_layer{}".format(sgcombo, i), "sg_eff_{}_dxy_by_layer{}.pdf".format(sgcombo, i), "sgeff", tag)

if drawSGSelPlots:

    sgcombos = ["bb12", "bb23", "bb34", "bb45", "bb56"]
    recovars = ["zLo_cut", "sdCut", "sdSlope", "sdMuls", "sdPVoff", "deltaPhi"]

    for sgcombo in sgcombos:

        p.dump_plot(fnames=[filename],
            dirname="plots/segment",
            dogrep=False,
            filter_pattern="Root__sg_{}_cutflow".format(sgcombo),
            extraoptions={"yaxis_log":False, "legend_smart":False, "print_yield":True, "remove_overflow":False, "remove_underflow":False},
            )

        p.dump_plot(fnames=[filename],
            dirname="plots/segment_log",
            dogrep=False,
            filter_pattern="Root__sg_{}_cutflow".format(sgcombo),
            extraoptions={"yaxis_log":True, "legend_smart":False, "print_yield":True},
            )

        for recovar in recovars:

            p.dump_plot(fnames=[filename],
                dirname="plots/segment",
                dogrep=False,
                filter_pattern="Root__sg_{}_{}".format(sgcombo, recovar),
                extraoptions={"yaxis_log":False, "legend_smart":False, "print_yield":True, "remove_overflow":True, "remove_underflow":False},
                )

if drawSGTruthSelPlots:

    sgcombos = ["bb12", "bb23", "bb34", "bb45", "bb56"]
    recovars = ["zLo_cut", "sdCut", "sdSlope", "sdMuls", "sdPVoff", "deltaPhi"]

    for sgcombo in sgcombos:

        for recovar in recovars:

            p.dump_plot(fnames=[filename],
                dirname="plots/segment",
                dogrep=False,
                filter_pattern="Root__sg_truth_{}_{}".format(sgcombo, recovar),
                extraoptions={"yaxis_log":False, "legend_smart":False, "print_yield":True, "remove_overflow":True, "remove_underflow":False},
                )

if drawTLplots:

    # tlcombos = ["bbbbbb"]
    tlcombos = ["bbbbbb", "bbbbbe", "bbbbee", "bbbeee", "bbeeee", "beeeee"]

    for tlcombo in tlcombos:
        for i in xrange(3):
            plot_eff("Root__tl_{}_matched_track_pt_by_layer{}".format(tlcombo, i), "Root__tl_{}_all_track_pt_by_layer{}".format(tlcombo, i), "tl_eff_{}_pt_by_layer{}.pdf".format(tlcombo, i), "tleff", tag)
            plot_eff("Root__tl_{}_matched_track_pt_by_layer{}".format(tlcombo, i), "Root__tl_{}_all_track_pt_by_layer{}".format(tlcombo, i), "tl_eff_{}_ptzoom_by_layer{}.pdf".format(tlcombo, i), "tleff", tag)
            plot_eff("Root__tl_{}_matched_track_eta_by_layer{}".format(tlcombo, i), "Root__tl_{}_all_track_eta_by_layer{}".format(tlcombo, i), "tl_eff_{}_eta_by_layer{}.pdf".format(tlcombo, i), "tleff", tag)
            plot_eff("Root__tl_{}_matched_track_eta_by_layer{}".format(tlcombo, i), "Root__tl_{}_all_track_eta_by_layer{}".format(tlcombo, i), "tl_eff_{}_etazoom_by_layer{}.pdf".format(tlcombo, i), "tleff", tag)
            plot_eff("Root__tl_{}_matched_track_dxy_by_layer{}".format(tlcombo, i), "Root__tl_{}_all_track_dxy_by_layer{}".format(tlcombo, i), "tl_eff_{}_dxy_by_layer{}.pdf".format(tlcombo, i), "tleff", tag)

if drawTLSelPlots:

    # tlcombos = ["bb1bb3", "bb1bb4", "bb1bb5", "bb2bb4", "bb3bb5", "bb3be5", "bb1be3", "bb2be4", "ee1ee3", "bb1ee3", "all"]
    tlcombos = [
            "bb1bb3",
            "bb1be3",
            "bb1ee3",
            "be1ee3",
            "ee1ee3",

            "bb2bb4",
            "bb2be4",
            "bb2ee4",
            "be2ee4",
            "ee2ee4",

            "bb3bb5",
            "bb3be5",
            "bb3ee5",
            "be3ee5",
            ]
    # tlcombos = ["bb1bb3", "bb2bb4", "bb3bb5"]

    for tlcombo in tlcombos:

        for do_und_ov_flow in [True, False]:

            suffix = "" if do_und_ov_flow else "_w_overflow"

            # p.dump_plot(fnames=[filename],
            #     dirname="plots/tracklet{}".format(suffix),
            #     dogrep=False,
            #     filter_pattern="Root__tl_{}_deltaBeta".format(tlcombo),
            #     extraoptions={"yaxis_log":False, "legend_smart":False, "print_yield":False, "remove_overflow":do_und_ov_flow, "remove_underflow":do_und_ov_flow},
            #     )

            # p.dump_plot(fnames=[filename],
            #     dirname="plots/tracklet{}".format(suffix),
            #     dogrep=False,
            #     filter_pattern="Root__tl_{}_betaIn".format(tlcombo),
            #     extraoptions={"yaxis_log":False, "legend_smart":False, "print_yield":False, "remove_overflow":do_und_ov_flow, "remove_underflow":do_und_ov_flow},
            #     )

            # p.dump_plot(fnames=[filename],
            #     dirname="plots/tracklet{}".format(suffix),
            #     dogrep=False,
            #     filter_pattern="Root__tl_{}_betaOut".format(tlcombo),
            #     extraoptions={"yaxis_log":False, "legend_smart":False, "print_yield":False, "remove_overflow":do_und_ov_flow, "remove_underflow":do_und_ov_flow},
            #     )

            # p.dump_plot(fnames=[filename],
            #     dirname="plots/tracklet{}".format(suffix),
            #     dogrep=False,
            #     filter_pattern="Root__tl_{}_deltaBeta_midpoint".format(tlcombo),
            #     extraoptions={"yaxis_log":False, "legend_smart":False, "print_yield":False, "remove_overflow":do_und_ov_flow, "remove_underflow":do_und_ov_flow},
            #     )

            # p.dump_plot(fnames=[filename],
            #     dirname="plots/tracklet{}".format(suffix),
            #     dogrep=False,
            #     filter_pattern="Root__tl_{}_deltaBeta_3rdCorr".format(tlcombo),
            #     extraoptions={"yaxis_log":False, "legend_smart":False, "print_yield":False, "remove_overflow":do_und_ov_flow, "remove_underflow":do_und_ov_flow},
            #     )

            # p.dump_plot(fnames=[filename],
            #     dirname="plots/tracklet{}".format(suffix),
            #     dogrep=False,
            #     filter_pattern="Root__tl_{}_deltaBeta_4thCorr".format(tlcombo),
            #     extraoptions={"yaxis_log":False, "legend_smart":False, "print_yield":False, "remove_overflow":do_und_ov_flow, "remove_underflow":do_und_ov_flow},
            #     )

            # p.dump_plot(fnames=[filename],
            #     dirname="plots/tracklet{}".format(suffix),
            #     dogrep=False,
            #     filter_pattern="Root__tl_{}_deltaBeta_3rdCorr_maxzoom".format(tlcombo),
            #     extraoptions={"yaxis_log":False, "legend_smart":False, "print_yield":False, "remove_overflow":do_und_ov_flow, "remove_underflow":do_und_ov_flow, "xaxis_ndivisions":405},
            #     )

            p.dump_plot(fnames=[filename],
                dirname="plots/tracklet{}".format(suffix),
                dogrep=False,
                filter_pattern="Root__tl_{}_deltaBeta_4thCorr_maxzoom".format(tlcombo),
                extraoptions={"yaxis_log":False, "legend_smart":False, "print_yield":False, "remove_overflow":do_und_ov_flow, "remove_underflow":do_und_ov_flow, "xaxis_ndivisions":405},
                )

            p.dump_plot(fnames=[filename],
                dirname="plots/tracklet{}".format(suffix),
                dogrep=False,
                filter_pattern="Root__tl_{}_deltaBeta_4thCorr_slava".format(tlcombo),
                extraoptions={"yaxis_log":False, "legend_smart":False, "print_yield":False, "remove_overflow":do_und_ov_flow, "remove_underflow":do_und_ov_flow, "xaxis_ndivisions":405},
                )

            # p.dump_plot(fnames=[filename],
            #     dirname="plots/tracklet{}".format(suffix),
            #     dogrep=False,
            #     filter_pattern="Root__tl_matched_track_{}_deltaBeta_4thCorr_maxzoom".format(tlcombo),
            #     extraoptions={"yaxis_log":False, "legend_smart":False, "print_yield":False, "remove_overflow":do_und_ov_flow, "remove_underflow":do_und_ov_flow, "xaxis_ndivisions":405},
            #     )

            # p.dump_plot(fnames=[filename],
            #     dirname="plots/tracklet{}".format(suffix),
            #     dogrep=False,
            #     filter_pattern="Root__tl_matched_track_{}_deltaBeta_maxzoom".format(tlcombo),
            #     extraoptions={"yaxis_log":False, "legend_smart":False, "print_yield":False, "remove_overflow":do_und_ov_flow, "remove_underflow":do_und_ov_flow, "xaxis_ndivisions":405},
            #     )

            # p.dump_plot(fnames=[filename],
            #     dirname="plots/tracklet{}".format(suffix),
            #     dogrep=False,
            #     filter_pattern="Root__tl_unmatched_track_{}_deltaBeta_4thCorr_maxzoom".format(tlcombo),
            #     extraoptions={"yaxis_log":False, "legend_smart":False, "print_yield":False, "remove_overflow":do_und_ov_flow, "remove_underflow":do_und_ov_flow, "xaxis_ndivisions":405},
            #     )

            # p.dump_plot(fnames=[filename],
            #     dirname="plots/tracklet{}".format(suffix),
            #     dogrep=False,
            #     filter_pattern="Root__tl_unmatched_track_{}_deltaBeta_maxzoom".format(tlcombo),
            #     extraoptions={"yaxis_log":False, "legend_smart":False, "print_yield":False, "remove_overflow":do_und_ov_flow, "remove_underflow":do_und_ov_flow, "xaxis_ndivisions":405},
            #     )

            # p.dump_plot(fnames=[filename],
            #     dirname="plots/tracklet{}".format(suffix),
            #     dogrep=False,
            #     filter_pattern="Root__tl_{}_deltaBeta_standard".format(tlcombo),
            #     extraoptions={"yaxis_log":False, "legend_smart":False, "print_yield":False, "remove_overflow":do_und_ov_flow, "remove_underflow":do_und_ov_flow},
            #     )

            # p.dump_plot(fnames=[filename],
            #     dirname="plots/tracklet{}".format(suffix),
            #     dogrep=False,
            #     filter_pattern="Root__tl_{}_deltaBeta_midpoint_standard".format(tlcombo),
            #     # extraoptions={"yaxis_log":False, "legend_smart":False, "print_yield":False, "remove_overflow":do_und_ov_flow, "remove_underflow":do_und_ov_flow},
            #     extraoptions={"yaxis_log":False, "legend_smart":False, "print_yield":False, "remove_overflow":do_und_ov_flow, "remove_underflow":do_und_ov_flow},
            #     )

            # p.dump_plot(fnames=[filename],
            #     dirname="plots/tracklet{}".format(suffix),
            #     dogrep=False,
            #     filter_pattern="Root__tl_{}_deltaBeta_3rdCorr_standard".format(tlcombo),
            #     extraoptions={"yaxis_log":False, "legend_smart":False, "print_yield":False, "remove_overflow":do_und_ov_flow, "remove_underflow":do_und_ov_flow},
            #     )

            p.dump_plot(fnames=[filename],
                dirname="plots/tracklet{}".format(suffix),
                dogrep=False,
                filter_pattern="Root__tl_{}_deltaBeta_4thCorr_standard".format(tlcombo),
                extraoptions={"yaxis_log":False, "legend_smart":False, "print_yield":False, "remove_overflow":do_und_ov_flow, "remove_underflow":do_und_ov_flow},
                )

            # p.dump_plot(fnames=[filename],
            #     dirname="plots/tracklet{}".format(suffix),
            #     dogrep=False,
            #     filter_pattern="Root__tl_{}_deltaBeta_zoom".format(tlcombo),
            #     extraoptions={"yaxis_log":False, "legend_smart":False, "print_yield":False, "remove_overflow":do_und_ov_flow, "remove_underflow":do_und_ov_flow},
            #     )

            # p.dump_plot(fnames=[filename],
            #     dirname="plots/tracklet{}".format(suffix),
            #     dogrep=False,
            #     filter_pattern="Root__tl_{}_deltaBeta_maxzoom".format(tlcombo),
            #     extraoptions={"yaxis_log":False, "legend_smart":False, "print_yield":False, "remove_overflow":do_und_ov_flow, "remove_underflow":do_und_ov_flow, "xaxis_ndivisions":405},
            #     )

            # p.dump_plot(fnames=[filename],
            #     dirname="plots/tracklet{}".format(suffix),
            #     dogrep=False,
            #     filter_pattern="Root__tl_{}_deltaBeta_slava".format(tlcombo),
            #     extraoptions={"yaxis_log":False, "legend_smart":False, "print_yield":False, "remove_overflow":do_und_ov_flow, "remove_underflow":do_und_ov_flow},
            #     )

            # p.dump_plot(fnames=[filename],
            #     dirname="plots/tracklet_log{}".format(suffix),
            #     dogrep=False,
            #     filter_pattern="Root__tl_{}_deltaBeta".format(tlcombo),
            #     extraoptions={"yaxis_log":True, "legend_smart":False, "print_yield":False},
            #     )

            # p.dump_plot(fnames=[filename],
            #     dirname="plots/tracklet{}".format(suffix),
            #     dogrep=False,
            #     filter_pattern="Root__tl_{}_cutflow".format(tlcombo),
            #     extraoptions={"yaxis_log":False, "legend_smart":False, "print_yield":True, "remove_overflow":do_und_ov_flow, "remove_underflow":do_und_ov_flow},
            #     )

            # p.dump_plot(fnames=[filename],
            #     dirname="plots/tracklet_log{}".format(suffix),
            #     dogrep=False,
            #     filter_pattern="Root__tl_{}_cutflow".format(tlcombo),
            #     extraoptions={"yaxis_log":True, "legend_smart":False, "print_yield":True},
            #     )

            # p.dump_plot(fnames=[filename],
            #     dirname="plots/tracklet{}".format(suffix),
            #     dogrep=False,
            #     filter_pattern="Root__tl_{}_deltaBeta_postCut".format(tlcombo),
            #     extraoptions={"yaxis_log":False, "legend_smart":False, "print_yield":False, "remove_overflow":do_und_ov_flow, "remove_underflow":do_und_ov_flow, "nbins":1},
            #     )

            # p.dump_plot(fnames=[filename],
            #     dirname="plots/tracklet{}".format(suffix),
            #     dogrep=False,
            #     filter_pattern="Root__tl_truth_{}_deltaBeta".format(tlcombo),
            #     extraoptions={"yaxis_log":False, "legend_smart":False, "print_yield":False, "remove_overflow":do_und_ov_flow, "remove_underflow":do_und_ov_flow},
            #     )

            # p.dump_plot(fnames=[filename],
            #     dirname="plots/tracklet_log{}".format(suffix),
            #     dogrep=False,
            #     filter_pattern="Root__tl_truth_{}_deltaBeta".format(tlcombo),
            #     extraoptions={"yaxis_log":True, "legend_smart":False, "print_yield":False, "remove_overflow":do_und_ov_flow, "remove_underflow":do_und_ov_flow},
            #     )

            # p.dump_plot(fnames=[filename],
            #     dirname="plots/tracklet{}".format(suffix),
            #     dogrep=False,
            #     filter_pattern="Root__tl_truth_{}_deltaBeta_4thCorr_maxzoom".format(tlcombo),
            #     extraoptions={"yaxis_log":False, "legend_smart":False, "print_yield":False, "remove_overflow":do_und_ov_flow, "remove_underflow":do_und_ov_flow},
            #     )

            # p.dump_plot(fnames=[filename],
            #     dirname="plots/tracklet_log{}".format(suffix),
            #     dogrep=False,
            #     filter_pattern="Root__tl_truth_{}_deltaBeta_4thCorr_maxzoom".format(tlcombo),
            #     extraoptions={"yaxis_log":True, "legend_smart":False, "print_yield":False, "remove_overflow":do_und_ov_flow, "remove_underflow":do_und_ov_flow},
            #     )

            # p.dump_plot(fnames=[filename],
            #     dirname="plots/tracklet{}".format(suffix),
            #     dogrep=False,
            #     filter_pattern="Root__tl_truth_{}_deltaBeta_4thCorr".format(tlcombo),
            #     extraoptions={"yaxis_log":False, "legend_smart":False, "print_yield":False, "remove_overflow":do_und_ov_flow, "remove_underflow":do_und_ov_flow},
            #     )

            # p.dump_plot(fnames=[filename],
            #     dirname="plots/tracklet_log{}".format(suffix),
            #     dogrep=False,
            #     filter_pattern="Root__tl_truth_{}_deltaBeta_4thCorr".format(tlcombo),
            #     extraoptions={"yaxis_log":True, "legend_smart":False, "print_yield":False, "remove_overflow":do_und_ov_flow, "remove_underflow":do_und_ov_flow},
            #     )

            # p.dump_plot(fnames=[filename],
            #     dirname="plots/tracklet{}".format(suffix),
            #     dogrep=False,
            #     filter_pattern="Root__tl_truth_{}_cutflow".format(tlcombo),
            #     extraoptions={"yaxis_log":False, "legend_smart":False, "print_yield":True, "remove_overflow":do_und_ov_flow, "remove_underflow":do_und_ov_flow},
            #     )

            # p.dump_plot(fnames=[filename],
            #     dirname="plots/tracklet{}".format(suffix),
            #     dogrep=False,
            #     filter_pattern="Root__tl_matched_pt2_track_{}_deltaBeta_4thCorr_maxzoom".format(tlcombo),
            #     extraoptions={"yaxis_log":False, "legend_smart":False, "print_yield":True, "remove_overflow":do_und_ov_flow, "remove_underflow":do_und_ov_flow},
            #     )

            # p.dump_plot(fnames=[filename],
            #     dirname="plots/tracklet{}".format(suffix),
            #     dogrep=False,
            #     filter_pattern="Root__tl_matched_pt1peak_track_{}_deltaBeta_4thCorr_maxzoom".format(tlcombo),
            #     extraoptions={"yaxis_log":False, "legend_smart":False, "print_yield":True, "remove_overflow":do_und_ov_flow, "remove_underflow":do_und_ov_flow},
            #     )

            # p.dump_plot(fnames=[filename],
            #     dirname="plots/tracklet{}".format(suffix),
            #     dogrep=False,
            #     filter_pattern="Root__tl_matched_pt1p5peak_track_{}_deltaBeta_4thCorr_maxzoom".format(tlcombo),
            #     extraoptions={"yaxis_log":False, "legend_smart":False, "print_yield":True, "remove_overflow":do_und_ov_flow, "remove_underflow":do_und_ov_flow},
            #     )

            # p.dump_plot(fnames=[filename],
            #     dirname="plots/tracklet{}".format(suffix),
            #     dogrep=False,
            #     filter_pattern="Root__tl_matched_pt2peak_track_{}_deltaBeta_4thCorr_maxzoom".format(tlcombo),
            #     extraoptions={"yaxis_log":False, "legend_smart":False, "print_yield":True, "remove_overflow":do_und_ov_flow, "remove_underflow":do_und_ov_flow},
            #     )

            # p.dump_plot(fnames=[filename],
            #     dirname="plots/tracklet{}".format(suffix),
            #     dogrep=False,
            #     filter_pattern="Root__tl_matched_pt2p5peak_track_{}_deltaBeta_4thCorr_maxzoom".format(tlcombo),
            #     extraoptions={"yaxis_log":False, "legend_smart":False, "print_yield":True, "remove_overflow":do_und_ov_flow, "remove_underflow":do_und_ov_flow},
            #     )

            # p.dump_plot(fnames=[filename],
            #     dirname="plots/tracklet{}".format(suffix),
            #     dogrep=False,
            #     filter_pattern="Root__tl_matched_pt3peak_track_{}_deltaBeta_4thCorr_maxzoom".format(tlcombo),
            #     extraoptions={"yaxis_log":False, "legend_smart":False, "print_yield":True, "remove_overflow":do_und_ov_flow, "remove_underflow":do_und_ov_flow},
            #     )

            # p.dump_plot(fnames=[filename],
            #     dirname="plots/tracklet{}".format(suffix),
            #     dogrep=False,
            #     filter_pattern="Root__tl_matched_pos_pt1peak_track_{}_deltaBeta_4thCorr_maxzoom".format(tlcombo),
            #     extraoptions={"yaxis_log":False, "legend_smart":False, "print_yield":True, "remove_overflow":do_und_ov_flow, "remove_underflow":do_und_ov_flow},
            #     )

            # p.dump_plot(fnames=[filename],
            #     dirname="plots/tracklet{}".format(suffix),
            #     dogrep=False,
            #     filter_pattern="Root__tl_matched_pos_pt1p5peak_track_{}_deltaBeta_4thCorr_maxzoom".format(tlcombo),
            #     extraoptions={"yaxis_log":False, "legend_smart":False, "print_yield":True, "remove_overflow":do_und_ov_flow, "remove_underflow":do_und_ov_flow},
            #     )

            # p.dump_plot(fnames=[filename],
            #     dirname="plots/tracklet{}".format(suffix),
            #     dogrep=False,
            #     filter_pattern="Root__tl_matched_pos_pt2peak_track_{}_deltaBeta_4thCorr_maxzoom".format(tlcombo),
            #     extraoptions={"yaxis_log":False, "legend_smart":False, "print_yield":True, "remove_overflow":do_und_ov_flow, "remove_underflow":do_und_ov_flow},
            #     )

            # p.dump_plot(fnames=[filename],
            #     dirname="plots/tracklet{}".format(suffix),
            #     dogrep=False,
            #     filter_pattern="Root__tl_matched_pos_pt2p5peak_track_{}_deltaBeta_4thCorr_maxzoom".format(tlcombo),
            #     extraoptions={"yaxis_log":False, "legend_smart":False, "print_yield":True, "remove_overflow":do_und_ov_flow, "remove_underflow":do_und_ov_flow},
            #     )

            # p.dump_plot(fnames=[filename],
            #     dirname="plots/tracklet{}".format(suffix),
            #     dogrep=False,
            #     filter_pattern="Root__tl_matched_pos_pt3peak_track_{}_deltaBeta_4thCorr_maxzoom".format(tlcombo),
            #     extraoptions={"yaxis_log":False, "legend_smart":False, "print_yield":True, "remove_overflow":do_und_ov_flow, "remove_underflow":do_und_ov_flow},
            #     )

            # p.dump_plot(fnames=[filename],
            #     dirname="plots/tracklet{}".format(suffix),
            #     dogrep=False,
            #     filter_pattern="Root__tl_matched_neg_pt1peak_track_{}_deltaBeta_4thCorr_maxzoom".format(tlcombo),
            #     extraoptions={"yaxis_log":False, "legend_smart":False, "print_yield":True, "remove_overflow":do_und_ov_flow, "remove_underflow":do_und_ov_flow},
            #     )

            # p.dump_plot(fnames=[filename],
            #     dirname="plots/tracklet{}".format(suffix),
            #     dogrep=False,
            #     filter_pattern="Root__tl_matched_neg_pt1p5peak_track_{}_deltaBeta_4thCorr_maxzoom".format(tlcombo),
            #     extraoptions={"yaxis_log":False, "legend_smart":False, "print_yield":True, "remove_overflow":do_und_ov_flow, "remove_underflow":do_und_ov_flow},
            #     )

            # p.dump_plot(fnames=[filename],
            #     dirname="plots/tracklet{}".format(suffix),
            #     dogrep=False,
            #     filter_pattern="Root__tl_matched_neg_pt2peak_track_{}_deltaBeta_4thCorr_maxzoom".format(tlcombo),
            #     extraoptions={"yaxis_log":False, "legend_smart":False, "print_yield":True, "remove_overflow":do_und_ov_flow, "remove_underflow":do_und_ov_flow},
            #     )

            # p.dump_plot(fnames=[filename],
            #     dirname="plots/tracklet{}".format(suffix),
            #     dogrep=False,
            #     filter_pattern="Root__tl_matched_neg_pt2p5peak_track_{}_deltaBeta_4thCorr_maxzoom".format(tlcombo),
            #     extraoptions={"yaxis_log":False, "legend_smart":False, "print_yield":True, "remove_overflow":do_und_ov_flow, "remove_underflow":do_und_ov_flow},
            #     )

            # p.dump_plot(fnames=[filename],
            #     dirname="plots/tracklet{}".format(suffix),
            #     dogrep=False,
            #     filter_pattern="Root__tl_matched_neg_pt3peak_track_{}_deltaBeta_4thCorr_maxzoom".format(tlcombo),
            #     extraoptions={"yaxis_log":False, "legend_smart":False, "print_yield":True, "remove_overflow":do_und_ov_flow, "remove_underflow":do_und_ov_flow},
            #     )

            # p.dump_plot(fnames=[filename],
            #     dirname="plots/tracklet{}".format(suffix),
            #     dogrep=False,
            #     filter_pattern="Root__pt_v_tl_matched_track_{}_deltaBeta_4thCorr_maxzoom".format(tlcombo),
            #     extraoptions={"yaxis_log":False, "legend_smart":False, "print_yield":True, "remove_overflow":do_und_ov_flow, "remove_underflow":do_und_ov_flow},
            #     )

            # p.dump_plot(fnames=[filename],
            #     dirname="plots/tracklet{}".format(suffix),
            #     dogrep=False,
            #     filter_pattern="Root__pt_v_tl_matched_pos_track_{}_deltaBeta_4thCorr_maxzoom".format(tlcombo),
            #     extraoptions={"yaxis_log":False, "legend_smart":False, "print_yield":True, "remove_overflow":do_und_ov_flow, "remove_underflow":do_und_ov_flow},
            #     )

            # p.dump_plot(fnames=[filename],
            #     dirname="plots/tracklet{}".format(suffix),
            #     dogrep=False,
            #     filter_pattern="Root__pt_v_tl_matched_neg_track_{}_deltaBeta_4thCorr_maxzoom".format(tlcombo),
            #     extraoptions={"yaxis_log":False, "legend_smart":False, "print_yield":True, "remove_overflow":do_und_ov_flow, "remove_underflow":do_und_ov_flow},
            #     )

            pos_neg_tracklet_comparison(filename, tlcombo, "pt1peak", suffix, do_und_ov_flow)
            pos_neg_tracklet_comparison(filename, tlcombo, "pt1p5peak", suffix, do_und_ov_flow)
            pos_neg_tracklet_comparison(filename, tlcombo, "pt2peak", suffix, do_und_ov_flow)
            pos_neg_tracklet_comparison(filename, tlcombo, "pt2p5peak", suffix, do_und_ov_flow)
            pos_neg_tracklet_comparison(filename, tlcombo, "pt3peak", suffix, do_und_ov_flow)
            pos_neg_tracklet_comparison(filename, tlcombo, "pt10", suffix, do_und_ov_flow)
            pos_neg_tracklet_comparison(filename, tlcombo, "pt20", suffix, do_und_ov_flow)
            pos_neg_tracklet_comparison(filename, tlcombo, "pt50", suffix, do_und_ov_flow)

            pos_neg_tracklet_comparison(filename, tlcombo, "pt1peak", suffix, do_und_ov_flow, "_zoom")
            pos_neg_tracklet_comparison(filename, tlcombo, "pt1p5peak", suffix, do_und_ov_flow, "_zoom")
            pos_neg_tracklet_comparison(filename, tlcombo, "pt2peak", suffix, do_und_ov_flow, "_zoom")
            pos_neg_tracklet_comparison(filename, tlcombo, "pt2p5peak", suffix, do_und_ov_flow, "_zoom")
            pos_neg_tracklet_comparison(filename, tlcombo, "pt3peak", suffix, do_und_ov_flow, "_zoom")

            pos_neg_tracklet_comparison(filename, tlcombo, "pt1peak", suffix, do_und_ov_flow, "")
            pos_neg_tracklet_comparison(filename, tlcombo, "pt1p5peak", suffix, do_und_ov_flow, "")
            pos_neg_tracklet_comparison(filename, tlcombo, "pt2peak", suffix, do_und_ov_flow, "")
            pos_neg_tracklet_comparison(filename, tlcombo, "pt2p5peak", suffix, do_und_ov_flow, "")
            pos_neg_tracklet_comparison(filename, tlcombo, "pt3peak", suffix, do_und_ov_flow, "")

            pos_neg_tracklet_comparison(filename, tlcombo, "pt1peak", suffix, do_und_ov_flow, "_maxzoom", "")
            pos_neg_tracklet_comparison(filename, tlcombo, "pt1p5peak", suffix, do_und_ov_flow, "_maxzoom", "")
            pos_neg_tracklet_comparison(filename, tlcombo, "pt2peak", suffix, do_und_ov_flow, "_maxzoom", "")
            pos_neg_tracklet_comparison(filename, tlcombo, "pt2p5peak", suffix, do_und_ov_flow, "_maxzoom", "")
            pos_neg_tracklet_comparison(filename, tlcombo, "pt3peak", suffix, do_und_ov_flow, "_maxzoom", "")
            pos_neg_tracklet_comparison(filename, tlcombo, "pt10", suffix, do_und_ov_flow, "_maxzoom", "")
            pos_neg_tracklet_comparison(filename, tlcombo, "pt20", suffix, do_und_ov_flow, "_maxzoom", "")
            pos_neg_tracklet_comparison(filename, tlcombo, "pt50", suffix, do_und_ov_flow, "_maxzoom", "")

            pos_neg_tracklet_comparison(filename, tlcombo, "pt1peak", suffix, do_und_ov_flow, "_zoom", "")
            pos_neg_tracklet_comparison(filename, tlcombo, "pt1p5peak", suffix, do_und_ov_flow, "_zoom", "")
            pos_neg_tracklet_comparison(filename, tlcombo, "pt2peak", suffix, do_und_ov_flow, "_zoom", "")
            pos_neg_tracklet_comparison(filename, tlcombo, "pt2p5peak", suffix, do_und_ov_flow, "_zoom", "")
            pos_neg_tracklet_comparison(filename, tlcombo, "pt3peak", suffix, do_und_ov_flow, "_zoom", "")

            pos_neg_tracklet_comparison(filename, tlcombo, "pt1peak", suffix, do_und_ov_flow, "", "")
            pos_neg_tracklet_comparison(filename, tlcombo, "pt1p5peak", suffix, do_und_ov_flow, "", "")
            pos_neg_tracklet_comparison(filename, tlcombo, "pt2peak", suffix, do_und_ov_flow, "", "")
            pos_neg_tracklet_comparison(filename, tlcombo, "pt2p5peak", suffix, do_und_ov_flow, "", "")
            pos_neg_tracklet_comparison(filename, tlcombo, "pt3peak", suffix, do_und_ov_flow, "", "")


if drawTCplots:
    # tccombos = ["bbbbbb"]
    tccombos = ["bbbbbb", "bbbbbe", "bbbbee", "bbbeee", "bbeeee", "beeeee"]

    for tccombo in tccombos:
        for i in xrange(1):
            plot_eff("Root__tc_{}_matched_track_pt_by_layer{}".format(tccombo, i), "Root__tc_{}_all_track_pt_by_layer{}".format(tccombo, i), "tc_eff_{}_pt_by_layer{}.pdf".format(tccombo, i), "tceff", tag)
            plot_eff("Root__tc_{}_matched_track_pt_by_layer{}".format(tccombo, i), "Root__tc_{}_all_track_pt_by_layer{}".format(tccombo, i), "tc_eff_{}_ptzoom_by_layer{}.pdf".format(tccombo, i), "tceff", tag)
            plot_eff("Root__tc_{}_matched_track_eta_by_layer{}".format(tccombo, i), "Root__tc_{}_all_track_eta_by_layer{}".format(tccombo, i), "tc_eff_{}_eta_by_layer{}.pdf".format(tccombo, i), "tceff", tag)
            plot_eff("Root__tc_{}_matched_track_eta_by_layer{}".format(tccombo, i), "Root__tc_{}_all_track_eta_by_layer{}".format(tccombo, i), "tc_eff_{}_etazoom_by_layer{}.pdf".format(tccombo, i), "tceff", tag)
            plot_eff("Root__tc_{}_matched_track_dxy_by_layer{}".format(tccombo, i), "Root__tc_{}_all_track_dxy_by_layer{}".format(tccombo, i), "tc_eff_{}_dxy_by_layer{}.pdf".format(tccombo, i), "tceff", tag)

if drawTCSelPlots:

    tlcombos = ["all"]

    for tlcombo in tlcombos:

        p.dump_plot(fnames=[filename],
            dirname="plots/trackcandidate",
            dogrep=False,
            filter_pattern="Root__tc_{}_inner_tl_deltaBeta".format(tlcombo),
            extraoptions={"yaxis_log":False, "legend_smart":False, "print_yield":False, "remove_overflow":do_und_ov_flow, "remove_underflow":do_und_ov_flow},
            )

        p.dump_plot(fnames=[filename],
            dirname="plots/trackcandidate",
            dogrep=False,
            filter_pattern="Root__tc_{}_nocut_inner_tl_deltaBeta".format(tlcombo),
            extraoptions={"yaxis_log":False, "legend_smart":False, "print_yield":False, "remove_overflow":do_und_ov_flow, "remove_underflow":do_und_ov_flow},
            )

        p.dump_plot(fnames=[filename],
            dirname="plots/trackcandidate",
            dogrep=False,
            filter_pattern="Root__tc_{}_outer_tl_deltaBeta".format(tlcombo),
            extraoptions={"yaxis_log":False, "legend_smart":False, "print_yield":False, "remove_overflow":do_und_ov_flow, "remove_underflow":do_und_ov_flow},
            )

        p.dump_plot(fnames=[filename],
            dirname="plots/trackcandidate",
            dogrep=False,
            filter_pattern="Root__tc_{}_nocut_outer_tl_deltaBeta".format(tlcombo),
            extraoptions={"yaxis_log":False, "legend_smart":False, "print_yield":False, "remove_overflow":do_und_ov_flow, "remove_underflow":do_und_ov_flow},
            )

        p.dump_plot(fnames=[filename],
            dirname="plots/trackcandidate",
            dogrep=False,
            filter_pattern="Root__tc_{}_inner_tl_betaIn_minus_outer_tl_betaOut".format(tlcombo),
            extraoptions={"yaxis_log":False, "legend_smart":False, "print_yield":False, "remove_overflow":do_und_ov_flow, "remove_underflow":do_und_ov_flow},
            )

        p.dump_plot(fnames=[filename],
            dirname="plots/trackcandidate",
            dogrep=False,
            filter_pattern="Root__tc_{}_dr".format(tlcombo),
            extraoptions={"yaxis_log":False, "legend_smart":False, "print_yield":False, "remove_overflow":do_und_ov_flow, "remove_underflow":do_und_ov_flow},
            )

        p.dump_plot(fnames=[filename],
            dirname="plots/trackcandidate",
            dogrep=False,
            filter_pattern="Root__tc_{}_dr_v_tc_{}_r".format(tlcombo,tlcombo),
            extraoptions={"yaxis_log":False, "legend_smart":False, "print_yield":False, "remove_overflow":do_und_ov_flow, "remove_underflow":do_und_ov_flow},
            )

        p.dump_plot(fnames=[filename],
            dirname="plots/trackcandidate",
            dogrep=False,
            filter_pattern="Root__truth_tc_{}_dr_v_truth_tc_{}_r".format(tlcombo,tlcombo),
            extraoptions={"yaxis_log":False, "legend_smart":False, "print_yield":False, "remove_overflow":do_und_ov_flow, "remove_underflow":do_und_ov_flow},
            )

        p.dump_plot(fnames=[filename],
            dirname="plots/trackcandidate",
            dogrep=False,
            filter_pattern="Root__truth_gt1pt_tc_{}_dr_v_truth_gt1pt_tc_{}_r".format(tlcombo,tlcombo),
            extraoptions={"yaxis_log":False, "legend_smart":False, "print_yield":False, "remove_overflow":do_und_ov_flow, "remove_underflow":do_und_ov_flow},
            )

        p.dump_plot(fnames=[filename],
            dirname="plots/trackcandidate",
            dogrep=False,
            filter_pattern="Root__tc_{}_cutflow".format(tlcombo),
            extraoptions={"yaxis_log":False, "legend_smart":False, "print_yield":True, "remove_overflow":do_und_ov_flow, "remove_underflow":do_und_ov_flow},
            )

        p.dump_plot(fnames=[filename],
            dirname="plots/trackcandidate",
            dogrep=False,
            filter_pattern="Root__tc_{}_inner_tl_betaAv_minus_outer_tl_betaAv".format(tlcombo),
            extraoptions={"yaxis_log":False, "legend_smart":False, "print_yield":True, "remove_overflow":do_und_ov_flow, "remove_underflow":do_und_ov_flow},
            )

if drawTPSelPlots:

    tpcombos = ["bb1bb2", "bb2bb3", "bb3bb4", "bb4bb5"]

    for tpcombo in tpcombos:

        p.dump_plot(fnames=[filename],
            dirname="plots/triplet",
            dogrep=False,
            filter_pattern="Root__tp_{}_deltaBeta".format(tpcombo),
            extraoptions={"yaxis_log":False, "legend_smart":False, "print_yield":False, "remove_overflow":False, "remove_underflow":False},
            )

        p.dump_plot(fnames=[filename],
            dirname="plots/triplet",
            dogrep=False,
            filter_pattern="Root__tp_{}_deltaBeta_midpoint".format(tpcombo),
            extraoptions={"yaxis_log":False, "legend_smart":False, "print_yield":False, "remove_overflow":False, "remove_underflow":False},
            )

        p.dump_plot(fnames=[filename],
            dirname="plots/triplet",
            dogrep=False,
            filter_pattern="Root__tp_{}_deltaBeta_3rdCorr".format(tpcombo),
            extraoptions={"yaxis_log":False, "legend_smart":False, "print_yield":False, "remove_overflow":False, "remove_underflow":False},
            )

        p.dump_plot(fnames=[filename],
            dirname="plots/triplet",
            dogrep=False,
            filter_pattern="Root__tp_{}_deltaBeta_4thCorr".format(tpcombo),
            extraoptions={"yaxis_log":False, "legend_smart":False, "print_yield":False, "remove_overflow":False, "remove_underflow":False},
            )

        p.dump_plot(fnames=[filename],
            dirname="plots/triplet",
            dogrep=False,
            filter_pattern="Root__tp_{}_deltaBeta_3rdCorr_maxzoom".format(tpcombo),
            extraoptions={"yaxis_log":False, "legend_smart":False, "print_yield":False, "remove_overflow":False, "remove_underflow":False},
            )

        p.dump_plot(fnames=[filename],
            dirname="plots/triplet",
            dogrep=False,
            filter_pattern="Root__tp_{}_deltaBeta_4thCorr_maxzoom".format(tpcombo),
            extraoptions={"yaxis_log":False, "legend_smart":False, "print_yield":False, "remove_overflow":False, "remove_underflow":False},
            )

        p.dump_plot(fnames=[filename],
            dirname="plots/triplet",
            dogrep=False,
            filter_pattern="Root__tp_{}_deltaBeta_standard".format(tpcombo),
            extraoptions={"yaxis_log":False, "legend_smart":False, "print_yield":False, "remove_overflow":False, "remove_underflow":False},
            )

        p.dump_plot(fnames=[filename],
            dirname="plots/triplet",
            dogrep=False,
            filter_pattern="Root__tp_{}_deltaBeta_midpoint_standard".format(tpcombo),
            # extraoptions={"yaxis_log":False, "legend_smart":False, "print_yield":False, "remove_overflow":False, "remove_underflow":False},
            extraoptions={"yaxis_log":False, "legend_smart":False, "print_yield":False, "remove_overflow":False, "remove_underflow":False},
            )

        p.dump_plot(fnames=[filename],
            dirname="plots/triplet",
            dogrep=False,
            filter_pattern="Root__tp_{}_deltaBeta_3rdCorr_standard".format(tpcombo),
            extraoptions={"yaxis_log":False, "legend_smart":False, "print_yield":False, "remove_overflow":False, "remove_underflow":False},
            )

        p.dump_plot(fnames=[filename],
            dirname="plots/triplet",
            dogrep=False,
            filter_pattern="Root__tp_{}_deltaBeta_4thCorr_standard".format(tpcombo),
            extraoptions={"yaxis_log":False, "legend_smart":False, "print_yield":False, "remove_overflow":False, "remove_underflow":False},
            )

        p.dump_plot(fnames=[filename],
            dirname="plots/triplet",
            dogrep=False,
            filter_pattern="Root__tp_{}_deltaBeta_zoom".format(tpcombo),
            extraoptions={"yaxis_log":False, "legend_smart":False, "print_yield":False, "remove_overflow":False, "remove_underflow":False},
            )

        p.dump_plot(fnames=[filename],
            dirname="plots/triplet",
            dogrep=False,
            filter_pattern="Root__tp_{}_deltaBeta_maxzoom".format(tpcombo),
            extraoptions={"yaxis_log":False, "legend_smart":False, "print_yield":False, "remove_overflow":False, "remove_underflow":False},
            )

        p.dump_plot(fnames=[filename],
            dirname="plots/triplet",
            dogrep=False,
            filter_pattern="Root__tp_{}_deltaBeta_slava".format(tpcombo),
            extraoptions={"yaxis_log":False, "legend_smart":False, "print_yield":False, "remove_overflow":False, "remove_underflow":False},
            )

        p.dump_plot(fnames=[filename],
            dirname="plots/triplet_log",
            dogrep=False,
            filter_pattern="Root__tp_{}_deltaBeta".format(tpcombo),
            extraoptions={"yaxis_log":True, "legend_smart":False, "print_yield":False},
            )

if drawMTVplots:
   plot_eff("Root__tc_matched_track_pt_mtv", "Root__tc_all_track_pt_mtv", "tc_eff_pt_mtv.pdf", "mtveff", tag)
   plot_eff("Root__tc_matched_track_pt_mtv", "Root__tc_all_track_pt_mtv", "tc_eff_ptzoom_mtv.pdf", "mtveff", tag)
   plot_eff("Root__tc_matched_track_eta_mtv", "Root__tc_all_track_eta_mtv", "tc_eff_eta_mtv.pdf", "mtveff", tag)
   plot_eff("Root__tc_matched_track_eta_mtv", "Root__tc_all_track_eta_mtv", "tc_eff_etazoom_mtv.pdf", "mtveff", tag)
   plot_eff("Root__tc_matched_track_dxy_mtv", "Root__tc_all_track_dxy_mtv", "tc_eff_dxy_mtv.pdf", "mtveff", tag)
   plot_eff("Root__tc_matched_track_phi_mtv", "Root__tc_all_track_phi_mtv", "tc_eff_phi_mtv.pdf", "mtveff", tag)
   plot_eff("Root__tc_notmatched_trackcandidate_pt_mtv", "Root__tc_all_trackcandidate_pt_mtv", "tc_fr_pt_mtv.pdf", "mtveff", tag)
   plot_eff("Root__tc_notmatched_trackcandidate_eta_mtv", "Root__tc_all_trackcandidate_eta_mtv", "tc_fr_eta_mtv.pdf", "mtveff", tag)
   # plot_eff("Root__tc_matched_track_pt_mtv_eta0_0p4", "Root__tc_all_track_pt_mtv_eta0_0p4", "tc_eff_pt_mtv_eta0_0p4.pdf", "mtveff", tag)
   # plot_eff("Root__tc_matched_track_pt_mtv_eta0_0p4", "Root__tc_all_track_pt_mtv_eta0_0p4", "tc_eff_ptzoom_mtv_eta0_0p4.pdf", "mtveff", tag)
   # plot_eff("Root__tc_matched_track_eta_mtv_eta0_0p4", "Root__tc_all_track_eta_mtv_eta0_0p4", "tc_eff_eta_mtv_eta0_0p4.pdf", "mtveff", tag)
   # plot_eff("Root__tc_matched_track_eta_mtv_eta0_0p4", "Root__tc_all_track_eta_mtv_eta0_0p4", "tc_eff_etazoom_mtv_eta0_0p4.pdf", "mtveff", tag)
   # plot_eff("Root__tc_matched_track_dxy_mtv_eta0_0p4", "Root__tc_all_track_dxy_mtv_eta0_0p4", "tc_eff_dxy_mtv_eta0_0p4.pdf", "mtveff", tag)
   # plot_eff("Root__tc_matched_track_pt_mtv_eta0p4_0p8", "Root__tc_all_track_pt_mtv_eta0p4_0p8", "tc_eff_pt_mtv_eta0p4_0p8.pdf", "mtveff", tag)
   # plot_eff("Root__tc_matched_track_pt_mtv_eta0p4_0p8", "Root__tc_all_track_pt_mtv_eta0p4_0p8", "tc_eff_ptzoom_mtv_eta0p4_0p8.pdf", "mtveff", tag)
   # plot_eff("Root__tc_matched_track_eta_mtv_eta0p4_0p8", "Root__tc_all_track_eta_mtv_eta0p4_0p8", "tc_eff_eta_mtv_eta0p4_0p8.pdf", "mtveff", tag)
   # plot_eff("Root__tc_matched_track_eta_mtv_eta0p4_0p8", "Root__tc_all_track_eta_mtv_eta0p4_0p8", "tc_eff_etazoom_mtv_eta0p4_0p8.pdf", "mtveff", tag)
   # plot_eff("Root__tc_matched_track_dxy_mtv_eta0p4_0p8", "Root__tc_all_track_dxy_mtv_eta0p4_0p8", "tc_eff_dxy_mtv_eta0p4_0p8.pdf", "mtveff", tag)


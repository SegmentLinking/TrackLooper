import ROOT as r
import os
from . import plottery as ply

"""
0 to print out all possible options and their defaults
1 to show two simply overlaid 1D hists
2 to show overlaid 1D hists with signals, data, ratio
3 to show three TGraph ROC curves
4 to show a TH2D with smart bin labels
"""
which_tests = [0, 1, 2, 3, 4]

for which_test in which_tests:

    if which_test == 0:

        ply.Options().usage()

    if which_test == 1:
        h1 = r.TH1F("h1","nb",10,0,10)
        h2 = r.TH1F("h2","nj",10,0,10)
        h1.FillRandom("gaus",300)
        h2.FillRandom("expo",300)
        ply.plot_hist(
            bgs=[h1,h2],
            options = {
                "do_stack": False,
                "yaxis_log": True,
                "output_name": "plottery/examples/test1.pdf",
                }
            )

    if which_test == 2:

        scalefact_all = 500
        scalefact_mc = 15

        nbins = 30
        h1 = r.TH1F("h1","h1",nbins,0,5)
        h1.FillRandom("gaus",int(scalefact_mc*6*scalefact_all))
        h1.Scale(1./scalefact_mc)

        h2 = r.TH1F("h2","h2",nbins,0,5)
        h2.FillRandom("expo",int(scalefact_mc*5.2*scalefact_all))
        h2.Scale(1./scalefact_mc)

        h3 = r.TH1F("h3","h3",nbins,0,5)
        h3.FillRandom("landau",int(scalefact_mc*8*scalefact_all))
        h3.Scale(1./scalefact_mc)

        hdata = r.TH1F("hdata","hdata",nbins,0,5)
        hdata.FillRandom("gaus",int(6*scalefact_all))
        hdata.FillRandom("expo",int(5.2*scalefact_all))
        hdata.FillRandom("landau",int(8*scalefact_all))
        hdata.FillRandom("expo",int(1*scalefact_all)) # signal injection

        hsig1 = r.TH1F("hsig1","hsig1",nbins,0,5)
        hsig1.FillRandom("expo",int(scalefact_mc*1*scalefact_all))
        hsig1.Scale(1./scalefact_mc)

        hsig2 = r.TH1F("hsig2","hsig2",nbins,0,5)
        hsig2.FillRandom("gaus",int(scalefact_mc*1*scalefact_all))
        hsig2.Scale(1./scalefact_mc)

        hsyst = r.TH1F("hsyst","hsyst",nbins,0,5)
        hsyst.FillRandom("gaus",int(scalefact_all/5.*1))
        hsyst.FillRandom("expo",int(scalefact_all/5.*4))

        ply.plot_hist(
                data=hdata,
                bgs=[h1,h2,h3],
                sigs = [hsig1, hsig2],
                syst = hsyst,
                sig_labels = ["SUSY", "Magic"],
                colors = [r.kRed-2, r.kAzure+2, r.kGreen-2],
                legend_labels = ["First", "Second", "Third"],
                options = {
                    "do_stack": True,
                    "legend_scalex": 0.7,
                    "legend_scaley": 1.5,
                    "extra_text": ["#slash{E}_{T} > 50 GeV","N_{jets} #geq 2","H_{T} > 300 GeV"],
                    # "yaxis_log": True,
                    "ratio_range":[0.8,1.2],
                    # "ratio_pull": True,
                    "hist_disable_xerrors": True,
                    "ratio_chi2prob": True,
                    "output_name": "plottery/examples/test2.pdf",
                    "legend_percentageinbox": True,
                    "cms_label": "Preliminary",
                    "lumi_value": "-inf",
                    "us_flag": True,
                    # "output_jsroot": True,
                    # "output_diff_previous": True,
                    }
                )




    elif which_test == 3:

        ply.plot_graph(
                [
                    # pairs of x coord and y coord lists --> normal line
                    ([0.1,0.2,0.3,0.4,0.5,0.6,0.7,1.0], [0.1,0.5,0.9,1.0,1.0,1.0,1.0,1.0]),
                    # pairs of x coord and y coord lists --> normal line
                    ([0.2,0.3,0.4,0.5,0.6,0.7,1.0], [0.3,0.5,0.7,0.8,0.9,0.95,1.0]),
                    # quadruplet of x, y, ydown,yup --> error band
                    ([0.1,0.2,0.3,0.4,0.5,0.6,0.7,1.0], [0.1,0.2,0.3,0.45,0.6,0.7,0.8,1.0],[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]),
                    ],
                colors = [r.kRed-2, r.kGreen-2, r.kAzure+2],
                legend_labels = ["red", "green", "blue"],
                options = {
                    "legend_alignment": "bottom right",
                    "legend_scalex": 0.7,
                    "xaxis_label": "bkg. eff.",
                    "yaxis_label": "sig. eff.",
                    "yaxis_log": True,
                    "yaxis_moreloglabels": True,
                    "yaxis_noexponents": True,
                    "xaxis_range": [0.1,1.0],
                    "yaxis_range": [0.1,1.0],
                    "title": "Crappy ROC curve",
                    "output_name": "plottery/examples/test3.pdf",
                    }
                )


    elif which_test == 4:

        xyg = r.TF2("xygaus","xygaus",0,10,0,10);
        xyg.SetParameters(1,5,2,5,2)  # amplitude, meanx,sigmax,meany,sigmay
        h2 = r.TH2F("h2","h2",10,0,10, 10,0,10)
        h2.FillRandom("xygaus",10000)
        ply.plot_hist_2d(
                h2,
                options = {
                    "zaxis_log": True,
                    "bin_text_smart": True,
                    "output_name": "plottery/examples/test4.pdf",
                    "us_flag": True,
                    "zaxis_noexponents": True,
                    }
                )


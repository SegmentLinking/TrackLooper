#!/bin/env python

import plottery_wrapper as p
import glob
import sys

if len(sys.argv) > 1:
    histfiles = [sys.argv[1]]
else:
    histfiles = glob.glob("*_hist.root")

for histfile in histfiles:
    suffix = histfile.replace("_hist.root", "")
    p.dump_plot(fnames=[histfile],
            filter_pattern="Wgt__",
            dogrep=True,
            dirname="plots_{}".format(suffix),
            extraoptions={
                "nbins":60,
                "print_yield":True,
                "lumi_value":1,
                },
            )
    p.dump_plot(fnames=[histfile],
            dirname="plots_{}_log".format(suffix),
            filter_pattern="Wgt__",
            dogrep=True,
            extraoptions={
                "yaxis_log":True,
                "lumi_value":1,
                },
            )


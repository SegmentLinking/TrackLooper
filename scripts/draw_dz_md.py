#!/bin/env python

import plottery_wrapper as p
from plottery import plottery as plt
import ROOT as r

import sys

filename = "debug.root"
if len(sys.argv) > 1:
    filename = sys.argv[1]

p.dump_plot(fnames=[filename],
    dirname="plots/md",
    dogrep=True,
    filter_pattern="Root__dz_md",
    extraoptions={"yaxis_log":True, "legend_smart":False, "print_yield":True, "remove_overflow":False, "remove_underflow":False, "print_mean":True},
    )

p.dump_plot(fnames=[filename],
    dirname="plots/md",
    dogrep=True,
    filter_pattern="Root__dz_true_md",
    extraoptions={"yaxis_log":True, "legend_smart":False, "print_yield":True, "remove_overflow":False, "remove_underflow":False, "print_mean":True},
    )

p.dump_plot(fnames=[filename],
    dirname="plots/md",
    dogrep=True,
    filter_pattern="Root__n_cross",
    extraoptions={"yaxis_log":False, "legend_smart":False, "print_yield":True, "remove_overflow":False, "remove_underflow":False, "print_mean":True},
    )

p.dump_plot(fnames=[filename],
    dirname="plots/md",
    dogrep=True,
    filter_pattern="Root__n_true_cross",
    extraoptions={"yaxis_log":False, "legend_smart":False, "print_yield":True, "remove_overflow":False, "remove_underflow":False, "print_mean":True},
    )

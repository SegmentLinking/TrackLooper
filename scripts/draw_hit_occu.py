#!/bin/env python

import plottery_wrapper as p
from plottery import plottery as plt
import ROOT as r

import sys

filename = "debug.root"
if len(sys.argv) > 1:
    filename = sys.argv[1]

p.dump_plot(fnames=[filename],
    dirname="plots/hit",
    dogrep=True,
    filter_pattern="Root__nhits_",
    extraoptions={"yaxis_log":False, "legend_smart":False, "print_yield":True, "remove_overflow":False, "remove_underflow":False, "print_mean":True},
    )

p.dump_plot(fnames=[filename],
    dirname="plots/mdoccu",
    dogrep=True,
    filter_pattern="Root__n_md_",
    extraoptions={"yaxis_log":False, "legend_smart":False, "print_yield":True, "remove_overflow":False, "remove_underflow":False, "print_mean":True},
    )

p.dump_plot(fnames=[filename],
    dirname="plots/sgoccu",
    dogrep=True,
    filter_pattern="Root__n_sg_",
    extraoptions={"yaxis_log":False, "legend_smart":False, "print_yield":True, "remove_overflow":False, "remove_underflow":False, "print_mean":True},
    )

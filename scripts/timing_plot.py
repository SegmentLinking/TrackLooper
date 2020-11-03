#!/bin/env python

import ROOT as r
import plottery_wrapper as p

tl_timing_data = open("tl_timing_data.txt")
tl_times = [ float(line.strip()) for line in tl_timing_data.readlines() ]
print tl_times

tl_time_hist = r.TH1F("segment_linking_timing", "", 10, 5, 50)
for tl_time in tl_times: tl_time_hist.Fill(tl_time)
p.plot_hist(bgs=[tl_time_hist], legend_labels=["PU200 t#bar{t}"], options={"output_name":"tl_time_hist.pdf", "xaxis_label":"Seconds"})

sg_timing_data = open("sg_timing_data.txt")
sg_times = [ float(line.strip()) for line in sg_timing_data.readlines() ]
print sg_times

sg_time_hist = r.TH1F("segment_linking_timing", "", 10, 0, 1.5)
for sg_time in sg_times: sg_time_hist.Fill(sg_time)
p.plot_hist(bgs=[sg_time_hist], legend_labels=["PU200 t#bar{t}"], options={"output_name":"sg_time_hist.pdf", "xaxis_label":"Seconds"})

md_timing_data = open("md_timing_data.txt")
md_times = [ float(line.strip()) for line in md_timing_data.readlines() ]
print md_times

md_time_hist = r.TH1F("segment_linking_timing", "", 10, 0, 1.5)
for md_time in md_times: md_time_hist.Fill(md_time)
p.plot_hist(bgs=[md_time_hist], legend_labels=["PU200 t#bar{t}"], options={"output_name":"md_time_hist.pdf", "xaxis_label":"Seconds"})

tl_considered_data = open("tl_considered_data.txt")
tl_considereds = [ float(line.strip()) for line in tl_considered_data.readlines() ]
print tl_considereds

sg_considered_data = open("sg_considered_data.txt")
sg_considereds = [ float(line.strip()) for line in sg_considered_data.readlines() ]
print sg_considereds

md_considered_data = open("md_considered_data.txt")
md_considereds = [ float(line.strip()) for line in md_considered_data.readlines() ]
print md_considereds

tl_rates = [ tl_considered / tl_time / 1e6 for tl_considered, tl_time in zip(tl_considereds, tl_times) ] 
sg_rates = [ sg_considered / sg_time / 1e6 for sg_considered, sg_time in zip(sg_considereds, sg_times) ] 
md_rates = [ md_considered / md_time / 1e6 for md_considered, md_time in zip(md_considereds, md_times) ] 

print tl_rates
tl_rate_hist = r.TH1F("segment_linking_rate", "", 10, 0, 1)
for tl_rate in tl_rates: tl_rate_hist.Fill(tl_rate)
p.plot_hist(bgs=[tl_rate_hist], legend_labels=["PU200 t#bar{t}"], options={"output_name":"tl_rate_hist.pdf", "xaxis_label":"Rate [MHz]"})

sg_rate_hist = r.TH1F("sg__rate", "", 10, 0, 3)
for sg_rate in sg_rates: sg_rate_hist.Fill(sg_rate)
p.plot_hist(bgs=[sg_rate_hist], legend_labels=["PU200 t#bar{t}"], options={"output_name":"sg_rate_hist.pdf", "xaxis_label":"Rate [MHz]"})

md_rate_hist = r.TH1F("md__rate", "", 10, 0, 3)
for md_rate in md_rates: md_rate_hist.Fill(md_rate)
p.plot_hist(bgs=[md_rate_hist], legend_labels=["PU200 t#bar{t}"], options={"output_name":"md_rate_hist.pdf", "xaxis_label":"Rate [MHz]"})


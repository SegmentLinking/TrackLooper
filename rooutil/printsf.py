#!/bin/env python

import os
import sys
import datetime
import ROOT as r

def printsf(funcname, xthreshs, ythreshs, sfs, errs, filename="", xvar="eta", yvar="pt", xvarabs=False, yvarabs=False, command=""):
    """
    Function to print scale factors (or fake rate) from arrays of numbers
    """
    # parse some options and process some stuff
    yvarabsstr = ""
    xvarabsstr = ""
    if yvarabs: yvarabsstr = "fabs"
    if xvarabs: xvarabsstr = "fabs"

    # Get the boundary
    xvarmaxthresh = xthreshs[-1] - abs(xthreshs[-1]) * 0.001
    xvarminthresh = xthreshs[0] - abs(xthreshs[0]) * 0.001
    yvarmaxthresh = ythreshs[-1] - abs(ythreshs[-1]) * 0.001
    yvarminthresh = ythreshs[0] - abs(ythreshs[0]) * 0.001

    # Form the function sring
    funcstr =  "// Auto generated from https://github.com/sgnoohc/rooutil/blob/master/printsf.py\n"
    funcstr += "// Created on {}\n".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    funcstr += "// Created by the command:\n"
    if len(command):
        funcstr += "//   > {}\n".format(command)
    funcstr += "float {}(float {}_raw, float {}_raw, int isyst=0)\n".format(funcname, yvar, xvar)
    funcstr += "{\n"
    funcstr += "    if (isyst != 1 && isyst != -1 && isyst != 0)\n"
    funcstr += "        printf(\"%s\",Form(\"WARNING - in function=%s, isyst=%d is not recommended!\\n\", __FUNCTION__, isyst));\n"
    funcstr += "    float {xvar} = std::min((float) {xvarmaxthresh}, std::max((float) {xvarminthresh}, {xvar}_raw)); // minimum values are just below the first bin upper edge\n".format(xvar=xvar, xvarmaxthresh=xvarmaxthresh, xvarminthresh=xvarminthresh)
    funcstr += "    float {yvar} = std::min((float) {yvarmaxthresh}, std::max((float) {yvarminthresh}, {yvar}_raw)); // minimum values are just below the first bin upper edge\n".format(yvar=yvar, yvarmaxthresh=yvarmaxthresh, yvarminthresh=yvarminthresh)
    for i, xthresh in enumerate(xthreshs):
        for j, ythresh in enumerate(ythreshs):
            sf = sfs[i][j]
            err = errs[i][j]
            if i == len(xthreshs) - 1 and j == len(ythreshs):
                funcstr += "    return {} + isyst * {};\n".format(yvarabsstr, yvar, ythresh, xvarabsstr, xvar, xthresh, sf, err)
            elif i == len(xthreshs) -1:
                funcstr += "    if ({}({}) < {}) return {} + isyst * {};\n".format(yvarabsstr, yvar, ythresh, sf, err)
            #elif j == len(ythreshs) -1:
            #    funcstr += "    if ({}({}) < {}) return {} + isyst * {};\n".format(yvarabsstr, yvar, ythresh, sf, err)
            else:
                funcstr += "    if ({}({}) < {} && {}({}) < {}) return {} + isyst * {};\n".format(yvarabsstr, yvar, ythresh, xvarabsstr, xvar, xthresh, sf, err)
    funcstr += "    printf(\"WARNING in {}(): the given phase-space (%f, %f) did not fall under any range!\\n\", {}, {}); \n".format(funcname, yvar, xvar)
    funcstr += "    return 1;\n"
    funcstr += "}\n"

    # print or write to file
    if len(filename) != 0:
        f = open(filename, "w")
        f.write(funcstr)
    else:
        print funcstr

def printsf_th2(funcname, th2, filename="", xvar="eta", yvar="pt", xvarabs=False, yvarabs=False, command=""):
    """
    Function to print scale factors (or fake rate) from TH2
    """
    sfs = []
    errs = []
    xthreshs = []
    ythreshs = []
    for i in xrange(1, th2.GetNbinsX() + 1): xthreshs.append(th2.GetXaxis().GetBinUpEdge(i))
    for i in xrange(1, th2.GetNbinsY() + 1): ythreshs.append(th2.GetYaxis().GetBinUpEdge(i))
    for i in xrange(1, th2.GetNbinsX() + 1):
        sfs.append([])
        errs.append([])
        for j in xrange(1, th2.GetNbinsY() + 1):
            sfs[i-1].append(th2.GetBinContent(i, j))
            errs[i-1].append(th2.GetBinError(i, j))

    printsf(funcname, xthreshs, ythreshs, sfs, errs, filename, xvar, yvar, xvarabs, yvarabs, command)

def printsf_tgraph1d(funcname, tgraph1d, filename="", xvar="eta", yvar="pt", xvarabs=False, yvarabs=False):
    """
    Function to print scale factors (or fake rate) from 1D TGraph
    WARNING: this only takes one side of the error
    """
    npoints = tgraph1d.GetN()
    xs = tgraph1d.GetX()
    ys = tgraph1d.GetY()
    xerrs = tgraph1d.GetEX()
    yerrs = tgraph1d.GetEY()
    xthreshs = [1000000]
    ythreshs = []
    sfs = [[]]
    errs = [[]]
    for index, x in enumerate(xs):
        ythreshs.append(x + tgraph1d.GetErrorXhigh(index))
    for index, y in enumerate(ys):
        sfs[0].append(y)
        errs[0].append(tgraph1d.GetErrorYhigh(index))
    printsf(funcname, xthreshs, ythreshs, sfs, errs, filename, xvar, yvar, xvarabs, yvarabs)

if __name__ == "__main__":
    default_func_name = "th2d_weight"
    def help():
        print "Usage:"
        print ""
        print "    python {} FILENAME TH2NAME \"CONFIGSTRING\" [OUTPUTFUNCNAME={}] [OUTPUTFILENAME]".format(sys.argv[0], default_func_name)
        print ""
        print "    CONFIGSTRING defines the xvar and yvar name and whether they are absolute or not"
        print "    NOTE: Make sure to escape string for CONFIGSTRING"
        print ""
        print "    e.g. |xvar|:yvar"
        print ""
        sys.exit()
    if len(sys.argv) < 3:
        help()
    filename = sys.argv[1]
    histname = sys.argv[2]
    configstring = sys.argv[3]
    xvar = configstring.split(":")[0]
    yvar = configstring.split(":")[1]
    xvarabs = True if len(xvar.replace("|", "")) != len(xvar) else False
    yvarabs = True if len(yvar.replace("|", "")) != len(yvar) else False
    xvar = xvar.replace("|", "")
    yvar = yvar.replace("|", "")
    try:
        funcname = sys.argv[4]
    except:
        funcname = default_func_name
    try:
        oname = sys.argv[5]
    except:
        oname = ""
    f = r.TFile(filename)
    h = f.Get(histname)
    printsf_th2(funcname, h, oname, xvar, yvar, xvarabs, yvarabs, " ".join(sys.argv))

#eof

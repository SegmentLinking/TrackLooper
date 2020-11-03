#!/bin/env python

#  .
# ..: P.Chang, philip@physics.ucsd.edu

# ================================================================
# Wrapper script to plottery. (https://github.com/aminnj/plottery)
# ================================================================

import ROOT as r
from plottery import plottery as p
from plottery import utils as u
import math
import sys
import uuid
import os
sys.path.append("{0}/syncfiles/pyfiles".format(os.path.realpath(__file__).rsplit("/",1)[0]))
from pytable import *
import tabletex
from errors import E
import errno    
import pyrootutil as ru

# ================================================================
# New TColors
# ================================================================
mycolors = []
mycolors.append(r.TColor(11005 , 103 / 255. , 0   / 255. , 31  / 255.))
mycolors.append(r.TColor(11004 , 178 / 255. , 24  / 255. , 43  / 255.))
mycolors.append(r.TColor(11003 , 214 / 255. , 96  / 255. , 77  / 255.))
mycolors.append(r.TColor(11002 , 244 / 255. , 165 / 255. , 130 / 255.))
mycolors.append(r.TColor(11001 , 253 / 255. , 219 / 255. , 199 / 255.))
mycolors.append(r.TColor(11000 , 247 / 255. , 247 / 255. , 247 / 255.))
mycolors.append(r.TColor(11011 , 209 / 255. , 229 / 255. , 240 / 255.))
mycolors.append(r.TColor(11012 , 146 / 255. , 197 / 255. , 222 / 255.))
mycolors.append(r.TColor(11013 , 67  / 255. , 147 / 255. , 195 / 255.))
mycolors.append(r.TColor(11014 , 33  / 255. , 102 / 255. , 172 / 255.))
mycolors.append(r.TColor(11015 , 5   / 255. , 48  / 255. , 97  / 255.))

mycolors.append(r.TColor(3001 , 239 / 255. , 138 / 255. , 98  / 255.))
mycolors.append(r.TColor(3000 , 247 / 255. , 247 / 255. , 247 / 255.))
mycolors.append(r.TColor(3011 , 103 / 255. , 169 / 255. , 207 / 255.))

mycolors.append(r.TColor(5001 , 251 / 255. , 180 / 255. , 174 / 255.))
mycolors.append(r.TColor(5002 , 179 / 255. , 205 / 255. , 227 / 255.))
mycolors.append(r.TColor(5003 , 204 / 255. , 235 / 255. , 197 / 255.))
mycolors.append(r.TColor(5004 , 222 / 255. , 203 / 255. , 228 / 255.))
mycolors.append(r.TColor(5005 , 254 / 255. , 217 / 255. , 166 / 255.))

mycolors.append(r.TColor(7000 ,   0/255. ,   0/255. ,   0/255.))
mycolors.append(r.TColor(7001 , 213/255. ,  94/255. ,   0/255.)) #r
mycolors.append(r.TColor(7002 , 230/255. , 159/255. ,   0/255.)) #o
mycolors.append(r.TColor(7003 , 240/255. , 228/255. ,  66/255.)) #y
mycolors.append(r.TColor(7004 ,   0/255. , 158/255. , 115/255.)) #g
mycolors.append(r.TColor(7005 ,   0/255. , 114/255. , 178/255.)) #b
mycolors.append(r.TColor(7006 ,  86/255. , 180/255. , 233/255.)) #k
mycolors.append(r.TColor(7007 , 204/255. , 121/255. , 167/255.)) #p
mycolors.append(r.TColor(7011 , 110/255. ,  54/255. ,   0/255.)) #alt r
mycolors.append(r.TColor(7012 , 161/255. , 117/255. ,   0/255.)) #alt o
mycolors.append(r.TColor(7013 , 163/255. , 155/255. ,  47/255.)) #alt y
mycolors.append(r.TColor(7014 ,   0/255. , 102/255. ,  79/255.)) #alt g
mycolors.append(r.TColor(7015 ,   0/255. ,  93/255. , 135/255.)) #alt b
mycolors.append(r.TColor(7016 , 153/255. , 153/255. , 153/255.)) #alt k
mycolors.append(r.TColor(7017 , 140/255. ,  93/255. , 119/255.)) #alt p

mycolors.append(r.TColor(9001 ,  60/255. , 186/255. ,  84/255.))
mycolors.append(r.TColor(9002 , 244/255. , 194/255. ,  13/255.))
mycolors.append(r.TColor(9003 , 219/255. ,  50/255. ,  54/255.))
mycolors.append(r.TColor(9004 ,  72/255. , 133/255. , 237/255.))

# Color schemes from Hannsjoerg for WWW analysis
mycolors.append(r.TColor(2001 , 91  / 255. , 187 / 255. , 241 / 255.)) #light-blue
mycolors.append(r.TColor(2002 , 60  / 255. , 144 / 255. , 196 / 255.)) #blue
mycolors.append(r.TColor(2003 , 230 / 255. , 159 / 255. , 0   / 255.)) #orange
mycolors.append(r.TColor(2004 , 180 / 255. , 117 / 255. , 0   / 255.)) #brown
mycolors.append(r.TColor(2005 , 245 / 255. , 236 / 255. , 69  / 255.)) #yellow
mycolors.append(r.TColor(2006 , 215 / 255. , 200 / 255. , 0   / 255.)) #dark yellow
mycolors.append(r.TColor(2007 , 70  / 255. , 109 / 255. , 171 / 255.)) #blue-violet
mycolors.append(r.TColor(2008 , 70  / 255. , 90  / 255. , 134 / 255.)) #violet
mycolors.append(r.TColor(2009 , 55  / 255. , 65  / 255. , 100 / 255.)) #dark violet
mycolors.append(r.TColor(2010 , 120 / 255. , 160 / 255. , 0   / 255.)) #light green
mycolors.append(r.TColor(2011 , 0   / 255. , 158 / 255. , 115 / 255.)) #green
mycolors.append(r.TColor(2012 , 204 / 255. , 121 / 255. , 167 / 255.)) #pink?

mycolors.append(r.TColor(4001 , 49  / 255. , 76  / 255. , 26  / 255. ))
mycolors.append(r.TColor(4002 , 33  / 255. , 164 / 255. , 105  / 255. ))
mycolors.append(r.TColor(4003 , 176 / 255. , 224 / 255. , 160 / 255. ))
mycolors.append(r.TColor(4004 , 210 / 255. , 245 / 255. , 200 / 255. ))
mycolors.append(r.TColor(4005 , 232 / 255. , 249 / 255. , 223 / 255. ))
mycolors.append(r.TColor(4006 , 253 / 255. , 156 / 255. , 207 / 255. ))
mycolors.append(r.TColor(4007 , 121 / 255. , 204 / 255. , 158 / 255. ))
mycolors.append(r.TColor(4008 , 158 / 255. ,   0 / 255. ,  42 / 255. ))
mycolors.append(r.TColor(4009 , 176 / 255. ,   0 / 255. , 195 / 255. ))
mycolors.append(r.TColor(4010 ,  20 / 255. , 195 / 255. ,   0 / 255. ))
mycolors.append(r.TColor(4011 , 145 / 255. ,   2 / 255. , 206 / 255. ))
mycolors.append(r.TColor(4012 , 255 / 255. ,   0 / 255. , 255 / 255. ))
mycolors.append(r.TColor(4013 , 243 / 255. ,  85 / 255. ,   0 / 255. ))
mycolors.append(r.TColor(4014 , 157 / 255. , 243 / 255. , 130 / 255. ))
mycolors.append(r.TColor(4015 , 235 / 255. , 117 / 255. , 249 / 255. ))
mycolors.append(r.TColor(4016 ,  90 / 255. , 211 / 255. , 221 / 255. ))
mycolors.append(r.TColor(4017 ,  85 / 255. , 181 / 255. ,  92 / 255. ))
mycolors.append(r.TColor(4018 , 172 / 255. ,  50 / 255. ,  60 / 255. ))
mycolors.append(r.TColor(4019 ,  42 / 255. , 111 / 255. , 130 / 255. ))

mycolors.append(r.TColor(4020 , 240 / 255. , 155 / 255. , 205 / 255. )) # ATLAS pink
mycolors.append(r.TColor(4021 ,  77 / 255. , 161 / 255. ,  60 / 255. )) # ATLAS green
mycolors.append(r.TColor(4022 ,  87 / 255. , 161 / 255. , 247 / 255. )) # ATLAS blue
mycolors.append(r.TColor(4023 , 196 / 255. , 139 / 255. , 253 / 255. )) # ATLAS darkpink
mycolors.append(r.TColor(4024 , 205 / 255. , 240 / 255. , 155 / 255. )) # Complementary

mycolors.append(r.TColor(4101 , 102 / 255. , 102 / 255. , 204 / 255. )) # ATLAS HWW / WW
mycolors.append(r.TColor(4102 ,  89 / 255. , 185 / 255. ,  26 / 255. )) # ATLAS HWW / DY
mycolors.append(r.TColor(4103 , 225 / 255. ,  91 / 255. , 226 / 255. )) # ATLAS HWW / VV
mycolors.append(r.TColor(4104 , 103 / 255. , 236 / 255. , 235 / 255. )) # ATLAS HWW / misid

mycolors.append(r.TColor(4201 ,  16 / 255. , 220 / 255. , 138 / 255. )) # Signal complementary

mycolors.append(r.TColor(4305 ,   0/255. , 208/255. , 145/255.)) # green made up


default_colors = []
default_colors.append(2005)
default_colors.append(2001)
default_colors.append(2003)
default_colors.append(2007)
default_colors.append(920)
default_colors.extend(range(2001, 2013))
default_colors.extend(range(7001, 7018))




#______________________________________________________________________________________________________________________
def makedir(dirpath):
    try:
        os.makedirs(dirpath)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(dirpath):
            pass
        else:
            raise


# ===============
# Histogram utils
# ===============

#______________________________________________________________________________________________________________________
def cloneTH1(obj, name=""):
    """
    Clone any TH1 object with the same name or new name.
    """
    if name == "":
        name = obj.GetName()
    rtn = obj.Clone(name)
    rtn.SetTitle("")
    if not rtn.GetSumw2N():
        rtn.Sumw2()
    rtn.SetDirectory(0)
    # https://root-forum.cern.ch/t/setbinlabel-causes-unexpected-behavior-when-handling-the-histograms/26202/2
    labels = rtn.GetXaxis().GetLabels()
    if labels:
        rtn.GetXaxis().SetRange(1, rtn.GetXaxis().GetNbins())
    return rtn;

#______________________________________________________________________________________________________________________
def get_total_hist(hists):
    """
    Sum all histograms and return a new copy of total bkg hist.
    """
    if len(hists) == 0:
        print "ERROR - the number of histograms are zero, while you asked me to sum them up."
    totalhist = cloneTH1(hists[0])
    totalhist.Reset()
    for hist in hists:
        totalhist.Add(hist)
    return totalhist

#______________________________________________________________________________________________________________________
def get_total_err_hist(hists):
    """
    Sum all histograms errors
    """
    if len(hists) == 0:
        print "ERROR - the number of histograms are zero, while you asked me to sum them up."
    totalhist = get_total_hist(hists)
    errhist = cloneTH1(totalhist)
    errhist.Reset()
    for i in xrange(0, totalhist.GetNbinsX() + 2):
        errhist.SetBinContent(i, totalhist.GetBinError(i))
    return errhist

#______________________________________________________________________________________________________________________
def normalize_by_first_bin(hists):
    def func(hist):
        norm = hist.GetBinContent(1)
        if norm != 0:
            hist.Scale(1./norm)
    if isinstance(hists, list):
        for hist in hists:
            func(hist)
    else:
        func(hists)
    return hists

#______________________________________________________________________________________________________________________
def add_diff_to_error(nomhist, errhist, errhistpairvar=None):
    """
    Add the difference between nomhist to errhist as an additional error to nomhist
    """
    if nomhist.GetNbinsX() != errhist.GetNbinsX(): print "ERROR - the nom hist and err hist have different dimension in X"
    if nomhist.GetNbinsY() != errhist.GetNbinsY(): print "ERROR - the nom hist and err hist have different dimension in Y"
    if nomhist.GetNbinsZ() != errhist.GetNbinsZ(): print "ERROR - the nom hist and err hist have different dimension in Z"

    if errhistpairvar:
        if nomhist.GetNbinsX() != errhistpairvar.GetNbinsX(): print "ERROR - the nom hist and err hist paired variation have different dimension in X"
        if nomhist.GetNbinsY() != errhistpairvar.GetNbinsY(): print "ERROR - the nom hist and err hist paired variation have different dimension in Y"
        if nomhist.GetNbinsZ() != errhistpairvar.GetNbinsZ(): print "ERROR - the nom hist and err hist paired variation have different dimension in Z"

    labels = nomhist.GetXaxis().GetLabels()
    if labels:
        nomhist.GetXaxis().SetRange(1, nomhist.GetXaxis().GetNbins())
        nomhist.GetYaxis().SetRange(1, nomhist.GetYaxis().GetNbins())
        nomhist.GetZaxis().SetRange(1, nomhist.GetZaxis().GetNbins())
        nomhist.GetXaxis().SetCanExtend(False)
        nomhist.GetYaxis().SetCanExtend(False)
        nomhist.GetZaxis().SetCanExtend(False)
        errhist.GetXaxis().SetRange(1, errhist.GetXaxis().GetNbins())
        errhist.GetYaxis().SetRange(1, errhist.GetYaxis().GetNbins())
        errhist.GetZaxis().SetRange(1, errhist.GetZaxis().GetNbins())
        errhist.GetXaxis().SetCanExtend(False)
        errhist.GetYaxis().SetCanExtend(False)
        errhist.GetZaxis().SetCanExtend(False)

    for iz in xrange(0, nomhist.GetNbinsZ()+2):
        for iy in xrange(0, nomhist.GetNbinsY()+2):
            for ix in xrange(0, nomhist.GetNbinsX()+2):
                nombc = nomhist.GetBinContent(ix, iy, iz)
                nombe = nomhist.GetBinError(ix, iy, iz)
                errbc = errhist.GetBinContent(ix, iy, iz)
                diff = nombc - errbc
                if errhistpairvar:
                    errbcpaired = errhistpairvar.GetBinContent(ix, iy, iz)
                    diffpaired = nombc - errbcpaired
                    if abs(diff) < abs(diffpaired):
                        diff = diffpaired
                newb = E(0, diff) + E(nombc, nombe)
                #print newb.val, newb.err, diff, nombe, nombc
                nomhist.SetBinContent(ix, iy, iz, newb.val)
                nomhist.SetBinError(ix, iy, iz, newb.err)
            if nomhist.GetDimension() == 1:
                return
        if nomhist.GetDimension() == 2:
            return

#______________________________________________________________________________________________________________________
def getYaxisRange(hist):
    maximum = 0
    if hist:
        for ibin in xrange(0, hist.GetNbinsX()+2):
        #for ibin in xrange(1, hist.GetNbinsX()+1):
            c = hist.GetBinContent(ibin)
            e = hist.GetBinError(ibin)
            v = c + e
            if v > maximum:
                maximum = v
    return maximum

#______________________________________________________________________________________________________________________
def getYaxisNonZeroMin(hist):
    minimum = 999999999999999999
    if hist:
        for ibin in xrange(1, hist.GetNbinsX()+1):
        #for ibin in xrange(1, hist.GetNbinsX()+1):
            c = hist.GetBinContent(ibin)
            e = hist.GetBinError(ibin)
            v = c + e
            if float(v) != float(0):
                if abs(v) < minimum:
                    minimum = abs(v)
    if minimum == 999999999999999999:
        minimum = 0.1
    return minimum

#______________________________________________________________________________________________________________________
def get_nonzeromin_yaxis_range(hists):
    minimum = 9999999999999999999
    for hist in hists:
        v = getYaxisNonZeroMin(hist)
        if v < minimum:
            minimum = v
    if minimum == 9999999999999999999:
        minimum = 0.1
    return minimum

#______________________________________________________________________________________________________________________
def get_max_yaxis_range(hists):
    maximum = 0
    for hist in hists:
        v = getYaxisRange(hist)
        if v > maximum:
            maximum = v
    return maximum

#______________________________________________________________________________________________________________________
def get_max_yaxis_range_order_half_modded(maximum):
    firstdigit = int(str(maximum)[0])
    maximum = max(maximum, 0.001)
    order = int(math.log10(maximum))
    if firstdigit <= 2:
        middle = (10.**(order - 1))
    else:
        middle = (10.**(order))
    return maximum + middle

#______________________________________________________________________________________________________________________
def remove_errors(hists):
    for hist in hists:
        for ibin in xrange(0, hist.GetNbinsX()+2):
            hist.SetBinError(ibin, 0)

#______________________________________________________________________________________________________________________
def rebin(hists, nbin):
    for hist in hists:
        if not hist: continue
        currnbin = hist.GetNbinsX()
        fac = currnbin / nbin
        if float(fac).is_integer() and fac > 0:
            hist.Rebin(fac)

#______________________________________________________________________________________________________________________
def single_divide_by_bin_width(hist):
    for ibin in xrange(1,hist.GetNbinsX()+2):
        hist.SetBinContent(ibin, hist.GetBinContent(ibin) / hist.GetBinWidth(ibin))
        hist.SetBinError(ibin, hist.GetBinError(ibin) / hist.GetBinWidth(ibin))

#______________________________________________________________________________________________________________________
def divide_by_bin_width(hists):
    for hist in hists:
        single_divide_by_bin_width(hist)

#______________________________________________________________________________________________________________________
def flatten_th2(th2):
    nx = th2.GetNbinsX()
    ny = th2.GetNbinsY()
    th1 = r.TH1F(th2.GetName(), th2.GetTitle(), nx*ny, 0, nx*ny)
    for ix in xrange(nx):
        for iy in xrange(ny):
            bc = th2.GetBinContent(ix+1, iy+1)
            be = th2.GetBinError(ix+1, iy+1)
            #th1.SetBinContent(ix+1+(iy)*nx, bc)
            #th1.SetBinError(ix+1+(iy)*nx, be)
            th1.SetBinContent(iy+1+(ix)*ny, bc)
            th1.SetBinError(iy+1+(ix)*ny, be)
    return th1

#______________________________________________________________________________________________________________________
def remove_underflow(hists):
    def func(hist):
        hist.SetBinContent(0, 0)
        hist.SetBinError(0, 0)
    if isinstance(hists, list):
        for hist in hists:
            func(hist)
    else:
        func(hists)
    return hists

#______________________________________________________________________________________________________________________
def remove_overflow(hists):
    def func(hist):
        hist.SetBinContent(hist.GetNbinsX()+1, 0)
        hist.SetBinError(hist.GetNbinsX()+1, 0)
    if isinstance(hists, list):
        for hist in hists:
            func(hist)
    else:
        func(hists)
    return hists

#______________________________________________________________________________________________________________________
def move_overflow(hists):
    def func(hist):
        of_bc = hist.GetBinContent(hist.GetNbinsX()+1)
        of_be = hist.GetBinError(hist.GetNbinsX()+1)
        lb_bc = hist.GetBinContent(hist.GetNbinsX())
        lb_be = hist.GetBinError(hist.GetNbinsX())
        lb_bc_new = lb_bc + of_bc
        lb_be_new = math.sqrt(lb_be**2 + of_be**2)
        hist.SetBinContent(hist.GetNbinsX(), lb_bc_new)
        hist.SetBinError(hist.GetNbinsX(), lb_be_new)
        hist.SetBinContent(hist.GetNbinsX()+1, 0)
        hist.SetBinError(hist.GetNbinsX()+1, 0)
    if isinstance(hists, list):
        for hist in hists:
            func(hist)
    else:
        func(hists)
    return hists

#______________________________________________________________________________________________________________________
def apply_nf(hists, nfs):
    def func(hist, nfs):
        if isinstance(nfs, list) and len(nfs) == 0:
            pass
        elif isinstance(nfs, float):
            for i in xrange(0, hist.GetNbinsX()+2):
                bc = hist.GetBinContent(i)
                be = hist.GetBinError(i)
                hist.SetBinContent(i, bc * nfs)
                hist.SetBinError(i, be * nfs)
        elif len(nfs) == hist.GetNbinsX():
            for i in xrange(1, hist.GetNbinsX()+1):
                bc = hist.GetBinContent(i)
                be = hist.GetBinError(i)
                hist.SetBinContent(i, bc * nfs[i-1])
                hist.SetBinError(i, be * nfs[i-1])
        elif len(nfs) == hist.GetNbinsX()+2:
            for i in xrange(0, hist.GetNbinsX()+2):
                bc = hist.GetBinContent(i)
                be = hist.GetBinError(i)
                hist.SetBinContent(i, bc * nfs[i])
                hist.SetBinError(i, be * nfs[i])
        elif len(nfs) == 1:
            for i in xrange(0, hist.GetNbinsX()+2):
                bc = hist.GetBinContent(i)
                be = hist.GetBinError(i)
                hist.SetBinContent(i, bc * nfs[0][0])
                hist.SetBinError(i, be * nfs[0][0])
    if isinstance(hists, list):
        for hist in hists:
            func(hist, nfs)
    else:
        func(hists, nfs)
    return hists

#______________________________________________________________________________________________________________________
def apply_nf_w_error(hists, nfs):
    def func(hist, nfs):
        if isinstance(nfs, list) and len(nfs) == 0:
            pass
        elif isinstance(nfs, float):
            for i in xrange(0, hist.GetNbinsX()+2):
                bc = hist.GetBinContent(i)
                be = hist.GetBinError(i)
                hist.SetBinContent(i, bc * nfs)
                hist.SetBinError(i, be * nfs)
        elif len(nfs) == hist.GetNbinsX():
            for i in xrange(1, hist.GetNbinsX()+1):
                bc = hist.GetBinContent(i)
                be = hist.GetBinError(i)
                bfe = be / bc if bc != 0 else 0
                nf = nfs[i-1][0]
                ne = nfs[i-1][1]
                nfe = ne / nf if nf != 0 else 0
                nbc = bc * nf
                nbe = math.sqrt(bfe**2 + nfe**2) * nbc
                hist.SetBinContent(i, nbc)
                hist.SetBinError(i, nbe)
        elif len(nfs) == hist.GetNbinsX()+2:
            for i in xrange(0, hist.GetNbinsX()+2):
                bc = hist.GetBinContent(i)
                be = hist.GetBinError(i)
                bfe = be / bc if bc != 0 else 0
                nf = nfs[i][0]
                ne = nfs[i][1]
                nfe = ne / nf if nf != 0 else 0
                nbc = bc * nf
                nbe = math.sqrt(bfe**2 + nfe**2) * nbc
                hist.SetBinContent(i, nbc)
                hist.SetBinError(i, nbe)
        elif len(nfs) == 1:
            for i in xrange(0, hist.GetNbinsX()+2):
                bc = hist.GetBinContent(i)
                be = hist.GetBinError(i)
                bfe = be / bc if bc != 0 else 0
                nf = nfs[0][0]
                ne = nfs[0][1]
                nfe = ne / nf if nf != 0 else 0
                nbc = bc * nf
                nbe = math.sqrt(bfe**2 + nfe**2) * nbc
                hist.SetBinContent(i, nbc)
                hist.SetBinError(i, nbe)
    if isinstance(hists, list):
        for hist in hists:
            func(hist, nfs)
    else:
        func(hists, nfs)
    return hists

#______________________________________________________________________________________________________________________
def apply_nf_2d(hists, nfs):
    def func(hist, nfs):
        if isinstance(nfs, list) and len(nfs) == 0:
            pass
        elif len(nfs) == 1:
            #hist.Scale(nfs[0][0])
            for i in xrange(0, hist.GetNbinsX()+2):
                for j in xrange(0, hist.GetNbinsY()+2):
                    bc = hist.GetBinContent(i, j)
                    be = hist.GetBinError(i, j)
                    nf = nfs[0][0]
                    hist.SetBinContent(i, j, bc * nf)
                    hist.SetBinError(i, j, be * nf)
        elif len(nfs) == hist.GetNbinsX():
            for i in xrange(1, hist.GetNbinsX()+1):
                for j in xrange(0, hist.GetNbinsY()+2):
                    bc = hist.GetBinContent(i, j)
                    be = hist.GetBinError(i, j)
                    nf = nfs[i-1][0]
                    hist.SetBinContent(i, j, bc * nf)
                    hist.SetBinError(i, j, be * nf)
        else:
            print "WARNING - apply_nf_w_error_2d: something went wrong."
    if isinstance(hists, list):
        for hist in hists:
            func(hist, nfs)
    else:
        func(hists, nfs)
    return hists

#______________________________________________________________________________________________________________________
def apply_nf_w_error_2d(hists, nfs):
    def func(hist, nfs):
        if isinstance(nfs, list) and len(nfs) == 0:
            pass
        elif len(nfs) == 1:
            for i in xrange(0, hist.GetNbinsX()+2):
                for j in xrange(0, hist.GetNbinsY()+2):
                    bc = hist.GetBinContent(i, j)
                    be = hist.GetBinError(i, j)
                    bfe = be / bc if bc != 0 else 0
                    nf = nfs[0][0]
                    ne = nfs[0][1]
                    nfe = ne / nf if nf != 0 else 0
                    nbc = bc * nf
                    nbe = math.sqrt(bfe**2 + nfe**2) * nbc
                    hist.SetBinContent(i, j, nbc)
                    hist.SetBinError(i, j, nbe)
        elif len(nfs) == hist.GetNbinsX():
            for i in xrange(1, hist.GetNbinsX()+1):
                for j in xrange(0, hist.GetNbinsY()+2):
                    bc = hist.GetBinContent(i, j)
                    be = hist.GetBinError(i, j)
                    bfe = be / bc if bc != 0 else 0
                    nf = nfs[i-1][0]
                    ne = nfs[i-1][1]
                    nfe = ne / nf if nf != 0 else 0
                    nbc = bc * nf
                    nbe = math.sqrt(bfe**2 + nfe**2) * nbc
                    hist.SetBinContent(i, j, nbc)
                    hist.SetBinError(i, j, nbe)
        else:
            print "WARNING - apply_nf_w_error_2d: something went wrong."
    if isinstance(hists, list):
        for hist in hists:
            func(hist, nfs)
    else:
        func(hists, nfs)
    return hists


# =================
# Significance scan
# =================

#______________________________________________________________________________________________________________________
# S / sqrt(B) fom
def fom_SoverB(s, serr, b, berr, totals, totalb):
    if b > 0:
        return s / b, 0
    else:
        return 0, 0

#______________________________________________________________________________________________________________________
# S / sqrt(B) fom
def fom_SoverSqrtB(s, serr, b, berr, totals, totalb):
    if b > 0 and s > 0:
        # return s / math.sqrt(b), 0
        return math.sqrt(2 * ((s + b) * math.log(1 + s / b) - s)), 0
    else:
        return 0, 0

#______________________________________________________________________________________________________________________
# S / sqrt(B +sB^2) fom
def fom_SoverSqrtBwErr(s, serr, b, berr, totals, totalb):
    if b > 0:
        return r.RooStats.NumberCountingUtils.BinomialExpZ(s, b, float(berr / b)), 0
        #return s / math.sqrt(b + berr*berr), 0
    else:
        return 0, 0

#______________________________________________________________________________________________________________________
# S / sqrt(B) fom
def fom_acceptance(s, serr, b, berr, totals, totalb):
    if totals != 0:
        return s / totals, 0
    else:
        return 0, 0

#______________________________________________________________________________________________________________________
# For each signal and total background return scan from left/right of fom (figure of merit) func.
#def plot_sigscan2d(sig, bkg, fom=fom_SoverSqrtB):
def plot_sigscan2d(sig, bkg, fom=fom_SoverB):
    nbin = sig.GetNbinsX()
    if nbin != bkg.GetNbinsX():
        print "Error - significance scan for the signal and background histograms have different size", nbin, bkg.GetNbinsX()
    scan = cloneTH1(sig)
    scan.Reset()
    xmin = scan.GetXaxis().GetBinLowEdge(1)
    xwidth = scan.GetXaxis().GetBinWidth(1)
    max_f = 0
    max_f_cut_low = 0
    max_f_cut_high = 0
    totalsig = sig.Integral(0, nbin + 1)
    totalbkg = bkg.Integral(0, nbin + 1)

    for i in xrange(1, nbin + 1):
        local_max_f = 0
        local_max_f_err = 0
        for j in xrange(i + 1, nbin + 1):
            sigerr = r.Double(0)
            sigint = sig.IntegralAndError(i, j, sigerr)
            bkgerr = r.Double(0)
            bkgint = bkg.IntegralAndError(i, j, bkgerr)
            f, ferr = fom(sigint, sigerr, bkgint, bkgerr, totalsig, totalbkg)
            if max_f < f:
                max_f = f
                max_f_cut_low = xmin + xwidth * (i - 1)
                max_f_cut_high = xmin + xwidth * j
            if local_max_f < f:
                local_max_f = f
                local_max_f_err = ferr
        scan.SetBinContent(i, local_max_f)
        scan.SetBinError(i, ferr)
    scan.SetName("{:.2f} ({:.2f},{:.2f})".format(max_f, max_f_cut_low, max_f_cut_high))
    return scan

#______________________________________________________________________________________________________________________
# For each signal and total background return scan from left/right of fom (figure of merit) func.
def plot_sigscan(sig, bkg, fom=fom_SoverSqrtB):
#def plot_sigscan(sig, bkg, fom=fom_SoverB):
    nbin = sig.GetNbinsX()
    if nbin != bkg.GetNbinsX():
        print "Error - significance scan for the signal and background histograms have different size", nbin, bkg.GetNbinsX()
    leftscan = cloneTH1(sig)
    leftscan.Reset()
    xmin = leftscan.GetXaxis().GetBinLowEdge(1)
    xwidth = leftscan.GetXaxis().GetBinWidth(1)
    max_f = 0
    max_f_cut = 0
    totalsig = sig.Integral(0, nbin + 1)
    totalbkg = bkg.Integral(0, nbin + 1)
    print totalsig, totalbkg
    for i in xrange(1, nbin + 1):
        sigerr = r.Double(0)
        sigint = sig.IntegralAndError(i, nbin + 1, sigerr)
        bkgerr = r.Double(0)
        bkgint = bkg.IntegralAndError(i, nbin + 1, bkgerr)
        f, ferr = fom(sigint, sigerr, bkgint, bkgerr, totalsig, totalbkg)
        leftscan.SetBinContent(i, f)
        leftscan.SetBinError(i, ferr)
        if max_f < f:
            max_f = f
            max_f_cut = xmin + xwidth * (i - 1)
    # print max_f
    leftscan.SetName("#rightarrow {:.2f} ({:.2f})".format(max_f, max_f_cut))
    rightscan = cloneTH1(sig)
    rightscan.Reset()
    max_f = 0
    max_f_cut = 0
    for i in reversed(xrange(1, nbin + 1)):
        sigerr = r.Double(0)
        sigint = sig.IntegralAndError(0, i, sigerr)
        bkgerr = r.Double(0)
        bkgint = bkg.IntegralAndError(0, i, bkgerr)
        f, ferr = fom(sigint, sigerr, bkgint, bkgerr, totalsig, totalbkg)
        rightscan.SetBinContent(i, f)
        rightscan.SetBinError(i, ferr)
        if max_f < f:
            max_f = f
            max_f_cut = xmin + xwidth * i
    rightscan.SetName("#leftarrow {:.2f} ({:.2f})".format(max_f, max_f_cut))
    return leftscan, rightscan

#______________________________________________________________________________________________________________________
# For each signal and indvidiual background plus systematics
def plot_sigscan_w_syst(sig, bkgs, systs, fom=fom_SoverSqrtBwErr):

    bkg = get_total_hist(bkgs)

    if len(bkgs) != len(systs) and len(systs) > 0:
        print "Error - The provided systs list does not have the same number of entries as the bkgs", bkgs, systs

    nbin = sig.GetNbinsX()
    if nbin != bkg.GetNbinsX():
        print "Error - significance scan for the signal and background histograms have different size", nbin, bkg.GetNbinsX()
    leftscan = cloneTH1(sig)
    leftscan.Reset()
    xmin = leftscan.GetXaxis().GetBinLowEdge(1)
    xwidth = leftscan.GetXaxis().GetBinWidth(1)
    max_f = -999
    max_f_cut = 0
    totalsig = sig.Integral(0, nbin + 1)
    totalbkg = bkg.Integral(0, nbin + 1)
    sigaccept = 0
    for i in xrange(1, nbin + 1):
        sigerr = r.Double(0)
        sigint = sig.IntegralAndError(i, nbin + 1, sigerr)
        bkgerr = r.Double(0)
        bkgint = bkg.IntegralAndError(i, nbin + 1, bkgerr)
        count_s = E(sigint, sigerr)
        count_b = E(bkgint, bkgerr)
        counts = []
        for index, bg in enumerate(bkgs):
            e = r.Double(0)
            c = bg.IntegralAndError(i, nbin + 1, e)
            ne = math.sqrt(e*e + c*systs[index]*c*systs[index])
            counts.append(E(c, ne))
        count_b_w_syst = E(0, 0)
        for count in counts:
            count_b_w_syst = count_b_w_syst + count
        bkgerr = count_b_w_syst.err
        f, ferr = fom(sigint, sigerr, bkgint, bkgerr, totalsig, totalbkg)
        #print i, f
        leftscan.SetBinContent(i, f)
        leftscan.SetBinError(i, ferr)
        if max_f < f:
            max_f = f
            max_f_cut = xmin + xwidth * (i - 1)
            sigaccept = sigint / totalsig
    leftscan.SetName("#rightarrow {:.4f} ({:.4f} {:.4f})".format(max_f, max_f_cut, sigaccept))

    rightscan = cloneTH1(sig)
    rightscan.Reset()
    max_f = -999
    max_f_cut = 0
    for i in reversed(xrange(1, nbin + 1)):
        sigerr = r.Double(0)
        sigint = sig.IntegralAndError(0, i, sigerr)
        bkgerr = r.Double(0)
        bkgint = bkg.IntegralAndError(0, i, bkgerr)
        count_s = E(sigint, sigerr)
        count_b = E(bkgint, bkgerr)
        counts = []
        for index, bg in enumerate(bkgs):
            e = r.Double(0)
            c = bg.IntegralAndError(0, i, e)
            ne = math.sqrt(e*e + c*systs[index]*c*systs[index])
            counts.append(E(c, ne))
        count_b_w_syst = E(0, 0)
        for count in counts:
            count_b_w_syst = count_b_w_syst + count
        f, ferr = fom(sigint, sigerr, bkgint, bkgerr, totalsig, totalbkg)
        rightscan.SetBinContent(i, f)
        rightscan.SetBinError(i, ferr)
        if max_f < f:
            max_f = f
            max_f_cut = xmin + xwidth * (i - 1)
    rightscan.SetName("#leftarrow {:.4f} ({:.4f})".format(max_f, max_f_cut))

    return leftscan, rightscan

# ====================
# Yield table printing
# ====================

#______________________________________________________________________________________________________________________
def yield_str(hist, i, prec=3, noerror=False):
    if noerror:
        return "{{:.{}f}}".format(prec).format(hist.GetBinContent(i))
    else:
        e = E(hist.GetBinContent(i), hist.GetBinError(i))
        return e.round(prec)
#______________________________________________________________________________________________________________________
def yield_tex_str(hist, i, prec=3, noerror=False):
    tmp = yield_str(hist, i, prec, noerror)
    tmp = tmp.__str__()
    sep = '\xc2\xb1'
    tmp = tmp.replace(sep, "$\pm$")
    return tmp

#______________________________________________________________________________________________________________________
def print_yield_table_from_list(hists, outputname, prec=2, binrange=[], noerror=False):
    x = Table()
    if len(hists) == 0:
        return
    # add bin column
    bins = binrange if len(binrange) != 0 else range(0, hists[0].GetNbinsX()+2)
    labels = hists[0].GetXaxis().GetLabels()
    if labels:
        x.add_column("Bin#", [ hists[0].GetXaxis().GetBinLabel(i) for i in bins])
    else:
        x.add_column("Bin#", ["Bin{}".format(i) for i in bins])
    for hist in hists:
        x.add_column(hist.GetName(), [ yield_str(hist, i, prec, noerror) for i in bins])
    fname = outputname
    fname = os.path.splitext(fname)[0]+'.txt'
    x.print_table()
    x.set_theme_basic()

    # Write text version
    makedir(os.path.dirname(fname))
    f = open(fname, "w")
    f.write("".join(x.get_table_string()))

#______________________________________________________________________________________________________________________
def print_yield_tex_table_from_list(hists, outputname, prec=2, caption="PUT YOUR CAPTION HERE", noerror=False, content_only=True):
    x = Table()
    if len(hists) == 0:
        return
    # add bin column
    labels = hists[0].GetXaxis().GetLabels()
    if labels:
        x.add_column("", [hists[0].GetXaxis().GetBinLabel(i) for i in xrange(1, hists[0].GetNbinsX()+1)])
    else:
        x.add_column("", ["Bin{}".format(i) for i in xrange(1, hists[0].GetNbinsX()+1)])
    for hist in hists:
        name = hist.GetName()
        if '#' in name:
            name = name.replace("#", "\\")
            name = "$" + name + "$"
        x.add_column(name, [ yield_tex_str(hist, i, prec, noerror) for i in xrange(1, hist.GetNbinsX()+1)])
    fname = outputname
    fname = os.path.splitext(fname)[0]+'.tex'
    x.set_theme_basic()

    # Change style for easier tex conversion
    x.d_style["INNER_INTERSECT"] = ''
    x.d_style["OUTER_RIGHT_INTERSECT"] = ''
    x.d_style["OUTER_BOTTOM_INTERSECT"] = ''
    x.d_style["OUTER_BOTTOM_LEFT"] = ''
    x.d_style["OUTER_BOTTOM_RIGHT"] = ''
    x.d_style["OUTER_TOP_INTERSECT"] = ''
    x.d_style["OUTER_TOP_LEFT"] = ''
    x.d_style["OUTER_TOP_RIGHT"] = ''
    x.d_style["INNER_HORIZONTAL"] = ''
    x.d_style["OUTER_BOTTOM_HORIZONTAL"] = ''
    x.d_style["OUTER_TOP_HORIZONTAL"] = ''

    x.d_style["OUTER_LEFT_VERTICAL"] = ''
    x.d_style["OUTER_RIGHT_VERTICAL"] = ''

#        self.d_style["INNER_HORIZONTAL"] = '-'
#        self.d_style["INNER_INTERSECT"] = '+'
#        self.d_style["INNER_VERTICAL"] = '|'
#        self.d_style["OUTER_LEFT_INTERSECT"] = '|'
#        self.d_style["OUTER_RIGHT_INTERSECT"] = '+'
#        self.d_style["OUTER_BOTTOM_HORIZONTAL"] = '-'
#        self.d_style["OUTER_BOTTOM_INTERSECT"] = '+'
#        self.d_style["OUTER_BOTTOM_LEFT"] = '+'
#        self.d_style["OUTER_BOTTOM_RIGHT"] = '+'
#        self.d_style["OUTER_TOP_HORIZONTAL"] = '-'
#        self.d_style["OUTER_TOP_INTERSECT"] = '+'
#        self.d_style["OUTER_TOP_LEFT"] = '+'
#        self.d_style["OUTER_TOP_RIGHT"] = '+'

    content = [ x for x in ("".join(x.get_table_string())).split('\n') if len(x) > 0 ]

    # Write tex from text version table
    f = open(fname, 'w')
    content = tabletex.makeTableTeX(content, complete=False)
    header = """\\begin{table}[!htb]
\\caption{"""
    header += caption
    header +="""}
\\resizebox{1.0\\textwidth}{!}{
"""
    footer = """}
\\end{table}
"""
    if not content_only:
        f.write(header)
    f.write(content)
    if not content_only:
        f.write(footer)

#______________________________________________________________________________________________________________________
def print_yield_table(hdata, hbkgs, hsigs, hsyst, options):
    hists = []
    hists.extend(hbkgs)
    hists.extend(hsigs)
    htotal = None
    if len(hbkgs) != 0:
        htotal = get_total_hist(hbkgs)
        htotal.SetName("Total")
        hists.append(htotal)
    if hdata and len(hbkgs) != 0:
        #print hdata
        #hratio = makeRatioHist(hdata, hbkgs)
        hratio = hdata.Clone("Ratio")
        hratio.Divide(htotal)
        #hists.append(htotal)
        hists.append(hdata)
        hists.append(hratio)
    prec = 2
    if "yield_prec" in options:
        prec = options["yield_prec"]
        del options["yield_prec"]
    print_yield_table_from_list(hists, options["output_name"], prec)
    print_yield_tex_table_from_list(hists, options["output_name"], prec, options["yield_table_caption"] if "yield_table_caption" in options else "PUT YOUR CAPTION HERE")
    if "yield_table_caption" in options: del options["yield_table_caption"]

def copy_nice_plot_index_php(options):
    plotdir = os.path.dirname(options["output_name"])
    if len(plotdir) == 0: plotdir = "./"
    os.system("cp {}/index.php {}/".format(os.path.realpath(__file__).rsplit("/",1)[0], plotdir))
    os.system("chmod 755 {}/index.php".format(plotdir))
#    os.system("cp {}/syncfiles/miscfiles/index.php {}/".format(os.path.realpath(__file__).rsplit("/",1)[0], plotdir))

def copy_nice_plot(plotdir):
    os.system("cp {}/syncfiles/miscfiles/index.php {}/".format(os.path.realpath(__file__).rsplit("/",1)[0], plotdir))

#______________________________________________________________________________________________________________________
def autobin(data, bgs):
    totalbkg = get_total_hist(bgs)

    accumulative = totalbkg.Clone("accumul")
    norm = accumulative.Integral() if accumulative.Integral() > 0 else 1
    accumulative.Scale(1. / norm)
    idx5 = -1
    idx95 = -1
    for i in xrange(1, accumulative.GetNbinsX()+2):
        intg = accumulative.Integral(0, i)
        if intg > 0.02 and idx5 < 0:
            idx5 = i
        if intg > 0.98 and idx95 < 0:
            idx95 = i
    minbin = -1
    if data:
        for i in xrange(idx5, idx95):
            bc = data.GetBinContent(i)
            if bc < minbin or minbin < 0:
                minbin = bc
    ndata = int(totalbkg.Integral(idx5, idx95))
    if ndata > 0:
        nbin = int(1 + 3.322 * math.log10(ndata))
    else:
        nbin = 4
    width = idx95 - idx5 + 1
    if data:
        frac = float(width) / float(data.GetNbinsX())
    else:
        frac = 1
    final_nbin = int(nbin / frac)
    if data:
        while data.GetNbinsX() % final_nbin != 0:
            if data.GetNbinsX() < final_nbin:
                return 4
            final_nbin += 1
    return final_nbin

# ====================
# The plottery wrapper
# ====================

#______________________________________________________________________________________________________________________
def plot_hist(data=None, bgs=[], sigs=[], syst=None, options={}, colors=[], sig_labels=[], legend_labels=[]):
    """
    Wrapper function to call Plottery.
    """

    # Set can extend turned off if label exists
    for h in bgs + sigs + [data] + [syst]:
        if h:
            labels = h.GetXaxis().GetLabels()
            if labels:
                h.SetCanExtend(False)

    # If print_all true, print all histogram content
    if "print_all" in options:
        if options["print_all"]:
            for bg in bgs: bg.Print("all")
            for sig in sigs: sig.Print("all")
            if data: data.Print("all")
        del options["print_all"]

    # Sanity check. If no histograms exit
    if not data and len(bgs) == 0 and len(sigs) == 0:
        print "[plottery_wrapper] >>> Nothing to do!"
        return

    # If a blind option is set, blind the data histogram to None
    # The later step will set the histogram of data to all zero
    if "blind" in options:
        if options["blind"]:
            data = None
            options["no_ratio"] = True
        del options["blind"]

    # signal scaling
    if "signal_scale" in options:
        if options["signal_scale"] == "auto":
            integral = get_total_hist(bgs).Integral()
            for sig in sigs:
                if sig.Integral() != 0:
                    sig.Scale(integral/sig.Integral())
            del options["signal_scale"]
        else:
            for sig in sigs:
                sig.Scale(options["signal_scale"])
            del options["signal_scale"]

    # autobin
    if "autobin" in options and options["autobin"]:
        options["nbins"] = autobin(data, bgs)
        del options["autobin"]
    elif "autobin" in options:
        del options["autobin"]

    # "nbins" initiate rebinning
    if "nbins" in options:
        nbins = options["nbins"]
        rebin(sigs, nbins)
        rebin(bgs, nbins)
        rebin([data], nbins)
        del options["nbins"]

    if "divide_by_bin_width" in options:
        if options["divide_by_bin_width"]:
            divide_by_bin_width(sigs)
            divide_by_bin_width(bgs)
            divide_by_bin_width([data])
            if "yaxis_label" not in options:
                options["yaxis_label"] = "Events / Bin Width"
        del options["divide_by_bin_width"]

    if "remove_underflow" in options:
        if options["remove_underflow"]:
            remove_underflow(sigs)
            remove_underflow(bgs)
            if data:
                remove_underflow([data])
        del options["remove_underflow"]

    if "remove_overflow" in options:
        if options["remove_overflow"]:
            remove_overflow(sigs)
            remove_overflow(bgs)
            if data:
                remove_overflow([data])
        del options["remove_overflow"]

    if "divide_by_first_bin" in options:
        if options["divide_by_first_bin"]:
            normalize_by_first_bin(sigs)
            normalize_by_first_bin(bgs)
            if data:
                normalize_by_first_bin([data])
        del options["divide_by_first_bin"]

    # If data is none clone one hist and fill with 0
    didnothaveanydata = False
    if not data:
        didnothaveanydata = True
        if len(bgs) != 0:
            data = bgs[0].Clone("Data")
            data.Reset()
        elif len(sigs) != 0:
            data = sigs[0].Clone("Data")
            data.Reset()

    # Compute some arguments that are missing (viz. colors, sig_labels, legend_labels)
    hsig_labels = sig_labels
    if len(sig_labels) == 0:
        for hsig in sigs:
            hsig_labels.append(hsig.GetName())
    hcolors = colors
    if len(colors) == 0:
        for index, hbg in enumerate(bgs):
            hcolors.append(default_colors[index])
    hlegend_labels = legend_labels
    if len(legend_labels) == 0:
        for hbg in bgs:
            hlegend_labels.append(hbg.GetName())

    # Set maximum of the plot
    totalbkg = None
    if len(bgs) != 0:
        totalbkg = get_total_hist(bgs)
    maxmult = 1.8
    if "ymax_scale" in options:
        maxmult = options["ymax_scale"]
        del options["ymax_scale"]
    yaxismax = get_max_yaxis_range_order_half_modded(get_max_yaxis_range([data, totalbkg]) * maxmult)
    yaxismin = get_nonzeromin_yaxis_range(bgs)
    #yaxismin = 1000

    if "yaxis_log" in options:
        if options["yaxis_log"] and ("yaxis_range" not in options or options["yaxis_range"] == []):
            options["yaxis_range"] = [yaxismin, 10000*(yaxismax-yaxismin)+yaxismax]
            print [yaxismin, 10000*(yaxismax-yaxismin)+yaxismax]

    # Once maximum is computed, set the y-axis label location
    if yaxismax < 0.01:
        options["yaxis_title_offset"] = 1.8
    elif yaxismax < 0.1:
        options["yaxis_title_offset"] = 1.6
    elif yaxismax < 10.:
        options["yaxis_title_offset"] = 1.6
    elif yaxismax < 100:
        options["yaxis_title_offset"] = 1.2
    elif yaxismax < 1000:
        options["yaxis_title_offset"] = 1.45
    elif yaxismax < 10000:
        options["yaxis_title_offset"] = 1.6
    else:
        options["yaxis_title_offset"] = 1.8

    # Print histogram content for debugging
    #totalbkg.Print("all")
    #if len(sigs) > 0:
    #    sigs[0].Print("all")
    #for hbg in bgs:
    #    hbg.Print("all")

    # Print yield table if the option is turned on
    if "print_yield" in options:
        if options["print_yield"]:
            print_yield_table(data, bgs, sigs, syst, options)
        del options["print_yield"]

    # Inject signal option
    if "inject_signal" in options:
        if options["inject_signal"]:
            if len(sigs) > 0:
                data = sigs[0].Clone("test")
                data.Reset()
                for hsig in sigs:
                    data.Add(hsig)
                for hbkg in bgs:
                    data.Add(hbkg)
                for i in xrange(1, data.GetNbinsX() + 1):
                    data.SetBinError(i, 0)
                options["legend_datalabel"] = "Sig+Bkg"
        del options["inject_signal"]

    # do KS test and add it to extra_text
    if "do_ks_test" in options:
        if options["do_ks_test"]:
            ksval = totalbkg.KolmogorovTest(data)
            options["extra_text"] = ["KS={:.2f}".format(ksval)]
        del options["do_ks_test"]

    # do smoothing
    if "do_smooth" in options:
        if options["do_smooth"]:
            for hsig in sigs:
                hsig.Smooth()
            for hbkg in bgs:
                hbkg.Smooth()
        del options["do_smooth"]

    if "print_mean" in options:
        if options["print_mean"]:
            mean = totalbkg.GetMean()
            try:
                options["extra_text"].append("mean={:.2f}".format(mean))
            except:
                options["extra_text"] = ["mean={:.2f}".format(mean)]
        del options["print_mean"]

    # If syst is not provided, compute one yourself from the bkg histograms
    if not syst:
        syst = get_total_err_hist(bgs)

    # The uncertainties are all accounted in the syst so remove all errors from bkgs
    remove_errors(bgs)

    # Get xaxis label from data, sig or bkg
    allhists = []
    for bg in bgs:
        allhists.append(bg)
    for sig in sigs:
        allhists.append(sig)
    if data:
        allhists.append(data)
    xaxis_label = allhists[0].GetXaxis().GetTitle()

    if "yaxis_range" in options and options["yaxis_range"] == []:
        del options["yaxis_range"]

    # Here are my default options for plottery
    #if not "canvas_width"             in options: options["canvas_width"]              = 604
    #if not "canvas_height"            in options: options["canvas_height"]             = 728
    if not "yaxis_log"                      in options: options["yaxis_log"]                      = False
    if not "canvas_width"                   in options: options["canvas_width"]                   = 454
    if not "canvas_height"                  in options: options["canvas_height"]                  = 553
    if not "yaxis_range"                    in options: options["yaxis_range"]                    = [0., yaxismax]
    if not "legend_ncolumns"                in options: options["legend_ncolumns"]                = 2 if len(bgs) >= 4 else 1
    if not "legend_alignment"               in options: options["legend_alignment"]               = "topright"
    #if not "legend_smart"                   in options: options["legend_smart"]                   = True if not options["yaxis_log"] else False
    if not "legend_smart"                   in options: options["legend_smart"]                   = True
    if not "legend_scalex"                  in options: options["legend_scalex"]                  = 0.8
    if not "legend_scaley"                  in options: options["legend_scaley"]                  = 0.8
    if not "legend_border"                  in options: options["legend_border"]                  = False
    if not "legend_rounded"                 in options: options["legend_rounded"]                 = False
    if not "legend_percentageinbox"         in options: options["legend_percentageinbox"]         = False
    if not "legend_opacity"                 in options: options["legend_opacity"]                 = 1
    if not "hist_line_none"                 in options: options["hist_line_none"]                 = False
    if not "hist_line_black"                in options: options["hist_line_black"]                = True
    if not "show_bkg_errors"                in options: options["show_bkg_errors"]                = False
    if not "ratio_range"                    in options: options["ratio_range"]                    = [0.7, 1.3]
    if not "ratio_name_size"                in options: options["ratio_name_size"]                = 0.13
    if not "ratio_name_offset"              in options: options["ratio_name_offset"]              = 0.6
    if not "ratio_xaxis_label_offset"       in options: options["ratio_xaxis_label_offset"]       = 0.06
    if not "ratio_yaxis_label_offset"       in options: options["ratio_yaxis_label_offset"]       = 0.03
    if not "ratio_xaxis_title_offset"       in options: options["ratio_xaxis_title_offset"]       = 1.60 if "xaxis_log" in options and options["xaxis_log"] else 1.40
    if not "ratio_xaxis_title_size"         in options: options["ratio_xaxis_title_size"]         = 0.13
    if not "ratio_label_size"               in options: options["ratio_label_size"]               = 0.13
    if not "canvas_tick_one_side"           in options: options["canvas_tick_one_side"]           = True
    if not "canvas_main_y1"                 in options: options["canvas_main_y1"]                 = 0.2
    if not "canvas_main_topmargin"          in options: options["canvas_main_topmargin"]          = 0.2 / 0.7 - 0.2
    if not "canvas_main_rightmargin"        in options: options["canvas_main_rightmargin"]        = 50. / 600.
    if not "canvas_main_bottommargin"       in options: options["canvas_main_bottommargin"]       = 0.2
    if not "canvas_main_leftmargin"         in options: options["canvas_main_leftmargin"]         = 130. / 600.
    if not "canvas_ratio_y2"                in options: options["canvas_ratio_y2"]                = 0.342
    if not "canvas_ratio_topmargin"         in options: options["canvas_ratio_topmargin"]         = 0.05
    if not "canvas_ratio_rightmargin"       in options: options["canvas_ratio_rightmargin"]       = 50. / 600.
    if not "canvas_ratio_bottommargin"      in options: options["canvas_ratio_bottommargin"]      = 0.4
    if not "canvas_ratio_leftmargin"        in options: options["canvas_ratio_leftmargin"]        = 130. / 600.
    if not "xaxis_title_size"               in options: options["xaxis_title_size"]               = 0.06
    if not "yaxis_title_size"               in options: options["yaxis_title_size"]               = 0.06
    if not "xaxis_title_offset"             in options: options["xaxis_title_offset"]             = 1.4 
    if not "yaxis_title_offset"             in options: options["yaxis_title_offset"]             = 1.4 
    if not "xaxis_label_size_scale"         in options: options["xaxis_label_size_scale"]         = 1.4
    if not "yaxis_label_size_scale"         in options: options["yaxis_label_size_scale"]         = 1.4
    if not "xaxis_label_offset_scale"       in options: options["xaxis_label_offset_scale"]       = 4.0
    if not "yaxis_label_offset_scale"       in options: options["yaxis_label_offset_scale"]       = 4.0
    if not "xaxis_tick_length_scale"        in options: options["xaxis_tick_length_scale"]        = -0.8
    if not "yaxis_tick_length_scale"        in options: options["yaxis_tick_length_scale"]        = -0.8
    if not "ratio_tick_length_scale"        in options: options["ratio_tick_length_scale"]        = -1.0
    if not "output_name"                    in options: options["output_name"]                    = "plots/plot.png"
    if not "cms_label"                      in options: options["cms_label"]                      = "Preliminary"
    if not "lumi_value"                     in options: options["lumi_value"]                     = "35.9"
    if not "bkg_err_fill_style"             in options: options["bkg_err_fill_style"]             = 3245
    if not "bkg_err_fill_color"             in options: options["bkg_err_fill_color"]             = 1
    if not "output_ic"                      in options: options["output_ic"]                      = 0
    if not "yaxis_moreloglabels"            in options: options["yaxis_moreloglabels"]            = False
    if not "yaxis_noexponents"              in options: options["yaxis_noexponents"]              = False
    if not "yaxis_exponent_offset"          in options: options["yaxis_exponent_offset"]          = -0.1
    if not "yaxis_exponent_vertical_offset" in options: options["yaxis_exponent_vertical_offset"] = 0.02
    if not "yaxis_ndivisions"               in options: options["yaxis_ndivisions"]               = 508
    if not "xaxis_ndivisions"               in options: options["xaxis_ndivisions"]               = 508
    if not "ratio_ndivisions"               in options: options["ratio_ndivisions"]               = 508
    if not "max_digits"                     in options: options["max_digits"]                     = 4
    if not "xaxis_label"                    in options: options["xaxis_label"]                    = xaxis_label
    if not "ratio_xaxis_title"              in options: options["ratio_xaxis_title"]              = xaxis_label
    if data == None:
        options["no_ratio"] = True
    if "no_ratio" in options:
        options["canvas_width"] = 566
        options["canvas_height"] = 553
        #options["canvas_width"] = (566 - 4) * 2 + 4
        #options["canvas_height"] = (553 - 28) * 2 + 28

    #if "no_ratio" in options:
    #    if options["no_ratio"]:
    #        options["canvas_ratio_y2"] = 0.0
    #        options["canvas_ratio_y1"] = 0.0
    #        options["canvas_ratio_x2"] = 0.0
    #        options["canvas_ratio_x1"] = 0.0
    #    del options["no_ratio"]

    # If you did not pass any data then set data back to None
    if didnothaveanydata:
        data = None

    # Call Plottery! I hope you win the Lottery!
    c1 = p.plot_hist(
            data          = data,
            bgs           = bgs,
            sigs          = sigs,
            syst          = syst,
            sig_labels    = hsig_labels,
            colors        = hcolors,
            legend_labels = hlegend_labels,
            options       = options
            )

    # Set permission
    os.system("chmod 644 {}".format(options["output_name"]))

    options["output_name"] = options["output_name"].replace("pdf","png")
    # Call Plottery! I hope you win the Lottery!
    c1 = p.plot_hist(
            data          = data,
            bgs           = bgs,
            sigs          = sigs,
            syst          = syst,
            sig_labels    = hsig_labels,
            colors        = hcolors,
            legend_labels = hlegend_labels,
            options       = options
            )

    # Set permission
    os.system("chmod 644 {}".format(options["output_name"]))

    # Call nice plots
    copy_nice_plot_index_php(options)

#______________________________________________________________________________________________________________________
def plot_cut_scan(data=None, bgs=[], sigs=[], syst=None, options={}, colors=[], sig_labels=[], legend_labels=[]):
    hsigs = []
    hbgs = []
    if syst:
        leftscan, rightscan = plot_sigscan_w_syst(sigs[0].Clone(), [bg.Clone() for bg in bgs], systs=syst)
    else:
        leftscan, rightscan = plot_sigscan(sigs[0].Clone(), get_total_hist(bgs).Clone())
    leftscan.Print("all")
    if leftscan.GetBinContent(1) != 0:
        leftscan.Scale(1./leftscan.GetBinContent(1))
    if rightscan.GetBinContent(rightscan.GetNbinsX()) != 0:
        rightscan.Scale(1./rightscan.GetBinContent(rightscan.GetNbinsX()))
    leftscan.SetFillStyle(1)
    hbgs.append(leftscan.Clone())
    hsigs.append(rightscan.Clone())
    scan2d = plot_sigscan2d(sigs[0].Clone(), get_total_hist(bgs).Clone())
    if scan2d.GetBinContent(1) != 0:
        scan2d.Scale(1./scan2d.GetBinContent(1))
    hsigs.append(scan2d.Clone())
    leftscan, rightscan = plot_sigscan(sigs[0].Clone(), get_total_hist(bgs).Clone(), fom_acceptance)
    hsigs.append(leftscan.Clone())
    hsigs.append(rightscan.Clone())
    options["bkg_err_fill_color"] = 0
    options["output_name"] = options["output_name"].replace(".png", "_cut_scan.png")
    options["output_name"] = options["output_name"].replace(".pdf", "_cut_scan.pdf")
    options["signal_scale"] = 1
    plot_hist(data=None, sigs=hsigs, bgs=hbgs, syst=None, options=options, colors=colors, sig_labels=sig_labels, legend_labels=legend_labels)

#______________________________________________________________________________________________________________________
def plot_soverb_scan(data=None, bgs=[], sigs=[], syst=None, options={}, colors=[], sig_labels=[], legend_labels=[]):
    hsigs = []
    hbgs = []
    if syst:
        leftscan, rightscan = plot_sigscan_w_syst(sigs[0].Clone(), [bg.Clone() for bg in bgs], systs=syst)
    else:
        leftscan, rightscan = plot_sigscan(sigs[0].Clone(), get_total_hist(bgs).Clone(), fom=fom_SoverB)
    leftscan.Print("all")
    if leftscan.GetBinContent(1) != 0:
        leftscan.Scale(1./leftscan.GetBinContent(1))
    if rightscan.GetBinContent(rightscan.GetNbinsX()) != 0:
        rightscan.Scale(1./rightscan.GetBinContent(rightscan.GetNbinsX()))
    leftscan.SetFillStyle(1)
    hbgs.append(leftscan.Clone())
    hsigs.append(rightscan.Clone())
    scan2d = plot_sigscan2d(sigs[0].Clone(), get_total_hist(bgs).Clone())
    if scan2d.GetBinContent(1) != 0:
        scan2d.Scale(1./scan2d.GetBinContent(1))
    hsigs.append(scan2d.Clone())
    leftscan, rightscan = plot_sigscan(sigs[0].Clone(), get_total_hist(bgs).Clone(), fom_acceptance)
    hsigs.append(leftscan.Clone())
    hsigs.append(rightscan.Clone())
    options["bkg_err_fill_color"] = 0
    options["output_name"] = options["output_name"].replace(".png", "_cut_scan.png")
    options["output_name"] = options["output_name"].replace(".pdf", "_cut_scan.pdf")
    options["signal_scale"] = 1
    plot_hist(data=None, sigs=hsigs, bgs=hbgs, syst=None, options=options, colors=colors, sig_labels=sig_labels, legend_labels=legend_labels)

#______________________________________________________________________________________________________________________
def plot_roc(fps=[],tps=[],legend_labels=[],colors=[],cutvals=[],scanreverse=[],options={},_persist=[]):

    #opts = Options(options, kind="graph")

    #style = utils.set_style()

    #c1 = r.TCanvas()
    #if opts["canvas_width"] and opts["canvas_height"]:
    #    width = opts["canvas_width"]
    #    height = opts["canvas_height"]
    #    c1 = r.TCanvas("c1", "c1", width, height)
    #_persist.append(c1) # need this to avoid segfault with garbage collection

    #pad_main = r.TPad("pad1","pad1",0.,0.,1.,1.)
    #if opts["canvas_main_topmargin"]: pad_main.SetTopMargin(opts["canvas_main_topmargin"])
    #if opts["canvas_main_rightmargin"]: pad_main.SetRightMargin(opts["canvas_main_rightmargin"])
    #if opts["canvas_main_bottommargin"]: pad_main.SetBottomMargin(opts["canvas_main_bottommargin"])
    #if opts["canvas_main_leftmargin"]: pad_main.SetLeftMargin(opts["canvas_main_leftmargin"])
    #if opts["canvas_tick_one_side"]: pad_main.SetTicks(0, 0)
    #pad_main.Draw()

    #pad_main.cd()

    map(u.move_in_overflows, tps)
    map(u.move_in_overflows, fps)

    #legend = get_legend(opts)

    # generalize later
    if len(tps) != len(fps):
        print len(tps), len(fps)
        print ">>> number of true positive hists and false positive hists must match"
        sys.exit(-1)

    debug = False

    ## do your thing
    valpairs = []
    pointpairs = []
    ref_seff = 0
    ref_beff = 0
    for index, _ in enumerate(tps):

        sighist = tps[index]
        bkghist = fps[index]
        cutval = cutvals[index] if len(cutvals) == len(tps) else -999

        if debug: print "[DEBUG] >>> here", sighist.GetName(), bkghist.GetName()

        error = r.Double()

        stot = sighist.IntegralAndError(0, sighist.GetNbinsX()+1, error)
        btot = bkghist.IntegralAndError(0, bkghist.GetNbinsX()+1, error)

        if debug: print '[DEBUG] >>>', stot, btot
        if debug: print '[DEBUG] >>> sighist.GetMean()', sighist.GetMean()
        if debug: print '[DEBUG] >>> bkghist.GetMean()', bkghist.GetMean()

        x=[]
        y=[]
        cuteffset = False

        for i in range(0, sighist.GetNbinsX()+2):
            if len(scanreverse) > 0:
                doreverse = scanreverse[index]
            else:
                doreverse = False
            s = sighist.IntegralAndError(sighist.GetNbinsX()-i, sighist.GetNbinsX()+1, error)
            b = bkghist.IntegralAndError(sighist.GetNbinsX()-i, bkghist.GetNbinsX()+1, error)
            if doreverse:
                s = sighist.IntegralAndError(0, 1 + i, error)
                b = bkghist.IntegralAndError(0, 1 + i, error)
            #s = sighist.IntegralAndError(0, i, error)
            #b = bkghist.IntegralAndError(0, i, error)
            seff = s / stot
            beff = b / btot
            curval = sighist.GetXaxis().GetBinUpEdge(sighist.GetNbinsX()) - i * sighist.GetXaxis().GetBinWidth(1)
            if doreverse:
                curval = sighist.GetXaxis().GetBinUpEdge(1 + i)
            print seff, beff, curval
#            if abs(ref_seff - seff) < 0.03:
#                print abs(ref_seff - seff) < 0.03
#                print ref_seff
#                print cuteffset
#                print cutval == -999, cutval
            if abs(ref_seff - seff) < 0.01 and ref_seff > 0 and not cuteffset and cutval == -999:
#                print 'here'
                cuteffset = True
                legend_labels[index] = "({0:.2f}, {1:.4f}) @ {2} ".format(seff, beff, curval) + legend_labels[index] if len(legend_labels[index]) > 0 else ""
                pointpairs.append(([beff], [seff]))
            if not doreverse:
                if curval <= cutval and not cuteffset:
                    legend_labels[index] = "({0:.2f}, {1:.4f}) @ {2} ".format(seff, beff, curval) + legend_labels[index] if len(legend_labels[index]) > 0 else ""
                    pointpairs.append(([beff], [seff]))
                    cuteffset = True
                    if ref_seff == 0: ref_seff = seff
                    if ref_beff == 0: ref_beff = beff
            else:
                if curval >= cutval and not cuteffset:
                    legend_labels[index] = "({0:.2f}, {1:.4f}) @ {2} ".format(seff, beff, curval) + legend_labels[index] if len(legend_labels[index]) > 0 else ""
                    pointpairs.append(([beff], [seff]))
                    cuteffset = True
                    if ref_seff == 0: ref_seff = seff
                    if ref_beff == 0: ref_beff = beff
            if debug:
                if abs(sighist.GetBinLowEdge(i) - 0.25) < 0.01: print seff, beff, sighist.GetBinLowEdge(i), seff*seff / math.sqrt(beff), seff / math.sqrt(beff)
                if abs(sighist.GetBinLowEdge(i) - 0.15) < 0.01: print seff, beff, sighist.GetBinLowEdge(i), seff*seff / math.sqrt(beff), seff / math.sqrt(beff)
                if abs(sighist.GetBinLowEdge(i) - 0.10) < 0.01: print seff, beff, sighist.GetBinLowEdge(i), seff*seff / math.sqrt(beff), seff / math.sqrt(beff)
                if abs(sighist.GetBinLowEdge(i) - 0.07) < 0.01: print seff, beff, sighist.GetBinLowEdge(i), seff*seff / math.sqrt(beff), seff / math.sqrt(beff)
                if abs(beff - 0.07) < 0.02: print seff, beff, sighist.GetBinLowEdge(i), seff*seff / math.sqrt(beff), seff / math.sqrt(beff)
                if abs(beff - 0.04) < 0.02: print seff, beff, sighist.GetBinLowEdge(i), seff*seff / math.sqrt(beff), seff / math.sqrt(beff)
                if abs(seff - 0.91) < 0.02: print seff, beff, sighist.GetBinLowEdge(i), seff*seff / math.sqrt(beff), seff / math.sqrt(beff)
            #if beff != 0:
            #    print seff, beff, sighist.GetBinLowEdge(i), seff*seff / math.sqrt(beff), seff / math.sqrt(beff), s, b, stot, btot
            x.append(beff)
            y.append(seff)

        valpairs.append((x,y))

        #graph = ROOT.TGraph(len(x))
        #for index, i in enumerate(x):
        #    graph.SetPoint(index, x[index], y[index])

        #graph.SetTitle(legend_labels[index])
        #graph.SetName(legend_labels[index])
        #graph.SetMinimum(0.)
        #graph.SetMaximum(1)
        ##graph.GetXaxis().SetRangeUser(0.05,1)
        #graph.SetLineColor(colors[index])
        #graph.SetLineWidth(colors[index])
        ##graph.GetXaxis().SetTitle("Eff Background")
        ##graph.GetYaxis().SetTitle("Eff Signal")
        ##self.histmanager.set_histaxis_settings(graph, 1.0)
        ##from copy import deepcopy
        #graphs.append(graph)
        ##self.objs.append(deepcopy(graph))

    #ymin, ymax = 0., 1. # generally ROC curves are always between 0. to 1.

    #for index, graph in enumerate(graphs):
    #    if index == 0:
    #        if opts["yaxis_range"]:
    #            graph.SetMinimum(opts["yaxis_range"][0])
    #            graph.SetMaximum(opts["yaxis_range"][1])
    #            ymin, ymax = opts["yaxis_range"]
    #        graph.SetMinimum(ymin)
    #        graph.SetMaximum(ymax)
    #        graph.Draw("alp")
    #    else:
    #        graph.Draw("lp")

    #draw_cms_lumi(pad_main, opts)
    #handle_axes(pad_main, stack, opts)
    #draw_extra_stuff(pad_main, opts)

    ## ymin ymax needs to be defined
    #if opts["legend_smart"]:
    #    utils.smart_legend(legend, bgs, data=data, ymin=ymin, ymax=ymax, opts=opts)
    #legend.Draw()

    #save(c1, opts)
    if not "legend_alignment"               in options: options["legend_alignment"]               = "bottomright"
    if not "legend_scalex"                  in options: options["legend_scalex"]                  = 1.5
    if not "legend_scaley"                  in options: options["legend_scaley"]                  = 0.8
    if not "legend_border"                  in options: options["legend_border"]                  = False

    valpairs.extend(pointpairs)

    draw_styles=[]
    for i in colors: draw_styles.append(1)

    colors.extend(colors)

    ll = []
    for x in legend_labels:
        if len(x) > 0:
            ll.append(x)
    legend_labels = ll

    c1 = p.plot_graph(valpairs, colors=colors, legend_labels=legend_labels, options=options, draw_styles=draw_styles)

    copy_nice_plot_index_php(options)

    return c1


#______________________________________________________________________________________________________________________
def plot_hist_2d(hist,options={}):
    p.plot_hist_2d(hist, options)
    options["output_name"] = options["output_name"].replace("pdf","png")
    p.plot_hist_2d(hist, options)

#______________________________________________________________________________________________________________________
def dump_plot_v1(fname, dirname="plots"):

    f = r.TFile(fname)
    
    hists = {}
    for key in f.GetListOfKeys():
        hists[key.GetName()] = f.Get(key.GetName())
    
    fn = fname.replace(".root", "")
    for hname in hists:
        if hists[hname].GetDimension() == 1:
            plot_hist(bgs=[hists[hname]], options={"output_name": dirname + "/" + fn + "_" + hname + ".pdf"})
        if hists[hname].GetDimension() == 2:
            plot_hist_2d(hist=hists[hname], options={"output_name": dirname + "/" + fn + "_" + hname + ".pdf"})

#______________________________________________________________________________________________________________________
def dump_plot(fnames=[], sig_fnames=[], data_fname=None, dirname="plots", legend_labels=[], signal_labels=None, donorm=False, filter_pattern="", signal_scale=1, extraoptions={}, usercolors=None, do_sum=False, output_name=None, dogrep=False, _plotter=plot_hist, doKStest=False, histmodfunc=None):

    # color_pallete
    colors_ = default_colors
    if usercolors:
        colors_ = usercolors + default_colors

    # Open all files and define color schemes
    sample_names = []
    tfs = {}
    clrs = {}
    issig = [] # Aggregate a list of signal samples
    for index, fname in enumerate(fnames + sig_fnames):
        n = os.path.basename(fname.replace(".root", ""))
        n += str(index)
        tfs[n] = r.TFile(fname)
        clrs[n] = colors_[index]
        sample_names.append(n)
        if index >= len(fnames):
            issig.append(n)

    if data_fname:
        n = os.path.basename(data_fname.replace(".root", ""))
        tfs[n] = r.TFile(data_fname)
        clrs[n] = colors_[index]
        sample_names.append(n)
    
    # Tag the data sample names
    data_sample_name = None
    if data_fname:
        n = os.path.basename(data_fname.replace(".root", ""))
        data_sample_name = n

    # Form a complete key list
    hist_names = []
    for n in tfs:
        for key in tfs[n].GetListOfKeys():
            if "TH" in tfs[n].Get(str(key.GetName())).ClassName():
                hist_names.append(str(key.GetName()))

    # Remove duplicate names
    hist_names = list(set(hist_names))

    # Sort
    hist_names.sort()

    # summed_hist if do_sum is true
    summed_hist = []

    # Loop over hist_names
    for hist_name in hist_names:
        
        # If to filter certain histograms
        if filter_pattern:
            if dogrep:
                doskip = True
                for item in filter_pattern.split(","):
                    if "*" in item:
                        match = True
                        for token in item.split("*"):
                            if token not in hist_name:
                                match = False
                                break
                        if match:
                            doskip = False
                            break
                    else:
                        if item in hist_name:
                            doskip = False
                            break
                if doskip:
                    continue
            else:
                doskip = True
                for item in filter_pattern.split(","):
                    if hist_name == item:
                        doskip = False
                        break
                if doskip:
                    continue

        hists = []
        colors = []
        for n in sample_names:
            h = tfs[n].Get(hist_name)
            if h:
                if signal_labels:
                    if n in issig:
                        h.SetName(signal_labels[issig.index(n)])
                    else:
                        h.SetName(n)
                else:
                    h.SetName(n)
                hists.append(h)
                colors.append(clrs[n])
            else:
                print "ERROR: did not find histogram", hist_name, "for the file", tfs[n].GetName()
                sys.exit(1)

        if do_sum:

            if len(summed_hist) > 0:

                for index, h in enumerate(hists):

                    summed_hist[index].Add(h)

            else:

                for h in hists:

                    summed_hist.append(h.Clone())

        else:

            if output_name:
                hist_name = output_name

            if len(hists) > 0:
                if hists[0].GetDimension() == 1:
                    if donorm: # shape comparison so use sigs to overlay one bkg with multiple shape comparison
                        options = {"signal_scale": "auto", "output_name": dirname + "/" + hist_name + ".pdf"}
                        options.update(extraoptions)
                        _plotter(bgs=[hists[0]], sigs=hists[1:], colors=colors, options=options, legend_labels=legend_labels)
                    else:
                        # Get the list of histograms and put them in either bkg or signals
                        sigs = [ hists[index] for index, n in enumerate(sample_names) if n in issig ] # list of signal histograms
                        bkgs = [ hists[index] for index, n in enumerate(sample_names) if (n not in issig) and n != data_sample_name ] # list of bkg histograms
                        data = [ hists[index] for index, n in enumerate(sample_names) if n == data_sample_name ][0] if data_sample_name else None
                        colors = [ colors[index] for index, n in enumerate(sample_names) if n not in issig ] # list of bkg colors
                        # But check if bkgs is at least 1
                        if len(bkgs) == 0:
                            bkgs = [ sigs.pop(0) ]
                        options = {"output_name": dirname + "/" + hist_name + ".pdf", "signal_scale": signal_scale, "do_ks_test":doKStest}
                        options.update(extraoptions)
                        _plotter(bgs=bkgs, sigs=sigs, data=data, colors=colors, options=options, legend_labels=legend_labels if _plotter==plot_hist else [])
                if hists[0].GetDimension() == 2:
                    if donorm:
                        for h in hists:
                            h.Scale(1./h.Integral())

                    # Compute range
                    zmax = hists[0].GetMaximum()
                    zmin = hists[0].GetMinimum()
                    for h in hists:
                        zmax = h.GetMaximum() if h.GetMaximum() > zmax else zmax
                        zmin = h.GetMinimum() if h.GetMinimum() > zmin else zmin
                    for h in hists:
                        if histmodfunc:
                            h = histmodfunc(h)
                        # plot_hist_2d(hist=h, options={"output_name": dirname + "/" + str(h.GetName()) + "_" + hist_name + "_log.pdf", "zaxis_log":True, "draw_option_2d":"lego2"})
                        # plot_hist_2d(hist=h, options={"output_name": dirname + "/" + str(h.GetName()) + "_" + hist_name + "_lin.pdf", "zaxis_log":False, "draw_option_2d":"lego2"})
                        # plot_hist_2d(hist=h, options={"output_name": dirname + "/" + str(h.GetName()) + "_" + hist_name + "_commonlog.pdf", "zaxis_log":True, "zaxis_range":[zmin, zmax], "draw_option_2d":"lego2"})
                        # plot_hist_2d(hist=h, options={"output_name": dirname + "/" + str(h.GetName()) + "_" + hist_name + "_commonlin.pdf", "zaxis_log":False, "zaxis_range":[zmin, zmax], "draw_option_2d":"lego2"})
                        options={"output_name": dirname + "/" + str(h.GetName()) + "_" + hist_name + "_log.pdf", "zaxis_log":True, "draw_option_2d":"colz"}
                        options.update(extraoptions)
                        plot_hist_2d(hist=h, options=options)
                        options={"output_name": dirname + "/" + str(h.GetName()) + "_" + hist_name + "_lin.pdf", "zaxis_log":False, "draw_option_2d":"colz"}
                        options.update(extraoptions)
                        plot_hist_2d(hist=h, options=options)
                        options={"output_name": dirname + "/" + str(h.GetName()) + "_" + hist_name + "_commonlog.pdf", "zaxis_log":True, "zaxis_range":[zmin, zmax], "draw_option_2d":"colz"}
                        options.update(extraoptions)
                        plot_hist_2d(hist=h, options=options)
                        options={"output_name": dirname + "/" + str(h.GetName()) + "_" + hist_name + "_commonlin.pdf", "zaxis_log":False, "zaxis_range":[zmin, zmax], "draw_option_2d":"colz"}
                        options.update(extraoptions)
                        plot_hist_2d(hist=h, options=options)

    if do_sum:

        hists = summed_hist

        if output_name:
            hist_name = output_name
        else:
            hist_name = hists[0].GetName()

        if hists[0].GetDimension() == 1:
            if donorm: # shape comparison so use sigs to overlay one bkg with multiple shape comparison
                options = {"signal_scale": "auto", "output_name": dirname + "/" + hist_name + ".pdf"}
                options.update(extraoptions)
                _plotter(bgs=[hists[0]], sigs=hists[1:], colors=colors, options=options, legend_labels=legend_labels)
            else:
                # Get the list of histograms and put them in either bkg or signals
                sigs = [ hists[index] for index, n in enumerate(sample_names) if n in issig ] # list of signal histograms
                bkgs = [ hists[index] for index, n in enumerate(sample_names) if (n not in issig) and n != data_sample_name ] # list of bkg histograms
                data = [ hists[index] for index, n in enumerate(sample_names) if n == data_sample_name ][0] if data_sample_name else None
                colors = [ colors[index] for index, n in enumerate(sample_names) if n not in issig ] # list of bkg colors
                # But check if bkgs is at least 1
                if len(bkgs) == 0:
                    bkgs = [ sigs.pop(0) ]
                options = {"output_name": dirname + "/" + hist_name + ".pdf", "signal_scale": signal_scale}
                options.update(extraoptions)
                _plotter(bgs=bkgs, sigs=sigs, data=data, colors=colors, options=options, legend_labels=legend_labels if _plotter==plot_hist else [])
        if hists[0].GetDimension() == 2:
            if donorm:
                for h in hists:
                    h.Scale(1./h.Integral())

            # Compute range
            zmax = hists[0].GetMaximum()
            zmin = hists[0].GetMinimum()
            for h in hists:
                zmax = h.GetMaximum() if h.GetMaximum() > zmax else zmax
                zmin = h.GetMinimum() if h.GetMinimum() > zmin else zmin
            for h in hists:
                plot_hist_2d(hist=h, options={"output_name": dirname + "/" + str(h.GetName()) + "_" + hist_name + "_log.pdf", "zaxis_log":True, "draw_option_2d":"lego2"})
                plot_hist_2d(hist=h, options={"output_name": dirname + "/" + str(h.GetName()) + "_" + hist_name + "_lin.pdf", "zaxis_log":False, "draw_option_2d":"lego2"})
                plot_hist_2d(hist=h, options={"output_name": dirname + "/" + str(h.GetName()) + "_" + hist_name + "_commonlog.pdf", "zaxis_log":True, "zaxis_range":[zmin, zmax], "draw_option_2d":"lego2"})
                plot_hist_2d(hist=h, options={"output_name": dirname + "/" + str(h.GetName()) + "_" + hist_name + "_commonlin.pdf", "zaxis_log":False, "zaxis_range":[zmin, zmax], "draw_option_2d":"lego2"})
                # plot_hist_2d(hist=h, options={"output_name": dirname + "/" + str(h.GetName()) + "_" + hist_name + "_log.pdf", "zaxis_log":True, "draw_option_2d":"colz"})
                # plot_hist_2d(hist=h, options={"output_name": dirname + "/" + str(h.GetName()) + "_" + hist_name + "_lin.pdf", "zaxis_log":False, "draw_option_2d":"colz"})
                # plot_hist_2d(hist=h, options={"output_name": dirname + "/" + str(h.GetName()) + "_" + hist_name + "_commonlog.pdf", "zaxis_log":True, "zaxis_range":[zmin, zmax], "draw_option_2d":"colz"})
                # plot_hist_2d(hist=h, options={"output_name": dirname + "/" + str(h.GetName()) + "_" + hist_name + "_commonlin.pdf", "zaxis_log":False, "zaxis_range":[zmin, zmax], "draw_option_2d":"colz"})

def plot_yields(fnames=[], sig_fnames=[], data_fname=None, regions=[], binlabels=[], output_name="yield", dirname="plots", legend_labels=[], signal_labels=None, donorm=False, signal_scale="", extraoptions={}, usercolors=None, hsuffix="_cutflow", _plotter=plot_hist):

    # color_pallete
    colors_ = default_colors
    if usercolors:
        colors_ = usercolors + default_colors

    # Open all files and define color schemes
    sample_names = []
    tfs = {}
    fns = {}
    clrs = {}
    issig = [] # Aggregate a list of signal samples
    for index, fname in enumerate(fnames + sig_fnames):
        n = os.path.basename(fname.replace(".root", ""))
        n += str(index)
        tfs[n] = r.TFile(fname)
        fns[n] = fname
        clrs[n] = colors_[index]
        sample_names.append(n)
        if index >= len(fnames):
            issig.append(n)

    if data_fname:
        n = os.path.basename(data_fname.replace(".root", ""))
        tfs[n] = r.TFile(data_fname)
        fns[n] = data_fname
        clrs[n] = colors_[index]
        sample_names.append(n)

    # Tag the data sample names
    data_sample_name = None
    if data_fname:
        n = os.path.basename(data_fname.replace(".root", ""))
        data_sample_name = n

    yield_hs = []
    for sn in sample_names:
        yield_hs.append(ru.get_yield_histogram( list_of_file_names=[ fns[sn] ], regions=regions, labels=binlabels, hsuffix=hsuffix))
        if signal_labels:
            if sn in issig:
                yield_hs[-1].SetName(signal_labels[issig.index(sn)])
            else:
                yield_hs[-1].SetName(sn)
        else:
            yield_hs[-1].SetName(sn)

    colors = []
    for n in sample_names:
        colors.append(clrs[n])

    # Get the list of histograms and put them in either bkg or signals
    sigs = [ yield_hs[index] for index, n in enumerate(sample_names) if n in issig ] # list of signal histograms
    bkgs = [ yield_hs[index] for index, n in enumerate(sample_names) if (n not in issig) and n != data_sample_name ] # list of bkg histograms
    data = [ yield_hs[index] for index, n in enumerate(sample_names) if n == data_sample_name ][0] if data_sample_name else None
    colors = [ colors[index] for index, n in enumerate(sample_names) if n not in issig ] # list of bkg colors
    # But check if bkgs is at least 1
    if len(bkgs) == 0:
        bkgs = [ sigs.pop(0) ]
    options = {"output_name": dirname + "/" + output_name + ".pdf", "signal_scale": signal_scale}
    options.update(extraoptions)
    _plotter(bgs=bkgs, sigs=sigs, data=data, colors=colors, options=options, legend_labels=legend_labels if _plotter==plot_hist else [])


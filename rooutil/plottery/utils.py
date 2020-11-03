import ROOT as r
import os
import math
import random
from array import array

class MyArc(r.TLine):

    def __init__(self, xc, yc, radius, phimin=180, phimax=360, ninterp=6):
        self.xc = xc
        self.yc = yc
        self.radius = radius
        self.phimin = phimin
        self.phimax = phimax
        self.ninterp = ninterp
        super(MyArc, self).__init__()

    def Draw(self, opt=""):
        xc, yc, radius, phimin, phimax = self.xc, self.yc, self.radius, self.phimin, self.phimax
        ninterp = self.ninterp
        dphi = 1.0*(phimax-phimin)/ninterp
        phis = [phimin + dphi*i for i in range(ninterp+1)]
        coords = []
        conv = math.pi/180.
        for iphi,phi in enumerate(phis):
            x = xc + radius*math.cos(phi*conv)
            y = yc + radius*math.sin(phi*conv)
            coords.append([x,y])
        for (x1,y1),(x2,y2) in zip(coords[:-1],coords[1:]):
            self.DrawLineNDC(x1,y1,x2,y2)

def set_style():

    tdr_style = r.TStyle("tdr_style","Style for P-TDR")

    #  For the canvas:
    tdr_style.SetCanvasBorderMode(0)
    tdr_style.SetCanvasColor(r.kWhite)
    tdr_style.SetCanvasDefH(550) # Height of canvas
    tdr_style.SetCanvasDefW(600) # Width of canvas
    tdr_style.SetCanvasDefX(0)   # Position on screen
    tdr_style.SetCanvasDefY(0)

    #  For the Pad:
    tdr_style.SetPadBorderMode(0)
    tdr_style.SetPadColor(r.kWhite)
    tdr_style.SetPadGridX(False)
    tdr_style.SetPadGridY(False)
    tdr_style.SetGridColor(0)
    tdr_style.SetGridStyle(3)
    tdr_style.SetGridWidth(1)

    #  For the frame:
    tdr_style.SetFrameBorderMode(0)
    tdr_style.SetFrameBorderSize(1)
    tdr_style.SetFrameFillColor(0)
    tdr_style.SetFrameFillStyle(0)
    tdr_style.SetFrameLineColor(1)
    tdr_style.SetFrameLineStyle(1)
    tdr_style.SetFrameLineWidth(1)

    # For the histo:
    tdr_style.SetHistLineColor(r.kBlack)
    tdr_style.SetHistLineWidth(2)
    tdr_style.SetEndErrorSize(2)
    tdr_style.SetMarkerStyle(20)

    # For the fit/function:
    tdr_style.SetOptFit(1)
    tdr_style.SetFitFormat("5.4g")
    tdr_style.SetFuncColor(2)
    tdr_style.SetFuncStyle(1)
    tdr_style.SetFuncWidth(1)

    # For the date:
    tdr_style.SetOptDate(0)

    # For the statistics box:
    tdr_style.SetOptFile(0)
    tdr_style.SetOptStat(0) #  To display the mean and RMS:   SetOptStat("mr")
    tdr_style.SetOptFit(0) #  To display the mean and RMS:   SetOptStat("mr")
    tdr_style.SetStatColor(r.kWhite)
    tdr_style.SetStatFont(42)
    tdr_style.SetStatFontSize(0.025)
    tdr_style.SetStatTextColor(1)
    tdr_style.SetStatFormat("6.4g")
    tdr_style.SetStatBorderSize(1)
    tdr_style.SetStatH(0.1)
    tdr_style.SetStatW(0.15)

    # Margins:
    tdr_style.SetPadTopMargin(0.08)
    tdr_style.SetPadBottomMargin(0.12)
    tdr_style.SetPadLeftMargin(0.13)
    tdr_style.SetPadRightMargin(0.04)

    # For the Global title:
    tdr_style.SetOptTitle(1)
    tdr_style.SetTitleFont(42)
    tdr_style.SetTitleColor(1)
    tdr_style.SetTitleTextColor(1)
    tdr_style.SetTitleFillColor(10)
    tdr_style.SetTitleFontSize(0.05)
    tdr_style.SetTitleX(0.5) #  Set the position of the title box
    tdr_style.SetTitleY(0.985) #  Set the position of the title box
    tdr_style.SetTitleAlign(23)
    tdr_style.SetTitleStyle(0)
    tdr_style.SetTitleBorderSize(0)
    tdr_style.SetTitleFillColor(0)

    r.TGaxis.SetExponentOffset(-0.06, 0, "y")
    r.TGaxis.SetExponentOffset(-0.86, -0.08, "x")

    # For the axis titles:
    tdr_style.SetTitleColor(1, "XYZ")
    tdr_style.SetTitleFont(42, "XYZ")
    tdr_style.SetTitleSize(0.045, "XYZ")
    tdr_style.SetTitleOffset(1.02, "X")
    tdr_style.SetTitleOffset(1.26, "Y")

    # For the axis labels:
    tdr_style.SetLabelColor(1, "XYZ")
    tdr_style.SetLabelFont(42, "XYZ")
    tdr_style.SetLabelOffset(0.007, "XYZ")
    tdr_style.SetLabelSize(0.040, "XYZ")

    # For the axis:
    tdr_style.SetAxisColor(1, "XYZ")
    tdr_style.SetStripDecimals(True)
    tdr_style.SetTickLength(0.03, "XYZ")
    tdr_style.SetNdivisions(510, "XYZ")
    tdr_style.SetPadTickX(1)  #  To get tick marks on the opposite side of the frame
    tdr_style.SetPadTickY(1)

    # Change for log plots:
    tdr_style.SetOptLogx(0)
    tdr_style.SetOptLogy(0)
    tdr_style.SetOptLogz(0)

    # Postscript options:
    tdr_style.SetPaperSize(20.,20.)
    tdr_style.cd()

    return tdr_style

def set_style_2d():
    style = set_style()
    style.SetPadBottomMargin(0.12)
    style.SetPadRightMargin(0.12)
    style.SetPadLeftMargin(0.10)
    style.SetTitleAlign(23)
    style.cd()

    return style

def set_palette(style, palette):
    if palette == "default":
        style.SetPalette(r.kBird) # default
        style.SetNumberContours(128)
    elif palette == "rainbow":
        style.SetPalette(r.kRainBow) # blue to red
        style.SetNumberContours(128)
    elif palette == "radiation":
        style.SetPalette(r.kInvertedDarkBodyRadiator) # default
        style.SetNumberContours(128)
    elif palette == "susy":
        stops = array('d', [0.00, 0.34, 0.61, 0.84, 1.00])
        red   = array('d', [0.50, 0.50, 1.00, 1.00, 1.00])
        green = array('d', [0.50, 1.00, 1.00, 0.60, 0.50])
        blue  = array('d', [1.00, 1.00, 0.50, 0.40, 0.50])
        r.TColor.CreateGradientColorTable(len(stops), stops, red, green, blue, 255)
        # print get_luminosities(len(stops), stops, red, green, blue, 255)
        style.SetNumberContours(255)

def get_brightdefault_colors():
    return [r.kBlack, r.kAzure, r.kRed, r.kGreen+1, r.kOrange-2, r.kMagenta]

def get_default_colors():
    return [r.kSpring-6, r.kAzure+7, r.kRed-7, r.kOrange-2, r.kCyan-7, r.kMagenta-7, r.kTeal+6, r.kGray+2, r.kGray, r.kBlue-2, r.kRed-2]

def get_default_marker_shapes():
    return [20,21,22,23,29,34]

def get_brightdefault_colors():
    return [r.kBlack, r.kAzure, r.kRed, r.kGreen+1, r.kOrange-2, r.kMagenta]

def hsv_to_rgb(h, s, v, scale=255.):
    """
    Takes hue, saturation, value 3-tuple
    and returns rgb 3-tuple
    """
    if s == 0.0: v*=scale; return [v, v, v]
    i = int(h*6.)
    f = 1.0*(h*6.)-i; p,q,t = int(scale*(v*(1.-s))), int(scale*(v*(1.-s*f))), int(scale*(v*(1.-s*(1.-f)))); v*=scale; i%=6
    if i == 0: return [v, t, p]
    if i == 1: return [q, v, p]
    if i == 2: return [p, v, t]
    if i == 3: return [p, q, v]
    if i == 4: return [t, p, v]
    if i == 5: return [v, p, q]

def rgb_to_hsv(r,g,b):
    """
    Reverse of hsv to rgb, but I think this is buggy.
    Check before using (i.e., rgb_to_hsv(hsv_to_rgb(x)) == x)
    """
    vmin = min(min(r,g),b)
    vmax = max(max(r,g),b)
    delta = 1.0*(vmax-vmin)
    value = 1.0*vmax
    if vmax > 0.:
        satur = delta/vmax
    else:
        satur = 0
        hue = -1
        return (hue,satur,value)
    if r == vmax:
        hue = 1.0*(g-b)/delta
    elif g == vmax:
        hue = 2.0+(b-r)/delta
    else:
        hue = 4.0+(r-g)/delta
    hue *= 60
    if (hue < 0): hue += 360
    return (hue,satur,value)

def interpolate_tuples(first, second, ndiv):
    """
    Given two n-tuples, and a number of divisions (ndiv), create
    ndiv n-tuples that are linearly spaced between first and second
    """
    def interp1d(one,two,ndiv):
        return [one+1.0*(two-one)*i/(ndiv-1) for i in range(ndiv)]
    return list(zip(*map(lambda x: interp1d(x[0],x[1],ndiv), zip(first,second))))

def get_legend_marker_info(legend):
    ncols = legend.GetNColumns()
    nrows = legend.GetNRows()
    x1 = legend.GetX1()
    y1 = legend.GetY1()
    x2 = legend.GetX2()
    y2 = legend.GetY2()
    margin = legend.GetMargin()*( x2-x1 )/ncols
    boxwidth = margin
    boxw = boxwidth*0.35
    yspace = (y2-y1)/nrows;
    draw_vertical = False
    coordsNDC = []

    for ientry in range(nrows*ncols):
        icol = ientry % ncols
        irow = ientry // ncols
        # note, we can't support more than 2 columns because
        # ROOT won't give us an easy way to get the relative sizes
        # of the columns (we need to know the text sizes) :(
        colfudge = -margin*0.35 # 2 cols
        xc = x1+0.5*margin+((x2-x1)/ncols+(colfudge))*icol
        yc = y2-0.5*yspace-irow*yspace
        coordsNDC.append([xc,yc])

    # if marker box is tall and skinny, the height of the text
    # is not the limitation. the width is. so we scale down the
    # label height a bit to accomodate the width
    label_height = (0.6-0.1*ncols)*yspace
    if label_height/boxw > 2.1:
        # if super tall/skinny, also draw 90deg rotated text
        draw_vertical = True
    if label_height/boxw > 1.6:
        label_height = 1.6*boxw

    return { "coords": coordsNDC, "label_height": label_height, "box_width": boxw, "draw_vertical": draw_vertical }

def get_stack_maximum(data, stack, opts={}):
    scalefact = 1.05
    if opts["yaxis_range"]:
        return opts["yaxis_range"][1]
    if data:
        return scalefact*max(data.GetMaximum(),stack.GetMaximum())
    else:
        return scalefact*stack.GetMaximum()

def compute_darkness(r,g,b):
    """
    Compute darkness := 1 - luminance, given RGB
    """
    return 1.0 - (0.299*r + 0.587*g + 0.114*b)

def interpolate_colors_rgb(first, second, ndiv, _persist=[]):
    """
    Create ndiv colors that are linearly interpolated between rgb triplets
    first and second
    """
    colorcodes = []
    for rgb in interpolate_tuples(first,second,ndiv):
        index = r.TColor.GetFreeColorIndex()
        _persist.append(r.TColor(index, *rgb))
        colorcodes.append(index)
    return colorcodes


def draw_flag(c1, cx, cy, size, _persist=[]):
    """
    Draw US flag
    # NOTE: May cause segfaults when flags are drawn
    # on more than 2 plots?
    """
    c1.cd();
    aspect_ratio = 1.33 # c1.GetWindowWidth()/c1.GetWindowHeight();
    xmin = cx-size/2.;
    xmax = cx+size/2.;
    ymin = cy-size/(2./aspect_ratio);
    ymax = cy+size/(2./aspect_ratio);
    fp = r.TPad("fp","fp",xmin,ymin,xmax,ymax);
    fp.SetFillStyle(0);
    fp.Draw();
    fp.cd();
    _persist.append(fp)
    A = 1.;
    B = 1.9;
    D = 0.76;
    G = 0.063/B;
    H = 0.063/B;
    E = 0.054;
    F = 0.054;
    for i in range(13):
        xlow = 0.;
        xhigh = 1.;
        ylow = 0.5*(1.-A/B) + i*(A/B)/13.;
        yhigh = 0.5*(1.-A/B) + (i+1)*(A/B)/13.;
        if (i >= 6): xlow = D/B;
        col = r.kWhite if i%2 else r.kRed-7
        box = r.TBox(xlow,ylow,xhigh,yhigh);
        box.SetFillColor(col);
        box.SetLineColor(col);
        box.Draw();
        _persist.append(box)

    starbox = r.TBox( 0., 0.5*(1-A/B)+6./13*(A/B), D/B, 1.-0.5*(1-A/B) );
    starbox.SetFillColor(r.kBlue-7);
    starbox.SetLineColor(r.kBlue-7);
    starbox.Draw();
    _persist.append(starbox)

    row = 0;
    inrow = 0;
    ybottom = 0.5*(1-A/B)+6./13*(1-A/B);
    starsize = 0.05+(xmax-xmin)*2.0;

    for i in range(50):

        x = -1.;
        y = -1.;
        if (inrow == 0): x = G;
        else: x = G+2*H*inrow;
        if (row == 0): y = ybottom+E;
        else: y = ybottom+E+(F*row)*(A/B);
        if (row%2!=0): x += H;

        tm = r.TMarker(x,y,r.kFullStar);
        tm.SetMarkerColor(r.kWhite);
        tm.SetMarkerSize(-1.0*starsize); # negative to flip so points upwards
        tm.Draw();
        _persist.append(tm)

        inrow += 1
        if (row%2 == 0):
            if (inrow == 6):
                inrow = 0;
                row += 1;
        else:
            if (inrow == 5):
                inrow = 0;
                row += 1;

    lab = r.TLatex(0.5,0.15,"#font[52]{Mostly made in USA}");
    lab.SetTextAlign(22);
    lab.SetTextSize(0.1);
    lab.SetTextColor(r.kGray+2);
    lab.Draw();
    _persist.append(lab)

    c1.cd();

def get_mean_sigma_1d_yvals(hist):
    """
    Return mean, sigma, yvals of a 1D hist (by basically "projecting" onto y-axis)
    """
    vals = list(hist)[1:-1]
    errs = [hist.GetBinError(ibin) for ibin in range(hist.GetNbinsX()+1)][1:-1]
    htmp = r.TH1D("htmp","htmp",150,min(vals),max(vals))
    if sum(errs) < 1e-6: errs = [1.+err for err in errs]
    for val,err in zip(vals,errs):
        if err < 1.e-6: continue
        htmp.Fill(val,1./err)
    mean, sigma = htmp.GetMean(), htmp.GetRMS()
    return mean, sigma, vals

def move_in_overflows(h):
    """
    Takes a histogram and moves the under and overflow bins
    into the first and last visible bins, respectively
    Errors are combined in quadrature
    """
    nbins = h.GetNbinsX()
    v_under = h[0]
    v_first = h[1]
    e_under = h.GetBinError(0)
    e_first = h.GetBinError(1)
    v_over = h[nbins+1]
    v_last = h[nbins]
    e_over = h.GetBinError(nbins+1)
    e_last = h.GetBinError(nbins)

    # When the bin labels are set the CanExtend messes things up.
    h.SetCanExtend(False)

    # Reset overflows to 0
    h.SetBinContent(0, 0)
    h.SetBinContent(nbins+1, 0)
    h.SetBinError(0, 0)
    h.SetBinError(nbins+1, 0)

    # Put them into first and last bins
    h.SetBinContent(1, v_first+v_under)
    h.SetBinContent(nbins, v_last+v_over)
    h.SetBinError(1, (e_first**2.+e_under**2.)**0.5)
    h.SetBinError(nbins, (e_last**2.+e_over**2.)**0.5)

def fill_fast(hist, xvals, yvals=None, weights=None):
    """
    partially stolen from root_numpy implementation
    using for loop with TH1::Fill() is slow, so use
    numpy to convert array to C-style array, and then FillN
    """
    import numpy as np
    xvals = np.asarray(xvals, dtype=np.double)
    two_d = False
    if yvals is not None:
        two_d = True
        yvals = np.asarray(yvals, dtype=np.double)
    if weights is None:
        weights = np.ones(len(xvals))
    else:
        weights = np.asarray(weights, dtype=np.double)
    if not two_d:
        hist.FillN(len(xvals),xvals,weights)
    else:
        hist.FillN(len(xvals),xvals,yvals,weights)

def draw_smart_2d_bin_labels(hist,opts):
    """
    Replicate the TEXT draw option for TH2 with TLatex drawn everywhere
    but calculate the background color of each bin and draw text as
    white or black depending on the darkness
    """
    darknesses = [] # darkness values
    lights = [] # lighter colors
    darks = [] # darker colors
    ncolors = r.gStyle.GetNumberContours()
    for ic in range(ncolors):
        code = r.gStyle.GetColorPalette(ic)
        color = r.gROOT.GetColor(code)
        red = color.GetRed()
        green = color.GetGreen()
        blue = color.GetBlue()
        darks.append(r.TColor.GetColorDark(code))
        lights.append(r.TColor.GetColorBright(code))
        darkness = compute_darkness(red, green, blue)
        darknesses.append(darkness)
    labels = []
    zlow, zhigh = max(1,hist.GetMinimum()), hist.GetMaximum()
    if opts["zaxis_range"]: zlow, zhigh = opts["zaxis_range"]
    t = r.TLatex()
    t.SetTextAlign(22)
    t.SetTextSize(0.025)
    fmt = opts["bin_text_format_smart"]
    for ix in range(1,hist.GetNbinsX()+1):
        for iy in range(1,hist.GetNbinsY()+1):
            xcent = hist.GetXaxis().GetBinCenter(ix)
            ycent = hist.GetYaxis().GetBinCenter(iy)
            val = hist.GetBinContent(ix,iy)
            err = hist.GetBinError(ix,iy)
            if val == 0: continue
            if opts["zaxis_log"]:
                frac = (math.log(min(val,zhigh))-math.log(zlow))/(math.log(zhigh)-math.log(zlow))
            else:
                frac = (min(val,zhigh)-zlow)/(zhigh-zlow)
            if frac > 1.: continue
            idx = int(frac*(len(darknesses)-1))
            if darknesses[idx] < 0.7:
                t.SetTextColor(r.kBlack)
            else:
                t.SetTextColor(r.kWhite)
            # t.SetTextColor(darks[idx])
            t.DrawLatex(xcent,ycent,fmt.format(val,err))
            labels.append(t)

def smart_legend(legend, bgs, data=None, ymin=0., ymax=None, Nx=25, Ny=25, niters=7, opts={}):
    """
    Given a TLegend, backgrounds, and optionally data,
    find a location where the TLegend doesn't overlap these objects
    preserving the width and height of the TLegend object
    by scanning over a Nx x Ny grid. If a non-overlapping position is not
    found, we decrease the legend width and height and try scanning again.
    Repeat this `niters` times before giving up.
    """


    debug = False # draw bounding boxes, etc

    def bar_in_box(coords_first,coords_second, exclude_if_below=True):
        # return true if any part of bar (top of bar represented by (x1,y1))
        # overlaps with box (bx1,bx2,by1,by2)
        # if !exclude_if_below, then we allow the box if it's above OR below the point
        x1,y1 = coords_first
        bx1,bx2,by1,by2 = coords_second
        does_x_overlap = bx1 <= x1 <= bx2
        if does_x_overlap:
            if exclude_if_below:
                return y1 > by1
            else:
                return by1 <= y1 <= by2
        else: return False

    def point_in_box(coords_first, coords_second):
        x1,y1 = coords_first
        bx1, bx2, by1, by2 = coords_second
        # return true if point is in box
        return (by1 <= y1 <= by2) or (bx1 <= x1 <= bx2)

    def is_good_legend(coords, pseudo_legend, exclude_if_below=True):
        # return true if this pseudolegend (given by 4-tuple pseudo_legend)
        # is a good legend (i.e., doesn't overlap with list of pairs in coords
        for coord in coords:
            if bar_in_box(coord, pseudo_legend, exclude_if_below=exclude_if_below): return False
        return True

    def distance_from_corner(pseudo_legend):
        # return euclidean distance of corner of pseudo legend cloest to plot
        # pane corner (note, this is rough)
        x1,x2,y1,y2 = pseudo_legend
        dist = 0.
        if 0.5*(y1+y2) > 0.5: dist += (1.0-y2)**2.
        else: dist += (y1)**2.
        if 0.5*(x1+x2) > 0.5: dist += (1.0-x2)**2.
        else: dist += (x1)**2.
        return dist**0.5

    allbgs = bgs[0].Clone("allbgs")
    allbgs.Reset()
    if opts["do_stack"]:
        for hist in bgs:
            allbgs.Add(hist)
    else:
        for ibin in range(1,allbgs.GetNbinsX()+1):
            allbgs.SetBinContent(ibin, max(hist.GetBinContent(ibin) for hist in bgs))
            if opts["draw_points"]:
                allbgs.SetBinContent(ibin, max(hist.GetBinContent(ibin)+hist.GetBinError(ibin) for hist in bgs))

    if not ymax:
        ymax = allbgs.GetMaximum()


    # get coordinates of legend corners
    leg_x1 = legend.GetX1()
    leg_x2 = legend.GetX2()
    leg_y1 = legend.GetY1()
    leg_y2 = legend.GetY2()
    legend_coords = (leg_x1,leg_x2,leg_y1,leg_y2)
    legend_width, legend_height = leg_x2 - leg_x1, leg_y2 - leg_y1
    xmin = allbgs.GetBinLowEdge(1)
    xmax = allbgs.GetBinLowEdge(allbgs.GetNbinsX()) + allbgs.GetBinWidth(allbgs.GetNbinsX())
    coords = []
    extra_coords = []

    # coords veto a legend if they are within the box, or
    # if the box is below the coord
    for ibin in range(1,allbgs.GetNbinsX()+1):
        xval = allbgs.GetBinCenter(ibin)
        yval = allbgs.GetBinContent(ibin)
        # if we have data, and it's higher than bgs, then use that value
        if data and data.GetBinContent(ibin)+data.GetBinError(ibin) > yval:
            xval = data.GetBinCenter(ibin)
            yval = data.GetBinContent(ibin)
            yval += 1.0*data.GetBinError(ibin)
        yfrac = (yval - ymin) / (ymax - ymin)
        xfrac = (xval - xmin) / (xmax - xmin)

        if opts["yaxis_log"]:
            ymin = max(ymin,0.1)
            yval = max(yval, 0.0001)
            yfrac = 1.*(math.log(min(yval,ymax))-math.log(ymin))/(math.log(ymax)-math.log(ymin))

        # convert from 0..1 inside plotting pane, to pad coordinates (stupid margins)
        xcoord = xfrac * (1. - r.gPad.GetLeftMargin() - r.gPad.GetRightMargin()) + r.gPad.GetLeftMargin()
        ycoord = yfrac * (1. - r.gPad.GetTopMargin() - r.gPad.GetBottomMargin()) + r.gPad.GetBottomMargin()
        coord = (xcoord, ycoord)
        coords.append(coord)

    # # NOTE: bugged. can't seem to get NDC for TLatex, only user
    # # extra_coords to veto a legend if they are within the box
    # for elem in r.gPad.GetListOfPrimitives():
    #     if not elem.InheritsFrom(r.TLatex.Class()): continue
    #     x1 = elem.GetX()
    #     elem.SetNDC()
    #     print elem.GetXsize()
    #     x2 = x1 + elem.GetXsize()
    #     y1 = elem.GetY()
    #     extra_coords.append([x1,y1])
    #     extra_coords.append([x2,y1])

    if debug:
        # draw x's to debug
        t = r.TLatex()
        t.SetTextAlign(22)
        t.SetTextFont(42)
        t.SetTextColor(r.kRed)
        t.SetTextSize(0.05)
        for coord in coords:
            t.DrawLatexNDC(coord[0],coord[1],"x")
        for coord in extra_coords:
            t.DrawLatexNDC(coord[0],coord[1],"x")

        line = r.TLine()
        line.SetLineColor(r.kRed)

    # generate list of legend coordinate candidates, preserving original width, height
    # move around the original legend by Nx increments in x and Ny in y
    # if we don't find anything, decrease the size and try again
    paddingx = 0.03
    paddingy = 0.03
    for iiter in range(niters):
        pseudo_legends = []
        for ix in range(Nx):
            pseudox1 = 1.0*ix/Nx + paddingx + r.gPad.GetLeftMargin()
            if pseudox1 > 1.-r.gPad.GetRightMargin()-paddingx: continue
            pseudox2 = pseudox1 + legend_width
            if pseudox2 > 1.-r.gPad.GetRightMargin()-paddingx: continue
            for iy in range(Ny):
                pseudoy1 = 1.0*iy/Ny + paddingy + r.gPad.GetBottomMargin()
                if pseudoy1 > 1.-r.gPad.GetTopMargin()-paddingy: continue
                pseudoy2 = pseudoy1 + legend_height
                if pseudoy2 > 1.-r.gPad.GetTopMargin()-paddingy: continue
                pseudo_legends.append([pseudox1,pseudox2,pseudoy1,pseudoy2])

        good_pseudo_legends = []
        for pseudo_legend in pseudo_legends:
            if not is_good_legend(coords, pseudo_legend): continue
            if not is_good_legend(extra_coords, pseudo_legend, exclude_if_below=False): continue
            good_pseudo_legends.append(pseudo_legend)

            if debug:
                xr = random.random()*0.05 - 0.025
                yr = random.random()*0.05 - 0.025
                x1, x2, y1, y2 = pseudo_legend
                line.SetLineColor(int(50*random.random()))
                line.DrawLineNDC(x1+xr,y1+yr,x2+xr,y1+yr)
                line.DrawLineNDC(x2+xr,y1+yr,x2+xr,y2+yr)
                line.DrawLineNDC(x2+xr,y2+yr,x1+xr,y2+yr)
                line.DrawLineNDC(x1+xr,y2+yr,x1+xr,y1+yr)

        good_pseudo_legends = sorted(good_pseudo_legends, key=lambda x: distance_from_corner(x))
        if len(good_pseudo_legends) > 0:
            legend.SetX1(good_pseudo_legends[0][0])
            legend.SetX2(good_pseudo_legends[0][1])
            legend.SetY1(good_pseudo_legends[0][2])
            legend.SetY2(good_pseudo_legends[0][3])
            break
        else:
            print(">>> Running another smart legend iteration decreasing legend height and width")
            legend_width *= 0.9
            legend_height *= 0.9
    else:
        print(">>> Tried to reduce legend width, height {} times, but still couldn't find a good position!".format(niters))


def diff_images(fname1, fname2, output="diff.png"):
    """
    Creates a file `output` that represents a diff of two input images
    `fname1` and `fname`. If these are .pdf, they will be first converted to .png.
    Example:
    >>> utils.diff_images("examples/test1.pdf", "examples/test3.pdf", output="diff.png")
    >>> os.system("ic diff.png")
    """
    import numpy as np
    import matplotlib.pylab as plt
    conversion_cmd = "gs -q -sDEVICE=pngalpha -o {outname} -sDEVICE=pngalpha -dUseCropBox -r{density} {inname}"
    # conversion_cmd = "convert -density {density} -trim {inname} -fuzz 1% {outname}"
    new_fnames = []
    for fname in [fname1, fname2]:
        if fname.rsplit(".",1)[-1] == "pdf":
            fname_in = fname
            fname_out = fname.replace(".pdf",".png")
            os.system(conversion_cmd.format(density=75, inname=fname_in, outname=fname_out))
            new_fnames.append(fname_out)
    if len(new_fnames) == 2: fname1, fname2 = new_fnames
    # img1 = plt.imread(fname1)[::2,::2] # downsample by factor of 2
    # img2 = plt.imread(fname2)[::2,::2]
    img1 = plt.imread(fname1)
    img2 = plt.imread(fname2)

    # Calculate the absolute difference on each channel separately
    error_r = np.fabs(np.subtract(img2[:,:,0], img1[:,:,0]))
    error_g = np.fabs(np.subtract(img2[:,:,1], img1[:,:,1]))
    error_b = np.fabs(np.subtract(img2[:,:,2], img1[:,:,2]))

    lum_img = np.sqrt(error_r*error_r + error_g+error_g + error_b*error_b)/np.sqrt(3)

    # # # Calculate the maximum error for each pixel
    # lum_img = np.maximum(np.maximum(error_r, error_g), error_b)

    # plt.set_cmap('Spectral')
    plt.set_cmap('gray')
    plt.imsave(output,-lum_img)

def draw_rounded_box(x1,y1,x2,y2,radius=0.05,width=2,color=r.kGray,alpha=0.5,expand=0.0,_persist=[]):
    x1 -= expand
    x2 += expand
    y1 -= expand
    y2 += expand

    lb = r.TLine(x1+radius,y1,x2-radius,y1)
    ll = r.TLine(x1,y1+radius,x1,y2-radius)
    lr = r.TLine(x2,y1+radius,x2,y2-radius)
    lt = r.TLine(x1+radius,y2,x2-radius,y2)

    abl = MyArc(x1+radius,y1+radius,radius,180,270)
    abr = MyArc(x2-radius,y1+radius,radius,0,-90)
    atl = MyArc(x1+radius,y2-radius,radius,90,180)
    atr = MyArc(x2-radius,y2-radius,radius,0,90)

    coll = [lb,ll,lr,lt,abl,abr,atl,atr]
    _persist.extend(coll)

    def f(obj):
        obj.SetBit(r.TLine.kLineNDC)
        obj.SetLineWidth(width)
        obj.SetLineColorAlpha(color,alpha)
        obj.Draw()

    list(map(f, coll))

def draw_shadow_rounded_box(x1,y1,x2,y2,radius=0.05,width=2,color=r.kGray,alpha=0.5,expand=0.0):
    for amult,ex in [
            (0.50, 0.002),
            (0.75, 0.001),
            (1.00, 0.000),
            ]:
        draw_rounded_box(x1,y1,x2,y2,radius+ex,color=color,width=width,alpha=amult*alpha,expand=ex)

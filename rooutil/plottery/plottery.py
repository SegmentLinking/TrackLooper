# coding: utf-8
import os
import ROOT as r
from . import utils
from array import array
import math
from itertools import cycle

r.gROOT.SetBatch(1) # please don't open an Xwindow
r.gEnv.SetValue("RooFit.Banner", "0") # turn off annoying RooFit banner
r.gErrorIgnoreLevel = r.kError # ignore Info/Warnings

class Options(object):
    """
    The Options object is just a nice wrapper around a dictionary
    with default values, some arithmetic, and warnings
    >>> import plottery as ply
    >>> # Passing d_opts1,d_opts2, or opts1 as the `options` kwarg to a plot
    >>> # function will have the same effect
    >>> d_opts1 = { "output_name": "test.pdf", "blah": 1, }
    >>> d_opts2 = { "blah2": 2, }
    >>> opts1 = ply.Options(d_opts1)
    >>> # You can add a dict or another Options object to an Options object
    >>> # to add new options or modify current ones
    >>> print opts1+d_opts2
    >>> print opts1+ply.Options(d_opts2)
    """

    def __init__(self, options={}, kind=None):

        # if we pass in a plain dict, then do the usual
        # thing, otherwise make a new options object
        # if an Options object is passed in
        if type(options) is dict:
            self.options = options
            self.kind = kind
        else:
            self.options = options.options
            self.kind = options.kind

        self.recognized_options = {

            # Canvas
            "canvas_width": {"type": "Int", "desc": "width of TCanvas in pixel", "default": None, "kinds": ["1dratio","graph","2d"], },
            "canvas_height": {"type": "Int", "desc": "height of TCanvas in pixel", "default": None, "kinds": ["1dratio","graph","2d"], },
            "canvas_main_y1": {"type": "Float", "desc": "main plot tpad y1", "default": 0.18, "kinds": ["1dratio","graph","2d"], },
            "canvas_main_topmargin": {"type": "Float", "desc": "ratio plot top margin", "default": None, "kinds": ["1dratio"], },
            "canvas_main_bottommargin": {"type": "Float", "desc": "ratio plot bottom margin", "default": None, "kinds": ["1dratio"], },
            "canvas_main_rightmargin": {"type": "Float", "desc": "ratio plot right margin", "default": None, "kinds": ["1dratio"], },
            "canvas_main_leftmargin": {"type": "Float", "desc": "ratio plot left margin", "default": None, "kinds": ["1dratio"], },
            "canvas_ratio_y2": {"type": "Float", "desc": "ratio tpad y2", "default": 0.19, "kinds": ["1dratio","graph","2d"], },
            "canvas_ratio_topmargin": {"type": "Float", "desc": "ratio plot top margin", "default": None, "kinds": ["1dratio"], },
            "canvas_ratio_bottommargin": {"type": "Float", "desc": "ratio plot bottom margin", "default": None, "kinds": ["1dratio"], },
            "canvas_ratio_rightmargin": {"type": "Float", "desc": "ratio plot right margin", "default": None, "kinds": ["1dratio"], },
            "canvas_ratio_leftmargin": {"type": "Float", "desc": "ratio plot left margin", "default": None, "kinds": ["1dratio"], },
            "canvas_tick_one_side": {"type": "Boolean", "desc": "ratio plot left margin", "default": False, "kinds": ["1dratio"], },

            # Legend
            "legend_coordinates": { "type": "List", "desc": "4 elements specifying TLegend constructor coordinates", "default": [0.63,0.67,0.93,0.87], "kinds": ["1dratio","graph"], },
            "legend_alignment": { "type": "String", "desc": "easy alignment of TLegend. String containing two words from: bottom, top, left, right", "default": "", "kinds": ["1dratio","graph"], },
            "legend_smart": { "type": "Boolean", "desc": "Smart alignment of legend to prevent overlaps", "default": True, "kinds": ["1dratio"], },
            "legend_border": { "type": "Boolean", "desc": "show legend border?", "default": True, "kinds": ["1dratio","graph"], },
            "legend_rounded": { "type": "Boolean", "desc": "rounded legend border", "default": True, "kinds": ["1dratio"], },
            "legend_scalex": { "type": "Float", "desc": "scale width of legend by this factor", "default": 1, "kinds": ["1dratio","graph"], },
            "legend_scaley": { "type": "Float", "desc": "scale height of legend by this factor", "default": 1, "kinds": ["1dratio","graph"], },
            "legend_opacity": { "type": "Float", "desc": "from 0 to 1 representing the opacity of the TLegend white background", "default": 0.5, "kinds": ["1dratio","graph"], },
            "legend_ncolumns": { "type": "Int", "desc": "number of columns in the legend", "default": 1, "kinds": ["1dratio","graph"], },
            "legend_column_separation": { "type": "Float", "desc": "column separation size", "default": None, "kinds": ["1dratio","graph"], },
            "legend_percentageinbox": { "type": "Boolean", "desc": "show relative process contributions as %age in the legend thumbnails", "default": True, "kinds": ["1dratio"], },
            "legend_datalabel": { "type": "String", "desc": "label for the data histogram in the legend", "default": "Data", "kinds": ["1dratio"], },

            # Axes
            "xaxis_log": { "type": "Boolean", "desc": "log scale x-axis", "default": False, "kinds": ["1dratio","graph","2d"], },
            "yaxis_log": { "type": "Boolean", "desc": "log scale y-axis", "default": False, "kinds": ["1dratio","graph","2d"], },
            "zaxis_log": { "type": "Boolean", "desc": "log scale z-axis", "default": False, "kinds": ["2d"], },

            "xaxis_label": { "type": "String", "desc": "label for x axis", "default": "", "kinds": ["1dratio","graph","2d"], },
            "yaxis_label": { "type": "String", "desc": "label for y axis", "default": "Events", "kinds": ["1dratio","graph","2d"], },
            "zaxis_label": { "type": "String", "desc": "label for z axis", "default": "", "kinds": ["2d"], },

            "xaxis_label_size_scale": { "type": "Float", "desc": "size of fonts for x axis", "default": 1.0, "kinds": ["1dratio","graph","2d"], },
            "yaxis_label_size_scale": { "type": "Float", "desc": "size of fonts for y axis", "default": 1.0, "kinds": ["1dratio","graph","2d"], },
            "zaxis_label_size_scale": { "type": "Float", "desc": "size of fonts for z axis", "default": 1.0, "kinds": ["2d"], },

            "xaxis_title_size": { "type": "Float", "desc": "size of fonts for x axis title", "default": None, "kinds": ["1dratio","graph","2d"], },
            "yaxis_title_size": { "type": "Float", "desc": "size of fonts for y axis title", "default": None, "kinds": ["1dratio","graph","2d"], },

            "xaxis_title_offset": { "type": "Float", "desc": "offset of x axis title", "default": None, "kinds": ["1dratio","graph","2d"], },
            "yaxis_title_offset": { "type": "Float", "desc": "offset of y axis title", "default": None, "kinds": ["1dratio","graph","2d"], },

            "xaxis_label_offset_scale": { "type": "Float", "desc": "x axis tickmark labels offset", "default": 1.0, "kinds": ["1dratio","graph","2d"], },
            "yaxis_label_offset_scale": { "type": "Float", "desc": "y axis tickmark labels offset", "default": 1.0, "kinds": ["1dratio","graph","2d"], },

            "xaxis_tick_length_scale": { "type": "Float", "desc": "x axis tickmark length scale", "default": 1.0, "kinds": ["1dratio","graph","2d"], },
            "yaxis_tick_length_scale": { "type": "Float", "desc": "y axis tickmark length scale", "default": 1.0, "kinds": ["1dratio","graph","2d"], },

            "xaxis_moreloglabels": { "type": "Boolean", "desc": "show denser labels with logscale for x axis", "default": True, "kinds": ["1dratio","graph","2d"], },
            "yaxis_moreloglabels": { "type": "Boolean", "desc": "show denser labels with logscale for y axis", "default": True, "kinds": ["1dratio","graph","2d"], },
            "zaxis_moreloglabels": { "type": "Boolean", "desc": "show denser labels with logscale for z axis", "default": True, "kinds": ["1dratio","graph","2d"], },
            "xaxis_noexponents": { "type": "Boolean", "desc": "don't show exponents in logscale labels for x axis", "default": False, "kinds": ["1dratio","graph","2d"], },
            "yaxis_noexponents": { "type": "Boolean", "desc": "don't show exponents in logscale labels for y axis", "default": False, "kinds": ["1dratio","graph","2d"], },
            "zaxis_noexponents": { "type": "Boolean", "desc": "don't show exponents in logscale labels for z axis", "default": False, "kinds": ["1dratio","graph","2d"], },

            "yaxis_exponent_offset": { "type": "Float", "desc": "offset x10^n left or right", "default": 0.0, "kinds": ["1dratio"], },
            "yaxis_exponent_vertical_offset": { "type": "Float", "desc": "offset x10^n up or down", "default": 0.0, "kinds": ["1dratio"], },

            "yaxis_ndivisions": { "type": "Int", "desc": "SetNdivisions integer for y-axis", "default": 510, "kinds": ["1dratio", "graph", "2d"], },
            "xaxis_ndivisions": { "type": "Int", "desc": "SetNdivisions integer for x-axis", "default": 510, "kinds": ["1dratio", "graph", "2d"], },

            "xaxis_range": { "type": "List", "desc": "2 elements to specify x axis range", "default": [], "kinds": ["1dratio","graph","2d"], },
            "yaxis_range": { "type": "List", "desc": "2 elements to specify y axis range", "default": [], "kinds": ["1dratio","graph","2d"], },
            "zaxis_range": { "type": "List", "desc": "2 elements to specify z axis range", "default": [], "kinds": ["2d"], },

            # Ratio
            "ratio_name": { "type": "String", "desc": "name of ratio pad", "default": "Data/MC", "kinds": ["1dratio"], },
            "ratio_name_size": { "type": "Float", "desc": "size of the name on the ratio pad (e.g. data/MC)", "default": 0.2, "kinds": ["1dratio"], },
            "ratio_name_offset": { "type": "Float", "desc": "offset to the name of ratio pad", "default": 0.25, "kinds": ["1dratio"], },
            "ratio_range": { "type": "List", "desc": "pair for min and max y-value for ratio; default auto re-sizes to 3 sigma range", "default": [-1,-1], "kinds": ["1dratio"], },
            "ratio_horizontal_lines": { "type": "List", "desc": "list of y-values to draw horizontal line", "default": [1.], "kinds": ["1dratio"], },
            "ratio_chi2prob": { "type": "Boolean", "desc": "show chi2 probability for ratio", "default": False, "kinds": ["1dratio"], },
            "ratio_pull": { "type": "Boolean", "desc": "show pulls instead of ratios in ratio pad", "default": False, "kinds": ["1dratio"], },
            "ratio_pull_numbers": { "type": "Boolean", "desc": "show numbers for pulls, and mean/sigma", "default": True, "kinds": ["1dratio"], },
            "ratio_ndivisions": { "type": "Int", "desc": "SetNdivisions integer for ratio", "default": 505, "kinds": ["1dratio"], },
            "ratio_numden_indices": { "type": "List", "desc": "Pair of numerator and denominator histogram indices (from `bgs`) for ratio", "default": None, "kinds": ["1dratio"], },
            "ratio_binomial_errors": { "type": "Boolean", "desc": "Use binomial error propagation when computing ratio eror bars", "default": False, "kinds": ["1dratio"], },
            "ratio_xaxis_title": { "type": "String", "desc": "X-axis label", "default": "", "kinds": ["1dratio"], },
            "ratio_xaxis_title_size": { "type": "Float", "desc": "X-axis label size", "default": None, "kinds": ["1dratio"], },
            "ratio_xaxis_title_offset": { "type": "FLoat", "desc": "X-axis label offset", "default": None, "kinds": ["1dratio"], },
            "ratio_label_size": { "type": "Float", "desc": "X-axis label size", "default": 0., "kinds": ["1dratio"], },
            "ratio_xaxis_label_offset": { "type": "Float", "desc": "offset to the x-axis labels (numbers)", "default": None, "kinds": ["1dratio"], },
            "ratio_yaxis_label_offset": { "type": "Float", "desc": "offset to the y-axis labels (numbers)", "default": None, "kinds": ["1dratio"], },
            "ratio_tick_length_scale": { "type": "Float", "desc": "Tick length scale of ratio pads", "default": 1.0, "kinds": ["1dratio"], },

            # Overall
            "title": { "type": "String", "desc": "plot title", "default": "", "kinds": ["1dratio","graph","2d"], },
            "draw_points": { "type": "Boolean", "desc": "draw points instead of fill", "default": False, "kinds": ["1d","1dratio"], },
            "draw_option_2d": { "type": "String", "desc": "hist draw option", "default": "colz", "kinds": ["2d"], },
            "bkg_err_fill_style": { "type": "Int", "desc": "Error shade draw style", "default": 1001, "kinds": ["1d", "1dratio"], },
            "bkg_err_fill_color": { "type": "Int", "desc": "Error shade color", "default": None, "kinds": ["1d", "1dratio"], },

            # CMS things
            "cms_label": {"type": "String", "desc": "E.g., 'Preliminary'; default hides label", "default": None, "kinds": ["1dratio","graph","2d"]},
            "lumi_value": {"type": "String", "desc": "E.g., 35.9; default hides lumi label", "default": "", "kinds": ["1dratio","graph","2d"]},
            "lumi_unit": {"type": "String", "desc": "Unit for lumi label", "default": "fb", "kinds": ["1dratio","graph","2d"]},

            # Misc
            "do_stack": { "type": "Boolean", "desc": "stack histograms", "default": True, "kinds": ["1dratio"], },
            "palette_name": { "type": "String", "desc": "color palette: 'default', 'rainbow', 'susy', etc.", "default": "default", "kinds": ["2d"], },
            "show_bkg_errors": { "type": "Boolean", "desc": "show error bar for background stack", "default": False, "kinds": ["1dratio"], },
            "show_bkg_smooth": { "type": "Boolean", "desc": "show smoothed background stack", "default": False, "kinds": ["1dratio"], },
            "bkg_sort_method": { "type": "Boolean", "desc": "how to sort background stack using integrals: 'unsorted', 'ascending', or 'descending'", "default": 'ascending', "kinds": ["1dratio"], },
            "no_ratio": { "type": "Boolean", "desc": "do not draw ratio plot", "default": False, "kinds": ["1dratio"], },
            "no_overflow": { "type": "Boolean", "desc": "do not draw overflow bins", "default": False, "kinds": ["1dratio"], },
            "stack_signal": { "type": "Boolean", "desc": "stack signal histograms", "default": False, "kinds": ["1dratio"], },

            "max_digits": { "type": "Int", "desc": "integer for max digits", "default": 5, "kinds" : ["1dratio", "graph", "2d"], },


            "bin_text_size": { "type": "Float", "desc": "size of text in bins (TH2::SetMarkerSize)", "default": 1.7, "kinds": ["2d"], },
            "bin_text_format": { "type": "String", "desc": "format string for text in TH2 bins", "default": ".1f", "kinds": ["2d"], },
            "bin_text_smart": { "type": "Boolean", "desc": "change bin text color for aesthetics", "default": False, "kinds": ["2d"], },
            "bin_text_format_smart": { "type": "String", "desc": "python-syntax format string for smart text in TH2 bins taking value and bin error", "default": "{0:.0f}#pm{1:.0f}", "kinds": ["2d"], },

            "hist_line_none": { "type": "Boolean", "desc": "No lines for histograms, only fill", "default": False, "kinds": ["1dratio"], },
            "hist_line_black": { "type": "Boolean", "desc": "Black lines for histograms", "default": False, "kinds": ["1dratio"], },
            "hist_disable_xerrors": { "type": "Boolean", "desc": "Disable the x-error bars on data for 1D hists", "default": True, "kinds": ["1dratio"], },

            "extra_text": { "type": "List", "desc": "list of strings for textboxes", "default": [], "kinds": [ "1dratio","graph"], },
            "extra_text_size": { "type": "Float", "desc": "size for extra text", "default": 0.04, "kinds": [ "1dratio","graph"], },
            "extra_text_xpos": { "type": "Float", "desc": "NDC x position (0 to 1) for extra text", "default": 0.3, "kinds": [ "1dratio","graph"], },
            "extra_text_ypos": { "type": "Float", "desc": "NDC y position (0 to 1) for extra text", "default": 0.87, "kinds": [ "1dratio","graph"], },

            "extra_lines": { "type": "List", "desc": "list of 4-tuples (x1,y1,x2,y2) for lines", "default": [], "kinds": [ "1dratio","graph"], },
            "no_overflow": {"type":"Boolean","desc":"Do not plot overflow bins","default": False, "kinds" : ["1dratio"],},

            # Fun
            "us_flag": { "type": "Boolean", "desc": "show the US flag in the corner", "default": False, "kinds": ["1dratio","graph","2d"], },
            "us_flag_coordinates": { "type": "List", "desc": "Specify flag location with (x pos, y pos, size)", "default": [0.68,0.96,0.06], "kinds": ["1dratio","graph","2d"], },

            # Output
            "output_name": { "type": "String", "desc": "output file name/path", "default": "plot.pdf", "kinds": ["1dratio","graph","2d"], },
            "output_ic": { "type": "Boolean", "desc": "run `ic` (imgcat) on output", "default": False, "kinds": ["1dratio","graph","2d"], },
            "output_jsroot": { "type": "Boolean", "desc": "output .json for jsroot", "default": False, "kinds": ["1dratio","graph","2d"], },
            "output_diff_previous": { "type": "Boolean", "desc": "diff the new output file with the previous", "default": False, "kinds": ["1dratio","graph","2d"], },

        }

        self.check_options()

    def usage(self):

        for key,obj in sorted(self.recognized_options.items()):
            default = obj["default"]
            desc = obj["desc"]
            typ = obj["type"]
            kinds = obj["kinds"]
            if self.kind and self.kind not in kinds: continue
            if type(default) is str: default = '"{}"'.format(default)
            print("* `{}` [{}]\n    {} (default: {})".format(key,typ,desc,default))

    def check_options(self):
        for name,val in self.options.items():
            if name not in self.recognized_options:
                print(">>> Option {} not in list of recognized options".format(name))
            else:
                obj = self.recognized_options[name]
                if self.kind not in obj["kinds"]:
                    print(">>> Option {} isn't declared to work with plot type of '{}'".format(name, self.kind))
                else:
                    pass
                    # print ">>> Carry on mate ... {} is fine".format(name)

    def __getitem__(self, key):
        if key in self.options:
            return self.options[key]
        else:
            if key in self.recognized_options:
                return self.recognized_options[key]["default"]
            else:
                print(">>> Hmm, can't find {} anywhere. Typo or intentional?".format(key))
                return None

    def get(self, key, default=None):
        val = self.__getitem__(key)
        if not val: return default
        else: return val

    def is_default(self, key):
        """
        returns True if user has not overriden this particular option
        """
        default = None
        if key in self.recognized_options:
            default = self.recognized_options[key]["default"]
        return (self.__getitem__(key) == default)

    def __setitem__(self, key, value):
        self.options[key] = value

    def __repr__(self):
        return str(self.options)

    def __contains__(self, key):
        return key in self.options

    def __add__(self, other):
        new_opts = {}
        new_opts.update(self.options)
        if type(other) is dict:
            new_opts.update(other)
        else:
            new_opts.update(other.options)
        return Options(new_opts,kind=self.kind)


def plot_graph(valpairs,colors=[],legend_labels=[],draw_styles=[],options={}):

    opts = Options(options, kind="graph")

    utils.set_style()

    c1 = r.TCanvas()
    if opts["canvas_width"] and opts["canvas_height"]:
        width = opts["canvas_width"]
        height = opts["canvas_height"]
        c1 = r.TCanvas("c1", "c1", width, height)
    legend = get_legend(opts)

    mg = r.TMultiGraph()
    drawopt = ""
    for parts in enumerate(valpairs):
        ipair = parts[0]
        rest = parts[1]
        typ = "xy"
        if len(rest) == 2:
            xs, ys = rest
            graph = r.TGraphAsymmErrors(len(xs), array('d',xs), array('d',ys))
            typ = "xy"
            legopt = "LP"
            drawopt = "ALP"
        elif len(rest) == 4:
            xs, ys, ylows, yhighs = rest
            zeros = array('d',[0. for _ in xs])
            graph = r.TGraphAsymmErrors(len(xs), array('d',xs), array('d',ys), zeros, zeros, array('d',ylows),array('d',yhighs))
            typ = "xyey"
            legopt, drawopt = "FLP","ALP3"
        elif len(rest) == 6:
            xs, ys, xlows, xhighs, ylows, yhighs = rest
            graph = r.TGraphAsymmErrors(len(xs), array('d',xs), array('d',ys), array('d',xlows), array('d',xhighs), array('d',ylows),array('d',yhighs))
            typ = "xyexey"
            legopt, drawopt = "FELP","ALP3"
        else:
            raise ValueError("don't recognize this format")

        if ipair < len(colors):
            graph.SetLineColor(colors[ipair])
            graph.SetLineWidth(4)
            graph.SetMarkerColor(colors[ipair])
            graph.SetMarkerSize(0.20*graph.GetLineWidth())
        if ipair < len(draw_styles):
            graph.SetLineStyle(draw_styles[ipair])
            graph.SetMarkerSize(0.)
        if ipair < len(legend_labels):
            legend.AddEntry(graph, legend_labels[ipair],legopt)

        if typ in ["xyey","xyexey"]:
            graph.SetFillColorAlpha(graph.GetLineColor(),0.25)

        mg.Add(graph,drawopt)

    mg.SetTitle(opts["title"])

    mg.Draw("A")

    if legend_labels: legend.Draw()

    draw_cms_lumi(c1, opts)
    handle_axes(c1, mg, opts)
    draw_extra_stuff(c1, opts)
    save(c1, opts)

    return c1

def get_legend(opts):
    x1,y1,x2,y2 = opts["legend_coordinates"]
    legend_alignment = opts["legend_alignment"]
    height = 0.2
    width = 0.3
    if "bottom" in legend_alignment: y1, y2 = 0.18, 0.18+height
    if "top" in legend_alignment: y1, y2 = 0.67, 0.67+height
    if "left" in legend_alignment: x1, x2 = 0.18, 0.18+width
    if "right" in legend_alignment: x1, x2 = 0.63, 0.63+width

    # scale width and height of legend keeping the sides
    # closest to the plot edges the same (so we expand/contact the legend inwards)
    scalex = opts["legend_scalex"]
    scaley = opts["legend_scaley"]
    toshift_x = (1.-scalex)*(x2-x1)
    toshift_y = (1.-scaley)*(y2-y1)
    if 0.5*(x1+x2) > 0.5: # second half, so keep the right side stationary
        x1 += toshift_x
    else: # keep left side pinned
        x2 -= toshift_x
    if 0.5*(y1+y2) > 0.5: # upper half, so keep the upper side stationary
        y1 += toshift_y
    else: # keep bottom side pinned
        y2 -= toshift_y

    legend = r.TLegend(x1,y1,x2,y2)


    if opts["legend_opacity"] == 1:
        legend.SetFillStyle(0)
    else:
        legend.SetFillColorAlpha(r.kWhite,1.0-opts["legend_opacity"])
    if opts["legend_border"]:
        legend.SetBorderSize(1)
    else:
        legend.SetBorderSize(0)
    legend.SetTextFont(42)
    legend.SetNColumns(opts["legend_ncolumns"])
    if opts["legend_column_separation"]: legend.SetColumnSeparation(opts["legend_column_separation"])
    return legend


def plot_hist(data=None,bgs=[],legend_labels=[],colors=[],sigs=[],sig_labels=[],syst=None,options={},_persist=[],marker_shapes = []):

    opts = Options(options, kind="1dratio")

    style = utils.set_style()

    c1 = r.TCanvas()
    if opts["canvas_width"] and opts["canvas_height"]:
        width = opts["canvas_width"]
        height = opts["canvas_height"]
        c1 = r.TCanvas("c1", "c1", width, height)
    _persist.append(c1) # need this to avoid segfault with garbage collection

    has_data = data and data.InheritsFrom(r.TH1.Class())
    do_ratio = (has_data or opts["ratio_numden_indices"]) and not opts["no_ratio"]
    if do_ratio:
        pad_main = r.TPad("pad1","pad1",0.0,opts["canvas_main_y1"],1.0,1.0)
        if opts["canvas_main_topmargin"]: pad_main.SetTopMargin(opts["canvas_main_topmargin"])
        if opts["canvas_main_rightmargin"]: pad_main.SetRightMargin(opts["canvas_main_rightmargin"])
        if opts["canvas_main_bottommargin"]: pad_main.SetBottomMargin(opts["canvas_main_bottommargin"])
        if opts["canvas_main_leftmargin"]: pad_main.SetLeftMargin(opts["canvas_main_leftmargin"])
        if opts["canvas_tick_one_side"]: pad_main.SetTicks(0, 0)
        pad_ratio = r.TPad("pad2","pad2",0.0, 0.00, 1.0, opts["canvas_ratio_y2"])
        if opts["canvas_ratio_topmargin"]: pad_ratio.SetTopMargin(opts["canvas_ratio_topmargin"])
        if opts["canvas_ratio_rightmargin"]: pad_ratio.SetRightMargin(opts["canvas_ratio_rightmargin"])
        if opts["canvas_ratio_bottommargin"]: pad_ratio.SetBottomMargin(opts["canvas_ratio_bottommargin"])
        if opts["canvas_ratio_leftmargin"]: pad_ratio.SetLeftMargin(opts["canvas_ratio_leftmargin"])
        if opts["canvas_tick_one_side"]: pad_ratio.SetTicks(0, 0)
        pad_main.Draw()
        pad_ratio.Draw()
    else:
        pad_main = r.TPad("pad1","pad1",0.,0.,1.,1.)
        if opts["canvas_main_topmargin"]: pad_main.SetTopMargin(opts["canvas_main_topmargin"])
        if opts["canvas_main_rightmargin"]: pad_main.SetRightMargin(opts["canvas_main_rightmargin"])
        if opts["canvas_main_bottommargin"]: pad_main.SetBottomMargin(opts["canvas_main_bottommargin"])
        if opts["canvas_main_leftmargin"]: pad_main.SetLeftMargin(opts["canvas_main_leftmargin"])
        if opts["canvas_tick_one_side"]: pad_main.SetTicks(0, 0)
        pad_main.Draw()

    pad_main.cd()

    # sort backgrounds, but make sure all parameters have same length
    if len(colors) < len(bgs):
        print(">>> Provided only {} colors for {} backgrounds, so using default palette".format(len(colors),len(bgs)))
        colors = utils.get_default_colors()
        if len(colors) < len(bgs):
            print(">>> Only {} default colors for {} backgrounds, so {} of them will be black.".format(len(colors),len(bgs),len(bgs)-len(colors)))
            for ibg in range(len(bgs)-len(colors)):
                colors.append(r.kBlack)

    if opts["draw_points"] and len(marker_shapes) < len(bgs):
        print(">>> Provided only {} marker shapes for {} point backgrounds, so using default shape collection".format(len(marker_shapes),len(bgs)))
        marker_shapes = utils.get_default_marker_shapes()


    if len(legend_labels) < len(bgs):
        print(">>> Provided only {} legend_labels for {} backgrounds, so using hist titles".format(len(legend_labels),len(bgs)))
        for ibg in range(len(bgs)-len(legend_labels)):
            legend_labels.append(bgs[ibg].GetTitle())
    sort_methods = {
            "descending": lambda x: -x[0].Integral(),
            "ascending": lambda x: x[0].Integral(), # highest integral on top of stack
            "unsorted": lambda x: 1, # preserve original ordering
            }
    which_method = opts["bkg_sort_method"]
    original_index_mapping = range(len(bgs))
    bgs, colors, legend_labels, original_index_mapping = list(zip(*sorted(zip(bgs,colors,legend_labels,original_index_mapping), key=sort_methods[which_method])))
    # map original indices of bgs to indices of sorted bgs
    original_index_mapping = { oidx: nidx for oidx,nidx in zip(original_index_mapping,list(range(len(bgs)))) }
    list(map(lambda x: x.Sumw2(), bgs))
    if not opts["no_overflow"]:
        list(map(utils.move_in_overflows, bgs))

    legend = get_legend(opts)

    if has_data:
        if not opts["no_overflow"]:
            utils.move_in_overflows(data)
        data.SetMarkerStyle(20)
        data.SetMarkerColor(r.kBlack)
        data.SetLineWidth(2)
        data.SetMarkerSize(0.8)
        data.SetLineColor(r.kBlack)
        legend.AddEntry(data, opts["legend_datalabel"], "LPE" if not opts["hist_disable_xerrors"] else "PE")

    stack = r.THStack("stack", "stack")
    for ibg,bg in enumerate(bgs):
        if ibg < len(colors):
            bg.SetLineColor(r.TColor.GetColorDark(colors[ibg]))
            if opts["hist_line_black"]:
                bg.SetLineColor(r.kBlack)
            bg.SetLineWidth(1)
            bg.SetMarkerColor(colors[ibg])
            bg.SetMarkerSize(0)
            bg.SetFillColorAlpha(colors[ibg],1 if opts["do_stack"] else 0.4)
            if opts["draw_points"]:
                bg.SetLineWidth(3)
                #bg.SetMarkerStyle(20)
                bg.SetMarkerStyle(marker_shapes[ibg % len(marker_shapes)])
                bg.SetLineColor(colors[ibg])
                bg.SetMarkerColor(colors[ibg])
                bg.SetMarkerSize(0.8)
            if opts["hist_line_none"]:
                bg.SetLineWidth(0)
        if ibg < len(legend_labels):
            entry_style = "F"
            if opts["draw_points"]:
                entry_style = "LPE"
            legend.AddEntry(bg, legend_labels[ibg], entry_style)
        stack.Add(bg)

    if opts["stack_signal"]:
        for isig_raw,sig in enumerate(sigs):
            isig = isig_raw + len(bgs) - 1
            print isig, isig_raw
            if isig < len(colors):
                # sig.SetLineColor(r.TColor.GetColorDark(colors[isig]))
                sig.SetLineColor(colors[isig])
                # if opts["hist_line_black"]:
                #     sig.SetLineColor(r.kBlack)
                # sig.SetLineWidth(1)
                sig.SetLineWidth(0)
                sig.SetMarkerColor(colors[isig])
                sig.SetMarkerSize(0)
                sig.SetFillColorAlpha(colors[isig],1 if opts["do_stack"] else 0.4)
                if opts["draw_points"]:
                    sig.SetLineWidth(3)
                    #sig.SetMarkerStyle(20)
                    sig.SetMarkerStyle(marker_shapes[isig % len(marker_shapes)])
                    sig.SetLineColor(colors[isig])
                    sig.SetMarkerColor(colors[isig])
                    sig.SetMarkerSize(0.8)
                if opts["hist_line_none"]:
                    sig.SetLineWidth(0)
            if isig < len(legend_labels):
                entry_style = "F"
                if opts["draw_points"]:
                    entry_style = "LPE"
                legend.AddEntry(sig, legend_labels[isig], entry_style)
            stack.Add(sig)
        sigs = []

    stack.SetTitle(opts["title"])

    drawopt = "nostackhist"
    extradrawopt = ""
    if opts["do_stack"]: drawopt = "hist"
    if opts["show_bkg_errors"]: drawopt += "e1"
    if opts["show_bkg_smooth"]: drawopt += "C"
    if opts["draw_points"]:
        drawopt += "PE"
        if opts["hist_disable_xerrors"]:
            drawopt += "X0"

    if opts["hist_disable_xerrors"]:
        extradrawopt += "X0"

    # When using  stack.GetHistogram().GetMaximum() to get ymax, this screws
    # up CMS Lumi drawing, but we can't just assume that get_stack_maximum
    # returns the actual maximum (even though that's what we set it to!) because
    # thstack multiplies the max by 1.05??? Odd
    # So here, we take into account that scaling for the rest of this function
    ymin, ymax = 0., utils.get_stack_maximum(data,stack,opts)
    stack.SetMaximum(ymax)
    stack.Draw(drawopt)
    ymax = 1.05*ymax if opts["do_stack"] else 1.00*ymax
    if opts["yaxis_range"]:
        stack.SetMinimum(opts["yaxis_range"][0])
        stack.SetMaximum(opts["yaxis_range"][1])
        ymin, ymax = opts["yaxis_range"]

    if syst:
        # Turn relative bin errors from syst into drawable histogram
        bgs_syst = syst.Clone("bgs_syst")
        bgs_syst.Reset()
        for hist in bgs:
            bgs_syst.Add(hist)
        for ibin in range(0,bgs_syst.GetNbinsX()+2):
            # Set the bin content of the systematic band to the total of the backgrounds
            # and the error to the actual value of the systematic histogram
            bgs_syst.SetBinContent(ibin, bgs_syst.GetBinContent(ibin))
            bgs_syst.SetBinError(ibin, syst.GetBinContent(ibin))
        if not opts["no_overflow"]: utils.move_in_overflows(bgs_syst)
        bgs_syst.SetMarkerSize(0)
        bgs_syst.SetMarkerColorAlpha(r.kWhite,0.)
        if not opts["bkg_err_fill_color"]: bgs_syst.SetFillColorAlpha(r.kGray+2,0.4)
        else: bgs_syst.SetFillColorAlpha(opts["bkg_err_fill_color"],0.4)
        bgs_syst.SetFillStyle(opts["bkg_err_fill_style"])

        # Compute the systematics band in the ratio, to be drawn later in the ratio
        all_bgs = syst.Clone("all_bgs")
        all_bgs.Reset()
        for hist in bgs:
            all_bgs.Add(hist)
        ratio_syst = bgs_syst.Clone("ratio_syst")
        ratio_syst.Sumw2()
        ratio_syst.Divide(all_bgs)
        ratio_syst.SetFillColorAlpha(r.kGray+2,0.4)
        if not opts["bkg_err_fill_color"]: ratio_syst.SetFillColorAlpha(r.kGray+2,0.4)
        else: ratio_syst.SetFillColorAlpha(opts["bkg_err_fill_color"],0.4)
        ratio_syst.SetFillStyle(opts["bkg_err_fill_style"])

        # Draw the main band in the main pad
        bgs_syst.Draw("E2 SAME")

    if has_data:
        data.Draw("samepe"+extradrawopt)

    if sigs:
        if not opts["no_overflow"]:
            map(utils.move_in_overflows, sigs)
        colors = cycle([r.kRed, r.kBlue, r.kOrange-4, r.kTeal-5])
        if len(sig_labels) < len(sigs):
            sig_labels = [sig.GetTitle() for sig in sigs]
        for hsig,signame,color in zip(sigs, sig_labels,colors):
            hsig.SetMarkerStyle(1) # 2 has errors
            hsig.SetMarkerColor(color)
            hsig.SetLineWidth(3)
            hsig.SetMarkerSize(0.8)
            hsig.SetLineColor(color)
            legend.AddEntry(hsig,signame, "LP")
            hsig.Draw("samehist")



    draw_cms_lumi(pad_main, opts)
    handle_axes(pad_main, stack, opts)
    draw_extra_stuff(pad_main, opts)

    if opts["legend_smart"] and not opts["yaxis_log"]:
        utils.smart_legend(legend, bgs, data=data, ymin=ymin, ymax=ymax, opts=opts)

    if opts["legend_rounded"]:
        legend.SetFillColor(0)
        legend.SetLineWidth(0)
        legend.Draw()
        x1, y1, x2, y2 = legend.GetX1(), legend.GetY1(), legend.GetX2(), legend.GetY2()
        radius = 0.010
        utils.draw_shadow_rounded_box(x1,y1,x2,y2,radius,color=r.kGray+1,alpha=0.9)
    else:
        legend.Draw()

    if opts["legend_percentageinbox"]:
        draw_percentageinbox(legend, bgs, sigs, opts, has_data=has_data)

    if do_ratio:
        pad_ratio.cd()

        if opts["ratio_numden_indices"]:
            orig_num_idx, orig_den_idx = opts["ratio_numden_indices"]
            numer = bgs[original_index_mapping[orig_num_idx]].Clone("numer")
            denom = bgs[original_index_mapping[orig_den_idx]].Clone("denom")
            if opts.is_default("ratio_name"):
                opts["ratio_name"] = "{}/{}".format(legend_labels[original_index_mapping[orig_num_idx]],legend_labels[original_index_mapping[orig_den_idx]])
        else:
            # construct numer and denom to be used everywhere
            numer = data.Clone("numer")
            denom = bgs[0].Clone("sumbgs")
            denom.Reset()
            denom = sum(bgs,denom)

        ratio = numer.Clone("ratio")
        if opts["ratio_binomial_errors"]:
            ratio.Divide(numer,denom,1,1,"b")
        else:
            ratio.Divide(denom)

        if opts["ratio_pull"]:
            for ibin in range(1,ratio.GetNbinsX()+1):
                ratio_val = ratio.GetBinContent(ibin)
                numer_val = numer.GetBinContent(ibin)
                numer_err = numer.GetBinError(ibin)
                denom_val = denom.GetBinContent(ibin)
                denom_err = denom.GetBinError(ibin)
                if syst:
                    # when doing a pull, the denominator is usually MC
                    # which is carries the syst error we need to add in
                    denom_err = (denom_err**2. + bgs_syst.GetBinError(ibin)**2.)**0.5
                # gaussian pull
                pull = (ratio_val-1.)/((numer_err**2.+denom_err**2.)**0.5)
                if numer_val > 1e-6:
                    # more correct pull, but is inf when 0 data, so fall back to gaus pull in that case
                    pull = r.RooStats.NumberCountingUtils.BinomialObsZ(numer_val,denom_val,denom_err/denom_val);
                ratio.SetBinContent(ibin,pull)
                ratio.SetBinError(ibin,0.)
            opts["ratio_range"] = [-3.0,3.0]
            opts["ratio_ndivisions"] = 208
            opts["ratio_horizontal_lines"] = [-1.,0.,1.]

        ratio.Draw("axis")

        if syst and not opts["ratio_pull"]:
            ratio_syst.Draw("E2 SAME")

        if opts["ratio_pull"] and opts["ratio_pull_numbers"]:
            t = r.TLatex()
            t.SetTextAlign(22)
            t.SetTextFont(42)
            t.SetTextColor(r.kBlack)
            t.SetTextSize(0.1)
            for ibin in range(1,ratio.GetNbinsX()+1):
                yval = ratio.GetBinContent(ibin)
                xval = ratio.GetBinCenter(ibin)
                yvaldraw = yval
                if yvaldraw > 2.35: yvaldraw -= 0.6
                else: yvaldraw += 0.6
                if abs(yval) > 2.: t.SetTextColor(r.kRed+1)
                elif abs(yval) > 1.: t.SetTextColor(r.kOrange+1)
                else: t.SetTextColor(r.kBlack)
                if abs(yval) > 3.: continue
                t.DrawLatex(xval,yvaldraw,"{:.1f}".format(yval))

        do_style_ratio(ratio, opts, pad_ratio)
        ratio.Draw("same PE"+extradrawopt)


        line = r.TLine()
        line.SetLineColor(r.kGray+2)
        line.SetLineWidth(1)
        for yval in opts["ratio_horizontal_lines"]:
            line.DrawLine(ratio.GetXaxis().GetBinLowEdge(1),yval,ratio.GetXaxis().GetBinUpEdge(ratio.GetNbinsX()),yval)

        if opts["ratio_chi2prob"] or (opts["ratio_pull"] and opts["ratio_pull_numbers"]):
            oldpad = r.gPad
            c1.cd()
            t = r.TLatex()
            t.SetTextAlign(22)
            t.SetTextFont(42)
            t.SetTextColor(r.kBlack)
            t.SetTextSize(0.03)
            yloc = pad_ratio.GetAbsHNDC()
            to_show = ""
            if opts["ratio_chi2prob"]:
                chi2 = 0.
                ndof = 0
                for ibin in range(1,ratio.GetNbinsX()+1):
                    err2 = ratio.GetBinError(ibin)**2.
                    if err2 < 1.e-6: continue
                    if syst:
                        err2 += ratio_syst.GetBinError(ibin)**2.
                    val = ratio.GetBinContent(ibin)
                    chi2 += (val-1.)**2./err2
                    ndof += 1
                prob = r.TMath.Prob(chi2,ndof-1)
                to_show = "P(#chi^{{2}}/ndof) = {:.2f}".format(prob)
            if opts["ratio_pull"] and opts["ratio_pull_numbers"]:
                mean, sigma, vals = utils.get_mean_sigma_1d_yvals(ratio)
                to_show = "Pulls: #mu = {:.2f}, #sigma = {:.2f}".format(mean,sigma)
            t.DrawLatexNDC(0.5,yloc+0.01,to_show)
            oldpad.cd()

        pad_main.cd()

    save(c1, opts)

    return c1


def do_style_ratio(ratio, opts, tpad):
    if opts["ratio_range"][1] <= opts["ratio_range"][0]:
        # if high <= low, compute range automatically (+-3 sigma interval)
        mean, sigma, vals = utils.get_mean_sigma_1d_yvals(ratio)
        low = max(mean-3*sigma,min(vals))-sigma/1e3
        high = min(mean+3*sigma,max(vals))+sigma/1e3
        opts["ratio_range"] = [low,high]
    ratio.SetMarkerStyle(20)
    ratio.SetMarkerSize(0.8)
    ratio.SetLineWidth(2)
    ratio.SetTitle("")
    if opts["xaxis_log"]:
        tpad.SetLogx(1)
        ratio.GetXaxis().SetMoreLogLabels(opts["xaxis_moreloglabels"])
        ratio.GetXaxis().SetNoExponent(opts["xaxis_noexponents"])
    ratio.GetYaxis().SetTitle(opts["ratio_name"])
    if opts["ratio_name_offset"]: ratio.GetYaxis().SetTitleOffset(opts["ratio_name_offset"])
    if opts["ratio_name_size"]: ratio.GetYaxis().SetTitleSize(opts["ratio_name_size"])
    ratio.GetYaxis().SetNdivisions(opts["ratio_ndivisions"])
    ratio.GetYaxis().SetLabelSize(0.13)
    if opts["ratio_yaxis_label_offset"]: ratio.GetYaxis().SetLabelOffset(opts["ratio_yaxis_label_offset"])

    if opts["xaxis_range"]: ratio.GetXaxis().SetRangeUser(*opts["xaxis_range"])
    ratio.GetYaxis().SetRangeUser(*opts["ratio_range"])
    ratio.GetXaxis().SetLabelSize(opts["ratio_label_size"])
    ratio.GetXaxis().SetTitle(opts["ratio_xaxis_title"])
    if opts["ratio_xaxis_title_size"]: ratio.GetXaxis().SetTitleSize(opts["ratio_xaxis_title_size"])
    if opts["ratio_xaxis_title_offset"] :ratio.GetXaxis().SetTitleOffset(opts["ratio_xaxis_title_offset"])
    if opts["ratio_xaxis_label_offset"]: ratio.GetXaxis().SetLabelOffset(opts["ratio_xaxis_label_offset"])
    ratio.GetXaxis().SetTickSize(0.06 * opts["ratio_tick_length_scale"])
    ratio.GetYaxis().SetTickSize(0.03 * opts["ratio_tick_length_scale"])

def draw_percentageinbox(legend, bgs, sigs, opts, has_data=False):
    t = r.TLatex()
    t.SetTextAlign(22)
    t.SetTextFont(42)
    t.SetTextColor(r.kWhite)
    info = utils.get_legend_marker_info(legend)
    t.SetTextSize(info["label_height"])
    all_entries = list(bgs) + list(sigs)
    total_integral = sum(bg.Integral() for bg in bgs)
    # we want the number to be centered, without the % symbol, so nudge the percentage text right a bit
    nudge_right = info["box_width"]*0.15
    if info["draw_vertical"]:
        t.SetTextAngle(90)
    else:
        t.SetTextAngle(0)
    for icoord, (xndc, yndc) in enumerate(info["coords"]):
        # if we have data, skip it and restart numbering from 0
        if has_data:
            if icoord == 0: continue
            icoord -= 1
        if icoord >= len(bgs): continue # don't do signals
        bg = all_entries[icoord]
        percentage = int(100.0*bg.Integral()*(1.+1.e-6)/total_integral)
        color = r.gROOT.GetColor(bg.GetFillColor())
        red = color.GetRed()
        green = color.GetGreen()
        blue = color.GetBlue()
        # same as utils.compute_darkness (https://root.cern.ch/doc/master/TColor_8h_source.html#l00027)
        # without the color.GetAlpha(), which is there because effective luminance is higher if there's transparency
        darkness = 1.-color.GetGrayscale()/color.GetAlpha()
        if darkness < 0.5:
            t.SetTextColor(r.kBlack)
        else:
            t.SetTextColor(r.kWhite)
        # t.SetTextColor(r.TColor.GetColorDark(bg.GetFillColor()))
        t.DrawLatexNDC(xndc+nudge_right,yndc,"%i#scale[0.5]{#lower[-0.2]{%%}}" % (percentage))


def handle_axes(c1, obj, opts):

    obj.GetXaxis().SetTitle(opts["xaxis_label"])
    if opts["xaxis_range"]: obj.GetXaxis().SetRangeUser(*opts["xaxis_range"])
    if opts["xaxis_log"]:
        c1.SetLogx(1)
        obj.GetXaxis().SetMoreLogLabels(opts["xaxis_moreloglabels"])
        obj.GetXaxis().SetNoExponent(opts["xaxis_noexponents"])
    if opts["xaxis_label_size_scale"]: obj.GetXaxis().SetLabelSize(obj.GetXaxis().GetLabelSize() * opts["xaxis_label_size_scale"])
    if opts["xaxis_label_offset_scale"]: obj.GetXaxis().SetLabelOffset(obj.GetXaxis().GetLabelOffset() * opts["xaxis_label_offset_scale"])
    if opts["xaxis_tick_length_scale"]: obj.GetXaxis().SetTickLength(obj.GetXaxis().GetTickLength() * opts["xaxis_tick_length_scale"])
    if opts["xaxis_title_size"]: obj.GetXaxis().SetTitleSize(opts["xaxis_title_size"])
    if opts["xaxis_title_offset"]: obj.GetXaxis().SetTitleOffset(opts["xaxis_title_offset"])

    obj.GetYaxis().SetTitle(opts["yaxis_label"])
    if opts["yaxis_range"]:
        obj.GetYaxis().SetRangeUser(*opts["yaxis_range"])
    if opts["yaxis_log"]:
        c1.SetLogy(1)
        obj.GetYaxis().SetMoreLogLabels(opts["yaxis_moreloglabels"])
        obj.GetYaxis().SetNoExponent(opts["yaxis_noexponents"])
    if opts["yaxis_label_size_scale"]: obj.GetYaxis().SetLabelSize(obj.GetYaxis().GetLabelSize() * opts["yaxis_label_size_scale"])
    if opts["yaxis_label_offset_scale"]: obj.GetYaxis().SetLabelOffset(obj.GetYaxis().GetLabelOffset() * opts["yaxis_label_offset_scale"])
    if opts["yaxis_tick_length_scale"]: obj.GetYaxis().SetTickLength(obj.GetYaxis().GetTickLength() * opts["yaxis_tick_length_scale"])
    if opts["yaxis_title_size"]: obj.GetYaxis().SetTitleSize(opts["yaxis_title_size"])
    if opts["yaxis_title_offset"]: obj.GetYaxis().SetTitleOffset(opts["yaxis_title_offset"])
    if opts["yaxis_exponent_offset"] or opts["yaxis_exponent_vertical_offset"]: r.TGaxis.SetExponentOffset(opts["yaxis_exponent_offset"], opts["yaxis_exponent_vertical_offset"])
    if opts["yaxis_ndivisions"]: obj.GetYaxis().SetNdivisions(opts["yaxis_ndivisions"])
    if opts["xaxis_ndivisions"]: obj.GetXaxis().SetNdivisions(opts["xaxis_ndivisions"])
    if opts["max_digits"]: r.TGaxis.SetMaxDigits(opts["max_digits"])
    if hasattr(obj, "GetZaxis"):
        obj.GetZaxis().SetTitle(opts["zaxis_label"])
        if opts["zaxis_range"]: obj.GetZaxis().SetRangeUser(*opts["zaxis_range"])
        if opts["zaxis_log"]:
            c1.SetLogz(1)
            obj.GetZaxis().SetMoreLogLabels(opts["zaxis_moreloglabels"])
            obj.GetZaxis().SetNoExponent(opts["zaxis_noexponents"])


def plot_hist_2d(hist,options={}):

    opts = Options(options, kind="2d")

    style = utils.set_style_2d()

    utils.set_palette(style, opts["palette_name"])

    c1 = r.TCanvas()
    if opts["canvas_width"] and opts["canvas_height"]:
        width = opts["canvas_width"]
        height = opts["canvas_height"]
        c1 = r.TCanvas("c1", "c1", width, height)

    hist.Draw(opts["draw_option_2d"])

    hist.SetTitle(opts["title"])

    hist.SetMarkerSize(opts["bin_text_size"])
    style.SetPaintTextFormat(opts["bin_text_format"])

    if opts["bin_text_smart"]:
        utils.draw_smart_2d_bin_labels(hist, opts)

    draw_cms_lumi(c1, opts)
    handle_axes(c1, hist, opts)
    draw_extra_stuff(c1, opts)
    save(c1, opts)

def draw_cms_lumi(c1, opts, _persist=[]):
    t = r.TLatex()
    t.SetTextAlign(11) # align bottom left corner of text
    t.SetTextColor(r.kBlack)
    t.SetTextSize(0.04)
    # get top left corner of current pad, and nudge up the y coord a bit
    xcms = r.gPad.GetX1() + r.gPad.GetLeftMargin()
    ycms = r.gPad.GetY2() - r.gPad.GetTopMargin() + 0.01
    xlumi = r.gPad.GetX2() - r.gPad.GetRightMargin()
    cms_label = opts["cms_label"]
    lumi_value = str(opts["lumi_value"])
    lumi_unit = opts["lumi_unit"]
    energy = 13
    if cms_label is not None:
        t.DrawLatexNDC(xcms,ycms,"#scale[1.25]{#font[61]{CMS}} #scale[1.1]{#font[52]{%s}}" % cms_label)
    if lumi_value:
        t.SetTextSize(0.04)
        t.SetTextAlign(31) # align bottom right
        t.SetTextFont(42) # align bottom right
        t.DrawLatexNDC(xlumi,ycms,"{lumi_str} {lumi_unit}^{{-1}} ({energy} TeV)".format(energy=energy, lumi_str=lumi_value, lumi_unit=lumi_unit))
    _persist.append(t)

def draw_extra_stuff(c1, opts):

    if opts["us_flag"]:
        utils.draw_flag(c1,*opts["us_flag_coordinates"])

    if opts["extra_text"]:
        t = r.TLatex()
        t.SetNDC()
        t.SetTextAlign(12)
        t.SetTextFont(42)
        t.SetTextColor(r.kBlack)
        # t.SetTextSize(0.04)
        t.SetTextSize(opts["extra_text_size"])
        for itext, text in enumerate(opts["extra_text"]):
            t.DrawLatex(opts["extra_text_xpos"],opts["extra_text_ypos"]-itext*5./4*t.GetTextSize(),text)

    if opts["extra_lines"]:
        for iline,lcoords in enumerate(opts["extra_lines"]):
            if len(lcoords) != 4:
                print(">>> Malformed line coordinates (length should be 4 but is {})".format(len(lcoords)))
                continue

            line = r.TLine()
            line.SetLineColor(r.kGray+2)
            line.SetLineWidth(1)
            line.SetLineStyle(2)
            line.DrawLine(*lcoords)


def save(c1, opts):

    fname = opts["output_name"]
    dirname = os.path.dirname(fname)
    if dirname and not os.path.isdir(dirname):
        print(">>> Plot should go inside {}/, but it doesn't exist.".format(dirname))
        print(">>> Instead of crashing, I'll do you a solid and make it".format(dirname))
        os.system("mkdir -p {}".format(dirname))

    orig_fname = None
    if opts["output_diff_previous"]:
        if os.path.exists(fname):
            orig_fname = fname.replace(".pdf","_orig.pdf")
            os.system("mv {} {}".format(fname, orig_fname))

    print(">>> Saving {}".format(fname))
    c1.SaveAs(fname)

    if opts["output_diff_previous"]:
        fname_diff = "diff.png"
        utils.diff_images(orig_fname,fname, output=fname_diff)
        os.system("ic {}".format(fname_diff))
        if orig_fname:
            os.system("rm {}".format(orig_fname))

    if opts["output_ic"]:
        os.system("ic {}".format(fname))
    if opts["output_jsroot"]:
        r.TBufferJSON.ExportToFile("{}.json".format(fname.rsplit(".",1)[0]),c1)

if __name__ == "__main__":

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

    plot_hist(
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
                "ratio_pull": True,
                "hist_disable_xerrors": True,
                "ratio_chi2prob": True,
                "output_name": "test1.pdf",
                "legend_percentageinbox": True,
                "cms_label": "Preliminary",
                "lumi_value": "-inf",
                "output_ic": True,
                "us_flag": True,
                # "output_jsroot": True,
                }
            )




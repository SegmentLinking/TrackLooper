# plottery
## Introduction
A ROOT plotter that makes you feel like a millionaire ("lottery", get it?). Some interesting features include
* Percentages in marker boxes in the legend
* Automatic legend placement to prevent overlaps
* A US flag stamp
* Chi2 probability calculation for ratio pads
* Automatic range for ratio pad
* Pull distribution markers have numbers representing n-sigma, and have mean/standard deviation shown

...and it supports
* TH1
* TGraph(AsymmErrors)
* TH2

A list of options is shown below, and there is a "self-documenting" class containing all of them in the source.

## Instructions
* You need ROOT
* Go to the parent directory
* Modify and execute `python3 -m plottery.examples` to see some examples (which get put into `plottery/examples/`)

## Design philosophies
* Generally, plotting scripts grow endlessly to encompass use-cases that crop up over the years.
In principle, plottery should comfortably handle 95% of use-cases to prevent the size from blowing up.
* Plottery is only a plotter. It is not a histogram-adder, a re-binner, a TTree looper, etc.
Features like that should be written around plottery, not within it.
* Options should be functionally grouped (e.g., options applying to legend should start with `legend_`, options
applying to the x-axis should start with `xaxis_`). See the list of supported options below for an idea. Also, this
makes it so printing out options alphabetically retains a logical grouping.
* Every commit :blue_book: should contain at least :one: emoji representing the theme of the commit. For example,
:new: can be used with a completely new feature, :beetle: for bugs, :question: if you're unsure if something is broken
by the commit, :anguished: to express frustration, and :poop: for those super-special commits.

## List of supported options
Note that the following list was obtained _verbatim_ with
```bash
python -c "__import__('plottery').Options().usage()"
```
To update the README in Vim go to the below line and type: jdGyy:@"<CR>
:r!python -c "__import__('plottery').Options().usage()"
* `bin_text_format` [String]
    format string for text in TH2 bins (default: ".1f")
* `bin_text_format_smart` [String]
    python-syntax format string for smart text in TH2 bins taking value and bin error (default: "{0:.0f}#pm{1:.0f}")
* `bin_text_size` [Float]
    size of text in bins (TH2::SetMarkerSize) (default: 1.7)
* `bin_text_smart` [Boolean]
    change bin text color for aesthetics (default: False)
* `bkg_err_fill_color` [Int]
    Error shade color (default: None)
* `bkg_err_fill_style` [Int]
    Error shade draw style (default: 1001)
* `bkg_sort_method` [Boolean]
    how to sort background stack using integrals: 'unsorted', 'ascending', or 'descending' (default: "ascending")
* `canvas_height` [Int]
    height of TCanvas in pixel (default: None)
* `canvas_main_bottommargin` [Float]
    ratio plot bottom margin (default: None)
* `canvas_main_leftmargin` [Float]
    ratio plot left margin (default: None)
* `canvas_main_rightmargin` [Float]
    ratio plot right margin (default: None)
* `canvas_main_topmargin` [Float]
    ratio plot top margin (default: None)
* `canvas_main_y1` [Float]
    main plot tpad y1 (default: 0.18)
* `canvas_ratio_bottommargin` [Float]
    ratio plot bottom margin (default: None)
* `canvas_ratio_leftmargin` [Float]
    ratio plot left margin (default: None)
* `canvas_ratio_rightmargin` [Float]
    ratio plot right margin (default: None)
* `canvas_ratio_topmargin` [Float]
    ratio plot top margin (default: None)
* `canvas_ratio_y2` [Float]
    ratio tpad y2 (default: 0.19)
* `canvas_tick_one_side` [Boolean]
    ratio plot left margin (default: False)
* `canvas_width` [Int]
    width of TCanvas in pixel (default: None)
* `cms_label` [String]
    E.g., 'Preliminary'; default hides label (default: None)
* `do_stack` [Boolean]
    stack histograms (default: True)
* `draw_option_2d` [String]
    hist draw option (default: "colz")
* `draw_points` [Boolean]
    draw points instead of fill (default: False)
* `extra_lines` [List]
    list of 4-tuples (x1,y1,x2,y2) for lines (default: [])
* `extra_text` [List]
    list of strings for textboxes (default: [])
* `extra_text_size` [Float]
    size for extra text (default: 0.04)
* `extra_text_xpos` [Float]
    NDC x position (0 to 1) for extra text (default: 0.3)
* `extra_text_ypos` [Float]
    NDC y position (0 to 1) for extra text (default: 0.87)
* `hist_disable_xerrors` [Boolean]
    Disable the x-error bars on data for 1D hists (default: True)
* `hist_line_black` [Boolean]
    Black lines for histograms (default: False)
* `hist_line_none` [Boolean]
    No lines for histograms, only fill (default: False)
* `legend_alignment` [String]
    easy alignment of TLegend. String containing two words from: bottom, top, left, right (default: "")
* `legend_border` [Boolean]
    show legend border? (default: True)
* `legend_column_separation` [Float]
    column separation size (default: None)
* `legend_coordinates` [List]
    4 elements specifying TLegend constructor coordinates (default: [0.63, 0.67, 0.93, 0.87])
* `legend_datalabel` [String]
    label for the data histogram in the legend (default: "Data")
* `legend_ncolumns` [Int]
    number of columns in the legend (default: 1)
* `legend_opacity` [Float]
    from 0 to 1 representing the opacity of the TLegend white background (default: 0.5)
* `legend_percentageinbox` [Boolean]
    show relative process contributions as %age in the legend thumbnails (default: True)
* `legend_scalex` [Float]
    scale width of legend by this factor (default: 1)
* `legend_scaley` [Float]
    scale height of legend by this factor (default: 1)
* `legend_smart` [Boolean]
    Smart alignment of legend to prevent overlaps (default: True)
* `lumi_unit` [String]
    Unit for lumi label (default: "fb")
* `lumi_value` [String]
    E.g., 35.9; default hides lumi label (default: "")
* `output_diff_previous` [Boolean]
    diff the new output file with the previous (default: False)
* `output_ic` [Boolean]
    run `ic` (imgcat) on output (default: False)
* `output_jsroot` [Boolean]
    output .json for jsroot (default: False)
* `output_name` [String]
    output file name/path (default: "plot.pdf")
* `palette_name` [String]
    color palette: 'default', 'rainbow', 'susy', etc. (default: "default")
* `ratio_binomial_errors` [Boolean]
    Use binomial error propagation when computing ratio eror bars (default: False)
* `ratio_chi2prob` [Boolean]
    show chi2 probability for ratio (default: False)
* `ratio_horizontal_lines` [List]
    list of y-values to draw horizontal line (default: [1.0])
* `ratio_label_size` [Float]
    X-axis label size (default: 0.0)
* `ratio_name` [String]
    name of ratio pad (default: "Data/MC")
* `ratio_name_offset` [Float]
    offset to the name of ratio pad (default: 0.25)
* `ratio_name_size` [Float]
    size of the name on the ratio pad (e.g. data/MC) (default: 0.2)
* `ratio_ndivisions` [Int]
    SetNdivisions integer for ratio (default: 505)
* `ratio_numden_indices` [List]
    Pair of numerator and denominator histogram indices (from `bgs`) for ratio (default: None)
* `ratio_pull` [Boolean]
    show pulls instead of ratios in ratio pad (default: False)
* `ratio_pull_numbers` [Boolean]
    show numbers for pulls, and mean/sigma (default: True)
* `ratio_range` [List]
    pair for min and max y-value for ratio; default auto re-sizes to 3 sigma range (default: [-1, -1])
* `ratio_tick_length_scale` [Float]
    Tick length scale of ratio pads (default: 1.0)
* `ratio_xaxis_label_offset` [Float]
    offset to the x-axis labels (numbers) (default: None)
* `ratio_xaxis_title` [String]
    X-axis label (default: "")
* `ratio_xaxis_title_offset` [FLoat]
    X-axis label offset (default: None)
* `ratio_xaxis_title_size` [Float]
    X-axis label size (default: None)
* `ratio_yaxis_label_offset` [Float]
    offset to the y-axis labels (numbers) (default: None)
* `show_bkg_errors` [Boolean]
    show error bar for background stack (default: False)
* `show_bkg_smooth` [Boolean]
    show smoothed background stack (default: False)
* `title` [String]
    plot title (default: "")
* `us_flag` [Boolean]
    show the US flag in the corner (default: False)
* `us_flag_coordinates` [List]
    Specify flag location with (x pos, y pos, size) (default: [0.68, 0.96, 0.06])
* `xaxis_label` [String]
    label for x axis (default: "")
* `xaxis_label_offset_scale` [Float]
    x axis tickmark labels offset (default: 1.0)
* `xaxis_label_size_scale` [Float]
    size of fonts for x axis (default: 1.0)
* `xaxis_log` [Boolean]
    log scale x-axis (default: False)
* `xaxis_moreloglabels` [Boolean]
    show denser labels with logscale for x axis (default: True)
* `xaxis_noexponents` [Boolean]
    don't show exponents in logscale labels for x axis (default: False)
* `xaxis_range` [List]
    2 elements to specify x axis range (default: [])
* `xaxis_tick_length_scale` [Float]
    x axis tickmark length scale (default: 1.0)
* `xaxis_title_offset` [Float]
    offset of x axis title (default: None)
* `xaxis_title_size` [Float]
    size of fonts for x axis title (default: None)
* `yaxis_label` [String]
    label for y axis (default: "Events")
* `yaxis_label_offset_scale` [Float]
    y axis tickmark labels offset (default: 1.0)
* `yaxis_label_size_scale` [Float]
    size of fonts for y axis (default: 1.0)
* `yaxis_log` [Boolean]
    log scale y-axis (default: False)
* `yaxis_moreloglabels` [Boolean]
    show denser labels with logscale for y axis (default: True)
* `yaxis_noexponents` [Boolean]
    don't show exponents in logscale labels for y axis (default: False)
* `yaxis_range` [List]
    2 elements to specify y axis range (default: [])
* `yaxis_tick_length_scale` [Float]
    y axis tickmark length scale (default: 1.0)
* `yaxis_title_offset` [Float]
    offset of y axis title (default: None)
* `yaxis_title_size` [Float]
    size of fonts for y axis title (default: None)
* `zaxis_label` [String]
    label for z axis (default: "")
* `zaxis_label_size_scale` [Float]
    size of fonts for z axis (default: 1.0)
* `zaxis_log` [Boolean]
    log scale z-axis (default: False)
* `zaxis_moreloglabels` [Boolean]
    show denser labels with logscale for z axis (default: True)
* `zaxis_noexponents` [Boolean]
    don't show exponents in logscale labels for z axis (default: False)
* `zaxis_range` [List]
    2 elements to specify z axis range (default: [])

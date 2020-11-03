#!/bin/env python

import ROOT
import re
import os
import errno    
import sys
from QFramework import *
from syncfiles.pyfiles.errors import E
import multiprocessing
import plottery_wrapper as p
import time


########################################################################################
def addWeightSystematics(cut, systvars, cutdict):
    for systvar in systvars:
        newname = str(cut.GetName()) + systvar
        newtitle = str(cut.GetTitle()) + systvar
        wgtdef = systvars[systvar]
        #print wgtdef
        newcut = TQCut(newname, newtitle, "1", wgtdef)
        cutdict[str(newcut.GetName())] = newcut
        cut.addCut(newcut)

########################################################################################
# Example usage:
#
#   test = copyEditCuts(
#           cut=tqcuts["SRDilep"],
#           name_edits={"SR":"AR"},
#           cut_edits={"SRDilep" : TQCut("ARDilep" , "ARDilep" , "(nVlep==2)*(nLlep==2)*(nTlep==1)*(lep_pt[0]>25.)*(lep_pt[1]>25.)" , "lepsf"+lepsfvar_suffix)},
#           cutdict=tqcuts,
#           )
#
#   tqcuts["ARDilep"].printCuts("trd")
#
#   tqcuts["Presel"].addCuts(tqcuts["ARDilep"])
#
#
def copyEditCuts(cut, name_edits, cut_edits, cutdict, terminate=[], parentcut=None):

    # Create a new cut
    if cut.GetName() in cut_edits:
        newcut = cut_edits[cut.GetName()]
    else:
        name = str(cut.GetName())
        title = str(cut.GetTitle())
        cutdef = str(cut.getCutExpression())
        wgtdef = str(cut.getWeightExpression())
        newname = reduce(lambda x, y: x.replace(y, name_edits[y]), name_edits, name)
        newtitle = reduce(lambda x, y: x.replace(y, name_edits[y]), name_edits, title)
        newcut = TQCut(newname, newtitle, cutdef, wgtdef)

    if str(newcut.GetName()) not in cutdict:
        cutdict[str(newcut.GetName())] = newcut

    if not parentcut:
        parentcut = newcut
    else:
        parentcut.addCut(newcut)

    if cut.GetName() in terminate:
        return

    if len(cut.getCuts()) == 0:
        return

    # if this cut is to be modded based on what was passed to cut_edits, then replace or add
    for c in cut.getCuts():
        copyEditCuts(c, name_edits, cut_edits, cutdict, terminate, newcut)


########################################################################################
def QE(samples, proc, cut):
    count = samples.getCounter(proc, cut).getCounter()
    error = samples.getCounter(proc, cut).getError()
    return E(count, error)

########################################################################################
def addCuts(base, prefix_base, cutdefs, doNm1=True):
    doSyst = False
    cuts = []
    prefix = prefix_base.split("base_")[1]
    for i, cutdef in enumerate(cutdefs):
        cutname = "cut{}_{}".format(i, prefix)
        if i == len(cutdefs) - 1 :
            cutname = "{}".format(prefix)
        cut = TQCut(cutname, cutname, cutdef[0], cutdef[1])
        cuts.append(cut)
    for i in xrange(len(cuts) - 1):
        cuts[i].addCut(cuts[i+1])
    base.addCut(cuts[0])
    if doNm1:
        for i, cutdef in enumerate(cutdefs):
            nm1cuts = [ cut[0] for j, cut in enumerate(cutdefs) if j!=i]
            nm1wgts = [ cut[1] for j, cut in enumerate(cutdefs) if j!=i]
            cutname = "{}_minus_{}".format(prefix, i)
            base.addCut(TQCut(cutname, cutname, combexpr(nm1cuts), combexpr(nm1wgts)))

########################################################################################
def createTQCut(cutname, cutdefs):

    # To hold the TQCuts
    cuts = []
    for i, cutdef in enumerate(cutdefs):

        # Create cut name
        this_cut_name = "cut{}_{}_{}".format(i, cutname, cutdef[0])

        ## If last cut, then the cut name is the "cutname"
        #if i == len(cutdefs) - 1: this_cut_name = "{}".format(cutname)

        # Create TQCut instance
        cut = TQCut(this_cut_name, "{} ".format(i) + cutdef[1] + " ({})".format(cutdef[0]), cutdef[2], cutdef[3])

        # Aggregate cuts
        cuts.append(cut)

    # Add the last cut again
    cutdef = cutdefs[-1]

    # Create TQCut instance
    cut = TQCut(cutname, cutname, cutdef[2], cutdef[3])

    # Aggregate cuts
    cuts.append(cut)

    # Link all the cuts in steps
    for i in xrange(len(cuts)-1):
        cuts[i].addCut(cuts[i+1])

    return cuts[0]

########################################################################################
def combexpr(exprlist):
    cutlist = [ expr[0] if len(expr) != 0 else "1" for expr in exprlist ]
    wgtlist = [ expr[1] if len(expr) != 0 else "1" for expr in exprlist ]
    return "({})".format(")*(".join(cutlist)), "({})".format(")*(".join(wgtlist))

########################################################################################
def atoi(text):
    return int(text) if text.isdigit() else text

########################################################################################
def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split('(\d+)', text) ]

########################################################################################
def printCutflow(samples, regionname):
    cuts = {}
    cutnames = []
    for counter in samples.getListOfCounterNames():
        if str(counter).find(regionname) != -1 and str(counter).find("cut") != -1:
            title = samples.getCounter("/data", str(counter)).GetTitle()
            cutnames.append(str(counter))
            cuts[str(counter)] = str(title)
    cutnames.sort(key=natural_keys)
    # Cutflow printing
    printer = TQCutflowPrinter(samples)
    for cut in cutnames:
        printer.addCutflowCut(cut, cuts[cut], True)
    addProcesses(printer, showdata=True)
    table = printer.createTable("style.firstColumnAlign=l")
    path = "cutflows/"
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise
    table.writeCSV("cutflows/{}.csv".format(regionname))
    table.writeHTML("cutflows/{}.html".format(regionname))
    table.writeLaTeX("cutflows/{}.tex".format(regionname))
    table.writePlain("cutflows/{}.txt".format(regionname))

########################################################################################
def getSampleListsDeprecated(samples):
    # Get all sample lists
    sample_names = []
    sample_full_names = {}
    for i in samples.getListOfSamples():
        if i.getNSamples(True) == 0:
            sample_name = i.GetName()
            nice_name = sample_name.replace(".root", "")
            sample_names.append(nice_name)
            sample_full_names[nice_name] = sample_name
#    for sample_name in sample_names:
#        print sample_name
    return sample_names, sample_full_names

########################################################################################
def getSampleLists(samples):
    # Get all sample lists
    sample_names = []
    sample_full_names = {}
    for i in samples.getListOfSamples():
        if i.getNSamples(True) == 0:
            sample_name = i.GetName()
            nice_name = sample_name.replace(".root", "")
            sample_names.append(nice_name)
            sample_full_names[nice_name] = sample_name
#    for sample_name in sample_names:
#        print sample_name
    return sample_full_names

########################################################################################
def connectNtuples(samples, config, path, priority="<2", excludepriority=""):
    parser = TQXSecParser(samples);
    parser.readCSVfile(config)
    parser.readMappingFromColumn("*path*")
    if priority.find(">") != -1:
        priority_value = int(priority.split(">")[1])
        parser.enableSamplesWithPriorityGreaterThan("priority", priority_value)
    elif priority.find("<") != -1:
        priority_value = int(priority.split("<")[1])
        parser.enableSamplesWithPriorityLessThan("priority", priority_value)
    if excludepriority.find(">") != -1:
        priority_value = int(excludepriority.split(">")[1])
        parser.disableSamplesWithPriorityGreaterThan("priority", priority_value)
    elif excludepriority.find("<") != -1:
        priority_value = int(excludepriority.split("<")[1])
        parser.disableSamplesWithPriorityLessThan("priority", priority_value)
    parser.addAllSamples(True)
    # By "visiting" the samples with the initializer we actually hook up the samples with root files
    init = TQSampleInitializer(path, 1)
    samples.visitMe(init)
    # Print the content for debugging purpose
    #samples.printContents("rtd")

########################################################################################
def addNtuples(samples, configstr, path, config_filename, priority="<2", excludepriority=""):
    parser = TQXSecParser(samples);
    f = open(config_filename, "w")
    f.write(configstr)
    f.close()
    parser.readCSVfile(config_filename)
    parser.readMappingFromColumn("*path*")
    if priority.find(">") != -1:
        priority_value = int(priority.split(">")[1])
        parser.enableSamplesWithPriorityGreaterThan("priority", priority_value)
    elif priority.find("<") != -1:
        priority_value = int(priority.split("<")[1])
        parser.enableSamplesWithPriorityLessThan("priority", priority_value)
    if excludepriority.find(">") != -1:
        priority_value = int(excludepriority.split(">")[1])
        parser.disableSamplesWithPriorityGreaterThan("priority", priority_value)
    elif excludepriority.find("<") != -1:
        priority_value = int(excludepriority.split("<")[1])
        parser.disableSamplesWithPriorityLessThan("priority", priority_value)
    parser.addAllSamples(True)
    # By "visiting" the samples with the initializer we actually hook up the samples with root files
    init = TQSampleInitializer(path, 1)
    samples.visitMe(init)
    # Print the content for debugging purpose
    #samples.printContents("rtd")

########################################################################################
def runParallel(njobs, func, samples, extra_args):
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    pool = multiprocessing.Pool(processes=njobs)
    for sample in samples.getListOfSamples():
        if sample.getNSamples(True) == 0:
            path = str(sample.getPath())
            job = pool.apply_async(func, args=(samples, path, extra_args, return_dict))
            #job.get()
    pool.close()
    pool.join()
    failed_jobs_exist = False
    return_dict = dict(return_dict)
    for sample_to_run in return_dict:
        if return_dict[sample_to_run] != "SUCCESS":
            print sample_to_run, "failed to finish properly"
            failed_jobs_exist = True
    return not failed_jobs_exist

########################################################################################
def pathToUniqStr(sample_to_run):
    sample_to_run_prefix = sample_to_run.replace("/","-")
    sample_to_run_prefix = sample_to_run_prefix.replace("?","q")
    sample_to_run_prefix = sample_to_run_prefix.replace("[","_")
    sample_to_run_prefix = sample_to_run_prefix.replace("]","_")
    sample_to_run_prefix = sample_to_run_prefix.replace("+","-")
    return sample_to_run_prefix

########################################################################################
def makedir(dirpath):
    try:
        os.makedirs(dirpath)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(dirpath):
            pass
        else:
            raise

########################################################################################
def exportTQCutsToTextFile(cuts, filename):

    # Dump to TQFolder
    folder = TQFolder("cuts")
    cuts.dumpToFolder(folder)

    # Export to text files
    folder.exportToTextFile(filename)

########################################################################################
def loadTQCutsFromTextFile(filename):
    # Load from cuts.cfg
    cut_definitions = TQFolder("cuts")
    cut_definitions.importFromTextFile(filename)
    tqcut = None
    for f in cut_definitions.getListOfFolders():
        if not tqcut:
            tqcut = TQCut.importFromFolder(cut_definitions.getFolder(f.GetName()))
        else:
            tqcut.importFromFolderInternal(cut_definitions.getFolder(f.GetName()))
    return tqcut

########################################################################################
def runSingle(samples, sample_to_run, options, return_dict={}):

    # Perhaps you run all in serial
    isparallel = (sample_to_run != "")

    # Load the cuts from the config file
    cuts = loadTQCutsFromTextFile(options["cuts"])

    # Set the return_dict before failure to check the status
    if isparallel: return_dict[sample_to_run] = "INIT"

    #
    # Book Analysis Jobs (Histogramming, Cutflow, Event lists, etc.)
    #

    # Cutflow is always booked
    cutflowjob = TQCutflowAnalysisJob("cutflow")
    cuts.addAnalysisJob(cutflowjob, "*")

    # If the histogram configuration file is provided
    if "histo" in options and options["histo"] != "":
        histojob = TQHistoMakerAnalysisJob()
        #histojob.importJobsFromTextFiles(options["histo"], cuts, "*", True if not isparallel else False)
        histojob.importJobsFromTextFiles(options["histo"], cuts, "*", False)

    # Eventlist jobs (use this if we want to print out some event information in a text format e.g. run, lumi, evt or other variables.)
    if "eventlist" in options and options["eventlist"] != "":
        eventlistjob = TQEventlistAnalysisJob("eventlist")
        #eventlistjob.importJobsFromTextFiles(options["eventlist"], cuts, "*", True if not isparallel else False)
        eventlistjob.importJobsFromTextFiles(options["eventlist"], cuts, "*", False)

    # Declare custom observables
    if "customobservables" in options and len(options["customobservables"]) != 0:
        for observable in options["customobservables"]:
            TQObservable.addObservable(options["customobservables"][observable], observable)

    ## Print cuts and numebr of booked analysis jobs for debugging purpose
    #if not isparallel:
    #    cuts.printCut("trd")

    #
    # Loop over the samples
    #

    # setup a visitor to actually loop over ROOT files
    vis = TQAnalysisSampleVisitor(cuts, True)

    # Run the job!
    if sample_to_run:
        samples.visitSampleFolders(vis, "{}".format(sample_to_run))
    else:
        samples.visitSampleFolders(vis)

    # Write the output histograms and cutflow cut values and etc.
    if sample_to_run == "":
        sample_to_run = "output"
    samples.writeToFile(os.path.join(options["output_dir"], pathToUniqStr(sample_to_run) + ".root"))

    if isparallel: return_dict[sample_to_run] = "SUCCESS"

    return True

########################################################################################
def merge_output(samples, options):
    individual_files = []
    print "Aggregating files to merge"
    for sample in samples.getListOfSamples():
        if sample.getNSamples(True) == 0:
            path = str(sample.getPath())
            individual_files.append(os.path.join(options["output_dir"], pathToUniqStr(path) + ".root"))
    print "Issuing tqmerge command"
    cmd = "python rooutil/qframework/share/tqmerge -o {}/output{}.root -t analysis {}".format(options["output_dir"], options["output_suffix"], " ".join(individual_files))
    print cmd
    os.system("python rooutil/qframework/share/tqmerge -o {}/output{}.root -t analysis {}".format(options["output_dir"], options["output_suffix"], " ".join(individual_files)))

########################################################################################
def loop(user_options):

    options = {

        # The main root TQSampleFolder name
        "master_sample_name" : "samples",

        # Where the ntuples are located
        "ntuple_path" : "/nfs-7/userdata/phchang/WWW_babies/WWW_v1.2.3/skim/",

        # Path to the config file that defines how the samples should be organized
        "sample_config_path" : "samples.cfg",

        # The samples with "priority" (defined in sample_config_pat) values satisfying the following condition is looped over
        "priority_value" : ">0",

        # The samples with "priority" (defined in sample_config_pat) values satisfying the following condition is NOT looped over
        "exclude_priority_value" : "<-1",

        # N-cores
        "ncore" : 16,

        # TQCuts config file
        "cuts" : "cuts.cfg",

        # Histogram config file
        "histo" : "histo.cfg",

        # Eventlist histogram
        "eventlist" : "eventlist.cfg",

        # Custom observables (dictionary)
        "customobservables" : {},

        # Custom observables (dictionary)
        "output_dir" : "outputs/",

        # specific path defined
        "output_suffix" : "",

        # Do merge
        "do_merge" : True,

        # specific path defined
        "path" : ""

    }

    # Update options with the user provided values
    options.update(user_options)

    # Create output dir
    makedir(options["output_dir"])

    # Create the master TQSampleFolder
    samples = TQSampleFolder(options["master_sample_name"])

    # Connect input baby ntuple
    connectNtuples(samples, options["sample_config_path"], options["ntuple_path"], options["priority_value"], options["exclude_priority_value"])

    # If a specific path is specified run one job
    looper_success = False
    if "path" in options and options["path"] != "":
        looper_success = runSingle(samples, options["path"], options)

    # Otherwise, run parallel jobs
    else:
        looper_success = runParallel(options["ncore"], runSingle, samples, options)

    print ">>>"

    if looper_success:
        print ">>> Successfully ran qutils.loop()"
        # Merge output
        if options["do_merge"]:
            merge_output(samples, options)
        return True
    else:
        print ">>> qutils.loop() FAILED!!! Check your configurations"
        return False

########################################################################################
def output_plotname(histname, options={}):
    nicename = str(histname).replace("/","-")
    nicename = nicename.replace("{","Bin_")
    nicename = nicename.replace("}","")
    nicename = nicename.replace(",","_")
    nicename = nicename.replace(" ","")
    if "yaxis_log" in options and options["yaxis_log"] == True:
        nicename += "_logy"
    else:
        nicename += "_liny"
    return nicename

########################################################################################
def plot(samples, histname, bkg_path=[], sig_path=[], data_path=None, systs=None, clrs=[], options={}, plotfunc=p.plot_hist):
    try:
        output_dir = "plots"
        if "output_dir" in options:
            output_dir = options["output_dir"]
            del options["output_dir"]
        # Options
        alloptions= {
                    "ratio_range":[0.0,2.0],
                    "nbins": 30,
                    "autobin": False,
                    "legend_scalex": 1.8,
                    "legend_scaley": 1.1,
                    "output_name": "{}/{}.pdf".format(output_dir, output_plotname(histname, options)),
                    "bkg_sort_method": "unsorted"
                    }
        alloptions.update(options)
        bkgs = []
        sigs = []
        for bkg, path in bkg_path: bkgs.append(samples.getHistogram(path, histname).Clone(bkg))
        for sig, path in sig_path: sigs.append(samples.getHistogram(path, histname).Clone(sig))
        # Check if the type is TH2F
        for bkg in bkgs:
            if bkg.GetDimension() > 1:
                # Skip because this is not TH1
                print ">>> Skipping hist = ", histname, " as it is TH2"
                return;
        for sig in sigs:
            if sig.GetDimension() > 1:
                # Skip because this is not TH1
                print ">>> Skipping hist = ", histname, " as it is TH2"
                return;
        # Check for blinding condition
        blind = False
        if "blind" in options:
            for keyword in options["blind"]:
                #print keyword, histname
                if histname.find(keyword) != -1:
                    blind = True
            alloptions["blind"] = blind
        if data_path:
            data = samples.getHistogram(data_path, histname).Clone("Data")
        else:
            data = None
        if len(clrs) == 0: colors = [ 920, 2007, 2005, 2003, 2001, 2 ]
        else: colors = clrs
        plotfunc(
                sigs = sigs,
                bgs  = bkgs,
                data = data,
                colors = colors,
                syst = systs,
                options=alloptions)
    except:
        print (samples, histname, bkg_path, sig_path, data_path, systs, clrs, options, plotfunc)

########################################################################################
def autoplot(samples, histnames=[], bkg_path=[], sig_path=[], data_path=None, systs=None, clrs=[], options={}, plotfunc=p.plot_hist):
    import multiprocessing
    jobs = []
    #if len(histnames) == 0:
    #    histnames = samples.getListOfHistogramNames()
    #    if histnames:
    #        pass
    #    else:
    #        histnames =[]
    histnames_from_tqsample = samples.getListOfHistogramNames()
    print histnames_from_tqsample
    #for index, histname in enumerate(histnames):
    for index, hn_from_tq in enumerate(histnames_from_tqsample):
        isin = False
        if len(histnames) > 0:
            for histname in histnames:
                if str(hn_from_tq).find(histname) != -1:
                    isin = True
        else:
            isin = True
        if isin:
            proc = multiprocessing.Process(target=plot, args=[samples, str(hn_from_tq)], kwargs={"bkg_path":bkg_path, "sig_path":sig_path, "data_path":data_path, "systs":systs, "clrs":clrs, "options":options, "plotfunc":plotfunc})
            jobs.append(proc)
            proc.start()
    for histname in histnames:
        if histname.find("{") != -1:
            proc = multiprocessing.Process(target=plot, args=[samples, histname], kwargs={"bkg_path":bkg_path, "sig_path":sig_path, "data_path":data_path, "systs":systs, "clrs":clrs, "options":options, "plotfunc":plotfunc})
            jobs.append(proc)
            proc.start()

    for job in jobs:
        job.join()

########################################################################################
def plot2d(samples, histname, bkg_path=[], sig_path=[], data_path=None, systs=None, clrs=[], options={}, plotfunc=p.plot_hist):
    output_dir = "plots"
    if "output_dir" in options:
        output_dir = options["output_dir"]
        del options["output_dir"]
    # Options
    alloptions= {
                "palette_name": "rainbow",
                #"draw_option_2d": "cont4",
                "output_name": "{}/{{}}_{}.pdf".format(output_dir, output_plotname(histname)),
                }
    alloptions.update(options)
    bkgs = []
    sigs = []
    for bkg, path in bkg_path: bkgs.append(samples.getHistogram(path, histname).Clone(bkg))
    for sig, path in sig_path: sigs.append(samples.getHistogram(path, histname).Clone(sig))
    # Check if the type is TH2F
    for bkg in bkgs:
        if bkg.GetDimension() != 2:
            # Skip because this is not TH1
            print ">>> Skipping hist = ", histname, " as it is not TH2"
            return;
    for sig in sigs:
        if sig.GetDimension() != 2:
            # Skip because this is not TH1
            print ">>> Skipping hist = ", histname, " as it is not TH2"
            return;
    # Check for blinding condition
    blind = False
    if "blind" in options:
        for keyword in options["blind"]:
            #print keyword, histname
            if histname.find(keyword) != -1:
                blind = True
        alloptions["blind"] = blind
    if data_path:
        data = samples.getHistogram(data_path, histname).Clone("Data")
    else:
        data = None
    if len(clrs) == 0: colors = [ 920, 2007, 2005, 2003, 2001, 2 ]
    else: colors = clrs

    allhist = []
    allhist.extend(bkgs)
    allhist.extend(sigs)

    allhistname = []
    for bkg, path in bkg_path:
        allhistname.append(path[1:].replace("/","-"))
    for sig, path in sig_path:
        allhistname.append(path[1:].replace("/","-"))
    if data:
        allhistname.append("Data")
    raw_path = alloptions["output_name"]
    for h, name in zip(allhist, allhistname):
        #h.Smooth()
        #h.Smooth()
        #h.Smooth()
        alloptions["output_name"] = raw_path.format(name)
        p.plot_hist_2d(h, options=alloptions)

########################################################################################
def autoplot2d(samples, histnames=[], bkg_path=[], sig_path=[], data_path=None, systs=None, clrs=[], options={}, plotfunc=p.plot_hist):
    import multiprocessing
    jobs = []
    #if len(histnames) == 0:
    #    histnames = samples.getListOfHistogramNames()
    #    if histnames:
    #        pass
    #    else:
    #        histnames =[]
    histnames_from_tqsample = samples.getListOfHistogramNames()
    #for index, histname in enumerate(histnames):
    for index, hn_from_tq in enumerate(histnames_from_tqsample):
        isin = False
        if len(histnames) > 0:
            for histname in histnames:
                if str(hn_from_tq).find(histname) != -1:
                    isin = True
        else:
            isin = True
        if isin:
            proc = multiprocessing.Process(target=plot2d, args=[samples, str(hn_from_tq)], kwargs={"bkg_path":bkg_path, "sig_path":sig_path, "data_path":data_path, "systs":systs, "clrs":clrs, "options":options, "plotfunc":plotfunc})
            jobs.append(proc)
            proc.start()
    for job in jobs:
        job.join()

########################################################################################
def table(samples, from_cut, bkg_path=[], sig_path=[], data_path=None, systs=None, options={}):
    printer = TQCutflowPrinter(samples)

    # Defining which columns. e.g. Backgrounds, total background, signal, data, ratio etc.
    printer.addCutflowProcess("|", "|")
    for bkg, path in bkg_path:
        printer.addCutflowProcess(path, bkg)
    printer.addCutflowProcess("|", "|")
    totalbkgpath = '+'.join([ path[1:] for bkg, path in bkg_path ])
    printer.addCutflowProcess(totalbkgpath, "Total Bkg.")
    printer.addCutflowProcess("|", "|")
    if len(sig_path) > 0:
        for sig, path in sig_path:
            printer.addCutflowProcess(path, sig)
        printer.addCutflowProcess("|", "|")
    if data_path:
        printer.addCutflowProcess(data_path, "Data")
        printer.addCutflowProcess("$ratio({}, {})".format(data_path, totalbkgpath), "Data / Total Bkg.")
        printer.addCutflowProcess("|", "|")

    if "show_detail" in options and options["show_detail"]:
        for sample in samples.getListOfSamples():
            if sample.getNSamples(True) == 0:
                path = str(sample.getPath())
                printer.addCutflowProcess(path, path)

    # Defining which rows. e.g. which cuts
    # If cut configuration file is not provided by "cuts": cuts.cfg argument
    # then we use getListOfCounterNames()
    # If provided, then we use it to build up a nice table
    # TODO Cut filter
    if "cuts" in options:
        tqcuts = loadTQCutsFromTextFile(options["cuts"])

        # Recursive function
        def addCutflowCuts(printer, cuts, cutlist=[], indent=0):
            if len(cutlist) != 0:
                for cut in cutlist:
                    if cut == "|":
                        printer.addCutflowCut("|", "|")
                    else:
                        c = cuts.getCut(cut)
                        printer.addCutflowCut(c.GetName(), str(c.GetTitle()))
            else:
                printer.addCutflowCut(cuts.GetName(), "&emsp;"*indent + '&#x21B3;' * (indent > 0) + str(cuts.GetTitle()))
                nextindent = indent + 1
                for cut in cuts.getCuts():
                    addCutflowCuts(printer, cut, cutlist, nextindent)

        addCutflowCuts(printer, tqcuts.getCut(from_cut), options["cuts_list"] if "cuts_list" in options else [])

    else:
        print "ERROR - Please provide options[\"cuts\"] = \"cuts.cfg\"!"

    # Write out to html, tex, txt, csv
    table = printer.createTable("style.firstColumnAlign=l")
    output_dir = "cutflows"
    if "output_dir" in options:
        output_dir = options["output_dir"]
    makedir(output_dir)
    if "output_name" not in options or options["output_name"] == "":
        output_name = from_cut
    else:
        output_name = options["output_name"]
    table.writeCSV  ("{}/{}.csv" .format(output_dir, output_name))
    table.writeHTML ("{}/{}.html".format(output_dir, output_name))
    table.writeLaTeX("{}/{}.tex" .format(output_dir, output_name))
    table.writePlain("{}/{}.txt" .format(output_dir, output_name))

    print ">>> Saving {}/{}.html".format(output_dir, output_name)

    # Stupid hack :( to fix the missing hashtag from qframework writeHTML function
    FileName = "{}/{}.html".format(output_dir, output_name)
    with open(FileName) as f:
        newText=f.read().replace('&21B3', '&#x21B3')

    with open(FileName, "w") as f:
        f.write(newText)

    # To place tabs in between texts for easy copy paste
    fname = "{}/{}.txt".format(output_dir, output_name)
    f = open(fname)
    g = open(fname+".tabbed", "w")
    lines = f.readlines()
    for line in lines:
        if line.find("|") != -1:
            g.write("\t".join(line.split()) + "\n")
        else:
            g.write(line)

########################################################################################
def autotable(samples, cutnames=[], bkg_path=[], sig_path=[], data_path=None, systs=None, options={}):
    import multiprocessing
    jobs = []
    if len(cutnames) == 0:
        print "ERROR - provided no cut names to create table from"
    for cutname in cutnames:
        proc = multiprocessing.Process(target=table, args=[samples, str(cutname)], kwargs={"bkg_path":bkg_path, "sig_path":sig_path, "data_path":data_path, "systs":systs, "options":options})
        jobs.append(proc)
        proc.start()
    for job in jobs:
        job.join()

########################################################################################
def get_cr_normalized_rate(options, key):
    # This is parsing an example like this:
    # ("SideSSmmFull" , "/typebkg/lostlep/[ttZ+WZ+Other]") : ("WZCRSSmmFull"    , "/data-typebkg/qflip-typebkg/photon-typebkg/prompt-typebkg/fakes-typebkg/lostlep/VBSWW-typebkg/lostlep/ttW-sig"),
    sr = options["nominal_sample"].getCounter(key[1], key[0])
    crdatapath = options["control_regions"][key][1]
    crprocpath = key[1]
    crname = options["control_regions"][key][0]
    nf = options["nominal_sample"].getCounter(crdatapath, crname)
    pr = options["nominal_sample"].getCounter(crprocpath, crname)
    nf.divide(pr)
    #print sr.getCounter()
    sr.multiply(nf)
    #print nf.getCounter(), sr.getCounter()
    #print "get_cr", key, sr.getCounter()
    return sr.getCounter()

########################################################################################
def make_thNmap(filepath, histpath, varx, vary="", varz=""):
    f = ROOT.TFile(filepath)
    h = f.Get(histpath)
    if not h:
        print filepath, histpath, "not found!"
        sys.exit()
    if vary == "":
        mapstr = "[TH1Map:{}:{}([{}])]".format(filepath, histpath, varx)
    elif varz == "":
        mapstr = "[TH2Map:{}:{}([{}],[{}])]".format(filepath, histpath, varx, vary)
    else:
        mapstr = "[TH3Map:{}:{}([{}],[{}],[{}])]".format(filepath, histpath, varx, vary, varz)
    print mapstr
    return mapstr


########################################################################################
def get_sr_rate(samples, path, r, suffix, options):
    if (r, path) not in options["control_regions"]:
        return samples.getCounter(path, r+suffix).getCounter()
    else:
        # The TF calculation
        cr = options["control_regions"][(r, path)][0]
        # nominal sr
        sr_nom = options["nominal_sample"].getCounter(path,  r)
        cr_nom = options["nominal_sample"].getCounter(path, cr)
        # syst
        sr_sys = samples.getCounter(path,  r+suffix)
        cr_sys = samples.getCounter(path, cr+suffix)
        sr_nom.divide(cr_nom)
        sr_sys.divide(cr_sys)
        sr_sys.divide(sr_nom)
        return get_cr_normalized_rate(options, (r, path)) * sr_sys.getCounter()

########################################################################################
def get_tf(r, path, options):
    # The TF calculation
    cr = options["control_regions"][(r, path)][0]
    cr_data = options["nominal_sample"].getCounter(options["data"], cr)
    rate = get_cr_normalized_rate(options, (r, path))
    #print "get_tf", (r, path), rate / cr_data.getCounter()
    return rate / cr_data.getCounter()

########################################################################################
def make_counting_experiment_statistics_data_card(options):

    #
    # The goal is to create a data card for https://twiki.cern.ch/twiki/bin/viewauth/CMS/SWGuideHiggsAnalysisCombinedLimit
    #

    column_width = 10
    for b in options["bins"]:
        if len(b) + 5 > column_width:
            column_width = len(b) + 5

    def form(s): return ("{:<"+str(column_width)+"s}").format(s)
    def flts(f): return ("{:<"+str(column_width)+"s}").format("{:<6.5f}".format(f)) if f > 0 else form("1e-9")

    # Channels (e.g. SR1, SR2, SR3, ...)
    nchannel = len(options["bins"])
    channels = [ form(x) for x in options["bins"]]

    # Processes (e.g. Higgs, ttbar, WW, W, Z, etc.)
    nprocess = len(options["bkgs"]) + 1
    processes = [ form(x) for x, path in ([options["sig"]] + options["bkgs"])]
    process_indices = [ form(str(index)) for index, x in enumerate([options["sig"]] + options["bkgs"])]
    bins_list = [ x * nprocess for x in channels ]
    processes_list = processes * nchannel

    # Creating list to access contents
    cuts_list = [ x for x in options["bins"] for i in range(nprocess) ]
    paths = [ path for x, path in ([options["sig"]] + options["bkgs"])]
    paths_list = paths * nchannel

    # nobservation to be printed
    nobs = [ form(str(int(options["nominal_sample"].getCounter(options["data"], r).getCounter()))) for r in options["bins"] ]

    # rates
    rates_val = []
    for r, path in zip(cuts_list, paths_list):
        key = (r, path)
        if key in options["control_regions"]:
            rates_val.append(get_cr_normalized_rate(options, key))
        else:
            rates_val.append(options["nominal_sample"].getCounter(path, r).getCounter())

    #rates_val = [ c.getCounter() for c in [ options["nominal_sample"].getCounter(path, r) if (r, proc.strip()) not in options["control_regions"] else get_cr_normalized_rate(options, options["control_regions"][(r, proc.strip())]) for r, path, proc in zip(cuts_list, paths_list, processes_list) ] ]
    rates_str = [ flts(cnt) for cnt in rates_val ]
            
    # items to be printed
    nchannel_formatted = nchannel
    channels_formatted = "".join(channels)
    bins_formatted = "".join(bins_list)
    processes_formatted = "".join(processes_list)
    process_indices_formatted = "".join(process_indices * nchannel)
    nobs_formatted = "".join(nobs)
    rates_formatted = "".join(rates_str)

    datacard  = "# Created {}\n".format(time.strftime("%Y-%m-%d %H:%M"))
    datacard += "# options = {}\n".format(options)

    datacard += """# Counting experiment with multiple channels
imax {nchannel}  number of channels
jmax *   number of backgrounds ('*' = automatic)
kmax *   number of nuisance parameters (sources of systematical uncertainties)
------------
# three channels, each with it's number of observed events
bin          {channels}
observation  {nobs}
------------
# now we list the expected events for signal and all backgrounds in those three bins
# the second 'process' line must have a positive number for backgrounds, and 0 for signal
# then we list the independent sources of uncertainties, and give their effect (syst. error)
# on each process and bin
bin                                           {bins}
process                                       {processes}
process                                       {process_indices}
rate                                          {rates}
------------
""".format(
        nchannel=nchannel_formatted,
        channels=channels_formatted,
        nobs=nobs_formatted,
        bins=bins_formatted,
        processes=processes_formatted,
        process_indices=process_indices_formatted,
        rates=rates_formatted,
        )

    ## Weight variation systematics that are saved in the "nominal_sample" TQSampleFolder
    ## The nomenclature of the coutner names must be <BIN_COUNTER><SYSTS>Up and <BIN_COUNTER><SYSTS>Down
    ## Or if the "syst_samples" are provided in the dictionary use that instead
    ## The keyword are the systematics and then the items list the processes to apply the systematics
    #
    # For example they will have the following format
    # "systematics" : [
    #     ("LepSF"         , { "procs_to_apply" : ["vbsww", "ttw", "photon", "qflip", "prompt"]                                                                          }),
    #     ("TrigSF"        , { "procs_to_apply" : ["vbsww", "ttw", "photon", "qflip", "prompt"]                                                                          }),
    #     ("BTagLF"        , { "procs_to_apply" : ["vbsww", "ttw", "photon", "qflip", "prompt"]                                                                          }),
    #     ("BTagHF"        , { "procs_to_apply" : ["vbsww", "ttw", "photon", "qflip", "prompt"]                                                                          }),
    #     ("Pileup"        , { "procs_to_apply" : ["vbsww", "ttw", "photon", "qflip", "prompt"]                                                                          }),
    #     ("FakeRateEl"    , { "procs_to_apply" : ["fake"]                                                                                                               }),
    #     ("FakeRateMu"    , { "procs_to_apply" : ["fake"]                                                                                                               }),
    #     ("FakeClosureEl" , { "procs_to_apply" : ["fake"]                                                                                                               }),
    #     ("FakeClosureMu" , { "procs_to_apply" : ["fake"]                                                                                                               }),
    #     ("PDF"           , { "procs_to_apply" : ["www"]                                                                                                                }),
    #     ("AlphaS"        , { "procs_to_apply" : ["www"]                                                                                                                }),
    #     ("Qsq"           , { "procs_to_apply" : ["www"]                                                                                                                }),
    #     ("JEC"           , { "procs_to_apply" : ["www", "vbsww", "ttw", "photon", "qflip", "prompt"], "syst_samples" : {"Up" : samples_jec_up, "Down": samples_jec_dn} }),
    #     ("MCStat"        , { "procs_to_apply" : ["www", "vbsww", "ttw", "photon", "qflip", "prompt"], "individual": True                                               }),
    #     ],
    for syst, systinfo in options["systematics"]:
        # If "syst_samples" are provided in the systinfo dictionary then use nominal cut counter of the provided sample to get the variations
        # If not provided, then attach a suffix to the counter name (these would be the weight variations)
        # If "syst_samples" not provided than it is a weight variational type so create a suffix to attach to the counter name
        syst_up_name_suffix = syst + "Up"   if "syst_samples" not in systinfo else ""
        syst_dn_name_suffix = syst + "Down" if "syst_samples" not in systinfo else ""
        samples_up = options["nominal_sample"] if "syst_samples" not in systinfo else systinfo["syst_samples"]["Up"]
        samples_dn = options["nominal_sample"] if "syst_samples" not in systinfo else systinfo["syst_samples"]["Down"]
        syst_up_rates_val = [ c for c in [ get_sr_rate(samples_up, path, r, syst_up_name_suffix, options) if process.strip() in systinfo["procs_to_apply"] else 0 for r, process, path in zip(cuts_list, processes * nchannel, paths_list) ] ]
        syst_dn_rates_val = [ c for c in [ get_sr_rate(samples_dn, path, r, syst_dn_name_suffix, options) if process.strip() in systinfo["procs_to_apply"] else 0 for r, process, path in zip(cuts_list, processes * nchannel, paths_list) ] ]
        syst_val_str = [ form("{:.5f}/{:<.5f}".format(max(dn, 0.001), up)) if (up > 0 or dn > 0) else form("-")  for up, dn in [ ((u / n, d / n) if n > 0 else (1, 1)) if p.strip() in systinfo["procs_to_apply"] else (-999, -999) for u, d, n, p in zip(syst_up_rates_val, syst_dn_rates_val, rates_val, processes * nchannel) ] ]
        syst_item = """{:<35s}lnN        {}\n""".format(syst, "".join(syst_val_str))
        datacard += syst_item

    # Statistical error per bin per channel add a statistical error from the MC
    for index, (r, process, path) in enumerate(zip(cuts_list, processes * nchannel, paths_list)):
        if process.strip() not in options["statistical"]:
            continue
        cnt = options["nominal_sample"].getCounter(path, r).getCounter()
        err = options["nominal_sample"].getCounter(path, r).getError()
        errors = [(0, 0)] * nprocess * nchannel
        errors[index] = ((cnt + err) / cnt, (cnt - err) / cnt) if cnt > 0 else (1, 1)
        syst_val_str = [ form("{:.5f}/{:<.5f}".format(max(dn, 0.001), up)) if (up > 0 or dn > 0) else form("-") for up, dn in errors ]
        systname = process.strip() + "_MCstat" + "_" + r
        syst_item = """{:<35s}lnN        {}\n""".format(systname, "".join(syst_val_str))
        datacard += syst_item

    # Control region statistical error
    # CR data stat error can be controlled via "gmN" error
    # In the options the control regions are provided in a following format
    #
    # "control_regions" : {
    #     ("SRSSeeFull"  , "/typebkg/lostlep/[ttZ+WZ+Other]") : ("WZCRSSeeFull", "/data-typebkg/[qflip+photon+prompt+fakes]-sig"),
    #     ("SideSSeeFull", "/typebkg/lostlep/[ttZ+WZ+Other]") : ("WZCRSSeeFull", "/data-typebkg/[qflip+photon+prompt+fakes]-sig"),
    #     },
    #
    # We first invert the regions such that we have a mapping per "CR" -> "SR's"
    crmap = {}
    for k, v in options["control_regions"].iteritems():
        crmap[v] = crmap.get(v, [])
        crmap[v].append(k)

    for key in sorted(crmap):
        syst_val_str = []
        for index, (r, process, path) in enumerate(zip(cuts_list, processes * nchannel, paths_list)):
            if (r, path) not in crmap[key]:
                syst_val_str.append(form("-"))
            else:
                syst_val_str.append(form("{:.5f}".format(get_tf(r, path, options))))
        systname = key[0] + "_CRstat"
        data = int(options["nominal_sample"].getCounter(options["data"], key[0]).getCounter())
        syst_item = """{:<35s}gmN {:<6d} {}\n""".format(systname, data, "".join(syst_val_str))
        datacard += syst_item

    for syst_name, proc, syst_val, filt_pattern in options["flat_systematics"]:
        syst_val_str = []
        for index, (r, process, path) in enumerate(zip(cuts_list, processes * nchannel, paths_list)):
            if process.strip() in proc and r.find(filt_pattern) != -1:
                syst_val_str.append(form(syst_val))
            else:
                syst_val_str.append(form("-"))
        syst_item = """{:<35s}lnN        {}\n""".format(syst_name, "".join(syst_val_str))
        datacard += syst_item

    return datacard


########################################################################################
def make_shape_experiment_statistics_data_card(options):

    #
    # The goal is to create a data card for https://twiki.cern.ch/twiki/bin/viewauth/CMS/SWGuideHiggsAnalysisCombinedLimit
    #

    column_width = 10
    for b, hp in options["hists"]:
        if len(b) + 5 > column_width:
            column_width = len(b) + 5

    # helper function
    def form(s): return ("{:<"+str(column_width)+"s}").format(s)
    def flts(f): return ("{:<"+str(column_width)+"s}").format("{:<6.5f}".format(f)) if f > 0 else form("1e-9")

    # hists : ("Name", "HistPath")
    nhist    = len(options["hists"])
    hist_names    = [ form(x) for x, path in options["hists"]]
    hist_paths    = [ path    for x, path in options["hists"]]

    # processes : ("Name", "SamplePath")
    nprocess = len(options["bkgs"]) + 1
    process_names = [ form(x) for x, path in ([options["sig"]] + options["bkgs"])]
    process_paths = [ path    for x, path in ([options["sig"]] + options["bkgs"])]
    process_indices = [ form(str(index)) for index, x in enumerate([options["sig"]] + options["bkgs"])]

    # helper function
    def abcabc(l, n): return l * n
    def aabbcc(l, n): return [ x for x in l for i in range(n) ]

    # Creating list to access contents in a nice single loop over a zipped list
    grand_list = zip(
            aabbcc(hist_names, nprocess),
            aabbcc(hist_paths, nprocess),
            abcabc(process_names, nhist),
            abcabc(process_paths, nhist),
            abcabc(process_indices, nhist)
            )

    #for h, hp, p, pp, i in grand_list:
    #    print h, hp, p, pp, i

    # Output histogram file
    f = ROOT.TFile(options["hist_output_file"], "recreate")

    # Get proc name
    def proc(pp):
        if options["sig"][1] == pp:
            return options["sig"][0]
        for i in options["bkgs"]:
            if i[1] == pp:
                return i[0]
        if options["data"] == pp:
            return "data_obs"
        print "ERROR - histogram path didn't match ", pp

    # Hists
    hist_dict = {}
    def Key(h, p): return (h.strip(), p.strip())
    def Hist(key): return hist_dict[key]
    def formsyst(hp, suffix):
        # There are two types of histogram names
        # Cut/hist+Cut/hist
        # or
        # {Cut,Cut,Cut}
        if suffix != "":
            if hp.find("{") != -1:
                hp = hp.replace("{","")
                hp = hp.replace("}","")
                return "{" + ",".join([ i + suffix for i in hp.split(",") ]) + "}"
            else:
                return "/".join([ i + suffix for i in hp.split("/") ])
        else:
            return hp
    def getHist(s, key, p, n, suffix=""):
        h = s.getHistogram(p, formsyst(n, suffix)).Clone("{}_{}".format(key[0], proc(key[1]))) # if suffix == "" else "{}_{}".format(key[0], proc(key[1]), suffix))
        h.SetCanExtend(False)
        for i in xrange(0, h.GetNbinsX()+2):
            bc = h.GetBinContent(i)
            h.SetBinContent(i, bc if bc > 0 else 1e-9)
        h.SetTitle("{}_{}".format(formsyst(n, suffix), p))
        return h
    def addHist(s, key, p, n, suffix=""):
        if key in hist_dict:
            print "ERROR - histogram already accessed and exists!", key
        else:
            hist_dict[key] = getHist(s, key, p, n, suffix)
    def get_cr_normalized_hist(options, key, pp, hp):
        # This is parsing an example like this:
        # ("SRSSeeFull"   , "lostlep") : ("{WZCRSSeeFull}"    , "/data-typebkg/qflip-typebkg/photon-typebkg/prompt-typebkg/fakes-typebkg/lostlep/VBSWW-typebkg/lostlep/ttW-sig"),
        addHist(options["nominal_sample"], key, pp, hp)
        crdatapath = options["control_regions"][key][1]
        crprocpath = key[1]
        crname = options["control_regions"][key][0]
        dd = options["nominal_sample"].getHistogram(crdatapath, crname).Clone("dd")
        pr = options["nominal_sample"].getHistogram(crprocpath, crname).Clone("pr")
        CR_data_error = []
        for i in xrange(0,dd.GetNbinsX()+2):
            CR_data_error.append(dd.GetBinError(i) / dd.GetBinContent(i) if dd.GetBinContent(i) > 0 else 0) # 0.5 * ROOT.TMath.ChisquareQuantile(1 - 0.3173 / 2, 2 * (0 + 1)) - 0
        nf = dd.Integral() / pr.Integral()
        Hist(key).Scale(nf)
        for i in xrange(1, Hist(key).GetNbinsX()+1):
            Hist(key).SetBinError(i, Hist(key).GetBinContent(i) * CR_data_error[i])
    def addTFSystHist(s, key, p, n, suffix="", nominal_histname=""):
        cr = options["control_regions"][(nominal_histname.strip(),key[1])][0]
        sr_nom = getHist(options["nominal_sample"], key, p, n, "")
        cr_nom = getHist(options["nominal_sample"], key, p,cr, "")
        sr_sys = getHist(s, key, p, n, suffix)
        cr_sys = getHist(s, key, p,cr, suffix)
        sr_nom.Divide(cr_nom)
        sr_sys.Divide(cr_sys)
        sr_sys.Divide(sr_nom)
        nom = Hist((nominal_histname.strip(),key[1])).Clone("{}_{}".format(key[0], proc(key[1])))
        nom.Multiply(sr_sys)
        if key in hist_dict:
            print "ERROR - histogram already accessed and exists!", key
        else:
            hist_dict[key] = nom
    def writeHists():
        keylist = hist_dict.keys()
        keylist.sort()
        for key in keylist:
            hist_dict[key].Write()

    # nobservation to be printed
    nobs = []
    for h, hp in options["hists"]:
        key = Key(h, options["data"])
        addHist(options["nominal_sample"], key, options["data"], hp)
        nobs.append(form(str(int(Hist(key).Integral()))))


    # rates for each bin and process
    rates_val = []
    for h, hp, p, pp, i in grand_list:
        key = Key(h, pp)
        if key in options["control_regions"]:
            get_cr_normalized_hist(options, key, pp, hp)
            rates_val.append(Hist(key).Integral())
        else:
            addHist(options["nominal_sample"], key, pp, hp)
            rates_val.append(Hist(key).Integral())
    rates_str = [ flts(cnt) for cnt in rates_val ]

    nhist_formatted = nhist
    channels_formatted = "".join(hist_names)
    bins_formatted = "".join(aabbcc(hist_names, nprocess))
    processes_formatted = "".join(abcabc(process_names, nhist))
    process_indices_formatted = "".join(abcabc(process_indices, nhist))
    nobs_formatted = "".join(nobs)
    rates_formatted = "".join(rates_str)

    datacard  = "# Created {}\n".format(time.strftime("%Y-%m-%d %H:%M"))
    datacard += "# options = {}\n".format(options)

    datacard += """# Counting experiment with multiple channels
imax {nhist}  number of channels
jmax *   number of backgrounds ('*' = automatic)
kmax *   number of nuisance parameters (sources of systematical uncertainties)
------------
shapes * * {hist_output_file} $CHANNEL_$PROCESS $CHANNEL$SYSTEMATIC_$PROCESS
------------
# three channels, each with it's number of observed events
bin          {channels}
observation  {nobs}
------------
# now we list the expected events for signal and all backgrounds in those three bins
# the second 'process' line must have a positive number for backgrounds, and 0 for signal
# then we list the independent sources of uncertainties, and give their effect (syst. error)
# on each process and bin
bin                                           {bins}
process                                       {process_indices}
process                                       {processes}
rate                                          {rates}
------------
""".format(
        hist_output_file=options["hist_output_file"],
        nhist=nhist_formatted,
        channels=channels_formatted,
        nobs=nobs_formatted,
        bins=bins_formatted,
        processes=processes_formatted,
        process_indices=process_indices_formatted,
        rates=rates_formatted,
        )

    ## Weight variation systematics that are saved in the "nominal_sample" TQSampleFolder
    ## The nomenclature of the coutner names must be <BIN_COUNTER><SYSTS>Up and <BIN_COUNTER><SYSTS>Down
    ## Or if the "syst_samples" are provided in the dictionary use that instead
    ## The keyword are the systematics and then the items list the processes to apply the systematics
    #
    # For example they will have the following format
    # "systematics" : [
    #     ("LepSF"         , { "procs_to_apply" : ["vbsww", "ttw", "photon", "qflip", "prompt"]                                                                          }),
    #     ("TrigSF"        , { "procs_to_apply" : ["vbsww", "ttw", "photon", "qflip", "prompt"]                                                                          }),
    #     ("BTagLF"        , { "procs_to_apply" : ["vbsww", "ttw", "photon", "qflip", "prompt"]                                                                          }),
    #     ("BTagHF"        , { "procs_to_apply" : ["vbsww", "ttw", "photon", "qflip", "prompt"]                                                                          }),
    #     ("Pileup"        , { "procs_to_apply" : ["vbsww", "ttw", "photon", "qflip", "prompt"]                                                                          }),
    #     ("FakeRateEl"    , { "procs_to_apply" : ["fake"]                                                                                                               }),
    #     ("FakeRateMu"    , { "procs_to_apply" : ["fake"]                                                                                                               }),
    #     ("FakeClosureEl" , { "procs_to_apply" : ["fake"]                                                                                                               }),
    #     ("FakeClosureMu" , { "procs_to_apply" : ["fake"]                                                                                                               }),
    #     ("PDF"           , { "procs_to_apply" : ["www"]                                                                                                                }),
    #     ("AlphaS"        , { "procs_to_apply" : ["www"]                                                                                                                }),
    #     ("Qsq"           , { "procs_to_apply" : ["www"]                                                                                                                }),
    #     ("JEC"           , { "procs_to_apply" : ["www", "vbsww", "ttw", "photon", "qflip", "prompt"], "syst_samples" : {"Up" : samples_jec_up, "Down": samples_jec_dn} }),
    #     ("MCStat"        , { "procs_to_apply" : ["www", "vbsww", "ttw", "photon", "qflip", "prompt"], "individual": True                                               }),
    #     ],
    for syst, systinfo in options["systematics"]:
        # If "syst_samples" are provided in the systinfo dictionary then use nominal cut counter of the provided sample to get the variations
        # If not provided, then attach a suffix to the counter name (these would be the weight variations)
        # If "syst_samples" not provided than it is a weight variational type so create a suffix to attach to the counter name
        syst_up_name_suffix = syst + "Up"   if "syst_samples" not in systinfo else ""
        syst_dn_name_suffix = syst + "Down" if "syst_samples" not in systinfo else ""
        samples_up = options["nominal_sample"] if "syst_samples" not in systinfo else systinfo["syst_samples"]["Up"]
        samples_dn = options["nominal_sample"] if "syst_samples" not in systinfo else systinfo["syst_samples"]["Down"]
        syst_up_rates_val = []
        syst_dn_rates_val = []
        syst_val_str = []
        for h, hp, p, pp, i in grand_list:
            if p.strip() in systinfo["procs_to_apply"]:
                nomkey = Key(h, pp)
                if Key(h, pp) not in options["control_regions"]:
                    key = Key(h.strip()+syst+"Up", pp)
                    addHist(samples_up, key, pp, hp, syst_up_name_suffix)
                    key = Key(h.strip()+syst+"Down", pp)
                    addHist(samples_dn, key, pp, hp, syst_dn_name_suffix)
                    syst_val_str.append(form("1"))
                else:
                    key = Key(h.strip()+syst+"Up", pp)
                    addTFSystHist(samples_up, key, pp, hp, syst_up_name_suffix, h)
                    key = Key(h.strip()+syst+"Down", pp)
                    addTFSystHist(samples_dn, key, pp, hp, syst_dn_name_suffix, h)
                    syst_val_str.append(form("1"))
            else:
                syst_val_str.append(form("-"))
        syst_item = """{:<35s}shape      {}\n""".format(syst, "".join(syst_val_str))
        datacard += syst_item

    for index, (h, hp, p, pp, i) in enumerate(grand_list):
        key = Key(h, pp)
        if p.strip() not in options["statistical"]:
            continue
        h_hist = Hist(key)
        for ibin in xrange(1, h_hist.GetNbinsX()+1):
            cnt = h_hist.GetBinContent(ibin)
            err = h_hist.GetBinError(ibin)
            histname = "{}".format(h.strip())
            systname = histname+p.strip()+"_"+"MCstat" + str(ibin)
            systkey_up = Key(histname+systname+"Up"  , pp)
            systkey_dn = Key(histname+systname+"Down", pp)
            hsys_up = h_hist.Clone("{}_{}".format(histname+systname+"Up"  , p))
            hsys_dn = h_hist.Clone("{}_{}".format(histname+systname+"Down", p))
            hsys_up.SetBinContent(ibin, cnt+cnt*min(err/cnt, 1))
            hsys_dn.SetBinContent(ibin, max(cnt-err, 1e-9))
            hist_dict[systkey_up] = hsys_up
            hist_dict[systkey_dn] = hsys_dn
            syst_val_str = [ form("-") ] * nprocess * nhist
            syst_val_str[index] = form("1")
            syst_item = """{:<35s}shape      {}\n""".format(systname, "".join(syst_val_str))
            datacard += syst_item

#    # Control region statistical error
#    # CR data stat error can be controlled via "gmN" error
#    # In the options the control regions are provided in a following format
#    #
#    # "control_regions" : {
#    #     ("SRSSeeFull"   , "/typebkg/lostlep/[ttZ+WZ+Other]") : ("{WZCRSSeeFull}"    , "/data-typebkg/qflip-typebkg/photon-typebkg/prompt-typebkg/fakes-typebkg/lostlep/VBSWW-typebkg/lostlep/ttW-sig"),
#    #     ("SRSSemFull"   , "/typebkg/lostlep/[ttZ+WZ+Other]") : ("{WZCRSSemFull}"    , "/data-typebkg/qflip-typebkg/photon-typebkg/prompt-typebkg/fakes-typebkg/lostlep/VBSWW-typebkg/lostlep/ttW-sig"),
#    #     },
#    #
#    # We first invert the regions such that we have a mapping per "CR" -> "SR's"
#    crmap = {}
#    for k, v in options["control_regions"].iteritems():
#        crmap[v] = crmap.get(v, [])
#        crmap[v].append(k)
#
#    for key in sorted(crmap):
#        syst_val_str = []
#        for index, (r, process, path) in enumerate(zip(cuts_list, processes * nchannel, paths_list)):
#            if (r, path) not in crmap[key]:
#                syst_val_str.append(form("-"))
#            else:
#                syst_val_str.append(form("{:.5f}".format(get_tf(r, path, options))))
#        systname = key[0] + "_CRstat"
#        data = int(options["nominal_sample"].getCounter(options["data"], key[0]).getCounter())
#        syst_item = """{:<35s}gmN {:<6d} {}\n""".format(systname, data, "".join(syst_val_str))
#        datacard += syst_item
#
#    for syst_name, proc, syst_val, filt_pattern in options["flat_systematics"]:
#        syst_val_str = []
#        for index, (r, process, path) in enumerate(zip(cuts_list, processes * nchannel, paths_list)):
#            if process.strip() in proc and r.find(filt_pattern) != -1:
#                syst_val_str.append(form(syst_val))
#            else:
#                syst_val_str.append(form("-"))
#        syst_item = """{:<35s}lnN        {}\n""".format(syst_name, "".join(syst_val_str))
#        datacard += syst_item

    # Write all histogram to output
    writeHists()

    return datacard

########################################################################################
def draw_eff(samples, numerator, denominator, mc, data, variable, options):

    mc_numerator   = samples.getHistogram(mc  , "{}/{}".format(numerator, variable))
    dt_numerator   = samples.getHistogram(data, "{}/{}".format(numerator, variable))
    mc_denominator = samples.getHistogram(mc  , "{}/{}".format(denominator, variable))
    dt_denominator = samples.getHistogram(data, "{}/{}".format(denominator, variable))

    if "nbins" in options:
        if mc_numerator.GetNbinsX() == 180:
            nbin = int(180 / options["nbins"])
            mc_numerator.Rebin(nbin)
            dt_numerator.Rebin(nbin)
            mc_denominator.Rebin(nbin)
            dt_denominator.Rebin(nbin)

    mc_eff = mc_numerator.Clone()
    dt_eff = dt_numerator.Clone()

    mc_eff.Divide(mc_numerator, mc_denominator, 1, 1, "B")
    dt_eff.Divide(dt_numerator, dt_denominator, 1, 1, "B")

    p.plot_hist(
            sigs = [],
            bgs  = [mc_eff],
            data = dt_eff,
            colors = [],
            syst = None,
            options=options
            )

    return mc_eff

########################################################################################
def draw_eff_num1_num2(samples, numerator1, numerator2, denominator, mc, variable, options):

    mc_numerator   = samples.getHistogram(mc, "{}/{}".format(numerator1, variable)).Clone(numerator1)
    dt_numerator   = samples.getHistogram(mc, "{}/{}".format(numerator2, variable)).Clone(numerator2)
    mc_denominator = samples.getHistogram(mc, "{}/{}".format(denominator, variable))
    dt_denominator = samples.getHistogram(mc, "{}/{}".format(denominator, variable))


    if "nbins" in options:
        if mc_numerator.GetNbinsX() == 180:
            nbin = int(180 / options["nbins"])
            mc_numerator.Rebin(nbin)
            dt_numerator.Rebin(nbin)
            mc_denominator.Rebin(nbin)
            dt_denominator.Rebin(nbin)

    mc_eff = mc_numerator.Clone()
    dt_eff = dt_numerator.Clone()

    mc_eff.Divide(mc_numerator, mc_denominator, 1, 1, "B")
    dt_eff.Divide(dt_numerator, dt_denominator, 1, 1, "B")

    p.plot_hist(
            sigs = [],
            bgs  = [mc_eff],
            data = dt_eff,
            colors = [],
            syst = None,
            options=options
            )

    return mc_eff


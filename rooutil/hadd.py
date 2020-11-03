#!/bin/env python

import ROOT as r
import sys
import os
import glob

def hadd(files, treename, output, neventsperfile):

    # Get all the nevents file from the reference tree name
    nevents = {}
    for to_merge_file_path in files:
        f = r.TFile(to_merge_file_path)
        t = f.Get(treename)
        nevents[to_merge_file_path] = int(t.GetEntries())

    # Now figure out what files to merge at a time
    clusters = []
    cluster = []
    nevents_in_current_cluster = 0
    for fn in files:
        if nevents_in_current_cluster + nevents[fn] > neventsperfile:
            if len(cluster) > 0:
                clusters.append(cluster)
            nevents_in_current_cluster = nevents[fn]
            cluster = [fn]
        else:
            nevents_in_current_cluster += nevents[fn]
            cluster.append(fn)

    # very last cluster needs to be added
    clusters.append(cluster)

    ## print the clustering info
    #for cluster in clusters:
    #    print cluster

    # Obtain the header path
    output_path_without_dot_root_full = output.replace(".root", "")

    # Base path
    output_path_without_dot_root_basename = os.path.basename(output_path_without_dot_root_full)

    # If it has hadoop
    is_hadoop = "hadoop" in output

    if is_hadoop:
        output_path_without_dot_root = output_path_without_dot_root_basename
    else:
        output_path_without_dot_root = output_path_without_dot_root_full

    # Run the commands
    print clusters
    for index, cluster in enumerate(clusters):

        command = "hadd -f {}_{}.root {}".format(output_path_without_dot_root, index+1, " ".join(cluster))
        print command
        os.system(command)
        if is_hadoop:
            command = "cp -v {}_{}.root {}_{}.root".format(output_path_without_dot_root, index+1, output_path_without_dot_root_full, index+1)
            print command
            os.system(command)
            command = "rm -v {}_{}.root".format(output_path_without_dot_root, index+1, output_path_without_dot_root_full, index+1)
            print command
            os.system(command)

def hadd_dir(dir_path, treename="t", nevents=50000, globber="*.root"):
    source_files = glob.glob("{}/{}".format(dir_path, globber))
    output_dir = "{}/merged".format(dir_path)
    output_path = "{}/output.root".format(output_dir, dir_path)
    os.system("mkdir -p {}".format(output_dir))
    hadd(source_files, treename, output_path, nevents)

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description="Hadd files")
    parser.add_argument('--output'     , '-o' , dest='output'     , default='output.root' , help='output root file path')
    parser.add_argument('--treename'   , '-t' , dest='treename'   , default='t'           , help='reference tree name to obtain number of events to determine how many files to merge at a time')
    parser.add_argument('--nevents'    , '-n' , dest='nevents'    , default=50000         , help='number of events to put at max per output merged files')
    parser.add_argument('files', metavar='FILE.csv', type=str, nargs='+', help='input files')
    args = parser.parse_args()

    hadd(args.files, args.treename, args.output, args.nevents)


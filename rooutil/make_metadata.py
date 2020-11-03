#!/bin/env python

import ROOT as r
import glob 
import sys
import multiprocessing
import json

def get_nevents(baby_file, return_dict):
    baby_tfile = r.TFile(baby_file)
    h = baby_tfile.Get("h_neventsinfile")
    ninput = h.GetBinContent(0)
    nprocessed = h.GetBinContent(1)
    nwgtprocessed = h.GetBinContent(15)
    return_dict[baby_file] = (ninput, nprocessed, nwgtprocessed)
    
def make_metadata(baby_dir):

    baby_files = glob.glob(baby_dir+"/*.root")

    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    pool = multiprocessing.Pool(processes=10)
    for baby_file in baby_files:
        job = pool.apply_async(get_nevents, args=(baby_file, return_dict))
    pool.close()
    pool.join()
    
    ntotalinput = 0
    ntotalprocessed = 0
    ntotalwgtprocessed = 0
    return_dict = dict(return_dict)
    keys = return_dict.keys()
    keys.sort()
    for baby_file in keys:
        ntotalinput += return_dict[baby_file][0]
        ntotalprocessed += return_dict[baby_file][1]
        ntotalwgtprocessed += return_dict[baby_file][2]
        #print baby_file, return_dict[baby_file]
    return_dict["n_total_input"] = ntotalinput
    return_dict["n_total_processed"] = ntotalprocessed
    return_dict["n_total_weighted_processed"] = ntotalwgtprocessed
    j = json.dumps(return_dict, indent=4)
    f = open(baby_dir + "/metadata.json", "w")
    print >> f, j
    f.close()

if __name__ == "__main__":

    def help():

        print "Usage:"
        print ""
        print "   python {} BABYDIRPATH".format(sys.argv[0])
        print ""
        print ""
        sys.exit(-1)


    try:
        baby_dir = sys.argv[1]
    except:
        help()

    make_metadata(baby_dir)


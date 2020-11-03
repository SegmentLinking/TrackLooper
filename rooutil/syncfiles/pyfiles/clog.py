#!/usr/bin/env python

import math, sys, os
import argparse
import commands

# OKGREEN = '\033[92m'
# FAIL = '\033[91m'
# ENDC = '\033[0m'

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('jobid', nargs="?", default=None)

    args = parser.parse_args()

    if not args.jobid:
        os.system("condor_history $USER -limit 20")
    else:
        status, output = commands.getstatusoutput("condor_history -l %s" % args.jobid)
        lines = output.split("\n")
        iwd = [line for line in lines if line.startswith("Iwd")][0].split("=",1)[-1].replace('"',"").strip()
        out = [line for line in lines if line.startswith("Out")][0].split("=",1)[-1].replace('"',"").strip()
        err = [line for line in lines if line.startswith("Err")][0].split("=",1)[-1].replace('"',"").strip()
        if not out.startswith("/"): out = iwd + "/" + out
        if not err.startswith("/"): err = iwd + "/" + err

        print out
        print err

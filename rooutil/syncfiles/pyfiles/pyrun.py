#!/usr/bin/env python

import time
import os
import sys
import commands
import argparse

try:
    from tqdm import format_meter, StatusPrinter
except:
    from pyfiles.tqdm import format_meter, StatusPrinter

from multiprocessing import Pool

"""
# Execute bash commands like xargs but with a prettier output

# Use 6 processes to sleep random amounts (piping in commands)
$ for i in `seq 1 50`; do echo sleep $(echo "$RANDOM/15000" | bc -l); done | pyrun.py -n 6

# Put commands in a file and use that as a command list
$ for i in `seq 1 50`; do echo sleep $(echo "$RANDOM/15000" | bc -l); done > cmds.sh
$ pyrun.py -f cmds.sh

# Use python API (basically a wrapper around multiprocessing libraries)
>>> from pyrun import Runner
>>> runner = Runner(nproc=5)
>>> runner.add_args(["sleep 1", "sleep 2", "sleep 3", "sleep 4"])
>>> runner.run()

# But what if I want to execute python functions and not shell commands?!
# Note that the function must take an index and then the args, for bookkeeping
>>> from pyrun import Runner
>>> def my_func((idx,args)):
        return idx, sum(args)
>>> runner = Runner(nproc=5, func=my_func)
>>> runner.add_args([(1,2),(3,4),(4,5),(6,7)])
>>> runner.run()
>>> print runner.get_outputs()
[(0, 3), (1, 7), (2, 9), (3, 13)]
"""

def f_test((idx,x)):
    time.sleep(x)
    toret = x+x
    return idx,toret

def f_bash((idx,x)):
    status, out = commands.getstatusoutput(x)
    return idx,status


class Runner(object):
    def __init__(self,nproc=5,func=f_bash,dot_type=1):
        self.pool = Pool(processes=nproc)
        self.args = []
        self.indices_with_args = []
        self.indices_status = []
        self.ntotal = 0
        self.ndone = 0
        self.elapsed = 0
        self.t0 = None
        self.func = func
        self.outputs = []
        self.dot_type = dot_type

    def add_args(self,args):
        next_idx = len(self.indices_with_args)
        for iarg,arg in enumerate(args):
            self.args.append(arg)
            self.indices_with_args.append((next_idx+iarg,arg))
            self.indices_status.append(0)
            self.ntotal += 1

    def get_args(self):
        # Make it possible to append to args during running
        # NOTE: This isn't actually possible because imap_unordered
        # listifies the whole generator first!!!
        i = 0
        while i < self.ntotal:
            yield self.indices_with_args[i]
            i += 1

    def get_outputs(self):
        # Return list of pairs (first element is the job index,
        # second element is the return value of the function)
        return self.outputs

    def get_dots(self,indices_status,which=1):
        # color = u"\033[92m"
        color = u"\033[38;2;12;185;103m"
        if which == 1:
            # return " "+"".join(u"\033[92m\u2022\033[0m" if x == 1 else u"\033[90m\u2219\033[0m" for x in indices_status)
            return " "+"".join(color+u"\u2022\033[0m" if x == 1 else u"\033[90m\u2219\033[0m" for x in indices_status)
        elif which in [2,3]:
            ncells = (6 if which == 2 else 8)
            chunks = [indices_status[i:i+ncells]+[0 for _ in range(ncells-len(indices_status[i:i+ncells]))] for i in range(0,len(indices_status),ncells)]
            chars = ""
            for chunk in chunks:
                on = [x for x in range(1,len(chunk)+1) if chunk[x-1]==1]
                char = ("\u"+hex(0x2800+reduce(lambda x,y:x|y,[1<<(x-1) for x in on],0))[2:]).decode("unicode-escape")
                chars += char
            return u" "+color+chars+u"\033[0m"

    def run(self):
        if not self.t0: 
            self.t0 = time.time()

        sp = StatusPrinter(sys.stderr)

        dots = self.get_dots(self.indices_status,which=self.dot_type)
        sp.print_status(format_meter(self.ndone, self.ntotal, self.elapsed, size=13,extra=dots))

        for idx,ret in self.pool.imap_unordered(self.func,self.get_args()):
            self.indices_status[idx] = True
            self.outputs.append((idx,ret))
            # Try to add another arg, but won't work (see reason in get_args)
            # if self.ndone == 6:
            #     self.add_args(["sleep 2"])
            self.ndone += 1
            self.elapsed = time.time()-self.t0
            dots = self.get_dots(self.indices_status,which=self.dot_type)
            sp.print_status(format_meter(self.ndone, self.ntotal, self.elapsed, size=13,extra=dots))
        print

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", help="text file with one command per line")
    parser.add_argument("-n", "--nproc", help="number of processes in pool", default="5")
    parser.add_argument("-d", "--dot_type", help="dot type (1,2,3)", default="1")
    cli_args = parser.parse_args()

    func_args = []
    if cli_args.file:
        fname = cli_args.file
        for line in open(fname,"r"):
            if line.strip():
                func_args.append(line.strip())
        print func_args
    else:
        for item in sys.stdin:
            func_args.append(item.strip())

    runner = Runner(nproc=int(cli_args.nproc),dot_type=int(cli_args.dot_type))
    runner.add_args(func_args)
    runner.run()


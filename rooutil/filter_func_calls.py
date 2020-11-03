#!/bin/env python

import os
import sys

#
# Searches function calls like "ptratio()" with just () ending in a source file
# Also can cross reference to CMS3.h or other class header files to eliminate ROOT or custom function calls with '()' ending
#

def help():
    print "python {} SOURCECODE [CLASSHEADERCODE]".format(sys.argv[0])
    print "e.g. python {} ScanChain.C CMS3.h".format(sys.argv[0])
    sys.exit()

fname = sys.argv[1]

f = open(fname)
lines = f.readlines()

rmchar = ".!,;=<>&\"?|*+/-:"
paran = "(){}[]"

# gather all the function calls ending with '()'
funcs = []
for line in lines:
    line = line.strip()
    for c in rmchar:
        line = line.replace(c, " ")
    if line.find("()") != -1:
        ls = line.split("()")
        for item in ls[:-1]:
            for p in paran:
                item = item.replace(p, " ")
            funcs.append(item.split()[-1])
funcs = list(set(funcs))
funcs.sort()

# if header file path provided gather all the available () and cross reference and print
if len(sys.argv) > 2:
    cname = sys.argv[2]
    c = open(cname)
    lines = c.readlines()
    funcnames = []
    for line in lines:
        line = line.strip()
        if line.find("&") != -1 and line.find("const") != -1:
            funcnames.append(line.split("&")[1].split("()")[0])
    funcnames = list(set(funcnames))
    funcnames.sort()

    for f in funcnames:
        if f in funcs:
            print f
else:
    for f in funcs:
        print f


# -*- coding: utf-8 -*-
#!/bin/env python

import sys
from  errors import E

def read_table(fname):

    f = open(fname)
    
    lines = [ l.strip() for l in f.readlines() ]
    
    categories = []
    yields = {}
    for line in lines:
        if "Bin#" in line:
            line = "".join(["	"] + line.split()[3:])
            line = line.replace("|", "			")
            categories = line.split()
            for category in categories:
                yields[category] = []
        if "Bin" in line:
            line = "".join(["	"] + line.split()[3:])
            line = line.replace("|", "	")
            line = line.replace(u"\u00B1".encode("utf-8"), ",")
            for category, item in zip(categories, line.split()):
                val, err = item.split(",")
                yields[category].append(E(float(val), float(err)))

    return yields

# -*- coding: utf-8 -*-
#!/bin/env python

import sys

f = open(sys.argv[1])

lines = [ l.strip() for l in f.readlines() ]

# for line in lines:
#     if "Bin#" in line:
#         line = "".join(["	"] + line.split()[3:])
#         line = line.replace("|", "			")
#         print line
#     if "Bin" in line:
#         line = "".join(["	"] + line.split()[3:])
#         line = line.replace("|", "	")
#         line = line.replace(u"\u00B1".encode("utf-8"), "	" + u"\u00B1".encode("utf-8") + "	")
#         print line

for line in lines:
    if "Bin#" in line:
        line = "".join(["	"] + line.split()[3:])
        line = line.replace("|", ",,,")
        print line
    if "Bin" in line:
        line = "".join(["	"] + line.split()[3:])
        line = line.replace("|", ",")
        line = line.replace(u"\u00B1".encode("utf-8"), "," + u"\u00B1".encode("utf-8") + ",")
        print line[:-1]

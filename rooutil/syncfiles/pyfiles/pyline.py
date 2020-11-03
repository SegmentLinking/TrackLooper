#!/usr/bin/env python

# Lets you use python syntax line-by-line for piped content
# Example:
#     ls -lrth ~ | pyline.py 'x.split("namin")[0]'

import math, sys, os
import datetime
import json
import ast

if __name__ == "__main__":
    pattern = None
    if(len(sys.argv) > 1): pattern = sys.argv[-1]
    for item in sys.stdin:
        # try:
            x = item.strip()
            if pattern: print eval(pattern)
            else: print x
        # except: pass


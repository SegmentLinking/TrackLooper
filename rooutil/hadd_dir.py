#!/bin/env python

import hadd
import sys

if len(sys.argv) == 2: hadd.hadd_dir(sys.argv[1])
if len(sys.argv) == 3: hadd.hadd_dir(sys.argv[1], "t", int(sys.argv[2]))

#!/usr/bin/env python

import math, sys, os

if __name__ == "__main__":
    if(len(sys.argv) < 2):
        print "first argument should be binwidth"
        sys.exit()

    binwidth=sys.argv[-1]
    # os.system("gnuplot -p -e 'set term dumb; binwidth=%s; bin(x,width)=width*floor(x/width); plot \"-\" using (bin($1,binwidth)):(1.0) smooth freq with boxes title \"\"'" % (binwidth))
    os.system("gnuplot -e 'set term dumb; binwidth=%s; bin(x,width)=width*floor(x/width); plot \"-\" using (bin($1,binwidth)):(1.0) smooth freq with boxes title \"\"'" % (binwidth))

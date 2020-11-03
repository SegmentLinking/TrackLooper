#!/usr/bin/env python

import math, sys, os
from pytable import Table
try:
    from collections import Counter
except:
    def Counter(vals):
        d = {}
        for v in vals:
            if v not in d: d[v] = 0
            d[v] += 1
        return d

def hum(num):
    # stolen from http://stackoverflow.com/questions/17973278/python-decimal-engineering-notation-for-mili-10e-3-and-micro-10e-6
    sgn = ''
    if num < 0: num, sgn = -num, '-'

    try: exp = int(math.floor(math.log10(num)))
    except: exp = 0

    exp3 = exp-(exp % 3)
    x3 = num/(10**exp3)

    if exp3 >= -24 and exp3 <= 24 and exp3 != 0: exp3_text = 'yzafpnum kMGTPEZY'[ ( exp3 - (-24)) / 3]
    elif exp3 == 0: exp3_text = ''
    else: exp3_text = 'e%s' % exp3
    return '%s%s%s' % (sgn,round(x3,2),exp3_text)

def statistics(ls):
    length = len(ls)
    totsum = sum(ls)
    mean = 1.0*totsum/length
    sigma = math.sqrt(1.0*sum([(mean-v)*(mean-v) for v in ls])/(length-1))
    maximum, minimum = max(ls), min(ls)
    #erroronmean = jackknife(ls)[1]
    return (length, mean, sigma, totsum, minimum, maximum, \
           hum(length), hum(mean), hum(sigma), hum(totsum), hum(minimum), hum(maximum))

def jackknife(ls):
    N = len(ls)
    totsum = sum(ls)
    avgs = []
    for e in ls:
        avgs.append( (totsum-e)/(N-1) )
    mu = float(sum(avgs))/len(avgs)
    sig = math.sqrt(1.0*N*sum([(mu-v)**2 for v in ls])/(N-1))
    return mu, sig

def freq(ls):
    dout = {}
    for elem in ls:
        if(elem not in dout.keys()): dout[elem] = 1
        else: dout[elem] += 1
    return dout

def makehisto(ls):
    d = freq(ls)
    maxval = max([d[k] for k in d.keys()])
    maxstrlen = max([len(k) for k in d.keys()])
    scaleto=80-maxstrlen
    for w in sorted(d, key=d.get, reverse=True):
        strbuff = "%%-%is | %%s (%%i)" % (maxstrlen)
        # strbuff = "%-9s | %s (%i)"
        if(maxval < scaleto):
            print strbuff % (w, "*" * d[w], d[w])
        else: # scale to scaleto width
            print strbuff % (w, "*" * max(1,int(float(scaleto)*d[w]/maxval)), d[w])

def get_table(vals, do_unicode=True, width=80):
    d = dict(Counter(vals))
    maxval = max([d[k] for k in d.keys()])
    def shorten(label):
        return label[:50]
    maxstrlen = max([len(shorten(k)) for k in d.keys()])
    scaleto=width-maxstrlen
    fillchar = "*"
    if do_unicode:
        fillchar = unichr(0x2589).encode('utf-8')
    tab = Table()
    for w in sorted(d, key=d.get, reverse=True):
        nfill = d[w] if maxval < scaleto else max(1,int(float(scaleto)*d[w]/maxval))
        strbuff = "{0} ({1})".format(fillchar*nfill,d[w])
        shortw = shorten(w)
        tab.add_row([shortw,strbuff])
    return tab

if __name__ == "__main__":
    do_ascii = False
    nums, words = [], []
    column = -1
    if(len(sys.argv) > 1): column = int(sys.argv[-1])
    for item in sys.stdin:
        try:
            if(column == -1): nums.append(float(item.strip()))
            else: nums.append(float(item.strip().split()[column-1]))
        except: 
            try:
                if(column == -1): words.append(item.strip())
                else: words.append(item.strip().split()[column-1])
            except: pass
        else: pass

    if(len(nums) <= 1):
        if(len(words) < 3):
            print "Can't calculate stuff with %i element!" % len(nums)
        else:
            # print "Found %i words, so histo will be made!" % len(words)
            if do_ascii:
                makehisto(words)
            else:
                get_table(words).print_table(ljustall=True, show_colnames=False)
    else: 
        print """
        length: {0} ({6})
        mean:   {1} ({7})
        sigma:  {2} ({8})
        sum:    {3} ({9})
        min:    {4} ({10})
        max:    {5} ({11})
        """.format(*statistics(nums))

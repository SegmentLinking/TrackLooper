#!/usr/bin/env python
# -*- coding: utf-8 -*-
__all__ = ['tqdm', 'trange']

import sys
import time
import os

GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
ENDC = '\033[0m'

PY2 = True
if sys.version_info[0] >= 3:
    PY2 = False
    unichr = chr

def hsv_to_rgb(h, s, v):
    if s == 0.0: v*=255; return [v, v, v]
    i = int(h*6.) # XXX assume int() truncates!
    f = (h*6.)-i; p,q,t = int(255*(v*(1.-s))), int(255*(v*(1.-s*f))), int(255*(v*(1.-s*(1.-f)))); v*=255; i%=6
    if i == 0: return [v, t, p]
    if i == 1: return [q, v, p]
    if i == 2: return [p, v, t]
    if i == 3: return [p, q, v]
    if i == 4: return [t, p, v]
    if i == 5: return [v, p, q]


def format_interval(t):
    mins, s = divmod(int(t), 60)
    h, m = divmod(mins, 60)
    if h:
        return '%d:%02d:%02d' % (h, m, s)
    else:
        return '%02d:%02d' % (m, s)


def format_meter(n, total, elapsed, do_rgb=True, do_ascii=False, size=35, extra=""):
    # n - number of finished iterations
    # total - total number of iterations, or None
    # elapsed - number of seconds passed since start
    if n > total:
        total = None
    
    elapsed_str = format_interval(elapsed)
    rate = '%5.2f' % (n / elapsed) if elapsed else '?'
    
    if total:
        frac = float(n) / total
        
        # N_BARS = 25
        # N_BARS = 75
        N_BARS = size
        # bar_length = int(frac*N_BARS)
        # bar = '#'*bar_length + '-'*(N_BARS-bar_length)
        percentage = '%3d%%' % (frac * 100)

        # if frac < 0.3:
        #     bar = RED+bar+ENDC
        # elif frac < 0.8:
        #     bar = YELLOW+bar+ENDC
        # else:
        #     bar = GREEN+bar+ENDC

        # bar = ""
        # for i in range(N_BARS):
        #     rgb = hsv_to_rgb(i*(100.0/N_BARS)*1.0/360,0.9,0.9)
        #     rgb = tuple(map(int, rgb))
        #     # print rgb
        #     if 1.0*i/N_BARS < frac:
        #         bar += "\033[48;2;%i;%i;%im \033[0m" % (rgb)
        #     else:
        #         bar += " "
        bar_length, frac_bar_length = divmod(int(frac * N_BARS * 8), 8)

        bar = ""
        rgb = (0,0,0)
        for i in range(bar_length):
            if 1.0*i/N_BARS < frac:
                if do_rgb:
                    rgb = hsv_to_rgb(1.0/3.6 * frac*i/bar_length,0.9,0.9)
                    rgb = tuple(map(int, rgb))
                bar += "\033[38;2;%i;%i;%im" % (rgb)
                if do_ascii:
                    bar += "%s\033[0m" % ("#")
                else:
                    bar += "%s\033[0m" % (unichr(0x2589))

        # FIXME this particular unicode character is shifted in mac terminal
        if frac_bar_length == 4: frac_bar_length += 1
        if frac_bar_length:
            bar += "\033[38;2;%i;%i;%im" % (rgb)
            if do_ascii:
                bar += "%s\033[0m" % ("#")
            else:
                bar += "%s\033[0m" % (unichr(0x2590-frac_bar_length))
        else:
            bar += ' '

        # whitespace padding
        if bar_length < N_BARS: bar += ' ' * max(N_BARS - bar_length - 1, 0)
        else: bar += ' ' * max(N_BARS - bar_length, 0)
        
        left_str = format_interval(elapsed / n * (total-n)) if n else '?'
        
        # return '|%s| %d/%d %s [elapsed: %s ETA: %s, %s Hz]' % (
        return '|%s| %d/%d %s [%s<%s, %s Hz]%s' % (
            bar, n, total, percentage, elapsed_str, left_str, rate, extra)
    
    else:
        return '%d [elapsed: %s, %s Hz]' % (n, elapsed_str, rate)


class StatusPrinter(object):
    def __init__(self, file):
        self.file = file
        self.last_printed_len = 0
    
    def print_status(self, s):
        if PY2:
            self.file.write('\r'+s.encode('utf-8')+' '*max(self.last_printed_len-len(s), 0))
        else:
            self.file.write('\r'+s+' '*max(self.last_printed_len-len(s), 0))
        # os.system('echo "\\033]1337;SetKeyLabel=F1=%s\\a"' % s.encode('utf-8').split("%")[-1])
        self.file.flush()
        self.last_printed_len = len(s)


def tqdm(iterable, desc='', total=None, leave=True, file=sys.stderr,
         mininterval=0.05, miniters=1, extra=""):
    """
    Get an iterable object, and return an iterator which acts exactly like the
    iterable, but prints a progress meter and updates it every time a value is
    requested.
    'desc' can contain a short string, describing the progress, that is added
    in the beginning of the line.
    'total' can give the number of expected iterations. If not given,
    len(iterable) is used if it is defined.
    'file' can be a file-like object to output the progress message to.
    If leave is False, tqdm deletes its traces from screen after it has
    finished iterating over all elements.
    If less than mininterval seconds or miniters iterations have passed since
    the last progress meter update, it is not updated again.
    """
    if total is None:
        try:
            total = len(iterable)
        except TypeError:
            total = None
    
    prefix = desc+': ' if desc else ''

    do_rgb = not os.getenv("STY")
    do_ascii = not not os.getenv("STY")
    
    sp = StatusPrinter(file)
    sp.print_status(prefix + format_meter(0, total, 0, do_rgb, do_ascii, extra=extra))
    
    start_t = last_print_t = time.time()
    last_print_n = 0
    n = 0
    for obj in iterable:
        yield obj
        # Now the object was created and processed, so we can print the meter.
        n += 1
        if n - last_print_n >= miniters:
            # We check the counter first, to reduce the overhead of time.time()
            cur_t = time.time()
            if cur_t - last_print_t >= mininterval:
                sp.print_status(prefix + format_meter(n, total, cur_t-start_t, do_rgb, do_ascii, extra=extra))
                last_print_n = n
                last_print_t = cur_t
    
    if not leave:
        sp.print_status('')
        sys.stdout.write('\r')
    else:
        if last_print_n < n:
            cur_t = time.time()
            sp.print_status(prefix + format_meter(n, total, cur_t-start_t, do_rgb, do_ascii, extra=extra))
        file.write('\n')


def trange(*args, **kwargs):
    """A shortcut for writing tqdm(range()) on py3 or tqdm(xrange()) on py2"""
    try:
        f = xrange
    except NameError:
        f = range
    
    return tqdm(f(*args), **kwargs)

def test():
    import time
    for i in tqdm(range(1000)):
        time.sleep(0.03)


if __name__ == "__main__":


    # can pipe two numbers (current, total) into tqdm
    # e.g.,
    #     for i in $(seq 1 100); do sleep 1s; echo $i 100 | tqdm.py ; done
    if not sys.stdin.isatty():
        content = str(sys.stdin.read()).strip()
        curr, tot = map(int,map(float,content.split()))
        tnow = time.time()

        if os.path.exists(".for_tqdm"):
            with open(".for_tqdm","r") as fhin:
                cthen, tthen = fhin.read().strip().split()
                cthen = int(cthen)
                tthen = float(tthen)
        else:
            cthen = 0
            tthen = 0

        # how much time elapsed from start
        if curr == cthen:
            elapsed = 999.
        else:
            elapsed = 1.0*curr/((curr-cthen)/(tnow-tthen))

        StatusPrinter(sys.stderr).print_status(format_meter(curr, tot, elapsed, True, False))
        sys.stdout.write('\r')

        with open(".for_tqdm","w") as fhout:
            fhout.write("{0} {1}".format(curr, tnow))

    else:
        test()


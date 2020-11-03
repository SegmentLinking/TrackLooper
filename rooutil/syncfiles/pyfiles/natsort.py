#!/usr/bin/env python

import sys
import re

def natsort(s):

    def soft_int(t):
        try:
            return int(t)
        except:
            return t.lower()
    return map(soft_int,re.split('([\-\+]\d+)', s))

if __name__ == "__main__":
    items = []
    for item in sys.stdin:
        items.append(item.strip())
    items = sorted(items, key=natsort)
    for item in items:
        print item



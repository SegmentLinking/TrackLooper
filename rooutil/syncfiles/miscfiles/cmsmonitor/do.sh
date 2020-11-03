#!/bin/bash

export SHELL=/bin/bash
export USER=namin
export LD_LIBRARY_PATH=/Users/namin/root/lib
export LIBPATH=/Users/namin/root/lib
export PATH=/Users/namin/root/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/X11/bin:/Library/TeX/texbin:/Users/namin/syncfiles/miscfiles/
export HOME=/Users/namin
export DYLD_LIBRARY_PATH=/Users/namin/root/lib
export PYTHONPATH=/Users/namin/root/lib:/Users/namin/syncfiles/pyfiles
export SHLIB_PATH=/Users/namin/root/lib
export DIR=/Users/namin/cron/monitor

touch $DIR/data.txt
python $DIR/monitor.py
python $DIR/plot.py
scp monitor.json namin@uaf-6.t2.ucsd.edu:~/public_html/
scp score.png namin@uaf-6.t2.ucsd.edu:~/public_html/
scp overview.html namin@uaf-6.t2.ucsd.edu:~/public_html/
scp {cms,lhc}.png namin@uaf-6.t2.ucsd.edu:~/public_html/

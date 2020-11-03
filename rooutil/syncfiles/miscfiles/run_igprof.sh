#!/usr/bin/env bash

tag=sichengIso_Mar29
pset=main_pset.py
igprof -d -pp -z -o igprof_${tag}.pp.gz cmsRun ${pset} >& igtest_${tag}.pp.log
# make sql file to put in web area
igprof-analyse --sqlite -d -v -g igprof_${tag}.pp.gz | sqlite3 igreport_${tag}_perf.sql3 >& /dev/null
# copy to uaf
cp igreport_${tag}_perf.sql3 ~/public_html/cgi-bin/data/
echo "http://uaf-6.t2.ucsd.edu/~namin/cgi-bin/igprof-navigator.py/igreport_${tag}_perf/"


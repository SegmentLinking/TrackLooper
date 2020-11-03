#!/bin/bash

igprof -pp -d -z -o igprof.pp.gz $*
cd /cvmfs/cms.cern.ch/slc6_amd64_gcc491/cms/cmssw-patch/CMSSW_7_4_7_patch1
eval `scramv1 runtime -sh`
cd -
# comment out the cmsRun line in run_igprof.sh and then run it again
igprof-analyse --sqlite -d -v -g igprof.pp.gz | sqlite3 igprof.pp.sql3 >& /dev/null
cp igprof.pp.sql3 ~/public_html/cgi-bin/data/
echo "http://${HOSTNAME}/~${USER}/cgi-bin/igprof-navigator.py/igprof.pp/"

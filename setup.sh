#!/bin/bash

source rooutil/thisrooutil.sh

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

#echo "Setting up ROOT"
# export SCRAM_ARCH=slc6_amd64_gcc530   # or whatever scram_arch you need for your desired CMSSW release
# export SCRAM_ARCH=slc6_amd64_gcc700   # or whatever scram_arch you need for your desired CMSSW release
# export CMSSW_VERSION=CMSSW_9_2_0
# export SCRAM_ARCH=slc6_amd64_gcc630   # or whatever scram_arch you need for your desired CMSSW release
# export CMSSW_VERSION=CMSSW_10_1_0
#export SCRAM_ARCH=slc7_amd64_gcc900   # or whatever scram_arch you need for your desired CMSSW release
#export SCRAM_ARCH=slc7_amd64_gcc700
#export CMSSW_VERSION=CMSSW_11_0_0_pre6
export SCRAM_ARCH=slc7_amd64_gcc900
export CMSSW_VERSION=CMSSW_11_2_0_pre5
source /cvmfs/cms.cern.ch/cmsset_default.sh
cd /cvmfs/cms.cern.ch/$SCRAM_ARCH/cms/cmssw/$CMSSW_VERSION/src
eval `scramv1 runtime -sh`
cd - > /dev/null

echo "Setup following ROOT.  Make sure it's slc7 variant. Otherwise the looper won't compile."
which root

#eof

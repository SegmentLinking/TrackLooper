#!/bin/bash

###########################################################################################################
# Setup environments
###########################################################################################################
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/code/rooutil/thisrooutil.sh
export SCRAM_ARCH=slc7_amd64_gcc900
export CMSSW_VERSION=CMSSW_11_2_0_pre5
source /cvmfs/cms.cern.ch/cmsset_default.sh
cd /cvmfs/cms.cern.ch/$SCRAM_ARCH/cms/cmssw/$CMSSW_VERSION/src
eval `scramv1 runtime -sh`
cd - > /dev/null
echo "Setup following ROOT.  Make sure it's slc7 variant. Otherwise the looper won't compile."
which root

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$DIR
export PATH=$PATH:$DIR/bin
export PATH=$PATH:$DIR/efficiency/bin
export TRACKLOOPERDIR=$DIR

###########################################################################################################
# Validation scripts
###########################################################################################################

# List of benchmark efficiencies are set as an environment variable
export LATEST_CPU_BENCHMARK_EFF_MUONGUN="/nfs-7/userdata/phchang/segmentlinking/benchmarks/7d8f188/eff_plots__CPU_7d8f188_muonGun/efficiencies.root"
export LATEST_CPU_BENCHMARK_EFF_PU200="/nfs-7/userdata/phchang/segmentlinking/benchmarks/3bb6b6b/PU200/eff_plots__CPU_3bb6b6b_PU200/efficiencies.root"

#eof

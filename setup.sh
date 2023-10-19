#!/bin/bash

###########################################################################################################
# Setup environments
###########################################################################################################
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/code/rooutil/thisrooutil.sh

export SCRAM_ARCH=el8_amd64_gcc11
export CMSSW_VERSION=CMSSW_13_0_0_pre4

source /cvmfs/cms.cern.ch/cmsset_default.sh
cd /cvmfs/cms.cern.ch/$SCRAM_ARCH/cms/cmssw/$CMSSW_VERSION/src
eval `scramv1 runtime -sh`
cd - > /dev/null
echo "Setup following ROOT. Make sure the appropriate setup file has been run. Otherwise the looper won't compile."
which root

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export LD_LIBRARY_PATH=$DIR/SDL/gpu:$DIR/SDL/cpu:$DIR:$LD_LIBRARY_PATH
export PATH=$DIR/bin:$PATH
export PATH=$DIR/efficiency/bin:$PATH
export PATH=$DIR/efficiency/python:$PATH
export TRACKLOOPERDIR=$DIR
export TRACKINGNTUPLEDIR=/data2/segmentlinking/CMSSW_12_2_0_pre2/
export LSTOUTPUTDIR=.

hostname=$(hostname)
if [[ $hostname == *cornell* ]]; then
  export LSTPERFORMANCEWEBDIR="/cdat/tem/${USER}/LSTPerformanceWeb"
else
  export LSTPERFORMANCEWEBDIR="/home/users/phchang/public_html/LSTPerformanceWeb"
fi

###########################################################################################################
# Validation scripts
###########################################################################################################

# List of benchmark efficiencies are set as an environment variable
export LATEST_CPU_BENCHMARK_EFF_MUONGUN="/data2/segmentlinking/muonGun_cpu_efficiencies.root"
export LATEST_CPU_BENCHMARK_EFF_PU200="/data2/segmentlinking/pu200_cpu_efficiencies.root"

# Alpaka, Boost, and CUDA dependencies
export BOOST_ROOT="/cvmfs/cms.cern.ch/el8_amd64_gcc11/external/boost/1.80.0-5305613b2f750cf1a05dcadf0d672647"
export ALPAKA_ROOT="/cvmfs/cms.cern.ch/el8_amd64_gcc11/external/alpaka/develop-20230621-9e2225ac6c979464a40749ef9d1e0331"
export CUDA_HOME=/cvmfs/cms.cern.ch/el8_amd64_gcc11/external/cuda/11.8.0-9f0af0f4206be7b705fe550319c49a11/
#eof

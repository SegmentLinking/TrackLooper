#!/bin/bash

###########################################################################################################
# Setup environments
###########################################################################################################
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/code/rooutil/thisrooutil.sh

export SCRAM_ARCH=el8_amd64_gcc10
export CMSSW_VERSION=CMSSW_12_5_0_pre5
export CUDA_HOME=/cvmfs/cms.cern.ch/el8_amd64_gcc10/external/cuda/11.5.2-c927b7e765e06433950d8a7eab9eddb4/

source /cvmfs/cms.cern.ch/cmsset_default.sh
cd /cvmfs/cms.cern.ch/$SCRAM_ARCH/cms/cmssw/$CMSSW_VERSION/src
eval `scramv1 runtime -sh`
cd - > /dev/null
echo "Setup following ROOT. Make sure the appropriate setup file has been run. Otherwise the looper won't compile."
which root

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export LD_LIBRARY_PATH=$DIR:$LD_LIBRARY_PATH
export PATH=$DIR/bin:$PATH
export PATH=$DIR/efficiency/bin:$PATH
export PATH=$DIR/efficiency/python:$PATH
export TRACKLOOPERDIR=$DIR
export TRACKINGNTUPLEDIR=/data2/segmentlinking/CMSSW_12_2_0_pre2/
export PIXELMAPDIR="/data2/segmentlinking/pixelmap_neta20_nphi72_nz24_ipt2"
export LSTOUTPUTDIR="."
export LSTPERFORMANCEWEBDIR="/cdat/tem/${USER}/LSTPerformanceWeb"

###########################################################################################################
# Validation scripts
###########################################################################################################

# List of benchmark efficiencies are set as an environment variable
export LATEST_CPU_BENCHMARK_EFF_MUONGUN="/data2/segmentlinking/muonGun_cpu_efficiencies.root"
export LATEST_CPU_BENCHMARK_EFF_PU200="/data2/segmentlinking/pu200_cpu_efficiencies.root"

source /cvmfs/cms.cern.ch/el8_amd64_gcc10/external/alpaka/develop-20220621-4e96939afa0cdb62448c73ead2bb07e0/etc/profile.d/init.sh
export BOOST_ROOT="/cvmfs/cms.cern.ch/el8_amd64_gcc10/external/boost/1.78.0-12075919175e8d078539685f9234134a"
export ALPAKA_ROOT="/cvmfs/cms.cern.ch/el8_amd64_gcc10/external/alpaka/develop-20220621-4e96939afa0cdb62448c73ead2bb07e0"
#eof

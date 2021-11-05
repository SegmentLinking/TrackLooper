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
export LD_LIBRARY_PATH=$DIR:$LD_LIBRARY_PATH
export PATH=$DIR/bin:$PATH
export PATH=$DIR/efficiency/bin:$PATH
export TRACKLOOPERDIR=$DIR
export PIXELMAPDIR="/data2/segmentlinking/pixelmap_neta20_nphi72_nz24_ipt2"

###########################################################################################################
# Validation scripts
###########################################################################################################

# List of benchmark efficiencies are set as an environment variable
export LATEST_CPU_BENCHMARK_EFF_MUONGUN="/data2/segmentlinking/muonGun_cpu_efficiencies.root"
export LATEST_CPU_BENCHMARK_EFF_PU200="/data2/segmentlinking/pu200_cpu_efficiencies.root"

export CUDA_PATH=/usr/local/cuda
export CUDABINDIR=${CUDA_PATH}/bin
export CUDALIBDIR=${CUDA_PATH}/lib64
export CUDAINCLUDE=${CUDA_PAT}/include
export PATH=${CUDABINDIR}:${MPIBINDIR}:$PATH
export LD_LIBRARY_PATH=${CUDAINCLUDE}:${CUDALIBDIR}:${NVIDIALIBDIR}:${MPILIBDIR}:$LD_LIBRARY_PATH
#eof

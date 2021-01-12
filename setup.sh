#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

source $DIR/code/rooutil/thisrooutil.sh

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

# Function to validate segment linking
validate_segment_linking() {

    if [[ ${1} == *"pionGun"* ]]; then
        PDGID=211
    elif [[ ${1} == *"muonGun"* ]]; then
        PDGID=13
    fi

    GITHASH=$(git rev-parse --short HEAD)
    OUTDIR=outputs_${GITHASH}_${1}
    NEVENTS=200 # Validationg first 200 events

    rm -rf ${OUTDIR}
    mkdir -p ${OUTDIR}

    # CPU baseline
    sh make_script.sh -m
    ./bin/sdl -n ${NEVENTS} -o ${OUTDIR}/cpu.root --cpu -i ${1}
    cd efficiency/
    sh run.sh -i ../${OUTDIR}/cpu.root -g ${PDGID} -p 4 -f
    cd ../

    run_gpu()
    {
        version=$1
        sample=$2
        shift
        shift
        # GPU unified
        sh make_script.sh -m $*
        ./bin/sdl -n ${NEVENTS} -o ${OUTDIR}/gpu_${version}.root -i ${sample}
        cd efficiency/
        sh run.sh -i ../${OUTDIR}/gpu_${version}.root -g ${PDGID} -p 4 -f
        cd ../
    }

    run_gpu unified ${1}
    run_gpu unified_cache ${1} -c
    run_gpu unified_cache_newgrid ${1} -c -g
    run_gpu unified_newgrid ${1} -g
    run_gpu explicit ${1} -x
    # run_gpu explicit_cache ${1} -x -c # Does not run on phi3
    # run_gpu explicit_cache_newgrid ${1} -x -c -g # Does not run on phi3
    run_gpu explicit_newgrid ${1} -x -g

    cd efficiency/
    sh compare.sh -i ../${OUTDIR}/cpu.root -f

}
export -f validate_segment_linking

#eof

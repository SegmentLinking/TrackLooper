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

###########################################################################################################
# Validation scripts
###########################################################################################################

# List of benchmark efficiencies are set as an environment variable
export LATEST_CPU_BENCHMARK_EFF_MUONGUN="/nfs-7/userdata/phchang/segmentlinking/benchmarks/7d8f188/eff_plots__CPU_7d8f188_muonGun/efficiencies.root"

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
export run_gpu

# Function to validate segment linking
validate_segment_linking() {

    if [[ ${1} == *"pionGun"* ]]; then
        PDGID=211
    elif [[ ${1} == *"muonGun"* ]]; then
        PDGID=13
    fi

    if [ -z ${2} ]; then
        NEVENTS=200 # If no number of events provided, validate on first 200 events
    else
        NEVENTS=${2} # If provided set the NEVENTS
    fi

    GITHASH=$(git rev-parse --short HEAD)
    OUTDIR=outputs_${GITHASH}_${1}

    # Delete old run
    rm -rf ${OUTDIR}
    mkdir -p ${OUTDIR}

    # Run different GPU configurations
    run_gpu unified ${1}
    run_gpu unified_cache ${1} -c
    run_gpu unified_cache_newgrid ${1} -c -g
    run_gpu unified_newgrid ${1} -g
    run_gpu explicit ${1} -x
    # run_gpu explicit_cache ${1} -x -c # Does not run on phi3
    # run_gpu explicit_cache_newgrid ${1} -x -c -g # Does not run on phi3
    run_gpu explicit_newgrid ${1} -x -g

    cd efficiency/
    sh compare.sh -i ${1} -t ${GITHASH} -f
    cd ../

}
export -f validate_segment_linking

#eof

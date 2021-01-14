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

run_gpu()
{
    version=$1
    sample=$2
    shift
    shift
    # GPU unified
    make_tracklooper -m $*
    sdl -n ${NEVENTS} -o ${OUTDIR}/gpu_${version}.root -i ${sample}
    make_efficiency -i ${OUTDIR}/gpu_${version}.root -g ${PDGID} -p 4 -f
}
export run_gpu

# Function to validate segment linking
validate_segment_linking() {

    SAMPLE=${1}
    if [[ ${SAMPLE} == *"pionGun"* ]]; then
        PDGID=211
    elif [[ ${SAMPLE} == *"muonGun"* ]]; then
        PDGID=13
    elif [[ ${SAMPLE} == *"PU200"* ]]; then
        PDGID=0
    fi

    if [ -z ${2} ]; then
        NEVENTS=200 # If no number of events provided, validate on first 200 events
        if [[ ${SAMPLE} == *"PU200"* ]]; then
            NEVENTS=30 # If PU200 then run 30 events
        fi
    else
        NEVENTS=${2} # If provided set the NEVENTS
    fi

    SPECIFICGPUVERSION=${3}

    GITHASH=$(git rev-parse --short HEAD)
    DIRTY=$(cat gitversion.txt | tail -n2)
    if [[ "git diff  " == "$(cat .gitversion.txt | tail -n2 | tr '\n' ' ')" ]]; then
        DIRTY="";
    else
        DIRTY="DIRTY";
    fi
    GITHASH=${GITHASH}${DIRTY}

    OUTDIR=outputs_${GITHASH}_${SAMPLE}

    # Delete old run
    rm -rf ${OUTDIR}
    mkdir -p ${OUTDIR}

    # Run different GPU configurations
    if [ -z ${SPECIFICGPUVERSION} ] || [[ "${SPECIFICGPUVERSION}" == "unified" ]]; then
        run_gpu unified ${SAMPLE}
    fi
    if [ -z ${SPECIFICGPUVERSION} ] || [[ "${SPECIFICGPUVERSION}" == "unified_cache" ]]; then
        run_gpu unified_cache ${SAMPLE} -c
    fi
    if [ -z ${SPECIFICGPUVERSION} ] || [[ "${SPECIFICGPUVERSION}" == "unified_cache_newgrid" ]]; then
        run_gpu unified_cache_newgrid ${SAMPLE} -c -g
    fi
    if [ -z ${SPECIFICGPUVERSION} ] || [[ "${SPECIFICGPUVERSION}" == "unified_newgrid" ]]; then
        run_gpu unified_newgrid ${SAMPLE} -g
    fi
    if [ -z ${SPECIFICGPUVERSION} ] || [[ "${SPECIFICGPUVERSION}" == "explicit" ]]; then
        run_gpu explicit ${SAMPLE} -x
    fi
    if [ -z ${SPECIFICGPUVERSION} ] || [[ "${SPECIFICGPUVERSION}" == "explicit_cache" ]]; then
        # run_gpu explicit_cache ${SAMPLE} -x -c # Does not run on phi3
        :
    fi
    if [ -z ${SPECIFICGPUVERSION} ] || [[ "${SPECIFICGPUVERSION}" == "explicit_cache_newgrid" ]]; then
        # run_gpu explicit_cache_newgrid ${SAMPLE} -x -c -g # Does not run on phi3
        :
    fi
    if [ -z ${SPECIFICGPUVERSION} ] || [[ "${SPECIFICGPUVERSION}" == "explicit_newgrid" ]]; then
        run_gpu explicit_newgrid ${SAMPLE} -x -g
    fi

    compare_efficiencies -i ${SAMPLE} -t ${GITHASH} -f

}
export -f validate_segment_linking

#eof

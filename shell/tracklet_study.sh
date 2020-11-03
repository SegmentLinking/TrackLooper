#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

TRACKLOOPERBASE=$(dirname $DIR)

TRACKINGNTUPLEDIR=/home/users/phchang/public_html/analysis/sdl/trackingNtuple

usage() {
    echo "Usage:"
    echo ""
    echo "  sh $0 SAMPLE TAG [DESCRIPTION]"
    echo ""
    echo "  SAMPLE = 0  : pu200 ttbar (500 evt) with tracklet"
    echo "           1  : 0.95 to 1.05 GeV 10-mu-gun"
    echo "           2  : 0.5 to 2.0 GeV 100-mu-gun"
    echo "           3  : 0.5 to 3.0 GeV 10-mu-gun"
    echo "           200: e 200 10-mu-gun"
    echo "           mu : Various mu-gun sample hadded"
    exit
}

if [ -z $1 ]; then usage; fi
if [ -z $2 ]; then usage; fi

JOBTAG=$2
DATASET=$1

if [[ $DATASET == "0" ]]; then SAMPLETAG=pu200; fi
if [[ $DATASET == "1" ]]; then SAMPLETAG=pt0p95_1p05; fi
if [[ $DATASET == "2" ]]; then SAMPLETAG=pt0p5_2p0; fi
if [[ $DATASET == "3" ]]; then SAMPLETAG=pt0p5_3p0; fi
if [[ $DATASET == "200" ]]; then SAMPLETAG=e200; fi
if [[ $DATASET == "mu" ]]; then SAMPLETAG=mu; fi

if [ -z ${SAMPLETAG} ]; then usage; fi

OUTPUTFILEBASENAME=fulleff_${SAMPLETAG}
OUTPUTFILE=fulleff_${SAMPLETAG}.root
OUTPUTDIR=${TRACKLOOPERBASE}/results/tracklet_study/${SAMPLETAG}_${JOBTAG}/

# Only if the directory does not exist one runs it again
if [ ! -d "${OUTPUTDIR}" ]; then

    mkdir -p ${OUTPUTDIR}
    rm -f ${OUTPUTDIR}/${OUTPUTFILE}
    rm -f ${OUTPUTDIR}/${OUTPUTFILEBASENAME}_*.root
    rm -f ${OUTPUTDIR}/${OUTPUTFILEBASENAME}_*.log

    if [[ $DATASET == "0" ]]; then
        SAMPLETYPE=pu200
        rm ${SAMPLETYPE}_*_${TAG}.root
        PU200SAMPLE=/nfs-7/userdata/bsathian/SDL_trackingNtuple/ttbar_highPU/trackingNtuple_with_PUinfo_500_evts.root
        NJOBS=20
        echo "" > .jobs.txt
        for i in $(seq 0 $((NJOBS-1))); do
            echo "time ./bin/sdl -m 4 -i ${PU200SAMPLE} -n -1 -t trackingNtuple/tree -o ${OUTPUTDIR}/${OUTPUTFILEBASENAME}_${i}.root -v 1 -x ${i} > ${OUTPUTDIR}/${OUTPUTFILEBASENAME}_${i}.log 2>&1 " >>  .jobs.txt
        done
    fi

    if [[ $DATASET == "1" ]]; then
        SAMPLETYPE=pt0p95_1p05
        rm ${SAMPLETYPE}_*_${TAG}.root
        NJOBS=1
        for i in $(seq 0 $((NJOBS-1))); do
            echo "time ./bin/sdl -m 4 -i ${TRACKINGNTUPLEDIR}/CMSSW_10_4_0/src/trackingNtuple_pt0p95_1p05_10MuGun.root -n -1 -t trackingNtuple/tree -o ${OUTPUTDIR}/${OUTPUTFILEBASENAME}_${i}.root -v 0 > ${OUTPUTDIR}/${OUTPUTFILEBASENAME}_${i}.log  2>&1 " >  .jobs.txt
        done
    fi

    if [[ $DATASET == "2" ]]; then
        SAMPLETYPE=pt0p5_2p0
        rm ${SAMPLETYPE}_*_${TAG}.root
        NJOBS=20
        echo "" > .jobs.txt
        for i in $(seq 0 $((NJOBS-1))); do
            echo "time ./bin/sdl -m 4 -j ${NJOBS} -I ${i} -i ${TRACKINGNTUPLEDIR}/CMSSW_10_4_0/src/trackingNtuple_100_pt0p5_2p0.root -n -1 -t trackingNtuple/tree -o ${OUTPUTDIR}/${OUTPUTFILEBASENAME}_${i}.root -v 0 > ${OUTPUTDIR}/${OUTPUTFILEBASENAME}_${i}.log  2>&1 " >>  .jobs.txt
        done
    fi

    if [[ $DATASET == "3" ]]; then
        SAMPLETYPE=pt0p5_3p0
        rm ${SAMPLETYPE}_*_${TAG}.root
        NJOBS=20
        echo "" > .jobs.txt
        for i in $(seq 0 $((NJOBS-1))); do
            echo "time ./bin/sdl -m 4 -j ${NJOBS} -I ${i} -i ${TRACKINGNTUPLEDIR}/CMSSW_10_4_0/src/trackingNtuple_100_pt0p5_3p0.root -n -1 -t trackingNtuple/tree -o ${OUTPUTDIR}/${OUTPUTFILEBASENAME}_${i}.root -v 0 > ${OUTPUTDIR}/${OUTPUTFILEBASENAME}_${i}.log  2>&1 " >>  .jobs.txt
        done
    fi

    if [[ $DATASET == "200" ]]; then
        SAMPLETYPE=e200
        rm ${SAMPLETYPE}_*_${TAG}.root
        NJOBS=20
        SAMPLEPATH=/hadoop/cms/store/user/slava77/CMSSW_10_4_0_patch1-tkNtuple/pass-e072c1a/27411.0_TenMuExtendedE_0_200/trackingNtuple.root
        echo "" > .jobs.txt
        for i in $(seq 0 $((NJOBS-1))); do
            echo "time ./bin/sdl -m 4 -j ${NJOBS} -I ${i} -i ${SAMPLEPATH} -n -1 -t trackingNtuple/tree -o ${OUTPUTDIR}/${OUTPUTFILEBASENAME}_${i}.root -v 0 > ${OUTPUTDIR}/${OUTPUTFILEBASENAME}_${i}.log  2>&1 " >>  .jobs.txt
        done
    fi

    if [[ $DATASET == "mu" ]]; then
        SAMPLETYPE=mu
        rm ${SAMPLETYPE}_*_${TAG}.root
        NJOBS=20
        SAMPLEPATH=${TRACKINGNTUPLEDIR}/CMSSW_10_4_0/src/trackingNtuple_ensemble_muon_guns.root
        echo "" > .jobs.txt
        for i in $(seq 0 $((NJOBS-1))); do
            echo "time ./bin/sdl -m 4 -j ${NJOBS} -I ${i} -i ${SAMPLEPATH} -n -1 -t trackingNtuple/tree -o ${OUTPUTDIR}/${OUTPUTFILEBASENAME}_${i}.root -v 0 > ${OUTPUTDIR}/${OUTPUTFILEBASENAME}_${i}.log  2>&1 " >>  .jobs.txt
        done
    fi

    echo "<== Submitting parallel jobs ..."
    sleep 1
    sh rooutil/xargs.sh -n 10 .jobs.txt
    wait

    # Hadding outputs
    hadd -f ${OUTPUTDIR}/${OUTPUTFILE} ${OUTPUTDIR}/${OUTPUTFILEBASENAME}_*.root

    # Writing some descriptive file
    echo "$3" > ${OUTPUTDIR}/description.txt
    git status >> ${OUTPUTDIR}/description.txt
    git diff >> ${OUTPUTDIR}/description.txt
    git log --graph --pretty=format:'%Cred%h%Creset -%C(yellow)%d%Creset %s %Cgreen(%cr) %C(bold blue)<%an>%Creset' --abbrev-commit >> ${OUTPUTDIR}/description.txt
    cd ${TRACKLOOPERBASE}/SDL/
    git status >> ${OUTPUTDIR}/description.txt
    git diff >> ${OUTPUTDIR}/description.txt
    git log --graph --pretty=format:'%Cred%h%Creset -%C(yellow)%d%Creset %s %Cgreen(%cr) %C(bold blue)<%an>%Creset' --abbrev-commit >> ${OUTPUTDIR}/description.txt
    cd ../

    # plotting
    cd ${OUTPUTDIR}
    python ${TRACKLOOPERBASE}/python/plot.py 5 ${OUTPUTFILE}

    # # Creating html for easier efficiency plots viewing
    # echo ""

    # echo "Creating HTML from markdown"
    # cd ${OUTPUTDIR}/
    # sh $DIR/write_markdown.sh ${SAMPLETAG} "$3" ${JOBTAG}

    # echo ""

    function getlink {
        echo ${PWD/\/home\/users\/phchang\/public_html/http:\/\/snt:tas@uaf-10.t2.ucsd.edu\/~$USER/}/$1
    }
    export -f getlink

    echo ">>> results are in ${OUTPUTDIR}"
    echo ">>> results can also be viewed via following link:"
    echo ">>>   $(getlink)"

else

    echo "The command has been already ran before"
    echo "i.e. $OUTPUTDIR already exists"
    echo "Delete $OUTPUTDIR to re-run"

fi


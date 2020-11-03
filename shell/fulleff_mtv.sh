#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

TRACKLOOPERBASE=$(dirname $DIR)

trap "kill 0" EXIT

TRACKINGNTUPLEDIR=/home/users/phchang/public_html/analysis/sdl/trackingNtuple

usage() {
    echo "Usage:"
    echo ""
    echo "  sh $0 SAMPLE TAG [DESCRIPTION]"
    echo ""
    echo "  SAMPLE = 1  : pt 0.5 to 2.0 hundred-mu-gun sample"
    echo "           2  : e 0 to 200 ten-mu-gun sample"
    echo "           3  : pu200 ttbar (252 events without sim info on pileup tracks) with doAnalysis option --pdg_id == 13"
    echo "           4  : pu200 ttbar (252 events without sim info on pileup tracks) with doAnalysis option --pdg_id == 211"
    echo "           5  : pu200 ttbar (252 events without sim info on pileup tracks) with doAnalysis option --pdg_id == 11"
    echo "           6  : DO NOT USE: displaced SUSY stuff (not sure what)"
    echo "           7  : pu200 ctau100 of some SUSY? with doAnalysis option --pdg_id == 13"
    echo "           8  : pu200 ctau100 of some SUSY? with doAnalysis option --pdg_id == 211"
    echo "           9  : pu200 ctau100 of some SUSY? with doAnalysis option --pdg_id == 11"
    echo "           10 : pt 0.5 to 3.0 hundred-mu-gun sample"
    echo "           11 : pt 0.5 to 5.0 ten-mu-gun sample"
    echo "           12 : pu200 ttbar (500 evt) with doAnalysis option --pdg_id == 13"
    echo "           13 : pu200 ttbar (500 evt) with doAnalysis option --pdg_id == 211"
    echo "           14 : pu200 ttbar (500 evt) with doAnalysis option --pdg_id == 11"
    echo "           15 : pu200 ttbar (500 evt) with doAnalysis option --pdg_id == 1 (1 means all charged particle)"
    echo "           16 : pt 0.5 to 50  hundred-mu-gun sample"
    echo "           17 : 5 cm 'cube' with pt 0.5 to 50 ten-mu-gun"
    exit
}

if [ -z $1 ]; then usage; fi
if [ -z $2 ]; then usage; fi
if [ -z "$3" ]; then usage; fi

if [[ $1 == "1" ]]; then
SAMPLE=${TRACKINGNTUPLEDIR}/CMSSW_10_4_0/src/trackingNtuple_pt0p5_1p5_10MuGun.root
SAMPLE=${TRACKINGNTUPLEDIR}/CMSSW_10_4_0/src/trackingNtuple_100_pt0p5_2p0.root
SAMPLETAG=pt0p5_2p0
SAMPLETAG=pt0p5_2p0_nosimhit13requirement
SAMPLETAG=pt0p5_2p0_nosimhit13requirement_nocurlers
SAMPLETAG=pt0p5_2p0_nosimhit13requirement_nocurlers_v1
SAMPLETAG=pt0p5_2p0_nosimhit13requirement_nocurlers_v2
SAMPLETAG=pt0p5_2p0_nosimhit13requirement_pdgid13
SAMPLETAG=pt0p5_2p0_tracklet_via_map
SAMPLETAG=pt0p5_2p0
PDGID=13
PTBOUND=4
NEVENTS=-1
DOPARALLEL=false
fi

if [[ $1 == "2" ]]; then
SAMPLE=/hadoop/cms/store/user/slava77/CMSSW_10_4_0_patch1-tkNtuple/pass-e072c1a/27411.0_TenMuExtendedE_0_200/trackingNtuple.root
SAMPLETAG=e200
PDGID=13
PTBOUND=0
NEVENTS=-1
DOPARALLEL=false
fi

if [[ $1 == "3" ]]; then
SAMPLE=${TRACKINGNTUPLEDIR}/CMSSW_10_4_0_mtd5/src/trackingNtuple_TTbar_PU200.root
SAMPLETAG=pu200_pdgid13
PDGID=13
PTBOUND=0
NEVENTS=252
DOPARALLEL=true
fi

if [[ $1 == "4" ]]; then
SAMPLE=${TRACKINGNTUPLEDIR}/CMSSW_10_4_0_mtd5/src/trackingNtuple_TTbar_PU200.root
SAMPLETAG=pu200_pdgid211
PDGID=211
PTBOUND=0
NEVENTS=252
DOPARALLEL=true
fi

if [[ $1 == "5" ]]; then
SAMPLE=${TRACKINGNTUPLEDIR}/CMSSW_10_4_0_mtd5/src/trackingNtuple_TTbar_PU200.root
SAMPLETAG=pu200_pdgid11
PDGID=11
PTBOUND=0
NEVENTS=252
DOPARALLEL=true
fi

if [[ $1 == "6" ]]; then
SAMPLE=${TRACKINGNTUPLEDIR}/CMSSW_10_4_0_mtd5/src/trackingNtuple_DisplacedSUSY_stopToChi_Gravitino_M_1000_700_10mm.root
SAMPLETAG=pu200_displaced_pdgid13
PDGID=13
PTBOUND=0
NEVENTS=252
DOPARALLEL=true
fi

if [[ $1 == "7" ]]; then
SAMPLE=${TRACKINGNTUPLEDIR}/CMSSW_10_4_0_mtd5/src/trackingNtuple_GMSB_L100_Ctau100.root
SAMPLETAG=pu200_displaced_gmsb_ctau100_pdgid13
PDGID=13
PTBOUND=0
NEVENTS=252
DOPARALLEL=true
fi

if [[ $1 == "8" ]]; then
SAMPLE=${TRACKINGNTUPLEDIR}/CMSSW_10_4_0_mtd5/src/trackingNtuple_GMSB_L100_Ctau100.root
SAMPLETAG=pu200_displaced_gmsb_ctau100_pdgid11
PDGID=11
PTBOUND=0
NEVENTS=252
DOPARALLEL=true
fi

if [[ $1 == "9" ]]; then
SAMPLE=${TRACKINGNTUPLEDIR}/CMSSW_10_4_0_mtd5/src/trackingNtuple_GMSB_L100_Ctau100.root
SAMPLETAG=pu200_displaced_gmsb_ctau100_pdgid211
PDGID=211
PTBOUND=0
NEVENTS=252
DOPARALLEL=true
fi

if [[ $1 == "10" ]]; then
SAMPLE=${TRACKINGNTUPLEDIR}/CMSSW_10_4_0/src/trackingNtuple_100_pt0p5_3p0.root
SAMPLETAG=pt0p5_3p0
PDGID=13
PTBOUND=5
NEVENTS=-1
DOPARALLEL=false
fi

if [[ $1 == "11" ]]; then
SAMPLE=${TRACKINGNTUPLEDIR}/CMSSW_10_4_0/src/trackingNtuple_10_pt0p5_5p0.root
SAMPLETAG=pt0p5_5p0
PDGID=13
PTBOUND=6
NEVENTS=-1
DOPARALLEL=false
fi

if [[ $1 == "12" ]]; then
SAMPLE=/nfs-7/userdata/bsathian/SDL_trackingNtuple/ttbar_highPU/trackingNtuple_with_PUinfo_500_evts.root
SAMPLETAG=pu200_w_truth_pdgid13
PDGID=13
PTBOUND=0
NEVENTS=-1
DOPARALLEL=true
fi

if [[ $1 == "13" ]]; then
SAMPLE=/nfs-7/userdata/bsathian/SDL_trackingNtuple/ttbar_highPU/trackingNtuple_with_PUinfo_500_evts.root
SAMPLETAG=pu200_w_truth_pdgid211
PDGID=211
PTBOUND=0
NEVENTS=-1
DOPARALLEL=true
fi

if [[ $1 == "14" ]]; then
SAMPLE=/nfs-7/userdata/bsathian/SDL_trackingNtuple/ttbar_highPU/trackingNtuple_with_PUinfo_500_evts.root
SAMPLETAG=pu200_w_truth_pdgid11
PDGID=11
PTBOUND=0
NEVENTS=-1
DOPARALLEL=true
fi

if [[ $1 == "15" ]]; then
SAMPLE=/nfs-7/userdata/bsathian/SDL_trackingNtuple/ttbar_highPU/trackingNtuple_with_PUinfo_500_evts.root
SAMPLETAG=pu200_w_truth_charged
PDGID=0
PTBOUND=0
NEVENTS=-1
DOPARALLEL=true
fi

if [[ $1 == "16" ]]; then
SAMPLE=${TRACKINGNTUPLEDIR}/CMSSW_10_4_0/src/trackingNtuple_100_pt0p5_50.root,${TRACKINGNTUPLEDIR}/CMSSW_10_4_0/src/trackingNtuple_100_pt0p5_2p0.root
SAMPLETAG=pt0p5_2p0_50
PDGID=13
PTBOUND=7
NEVENTS=-1
DOPARALLEL=false
fi

if [[ $1 == "17" ]]; then
SAMPLE=${TRACKINGNTUPLEDIR}/CMSSW_10_4_0/src/trackingNtuple_10_pt0p5_50_5cm_cube.root
SAMPLETAG=pt0p5_50_5cm_cube
PDGID=13
PTBOUND=7
NEVENTS=-1
DOPARALLEL=false
fi

JOBTAG=$2

OUTPUTFILEBASENAME=fulleff_${SAMPLETAG}
OUTPUTFILE=fulleff_${SAMPLETAG}_mtv.root
OUTPUTDIR=${TRACKLOOPERBASE}/results/mtv_eff/${SAMPLETAG}_${JOBTAG}/

# Only if the directory does not exist one runs it again
if [ ! -d "${OUTPUTDIR}" ]; then

    mkdir -p ${OUTPUTDIR}
    rm -f ${OUTPUTDIR}/${OUTPUTFILE}
    rm -f ${OUTPUTDIR}/${OUTPUTFILEBASENAME}_*.root
    rm -f ${OUTPUTDIR}/${OUTPUTFILEBASENAME}_*.log

    # If need to run parallel run in parallel
    if [ $DOPARALLEL = true ]; then
        echo "Running parallel"
        rm .jobs.txt
        for i in $(seq 0 10); do
            CMD="${TRACKLOOPERBASE}/bin/mtv -v 1 -i ${SAMPLE} -t trackingNtuple/tree -n ${NEVENTS} -p ${PTBOUND} -o ${OUTPUTDIR}/${OUTPUTFILEBASENAME}_${i}.root -g ${PDGID} -x ${i} > ${OUTPUTDIR}/${OUTPUTFILEBASENAME}_${i}.log 2>&1"
            echo $CMD >> .jobs.txt
        done
        sh rooutil/xargs.sh -n 10 .jobs.txt
    else
        rm .jobs.txt
        NJOBS=16
        for i in $(seq 0 $((NJOBS-1))); do
            CMD="${TRACKLOOPERBASE}/bin/mtv -j ${NJOBS} -I ${i} -i ${SAMPLE} -t trackingNtuple/tree -n ${NEVENTS} -p ${PTBOUND} -o ${OUTPUTDIR}/${OUTPUTFILEBASENAME}_${i}.root -g ${PDGID} > ${OUTPUTDIR}/${OUTPUTFILEBASENAME}_${i}.log 2>&1"
            echo $CMD >> .jobs.txt
        done
        sh rooutil/xargs.sh -n 10 .jobs.txt
    fi

    # Hadding output
    HADDOUTPUTFILE=${OUTPUTDIR}/fulleff_${SAMPLETAG}_mtv.root
    rm -f ${HADDOUTPUTFILE}
    hadd -f ${HADDOUTPUTFILE} ${OUTPUTDIR}/fulleff_${SAMPLETAG}*.root

    #
    # Plotting
    #
    cd ${OUTPUTDIR}
    python ${TRACKLOOPERBASE}/python/plot.py 8 ${OUTPUTFILE} ${SAMPLETAG}

    echo "$3" > description.txt

    # Creating html for easier efficiency plots viewing
    echo ""
    
    echo "Creating HTML from markdown"
    cd ${OUTPUTDIR}/
    sh $DIR/write_markdown.sh ${SAMPLETAG} "$3" ${JOBTAG} mtv_eff
    
    echo ""
    
    function getlink {
        echo ${PWD/\/home\/users\/phchang\/public_html/http:\/\/snt:tas@uaf-10.t2.ucsd.edu\/~$USER/}/$1
    }
    
    export -f getlink
    echo ">>> results are in ${OUTPUTDIR}"
    echo ">>> results can also be viewed via following link:"
    echo ">>>   $(getlink)"

else

    #
    # Plotting
    #
    cd ${OUTPUTDIR}
    python ${TRACKLOOPERBASE}/python/plot.py 8 ${OUTPUTFILE} ${SAMPLETAG}

    echo "$3" > description.txt

    # Creating html for easier efficiency plots viewing
    echo ""
    
    echo "Creating HTML from markdown"
    cd ${OUTPUTDIR}/
    sh $DIR/write_markdown.sh ${SAMPLETAG} "$3" ${JOBTAG} mtv_eff
    
    echo ""
    
    function getlink {
        echo ${PWD/\/home\/users\/phchang\/public_html/http:\/\/snt:tas@uaf-10.t2.ucsd.edu\/~$USER/}/$1
    }
    
    export -f getlink
    echo ">>> results are in ${OUTPUTDIR}"
    echo ">>> results can also be viewed via following link:"
    echo ">>>   $(getlink)"

    echo "The command has been already ran before"
    echo "So I re-ran the plotting only"
    echo "i.e. $OUTPUTDIR already exists"
    echo "Delete $OUTPUTDIR to re-run"

fi


#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

TRACKLOOPERBASE=$(dirname $DIR)

# trap "kill 0" EXIT

TRACKINGNTUPLEDIR=/home/users/phchang/public_html/analysis/sdl/trackingNtuple

usage() {
    echo "Usage:"
    echo ""
    echo "  sh $0 OPTIONS"
    echo ""
    echo "  Options with arguments"
    echo "    -d DESCRIPTION    (description to save)"
    echo "    -s SAMPLEID       (sample ID see below for the list of sample ID)"
    echo "    -t JOBTAG         (job tag)"
    echo "   [-n NMATCH]        (nmatch used for matching between track candidate to sim track for mtv-like plots)"
    echo ""
    echo "  Options without arguemtns"
    echo "    -m (to do do mtv-like efficiency)"
    echo "    -w (to write SDL ntuples)"
    echo ""
    echo "  SAMPLEID = 1  : pt 0.5 to 2.0 hundred-mu-gun sample"
    echo "             2  : e 0 to 200 ten-mu-gun sample"
    echo "             3  : pu200 ttbar (252 events without sim info on pileup tracks) with sdl option --pdg_id == 13"
    echo "             4  : pu200 ttbar (252 events without sim info on pileup tracks) with sdl option --pdg_id == 211"
    echo "             5  : pu200 ttbar (252 events without sim info on pileup tracks) with sdl option --pdg_id == 11"
    echo "             6  : DO NOT USE: displaced SUSY stuff (not sure what)"
    echo "             7  : pu200 ctau100 of some SUSY? with sdl option --pdg_id == 13"
    echo "             8  : pu200 ctau100 of some SUSY? with sdl option --pdg_id == 211"
    echo "             9  : pu200 ctau100 of some SUSY? with sdl option --pdg_id == 11"
    echo "             10 : pt 0.5 to 3.0 hundred-mu-gun sample"
    echo "             11 : pt 0.5 to 5.0 ten-mu-gun sample"
    echo "             12 : pu200 ttbar (500 evt) with sdl option --pdg_id == 13"
    echo "             13 : pu200 ttbar (500 evt) with sdl option --pdg_id == 211"
    echo "             14 : pu200 ttbar (500 evt) with sdl option --pdg_id == 11"
    echo "             15 : pu200 ttbar (500 evt) with sdl option --pdg_id == 0 (0 means all charged particle)"
    echo "             17 : 5 cm 'cube' with pt 0.5 to 50 ten-mu-gun"
    echo "             18 : 50 cm 'cube' with pt 0.5 to 50 ten-mu-gun"
    echo "             19 : ttbar (pu 200?)"
    echo "             20 : 100 pion gun pt 0.5 to 50 GeV"
    exit
}

# Command-line opts
DOMTV=false
WRITESDLNTUPLE=false
while getopts ":d:s:t:n:mwh" OPTION; do
  case $OPTION in
    d) DESCRIPTION=${OPTARG};;
    s) SAMPLEID=${OPTARG};;
    t) JOBTAG=${OPTARG};;
    n) NMATCH=${OPTARG};;
    m) DOMTV=true;;
    w) WRITESDLNTUPLE=true;;
    h) usage;;
    :) usage;;
  esac
done

if [ -z "${DESCRIPTION}" ]; then usage; fi
if [ -z ${SAMPLEID}  ]; then usage; fi
if [ -z ${JOBTAG}  ]; then usage; fi
if [ -z ${NMATCH}  ]; then NMATCH=9; fi
if [ -z ${DOMTV}  ]; then usage; fi

# to shift away the parsed options
shift $(($OPTIND - 1))

#######################################################################
# Below long list of if statement determines the following four options
#######################################################################
# SAMPLE
# SAMPLETAG
# PDGID
# PTBOUND
# DOPEREVENT

DOPEREVENT=false

if [[ ${SAMPLEID} == "1" ]]; then
SAMPLE=${TRACKINGNTUPLEDIR}/CMSSW_10_4_0/src/trackingNtuple_100_pt0p5_2p0.root
SAMPLETAG=pt0p5_2p0
PDGID=13
PTBOUND=4
fi

if [[ ${SAMPLEID} == "2" ]]; then
SAMPLE=/nfs-7/userdata/phchang/trackingNtuple/trackingNtuple_10MuGun.root
SAMPLETAG=e200
PDGID=13
PTBOUND=0
fi

if [[ ${SAMPLEID} == "3" ]]; then
SAMPLE=${TRACKINGNTUPLEDIR}/CMSSW_10_4_0_mtd5/src/trackingNtuple_TTbar_PU200.root
SAMPLETAG=pu200_pdgid13
PDGID=13
PTBOUND=0
fi

if [[ ${SAMPLEID} == "4" ]]; then
SAMPLE=${TRACKINGNTUPLEDIR}/CMSSW_10_4_0_mtd5/src/trackingNtuple_TTbar_PU200.root
SAMPLETAG=pu200_pdgid211
PDGID=211
PTBOUND=0
fi

if [[ ${SAMPLEID} == "5" ]]; then
SAMPLE=${TRACKINGNTUPLEDIR}/CMSSW_10_4_0_mtd5/src/trackingNtuple_TTbar_PU200.root
SAMPLETAG=pu200_pdgid11
PDGID=11
PTBOUND=0
fi

if [[ ${SAMPLEID} == "6" ]]; then
SAMPLE=${TRACKINGNTUPLEDIR}/CMSSW_10_4_0_mtd5/src/trackingNtuple_DisplacedSUSY_stopToChi_Gravitino_M_1000_700_10mm.root
SAMPLETAG=pu200_displaced_pdgid13
PDGID=13
PTBOUND=0
fi

if [[ ${SAMPLEID} == "7" ]]; then
SAMPLE=${TRACKINGNTUPLEDIR}/CMSSW_10_4_0_mtd5/src/trackingNtuple_GMSB_L100_Ctau100.root
SAMPLETAG=pu200_displaced_gmsb_ctau100_pdgid13
PDGID=13
PTBOUND=0
fi

if [[ ${SAMPLEID} == "8" ]]; then
SAMPLE=${TRACKINGNTUPLEDIR}/CMSSW_10_4_0_mtd5/src/trackingNtuple_GMSB_L100_Ctau100.root
SAMPLETAG=pu200_displaced_gmsb_ctau100_pdgid11
PDGID=11
PTBOUND=0
fi

if [[ ${SAMPLEID} == "9" ]]; then
SAMPLE=${TRACKINGNTUPLEDIR}/CMSSW_10_4_0_mtd5/src/trackingNtuple_GMSB_L100_Ctau100.root
SAMPLETAG=pu200_displaced_gmsb_ctau100_pdgid211
PDGID=211
PTBOUND=0
fi

if [[ ${SAMPLEID} == "10" ]]; then
# SAMPLE=${TRACKINGNTUPLEDIR}/CMSSW_10_4_0/src/trackingNtuple_100_pt0p5_3p0.root
SAMPLE=${TRACKINGNTUPLEDIR}/CMSSW_10_4_0/src/trackingNtuple_100mu_pt0p5_3p0.root
SAMPLETAG=pt0p5_3p0
PDGID=13
PTBOUND=5
fi

if [[ ${SAMPLEID} == "11" ]]; then
SAMPLE=${TRACKINGNTUPLEDIR}/CMSSW_10_4_0/src/trackingNtuple_10_pt0p5_5p0.root
SAMPLETAG=pt0p5_5p0
PDGID=13
PTBOUND=6
fi

if [[ ${SAMPLEID} == "12" ]]; then
SAMPLE=/nfs-7/userdata/bsathian/SDL_trackingNtuple/ttbar_highPU/trackingNtuple_with_PUinfo_500_evts.root
SAMPLETAG=pu200_w_truth_pdgid13
PDGID=13
PTBOUND=0
DOPEREVENT=true
fi

if [[ ${SAMPLEID} == "13" ]]; then
SAMPLE=/nfs-7/userdata/bsathian/SDL_trackingNtuple/ttbar_highPU/trackingNtuple_with_PUinfo_500_evts.root
SAMPLETAG=pu200_w_truth_pdgid211
PDGID=211
PTBOUND=0
DOPEREVENT=true
fi

if [[ ${SAMPLEID} == "14" ]]; then
SAMPLE=/nfs-7/userdata/bsathian/SDL_trackingNtuple/ttbar_highPU/trackingNtuple_with_PUinfo_500_evts.root
SAMPLETAG=pu200_w_truth_pdgid11
PDGID=11
PTBOUND=0
DOPEREVENT=true
fi

if [[ ${SAMPLEID} == "15" ]]; then
SAMPLE=/nfs-7/userdata/bsathian/SDL_trackingNtuple/ttbar_highPU/trackingNtuple_with_PUinfo_500_evts.root
SAMPLETAG=pu200_w_truthinfo_charged
PDGID=0
PTBOUND=0
DOPEREVENT=true
fi

if [[ ${SAMPLEID} == "17" ]]; then
SAMPLE=${TRACKINGNTUPLEDIR}/CMSSW_10_4_0/src/trackingNtuple_10_pt0p5_50_5cm_cube.root
SAMPLETAG=pt0p5_50_5cm_cube
PDGID=13
PTBOUND=0
fi

if [[ ${SAMPLEID} == "18" ]]; then
SAMPLE=${TRACKINGNTUPLEDIR}/CMSSW_10_4_0/src/trackingNtuple_10_pt0p5_50_50cm_cube.root
SAMPLETAG=pt0p5_50_50cm_cube
PDGID=13
PTBOUND=7
fi

if [[ ${SAMPLEID} == "19" ]]; then
SAMPLE=/home/users/phchang/public_html/analysis/sdl/trackingNtuple/CMSSW_10_4_0_mtd5/src/trackingNtuple_TTbar.root
SAMPLETAG=ttbar
PDGID=13
PTBOUND=7
fi

if [[ ${SAMPLEID} == "20" ]]; then
SAMPLE=/home/users/phchang/public_html/analysis/sdl/trackingNtuple/CMSSW_10_4_0/src/trackingNtuple_100pion_pt0p5_50p0.root
SAMPLETAG=pion
PDGID=211
PTBOUND=8
fi


########################################
# Mode handling
########################################
MODE=algo_eff
MODENUMBER=3

if [[ ${DOMTV} == true ]]; then
    MODE=mtv_eff
    MODENUMBER=2
    #################################################
    # When the option is DOMTV then PDGID is set to 0
    #################################################
    # PDGID=0
fi

if [[ ${WRITESDLNTUPLE} == true ]]; then
    MODE=write_sdl_ntuple
    MODENUMBER=5
    PDGID=0
fi

##########################################################################################
# Set output file related names
##########################################################################################
OUTPUTFILEBASENAME=fulleff_${SAMPLETAG}
OUTPUTFILE=fulleff_${SAMPLETAG}.root
OUTPUTDIR=${TRACKLOOPERBASE}/results/${MODE}/${SAMPLETAG}_${JOBTAG}/

# Only if the directory does not exist one runs it again
if [ ! -d "${OUTPUTDIR}" ]; then

    mkdir -p ${OUTPUTDIR}
    rm -f ${OUTPUTDIR}/${OUTPUTFILE}
    rm -f ${OUTPUTDIR}/${OUTPUTFILEBASENAME}_*.root
    rm -f ${OUTPUTDIR}/${OUTPUTFILEBASENAME}_*.log

    rm .jobs.txt
    if [[ ${DOPEREVENT} == true ]]; then
        # NJOBS=500
        # NJOBS=36
        NJOBS=1
    else
        NJOBS=16
    fi
    for i in $(seq 0 $((NJOBS-1))); do
        # (set -x ;$TRACKLOOPERBASE/bin/sdl -j ${NJOBS} -I ${i} -i ${SAMPLE} -n -1 -t trackingNtuple/tree -m ${MODENUMBER} -p ${PTBOUND} -o ${OUTPUTDIR}/${OUTPUTFILEBASENAME}_${i}.root -g ${PDGID} > ${OUTPUTDIR}/${OUTPUTFILEBASENAME}_${i}.log 2>&1) &
        if [[ ${DOPEREVENT} == true ]]; then
            echo "$TRACKLOOPERBASE/bin/sdl -N ${NMATCH} -x ${i} -i ${SAMPLE} -n -1 -t trackingNtuple/tree -m ${MODENUMBER} -p ${PTBOUND} -o ${OUTPUTDIR}/${OUTPUTFILEBASENAME}_${i}.root -g ${PDGID} -v 3 > ${OUTPUTDIR}/${OUTPUTFILEBASENAME}_${i}.log 2>&1" >> .jobs.txt
        else
            echo "$TRACKLOOPERBASE/bin/sdl -N ${NMATCH} -j ${NJOBS} -I ${i} -i ${SAMPLE} -n -1 -t trackingNtuple/tree -m ${MODENUMBER} -p ${PTBOUND} -o ${OUTPUTDIR}/${OUTPUTFILEBASENAME}_${i}.root -g ${PDGID} > ${OUTPUTDIR}/${OUTPUTFILEBASENAME}_${i}.log 2>&1" >> .jobs.txt
            # echo "$TRACKLOOPERBASE/bin/sdl -N ${NMATCH} -j ${NJOBS} -I ${i} -i ${SAMPLE} -n 100 -t trackingNtuple/tree -m ${MODENUMBER} -p ${PTBOUND} -o ${OUTPUTDIR}/${OUTPUTFILEBASENAME}_${i}.root -g ${PDGID} > ${OUTPUTDIR}/${OUTPUTFILEBASENAME}_${i}.log 2>&1" >> .jobs.txt
        fi
    done

    echo "<== Submitting parallel jobs ..."

    xargs.sh .jobs.txt

    cp .jobs.txt ${OUTPUTDIR}/jobs.txt

    echo "<== Finished parallel jobs ..."

    # Hadding outputs
    hadd -f ${OUTPUTDIR}/${OUTPUTFILE} ${OUTPUTDIR}/${OUTPUTFILEBASENAME}_*.root

    # Writing some descriptive file
    echo "${DESCRIPTION}" > ${OUTPUTDIR}/description.txt
    git status >> ${OUTPUTDIR}/description.txt
    git diff >> ${OUTPUTDIR}/description.txt
    git log --graph --pretty=format:'%Cred%h%Creset -%C(yellow)%d%Creset %s %Cgreen(%cr) %C(bold blue)<%an>%Creset' --abbrev-commit >> ${OUTPUTDIR}/description.txt
    cd ${TRACKLOOPERBASE}/SDL/
    git status >> ${OUTPUTDIR}/description.txt
    git diff >> ${OUTPUTDIR}/description.txt
    git log --graph --pretty=format:'%Cred%h%Creset -%C(yellow)%d%Creset %s %Cgreen(%cr) %C(bold blue)<%an>%Creset' --abbrev-commit >> ${OUTPUTDIR}/description.txt
    cd ../

    if [[ ${WRITESDLNTUPLE} == true ]]; then
        :
        git status > gitlog; git log >> gitlog; git diff >> gitlog;
        mv gitlog ${OUTPUTDIR}
        cd efficiency/
        make -j
        sh run.sh ${OUTPUTDIR}/${OUTPUTFILE} ${PTBOUND}
        echo "Plotting standard efficiency plots ..."
        sh plot.sh ${SAMPLETAG} $(git rev-parse --short HEAD) > ${OUTPUTDIR}/plot.log 2>&1
        echo "Plotting Done!"
        mv plots/ ${OUTPUTDIR}/
        cp index.html ${OUTPUTDIR}/plots/
        cd ${OUTPUTDIR}
    else

        # plotting
        if [[ ${MODE} == *"mtv_eff"* ]]; then
            cd ${TRACKLOOPERBASE}/results/mtv_eff/${SAMPLETAG}_${JOBTAG}/
            echo "python ${TRACKLOOPERBASE}/python/plot.py 8 fulleff_${SAMPLETAG}.root ${SAMPLETAG}" > plot.log
            python ${TRACKLOOPERBASE}/python/plot.py 8 fulleff_${SAMPLETAG}.root ${SAMPLETAG} >> plot.log
        else
            echo sh $DIR/fulleff_plot.sh ${SAMPLETAG} ${JOBTAG}
            sh $DIR/fulleff_plot.sh ${SAMPLETAG} ${JOBTAG}
        fi

        # Creating html for easier efficiency plots viewing
        echo ""

        echo "Creating HTML from markdown"
        cd ${OUTPUTDIR}/
        echo sh $DIR/write_markdown.sh ${SAMPLETAG} "${DESCRIPTION}" ${JOBTAG} ${MODE}
        sh $DIR/write_markdown.sh ${SAMPLETAG} "${DESCRIPTION}" ${JOBTAG} ${MODE}

        echo ""

    fi

    function getlink {
        echo ${PWD/\/home\/users\/phchang\/public_html/http:\/\/snt:tas@uaf-10.t2.ucsd.edu\/~$USER/}/$1
    }

    if [[ ${WRITESDLNTUPLE} == true ]]; then
        export -f getlink
        echo ">>> results are in ${OUTPUTDIR}"
        echo ">>> results can also be viewed via following link:"
        echo ">>>   $(getlink plots/)"
    else
        export -f getlink
        echo ">>> results are in ${OUTPUTDIR}"
        echo ">>> results can also be viewed via following link:"
        echo ">>>   $(getlink)"
    fi

else

    echo "The command has been already ran before"
    echo "i.e. $OUTPUTDIR already exists"
    echo "Delete $OUTPUTDIR to re-run"

fi

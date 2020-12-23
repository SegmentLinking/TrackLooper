#!/bin/bash

##############################################################################
#
#
# Validation Plot Maker
#
#
##############################################################################

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Help
usage()
{
  echo "ERROR - Usage:"
  echo
  echo "      sh $(basename $0) OPTIONSTRINGS ..."
  echo
  echo "Options:"
  echo "  -h    Help                   (Display this message)"
  echo "  -i    input eff ntuple       (e.g. -i /path/to/my/eff_ntuple.root)"
  echo "  -p    pt boundaries          (Pt bin boundary settings see below for more detail)"
  echo "  -g    pdgid for denom tracks (e.g. -g 13 for muon or -g 0 for all charged particle)"
  echo
  exit
}

# Parsing command-line opts
while getopts ":i:p:g:h" OPTION; do
  case $OPTION in
    i) INPUTFILE=${OPTARG};;
    p) PTBOUND=${OPTARG};;
    g) PDGID=${OPTARG};;
    h) usage;;
    :) usage;;
  esac
done

# If the command line options are not provided set it to default value of false
if [ -z ${INPUTFILE} ]; then usage; fi
if [ -z ${PTBOUND}  ]; then usage; fi
if [ -z ${PDGID} ]; then usage; fi

# Shift away the parsed options
shift $(($OPTIND - 1))

# Verbose
echo "====================================================="  | tee -a ${LOG}
echo "Validation Plot Maker Script                         "  | tee -a ${LOG}
echo "====================================================="  | tee -a ${LOG}
echo ""                                                       | tee -a ${LOG}
echo "  INPUTFILE         : ${INPUTFILE}"                     | tee -a ${LOG}
echo "  PTBOUND           : ${PTBOUND}"                       | tee -a ${LOG}
echo "  PDGID             : ${PDGID}"                         | tee -a ${LOG}
echo ""                                                       | tee -a ${LOG}

rm .jobs.txt
NJOBS=16

rm -rf outputs/
mkdir -p outputs/

for i in $(seq 0 $((NJOBS-1))); do
    echo "./doAnalysis -i ${INPUTFILE} -p ${PTBOUND} -g ${PDGID} -t tree -o outputs/output_${i}.root -j ${NJOBS} -I ${i} > outputs/output_${i}.log 2>&1" >> .jobs.txt
done

xargs.sh .jobs.txt

hadd -f efficiency.root outputs/*.root 

# Get tag from the efficiency file
PYTHON_CODE=$(cat <<END
# python code starts here
import ROOT as r

f = r.TFile("${INPUTFILE}")
t = f.Get("code_tag_data")
t.Print("")

# python code ends here
END
)

FULLTAG=$(python -c "$PYTHON_CODE"  | head -n2 | tail -n1)
TAG=${FULLTAG:0:7}

# Get input sample from the efficiency file
PYTHON_CODE=$(cat <<END
# python code starts here
import ROOT as r

f = r.TFile("${INPUTFILE}")
t = f.Get("input")
t.Print("")

# python code ends here
END
)

FULLSTRSAMPLE=$(python -c "$PYTHON_CODE")
SAMPLE=${FULLSTRSAMPLE//"TObjString = "/}

echo ${TAG} ${SAMPLE}

sh plot.sh $(basename ${SAMPLE}) ${TAG}


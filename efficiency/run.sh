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
  echo "  -f    force                  (Force re-run of the efficiency)"
  echo "  -i    input eff ntuple       (e.g. -i /path/to/my/eff_ntuple.root)"
  echo "  -p    pt boundaries          (Pt bin boundary settings see below for more detail)"
  echo "  -g    pdgid for denom tracks (e.g. -g 13 for muon or -g 0 for all charged particle)"
  echo
  exit
}

# Parsing command-line opts
while getopts ":i:p:g:fh" OPTION; do
  case $OPTION in
    i) INPUTFILE=${OPTARG};;
    p) PTBOUND=${OPTARG};;
    g) PDGID=${OPTARG};;
    f) FORCE=true;;
    h) usage;;
    :) usage;;
  esac
done

# If the command line options are not provided set it to default value of false
if [ -z ${INPUTFILE} ]; then usage; fi
if [ -z ${PTBOUND}  ]; then usage; fi
if [ -z ${PDGID} ]; then usage; fi
if [ -z ${FORCE} ]; then FORCE=false; fi

# Shift away the parsed options
shift $(($OPTIND - 1))

# Verbose
echo "====================================================="
echo "Validation Plot Maker Script                         "
echo "====================================================="
echo ""
echo "  INPUTFILE         : ${INPUTFILE}"
echo "  PTBOUND           : ${PTBOUND}"
echo "  PDGID             : ${PDGID}"
echo "  FORCE             : ${FORCE}"
echo ""

# Get tag from the efficiency file
echo "Parsing repository git hash tag..."
PYTHON_CODE=$(cat <<END
# python code starts here
import ROOT as r

f = r.TFile("${INPUTFILE}")
t = f.Get("code_tag_data")
t.Print("")

# python code ends here
END
)

# Print full info on the source code
python -c "$PYTHON_CODE" > gitversion.txt

FULLTAG=$(cat gitversion.txt | head -n2 | tail -n1)
TAG=${FULLTAG:0:7}

# Get input sample from the efficiency file
echo "Parsing input sample name..."
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

# Get whether it is GPU or CPU
echo "Parsing code version..."
PYTHON_CODE=$(cat <<END
# python code starts here
import ROOT as r

f = r.TFile("${INPUTFILE}")
t = f.Get("version")
t.Print("")

# python code ends here
END
)

FULLSTRVERSION=$(python -c "$PYTHON_CODE")
VERSION=${FULLSTRVERSION//"TObjString = "/}

# Get full command line used to run the segment linking
echo "Parsing command line used..."
PYTHON_CODE=$(cat <<END
# python code starts here
import ROOT as r

f = r.TFile("${INPUTFILE}")
t = f.Get("full_cmd_line")
t.Print("")

# python code ends here
END
)

FULLCMDLINE=$(python -c "$PYTHON_CODE")
CMDLINE=${FULLCMDLINE//"TObjString = "/}

echo ${CMDLINE} > cmdline.txt

mkdir -p results/

OUTDIR=results/eff_plots__${VERSION}_${TAG}_${SAMPLE}

if [ -d "${OUTDIR}" ]; then
    if $FORCE; then
        :
    else
        echo "Already ran efficiency for TAG=${TAG} SAMPLE=${SAMPLE} VERSION=${VERSION} !!!"
        echo "Are you sure you want to re-run and overwrite?  Add -f in the command line to overwrite."
        exit
    fi
fi

echo ""
echo "  TAG               : ${TAG}"
echo "  SAMPLE            : ${SAMPLE}"
echo "  VERSION           : ${VERSION}"
echo "  CMDLINE           : ${CMDLINE}"
echo ""

# Delete the output if already existing
rm -rf ${OUTDIR}

echo "Running efficiency...             (log: ${OUTDIR}/run.log)"

# Run the efficiency histograms
rm -f .jobs.txt
NJOBS=16

rm -rf outputs/
mkdir -p outputs/

if [ ! -f doAnalysis ]; then
    echo "Need to make efficiency code first"
    make clean; make -j
fi

for i in $(seq 0 $((NJOBS-1))); do
    echo "./doAnalysis -i ${INPUTFILE} -p ${PTBOUND} -g ${PDGID} -t tree -o outputs/output_${i}.root -j ${NJOBS} -I ${i} > outputs/output_${i}.log 2>&1" >> .jobs.txt
done

xargs.sh .jobs.txt > run.log 2>&1

echo "Hadding histograms...             (log: ${OUTDIR}/hadd.log)"
hadd -f efficiency.root outputs/*.root > hadd.log 2>&1

# Make plots
echo "Making plots...                   (log: ${OUTDIR}/plots.log)"
sh plot.sh $(basename ${SAMPLE}) ${TAG}_${VERSION} > plots.log 2>&1

# copy the plots
echo "Copying plots..."
cp -r plots ${OUTDIR}

# copy the source code info
mv gitversion.txt ${OUTDIR}/
mv plots.log ${OUTDIR}/
mv run.log ${OUTDIR}/
mv hadd.log ${OUTDIR}/
mv efficiency.root ${OUTDIR}/
mv outputs ${OUTDIR}/
mv cmdline.txt ${OUTDIR}/

# clean up so that the next run would be spoiled
rm -rf plots

# Output
echo "Done! Output is located at ${OUTDIR}/index.html"

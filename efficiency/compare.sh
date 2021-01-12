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
  echo
  exit
}

# Parsing command-line opts
while getopts ":i:fh" OPTION; do
  case $OPTION in
    i) INPUTFILE=${OPTARG};;
    f) FORCE=true;;
    h) usage;;
    :) usage;;
  esac
done

# If the command line options are not provided set it to default value of false
if [ -z ${INPUTFILE} ]; then usage; fi
if [ -z ${FORCE} ]; then FORCE=false; fi

# Shift away the parsed options
shift $(($OPTIND - 1))

# Verbose
echo "====================================================="
echo "Validation Plot Maker Script                         "
echo "====================================================="
echo ""
echo "  INPUTFILE         : ${INPUTFILE}"
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

mkdir -p results/

OUTDIR=results/eff_comparison_plots__${TAG}_${SAMPLE}

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
echo ""

# Delete the output if already existing
rm -rf ${OUTDIR}

# Make plots
mkdir plots
echo "Making comparison plots...              (log: ${OUTDIR}/plots.log)"
python plot_compare.py ${TAG} ${SAMPLE} > plots.log 2>&1

# copy the plots
echo "Copying plots..."
cp -r plots ${OUTDIR}

# copy the source code info
mv gitversion.txt ${OUTDIR}/
mv plots.log ${OUTDIR}/
cp index.php ${OUTDIR}/

# clean up so that the next run would be spoiled
rm -rf plots

# Output
echo "Done! Output is located at ${OUTDIR}/"

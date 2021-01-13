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
  echo "  -i    input sample type      (e.g. -i muonGun)"
  echo "  -t    git hash tag           (e.g. -t 78df188)"
  echo
  exit
}

# Parsing command-line opts
while getopts ":i:t:fh" OPTION; do
  case $OPTION in
    i) SAMPLE=${OPTARG};;
    t) TAG=${OPTARG};;
    f) FORCE=true;;
    h) usage;;
    :) usage;;
  esac
done

# If the command line options are not provided set it to default value of false
if [ -z ${SAMPLE} ]; then usage; fi
if [ -z ${TAG} ]; then usage; fi
if [ -z ${FORCE} ]; then FORCE=false; fi

# Shift away the parsed options
shift $(($OPTIND - 1))

# Create output directory
mkdir -p results/

# Set output directory
OUTDIR=results/eff_comparison_plots__${TAG}_${SAMPLE}

if [ -d "${OUTDIR}" ]; then
    if $FORCE; then
        :
    else
        echo "Already ran efficiency for TAG=${TAG} SAMPLE=${SAMPLE} !!!"
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

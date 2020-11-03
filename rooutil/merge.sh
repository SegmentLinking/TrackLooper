#!/bin/bash

# This is a baby making condor executable for CondorTask of ProjectMetis. Passed in arguments are:
# arguments = [outdir, outname_noext, inputs_commasep, index, cmssw_ver, scramarch, self.arguments]

OUTPUTDIR=$1
OUTPUTNAME=$2
INPUTFILENAMES=$3
IFILE=$4
CMSSW_VERSION=$5
SCRAM_ARCH=$6
ISFASTSIM=$7

OUTPUTDIR=`dirname $OUTPUTDIR`
OUTPUTNAME=$(echo $OUTPUTNAME | sed 's/\.root//')

echo -e "\n--- begin header output ---\n" #                     <----- section division
echo "OUTPUTDIR: $OUTPUTDIR"
echo "OUTPUTNAME: $OUTPUTNAME"
echo "INPUTFILENAMES: $INPUTFILENAMES"
echo "IFILE: $IFILE"
echo "CMSSW_VERSION: $CMSSW_VERSION"
echo "SCRAM_ARCH: $SCRAM_ARCH"

echo "hostname: $(hostname)"
echo "uname -a: $(uname -a)"
echo "time: $(date +%s)"
echo "args: $@"

echo -e "\n--- end header output ---\n" #                       <----- section division

# Unpack the passed in tarfile
tar -xzf package.tar.gz
ls -ltrha
echo ----------------------------------------------

# Setup Enviroment
export SCRAM_ARCH=$SCRAM_ARCH
source /cvmfs/cms.cern.ch/cmsset_default.sh
pushd /cvmfs/cms.cern.ch/$SCRAM_ARCH/cms/cmssw/$CMSSW_VERSION/src/ > /dev/null
eval `scramv1 runtime -sh`
popd > /dev/null

# The output name is the sample name for stop baby
SAMPLE_NAME=$OUTPUTNAME
NEVENTS=-1

echo "Running Merge Jobs:"
mergeout_folder=$OUTPUTDIR/merged
skimout_folder=$OUTPUTDIR/skimmed
mkdir -p $mergeout_folder
mkdir -p $skimout_folder

# Perform Merge
# out_file=${mergeout_folder}/merged_${OUTPUTNAME}.root
out_file=${OUTPUTNAME}.root
in_folder=${OUTPUTDIR}
echo "Will merge $in_folder/*.root into $out_file"
echo "python rooutil/hadd.py -t t -o ${out_file} ${in_folder}/*.root"
python rooutil/hadd.py -t t -o ${out_file} ${in_folder}/*.root
if [ $? -eq 0 ]; then
    echo "Successfully hadded files"
    :
else
    echo "Failed to hadd the files"
    exit
fi


echo ----------------------------------------------
ls -ltrha
echo ----------------------------------------------

echo -e "\n--- end running ---\n" #                             <----- section division

[[ ! -f ${OUTPUTNAME}_1.root ]] && mv ${OUTPUTNAME}.root ${OUTPUTNAME}_1.root

# Copy back the output file
for mergeout in ${OUTPUTNAME}*.root; do
    gfal-copy -p -f -t 4200 --verbose file://`pwd`/$mergeout gsiftp://gftp.t2.ucsd.edu${mergeout_folder}/${mergeout} --checksum ADLER32
done

echo -e "\n--- cleaning up ---\n" #                             <----- section division
rm -r *.root

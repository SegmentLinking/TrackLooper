#!/bin/bash

usage()
{
    echo "Usage:"
    echo
    echo "      sh $(basename $0) [-t TREENAME=Events] ROOTFILE"
    echo
    exit
}

# Parsing command-line opts
while getopts ":tdh" OPTION; do
    case $OPTION in
        t) TREENAME=${OPTARG};;
        d) DELETE=true;;
        h) usage;;
        :) usage;;
    esac
done

# Set default tree name if not provided
if [ -z ${TREENAME} ]; then TREENAME=Events; fi

# to shift away the parsed options
shift $(($OPTIND - 1))

# Check root file argument
if [ -z $1 ]; then
    echo "Error: no root file name was provided."
    usage
else
    FILENAME=$1
fi

# Print disclaimer to cover my ass
if [ "$DELETE" == true ]; then
    echo "[RSR] WARNING: DELETE MODE IS ON!!!"
    echo "[RSR] If the script finds a bad root file it will DELETE IT!!!"
    echo "[RSR] DON'T SAY I DIDN'T WARN YA"
fi

# Write the script to a temporary file in order to avoid clash when running parallel
MACRONAME=$(mktemp stupid_numbers_XXXXXXXXX)
MACRO=/tmp/${MACRONAME}.C

# Dumping the macro to the tmp file
#------------------------------------------------------------
echo "#include \"TFile.h\"
#include \"TTree.h\"
#include \"TString.h\"

int ${MACRONAME}(TString fname, TString treename)
{
    TFile* f = new TFile(fname, \"open\");
    if (!f)
    {
        printf(\"[RSR] file is screwed up\n\");
        return 1;
    }
    TTree* t = (TTree*) f->Get(treename);
    if (!t)
    {
        printf(\"[RSR] tree is screwed up\n\");
        return 1;
    }

    Long64_t nentries = t->GetEntries();
    printf(\"[RSR] ntuple has %lld events\n\", nentries);

    bool foundBad = false;
    for (Long64_t ientry = 0; ientry < t->GetEntries(); ++ientry)
    {
        Int_t bytes_read = t->GetEntry();
        if (bytes_read < 0)
        {
            foundBad = true;
            printf(\"[RSR] found bad event %lld\n\", nentries);
            return 1;
        }
    }

    if (!foundBad)
    {
        printf(\"[RSR] passed the rigorous sweeproot\n\");
        return 0;
    }
    return 0;
}" > $MACRO
#------------------------------------------------------------

rm ${MACRONAME}.C

# Perform a rigorous sweep
root -l -b -q ${MACRO}+\(\"$FILENAME\",\"$TREENAME\"\)

# If return code is non-zero delete the file
if [ $? -ne 0 ]; then
    echo "[RSR] failed the rigorous sweeproot"
    if [ "$DELETE" == true ]; then
        echo "[RSR] Deleting the file $FILENAME"
        rm $FILENAME
    fi
    exit 1
fi

exit 0

#eof

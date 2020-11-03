#!/bin/bash

usage()
{
    echo "Usage:"
    echo ""
    echo "  $0 ROOTFILE OUTPUTFILE SCANSTRING"
    echo ""
    echo ""
    exit
}

if [ -z $1 ]; then usage; fi
if [ -z $2 ]; then usage; fi
if [ -z $3 ]; then usage; fi

ROOTFILE=$1
OUTPUTFILE=$2
COLUMNS=$3

echo "{
    TFile *_file0 = TFile::Open(\"${ROOTFILE}\");
    ((TTreePlayer*)(t->GetPlayer()))->SetScanRedirect(true); 
    ((TTreePlayer*)(t->GetPlayer()))->SetScanFileName(\"${OUTPUTFILE}\"); 
    t->Scan(\"${COLUMNS}\",\"\",\"\");
}" > /tmp/temp_macro.C

root -l -b -q /tmp/temp_macro.C

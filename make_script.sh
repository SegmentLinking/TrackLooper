#!/bin/bash

##############################################################################
#
#
# Line Segment Tracking Standalone Code Make Script
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
  echo "  -x    explicit memory        (Make library with explicit memory enabled)"
  echo "  -c    cache                  (Make library with cache enabled)"
  echo "  -s    show log               (Full compilation script to stdout)"
  echo
  exit
}

# Parsing command-line opts
while getopts ":cxsh" OPTION; do
  case $OPTION in
    c) MAKECACHE=true;;
    x) MAKEEXPLICIT=true;;
    s) SHOWLOG=true;;
    h) usage;;
    :) usage;;
  esac
done

# If the command line options are not provided set it to default value of false
if [ -z ${MAKECACHE} ]; then MAKECACHE=false; fi
if [ -z ${MAKEEXPLICIT}  ]; then MAKEEXPLICIT=false; fi
if [ -z ${SHOWLOG} ]; then SHOWLOG=false; fi

# Shift away the parsed options
shift $(($OPTIND - 1))

# create log file
LOG=.make.log.$(date +%s)

# Verbose
date | tee -a ${LOG}
echo "====================================================="  | tee -a ${LOG}
echo "Line Segment Tracking Compilation Script             "  | tee -a ${LOG}
echo "====================================================="  | tee -a ${LOG}
echo "Compilation options set to..."                          | tee -a ${LOG}
echo ""                                                       | tee -a ${LOG}
echo "  MAKECACHE      : ${MAKECACHE}"                        | tee -a ${LOG}
echo "  MAKEEXPLICIT   : ${MAKEEXPLICIT}"                     | tee -a ${LOG}
echo "  SHOWLOG        : ${SHOWLOG}"                          | tee -a ${LOG}
echo ""                                                       | tee -a ${LOG}
echo "  (cf. Run > sh $(basename $0) -h to see all options)"  | tee -a ${LOG}
echo ""                                                       | tee -a ${LOG}

# Target to make default is unified
MAKETARGET=unified

# If make explicit is true then make library with explicit memory on GPU
if $MAKEEXPLICIT; then MAKETARGET=explicit; fi

# If make cache is true then make library with cache enabled
if $MAKECACHE; then MAKETARGET=${MAKETARGET}_cache; fi

echo "Line Segment Tracking GPU library with MAKETARGET=${MAKETARGET} is being compiled...." | tee -a ${LOG}

echo "---------------------------------------------------------------------------------------------" >> ${LOG} 2>&1 
echo "---------------------------------------------------------------------------------------------" >> ${LOG} 2>&1 
echo "---------------------------------------------------------------------------------------------" >> ${LOG} 2>&1 
if $SHOWLOG; then
    (cd SDL && make clean && make -j 32 ${MAKETARGET} && cd -) 2>&1 | tee -a ${LOG}
else
    (cd SDL && make clean && make -j 32 ${MAKETARGET} && cd -) >> ${LOG} 2>&1 
fi

if [ ! -f SDL/sdl.so ]; then
    echo "ERROR: SDL/sdl.so failed to compile!" | tee -a ${LOG}
    echo "See ${LOG} file for more detail..." | tee -a ${LOG}
    exit
fi

echo "Line Segment Tracking GPU library compilation with MAKETARGET=${MAKETARGET} successful!" | tee -a ${LOG}

echo "" | tee -a ${LOG}
echo "Line Segment Tracking binaries are being compiled...." | tee -a ${LOG}

echo "---------------------------------------------------------------------------------------------" >> ${LOG} 2>&1 
echo "---------------------------------------------------------------------------------------------" >> ${LOG} 2>&1 
echo "---------------------------------------------------------------------------------------------" >> ${LOG} 2>&1 
if $SHOWLOG; then
    make -j 2>&1 | tee -a ${LOG}
else
    make -j >> ${LOG} 2>&1 
fi

if [ ! -f bin/doAnalysis ]; then
    echo "ERROR: bin/doAnalysis failed to compile!" | tee -a ${LOG}
    echo "See ${LOG} file for more detail..." | tee -a ${LOG}
    exit
fi

if [ ! -f bin/sdl ]; then
    echo "ERROR: bin/sdl failed to compile!" | tee -a ${LOG}
    echo "See ${LOG} file for more detail..." | tee -a ${LOG}
    exit
fi

echo "" >> ${LOG}
echo "" >> ${LOG}
echo "" >> ${LOG}
echo "Line Segment Tracking binaries compilation successful!" | tee -a ${LOG}
echo "" | tee -a ${LOG}
echo "Compilation is logged at ${LOG}" | tee -a ${LOG}

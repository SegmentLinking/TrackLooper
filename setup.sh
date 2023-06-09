#!/bin/bash

module load cuda-11.7.1
module load cuda11.7/toolkit/11.7.1

###########################################################################################################
# Setup environments
###########################################################################################################
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/code/rooutil/thisrooutil.sh
which root

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export LD_LIBRARY_PATH=$DIR:$LD_LIBRARY_PATH
export PATH=$DIR/bin:$PATH
export PATH=$DIR/efficiency/bin:$PATH
export PATH=$DIR/efficiency/python:$PATH
export TRACKLOOPERDIR=$DIR
export TRACKINGNTUPLEDIR=~
export LSTOUTPUTDIR=.
export LSTPERFORMANCEWEBDIR=~/LSTPerformanceWeb
#eof

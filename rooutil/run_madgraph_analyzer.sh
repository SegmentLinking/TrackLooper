#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

if [[ "$1" == *".gz"* ]]; then
    gunzip $1
    LHEFILENAME=${1/.lhe.gz/.lhe}
else
    LHEFILENAME=$1
fi

ROOTFILENAME=${LHEFILENAME/.lhe/.root}
HISTFILENAME=${LHEFILENAME/.lhe/_hist.root}
if [ -f $ROOTFILENAME ];then
    :
else
    python ${DIR}/lhe2root.py ${LHEFILENAME} ${ROOTFILENAME}
fi

if [ -n "$2" ]; then
    PAIRS="-p $2"
fi

if [ -n "$3" ]; then
    EXTRA="$3"
fi

rm ${HISTFILENAME}
./doAnalysis -i ${ROOTFILENAME} -t Physics -o ${HISTFILENAME} ${PAIRS} ${EXTRA}
# python $DIR/plot_madgraph_analyzer.py ${HISTFILENAME}

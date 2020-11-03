#!/bin/bash

trap "kill 0" EXIT

usage() {
    echo "Usage:"
    echo ""
    echo "   $0 TAG DESCRIPTION"
    echo ""
    exit
}

if [ -z "$1" ]; then usage; fi
if [ -z "$2" ]; then usage; fi

TAG=$1
DESCRIPTION="$2"

sh fulleff.sh 1 _${TAG} "${DESCRIPTION}"
# sh fulleff.sh 2 _${TAG} "${DESCRIPTION}"
# sh fulleff.sh 3 _${TAG} "${DESCRIPTION}"
# sh fulleff.sh 4 _${TAG} "${DESCRIPTION}"
# sh fulleff.sh 5 _${TAG} "${DESCRIPTION}"
# sh fulleff.sh 7 _${TAG} "${DESCRIPTION}"
# sh fulleff.sh 8 _${TAG} "${DESCRIPTION}"
# sh fulleff.sh 9 _${TAG} "${DESCRIPTION}"

sh fulleff_mtv.sh 1 _${TAG} "${DESCRIPTION}"
# sh fulleff_mtv.sh 2 _${TAG} "${DESCRIPTION}"
# sh fulleff_mtv.sh 3 _${TAG} "${DESCRIPTION}"
# sh fulleff_mtv.sh 4 _${TAG} "${DESCRIPTION}"
# sh fulleff_mtv.sh 5 _${TAG} "${DESCRIPTION}"
# sh fulleff_mtv.sh 7 _${TAG} "${DESCRIPTION}"
# sh fulleff_mtv.sh 8 _${TAG} "${DESCRIPTION}"
# sh fulleff_mtv.sh 9 _${TAG} "${DESCRIPTION}"

# # sh fulleff.sh 6 _${TAG} "${DESCRIPTION}"
# # sh fulleff_mtv.sh 6 _${TAG} "${DESCRIPTION}"

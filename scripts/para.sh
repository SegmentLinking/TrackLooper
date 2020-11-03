#!/bin/bash

rm pu140_*.root

for i in $(seq 0 99); do
    echo "time ./doAnalysis -i /home/users/phchang/public_html/analysis/sdl/trackingNtuple/CMSSW_10_4_0_mtd5/src/trackingNtuple_NuGun_PU140.root -n -1 -t trackingNtuple/tree -o pu140_${i}.root -x ${i}" >> arun.sh
done

wait

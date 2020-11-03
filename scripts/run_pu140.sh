#!/bin/bash

function usage {
    echo "Usage:"
    echo "    $0 TAG"
    exit
}

if [ -z $1 ]; then
    usage
fi

TAG=$1

git st > gitversion.txt
git diff >> gitversion.txt
cd SDL/
git st >> ../gitversion.txt
git diff >> ../gitversion.txt
cd ../

rm pu140_*_${TAG}.root
time ./doAnalysis -i /home/users/phchang/public_html/analysis/sdl/trackingNtuple/CMSSW_10_4_0_mtd5/src/trackingNtuple_NuGun_PU140.root -n -1 -t trackingNtuple/tree -o pu140_0_${TAG}.root  -x 0  > pu140_0.log 2>&1 & 
time ./doAnalysis -i /home/users/phchang/public_html/analysis/sdl/trackingNtuple/CMSSW_10_4_0_mtd5/src/trackingNtuple_NuGun_PU140.root -n -1 -t trackingNtuple/tree -o pu140_1_${TAG}.root  -x 1  > pu140_1.log 2>&1 &
time ./doAnalysis -i /home/users/phchang/public_html/analysis/sdl/trackingNtuple/CMSSW_10_4_0_mtd5/src/trackingNtuple_NuGun_PU140.root -n -1 -t trackingNtuple/tree -o pu140_2_${TAG}.root  -x 2  > pu140_2.log 2>&1 &
time ./doAnalysis -i /home/users/phchang/public_html/analysis/sdl/trackingNtuple/CMSSW_10_4_0_mtd5/src/trackingNtuple_NuGun_PU140.root -n -1 -t trackingNtuple/tree -o pu140_3_${TAG}.root  -x 3  > pu140_3.log 2>&1 &
time ./doAnalysis -i /home/users/phchang/public_html/analysis/sdl/trackingNtuple/CMSSW_10_4_0_mtd5/src/trackingNtuple_NuGun_PU140.root -n -1 -t trackingNtuple/tree -o pu140_4_${TAG}.root  -x 4  > pu140_4.log 2>&1 &
time ./doAnalysis -i /home/users/phchang/public_html/analysis/sdl/trackingNtuple/CMSSW_10_4_0_mtd5/src/trackingNtuple_NuGun_PU140.root -n -1 -t trackingNtuple/tree -o pu140_5_${TAG}.root  -x 5  > pu140_5.log 2>&1 &
time ./doAnalysis -i /home/users/phchang/public_html/analysis/sdl/trackingNtuple/CMSSW_10_4_0_mtd5/src/trackingNtuple_NuGun_PU140.root -n -1 -t trackingNtuple/tree -o pu140_6_${TAG}.root  -x 6  > pu140_6.log 2>&1 &
time ./doAnalysis -i /home/users/phchang/public_html/analysis/sdl/trackingNtuple/CMSSW_10_4_0_mtd5/src/trackingNtuple_NuGun_PU140.root -n -1 -t trackingNtuple/tree -o pu140_7_${TAG}.root  -x 7  > pu140_7.log 2>&1 &
time ./doAnalysis -i /home/users/phchang/public_html/analysis/sdl/trackingNtuple/CMSSW_10_4_0_mtd5/src/trackingNtuple_NuGun_PU140.root -n -1 -t trackingNtuple/tree -o pu140_8_${TAG}.root  -x 8  > pu140_8.log 2>&1 &
time ./doAnalysis -i /home/users/phchang/public_html/analysis/sdl/trackingNtuple/CMSSW_10_4_0_mtd5/src/trackingNtuple_NuGun_PU140.root -n -1 -t trackingNtuple/tree -o pu140_9_${TAG}.root  -x 9  > pu140_9.log 2>&1 &
time ./doAnalysis -i /home/users/phchang/public_html/analysis/sdl/trackingNtuple/CMSSW_10_4_0_mtd5/src/trackingNtuple_NuGun_PU140.root -n -1 -t trackingNtuple/tree -o pu140_10_${TAG}.root -x 10 > pu140_10.log 2>&1 &
time ./doAnalysis -i /home/users/phchang/public_html/analysis/sdl/trackingNtuple/CMSSW_10_4_0_mtd5/src/trackingNtuple_NuGun_PU140.root -n -1 -t trackingNtuple/tree -o pu140_11_${TAG}.root -x 11 > pu140_11.log 2>&1 &
time ./doAnalysis -i /home/users/phchang/public_html/analysis/sdl/trackingNtuple/CMSSW_10_4_0_mtd5/src/trackingNtuple_NuGun_PU140.root -n -1 -t trackingNtuple/tree -o pu140_12_${TAG}.root -x 12 > pu140_12.log 2>&1 &
time ./doAnalysis -i /home/users/phchang/public_html/analysis/sdl/trackingNtuple/CMSSW_10_4_0_mtd5/src/trackingNtuple_NuGun_PU140.root -n -1 -t trackingNtuple/tree -o pu140_13_${TAG}.root -x 13 > pu140_13.log 2>&1 &
time ./doAnalysis -i /home/users/phchang/public_html/analysis/sdl/trackingNtuple/CMSSW_10_4_0_mtd5/src/trackingNtuple_NuGun_PU140.root -n -1 -t trackingNtuple/tree -o pu140_14_${TAG}.root -x 14 > pu140_14.log 2>&1 &
time ./doAnalysis -i /home/users/phchang/public_html/analysis/sdl/trackingNtuple/CMSSW_10_4_0_mtd5/src/trackingNtuple_NuGun_PU140.root -n -1 -t trackingNtuple/tree -o pu140_15_${TAG}.root -x 15 > pu140_15.log 2>&1 &
time ./doAnalysis -i /home/users/phchang/public_html/analysis/sdl/trackingNtuple/CMSSW_10_4_0_mtd5/src/trackingNtuple_NuGun_PU140.root -n -1 -t trackingNtuple/tree -o pu140_16_${TAG}.root -x 16 > pu140_16.log 2>&1 &
time ./doAnalysis -i /home/users/phchang/public_html/analysis/sdl/trackingNtuple/CMSSW_10_4_0_mtd5/src/trackingNtuple_NuGun_PU140.root -n -1 -t trackingNtuple/tree -o pu140_17_${TAG}.root -x 17 > pu140_17.log 2>&1 &
time ./doAnalysis -i /home/users/phchang/public_html/analysis/sdl/trackingNtuple/CMSSW_10_4_0_mtd5/src/trackingNtuple_NuGun_PU140.root -n -1 -t trackingNtuple/tree -o pu140_18_${TAG}.root -x 18 > pu140_18.log 2>&1 &
time ./doAnalysis -i /home/users/phchang/public_html/analysis/sdl/trackingNtuple/CMSSW_10_4_0_mtd5/src/trackingNtuple_NuGun_PU140.root -n -1 -t trackingNtuple/tree -o pu140_19_${TAG}.root -x 19 > pu140_19.log 2>&1 &
time ./doAnalysis -i /home/users/phchang/public_html/analysis/sdl/trackingNtuple/CMSSW_10_4_0_mtd5/src/trackingNtuple_NuGun_PU140.root -n -1 -t trackingNtuple/tree -o pu140_20_${TAG}.root -x 20 > pu140_20.log 2>&1 &
time ./doAnalysis -i /home/users/phchang/public_html/analysis/sdl/trackingNtuple/CMSSW_10_4_0_mtd5/src/trackingNtuple_NuGun_PU140.root -n -1 -t trackingNtuple/tree -o pu140_21_${TAG}.root -x 21 > pu140_21.log 2>&1 &
time ./doAnalysis -i /home/users/phchang/public_html/analysis/sdl/trackingNtuple/CMSSW_10_4_0_mtd5/src/trackingNtuple_NuGun_PU140.root -n -1 -t trackingNtuple/tree -o pu140_22_${TAG}.root -x 22 > pu140_22.log 2>&1 &
time ./doAnalysis -i /home/users/phchang/public_html/analysis/sdl/trackingNtuple/CMSSW_10_4_0_mtd5/src/trackingNtuple_NuGun_PU140.root -n -1 -t trackingNtuple/tree -o pu140_23_${TAG}.root -x 23 > pu140_23.log 2>&1 &
time ./doAnalysis -i /home/users/phchang/public_html/analysis/sdl/trackingNtuple/CMSSW_10_4_0_mtd5/src/trackingNtuple_NuGun_PU140.root -n -1 -t trackingNtuple/tree -o pu140_24_${TAG}.root -x 24 > pu140_24.log 2>&1 &
time ./doAnalysis -i /home/users/phchang/public_html/analysis/sdl/trackingNtuple/CMSSW_10_4_0_mtd5/src/trackingNtuple_NuGun_PU140.root -n -1 -t trackingNtuple/tree -o pu140_25_${TAG}.root -x 25 > pu140_25.log 2>&1 &
time ./doAnalysis -i /home/users/phchang/public_html/analysis/sdl/trackingNtuple/CMSSW_10_4_0_mtd5/src/trackingNtuple_NuGun_PU140.root -n -1 -t trackingNtuple/tree -o pu140_26_${TAG}.root -x 26 > pu140_26.log 2>&1 &
time ./doAnalysis -i /home/users/phchang/public_html/analysis/sdl/trackingNtuple/CMSSW_10_4_0_mtd5/src/trackingNtuple_NuGun_PU140.root -n -1 -t trackingNtuple/tree -o pu140_27_${TAG}.root -x 27 > pu140_27.log 2>&1 &
time ./doAnalysis -i /home/users/phchang/public_html/analysis/sdl/trackingNtuple/CMSSW_10_4_0_mtd5/src/trackingNtuple_NuGun_PU140.root -n -1 -t trackingNtuple/tree -o pu140_28_${TAG}.root -x 28 > pu140_28.log 2>&1 &
time ./doAnalysis -i /home/users/phchang/public_html/analysis/sdl/trackingNtuple/CMSSW_10_4_0_mtd5/src/trackingNtuple_NuGun_PU140.root -n -1 -t trackingNtuple/tree -o pu140_29_${TAG}.root -x 29 > pu140_29.log 2>&1 &

wait

hadd -f pu140_${TAG}.root pu140_*_${TAG}.root

mkdir -p results/${TAG}

mv pu140_${TAG}.root results/${TAG}
mv pu140_*_${TAG}.root results/${TAG}
mv pu140_*.log results/${TAG}
mv gitversion.txt results/${TAG}



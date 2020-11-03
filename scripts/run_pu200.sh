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

SAMPLE=../trackingNtuple/CMSSW_10_4_0_mtd5/src/trackingNtuple_TTbar_PU200.root

rm pu200_*_${TAG}.root
echo "time ./doAnalysis -i ${SAMPLE} -n -1 -t trackingNtuple/tree -o pu200_0_${TAG}.root  -x 0  > pu200_0.log 2>&1 &"
echo "time ./doAnalysis -i ${SAMPLE} -n -1 -t trackingNtuple/tree -o pu200_1_${TAG}.root  -x 1  > pu200_1.log 2>&1 &"
echo "time ./doAnalysis -i ${SAMPLE} -n -1 -t trackingNtuple/tree -o pu200_2_${TAG}.root  -x 2  > pu200_2.log 2>&1 &"
echo "time ./doAnalysis -i ${SAMPLE} -n -1 -t trackingNtuple/tree -o pu200_3_${TAG}.root  -x 3  > pu200_3.log 2>&1 &"
echo "time ./doAnalysis -i ${SAMPLE} -n -1 -t trackingNtuple/tree -o pu200_4_${TAG}.root  -x 4  > pu200_4.log 2>&1 &"
echo "time ./doAnalysis -i ${SAMPLE} -n -1 -t trackingNtuple/tree -o pu200_5_${TAG}.root  -x 5  > pu200_5.log 2>&1 &"
echo "time ./doAnalysis -i ${SAMPLE} -n -1 -t trackingNtuple/tree -o pu200_6_${TAG}.root  -x 6  > pu200_6.log 2>&1 &"
echo "time ./doAnalysis -i ${SAMPLE} -n -1 -t trackingNtuple/tree -o pu200_7_${TAG}.root  -x 7  > pu200_7.log 2>&1 &"
echo "time ./doAnalysis -i ${SAMPLE} -n -1 -t trackingNtuple/tree -o pu200_8_${TAG}.root  -x 8  > pu200_8.log 2>&1 &"
echo "time ./doAnalysis -i ${SAMPLE} -n -1 -t trackingNtuple/tree -o pu200_9_${TAG}.root  -x 9  > pu200_9.log 2>&1 &"
# echo "time ./doAnalysis -i ${SAMPLE} -n -1 -t trackingNtuple/tree -o pu200_10_${TAG}.root -x 10 > pu200_10.log 2>&1 &"
# echo "time ./doAnalysis -i ${SAMPLE} -n -1 -t trackingNtuple/tree -o pu200_11_${TAG}.root -x 11 > pu200_11.log 2>&1 &"
# echo "time ./doAnalysis -i ${SAMPLE} -n -1 -t trackingNtuple/tree -o pu200_12_${TAG}.root -x 12 > pu200_12.log 2>&1 &"
# echo "time ./doAnalysis -i ${SAMPLE} -n -1 -t trackingNtuple/tree -o pu200_13_${TAG}.root -x 13 > pu200_13.log 2>&1 &"
# echo "time ./doAnalysis -i ${SAMPLE} -n -1 -t trackingNtuple/tree -o pu200_14_${TAG}.root -x 14 > pu200_14.log 2>&1 &"
# echo "time ./doAnalysis -i ${SAMPLE} -n -1 -t trackingNtuple/tree -o pu200_15_${TAG}.root -x 15 > pu200_15.log 2>&1 &"
# echo "time ./doAnalysis -i ${SAMPLE} -n -1 -t trackingNtuple/tree -o pu200_16_${TAG}.root -x 16 > pu200_16.log 2>&1 &"
# echo "time ./doAnalysis -i ${SAMPLE} -n -1 -t trackingNtuple/tree -o pu200_17_${TAG}.root -x 17 > pu200_17.log 2>&1 &"
# echo "time ./doAnalysis -i ${SAMPLE} -n -1 -t trackingNtuple/tree -o pu200_18_${TAG}.root -x 18 > pu200_18.log 2>&1 &"
# echo "time ./doAnalysis -i ${SAMPLE} -n -1 -t trackingNtuple/tree -o pu200_19_${TAG}.root -x 19 > pu200_19.log 2>&1 &"
# echo "time ./doAnalysis -i ${SAMPLE} -n -1 -t trackingNtuple/tree -o pu200_20_${TAG}.root -x 20 > pu200_20.log 2>&1 &"
# echo "time ./doAnalysis -i ${SAMPLE} -n -1 -t trackingNtuple/tree -o pu200_21_${TAG}.root -x 21 > pu200_21.log 2>&1 &"
# echo "time ./doAnalysis -i ${SAMPLE} -n -1 -t trackingNtuple/tree -o pu200_22_${TAG}.root -x 22 > pu200_22.log 2>&1 &"
# echo "time ./doAnalysis -i ${SAMPLE} -n -1 -t trackingNtuple/tree -o pu200_23_${TAG}.root -x 23 > pu200_23.log 2>&1 &"
# echo "time ./doAnalysis -i ${SAMPLE} -n -1 -t trackingNtuple/tree -o pu200_24_${TAG}.root -x 24 > pu200_24.log 2>&1 &"
# echo "time ./doAnalysis -i ${SAMPLE} -n -1 -t trackingNtuple/tree -o pu200_25_${TAG}.root -x 25 > pu200_25.log 2>&1 &"
# echo "time ./doAnalysis -i ${SAMPLE} -n -1 -t trackingNtuple/tree -o pu200_26_${TAG}.root -x 26 > pu200_26.log 2>&1 &"
# echo "time ./doAnalysis -i ${SAMPLE} -n -1 -t trackingNtuple/tree -o pu200_27_${TAG}.root -x 27 > pu200_27.log 2>&1 &"
# echo "time ./doAnalysis -i ${SAMPLE} -n -1 -t trackingNtuple/tree -o pu200_28_${TAG}.root -x 28 > pu200_28.log 2>&1 &"
# echo "time ./doAnalysis -i ${SAMPLE} -n -1 -t trackingNtuple/tree -o pu200_29_${TAG}.root -x 29 > pu200_29.log 2>&1 &"
time ./doAnalysis -i ${SAMPLE} -n -1 -t trackingNtuple/tree -o pu200_0_${TAG}.root  -x 0  > pu200_0.log 2>&1 & 
time ./doAnalysis -i ${SAMPLE} -n -1 -t trackingNtuple/tree -o pu200_1_${TAG}.root  -x 1  > pu200_1.log 2>&1 &
time ./doAnalysis -i ${SAMPLE} -n -1 -t trackingNtuple/tree -o pu200_2_${TAG}.root  -x 2  > pu200_2.log 2>&1 &
time ./doAnalysis -i ${SAMPLE} -n -1 -t trackingNtuple/tree -o pu200_3_${TAG}.root  -x 3  > pu200_3.log 2>&1 &
time ./doAnalysis -i ${SAMPLE} -n -1 -t trackingNtuple/tree -o pu200_4_${TAG}.root  -x 4  > pu200_4.log 2>&1 &
time ./doAnalysis -i ${SAMPLE} -n -1 -t trackingNtuple/tree -o pu200_5_${TAG}.root  -x 5  > pu200_5.log 2>&1 &
time ./doAnalysis -i ${SAMPLE} -n -1 -t trackingNtuple/tree -o pu200_6_${TAG}.root  -x 6  > pu200_6.log 2>&1 &
time ./doAnalysis -i ${SAMPLE} -n -1 -t trackingNtuple/tree -o pu200_7_${TAG}.root  -x 7  > pu200_7.log 2>&1 &
time ./doAnalysis -i ${SAMPLE} -n -1 -t trackingNtuple/tree -o pu200_8_${TAG}.root  -x 8  > pu200_8.log 2>&1 &
time ./doAnalysis -i ${SAMPLE} -n -1 -t trackingNtuple/tree -o pu200_9_${TAG}.root  -x 9  > pu200_9.log 2>&1 &
# time ./doAnalysis -i ${SAMPLE} -n -1 -t trackingNtuple/tree -o pu200_10_${TAG}.root -x 10 > pu200_10.log 2>&1 &
# time ./doAnalysis -i ${SAMPLE} -n -1 -t trackingNtuple/tree -o pu200_11_${TAG}.root -x 11 > pu200_11.log 2>&1 &
# time ./doAnalysis -i ${SAMPLE} -n -1 -t trackingNtuple/tree -o pu200_12_${TAG}.root -x 12 > pu200_12.log 2>&1 &
# time ./doAnalysis -i ${SAMPLE} -n -1 -t trackingNtuple/tree -o pu200_13_${TAG}.root -x 13 > pu200_13.log 2>&1 &
# time ./doAnalysis -i ${SAMPLE} -n -1 -t trackingNtuple/tree -o pu200_14_${TAG}.root -x 14 > pu200_14.log 2>&1 &
# time ./doAnalysis -i ${SAMPLE} -n -1 -t trackingNtuple/tree -o pu200_15_${TAG}.root -x 15 > pu200_15.log 2>&1 &
# time ./doAnalysis -i ${SAMPLE} -n -1 -t trackingNtuple/tree -o pu200_16_${TAG}.root -x 16 > pu200_16.log 2>&1 &
# time ./doAnalysis -i ${SAMPLE} -n -1 -t trackingNtuple/tree -o pu200_17_${TAG}.root -x 17 > pu200_17.log 2>&1 &
# time ./doAnalysis -i ${SAMPLE} -n -1 -t trackingNtuple/tree -o pu200_18_${TAG}.root -x 18 > pu200_18.log 2>&1 &
# time ./doAnalysis -i ${SAMPLE} -n -1 -t trackingNtuple/tree -o pu200_19_${TAG}.root -x 19 > pu200_19.log 2>&1 &
# time ./doAnalysis -i ${SAMPLE} -n -1 -t trackingNtuple/tree -o pu200_20_${TAG}.root -x 20 > pu200_20.log 2>&1 &
# time ./doAnalysis -i ${SAMPLE} -n -1 -t trackingNtuple/tree -o pu200_21_${TAG}.root -x 21 > pu200_21.log 2>&1 &
# time ./doAnalysis -i ${SAMPLE} -n -1 -t trackingNtuple/tree -o pu200_22_${TAG}.root -x 22 > pu200_22.log 2>&1 &
# time ./doAnalysis -i ${SAMPLE} -n -1 -t trackingNtuple/tree -o pu200_23_${TAG}.root -x 23 > pu200_23.log 2>&1 &
# time ./doAnalysis -i ${SAMPLE} -n -1 -t trackingNtuple/tree -o pu200_24_${TAG}.root -x 24 > pu200_24.log 2>&1 &
# time ./doAnalysis -i ${SAMPLE} -n -1 -t trackingNtuple/tree -o pu200_25_${TAG}.root -x 25 > pu200_25.log 2>&1 &
# time ./doAnalysis -i ${SAMPLE} -n -1 -t trackingNtuple/tree -o pu200_26_${TAG}.root -x 26 > pu200_26.log 2>&1 &
# time ./doAnalysis -i ${SAMPLE} -n -1 -t trackingNtuple/tree -o pu200_27_${TAG}.root -x 27 > pu200_27.log 2>&1 &
# time ./doAnalysis -i ${SAMPLE} -n -1 -t trackingNtuple/tree -o pu200_28_${TAG}.root -x 28 > pu200_28.log 2>&1 &
# time ./doAnalysis -i ${SAMPLE} -n -1 -t trackingNtuple/tree -o pu200_29_${TAG}.root -x 29 > pu200_29.log 2>&1 &

wait

hadd -f pu200_${TAG}.root pu200_*_${TAG}.root

mkdir -p results/${TAG}

mv pu200_${TAG}.root results/${TAG}
mv pu200_*_${TAG}.root results/${TAG}
mv pu200_*.log results/${TAG}
mv gitversion.txt results/${TAG}



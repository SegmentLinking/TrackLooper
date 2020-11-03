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

rm muon_*eff_${TAG}.root
echo ./doAnalysis -i trackingNtuple.root -n -1 -t trackingNtuple/tree -n -1 -e 1 -p 0 -o muon_mdeff_${TAG}.root 
./doAnalysis -i trackingNtuple.root -n -1 -t trackingNtuple/tree -n -1 -e 1 -p 0 -o muon_mdeff_${TAG}.root 
echo ./doAnalysis -i trackingNtuple.root -n -1 -t trackingNtuple/tree -n -1 -e 2 -p 0 -o muon_sgeff_${TAG}.root 
./doAnalysis -i trackingNtuple.root -n -1 -t trackingNtuple/tree -n -1 -e 2 -p 0 -o muon_sgeff_${TAG}.root 
echo ./doAnalysis -i trackingNtuple.root -n -1 -t trackingNtuple/tree -n -1 -e 3 -p 0 -o muon_tleff_${TAG}.root 
./doAnalysis -i trackingNtuple.root -n -1 -t trackingNtuple/tree -n -1 -e 3 -p 0 -o muon_tleff_${TAG}.root 

rm lowmuon_*eff_${TAG}.root
echo ./doAnalysis -i ../trackingNtuple/CMSSW_10_4_0/src/trackingNtuple_lowE.root -n -1 -t trackingNtuple/tree -n -1 -e 1 -p 3 -o lowmuon_mdeff_${TAG}.root
./doAnalysis -i ../trackingNtuple/CMSSW_10_4_0/src/trackingNtuple_lowE.root -n -1 -t trackingNtuple/tree -n -1 -e 1 -p 3 -o lowmuon_mdeff_${TAG}.root
echo ./doAnalysis -i ../trackingNtuple/CMSSW_10_4_0/src/trackingNtuple_lowE.root -n -1 -t trackingNtuple/tree -n -1 -e 2 -p 3 -o lowmuon_sgeff_${TAG}.root
./doAnalysis -i ../trackingNtuple/CMSSW_10_4_0/src/trackingNtuple_lowE.root -n -1 -t trackingNtuple/tree -n -1 -e 2 -p 3 -o lowmuon_sgeff_${TAG}.root
echo ./doAnalysis -i ../trackingNtuple/CMSSW_10_4_0/src/trackingNtuple_lowE.root -n -1 -t trackingNtuple/tree -n -1 -e 3 -p 3 -o lowmuon_tleff_${TAG}.root
./doAnalysis -i ../trackingNtuple/CMSSW_10_4_0/src/trackingNtuple_lowE.root -n -1 -t trackingNtuple/tree -n -1 -e 3 -p 3 -o lowmuon_tleff_${TAG}.root

rm muonpt1_*eff_${TAG}.root
echo ./doAnalysis -i ../trackingNtuple/CMSSW_10_4_0/src/trackingNtuple_pt1.root -n -1 -t trackingNtuple/tree -n -1 -e 1 -p 1 -o muonpt1_mdeff_${TAG}.root
./doAnalysis -i ../trackingNtuple/CMSSW_10_4_0/src/trackingNtuple_pt1.root -n -1 -t trackingNtuple/tree -n -1 -e 1 -p 1 -o muonpt1_mdeff_${TAG}.root
echo ./doAnalysis -i ../trackingNtuple/CMSSW_10_4_0/src/trackingNtuple_pt1.root -n -1 -t trackingNtuple/tree -n -1 -e 2 -p 1 -o muonpt1_sgeff_${TAG}.root
./doAnalysis -i ../trackingNtuple/CMSSW_10_4_0/src/trackingNtuple_pt1.root -n -1 -t trackingNtuple/tree -n -1 -e 2 -p 1 -o muonpt1_sgeff_${TAG}.root
echo ./doAnalysis -i ../trackingNtuple/CMSSW_10_4_0/src/trackingNtuple_pt1.root -n -1 -t trackingNtuple/tree -n -1 -e 3 -p 1 -o muonpt1_tleff_${TAG}.root
./doAnalysis -i ../trackingNtuple/CMSSW_10_4_0/src/trackingNtuple_pt1.root -n -1 -t trackingNtuple/tree -n -1 -e 3 -p 1 -o muonpt1_tleff_${TAG}.root

rm muonpt0p95to1p05_*eff_${TAG}.root
echo ./doAnalysis -i ../trackingNtuple/CMSSW_10_4_0/src/trackingNtuple_pt0p95_1p05_10MuGun.root -n -1 -t trackingNtuple/tree -n -1 -e 1 -p 2 -o muonpt0p95to1p05_mdeff_${TAG}.root
./doAnalysis -i ../trackingNtuple/CMSSW_10_4_0/src/trackingNtuple_pt0p95_1p05_10MuGun.root -n -1 -t trackingNtuple/tree -n -1 -e 1 -p 2 -o muonpt0p95to1p05_mdeff_${TAG}.root
echo ./doAnalysis -i ../trackingNtuple/CMSSW_10_4_0/src/trackingNtuple_pt0p95_1p05_10MuGun.root -n -1 -t trackingNtuple/tree -n -1 -e 2 -p 2 -o muonpt0p95to1p05_sgeff_${TAG}.root
./doAnalysis -i ../trackingNtuple/CMSSW_10_4_0/src/trackingNtuple_pt0p95_1p05_10MuGun.root -n -1 -t trackingNtuple/tree -n -1 -e 2 -p 2 -o muonpt0p95to1p05_sgeff_${TAG}.root
echo ./doAnalysis -i ../trackingNtuple/CMSSW_10_4_0/src/trackingNtuple_pt0p95_1p05_10MuGun.root -n -1 -t trackingNtuple/tree -n -1 -e 3 -p 2 -o muonpt0p95to1p05_tleff_${TAG}.root
./doAnalysis -i ../trackingNtuple/CMSSW_10_4_0/src/trackingNtuple_pt0p95_1p05_10MuGun.root -n -1 -t trackingNtuple/tree -n -1 -e 3 -p 2 -o muonpt0p95to1p05_tleff_${TAG}.root

rm muonpt0p5to1p5_*eff_${TAG}.root
echo ./doAnalysis -i ../trackingNtuple/CMSSW_10_4_0/src/trackingNtuple_pt0p5_1p5_10MuGun.root -n -1 -t trackingNtuple/tree -n -1 -e 1 -p 3 -o muonpt0p5to1p5_mdeff_${TAG}.root
./doAnalysis -i ../trackingNtuple/CMSSW_10_4_0/src/trackingNtuple_pt0p5_1p5_10MuGun.root -n -1 -t trackingNtuple/tree -n -1 -e 1 -p 3 -o muonpt0p5to1p5_mdeff_${TAG}.root
echo ./doAnalysis -i ../trackingNtuple/CMSSW_10_4_0/src/trackingNtuple_pt0p5_1p5_10MuGun.root -n -1 -t trackingNtuple/tree -n -1 -e 2 -p 3 -o muonpt0p5to1p5_sgeff_${TAG}.root
./doAnalysis -i ../trackingNtuple/CMSSW_10_4_0/src/trackingNtuple_pt0p5_1p5_10MuGun.root -n -1 -t trackingNtuple/tree -n -1 -e 2 -p 3 -o muonpt0p5to1p5_sgeff_${TAG}.root
echo ./doAnalysis -i ../trackingNtuple/CMSSW_10_4_0/src/trackingNtuple_pt0p5_1p5_10MuGun.root -n -1 -t trackingNtuple/tree -n -1 -e 3 -p 3 -o muonpt0p5to1p5_tleff_${TAG}.root
./doAnalysis -i ../trackingNtuple/CMSSW_10_4_0/src/trackingNtuple_pt0p5_1p5_10MuGun.root -n -1 -t trackingNtuple/tree -n -1 -e 3 -p 3 -o muonpt0p5to1p5_tleff_${TAG}.root

mkdir -p results/${TAG}_eff_study

mv muon_*eff_${TAG}.root results/${TAG}_eff_study
mv lowmuon_*eff_${TAG}.root results/${TAG}_eff_study
mv muonpt1_*eff_${TAG}.root results/${TAG}_eff_study
mv muonpt0p5to1p5_*eff_${TAG}.root results/${TAG}_eff_study
mv muonpt0p95to1p05_*eff_${TAG}.root results/${TAG}_eff_study
mv gitversion.txt results/${TAG}_eff_study

muonsamples="muon lowmuon muonpt1 muonpt0p95to1p05 muonpt0p5to1p5"

for muonsample in ${muonsamples}; do
    echo python plot.py 1 results/${TAG}_eff_study/${muonsample}_mdeff_${TAG}.root
    python plot.py 1 results/${TAG}_eff_study/${muonsample}_mdeff_${TAG}.root
    mv plots/mdeff results/${TAG}_eff_study/mdeff_${muonsample}
    echo python plot.py 2 results/${TAG}_eff_study/${muonsample}_sgeff_${TAG}.root
    python plot.py 2 results/${TAG}_eff_study/${muonsample}_sgeff_${TAG}.root
    mv plots/sgeff results/${TAG}_eff_study/sgeff_${muonsample}
    echo python plot.py 3 results/${TAG}_eff_study/${muonsample}_tleff_${TAG}.root
    python plot.py 3 results/${TAG}_eff_study/${muonsample}_tleff_${TAG}.root
    mv plots/tleff results/${TAG}_eff_study/tleff_${muonsample}
done

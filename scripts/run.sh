make clean; make -j
./doAnalysis -i trackingNtuple.root -n -1 -t trackingNtuple/tree -n -1 -d
python plot.py

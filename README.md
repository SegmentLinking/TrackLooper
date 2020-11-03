# TrackLooper

## Install

    git clone --recurse-submodules git@github.com:bsathian/TrackLooper.git
    cd TrackLooper
    source setup.sh # set ROOT
    make clean;
    cd SDL && make -j && cd -
    make -j all;

## Running efficiency plot


## Obtaining detector geometry via the centroids

    ./bin/sdl -i "/nfs-7/userdata/phchang/trackingNtuple/trackingNtuple_*.root" -t trackingNtuple/tree -d -m 1 # creates data/all_sim_hits.root
    python scripts/module_bound_fit.py # creates data/phase2.txt

## Building map

Folloing command creates ```data/conn.txt```.  Afterwards, copy it to a ```conn.py``` format.

    ./bin/sdl -m 0 -i /nfs-7/userdata/phchang/trackingNtuple/trackingNtuple_400_pt_0p8_2p0.root,/nfs-7/userdata/phchang/trackingNtuple/trackingNtuple_10MuGun.root,/nfs-7/userdata/phchang/trackingNtuple/trackingNtuple_hundred_pt_0p8_2p0.root

Then run the following command. (Watch out for the import statements)

    python SDLpython/build_module_connection.py


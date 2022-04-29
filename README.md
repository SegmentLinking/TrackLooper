# TrackLooper for Hackathon

## Pre-requisites
### Nvidia machine specific instructions

* Immediately after login, type `set +T` otherwise none of the `source` commands will work!

### ROOT Install instructions

* Install the .tar.gz version of root from the instructions (https://root.cern/download/root_v6.26.02.Linux-ubuntu20-x86_64-gcc9.4.tar.gz)
* Change the C++ standard to `c++14` to `c++17` in `root/bin/root-config` file
* Activate ROOT using `source root/bin/thisroot.sh` (can add this to your `.bashrc` if you're going to use this regularly)

### Load CUDA modules provided by the cluster
* Type `module load cuda/11.4.4` (Version number is important!)

## Quick start guide

### Download the input file that contains the hits

    wget http://uaf-10.t2.ucsd.edu/~bsathian/LST_files/trackingNtuple_with_PUinfo_500_evts.root

### Go to your working directory

    mkdir /go/to/your/working/directory
    cd /go/to/your/working/directory
    
### Clone the repository

    git clone git@github.com:SegmentLinking/TrackLooper.git
    cd TrackLooper/
    git checkout hackathon

Download the new module maps

    cd data/
    wget http://uaf-10.t2.ucsd.edu/~bsathian/LST_files/lst_module_maps.tar.gz
    tar -xzvf lst_module_maps.tar.gz 

Once every new shell, source the setup script to initilaize the enviornment.

    source setup.sh

Compile the code with option flags

    sdl_make_tracklooper -m8
    -x: run with explicit instead of unified memory
    -c: run with the cmssw caching allocator
    -l: toggle on preloading of hits
    -h: show help screen with all options
 Run the code
 
    ./bin/sdl -n <nevents> -v <verbose> -w <writeout> -s <streams> -i <dataset>
    -i: PU200; muonGun, etc
    -n: number of events
    -v: 0-no printout; 1- timing printout only; 2- multiplicity printout
    -s: number of streams/events in flight
    -w: 0- no writout; 2- full ntuple writeout
    


## Validation
Run the validation on sample

    sdl_validate_segment_linking <dataset> 
    Runs Explicit and unified versions over 200 events by default
    dataset: PU200, muonGun, pionGun, etc

Run the validation on specific version of GPU implementation

    sdl_validate_segment_linking <dataset> unified
    sdl_validate_segment_linking <dataset> unified_cache
    sdl_validate_segment_linking <dataset> explicit
    sdl_validate_segment_linking <dataset> explicit_newgrid
    (can optionally add in number of events as 3rd option)

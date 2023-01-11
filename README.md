# TrackLooper


## Quick Start


### Setting up LSTPerformanceWeb (only for lnx7188)

For lnx7188 this needs to be done once

    cd /cdat/tem/${USER}/
    git clone git@github.com:SegmentLinking/LSTPerformanceWeb.git

### Running the code

    git clone --recursive git@github.com:SegmentLinking/TrackLooper.git
    cd TrackLooper/
    source setup.sh
    # source setup_cgpu.sh # if you are on cgpu-1
    sdl_make_tracklooper -mc
    sdl -i PU200 -o LSTNtuple.root
    createPerfNumDenHists -i LSTNtuple.root -o LSTNumDen.root
    lst_plot_performance.py LSTNumDen.root -t "myTag"
    # python3 efficiency/python/lst_plot_performance.py LSTNumDen.root -t "myTag" # if you are on cgpu-1

The above can be even simplified

    git clone --recursive git@github.com:SegmentLinking/TrackLooper.git
    cd TrackLooper/
    source setup.sh
    # source setup_cgpu.sh # if you are on cgpu-1
    sdl_run -f -mc -s PU200 -n -1 -t myTag

## Instructions

Log on to phi3 or lnx7188
Go to your working directory

    mkdir /go/to/your/working/directory
    cd /go/to/your/working/directory
    
Clone the repository

    git clone git@github.com:SegmentLinking/TrackLooper.git
    cd TrackLooper/

Once every new shell, source the setup script to initilaize the enviornment.

    source setup.sh

Compile the code with option flags

    sdl_make_tracklooper -mc
    -c: run with the cmssw caching allocator
    -h: show help screen with all options

Run the code
 
    sdl -n <nevents> -v <verbose> -w <writeout> -s <streams> -i <dataset> -o <output>

    -i: PU200; muonGun, etc
    -n: number of events
    -v: 0-no printout; 1- timing printout only; 2- multiplicity printout
    -s: number of streams/events in flight
    -w: 0- no writout; 1- minimum writeout; 2- full ntuple writeout
    -o: provide an output root file name (e.g. LSTNtuple.root)
    
Plotting numerators and denominators of performance plots

    createPerfNumDenHists -i <input> -o <output> [-g <pdgids> -n <nevents>]

    -i: Path to LSTNtuple.root
    -o: provide an output root file name (e.g. num_den_hist.root)
    -n: (optional) number of events
    -g: (optional) comma separated pdgids to add more efficiency plots with different sim particle slices
    
Plotting performance plots

    lst_plot_performance.py num_den_hist.root -t "mywork"

When running on ```cgpu-1``` remember to specify python3 as there is no python.
The shebang on the ```lst_plot_performance.py``` is not updated as ```lnx7188``` works with python2....

    python3 efficiency/python/lst_plot_performance.py num_den_hist.root -t "mywork" # If running on cgpu-1
                                                                                                                                                           
Comparing two different runs

    lst_plot_performance.py \
        num_den_hist_1.root \     # Reference
        num_den_hist_2.root \     # New work
        -l BaseLine,MyNewWork \   # Labeling
        -t "mywork"

## Validation
Run the validation on sample

    sdl_validate_segment_linking <dataset> 
    Runs Explicit version over 200 events by default
    dataset: PU200, muonGun, pionGun, etc

Run the validation on specific version of GPU implementation

    sdl_validate_segment_linking <dataset> explicit
    sdl_validate_segment_linking <dataset> explicit_cache
    (can optionally add in number of events as 3rd option)


## CMSSW Integration
This is the a complete set of instruction on how the TrackLooper code
can be linked as an external tool in CMSSW:

### Build TrackLooper
```bash
git clone git@github.com:SegmentLinking/TrackLooper.git
cd TrackLooper/
source setup.sh
sdl_make_tracklooper -m8xc2
cd ..
```

### Set up `TrackLooper` as an external
```bash
export SCRAM_ARCH=slc7_amd64_gcc10
cmsrel CMSSW_12_6_0_pre2
cd CMSSW_12_6_0_pre2/src
cmsenv
git cms-init
cat <<EOF >lst.xml
<tool name="lst" version="1.0">
  <client>
    <environment name="LSTBASE" default="$PWD/../../TrackLooper"/>
    <environment name="LIBDIR" default="\$LSTBASE/SDL"/>
    <environment name="INCLUDE" default="\$LSTBASE"/>
  </client>
  <runtime name="LST_BASE" value="\$LSTBASE"/>
  <lib name="sdl"/>
</tool>
EOF
scram setup lst.xml
cmsenv
git cms-checkdeps -a
scram b -j 12
```

Including the line
```
<use name="lst"/>
```
in the relevant package `BuildFile.xml` allows for
including our headers in the code of that package.

# TrackLooper

## Quick start guide

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

# TrackLooper

## Quick start guide

Log on to phi3

    git clone git@github.com:SegmentLinking/TrackLooper.git
    cd TrackLooper/
    source setup.sh

Go to your working directory

    mkdir /go/to/your/working/directory
    cd /go/to/your/working/directory

Run the code

    make_tracklooper -m
    sdl -i muonGun -o muonGun_200evt_gpu.root -n 200
    sdl -i muonGun -o muonGun_200evt_cpu.root -n 200 --cpu
    make_efficiency -i ../muonGun_200evt_gpu.root -p 4 -g 13
    make_efficiency -i ../muonGun_200evt_cpu.root -p 4 -g 13

## Validation

Log on to phi3

    git clone git@github.com:SegmentLinking/TrackLooper.git
    cd TrackLooper/
    source setup.sh

Go to your working directory

    mkdir /go/to/your/working/directory
    cd /go/to/your/working/directory

Run the validation on muonGun sample for all 6 different configurations

    validate_segment_linking muonGun

Run the validation on specific version of GPU implementation

    validate_segment_linking muonGun unified
    validate_segment_linking muonGun unified_cache
    validate_segment_linking muonGun unified_newgrid
    validate_segment_linking muonGun unified_cache_newgrid
    validate_segment_linking muonGun explicit
    validate_segment_linking muonGun explicit_newgrid

Run the validation on pionGun sample for all 6 different configurations

    validate_segment_linking pionGun

Run the validation on specific version of GPU implementation

    validate_segment_linking pionGun unified
    validate_segment_linking pionGun unified_cache
    validate_segment_linking pionGun unified_newgrid
    validate_segment_linking pionGun unified_cache_newgrid
    validate_segment_linking pionGun explicit
    validate_segment_linking pionGun explicit_newgrid

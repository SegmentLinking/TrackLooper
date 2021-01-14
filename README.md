# TrackLooper

## Quick start guide

Log on to phi3

    git clone git@github.com:SegmentLinking/TrackLooper.git
    cd TrackLooper/
    source setup.sh
    cd /go/to/your/working/directory
    make_tracklooper -m
    sdl -i muonGun -o muonGun_200evt_gpu.root -n 200
    sdl -i muonGun -o muonGun_200evt_cpu.root -n 200 --cpu
    make_efficiency -i ../muonGun_200evt_gpu.root -p 4 -g 13
    make_efficiency -i ../muonGun_200evt_cpu.root -p 4 -g 13

## Validations on first 200 events of muon gun sample

Log on to phi3

    git clone git@github.com:SegmentLinking/TrackLooper.git
    cd TrackLooper/
    source setup.sh
    cd /go/to/your/working/directory
    validate_segment_linking muonGun
    validate_segment_linking pionGun

This will run CPU + 6 GPU configurations and create a comparison plot

#!/bin/bash

validate_muons() {

    # Validationg first 200 events

    rm -rf outputs
    mkdir -p outputs

    # CPU baseline
    sh make_script.sh -m
    ./bin/sdl -n 200 -o outputs/cpu.root --cpu
    cd efficiency/
    sh run.sh -i ../outputs/cpu.root -g 13 -p 4 -f
    cd ../

    run_gpu()
    {
        version=$1
        shift
        # GPU unified
        sh make_script.sh -m $*
        ./bin/sdl -n 200 -o outputs/gpu_${version}.root
        cd efficiency/
        sh run.sh -i ../outputs/gpu_${version}.root -g 13 -p 4 -f
        cd ../
    }

    run_gpu unified 
    run_gpu unified_cache -c
    run_gpu unified_cache_newgrid -c -g
    run_gpu unified_newgrid -g
    run_gpu explicit -x
    # run_gpu explicit_cache -x -c # Does not run on phi3
    # run_gpu explicit_cache_newgrid -x -c -g # Does not run on phi3
    run_gpu explicit_newgrid -x -g

    cd efficiency/
    sh compare.sh -i ../outputs/cpu.root -f

}

validate_muons

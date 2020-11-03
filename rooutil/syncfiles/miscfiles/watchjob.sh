#!/bin/bash

jobs=$@
njobs=$(echo $jobs | wc -w)

for i in {1..600} ; do
    echo "iteration $i"
    sleep 2m
    if [ $(condor_q -nobatch $jobs | wc -l) -eq $(condor_q -nobatch kittycats | wc -l) ]; then 
        echo "Job(s) $jobs ended on $(date)" | mail -s "[UAFNotify] $njobs job(s) ended on $(date)" amin.nj@gmail.com
        break
    else
        echo "Jobs still running"
    fi
done


# TrackLooper for Hackathon Princeton Open Hackathon 2023

## ROOT instalation
	wget https://root.cern/download/root_v6.26.02.Linux-ubuntu20-x86_64-gcc9.4.tar.gz
	tar -xzvf root_v6.26.02.Linux-ubuntu20-x86_64-gcc9.4.tar.gz

Change the C++ standard to c++14 to c++17 in root/bin/root-config file

## Fetching of input files
	wget http://uaf-10.t2.ucsd.edu/~evourlio/LSTFilesForPrincetonOpenhackathon2023/trackingNtuple_ttbar_PU200.root

## Work area
	mkdir workdir # Important that we go exactly one level down
	cd workdir
	git clone git@github.com:SegmentLinking/TrackLooper.git
	cd TrackLooper/
	git checkout princetonOpenHackathon2023

## Work setup
	set +T
	source root/bin/thisroot.sh
	source setup.sh
	sdl_make_tracklooper -mc # Code compilation

## Code running
	srun --ntasks=1 --nodes=1 --cpus-per-task=1 --partition=gpu --time=0:04:00 --gres=gpu:1 --pty /bin/bash # Example for requesting resources
    	./bin/sdl -n <nevents> -v <verbose> -w <writeout> -s <streams> -i <dataset>
        -i: input sample; use PU200
        -n: number of events; -1 for all events = 175
    	-v: 0-no printout; 1- timing printout only; 2- multiplicity printout
    	-s: number of streams/events in flight
    	-w: 0- no writout; 1- full ntuple writeout

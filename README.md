# TrackLooper for Hackathon Princeton Open Hackathon 2023

## To be run only once

### ROOT installation
	wget https://root.cern/download/root_v6.26.02.Linux-ubuntu20-x86_64-gcc9.4.tar.gz
	tar -xzvf root_v6.26.02.Linux-ubuntu20-x86_64-gcc9.4.tar.gz

- **Change the C++ standard to `c++14` to `c++17` in `root/bin/root-config` file**

### Fetching of input files
	wget http://uaf-10.t2.ucsd.edu/~evourlio/LSTFilesForPrincetonOpenhackathon2023/trackingNtuple_ttbar_PU200.root

### Work area
	mkdir workdir # Important that we go exactly one level down
	cd workdir
	git clone git@github.com:SegmentLinking/TrackLooper.git
	cd TrackLooper/
	git checkout princetonOpenHackathon2023

## To be run in every login

### Work setup
	set +T
	source root/bin/thisroot.sh
	cd workdir/TrackLooper
	source setup.sh
	sdl_make_tracklooper -mc # Code compilation

### Code running
	srun --ntasks=1 --nodes=1 --cpus-per-task=1 --partition=gpu --time=1:00:00 --gres=gpu:1 --pty /bin/bash # Example for requesting resources
	sdl -n <nevents> -v <verbose> -w <writeout> -s <streams> -i <dataset>
	-i: input sample; use PU200
	-n: number of events; -1 for all events = 175
	-v: 0-no printout; 1- timing printout only; 2- multiplicity printout
	-s: number of streams/events in flight
	-w: 0- no writout; 1- full ntuple writeout

## Performance and validation
- One quick easy to validate the code is to check the multiplicity of produced objects for a few events, e.g. for the 1st event:
```
	sdl -i PU200 -n 1 -v 2
```
- To check the timing, one can use the following command:
```
	sdl_timing PU200 explicit_cache 100
```
- To get the profiling report, we used:
```
	ncu --set full -o profiling_DATE_COMMIT -f --import-source on ./bin/sdl -n 1 -v 0 -i PU200
```

## Useful links
- Latest algorithm paper: https://arxiv.org/abs/2209.13711
- Latest complete set of slides on the algorithm: https://indico.jlab.org/event/459/contributions/11399/attachments/9632/14023/PhilipChang20230508_LST_CHEP2023Draft_v4.pdf
- Profiling report (on V100): https://www.classe.cornell.edu/~evourlio/www/SDL_GPU/profiling_cornell_master_3c0ba9a.ncu-rep
- Profiling report (on NVIDIA cluster A100): https://uaf-10.t2.ucsd.edu/~evourlio/SDL/profiling_20230609.ncu-rep

### SDL Refactored code

Instructions
1. Clone the `balajiCUDA` branch of the `TrackLooper` repository (https://github.com/bsathian/TrackLooper) recursively. This usually pulls the latest commit from this `SDL` repository. If not, clone this repository separately, replacing the `SDL` folder in the TrackLooper repository
2. Run `source setup.sh` in the TrackLooper folder (one level above after cloning). This sets up the environment.
3. `make` once inside the `SDL` folder to create the `sdl.so` shared object
4. `make` outside in the parent `TrackLooper` folder to create the executable called `doAnalysis` inside the `bin` folder
5. Download http://uaf-10.t2.ucsd.edu/~bsathian/SDL/test_10_events.root
6. Run the following command
`./bin/doAnalysis -i test_10_events.root -t trackingNtuple/tree -n 1 -d -v 1`
7. Tweak the argument of -n to change the number of events (up to 10)

# TrackLooper


## Quick Start


### Setting up LSTPerformanceWeb (only for lnx7188)

For lnx7188 this needs to be done once

    cd /cdat/tem/${USER}/
    git clone git@github.com:SegmentLinking/LSTPerformanceWeb.git

### Setting up the code

    git clone git@github.com:SegmentLinking/TrackLooper.git
    cd TrackLooper/
    # Source one of the commands below, depending on the site
    source setup_ucsd.sh # if on UCSD
    source setup_lnx7188.sh # if on Cornell
    source setup_hpg.sh # if on Florida

### Running the code

    sdl_make_tracklooper -mc
    sdl -i PU200 -o LSTNtuple.root
    createPerfNumDenHists -i LSTNtuple.root -o LSTNumDen.root
    lst_plot_performance.py LSTNumDen.root -t "myTag"
    # python3 efficiency/python/lst_plot_performance.py LSTNumDen.root -t "myTag" # if you are on cgpu-1

The above can be even simplified

    sdl_run -f -mc -s PU200 -n -1 -t myTag

## Command explanations

Compile the code with option flags

    sdl_make_tracklooper -mc
    -m: make clean binaries
    -c: run with the cmssw caching allocator
    -h: show help screen with all options

Run the code
 
    sdl -n <nevents> -v <verbose> -w <writeout> -s <streams> -i <dataset> -o <output>

    -i: PU200; muonGun, etc
    -n: number of events; default: all
    -v: 0-no printout; 1- timing printout only; 2- multiplicity printout; default: 0
    -s: number of streams/events in flight; default: 1
    -w: 0- no writeout; 1- minimum writeout; default: 1
    -o: provide an output root file name (e.g. LSTNtuple.root); default: debug.root
    -l: add lower level object (pT3, pT5, T5, etc.) branches to the output
    
Plotting numerators and denominators of performance plots

    createPerfNumDenHists -i <input> -o <output> [-g <pdgids> -n <nevents>]

    -i: Path to LSTNtuple.root
    -o: provide an output root file name (e.g. num_den_hist.root)
    -n: (optional) number of events
    -g: (optional) comma separated pdgids to add more efficiency plots with different sim particle slices
    
Plotting performance plots

    lst_plot_performance.py num_den_hist.root -t "mywork"

There are several options you can provide to restrict number of plots being produced.
And by default, it creates a certain set of objects.
One can specifcy the type, range, metric, etc.
To see the full information type

    lst_plot_performance.py --help

To give an example of plotting efficiency, object type of lower level T5, for |eta| < 2.5 only.

    lst_plot_performance.py num_den_hist.root -t "mywork" -m eff -o T5_lower -s loweta

NOTE: in order to plot lower level object, ```-l``` option must have been used during ```sdl``` step!

When running on ```cgpu-1``` remember to specify python3 as there is no python.
The shebang on the ```lst_plot_performance.py``` is not updated as ```lnx7188``` works with python2...

    python3 efficiency/python/lst_plot_performance.py num_den_hist.root -t "mywork" # If running on cgpu-1
                                                                                                                                                           
Comparing two different runs

    lst_plot_performance.py \
        num_den_hist_1.root \     # Reference
        num_den_hist_2.root \     # New work
        -L BaseLine,MyNewWork \   # Labeling
        -t "mywork" \
        --compare

## CMSSW Integration
This is the a complete set of instruction on how the TrackLooper code
can be linked as an external tool in CMSSW:

### Build TrackLooper
```bash
git clone git@github.com:SegmentLinking/TrackLooper.git
cd TrackLooper/
# Source one of the commands below, depending on the site
source setup_ucsd.sh # if on UCSD
source setup_lnx7188.sh # if on Cornell
source setup_hpg.sh # if on Florida
sdl_make_tracklooper -mc
cd ..
```

### Set up `TrackLooper` as an external
```bash
mkdir workingFolder # Create the folder you will be working in
cd workingFolder
cmsrel CMSSW_13_0_0_pre4
cd CMSSW_13_0_0_pre4/src
cmsenv
git cms-init
git remote add SegLink git@github.com:SegmentLinking/cmssw.git
git fetch SegLink CMSSW_13_0_0_pre4_LST_X
git cms-addpkg RecoTracker Configuration
git checkout CMSSW_13_0_0_pre4_LST_X
cat <<EOF >lst.xml
<tool name="lst" version="1.0">
  <client>
    <environment name="LSTBASE" default="$PWD/../../../TrackLooper"/>
    <environment name="LIBDIR" default="\$LSTBASE/SDL"/>
    <environment name="INCLUDE" default="\$LSTBASE"/>
  </client>
  <runtime name="LST_BASE" value="\$LSTBASE"/>
  <lib name="sdl"/>
</tool>
EOF
scram setup lst.xml
cmsenv
git cms-checkdeps -a -A
scram b -j 12
```

### Run the LST reconstruction in CMSSW
A simple test configuration of the LST reconstruction can be run with the command:
```bash
cmsRun RecoTracker/LST/test/LSTAlpakaTester.py
```

For a more complete workflow, one can run a modified version of the 21034.1 workflow.
To get the commands of this workflow, one can run:
```bash
runTheMatrix.py -w upgrade -n -e -l 21034.1
```

For convenience, the workflow has been run for 100 events and the output is stored here:
```bash
/ceph/cms/store/user/evourlio/LST/step2_21034.1_100Events.root
```

For enabling the LST reconstruction in the CMSSW tracking workflow, a modified step3 needs to be run.
This is based on the step3 command of the 21034.1 workflow with the following changes:
   - Use dummy PU input files by changing the argument of the `--pileup_input` flag to `file:file.root` (this will be fixed manually in the configuration file)
   - Add at the end of the command: `--procModifiers gpu,trackingLST,trackingIters01 --no_exec`

Run the command and modify the output configuration file with the following:
   - Add the following lines below the part where the import of the standard configurations happens:
     ```python
     process.load('Configuration.StandardSequences.Accelerators_cff')
     process.AlpakaServiceCudaAsync = cms.Service('AlpakaServiceCudaAsync')
     process.AlpakaServiceSerialSync = cms.Service('AlpakaServiceSerialSync')
     ```
   - Find the line that starts with process.mix.input.fileNames and change it to (supposing that the PU files are available locally, as is the case at the UCSD machines):
     ```python
     process.mix.input.fileNames = cms.untracked.vstring(['file:/data2/segmentlinking/PUSamplesForCMSSW1263/CMSSW_12_3_0_pre5/RelValMinBias_14TeV/GEN-SIM/123X_mcRun4_realistic_v4_2026D88noPU-v1/066fc95d-1cef-4469-9e08-3913973cd4ce.root', 'file:/data2/segmentlinking/PUSamplesForCMSSW1263/CMSSW_12_3_0_pre5/RelValMinBias_14TeV/GEN-SIM/123X_mcRun4_realistic_v4_2026D88noPU-v1/07928a25-231b-450d-9d17-e20e751323a1.root', 'file:/data2/segmentlinking/PUSamplesForCMSSW1263/CMSSW_12_3_0_pre5/RelValMinBias_14TeV/GEN-SIM/123X_mcRun4_realistic_v4_2026D88noPU-v1/26bd8fb0-575e-4201-b657-94cdcb633045.root', 'file:/data2/segmentlinking/PUSamplesForCMSSW1263/CMSSW_12_3_0_pre5/RelValMinBias_14TeV/GEN-SIM/123X_mcRun4_realistic_v4_2026D88noPU-v1/4206a9c5-44c2-45a5-aab2-1a8a6043a08a.root', 'file:/data2/segmentlinking/PUSamplesForCMSSW1263/CMSSW_12_3_0_pre5/RelValMinBias_14TeV/GEN-SIM/123X_mcRun4_realistic_v4_2026D88noPU-v1/55a372bf-a234-4111-8ce0-ead6157a1810.root', 'file:/data2/segmentlinking/PUSamplesForCMSSW1263/CMSSW_12_3_0_pre5/RelValMinBias_14TeV/GEN-SIM/123X_mcRun4_realistic_v4_2026D88noPU-v1/59ad346c-f405-4288-96d7-795f81c43fe8.root', 'file:/data2/segmentlinking/PUSamplesForCMSSW1263/CMSSW_12_3_0_pre5/RelValMinBias_14TeV/GEN-SIM/123X_mcRun4_realistic_v4_2026D88noPU-v1/7280f5ec-b71d-4579-a730-7ce2de0ff906.root', 'file:/data2/segmentlinking/PUSamplesForCMSSW1263/CMSSW_12_3_0_pre5/RelValMinBias_14TeV/GEN-SIM/123X_mcRun4_realistic_v4_2026D88noPU-v1/b93adc85-715f-477a-afc9-65f3241933ee.root', 'file:/data2/segmentlinking/PUSamplesForCMSSW1263/CMSSW_12_3_0_pre5/RelValMinBias_14TeV/GEN-SIM/123X_mcRun4_realistic_v4_2026D88noPU-v1/c7a0aa46-f55c-4b01-977f-34a397b71fba.root', 'file:/data2/segmentlinking/PUSamplesForCMSSW1263/CMSSW_12_3_0_pre5/RelValMinBias_14TeV/GEN-SIM/123X_mcRun4_realistic_v4_2026D88noPU-v1/e77fa467-97cb-4943-884f-6965b4eb0390.root'])
     ```
   - Modify the input and output file names accordingly, as well as the number of events.

Then, run the configuration file with `cmsRun`.

To get the DQM files, one would have to run step4 of the 21034.1 workflow with the following modifications:
   - Modify the `--pileup_input` flag as above, add `--no_exec` to the end of command and then run it.
   - Modify the output configuration file by including the appropriate PU files, as above, and by changing the input file (the one containing `inDQM` from the previous step) and number of events accordingly.

Running the configuration file with `cmsRun`, the output file will have a name starting with `DQM`. The name is the same every time this step runs,
so it is good practice to rename the file, e.g. to `tracking_Iters01LST.root`.
The MTV plots can be produced with the command:
```bash
makeTrackValidationPlots.py --extended tracking_Iters01LST.root
```

**Note:** In case one wants to run step2 as well, similar modifications as in step4 (PU files, `--no_exec` flag and input file/number of events) need to be applied.

### Inclusion of LST in other CMSSW packages
Including the line
```
<use name="lst"/>
```
in the relevant package `BuildFile.xml` allows for
including our headers in the code of that package.

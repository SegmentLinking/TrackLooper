# TrackLooper

Instructions to reproduce the results in DP-2023/019, presented at ACAT 2024.

## LST Integration in CMS Phase 2 HLT
This is the complete set of instruction on how the TrackLooper code
can be linked as an external tool in CMSSW and used in the Phase 2 HLT:

### Build TrackLooper
```bash
git clone git@github.com:SegmentLinking/TrackLooper.git
cd TrackLooper/
git checkout DP2023019_ACAT2024
# Source one of the commands below, depending on the site
source setup.sh # if on UCSD or Cornell
source setup_hpg.sh # if on Florida
# Compile with one of the commands below, depending on the configuration
sed -i '347s=.*=            if (dR2 < 5e-3f)=' SDL/TrackCandidate.h; sdl_make_tracklooper -mc # if running CKFOnLegacyTriplets
sed -i '347s=.*=            if (dR2 < 1e-3f)=' SDL/TrackCandidate.h; sdl_make_tracklooper -mc # if running CKFOnLSTQuads
sed -i '347s=.*=            if (dR2 < 1e-3f)=' SDL/TrackCandidate.h; sdl_make_tracklooper -mc23 # if running CKFOnLSTQuadsAndTriplets
cd ..
```

### Set up `TrackLooper` as an external CMSSW package
```bash
mkdir DP2023019_ACAT2024 # Create the folder you will be working in
cd DP2023019_ACAT2024
cmsrel CMSSW_13_2_0_pre2
cd CMSSW_13_2_0_pre2/src
cmsenv
git cms-init
git remote add SegLink git@github.com:SegmentLinking/cmssw.git
git fetch SegLink CMSSW_13_2_0_pre2_HLT_LST_DP2023019_ACAT2024
git cms-addpkg RecoTracker Configuration HLTrigger
git checkout CMSSW_13_2_0_pre2_HLT_LST_DP2023019_ACAT2024
#To include both the CPU library and GPU library into CMSSW, create 2 xml files. Before writing the following xml file, check that libsdl_cpu.so and libsdl_gpu.so can be found under the ../../../TrackLooper/SDL/ folder.
cat <<EOF >lst_cpu.xml
<tool name="lst_cpu" version="1.0">
  <client>
    <environment name="LSTBASE" default="$PWD/../../../TrackLooper"/>
    <environment name="LIBDIR" default="\$LSTBASE/SDL"/>
    <environment name="INCLUDE" default="\$LSTBASE"/>
  </client>
  <runtime name="LST_BASE" value="\$LSTBASE"/>
  <lib name="sdl_cpu"/>
</tool>
EOF
cat <<EOF >lst_cuda.xml
<tool name="lst_cuda" version="1.0">
  <client>
    <environment name="LSTBASE" default="$PWD/../../../TrackLooper"/>
    <environment name="LIBDIR" default="\$LSTBASE/SDL"/>
    <environment name="INCLUDE" default="\$LSTBASE"/>
  </client>
  <runtime name="LST_BASE" value="\$LSTBASE"/>
  <lib name="sdl_cuda"/>
</tool>
EOF
scram setup lst_cpu.xml
scram setup lst_cuda.xml
cmsenv
git cms-checkdeps -a -A
scram b -j 16
```

### Run the LST reconstruction in CMSSW

The configuration to run LST in the Phase 2 HLT context can be produced running:
```bash
cmsDriver.py Phase2_Patatrack -s HLT:75e33 \
    --processName=HLTX \
     --conditions auto:phase2_realistic_T21 \
     --geometry Extended2026D95 \
     --era Phase2C17I13M9 \
     --eventcontent FEVTDEBUGHLT \
     --filein=file:018f9975-045e-4a38-9ea3-da224b1262c5.root \
     -n 100 \
     --nThreads 10 \
     --nStreams 8 \
     --customise HLTrigger/Configuration/customizePhase2HLTTracking.customisePhase2HLTForTrackingOnly,HLTrigger/Configuration/customizePhase2HLTTracking.customisePhase2HLTForPatatrackLSTCKFOnX,HLTrigger/Configuration/customizePhase2HLTTracking.addTrackingValidation
```
The already-produced input file can be found in:
`/home/users/evourlio/LSTinHLT_P2UG/cmsswAlpaka/src/018f9975-045e-4a38-9ea3-da224b1262c5.root`

For the *CKFOnLegacyTriplets* configuration, `customisePhase2HLTForPatatrackLSTCKFOnX` should be replaced by `customisePhase2HLTForPatatrackLSTCKFOnLegacyTriplets` in the above command.

For the *CKFOnLSTQuadsAndTriplets* and *CKFOnLSTQuadsAndTriplets* configurations, `customisePhase2HLTForPatatrackLSTCKFOnX` should be replaced by `customisePhase2HLTForPatatrackLSTCKFOnLSTSeeds` in the above command.

With the DQM file produced from the previous step, the harvesting step can be run.
```
harvestTrackValidationPlots.py Phase2HLT_DQM.root -o plots.root
```
MTV validation plots can be produced with:
```
makeTrackValidationPlots.py plots.root
```
The nice plotter used to produce the DP note plots can be found in this repo under the name `myplotter.py` (review before using).


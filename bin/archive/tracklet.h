#ifndef process_h
#define process_h

#include <vector>
#include <map>
#include <tuple>

#include "trktree.h"
#include "rooutil.h"
#include "cxxopts.h"

#include "SDL/Event.h" // SDL::Event
#include "SDL/Module.h" // SDL::Module
#include "SDL/PrintUtil.h" // SDL::out
#include "SDL/EndcapGeometry.h" // SDL::EndcapGeometry
#include "SDL/ModuleConnectionMap.h" // SDL::ModuleConnectionMap

// Efficiency study modules
#include "Study.h"
#include "StudyEfficiency.h"
#include "StudySegmentEfficiency.h"
#include "StudyEndcapInefficiency.h"
#include "StudyBarreldPhiChangeCutThresholdValidity.h"
#include "StudyOccupancy.h"
#include "StudyMDOccupancy.h"
#include "StudyLinkedModule.h"
#include "StudyTrackletEfficiency.h"
#include "StudyTrackletSelection.h"
#include "StudyTrackletSelectionOnTruths.h"
#include "StudySegmentSelection.h"
#include "StudyHitOccupancy.h"
#include "StudyMiniDoubletOccupancy.h"
#include "StudySegmentOccupancy.h"
#include "StudyTrackCandidateSelection.h"
#include "StudyMiniDoubletEfficiency.h"
#include "StudySDLEfficiency.h"
#include "StudyTripletSelection.h"
#include "StudySDLInefficiency.h"
#include "StudyMTVEfficiency.h"
#include "StudyConditionalHitEfficiency.h"
#include "StudySDLEfficiencyv2.h"
#include "StudySDLMiniDoubletDebugNtupleWriter.h"
#include "StudySDLSegmentDebugNtupleWriter.h"
#include "StudySDLTrackletDebugNtupleWriter.h"
#include "StudySDLTrackCandidateDebugNtupleWriter.h"

#include "constants.h"

#include "AnalysisConfig.h"

#include "trkCore.h"

#endif

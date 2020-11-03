#ifndef process_h
#define process_h

#include <vector>
#include <map>
#include <tuple>
//#include <cuda_runtime.h>

#include "trktree.h"
#include "rooutil.h"
#include "cxxopts.h"
#include "trkCore.h"
#include "SDL/Event.cuh" // SDL::Event
#include "SDL/Module.cuh" // SDL::Module
#include "SDL/EndcapGeometry.h" // SDL::EndcapGeometry
#include "SDL/ModuleConnectionMap.h" // SDL::ModuleConnectionMap

// Efficiency study modules
#include "Study.h"

#include "StudyTrackletOccupancy.h"

#include "constants.h"

#include "AnalysisConfig.h"

#include "AnalysisInterface/EventForAnalysisInterface.h"

void printModuleConnectionInfo(std::ofstream&);
// bool hasAll12HitsInBarrel(unsigned int);
void processModuleBoundaryInfo();
int layer(int lay, int det);

#endif

#ifndef StudySDLTrackletDebugNtupleWriter_h
#define StudySDLTrackletDebugNtupleWriter_h

#include "SDL/Event.h"

#include "Study.h"

#include <vector>
#include <tuple>

#include "TString.h"
#include "trktree.h"
#include "constants.h"

#include "AnalysisConfig.h"

#include "trkCore.h"

class StudySDLTrackletDebugNtupleWriter : public Study
{
public:
    StudySDLTrackletDebugNtupleWriter();
    virtual void bookStudy();
    virtual void doStudy(SDL::Event& recoevent, std::vector<std::tuple<unsigned int, SDL::Event*>> simtrkevents);

};

#endif

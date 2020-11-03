#ifndef StudySDLSegmentDebugNtupleWriter_h
#define StudySDLSegmentDebugNtupleWriter_h

#include "SDL/Event.h"

#include "Study.h"

#include <vector>
#include <tuple>

#include "TString.h"
#include "trktree.h"
#include "constants.h"

#include "AnalysisConfig.h"

#include "trkCore.h"

class StudySDLSegmentDebugNtupleWriter : public Study
{
public:
    StudySDLSegmentDebugNtupleWriter();
    virtual void bookStudy();
    virtual void doStudy(SDL::Event& recoevent, std::vector<std::tuple<unsigned int, SDL::Event*>> simtrkevents);

};

#endif

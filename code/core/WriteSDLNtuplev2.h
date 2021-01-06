#ifndef WriteSDLNtuplev2_h
#define WriteSDLNtuplev2_h

//#include "SDL/Event.h"
#include "AnalysisInterface/EventForAnalysisInterface.h"

#include "Study.h"

#include <vector>
#include <tuple>

#include "TString.h"
#include "trktree.h"
#include "constants.h"

#include "AnalysisConfig.h"

#include "trkCore.h"

#include "SDLMath.h"

#include "AnalysisInterface/MathUtil.h"

#include <cppitertools/itertools.hpp>

class WriteSDLNtuplev2 : public Study
{

public:
    const char* studyname;

    WriteSDLNtuplev2(const char* studyName);

    virtual void bookStudy();
    virtual void doStudy(SDL::EventForAnalysisInterface& recoevent, std::vector<std::tuple<unsigned int, SDL::EventForAnalysisInterface*>> simtrkevents);

    void createHitsSimHitsSimTracksBranches();
    void createPixelSeedBranches();
    void createMiniDoubletBranches();
    void createSegmentBranches();
    void createPixelSegmentBranches();
    void createTripletBranches();
    void createQuadrupletBranches();
    void createPixelQuadrupletBranches();
    void createTrackCandidateBranches();

    void setHitsSimHitsSimTracksBranches();
    void setPixelSeedBranches();
    void setMiniDoubletBranches(SDL::EventForAnalysisInterface& recoevent);
    void setSegmentBranches(SDL::EventForAnalysisInterface& recoevent);
    void setPixelSegmentBranches(SDL::EventForAnalysisInterface& recoevent);
    void setTripletBranches(SDL::EventForAnalysisInterface& recoevent);
    void setQuadrupletBranches(SDL::EventForAnalysisInterface& recoevent);
    void setPixelQuadrupletBranches(SDL::EventForAnalysisInterface& recoevent);
    void setTrackCandidateBranches(SDL::EventForAnalysisInterface& recoevent);

};

#endif

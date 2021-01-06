#ifndef Study_h
#define Study_h

#include <vector>
#include <tuple>

#include "AnalysisInterface/EventForAnalysisInterface.h"

class Study
{

public:

    virtual void bookStudy();
    virtual void doStudy(SDL::EventForAnalysisInterface& recoevent, std::vector<std::tuple<unsigned int, SDL::EventForAnalysisInterface*>> simtrkevents);

};


#endif

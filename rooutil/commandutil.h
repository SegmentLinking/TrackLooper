#ifndef commandutil_h
#define commandutil_h

// C++
#include <tuple>
#include <stdlib.h>
#include <iostream>

// ROOT
#include "TString.h"

namespace RooUtil
{
    namespace CommandUtil
    {
        std::tuple<TString, TString, int> parseArgs(int argc, char** argv);
    }
}

#endif

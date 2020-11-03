#ifndef module_h
#define module_h

#include "ttreex.h"

#include <vector>
#include <memory>

namespace RooUtil
{
    class Module
    {
        public:
            TTreeX* tx;
            void SetTTreeX(RooUtil::TTreeX*);
            virtual void AddOutput();
            virtual void FillOutput(); // TODO rename
    };

    class Processor
    {
        public:
            TTreeX* tx;
            std::vector<unique_ptr<Module>> modules;
            Processor(RooUtil::TTreeX*);
            void AddModule(RooUtil::Module*);
            void AddOutputs();
            void SetOutputs(); // TODO rename
            void FillTree(); // TODO rename
            void FillOutputs(); // TODO rename
    };
}

#endif

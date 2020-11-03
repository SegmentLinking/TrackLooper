#ifndef tmvautil_h
#define tmvautil_h

#include <vector>

#include "TMVA/Reader.h"
#include "ttreex.h"

namespace RooUtil
{

    namespace TMVAUtil
    {
        TMVA::Reader* createReader(TString methodType, TString xmlpath);
        std::vector<float> getInputValues(TMVA::Reader* reader, RooUtil::TTreeX& tx);

        class ReaderX
        {
            public:
                ReaderX(TString methodType, TString xmlpath);
                float eval(RooUtil::TTreeX& tx);

                TString methodType;
                TMVA::Reader* reader;

        };
    }

}

#endif

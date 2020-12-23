#ifndef Algo_h
#define Algo_h

namespace SDL
{
    // Mini-Doublet algorithm enum
    enum MDAlgo
    {
        Default_MDAlgo = 0,
        AllComb_MDAlgo
    };

    // Segment algorithm enum
    enum SGAlgo
    {
        Default_SGAlgo = 0,
        AllComb_SGAlgo
    };

    // Tracklet algorithm enum
    enum TLAlgo
    {
        Default_TLAlgo = 0,
        DefaultNm1_TLAlgo,
        AllComb_TLAlgo
    };

    // Triplet algorithm enum
    enum TPAlgo
    {
        Default_TPAlgo = 0,
        AllComb_TPAlgo
    };

    // Track candidate algorithm enum
    enum TCAlgo
    {
        Default_TCAlgo = 0,
        AllComb_TCAlgo
    };

    // SDL processing enum
    enum SDLMode
    {
        All_SDLAlgo = 0,
        Tracklet_SDLAlgo,
        Triplet_SDLAlgo
    };
};

#endif

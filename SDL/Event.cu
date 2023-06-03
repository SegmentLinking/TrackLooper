#include "Event.cuh"

struct SDL::modules* SDL::modulesInGPU = nullptr;
struct SDL::pixelMap* SDL::pixelMapping = nullptr;
uint16_t SDL::nModules;
uint16_t SDL::nLowerModules;

SDL::Event::Event(cudaStream_t estream, bool verbose): queue(alpaka::getDevByIdx<Acc>(0u))
{
    int version;
    int driver;
    cudaRuntimeGetVersion(&version);
    cudaDriverGetVersion(&driver);
    //printf("version: %d Driver %d\n",version, driver);
    stream = estream;
    addObjects = verbose;
    hitsInGPU = nullptr;
    mdsInGPU = nullptr;
    segmentsInGPU = nullptr;
    tripletsInGPU = nullptr;
    quintupletsInGPU = nullptr;
    trackCandidatesInGPU = nullptr;
    pixelTripletsInGPU = nullptr;
    pixelQuintupletsInGPU = nullptr;
    rangesInGPU = nullptr;

    hitsInCPU = nullptr;
    rangesInCPU = nullptr;
    mdsInCPU = nullptr;
    segmentsInCPU = nullptr;
    tripletsInCPU = nullptr;
    trackCandidatesInCPU = nullptr;
    modulesInCPU = nullptr;
    modulesInCPUFull = nullptr;
    quintupletsInCPU = nullptr;
    pixelTripletsInCPU = nullptr;
    pixelQuintupletsInCPU = nullptr;

    //reset the arrays
    for(int i = 0; i<6; i++)
    {
        n_hits_by_layer_barrel_[i] = 0;
        n_minidoublets_by_layer_barrel_[i] = 0;
        n_segments_by_layer_barrel_[i] = 0;
        n_triplets_by_layer_barrel_[i] = 0;
        n_trackCandidates_by_layer_barrel_[i] = 0;
        n_quintuplets_by_layer_barrel_[i] = 0;
        if(i<5)
        {
            n_hits_by_layer_endcap_[i] = 0;
            n_minidoublets_by_layer_endcap_[i] = 0;
            n_segments_by_layer_endcap_[i] = 0;
            n_triplets_by_layer_endcap_[i] = 0;
            n_trackCandidates_by_layer_endcap_[i] = 0;
            n_quintuplets_by_layer_endcap_[i] = 0;
        }
    }
}

SDL::Event::~Event()
{
#ifdef CACHE_ALLOC
    if(rangesInGPU){rangesInGPU->freeMemoryCache();}
    if(hitsInGPU){hitsInGPU->freeMemoryCache();}
    if(mdsInGPU){mdsInGPU->freeMemoryCache();}
    if(tripletsInGPU){tripletsInGPU->freeMemoryCache();}
    if(quintupletsInGPU){quintupletsInGPU->freeMemoryCache();}
    if(pixelQuintupletsInGPU){pixelQuintupletsInGPU->freeMemoryCache();}
    if(pixelTripletsInGPU){pixelTripletsInGPU->freeMemoryCache();}
    if(trackCandidatesInGPU){trackCandidatesInGPU->freeMemoryCache();}
#else
    if(rangesInGPU){rangesInGPU->freeMemory();}
    if(hitsInGPU){hitsInGPU->freeMemory();}
    if(mdsInGPU){mdsInGPU->freeMemory(stream);}
    if(tripletsInGPU){tripletsInGPU->freeMemory(stream);}
    if(quintupletsInGPU){quintupletsInGPU->freeMemory(stream);}
    if(pixelQuintupletsInGPU){pixelQuintupletsInGPU->freeMemory(stream);}
    if(pixelTripletsInGPU){pixelTripletsInGPU->freeMemory(stream);}
    if(trackCandidatesInGPU){trackCandidatesInGPU->freeMemory(stream);}
#endif
    if(rangesInGPU != nullptr){cms::cuda::free_host(rangesInGPU);}
    if(mdsInGPU != nullptr){cms::cuda::free_host(mdsInGPU);}
    if(segmentsInGPU != nullptr){delete segmentsInGPU;}
    if(tripletsInGPU!= nullptr){cms::cuda::free_host(tripletsInGPU);}
    if(trackCandidatesInGPU!= nullptr){cms::cuda::free_host(trackCandidatesInGPU);}
    if(hitsInGPU!= nullptr){cms::cuda::free_host(hitsInGPU);}
    if(pixelTripletsInGPU!= nullptr){cms::cuda::free_host(pixelTripletsInGPU);}
    if(pixelQuintupletsInGPU!= nullptr){cms::cuda::free_host(pixelQuintupletsInGPU);}
    if(quintupletsInGPU!= nullptr){cms::cuda::free_host(quintupletsInGPU);}

    if(hitsInCPU != nullptr)
    {
        delete[] hitsInCPU->idxs;
        delete[] hitsInCPU->xs;
        delete[] hitsInCPU->ys;
        delete[] hitsInCPU->zs;
        delete[] hitsInCPU->moduleIndices;
        delete hitsInCPU->nHits;
        delete hitsInCPU;
    }
    if(rangesInCPU != nullptr)
    {
        delete[] rangesInCPU->quintupletModuleIndices;
        delete rangesInCPU;
    }

    if(mdsInCPU != nullptr)
    {
        delete[] mdsInCPU->anchorHitIndices;
        delete[] mdsInCPU->nMDs;
        delete mdsInCPU->nMemoryLocations;
        delete[] mdsInCPU->totOccupancyMDs;
        delete mdsInCPU;
    }

    if(segmentsInCPU != nullptr)
    {
        delete segmentsInCPU;
    }

    if(tripletsInCPU != nullptr)
    {
        delete[] tripletsInCPU->segmentIndices;
        delete[] tripletsInCPU->nTriplets;
        delete[] tripletsInCPU->totOccupancyTriplets;
        delete[] tripletsInCPU->betaIn;
        delete[] tripletsInCPU->betaOut;
        delete[] tripletsInCPU->pt_beta;
        delete[] tripletsInCPU->hitIndices;
        delete[] tripletsInCPU->logicalLayers;
        delete[] tripletsInCPU->lowerModuleIndices;
        delete tripletsInCPU->nMemoryLocations;
#ifdef CUT_VALUE_DEBUG
        delete[] tripletsInCPU->zOut;
        delete[] tripletsInCPU->zLo;
        delete[] tripletsInCPU->zHi;
        delete[] tripletsInCPU->zLoPointed;
        delete[] tripletsInCPU->zHiPointed;
        delete[] tripletsInCPU->sdlCut;
        delete[] tripletsInCPU->betaInCut;
        delete[] tripletsInCPU->betaOutCut;
        delete[] tripletsInCPU->deltaBetaCut;
        delete[] tripletsInCPU->rtLo;
        delete[] tripletsInCPU->rtHi;
        delete[] tripletsInCPU->kZ;
#endif
        delete tripletsInCPU;
    }
    if(quintupletsInCPU != nullptr)
    {
        delete[] quintupletsInCPU->tripletIndices;
        delete[] quintupletsInCPU->nQuintuplets;
        delete[] quintupletsInCPU->totOccupancyQuintuplets;
        delete[] quintupletsInCPU->lowerModuleIndices;
        delete[] quintupletsInCPU->innerRadius;
        delete[] quintupletsInCPU->outerRadius;
        delete[] quintupletsInCPU->regressionRadius;
        delete[] quintupletsInCPU->bridgeRadius;
        delete[] quintupletsInCPU->chiSquared;
        delete[] quintupletsInCPU->rzChiSquared;
        delete[] quintupletsInCPU->nonAnchorChiSquared;
        delete quintupletsInCPU;
    }

    if(pixelTripletsInCPU != nullptr)
    {
        delete[] pixelTripletsInCPU->tripletIndices;
        delete[] pixelTripletsInCPU->pixelSegmentIndices;
        delete[] pixelTripletsInCPU->pixelRadius;
        delete[] pixelTripletsInCPU->tripletRadius;
        delete pixelTripletsInCPU->nPixelTriplets;
        delete pixelTripletsInCPU->totOccupancyPixelTriplets;
        delete[] pixelTripletsInCPU->rzChiSquared;
        delete[] pixelTripletsInCPU->rPhiChiSquared;
        delete[] pixelTripletsInCPU->rPhiChiSquaredInwards;
        delete pixelTripletsInCPU;
    }

    if(pixelQuintupletsInCPU != nullptr)
    {
        delete[] pixelQuintupletsInCPU->pixelIndices;
        delete[] pixelQuintupletsInCPU->T5Indices;
        delete[] pixelQuintupletsInCPU->isDup;
        delete[] pixelQuintupletsInCPU->score;
        delete pixelQuintupletsInCPU->nPixelQuintuplets;
        delete pixelQuintupletsInCPU->totOccupancyPixelQuintuplets;
        delete[] pixelQuintupletsInCPU->rzChiSquared;
        delete[] pixelQuintupletsInCPU->rPhiChiSquared;
        delete[] pixelQuintupletsInCPU->rPhiChiSquaredInwards;
        delete pixelQuintupletsInCPU;
    }

    if(trackCandidatesInCPU != nullptr)
    {
        delete[] trackCandidatesInCPU->objectIndices;
        delete[] trackCandidatesInCPU->trackCandidateType;
        delete[] trackCandidatesInCPU->nTrackCandidates;
        delete[] trackCandidatesInCPU->hitIndices;
        delete[] trackCandidatesInCPU->logicalLayers;
        delete trackCandidatesInCPU;
    }


    if(modulesInCPU != nullptr)
    {
        delete[] modulesInCPU->nLowerModules;
        delete[] modulesInCPU->nModules;
        delete[] modulesInCPU->detIds;
        delete[] modulesInCPU->isLower;
        delete[] modulesInCPU->layers;
        delete[] modulesInCPU->subdets;
        delete[] modulesInCPU->rings;
        delete[] modulesInCPU->rods;
        delete[] modulesInCPU->modules;
        delete[] modulesInCPU->sides;
        delete[] modulesInCPU->eta;
        delete[] modulesInCPU->r;
        delete[] modulesInCPU;
    }
    if(modulesInCPUFull != nullptr)
    {
        delete[] modulesInCPUFull->detIds;
        delete[] modulesInCPUFull->moduleMap;
        delete[] modulesInCPUFull->nConnectedModules;
        delete[] modulesInCPUFull->drdzs;
        delete[] modulesInCPUFull->slopes;
        delete[] modulesInCPUFull->nModules;
        delete[] modulesInCPUFull->nLowerModules;
        delete[] modulesInCPUFull->layers;
        delete[] modulesInCPUFull->rings;
        delete[] modulesInCPUFull->modules;
        delete[] modulesInCPUFull->rods;
        delete[] modulesInCPUFull->subdets;
        delete[] modulesInCPUFull->sides;
        delete[] modulesInCPUFull->eta;
        delete[] modulesInCPUFull->r;
        delete[] modulesInCPUFull->isInverted;
        delete[] modulesInCPUFull->isLower;


        delete[] modulesInCPUFull->moduleType;
        delete[] modulesInCPUFull->moduleLayerType;
        delete[] modulesInCPUFull;
    }
}

void SDL::Event::resetEvent()
{
#ifdef CACHE_ALLOC
    if(hitsInGPU){hitsInGPU->freeMemoryCache();}
    if(mdsInGPU){mdsInGPU->freeMemoryCache();}
    if(quintupletsInGPU){quintupletsInGPU->freeMemoryCache();}
    if(rangesInGPU){rangesInGPU->freeMemoryCache();}
    if(tripletsInGPU){tripletsInGPU->freeMemoryCache();}
    if(pixelQuintupletsInGPU){pixelQuintupletsInGPU->freeMemoryCache();}
    if(pixelTripletsInGPU){pixelTripletsInGPU->freeMemoryCache();}
    if(trackCandidatesInGPU){trackCandidatesInGPU->freeMemoryCache();}
#else
    if(hitsInGPU){hitsInGPU->freeMemory();}
    if(quintupletsInGPU){quintupletsInGPU->freeMemory(stream);}
    if(rangesInGPU){rangesInGPU->freeMemory();}
    if(mdsInGPU){mdsInGPU->freeMemory(stream);}
    if(tripletsInGPU){tripletsInGPU->freeMemory(stream);}
    if(pixelQuintupletsInGPU){pixelQuintupletsInGPU->freeMemory(stream);}
    if(pixelTripletsInGPU){pixelTripletsInGPU->freeMemory(stream);}
    if(trackCandidatesInGPU){trackCandidatesInGPU->freeMemory(stream);}
#endif
    //reset the arrays
    for(int i = 0; i<6; i++)
    {
        n_hits_by_layer_barrel_[i] = 0;
        n_minidoublets_by_layer_barrel_[i] = 0;
        n_segments_by_layer_barrel_[i] = 0;
        n_triplets_by_layer_barrel_[i] = 0;
        n_trackCandidates_by_layer_barrel_[i] = 0;
        n_quintuplets_by_layer_barrel_[i] = 0;
        if(i<5)
        {
            n_hits_by_layer_endcap_[i] = 0;
            n_minidoublets_by_layer_endcap_[i] = 0;
            n_segments_by_layer_endcap_[i] = 0;
            n_triplets_by_layer_endcap_[i] = 0;
            n_trackCandidates_by_layer_endcap_[i] = 0;
            n_quintuplets_by_layer_endcap_[i] = 0;
        }
    }
    if(hitsInGPU){cms::cuda::free_host(hitsInGPU);
      hitsInGPU = nullptr;}
    if(mdsInGPU){cms::cuda::free_host(mdsInGPU);
      mdsInGPU = nullptr;}
    if(rangesInGPU){cms::cuda::free_host(rangesInGPU);
      rangesInGPU = nullptr;}
    if(segmentsInGPU){delete segmentsInGPU;
      segmentsInGPU = nullptr;}
    if(tripletsInGPU){cms::cuda::free_host(tripletsInGPU);
      tripletsInGPU = nullptr;}
    if(quintupletsInGPU){cms::cuda::free_host(quintupletsInGPU);
      quintupletsInGPU = nullptr;}
    if(trackCandidatesInGPU){cms::cuda::free_host(trackCandidatesInGPU);
      trackCandidatesInGPU = nullptr;}
    if(pixelTripletsInGPU){cms::cuda::free_host(pixelTripletsInGPU);
      pixelTripletsInGPU = nullptr;}
    if(pixelQuintupletsInGPU){cms::cuda::free_host(pixelQuintupletsInGPU);
      pixelQuintupletsInGPU = nullptr;}

    if(hitsInCPU != nullptr)
    {
        delete[] hitsInCPU->idxs;
        delete[] hitsInCPU->xs;
        delete[] hitsInCPU->ys;
        delete[] hitsInCPU->zs;
        delete[] hitsInCPU->moduleIndices;
        delete hitsInCPU->nHits;
        delete hitsInCPU;
        hitsInCPU = nullptr;
    }
    if(rangesInCPU != nullptr)
    {
        delete[] rangesInCPU->hitRanges;
        delete[] rangesInCPU->quintupletModuleIndices;
        delete rangesInCPU;
        rangesInCPU = nullptr;
    }

    if(mdsInCPU != nullptr)
    {
        delete[] mdsInCPU->anchorHitIndices;
        delete[] mdsInCPU->nMDs;
        delete[] mdsInCPU->totOccupancyMDs;
        delete mdsInCPU;
        mdsInCPU = nullptr;
    }

    if(segmentsInCPU != nullptr)
    {
        delete segmentsInCPU;
        segmentsInCPU = nullptr;
    }

    if(tripletsInCPU != nullptr)
    {
        delete[] tripletsInCPU->segmentIndices;
        delete[] tripletsInCPU->nTriplets;
        delete[] tripletsInCPU->totOccupancyTriplets;
        delete[] tripletsInCPU->betaIn;
        delete[] tripletsInCPU->betaOut;
        delete[] tripletsInCPU->pt_beta;
        delete[] tripletsInCPU->logicalLayers;
        delete[] tripletsInCPU->lowerModuleIndices;
        delete[] tripletsInCPU->hitIndices;
        delete tripletsInCPU;
        tripletsInCPU = nullptr;
    }
    if(quintupletsInCPU != nullptr)
    {
        delete[] quintupletsInCPU->tripletIndices;
        delete[] quintupletsInCPU->nQuintuplets;
        delete[] quintupletsInCPU->totOccupancyQuintuplets;
        delete[] quintupletsInCPU->lowerModuleIndices;
        delete[] quintupletsInCPU->innerRadius;
        delete[] quintupletsInCPU->outerRadius;
        delete[] quintupletsInCPU->regressionRadius;
        delete[] quintupletsInCPU->bridgeRadius;
        delete[] quintupletsInCPU->chiSquared;
        delete[] quintupletsInCPU->rzChiSquared;
        delete[] quintupletsInCPU->nonAnchorChiSquared;
        delete quintupletsInCPU;
        quintupletsInCPU = nullptr;
    }
    if(pixelTripletsInCPU != nullptr)
    {
        delete[] pixelTripletsInCPU->tripletIndices;
        delete[] pixelTripletsInCPU->pixelSegmentIndices;
        delete[] pixelTripletsInCPU->pixelRadius;
        delete[] pixelTripletsInCPU->tripletRadius;
        delete pixelTripletsInCPU->nPixelTriplets;
        delete pixelTripletsInCPU->totOccupancyPixelTriplets;
        delete[] pixelTripletsInCPU->rzChiSquared;
        delete[] pixelTripletsInCPU->rPhiChiSquared;
        delete[] pixelTripletsInCPU->rPhiChiSquaredInwards;
        delete pixelTripletsInCPU;
        pixelTripletsInCPU = nullptr;
    }

    if(pixelQuintupletsInCPU != nullptr)
    {
        delete[] pixelQuintupletsInCPU->pixelIndices;
        delete[] pixelQuintupletsInCPU->T5Indices;
        delete[] pixelQuintupletsInCPU->isDup;
        delete[] pixelQuintupletsInCPU->score;
        delete pixelQuintupletsInCPU->nPixelQuintuplets;
        delete pixelQuintupletsInCPU->totOccupancyPixelQuintuplets;
        delete[] pixelQuintupletsInCPU->rzChiSquared;
        delete[] pixelQuintupletsInCPU->rPhiChiSquared;
        delete[] pixelQuintupletsInCPU->rPhiChiSquaredInwards;
        delete pixelQuintupletsInCPU;
        pixelQuintupletsInCPU = nullptr;
    }
    if(trackCandidatesInCPU != nullptr)
    {
        delete[] trackCandidatesInCPU->objectIndices;
        delete[] trackCandidatesInCPU->trackCandidateType;
        delete[] trackCandidatesInCPU->nTrackCandidates;
        delete[] trackCandidatesInCPU->logicalLayers;
        delete[] trackCandidatesInCPU->hitIndices;
        delete[] trackCandidatesInCPU->lowerModuleIndices;
        delete trackCandidatesInCPU;
        trackCandidatesInCPU = nullptr;
    }

    if(modulesInCPU != nullptr)
    {
        delete[] modulesInCPU->nLowerModules;
        delete[] modulesInCPU->nModules;
        delete[] modulesInCPU->detIds;
        delete[] modulesInCPU->isLower;
        delete[] modulesInCPU->layers;
        delete[] modulesInCPU->subdets;
        delete[] modulesInCPU->rings;
        delete[] modulesInCPU->rods;
        delete[] modulesInCPU->modules;
        delete[] modulesInCPU->sides;
        delete[] modulesInCPU->eta;
        delete[] modulesInCPU->r;
        delete[] modulesInCPU;
        modulesInCPU = nullptr;
    }
    if(modulesInCPUFull != nullptr)
    {
        delete[] modulesInCPUFull->detIds;
        delete[] modulesInCPUFull->moduleMap;
        delete[] modulesInCPUFull->nConnectedModules;
        delete[] modulesInCPUFull->drdzs;
        delete[] modulesInCPUFull->slopes;
        delete[] modulesInCPUFull->nModules;
        delete[] modulesInCPUFull->nLowerModules;
        delete[] modulesInCPUFull->layers;
        delete[] modulesInCPUFull->rings;
        delete[] modulesInCPUFull->modules;
        delete[] modulesInCPUFull->rods;
        delete[] modulesInCPUFull->sides;
        delete[] modulesInCPUFull->subdets;
        delete[] modulesInCPUFull->eta;
        delete[] modulesInCPUFull->r;
        delete[] modulesInCPUFull->isInverted;
        delete[] modulesInCPUFull->isLower;


        delete[] modulesInCPUFull->moduleType;
        delete[] modulesInCPUFull->moduleLayerType;
        delete[] modulesInCPUFull;
        modulesInCPUFull = nullptr;
    }

}

void SDL::initModules(const char* moduleMetaDataFilePath)
{
    cudaStream_t default_stream = 0;
    if(modulesInGPU == nullptr)
    {
        cudaMallocHost(&modulesInGPU, sizeof(struct SDL::modules));
        cudaMallocHost(&pixelMapping, sizeof(struct SDL::pixelMap));
        //nModules gets filled here
        loadModulesFromFile(*modulesInGPU,nModules,nLowerModules, *pixelMapping, default_stream, moduleMetaDataFilePath);
        cudaStreamSynchronize(default_stream);
    }
    //resetObjectRanges(*modulesInGPU,nModules, default_stream);
}

void SDL::cleanModules()
{
    freeModules(*modulesInGPU, *pixelMapping);
    cudaFreeHost(modulesInGPU);
    cudaFreeHost(pixelMapping);
}

void SDL::Event::resetObjectsInModule()
{
    resetObjectRanges(*rangesInGPU,nModules,stream);
}

ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE int binary_search(
    unsigned int *data, // Array that we are searching over
    unsigned int search_val, // Value we want to find in data array
    unsigned int ndata) // Number of elements in data array
{
    unsigned int low = 0;
    unsigned int high = ndata - 1;

    while(low <= high)
    {
        unsigned int mid = (low + high)/2;
        unsigned int test_val = data[mid];
        if (test_val == search_val)
            return mid;
        else if (test_val > search_val)
            high = mid - 1;
        else
            low = mid + 1;
    }
    // Couldn't find search value in array.
    return -1;
}

struct moduleRangesKernel
{
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TAcc>
    ALPAKA_FN_ACC void operator()(
        TAcc const & acc,
        struct SDL::modules *modulesInGPU,
        struct SDL::hits *hitsInGPU,
        int const & nLowerModules) const
    {
        using Dim = alpaka::Dim<TAcc>;
        using Idx = alpaka::Idx<TAcc>;
        using Vec = alpaka::Vec<Dim, Idx>;

        Vec const globalThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
        Vec const gridThreadExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);

        for(int lowerIndex = globalThreadIdx[2]; lowerIndex < nLowerModules; lowerIndex += gridThreadExtent[2])
        {
            uint16_t upperIndex = modulesInGPU->partnerModuleIndices[lowerIndex];
            if (hitsInGPU->hitRanges[lowerIndex * 2] != -1 && hitsInGPU->hitRanges[upperIndex * 2] != -1)
            {
                hitsInGPU->hitRangesLower[lowerIndex] =  hitsInGPU->hitRanges[lowerIndex * 2]; 
                hitsInGPU->hitRangesUpper[lowerIndex] =  hitsInGPU->hitRanges[upperIndex * 2];
                hitsInGPU->hitRangesnLower[lowerIndex] = hitsInGPU->hitRanges[lowerIndex * 2 + 1] - hitsInGPU->hitRanges[lowerIndex * 2] + 1;
                hitsInGPU->hitRangesnUpper[lowerIndex] = hitsInGPU->hitRanges[upperIndex * 2 + 1] - hitsInGPU->hitRanges[upperIndex * 2] + 1;
            }
        }
    }
};

struct hitLoopKernel
{
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TAcc>
    ALPAKA_FN_ACC void operator()(
        TAcc const & acc,
        uint16_t Endcap, // Integer corresponding to endcap in module subdets
        uint16_t TwoS, // Integer corresponding to TwoS in moduleType
        unsigned int nModules, // Number of modules
        unsigned int nEndCapMap, // Number of elements in endcap map
        unsigned int* geoMapDetId, // DetId's from endcap map
        float* geoMapPhi, // Phi values from endcap map
        struct SDL::modules *modulesInGPU,
        struct SDL::hits *hitsInGPU,
        int const & nHits) const // Total number of hits in event
    {
        using Dim = alpaka::Dim<TAcc>;
        using Idx = alpaka::Idx<TAcc>;
        using Vec = alpaka::Vec<Dim, Idx>;

        Vec const globalThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
        Vec const gridThreadExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);

        for(int ihit = globalThreadIdx[2]; ihit < nHits; ihit += gridThreadExtent[2])
        {
            float ihit_x = hitsInGPU->xs[ihit];
            float ihit_y = hitsInGPU->ys[ihit];
            float ihit_z = hitsInGPU->zs[ihit];
            int iDetId = hitsInGPU->detid[ihit];
    
            hitsInGPU->rts[ihit] = alpaka::math::sqrt(acc, ihit_x*ihit_x + ihit_y*ihit_y);
            hitsInGPU->phis[ihit] = SDL::phi(acc, ihit_x,ihit_y);
            // Acosh has no supported implementation in Alpaka right now.
            hitsInGPU->etas[ihit] = ((ihit_z>0)-(ihit_z<0)) * SDL::temp_acosh(acc, alpaka::math::sqrt(acc, ihit_x*ihit_x+ihit_y*ihit_y+ihit_z*ihit_z)/hitsInGPU->rts[ihit]);
            int found_index = binary_search(modulesInGPU->mapdetId, iDetId, nModules);
            uint16_t lastModuleIndex = modulesInGPU->mapIdx[found_index];
    
            hitsInGPU->moduleIndices[ihit] = lastModuleIndex;
    
            if(modulesInGPU->subdets[lastModuleIndex] == Endcap && modulesInGPU->moduleType[lastModuleIndex] == TwoS)
            {
                int found_index = binary_search(geoMapDetId, iDetId, nEndCapMap);
                float phi = 0;
                // Unclear why these are not in map, but CPU map returns phi = 0 for all exceptions.
                if (found_index != -1)
                    phi = geoMapPhi[found_index];
                float cos_phi = alpaka::math::cos(acc, phi);
                hitsInGPU->highEdgeXs[ihit] = ihit_x + 2.5f * cos_phi;
                hitsInGPU->lowEdgeXs[ihit] = ihit_x - 2.5f * cos_phi;
                float sin_phi = alpaka::math::sin(acc, phi);
                hitsInGPU->highEdgeYs[ihit] = ihit_y + 2.5f * sin_phi;
                hitsInGPU->lowEdgeYs[ihit] = ihit_y - 2.5f * sin_phi;
            }
            // Need to set initial value if index hasn't been seen before.
            int old = alpaka::atomicOp<alpaka::AtomicCas>(acc, &(hitsInGPU->hitRanges[lastModuleIndex * 2]), -1, ihit);
            // For subsequent visits, stores the min value.
            if (old != -1)
                alpaka::atomicOp<alpaka::AtomicMin>(acc, &hitsInGPU->hitRanges[lastModuleIndex * 2], ihit);

            alpaka::atomicOp<alpaka::AtomicMax>(acc, &hitsInGPU->hitRanges[lastModuleIndex * 2 + 1], ihit);
        }
    }
};

void SDL::Event::addHitToEvent(std::vector<float> x, std::vector<float> y, std::vector<float> z, std::vector<unsigned int> detId, std::vector<unsigned int> idxInNtuple)
{
    // Use the actual number of hits instead of a max.
    const int nHits = x.size();

    // Get current device for future use.
    cudaGetDevice(&dev);

    // Initialize space on device/host for next event.
    if (hitsInGPU == nullptr)
    {
        hitsInGPU = (SDL::hits*)cms::cuda::allocate_host(sizeof(SDL::hits), stream);
        // Unclear why but this has to be 2*nHits to avoid crashing.
        createHitsInExplicitMemory(*hitsInGPU, nModules, 2*nHits, stream, 1);
    }
    if (rangesInGPU == nullptr)
    {
        rangesInGPU = (SDL::objectRanges*)cms::cuda::allocate_host(sizeof(SDL::objectRanges), stream);
    	createRangesInExplicitMemory(*rangesInGPU, nModules, stream, nLowerModules);
        resetObjectsInModule();
    }
    cudaStreamSynchronize(stream);
    // Copy the host arrays to the GPU.
    cudaMemcpyAsync(hitsInGPU->xs, &x[0], nHits*sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(hitsInGPU->ys, &y[0], nHits*sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(hitsInGPU->zs, &z[0], nHits*sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(hitsInGPU->detid, &detId[0], nHits*sizeof(unsigned int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(hitsInGPU->idxs, &idxInNtuple[0], nHits*sizeof(unsigned int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(hitsInGPU->nHits, &nHits, sizeof(unsigned int), cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);

    Vec const threadsPerBlock1(static_cast<Idx>(1), static_cast<Idx>(1), static_cast<Idx>(256));
    Vec const blocksPerGrid1(static_cast<Idx>(1), static_cast<Idx>(1), static_cast<Idx>(MAX_BLOCKS));
    WorkDiv const hit_loop_workdiv(blocksPerGrid1, threadsPerBlock1, elementsPerThread);

    hitLoopKernel hit_loop_kernel;
    auto const hit_loop_task(alpaka::createTaskKernel<Acc>(
        hit_loop_workdiv,
        hit_loop_kernel,
        Endcap,
        TwoS,
        nModules,
        SDL::endcapGeometry.nEndCapMap,
        SDL::endcapGeometry.geoMapDetId,
        SDL::endcapGeometry.geoMapPhi,
        modulesInGPU,
        hitsInGPU,
        nHits));

    alpaka::enqueue(queue, hit_loop_task);
    alpaka::wait(queue);

    Vec const threadsPerBlock2(static_cast<Idx>(1), static_cast<Idx>(1), static_cast<Idx>(256));
    Vec const blocksPerGrid2(static_cast<Idx>(1), static_cast<Idx>(1), static_cast<Idx>(MAX_BLOCKS));
    WorkDiv const module_ranges_workdiv(blocksPerGrid2, threadsPerBlock2, elementsPerThread);

    moduleRangesKernel module_ranges_kernel;
    auto const module_ranges_task(alpaka::createTaskKernel<Acc>(
        module_ranges_workdiv,
        module_ranges_kernel,
        modulesInGPU,
        hitsInGPU,
        nLowerModules));

    // Waiting isn't needed after second kernel call. Saves ~100 us.
    // This is because addPixelSegmentToEvent (which is run next) doesn't rely on hitsinGPU->hitrange variables.
    // Also, modulesInGPU->partnerModuleIndices is not alterned in addPixelSegmentToEvent.
    alpaka::enqueue(queue, module_ranges_task);
}

struct addPixelSegmentToEventKernel
{
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TAcc>
    ALPAKA_FN_ACC void operator()(
        TAcc const & acc,
        struct SDL::modules& modulesInGPU,
        struct SDL::objectRanges& rangesInGPU,
        struct SDL::hits& hitsInGPU,
        struct SDL::miniDoublets& mdsInGPU,
        struct SDL::segments<TAcc>& segmentsInGPU,
        unsigned int* hitIndices0,
        unsigned int* hitIndices1,
        unsigned int* hitIndices2,
        unsigned int* hitIndices3,
        float* dPhiChange,
        uint16_t pixelModuleIndex,
        const int size) const
    {
        using Dim = alpaka::Dim<TAcc>;
        using Idx = alpaka::Idx<TAcc>;
        using Vec = alpaka::Vec<Dim, Idx>;

        Vec const globalThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
        Vec const gridThreadExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);

        for(int tid = globalThreadIdx[2]; tid < size; tid += gridThreadExtent[2])
        {
            unsigned int innerMDIndex = rangesInGPU.miniDoubletModuleIndices[pixelModuleIndex] + 2*(tid);
            unsigned int outerMDIndex = rangesInGPU.miniDoubletModuleIndices[pixelModuleIndex] + 2*(tid) +1;
            unsigned int pixelSegmentIndex = rangesInGPU.segmentModuleIndices[pixelModuleIndex] + tid;

            addMDToMemory(mdsInGPU, hitsInGPU, modulesInGPU, hitIndices0[tid], hitIndices1[tid], pixelModuleIndex, 0,0,0,0,0,0,0,0,0,innerMDIndex);
            addMDToMemory(mdsInGPU, hitsInGPU, modulesInGPU, hitIndices2[tid], hitIndices3[tid], pixelModuleIndex, 0,0,0,0,0,0,0,0,0,outerMDIndex);

            //in outer hits - pt, eta, phi
            float slope = SDL::temp_sinh(acc, hitsInGPU.ys[mdsInGPU.outerHitIndices[innerMDIndex]]);
            float intercept = hitsInGPU.zs[mdsInGPU.anchorHitIndices[innerMDIndex]] - slope * hitsInGPU.rts[mdsInGPU.anchorHitIndices[innerMDIndex]];
            float score_lsq=(hitsInGPU.rts[mdsInGPU.anchorHitIndices[outerMDIndex]] * slope + intercept) - (hitsInGPU.zs[mdsInGPU.anchorHitIndices[outerMDIndex]]);
            score_lsq = score_lsq * score_lsq;

            unsigned int hits1[4];
            hits1[0] = hitsInGPU.idxs[mdsInGPU.anchorHitIndices[innerMDIndex]];
            hits1[1] = hitsInGPU.idxs[mdsInGPU.anchorHitIndices[outerMDIndex]];
            hits1[2] = hitsInGPU.idxs[mdsInGPU.outerHitIndices[innerMDIndex]];
            hits1[3] = hitsInGPU.idxs[mdsInGPU.outerHitIndices[outerMDIndex]];
            addPixelSegmentToMemory(acc, segmentsInGPU, mdsInGPU, innerMDIndex, outerMDIndex, pixelModuleIndex, hits1, hitIndices0[tid], hitIndices2[tid], dPhiChange[tid], pixelSegmentIndex, tid, score_lsq);
        }
    }
};

void SDL::Event::addPixelSegmentToEvent(std::vector<unsigned int> hitIndices0,std::vector<unsigned int> hitIndices1,std::vector<unsigned int> hitIndices2,std::vector<unsigned int> hitIndices3, std::vector<float> dPhiChange, std::vector<float> ptIn, std::vector<float> ptErr, std::vector<float> px, std::vector<float> py, std::vector<float> pz, std::vector<float> eta, std::vector<float> etaErr, std::vector<float> phi, std::vector<int> charge, std::vector<unsigned int> seedIdx, std::vector<int> superbin, std::vector<int8_t> pixelType, std::vector<char> isQuad)
{
    const int size = ptIn.size();
    unsigned int mdSize = 2 * size;
    uint16_t pixelModuleIndex = (*detIdToIndex)[1];

    if(mdsInGPU == nullptr)
    {
        mdsInGPU = (SDL::miniDoublets*)cms::cuda::allocate_host(sizeof(SDL::miniDoublets), stream);
        unsigned int nTotalMDs;
        cudaMemsetAsync(&rangesInGPU->miniDoubletModuleOccupancy[nLowerModules],N_MAX_PIXEL_MD_PER_MODULES, sizeof(unsigned int),stream);

        Vec const threadsPerBlockCreateMD(static_cast<Idx>(1), static_cast<Idx>(1), static_cast<Idx>(1024));
        Vec const blocksPerGridCreateMD(static_cast<Idx>(1), static_cast<Idx>(1), static_cast<Idx>(1));
        WorkDiv const createMDArrayRangesGPU_workDiv(blocksPerGridCreateMD, threadsPerBlockCreateMD, elementsPerThread);

        SDL::createMDArrayRangesGPU createMDArrayRangesGPU_kernel;
        auto const createMDArrayRangesGPUTask(alpaka::createTaskKernel<Acc>(
            createMDArrayRangesGPU_workDiv,
            createMDArrayRangesGPU_kernel,
            *modulesInGPU,
            *rangesInGPU));

        alpaka::enqueue(queue, createMDArrayRangesGPUTask);
        alpaka::wait(queue);

        cudaMemcpyAsync(&nTotalMDs,rangesInGPU->device_nTotalMDs,sizeof(unsigned int),cudaMemcpyDeviceToHost,stream);
        cudaStreamSynchronize(stream);
        nTotalMDs+= N_MAX_PIXEL_MD_PER_MODULES;

        createMDsInExplicitMemory(*mdsInGPU, nTotalMDs, nLowerModules, N_MAX_PIXEL_MD_PER_MODULES,stream);

        cudaMemcpyAsync(mdsInGPU->nMemoryLocations, &nTotalMDs, sizeof(unsigned int), cudaMemcpyHostToDevice, stream);
        cudaStreamSynchronize(stream);
    }
    if(segmentsInGPU == nullptr)
    {
        // can be optimized here: because we didn't distinguish pixel segments and outer-tracker segments and call them both "segments", so they use the index continuously.
        // If we want to further study the memory footprint in detail, we can separate the two and allocate different memories to them

        Vec const threadsPerBlockCreateSeg(static_cast<Idx>(1), static_cast<Idx>(1), static_cast<Idx>(1024));
        Vec const blocksPerGridCreateSeg(static_cast<Idx>(1), static_cast<Idx>(1), static_cast<Idx>(1));
        WorkDiv const createSegmentArrayRanges_workDiv(blocksPerGridCreateSeg, threadsPerBlockCreateSeg, elementsPerThread);

        SDL::createSegmentArrayRanges createSegmentArrayRanges_kernel;
        auto const createSegmentArrayRangesTask(alpaka::createTaskKernel<Acc>(
            createSegmentArrayRanges_workDiv,
            createSegmentArrayRanges_kernel,
            *modulesInGPU,
            *rangesInGPU,
            *mdsInGPU));

        alpaka::enqueue(queue, createSegmentArrayRangesTask);
        alpaka::wait(queue);

        cudaMemcpyAsync(&nTotalSegments,rangesInGPU->device_nTotalSegs,sizeof(unsigned int),cudaMemcpyDeviceToHost,stream);
        cudaStreamSynchronize(stream);
        nTotalSegments += N_MAX_PIXEL_SEGMENTS_PER_MODULE;

        segmentsInGPU = new SDL::segments<Acc>(nTotalSegments, nLowerModules, N_MAX_PIXEL_SEGMENTS_PER_MODULE, devAcc, queue);

        cudaMemcpyAsync(segmentsInGPU->nMemoryLocations, &nTotalSegments, sizeof(unsigned int), cudaMemcpyHostToDevice, stream);;
        cudaStreamSynchronize(stream);
    }

    auto hitIndices0_dev = allocBufWrapper<unsigned int>(devAcc, size);
    auto hitIndices1_dev = allocBufWrapper<unsigned int>(devAcc, size);
    auto hitIndices2_dev = allocBufWrapper<unsigned int>(devAcc, size);
    auto hitIndices3_dev = allocBufWrapper<unsigned int>(devAcc, size);
    auto dPhiChange_dev = allocBufWrapper<float>(devAcc, size);

    alpaka::memcpy(queue, hitIndices0_dev, hitIndices0, size);
    alpaka::memcpy(queue, hitIndices1_dev, hitIndices1, size);
    alpaka::memcpy(queue, hitIndices2_dev, hitIndices2, size);
    alpaka::memcpy(queue, hitIndices3_dev, hitIndices3, size);
    alpaka::memcpy(queue, dPhiChange_dev, dPhiChange, size);

    alpaka::memcpy(queue, segmentsInGPU->ptIn_buf, ptIn, size);
    alpaka::memcpy(queue, segmentsInGPU->ptErr_buf, ptErr, size);
    alpaka::memcpy(queue, segmentsInGPU->px_buf, px, size);
    alpaka::memcpy(queue, segmentsInGPU->py_buf, py, size);
    alpaka::memcpy(queue, segmentsInGPU->pz_buf, pz, size);
    alpaka::memcpy(queue, segmentsInGPU->etaErr_buf, etaErr, size);
    alpaka::memcpy(queue, segmentsInGPU->isQuad_buf, isQuad, size);
    alpaka::memcpy(queue, segmentsInGPU->eta_buf, eta, size);
    alpaka::memcpy(queue, segmentsInGPU->phi_buf, phi, size);
    alpaka::memcpy(queue, segmentsInGPU->charge_buf, charge, size);
    alpaka::memcpy(queue, segmentsInGPU->seedIdx_buf, seedIdx, size);
    alpaka::memcpy(queue, segmentsInGPU->superbin_buf, superbin, size);
    alpaka::memcpy(queue, segmentsInGPU->pixelType_buf, pixelType, size);

    cudaMemcpyAsync(&(segmentsInGPU->nSegments)[pixelModuleIndex], &size, sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(&(segmentsInGPU->totOccupancySegments)[pixelModuleIndex], &size, sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(&(mdsInGPU->nMDs)[pixelModuleIndex], &mdSize, sizeof(unsigned int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(&(mdsInGPU->totOccupancyMDs)[pixelModuleIndex], &mdSize, sizeof(unsigned int), cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);
    alpaka::wait(queue);

    Vec const threadsPerBlock(static_cast<Idx>(1), static_cast<Idx>(1), static_cast<Idx>(256));
    Vec const blocksPerGrid(static_cast<Idx>(1), static_cast<Idx>(1), static_cast<Idx>(MAX_BLOCKS));
    WorkDiv const addPixelSegmentToEvent_workdiv(blocksPerGrid, threadsPerBlock, elementsPerThread);

    addPixelSegmentToEventKernel addPixelSegmentToEvent_kernel;
    auto const addPixelSegmentToEvent_task(alpaka::createTaskKernel<Acc>(
        addPixelSegmentToEvent_workdiv,
        addPixelSegmentToEvent_kernel,
        *modulesInGPU,
        *rangesInGPU,
        *hitsInGPU,
        *mdsInGPU,
        *segmentsInGPU,
        alpaka::getPtrNative(hitIndices0_dev),
        alpaka::getPtrNative(hitIndices1_dev),
        alpaka::getPtrNative(hitIndices2_dev),
        alpaka::getPtrNative(hitIndices3_dev),
        alpaka::getPtrNative(dPhiChange_dev),
        pixelModuleIndex,
        size));

    alpaka::enqueue(queue, addPixelSegmentToEvent_task);
    alpaka::wait(queue);
}

void SDL::Event::addMiniDoubletsToEventExplicit()
{
    unsigned int* nMDsCPU;
    nMDsCPU = (unsigned int*)cms::cuda::allocate_host(nLowerModules * sizeof(unsigned int), stream);
    cudaMemcpyAsync(nMDsCPU,mdsInGPU->nMDs,nLowerModules*sizeof(unsigned int),cudaMemcpyDeviceToHost,stream);
    cudaStreamSynchronize(stream);

    short* module_subdets;
    module_subdets = (short*)cms::cuda::allocate_host(nLowerModules* sizeof(short), stream);
    cudaMemcpyAsync(module_subdets,modulesInGPU->subdets,nLowerModules*sizeof(short),cudaMemcpyDeviceToHost,stream);
    short* module_layers;
    module_layers = (short*)cms::cuda::allocate_host(nLowerModules * sizeof(short), stream);
    cudaMemcpyAsync(module_layers,modulesInGPU->layers,nLowerModules*sizeof(short),cudaMemcpyDeviceToHost,stream);
    int* module_hitRanges;
    module_hitRanges = (int*)cms::cuda::allocate_host(nLowerModules* 2*sizeof(int), stream);
    cudaMemcpyAsync(module_hitRanges,hitsInGPU->hitRanges,nLowerModules*2*sizeof(int),cudaMemcpyDeviceToHost,stream);

    cudaStreamSynchronize(stream);

    for(unsigned int i = 0; i<nLowerModules; i++)
    {
        if(!(nMDsCPU[i] == 0 or module_hitRanges[i * 2] == -1))
        {
            if(module_subdets[i] == Barrel)
            {
                n_minidoublets_by_layer_barrel_[module_layers[i] -1] += nMDsCPU[i];
            }
            else
            {
                n_minidoublets_by_layer_endcap_[module_layers[i] - 1] += nMDsCPU[i];
            }

        }
    }
    cms::cuda::free_host(nMDsCPU);
    cms::cuda::free_host(module_subdets);
    cms::cuda::free_host(module_layers);
    cms::cuda::free_host(module_hitRanges);
}

void SDL::Event::addSegmentsToEventExplicit()
{
    unsigned int* nSegmentsCPU;
    nSegmentsCPU = (unsigned int*)cms::cuda::allocate_host(nLowerModules * sizeof(unsigned int), stream);
    cudaMemcpyAsync(nSegmentsCPU,segmentsInGPU->nSegments,nLowerModules*sizeof(unsigned int),cudaMemcpyDeviceToHost,stream);

    short* module_subdets;
    module_subdets = (short*)cms::cuda::allocate_host(nLowerModules* sizeof(short), stream);
    cudaMemcpyAsync(module_subdets,modulesInGPU->subdets,nLowerModules*sizeof(short),cudaMemcpyDeviceToHost,stream);
    short* module_layers;
    module_layers = (short*)cms::cuda::allocate_host(nLowerModules * sizeof(short), stream);
    cudaMemcpyAsync(module_layers,modulesInGPU->layers,nLowerModules*sizeof(short),cudaMemcpyDeviceToHost,stream);

    cudaStreamSynchronize(stream);
    for(unsigned int i = 0; i<nLowerModules; i++)
    {
        if(!(nSegmentsCPU[i] == 0))
        {
            if(module_subdets[i] == Barrel)
            {
                n_segments_by_layer_barrel_[module_layers[i] - 1] += nSegmentsCPU[i];
            }
            else
            {
                n_segments_by_layer_endcap_[module_layers[i] -1] += nSegmentsCPU[i];
            }
        }
    }
    cms::cuda::free_host(nSegmentsCPU);
    cms::cuda::free_host(module_subdets);
    cms::cuda::free_host(module_layers);
}

void SDL::Event::createMiniDoublets()
{
    //hardcoded range numbers for this will come from studies!
    unsigned int nTotalMDs;
    cudaMemsetAsync(&rangesInGPU->miniDoubletModuleOccupancy[nLowerModules],N_MAX_PIXEL_MD_PER_MODULES, sizeof(unsigned int),stream);

    Vec const threadsPerBlockCreateMD(static_cast<Idx>(1), static_cast<Idx>(1), static_cast<Idx>(1024));
    Vec const blocksPerGridCreateMD(static_cast<Idx>(1), static_cast<Idx>(1), static_cast<Idx>(1));
    WorkDiv const createMDArrayRangesGPU_workDiv(blocksPerGridCreateMD, threadsPerBlockCreateMD, elementsPerThread);

    SDL::createMDArrayRangesGPU createMDArrayRangesGPU_kernel;
    auto const createMDArrayRangesGPUTask(alpaka::createTaskKernel<Acc>(
        createMDArrayRangesGPU_workDiv,
        createMDArrayRangesGPU_kernel,
        *modulesInGPU,
        *rangesInGPU));

    alpaka::enqueue(queue, createMDArrayRangesGPUTask);
    alpaka::wait(queue);

    cudaMemcpyAsync(&nTotalMDs,rangesInGPU->device_nTotalMDs,sizeof(unsigned int),cudaMemcpyDeviceToHost,stream);
    cudaStreamSynchronize(stream);
    nTotalMDs+=N_MAX_PIXEL_MD_PER_MODULES;

    if(mdsInGPU == nullptr)
    {
        mdsInGPU = (SDL::miniDoublets*)cms::cuda::allocate_host(sizeof(SDL::miniDoublets), stream);

        //FIXME: Add memory locations for pixel MDs
        createMDsInExplicitMemory(*mdsInGPU, nTotalMDs, nLowerModules, N_MAX_PIXEL_MD_PER_MODULES, stream);

    }
    cudaStreamSynchronize(stream);

    int maxThreadsPerModule=0;
    int* module_hitRanges;
    module_hitRanges = (int*)cms::cuda::allocate_host(nModules* 2*sizeof(int), stream);
    cudaMemcpyAsync(module_hitRanges,hitsInGPU->hitRanges,nModules*2*sizeof(int),cudaMemcpyDeviceToHost,stream);
    bool* module_isLower;
    module_isLower = (bool*)cms::cuda::allocate_host(nModules*sizeof(bool), stream);
    cudaMemcpyAsync(module_isLower,modulesInGPU->isLower,nModules*sizeof(bool),cudaMemcpyDeviceToHost,stream);
    bool* module_isInverted;
    module_isInverted = (bool*)cms::cuda::allocate_host(nModules*sizeof(bool), stream);
    cudaMemcpyAsync(module_isInverted,modulesInGPU->isInverted,nModules*sizeof(bool),cudaMemcpyDeviceToHost,stream);
    int* module_partnerModuleIndices;
    module_partnerModuleIndices = (int*)cms::cuda::allocate_host(nLowerModules * sizeof(unsigned int), stream);
    cudaMemcpyAsync(module_partnerModuleIndices, modulesInGPU->partnerModuleIndices, nLowerModules * sizeof(unsigned int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    for (uint16_t lowerModuleIndex=0; lowerModuleIndex<nLowerModules; lowerModuleIndex++) 
    {
        uint16_t upperModuleIndex = module_partnerModuleIndices[lowerModuleIndex];
        int lowerHitRanges = module_hitRanges[lowerModuleIndex*2];
        int upperHitRanges = module_hitRanges[upperModuleIndex*2];
        if(lowerHitRanges!=-1 && upperHitRanges!=-1) 
        {
            int nLowerHits = module_hitRanges[lowerModuleIndex * 2 + 1] - lowerHitRanges + 1;
            int nUpperHits = module_hitRanges[upperModuleIndex * 2 + 1] - upperHitRanges + 1;
            maxThreadsPerModule = maxThreadsPerModule > (nLowerHits*nUpperHits) ? maxThreadsPerModule : nLowerHits*nUpperHits;
        }
    }
    cms::cuda::free_host(module_hitRanges);
    cms::cuda::free_host(module_partnerModuleIndices);
    cms::cuda::free_host(module_isLower);
    cms::cuda::free_host(module_isInverted);

    Vec const threadsPerBlockCreateMDInGPU(static_cast<Idx>(1), static_cast<Idx>(16), static_cast<Idx>(32));
    Vec const blocksPerGridCreateMDInGPU(static_cast<Idx>(1), static_cast<Idx>(nLowerModules/threadsPerBlock[1]), static_cast<Idx>(1));
    WorkDiv const createMiniDoubletsInGPUv2_workDiv(blocksPerGridCreateMDInGPU, threadsPerBlockCreateMDInGPU, elementsPerThread);

    SDL::createMiniDoubletsInGPUv2 createMiniDoubletsInGPUv2_kernel;
    auto const createMiniDoubletsInGPUv2Task(alpaka::createTaskKernel<Acc>(
        createMiniDoubletsInGPUv2_workDiv,
        createMiniDoubletsInGPUv2_kernel,
        *modulesInGPU,
        *hitsInGPU,
        *mdsInGPU,
        *rangesInGPU));

    alpaka::enqueue(queue, createMiniDoubletsInGPUv2Task);

    Vec const threadsPerBlockAddMD(static_cast<Idx>(1), static_cast<Idx>(1), static_cast<Idx>(1024));
    Vec const blocksPerGridAddMD(static_cast<Idx>(1), static_cast<Idx>(1), static_cast<Idx>(1));
    WorkDiv const addMiniDoubletRangesToEventExplicit_workDiv(blocksPerGridAddMD, threadsPerBlockAddMD, elementsPerThread);

    SDL::addMiniDoubletRangesToEventExplicit addMiniDoubletRangesToEventExplicit_kernel;
    auto const addMiniDoubletRangesToEventExplicitTask(alpaka::createTaskKernel<Acc>(
        addMiniDoubletRangesToEventExplicit_workDiv,
        addMiniDoubletRangesToEventExplicit_kernel,
        *modulesInGPU,
        *mdsInGPU,
        *rangesInGPU,
        *hitsInGPU));

    alpaka::enqueue(queue, addMiniDoubletRangesToEventExplicitTask);
    alpaka::wait(queue);

    if(addObjects)
    {
        addMiniDoubletsToEventExplicit();
    }

}

void SDL::Event::createSegmentsWithModuleMap()
{
    if(segmentsInGPU == nullptr)
    {
        segmentsInGPU = new SDL::segments<Acc>(nTotalSegments, nLowerModules, N_MAX_PIXEL_SEGMENTS_PER_MODULE, devAcc, queue);
    }

    Vec const threadsPerBlockCreateSeg(static_cast<Idx>(1), static_cast<Idx>(1), static_cast<Idx>(64));
    Vec const blocksPerGridCreateSeg(static_cast<Idx>(1), static_cast<Idx>(1), static_cast<Idx>(nLowerModules));
    WorkDiv const createSegmentsInGPUv2_workDiv(blocksPerGridCreateSeg, threadsPerBlockCreateSeg, elementsPerThread);

    SDL::createSegmentsInGPUv2 createSegmentsInGPUv2_kernel;
    auto const createSegmentsInGPUv2Task(alpaka::createTaskKernel<Acc>(
        createSegmentsInGPUv2_workDiv,
        createSegmentsInGPUv2_kernel,
        *modulesInGPU,
        *mdsInGPU,
        *segmentsInGPU,
        *rangesInGPU));

    alpaka::enqueue(queue, createSegmentsInGPUv2Task);

    Vec const threadsPerBlockAddSeg(static_cast<Idx>(1), static_cast<Idx>(1), static_cast<Idx>(1024));
    Vec const blocksPerGridAddSeg(static_cast<Idx>(1), static_cast<Idx>(1), static_cast<Idx>(1));
    WorkDiv const addSegmentRangesToEventExplicit_workDiv(blocksPerGridAddSeg, threadsPerBlockAddSeg, elementsPerThread);

    SDL::addSegmentRangesToEventExplicit addSegmentRangesToEventExplicit_kernel;
    auto const addSegmentRangesToEventExplicitTask(alpaka::createTaskKernel<Acc>(
        addSegmentRangesToEventExplicit_workDiv,
        addSegmentRangesToEventExplicit_kernel,
        *modulesInGPU,
        *segmentsInGPU,
        *rangesInGPU));

    alpaka::enqueue(queue, addSegmentRangesToEventExplicitTask);
    alpaka::wait(queue);

    if(addObjects)
    {
        addSegmentsToEventExplicit();
    }
}


void SDL::Event::createTriplets()
{
    if(tripletsInGPU == nullptr)
    {
        tripletsInGPU = (SDL::triplets*)cms::cuda::allocate_host(sizeof(SDL::triplets), stream);
        unsigned int maxTriplets;

        Vec const threadsPerBlockCreateTrip(static_cast<Idx>(1), static_cast<Idx>(1), static_cast<Idx>(1024));
        Vec const blocksPerGridCreateTrip(static_cast<Idx>(1), static_cast<Idx>(1), static_cast<Idx>(1));
        WorkDiv const createTripletArrayRanges_workDiv(blocksPerGridCreateTrip, threadsPerBlockCreateTrip, elementsPerThread);

        SDL::createTripletArrayRanges createTripletArrayRanges_kernel;
        auto const createTripletArrayRangesTask(alpaka::createTaskKernel<Acc>(
            createTripletArrayRanges_workDiv,
            createTripletArrayRanges_kernel,
            *modulesInGPU,
            *rangesInGPU,
            *segmentsInGPU));

        alpaka::enqueue(queue, createTripletArrayRangesTask);
        alpaka::wait(queue);

        cudaMemcpyAsync(&maxTriplets,rangesInGPU->device_nTotalTrips,sizeof(unsigned int),cudaMemcpyDeviceToHost,stream);
        cudaStreamSynchronize(stream);

        createTripletsInExplicitMemory(*tripletsInGPU, maxTriplets, nLowerModules,stream);

        cudaMemcpyAsync(tripletsInGPU->nMemoryLocations, &maxTriplets, sizeof(unsigned int), cudaMemcpyHostToDevice, stream);
        cudaStreamSynchronize(stream);
    }
    //TODO:Move this also inside the ranges function
    uint16_t nonZeroModules=0;
    unsigned int max_InnerSeg=0;
    uint16_t *index = (uint16_t*)malloc(nLowerModules*sizeof(unsigned int));
    uint16_t *index_gpu;
    index_gpu = (uint16_t*)cms::cuda::allocate_device(dev, nLowerModules*sizeof(uint16_t), stream);
    unsigned int *nSegments = (unsigned int*)malloc(nLowerModules*sizeof(unsigned int));
    cudaMemcpyAsync((void *)nSegments, segmentsInGPU->nSegments, nLowerModules*sizeof(unsigned int), cudaMemcpyDeviceToHost,stream);
    cudaStreamSynchronize(stream);

    uint16_t* module_nConnectedModules;
    module_nConnectedModules = (uint16_t*)cms::cuda::allocate_host(nLowerModules* sizeof(uint16_t), stream);
    cudaMemcpyAsync(module_nConnectedModules,modulesInGPU->nConnectedModules,nLowerModules*sizeof(uint16_t),cudaMemcpyDeviceToHost,stream);
    cudaStreamSynchronize(stream);

    for (uint16_t innerLowerModuleIndex = 0; innerLowerModuleIndex <nLowerModules; innerLowerModuleIndex++)
    {
        uint16_t nConnectedModules = module_nConnectedModules[innerLowerModuleIndex];
        unsigned int nInnerSegments = nSegments[innerLowerModuleIndex];
        if (nConnectedModules != 0 and nInnerSegments != 0) 
        {
            index[nonZeroModules] = innerLowerModuleIndex;
            nonZeroModules++;
        }
        max_InnerSeg = max(max_InnerSeg, nInnerSegments);
    }
    cms::cuda::free_host(module_nConnectedModules);
    cudaMemcpyAsync(index_gpu, index, nonZeroModules*sizeof(uint16_t), cudaMemcpyHostToDevice,stream);
    cudaStreamSynchronize(stream);

    Vec const threadsPerBlockCreateTrip(static_cast<Idx>(1), static_cast<Idx>(16), static_cast<Idx>(16));
    Vec const blocksPerGridCreateTrip(static_cast<Idx>(MAX_BLOCKS), static_cast<Idx>(1), static_cast<Idx>(1));
    WorkDiv const createTripletsInGPUv2_workDiv(blocksPerGridCreateTrip, threadsPerBlockCreateTrip, elementsPerThread);

    SDL::createTripletsInGPUv2 createTripletsInGPUv2_kernel;
    auto const createTripletsInGPUv2Task(alpaka::createTaskKernel<Acc>(
        createTripletsInGPUv2_workDiv,
        createTripletsInGPUv2_kernel,
        *modulesInGPU,
        *mdsInGPU,
        *segmentsInGPU,
        *tripletsInGPU,
        *rangesInGPU,
        index_gpu,
        nonZeroModules));

    alpaka::enqueue(queue, createTripletsInGPUv2Task);

    Vec const threadsPerBlockAddTrip(static_cast<Idx>(1), static_cast<Idx>(1), static_cast<Idx>(1024));
    Vec const blocksPerGridAddTrip(static_cast<Idx>(1), static_cast<Idx>(1), static_cast<Idx>(1));
    WorkDiv const addTripletRangesToEventExplicit_workDiv(blocksPerGridAddTrip, threadsPerBlockAddTrip, elementsPerThread);

    SDL::addTripletRangesToEventExplicit addTripletRangesToEventExplicit_kernel;
    auto const addTripletRangesToEventExplicitTask(alpaka::createTaskKernel<Acc>(
        addTripletRangesToEventExplicit_workDiv,
        addTripletRangesToEventExplicit_kernel,
        *modulesInGPU,
        *tripletsInGPU,
        *rangesInGPU));

    alpaka::enqueue(queue, addTripletRangesToEventExplicitTask);
    alpaka::wait(queue);

    free(nSegments);
    free(index);
    cms::cuda::free_device(dev, index_gpu);

    if(addObjects)
    {
        addTripletsToEventExplicit();
    }
}

void SDL::Event::createTrackCandidates()
{
    uint16_t nEligibleModules;
    cudaMemcpyAsync(&nEligibleModules,rangesInGPU->nEligibleT5Modules,sizeof(uint16_t),cudaMemcpyDeviceToHost,stream);
    if(trackCandidatesInGPU == nullptr)
    {
        trackCandidatesInGPU = (SDL::trackCandidates*)cms::cuda::allocate_host(sizeof(SDL::trackCandidates), stream);
        createTrackCandidatesInExplicitMemory(*trackCandidatesInGPU, N_MAX_TRACK_CANDIDATES + N_MAX_PIXEL_TRACK_CANDIDATES,stream);
    }

    Vec const threadsPerBlock_crossCleanpT3(static_cast<Idx>(1), static_cast<Idx>(16), static_cast<Idx>(64));
    Vec const blocksPerGrid_crossCleanpT3(static_cast<Idx>(1), static_cast<Idx>(4), static_cast<Idx>(20));
    WorkDiv const crossCleanpT3_workDiv(blocksPerGrid_crossCleanpT3, blocksPerGrid_crossCleanpT3, elementsPerThread);

    SDL::crossCleanpT3 crossCleanpT3_kernel;
    auto const crossCleanpT3Task(alpaka::createTaskKernel<Acc>(
        crossCleanpT3_workDiv,
        crossCleanpT3_kernel,
        *modulesInGPU,
        *rangesInGPU,
        *pixelTripletsInGPU,
        *segmentsInGPU,
        *pixelQuintupletsInGPU));

    alpaka::enqueue(queue, crossCleanpT3Task);
    alpaka::wait(queue);

    //adding objects
    Vec const threadsPerBlock_addpT3asTrackCandidatesInGPU(static_cast<Idx>(1), static_cast<Idx>(1), static_cast<Idx>(512));
    Vec const blocksPerGrid_addpT3asTrackCandidatesInGPU(static_cast<Idx>(1), static_cast<Idx>(1), static_cast<Idx>(1));
    WorkDiv const addpT3asTrackCandidatesInGPU_workDiv(blocksPerGrid_addpT3asTrackCandidatesInGPU, threadsPerBlock_addpT3asTrackCandidatesInGPU, elementsPerThread);

    SDL::addpT3asTrackCandidatesInGPU addpT3asTrackCandidatesInGPU_kernel;
    auto const addpT3asTrackCandidatesInGPUTask(alpaka::createTaskKernel<Acc>(
        addpT3asTrackCandidatesInGPU_workDiv,
        addpT3asTrackCandidatesInGPU_kernel,
        nLowerModules,
        *pixelTripletsInGPU,
        *trackCandidatesInGPU,
        *segmentsInGPU,
        *rangesInGPU));

    alpaka::enqueue(queue, addpT3asTrackCandidatesInGPUTask);
    alpaka::wait(queue);

    Vec const threadsPerBlockRemoveDupQuints(static_cast<Idx>(1), static_cast<Idx>(16), static_cast<Idx>(32));
    Vec const blocksPerGridRemoveDupQuints(static_cast<Idx>(1), static_cast<Idx>(max(nEligibleModules/16,1)), static_cast<Idx>(max(nEligibleModules/32,1)));
    WorkDiv const removeDupQuintupletsInGPUBeforeTC_workDiv(blocksPerGridRemoveDupQuints, threadsPerBlockRemoveDupQuints, elementsPerThread);

    SDL::removeDupQuintupletsInGPUBeforeTC removeDupQuintupletsInGPUBeforeTC_kernel;
    auto const removeDupQuintupletsInGPUBeforeTCTask(alpaka::createTaskKernel<Acc>(
        removeDupQuintupletsInGPUBeforeTC_workDiv,
        removeDupQuintupletsInGPUBeforeTC_kernel,
        *quintupletsInGPU,
        *rangesInGPU));

    alpaka::enqueue(queue, removeDupQuintupletsInGPUBeforeTCTask);
    alpaka::wait(queue);

    Vec const threadsPerBlock_crossCleanT5(static_cast<Idx>(32), static_cast<Idx>(1), static_cast<Idx>(32));
    Vec const blocksPerGrid_crossCleanT5(static_cast<Idx>((13296/32) + 1), static_cast<Idx>(1), static_cast<Idx>(MAX_BLOCKS));
    WorkDiv const crossCleanT5_workDiv(blocksPerGrid_crossCleanT5, threadsPerBlock_crossCleanT5, elementsPerThread);

    SDL::crossCleanT5 crossCleanT5_kernel;
    auto const crossCleanT5Task(alpaka::createTaskKernel<Acc>(
        crossCleanT5_workDiv,
        crossCleanT5_kernel,
        *modulesInGPU,
        *quintupletsInGPU,
        *pixelQuintupletsInGPU,
        *pixelTripletsInGPU,
        *rangesInGPU));

    alpaka::enqueue(queue, crossCleanT5Task);
    alpaka::wait(queue);

    Vec const threadsPerBlock_addT5asTrackCandidateInGPU(static_cast<Idx>(1), static_cast<Idx>(8), static_cast<Idx>(128));
    Vec const blocksPerGrid_addT5asTrackCandidateInGPU(static_cast<Idx>(1), static_cast<Idx>(8), static_cast<Idx>(10));
    WorkDiv const addT5asTrackCandidateInGPU_workDiv(blocksPerGrid_addT5asTrackCandidateInGPU, threadsPerBlock_addT5asTrackCandidateInGPU, elementsPerThread);

    SDL::addT5asTrackCandidateInGPU addT5asTrackCandidateInGPU_kernel;
    auto const addT5asTrackCandidateInGPUTask(alpaka::createTaskKernel<Acc>(
        addT5asTrackCandidateInGPU_workDiv,
        addT5asTrackCandidateInGPU_kernel,
        nLowerModules,
        *quintupletsInGPU,
        *trackCandidatesInGPU,
        *rangesInGPU));

    alpaka::enqueue(queue, addT5asTrackCandidateInGPUTask);
    alpaka::wait(queue);

    Vec const threadsPerBlockCheckHitspLS(static_cast<Idx>(1), static_cast<Idx>(16), static_cast<Idx>(16));
    Vec const blocksPerGridCheckHitspLS(static_cast<Idx>(1), static_cast<Idx>(MAX_BLOCKS*4), static_cast<Idx>(MAX_BLOCKS/4));
    WorkDiv const checkHitspLS_workDiv(blocksPerGridCheckHitspLS, threadsPerBlockCheckHitspLS, elementsPerThread);

    SDL::checkHitspLS checkHitspLS_kernel;
    auto const checkHitspLSTask(alpaka::createTaskKernel<Acc>(
        checkHitspLS_workDiv,
        checkHitspLS_kernel,
        *modulesInGPU,
        *segmentsInGPU,
        true));

    alpaka::enqueue(queue, checkHitspLSTask);
    alpaka::wait(queue);

    Vec const threadsPerBlock_crossCleanpLS(static_cast<Idx>(1), static_cast<Idx>(16), static_cast<Idx>(32));
    Vec const blocksPerGrid_crossCleanpLS(static_cast<Idx>(1), static_cast<Idx>(4), static_cast<Idx>(20));
    WorkDiv const crossCleanpLS_workDiv(blocksPerGrid_crossCleanpLS, threadsPerBlock_crossCleanpLS, elementsPerThread);

    SDL::crossCleanpLS crossCleanpLS_kernel;
    auto const crossCleanpLSTask(alpaka::createTaskKernel<Acc>(
        crossCleanpLS_workDiv,
        crossCleanpLS_kernel,
        *modulesInGPU,
        *rangesInGPU,
        *pixelTripletsInGPU,
        *trackCandidatesInGPU,
        *segmentsInGPU,
        *mdsInGPU,
        *hitsInGPU,
        *quintupletsInGPU));

    alpaka::enqueue(queue, crossCleanpLSTask);
    alpaka::wait(queue);

    Vec const threadsPerBlock_addpLSasTrackCandidateInGPU(static_cast<Idx>(1), static_cast<Idx>(1), static_cast<Idx>(384));
    Vec const blocksPerGrid_addpLSasTrackCandidateInGPU(static_cast<Idx>(1), static_cast<Idx>(1), static_cast<Idx>(MAX_BLOCKS));
    WorkDiv const addpLSasTrackCandidateInGPU_workDiv(blocksPerGrid_addpLSasTrackCandidateInGPU, threadsPerBlock_addpLSasTrackCandidateInGPU, elementsPerThread);

    SDL::addpLSasTrackCandidateInGPU addpLSasTrackCandidateInGPU_kernel;
    auto const addpLSasTrackCandidateInGPUTask(alpaka::createTaskKernel<Acc>(
        addpLSasTrackCandidateInGPU_workDiv,
        addpLSasTrackCandidateInGPU_kernel,
        nLowerModules,
        *trackCandidatesInGPU,
        *segmentsInGPU));

    alpaka::enqueue(queue, addpLSasTrackCandidateInGPUTask);
    alpaka::wait(queue);
}

void SDL::Event::createPixelTriplets()
{
    if(pixelTripletsInGPU == nullptr)
    {
        pixelTripletsInGPU = (SDL::pixelTriplets*)cms::cuda::allocate_host(sizeof(SDL::pixelTriplets), stream);
    }

    createPixelTripletsInExplicitMemory(*pixelTripletsInGPU, N_MAX_PIXEL_TRIPLETS,stream);

    unsigned int pixelModuleIndex = nLowerModules;
    int* superbins;
    int8_t* pixelTypes;
    unsigned int *nTriplets;
    unsigned int nInnerSegments = 0;
    cudaMemcpyAsync(&nInnerSegments, &(segmentsInGPU->nSegments[pixelModuleIndex]), sizeof(int), cudaMemcpyDeviceToHost,stream);
    nTriplets = (unsigned int*)cms::cuda::allocate_host(nLowerModules * sizeof(unsigned int), stream);
    cudaMemcpyAsync(nTriplets, tripletsInGPU->nTriplets, nLowerModules * sizeof(unsigned int), cudaMemcpyDeviceToHost,stream);
    superbins = (int*)cms::cuda::allocate_host(N_MAX_PIXEL_SEGMENTS_PER_MODULE*sizeof(int), stream);
    pixelTypes = (int8_t*)cms::cuda::allocate_host(N_MAX_PIXEL_SEGMENTS_PER_MODULE*sizeof(int8_t), stream);

    cudaMemcpyAsync(superbins,segmentsInGPU->superbin,N_MAX_PIXEL_SEGMENTS_PER_MODULE*sizeof(int),cudaMemcpyDeviceToHost,stream);
    cudaMemcpyAsync(pixelTypes,segmentsInGPU->pixelType,N_MAX_PIXEL_SEGMENTS_PER_MODULE*sizeof(int8_t),cudaMemcpyDeviceToHost,stream);

    unsigned int* connectedPixelSize_host;
    unsigned int* connectedPixelIndex_host;
    connectedPixelSize_host = (unsigned int*)cms::cuda::allocate_host(nInnerSegments* sizeof(unsigned int), stream);
    connectedPixelIndex_host = (unsigned int*)cms::cuda::allocate_host(nInnerSegments* sizeof(unsigned int), stream);
    unsigned int* connectedPixelSize_dev;
    unsigned int* connectedPixelIndex_dev;
    connectedPixelSize_dev = (unsigned int*)cms::cuda::allocate_device(dev, nInnerSegments*sizeof(unsigned int), stream);
    connectedPixelIndex_dev = (unsigned int*)cms::cuda::allocate_device(dev, nInnerSegments*sizeof(unsigned int), stream);

    cudaStreamSynchronize(stream);
    int pixelIndexOffsetPos = pixelMapping->connectedPixelsIndex[44999] + pixelMapping->connectedPixelsSizes[44999];
    int pixelIndexOffsetNeg = pixelMapping->connectedPixelsIndexPos[44999] + pixelMapping->connectedPixelsSizes[44999] + pixelIndexOffsetPos;

    // TODO: check if a map/reduction to just eligible pLSs would speed up the kernel
    // the current selection still leaves a significant fraction of unmatchable pLSs
    for (unsigned int i = 0; i < nInnerSegments; i++)
    {// loop over # pLS
        int8_t pixelType = pixelTypes[i];// get pixel type for this pLS
        int superbin = superbins[i]; //get superbin for this pixel
        if((superbin < 0) or (superbin >= 45000) or (pixelType > 2) or (pixelType < 0))
        {
            connectedPixelSize_host[i] = 0;
            connectedPixelIndex_host[i] = 0;
            continue;
        }

        if(pixelType ==0)
        { // used pixel type to select correct size-index arrays
            connectedPixelSize_host[i]  = pixelMapping->connectedPixelsSizes[superbin]; //number of connected modules to this pixel
            auto connectedIdxBase = pixelMapping->connectedPixelsIndex[superbin];
            connectedPixelIndex_host[i] = connectedIdxBase;// index to get start of connected modules for this superbin in map
            // printf("i %d out of nInnerSegments %d type %d superbin %d connectedPixelIndex %d connectedPixelSize %d\n",
            //        i, nInnerSegments, pixelType, superbin, connectedPixelIndex_host[i], connectedPixelSize_host[i]);
        }
        else if(pixelType ==1)
        {
            connectedPixelSize_host[i] = pixelMapping->connectedPixelsSizesPos[superbin]; //number of pixel connected modules
            auto connectedIdxBase = pixelMapping->connectedPixelsIndexPos[superbin]+pixelIndexOffsetPos;
            connectedPixelIndex_host[i] = connectedIdxBase;// index to get start of connected pixel modules
        }
        else if(pixelType ==2)
        {
            connectedPixelSize_host[i] = pixelMapping->connectedPixelsSizesNeg[superbin]; //number of pixel connected modules
            auto connectedIdxBase = pixelMapping->connectedPixelsIndexNeg[superbin] + pixelIndexOffsetNeg;
            connectedPixelIndex_host[i] = connectedIdxBase;// index to get start of connected pixel modules
        }
    }

    cudaMemcpyAsync(connectedPixelSize_dev, connectedPixelSize_host, nInnerSegments*sizeof(unsigned int), cudaMemcpyHostToDevice,stream);
    cudaMemcpyAsync(connectedPixelIndex_dev, connectedPixelIndex_host, nInnerSegments*sizeof(unsigned int), cudaMemcpyHostToDevice,stream);
    cudaStreamSynchronize(stream);

    cms::cuda::free_host(connectedPixelSize_host);
    cms::cuda::free_host(connectedPixelIndex_host);
    cms::cuda::free_host(superbins);
    cms::cuda::free_host(pixelTypes);
    cms::cuda::free_host(nTriplets);

    Vec const threadsPerBlock(static_cast<Idx>(1), static_cast<Idx>(4), static_cast<Idx>(32));
    Vec const blocksPerGrid(static_cast<Idx>(16 /* above median of connected modules*/), static_cast<Idx>(4096), static_cast<Idx>(1));
    WorkDiv const createPixelTripletsInGPUFromMapv2_workDiv(blocksPerGrid, threadsPerBlock, elementsPerThread);

    SDL::createPixelTripletsInGPUFromMapv2 createPixelTripletsInGPUFromMapv2_kernel;
    auto const createPixelTripletsInGPUFromMapv2Task(alpaka::createTaskKernel<Acc>(
        createPixelTripletsInGPUFromMapv2_workDiv,
        createPixelTripletsInGPUFromMapv2_kernel,
        *modulesInGPU,
        *rangesInGPU,
        *mdsInGPU,
        *segmentsInGPU,
        *tripletsInGPU,
        *pixelTripletsInGPU,
        connectedPixelSize_dev,
        connectedPixelIndex_dev,
        nInnerSegments));

    alpaka::enqueue(queue, createPixelTripletsInGPUFromMapv2Task);
    alpaka::wait(queue);

    cms::cuda::free_device(dev, connectedPixelSize_dev);
    cms::cuda::free_device(dev, connectedPixelIndex_dev);


#ifdef Warnings
    int nPixelTriplets;
    cudaMemcpyAsync(&nPixelTriplets, pixelTripletsInGPU->nPixelTriplets,  sizeof(int), cudaMemcpyDeviceToHost,stream);
    cudaStreamSynchronize(stream);
    std::cout<<"number of pixel triplets = "<<nPixelTriplets<<std::endl;
#endif

    //pT3s can be cleaned here because they're not used in making pT5s!
    Vec const threadsPerBlockDupPixTrip(static_cast<Idx>(1), static_cast<Idx>(16), static_cast<Idx>(16));
    //seems like more blocks lead to conflicting writes
    Vec const blocksPerGridDupPixTrip(static_cast<Idx>(1), static_cast<Idx>(40), static_cast<Idx>(1));
    WorkDiv const removeDupPixelTripletsInGPUFromMap_workDiv(blocksPerGridDupPixTrip, threadsPerBlockDupPixTrip, elementsPerThread);

    SDL::removeDupPixelTripletsInGPUFromMap removeDupPixelTripletsInGPUFromMap_kernel;
    auto const removeDupPixelTripletsInGPUFromMapTask(alpaka::createTaskKernel<Acc>(
        removeDupPixelTripletsInGPUFromMap_workDiv,
        removeDupPixelTripletsInGPUFromMap_kernel,
        *pixelTripletsInGPU,
        false));

    alpaka::enqueue(queue, removeDupPixelTripletsInGPUFromMapTask);
    alpaka::wait(queue);
}

void SDL::Event::createQuintuplets()
{
    uint16_t nEligibleT5Modules = 0;

#ifdef CACHE_ALLOC
    rangesInGPU->indicesOfEligibleT5Modules = (uint16_t*)cms::cuda::allocate_device(dev, nLowerModules * sizeof(uint16_t), stream);
#else
    cudaMalloc(&(rangesInGPU->indicesOfEligibleT5Modules), nLowerModules * sizeof(uint16_t));
#endif
    cudaMemsetAsync(rangesInGPU->quintupletModuleIndices, -1, sizeof(int) * (nLowerModules),stream);
    cudaStreamSynchronize(stream);
    unsigned int nTotalQuintuplets;

    Vec const threadsPerBlockCreateQuints(static_cast<Idx>(1), static_cast<Idx>(1), static_cast<Idx>(1024));
    Vec const blocksPerGridCreateQuints(static_cast<Idx>(1), static_cast<Idx>(1), static_cast<Idx>(1));
    WorkDiv const createEligibleModulesListForQuintupletsGPU_workDiv(blocksPerGridCreateQuints, threadsPerBlockCreateQuints, elementsPerThread);

    SDL::createEligibleModulesListForQuintupletsGPU createEligibleModulesListForQuintupletsGPU_kernel;
    auto const createEligibleModulesListForQuintupletsGPUTask(alpaka::createTaskKernel<Acc>(
        createEligibleModulesListForQuintupletsGPU_workDiv,
        createEligibleModulesListForQuintupletsGPU_kernel,
        *modulesInGPU,
        *tripletsInGPU,
        *rangesInGPU));

    alpaka::enqueue(queue, createEligibleModulesListForQuintupletsGPUTask);
    alpaka::wait(queue);

    cudaMemcpyAsync(&nEligibleT5Modules,rangesInGPU->nEligibleT5Modules,sizeof(uint16_t),cudaMemcpyDeviceToHost,stream);
    cudaMemcpyAsync(&nTotalQuintuplets,rangesInGPU->device_nTotalQuints,sizeof(unsigned int),cudaMemcpyDeviceToHost,stream);
    cudaStreamSynchronize(stream);

    if(quintupletsInGPU == nullptr)
    {
        quintupletsInGPU = (SDL::quintuplets*)cms::cuda::allocate_host(sizeof(SDL::quintuplets), stream);
        createQuintupletsInExplicitMemory(*quintupletsInGPU, nTotalQuintuplets, nLowerModules, nEligibleT5Modules,stream);
        cudaMemcpyAsync(quintupletsInGPU->nMemoryLocations, &nTotalQuintuplets, sizeof(unsigned int), cudaMemcpyHostToDevice, stream);
        cudaStreamSynchronize(stream);
    }

    Vec const threadsPerBlockQuints(static_cast<Idx>(1), static_cast<Idx>(8), static_cast<Idx>(32));
    Vec const blocksPerGridQuints(static_cast<Idx>(max(nEligibleT5Modules,1)), static_cast<Idx>(1), static_cast<Idx>(1));
    WorkDiv const createQuintupletsInGPUv2_workDiv(blocksPerGridQuints, threadsPerBlockQuints, elementsPerThread);

    SDL::createQuintupletsInGPUv2 createQuintupletsInGPUv2_kernel;
    auto const createQuintupletsInGPUv2Task(alpaka::createTaskKernel<Acc>(
        createQuintupletsInGPUv2_workDiv,
        createQuintupletsInGPUv2_kernel,
        *modulesInGPU,
        *mdsInGPU,
        *segmentsInGPU,
        *tripletsInGPU,
        *quintupletsInGPU,
        *rangesInGPU,
        nEligibleT5Modules));

    alpaka::enqueue(queue, createQuintupletsInGPUv2Task);
    alpaka::wait(queue);

    Vec const threadsPerBlockDupQuint(static_cast<Idx>(1), static_cast<Idx>(16), static_cast<Idx>(16));
    Vec const blocksPerGridDupQuint(static_cast<Idx>(MAX_BLOCKS), static_cast<Idx>(1), static_cast<Idx>(1));
    WorkDiv const removeDupQuintupletsInGPUAfterBuild_workDiv(blocksPerGridDupQuint, threadsPerBlockDupQuint, elementsPerThread);

    SDL::removeDupQuintupletsInGPUAfterBuild removeDupQuintupletsInGPUAfterBuild_kernel;
    auto const removeDupQuintupletsInGPUAfterBuildTask(alpaka::createTaskKernel<Acc>(
        removeDupQuintupletsInGPUAfterBuild_workDiv,
        removeDupQuintupletsInGPUAfterBuild_kernel,
        *modulesInGPU,
        *quintupletsInGPU,
        *rangesInGPU));

    alpaka::enqueue(queue, removeDupQuintupletsInGPUAfterBuildTask);

    Vec const threadsPerBlockAddQuint(static_cast<Idx>(1), static_cast<Idx>(1), static_cast<Idx>(1024));
    Vec const blocksPerGridAddQuint(static_cast<Idx>(1), static_cast<Idx>(1), static_cast<Idx>(1));
    WorkDiv const addQuintupletRangesToEventExplicit_workDiv(blocksPerGridAddQuint, threadsPerBlockAddQuint, elementsPerThread);

    SDL::addQuintupletRangesToEventExplicit addQuintupletRangesToEventExplicit_kernel;
    auto const addQuintupletRangesToEventExplicitTask(alpaka::createTaskKernel<Acc>(
        addQuintupletRangesToEventExplicit_workDiv,
        addQuintupletRangesToEventExplicit_kernel,
        *modulesInGPU,
        *quintupletsInGPU,
        *rangesInGPU));

    alpaka::enqueue(queue, addQuintupletRangesToEventExplicitTask);
    alpaka::wait(queue);

    if(addObjects)
    {
        addQuintupletsToEventExplicit();
    }
}

void SDL::Event::pixelLineSegmentCleaning()
{
    Vec const threadsPerBlockCheckHitspLS(static_cast<Idx>(1), static_cast<Idx>(16), static_cast<Idx>(16));
    Vec const blocksPerGridCheckHitspLS(static_cast<Idx>(1), static_cast<Idx>(MAX_BLOCKS*4), static_cast<Idx>(MAX_BLOCKS/4));
    WorkDiv const checkHitspLS_workDiv(blocksPerGridCheckHitspLS, threadsPerBlockCheckHitspLS, elementsPerThread);

    SDL::checkHitspLS checkHitspLS_kernel;
    auto const checkHitspLSTask(alpaka::createTaskKernel<Acc>(
        checkHitspLS_workDiv,
        checkHitspLS_kernel,
        *modulesInGPU,
        *segmentsInGPU,
        false));

    alpaka::enqueue(queue, checkHitspLSTask);
    alpaka::wait(queue);
}

void SDL::Event::createPixelQuintuplets()
{
    if(pixelQuintupletsInGPU == nullptr)
    {
        pixelQuintupletsInGPU = (SDL::pixelQuintuplets*)cms::cuda::allocate_host(sizeof(SDL::pixelQuintuplets), stream);
        createPixelQuintupletsInExplicitMemory(*pixelQuintupletsInGPU, N_MAX_PIXEL_QUINTUPLETS,stream);
    }
    if(trackCandidatesInGPU == nullptr)
    {
        trackCandidatesInGPU = (SDL::trackCandidates*)cms::cuda::allocate_host(sizeof(SDL::trackCandidates), stream);
        createTrackCandidatesInExplicitMemory(*trackCandidatesInGPU, N_MAX_TRACK_CANDIDATES + N_MAX_PIXEL_TRACK_CANDIDATES,stream);
    } 

    unsigned int pixelModuleIndex;
    int* superbins;
    int8_t* pixelTypes;
    int *nQuintuplets;

    unsigned int* connectedPixelSize_host;
    unsigned int* connectedPixelIndex_host;
    unsigned int* connectedPixelSize_dev;
    unsigned int* connectedPixelIndex_dev;

    nQuintuplets = (int*)cms::cuda::allocate_host(nLowerModules * sizeof(int), stream);
    cudaMemcpyAsync(nQuintuplets, quintupletsInGPU->nQuintuplets, nLowerModules * sizeof(int), cudaMemcpyDeviceToHost,stream);

    superbins = (int*)cms::cuda::allocate_host(N_MAX_PIXEL_SEGMENTS_PER_MODULE*sizeof(int), stream);
    pixelTypes = (int8_t*)cms::cuda::allocate_host(N_MAX_PIXEL_SEGMENTS_PER_MODULE*sizeof(int8_t), stream);

    cudaMemcpyAsync(superbins,segmentsInGPU->superbin,N_MAX_PIXEL_SEGMENTS_PER_MODULE*sizeof(int),cudaMemcpyDeviceToHost,stream);
    cudaMemcpyAsync(pixelTypes,segmentsInGPU->pixelType,N_MAX_PIXEL_SEGMENTS_PER_MODULE*sizeof(int8_t),cudaMemcpyDeviceToHost,stream);

    cudaStreamSynchronize(stream);
    pixelModuleIndex = nLowerModules;
    unsigned int nInnerSegments = 0;
    cudaMemcpyAsync(&nInnerSegments, &(segmentsInGPU->nSegments[pixelModuleIndex]), sizeof(unsigned int), cudaMemcpyDeviceToHost,stream);
    connectedPixelSize_host = (unsigned int*)cms::cuda::allocate_host(nInnerSegments* sizeof(unsigned int), stream);
    connectedPixelIndex_host = (unsigned int*)cms::cuda::allocate_host(nInnerSegments* sizeof(unsigned int), stream);
    connectedPixelSize_dev = (unsigned int*)cms::cuda::allocate_device(dev,nInnerSegments* sizeof(unsigned int),stream);
    connectedPixelIndex_dev = (unsigned int*)cms::cuda::allocate_device(dev,nInnerSegments* sizeof(unsigned int),stream);
    cudaStreamSynchronize(stream);

    int pixelIndexOffsetPos = pixelMapping->connectedPixelsIndex[44999] + pixelMapping->connectedPixelsSizes[44999];
    int pixelIndexOffsetNeg = pixelMapping->connectedPixelsIndexPos[44999] + pixelMapping->connectedPixelsSizes[44999] + pixelIndexOffsetPos;

    for (unsigned int i = 0; i < nInnerSegments; i++)
    {// loop over # pLS
        int8_t pixelType = pixelTypes[i];// get pixel type for this pLS
        int superbin = superbins[i]; //get superbin for this pixel
        if((superbin < 0) or (superbin >= 45000) or (pixelType > 2) or (pixelType < 0))
        {
            connectedPixelIndex_host[i] = 0;
            connectedPixelSize_host[i] = 0;
            continue;
        }

        if(pixelType ==0)
        { // used pixel type to select correct size-index arrays
            connectedPixelSize_host[i]  = pixelMapping->connectedPixelsSizes[superbin]; //number of connected modules to this pixel
            unsigned int connectedIdxBase = pixelMapping->connectedPixelsIndex[superbin];
            connectedPixelIndex_host[i] = connectedIdxBase;
        }
        else if(pixelType ==1)
        {
            connectedPixelSize_host[i] = pixelMapping->connectedPixelsSizesPos[superbin]; //number of pixel connected modules
            unsigned int connectedIdxBase = pixelMapping->connectedPixelsIndexPos[superbin]+pixelIndexOffsetPos;
            connectedPixelIndex_host[i] = connectedIdxBase;
        }
        else if(pixelType ==2)
        {
            connectedPixelSize_host[i] = pixelMapping->connectedPixelsSizesNeg[superbin]; //number of pixel connected modules
            unsigned int connectedIdxBase = pixelMapping->connectedPixelsIndexNeg[superbin] + pixelIndexOffsetNeg;
            connectedPixelIndex_host[i] = connectedIdxBase;
        }
    }

    cudaMemcpyAsync(connectedPixelSize_dev, connectedPixelSize_host, nInnerSegments*sizeof(unsigned int), cudaMemcpyHostToDevice,stream);
    cudaMemcpyAsync(connectedPixelIndex_dev, connectedPixelIndex_host, nInnerSegments*sizeof(unsigned int), cudaMemcpyHostToDevice,stream);
    cudaStreamSynchronize(stream);

    Vec const threadsPerBlockCreatePixQuints(static_cast<Idx>(1), static_cast<Idx>(16), static_cast<Idx>(16));
    Vec const blocksPerGridCreatePixQuints(static_cast<Idx>(16), static_cast<Idx>(MAX_BLOCKS), static_cast<Idx>(1));
    WorkDiv const createPixelQuintupletsInGPUFromMapv2_workDiv(blocksPerGridCreatePixQuints, threadsPerBlockCreatePixQuints, elementsPerThread);

    SDL::createPixelQuintupletsInGPUFromMapv2 createPixelQuintupletsInGPUFromMapv2_kernel;
    auto const createPixelQuintupletsInGPUFromMapv2Task(alpaka::createTaskKernel<Acc>(
        createPixelQuintupletsInGPUFromMapv2_workDiv,
        createPixelQuintupletsInGPUFromMapv2_kernel,
        *modulesInGPU,
        *mdsInGPU,
        *segmentsInGPU,
        *tripletsInGPU,
        *quintupletsInGPU,
        *pixelQuintupletsInGPU,
        connectedPixelSize_dev,
        connectedPixelIndex_dev,
        nInnerSegments,
        *rangesInGPU));

    alpaka::enqueue(queue, createPixelQuintupletsInGPUFromMapv2Task);
    alpaka::wait(queue);

    cms::cuda::free_host(superbins);
    cms::cuda::free_host(pixelTypes);
    cms::cuda::free_host(nQuintuplets);
    cms::cuda::free_host(connectedPixelSize_host);
    cms::cuda::free_host(connectedPixelIndex_host);
    cms::cuda::free_device(dev, connectedPixelSize_dev);
    cms::cuda::free_device(dev, connectedPixelIndex_dev);

    Vec const threadsPerBlockDupPix(static_cast<Idx>(1), static_cast<Idx>(16), static_cast<Idx>(16));
    Vec const blocksPerGridDupPix(static_cast<Idx>(1), static_cast<Idx>(MAX_BLOCKS), static_cast<Idx>(1));
    WorkDiv const removeDupPixelQuintupletsInGPUFromMap_workDiv(blocksPerGridDupPix, threadsPerBlockDupPix, elementsPerThread);

    SDL::removeDupPixelQuintupletsInGPUFromMap removeDupPixelQuintupletsInGPUFromMap_kernel;
    auto const removeDupPixelQuintupletsInGPUFromMapTask(alpaka::createTaskKernel<Acc>(
        removeDupPixelQuintupletsInGPUFromMap_workDiv,
        removeDupPixelQuintupletsInGPUFromMap_kernel,
        *pixelQuintupletsInGPU,
        false));

    alpaka::enqueue(queue, removeDupPixelQuintupletsInGPUFromMapTask);
    alpaka::wait(queue);

    Vec const threadsPerBlockAddpT5asTrackCan(static_cast<Idx>(1), static_cast<Idx>(1), static_cast<Idx>(256));
    Vec const blocksPerGridAddpT5asTrackCan(static_cast<Idx>(1), static_cast<Idx>(1), static_cast<Idx>(1));
    WorkDiv const addpT5asTrackCandidateInGPU_workDiv(blocksPerGridAddpT5asTrackCan, threadsPerBlockAddpT5asTrackCan, elementsPerThread);

    SDL::addpT5asTrackCandidateInGPU addpT5asTrackCandidateInGPU_kernel;
    auto const addpT5asTrackCandidateInGPUTask(alpaka::createTaskKernel<Acc>(
        addpT5asTrackCandidateInGPU_workDiv,
        addpT5asTrackCandidateInGPU_kernel,
        nLowerModules,
        *pixelQuintupletsInGPU,
        *trackCandidatesInGPU,
        *segmentsInGPU,
        *rangesInGPU));

    alpaka::enqueue(queue, addpT5asTrackCandidateInGPUTask);
    alpaka::wait(queue);
#ifdef Warnings
    int nPixelQuintuplets;
    cudaMemcpyAsync(&nPixelQuintuplets, &(pixelQuintupletsInGPU->nPixelQuintuplets), sizeof(int), cudaMemcpyDeviceToHost,stream);
    cudaStreamSynchronize(stream);
    std::cout<<"number of pixel quintuplets = "<<nPixelQuintuplets<<std::endl;
#endif   
}

void SDL::Event::addQuintupletsToEventExplicit()
{
    unsigned int* nQuintupletsCPU;
    nQuintupletsCPU = (unsigned int*)cms::cuda::allocate_host(nLowerModules * sizeof(unsigned int), stream);
    cudaMemcpyAsync(nQuintupletsCPU,quintupletsInGPU->nQuintuplets,nLowerModules*sizeof(unsigned int),cudaMemcpyDeviceToHost,stream);

    short* module_subdets;
    module_subdets = (short*)cms::cuda::allocate_host(nModules* sizeof(short), stream);
    cudaMemcpyAsync(module_subdets,modulesInGPU->subdets,nModules*sizeof(short),cudaMemcpyDeviceToHost,stream);
    short* module_layers;
    module_layers = (short*)cms::cuda::allocate_host(nLowerModules * sizeof(short), stream);
    cudaMemcpyAsync(module_layers,modulesInGPU->layers,nLowerModules*sizeof(short),cudaMemcpyDeviceToHost,stream);
    int* module_quintupletModuleIndices;
    module_quintupletModuleIndices = (int*)cms::cuda::allocate_host(nLowerModules * sizeof(int), stream);
    cudaMemcpyAsync(module_quintupletModuleIndices, rangesInGPU->quintupletModuleIndices, nLowerModules * sizeof(int), cudaMemcpyDeviceToHost,stream);
    cudaStreamSynchronize(stream);
    for(uint16_t i = 0; i<nLowerModules; i++)
    {
        if(!(nQuintupletsCPU[i] == 0 or module_quintupletModuleIndices[i] == -1))
        {
            if(module_subdets[i] == Barrel)
            {
                n_quintuplets_by_layer_barrel_[module_layers[i] - 1] += nQuintupletsCPU[i];
            }
            else
            {
                n_quintuplets_by_layer_endcap_[module_layers[i] - 1] += nQuintupletsCPU[i];
            }
        }
    }
    cms::cuda::free_host(nQuintupletsCPU);
    cms::cuda::free_host(module_layers);
    cms::cuda::free_host(module_subdets);
    cms::cuda::free_host(module_quintupletModuleIndices);
}

void SDL::Event::addTripletsToEventExplicit()
{
    unsigned int* nTripletsCPU;
    nTripletsCPU = (unsigned int*)cms::cuda::allocate_host(nLowerModules * sizeof(unsigned int), stream);
    cudaMemcpyAsync(nTripletsCPU,tripletsInGPU->nTriplets,nLowerModules*sizeof(unsigned int),cudaMemcpyDeviceToHost,stream);

    short* module_subdets;
    module_subdets = (short*)cms::cuda::allocate_host(nLowerModules* sizeof(short), stream);
    cudaMemcpyAsync(module_subdets,modulesInGPU->subdets,nLowerModules*sizeof(short),cudaMemcpyDeviceToHost,stream);
    short* module_layers;
    module_layers = (short*)cms::cuda::allocate_host(nLowerModules * sizeof(short), stream);
    cudaMemcpyAsync(module_layers,modulesInGPU->layers,nLowerModules*sizeof(short),cudaMemcpyDeviceToHost,stream);

    cudaStreamSynchronize(stream);
    for(uint16_t i = 0; i<nLowerModules; i++)
    {
        if(nTripletsCPU[i] != 0)
        {
            if(module_subdets[i] == Barrel)
            {
                n_triplets_by_layer_barrel_[module_layers[i] - 1] += nTripletsCPU[i];
            }
            else
            {
                n_triplets_by_layer_endcap_[module_layers[i] - 1] += nTripletsCPU[i];
            }
        }
    }

    cms::cuda::free_host(nTripletsCPU);
    cms::cuda::free_host(module_layers);
    cms::cuda::free_host(module_subdets);
}

unsigned int SDL::Event::getNumberOfHits()
{
    unsigned int hits = 0;
    for(auto &it:n_hits_by_layer_barrel_)
    {
        hits += it;
    }
    for(auto& it:n_hits_by_layer_endcap_)
    {
        hits += it;
    }

    return hits;
}

unsigned int SDL::Event::getNumberOfHitsByLayer(unsigned int layer)
{
    if(layer == 6)
        return n_hits_by_layer_barrel_[layer];
    else
        return n_hits_by_layer_barrel_[layer] + n_hits_by_layer_endcap_[layer];
}

unsigned int SDL::Event::getNumberOfHitsByLayerBarrel(unsigned int layer)
{
    return n_hits_by_layer_barrel_[layer];
}

unsigned int SDL::Event::getNumberOfHitsByLayerEndcap(unsigned int layer)
{
    return n_hits_by_layer_endcap_[layer];
}

unsigned int SDL::Event::getNumberOfMiniDoublets()
{
     unsigned int miniDoublets = 0;
    for(auto &it:n_minidoublets_by_layer_barrel_)
    {
        miniDoublets += it;
    }
    for(auto &it:n_minidoublets_by_layer_endcap_)
    {
        miniDoublets += it;
    }

    return miniDoublets;
}

unsigned int SDL::Event::getNumberOfMiniDoubletsByLayer(unsigned int layer)
{
    if(layer == 6)
        return n_minidoublets_by_layer_barrel_[layer];
    else
        return n_minidoublets_by_layer_barrel_[layer] + n_minidoublets_by_layer_endcap_[layer];
}

unsigned int SDL::Event::getNumberOfMiniDoubletsByLayerBarrel(unsigned int layer)
{
    return n_minidoublets_by_layer_barrel_[layer];
}

unsigned int SDL::Event::getNumberOfMiniDoubletsByLayerEndcap(unsigned int layer)
{
    return n_minidoublets_by_layer_endcap_[layer];
}

unsigned int SDL::Event::getNumberOfSegments()
{
    unsigned int segments = 0;
    for(auto &it:n_segments_by_layer_barrel_)
    {
        segments += it;
    }
    for(auto &it:n_segments_by_layer_endcap_)
    {
        segments += it;
    }

    return segments;
}

unsigned int SDL::Event::getNumberOfSegmentsByLayer(unsigned int layer)
{
     if(layer == 6)
        return n_segments_by_layer_barrel_[layer];
    else
        return n_segments_by_layer_barrel_[layer] + n_segments_by_layer_endcap_[layer];
}

unsigned int SDL::Event::getNumberOfSegmentsByLayerBarrel(unsigned int layer)
{
    return n_segments_by_layer_barrel_[layer];
}

unsigned int SDL::Event::getNumberOfSegmentsByLayerEndcap(unsigned int layer)
{
    return n_segments_by_layer_endcap_[layer];
}

unsigned int SDL::Event::getNumberOfTriplets()
{
    unsigned int triplets = 0;
    for(auto &it:n_triplets_by_layer_barrel_)
    {
        triplets += it;
    }
    for(auto &it:n_triplets_by_layer_endcap_)
    {
        triplets += it;
    }

    return triplets;
}

unsigned int SDL::Event::getNumberOfTripletsByLayer(unsigned int layer)
{
    if(layer == 6)
        return n_triplets_by_layer_barrel_[layer];
    else
        return n_triplets_by_layer_barrel_[layer] + n_triplets_by_layer_endcap_[layer];
}

unsigned int SDL::Event::getNumberOfTripletsByLayerBarrel(unsigned int layer)
{
    return n_triplets_by_layer_barrel_[layer];
}

unsigned int SDL::Event::getNumberOfTripletsByLayerEndcap(unsigned int layer)
{
    return n_triplets_by_layer_endcap_[layer];
}

int SDL::Event::getNumberOfPixelTriplets()
{
    int nPixelTriplets;
    cudaMemcpyAsync(&nPixelTriplets, pixelTripletsInGPU->nPixelTriplets, sizeof(int), cudaMemcpyDeviceToHost,stream);
    cudaStreamSynchronize(stream);
    return nPixelTriplets;
}

int SDL::Event::getNumberOfPixelQuintuplets()
{
    int nPixelQuintuplets;
    cudaMemcpyAsync(&nPixelQuintuplets, pixelQuintupletsInGPU->nPixelQuintuplets, sizeof(int), cudaMemcpyDeviceToHost,stream);
    cudaStreamSynchronize(stream);
    return nPixelQuintuplets;
}
unsigned int SDL::Event::getNumberOfQuintuplets()
{
    unsigned int quintuplets = 0;
    for(auto &it:n_quintuplets_by_layer_barrel_)
    {
        quintuplets += it;
    }
    for(auto &it:n_quintuplets_by_layer_endcap_)
    {
        quintuplets += it;
    }

    return quintuplets;
}

unsigned int SDL::Event::getNumberOfQuintupletsByLayer(unsigned int layer)
{
    if(layer == 6)
        return n_quintuplets_by_layer_barrel_[layer];
    else
        return n_quintuplets_by_layer_barrel_[layer] + n_quintuplets_by_layer_endcap_[layer];
}

unsigned int SDL::Event::getNumberOfQuintupletsByLayerBarrel(unsigned int layer)
{
    return n_quintuplets_by_layer_barrel_[layer];
}

unsigned int SDL::Event::getNumberOfQuintupletsByLayerEndcap(unsigned int layer)
{
    return n_quintuplets_by_layer_endcap_[layer];
}

int SDL::Event::getNumberOfTrackCandidates()
{    
    int nTrackCandidates;
    cudaMemcpyAsync(&nTrackCandidates, trackCandidatesInGPU->nTrackCandidates, sizeof(int), cudaMemcpyDeviceToHost,stream);
    cudaStreamSynchronize(stream);

    return nTrackCandidates;
}

int SDL::Event::getNumberOfPT5TrackCandidates()
{
    int nTrackCandidatesPT5;
    cudaMemcpyAsync(&nTrackCandidatesPT5, trackCandidatesInGPU->nTrackCandidatespT5, sizeof(int), cudaMemcpyDeviceToHost,stream);
    cudaStreamSynchronize(stream);

    return nTrackCandidatesPT5;
}

int SDL::Event::getNumberOfPT3TrackCandidates()
{
    int nTrackCandidatesPT3;
    cudaMemcpyAsync(&nTrackCandidatesPT3, trackCandidatesInGPU->nTrackCandidatespT3, sizeof(int), cudaMemcpyDeviceToHost,stream);
    cudaStreamSynchronize(stream);

    return nTrackCandidatesPT3;
}

int SDL::Event::getNumberOfPLSTrackCandidates()
{
    unsigned int nTrackCandidatesPLS;
    cudaMemcpyAsync(&nTrackCandidatesPLS, trackCandidatesInGPU->nTrackCandidatespLS, sizeof(int), cudaMemcpyDeviceToHost,stream);
    cudaStreamSynchronize(stream);

    return nTrackCandidatesPLS;
}

int SDL::Event::getNumberOfPixelTrackCandidates()
{
    int nTrackCandidates;
    int nTrackCandidatesT5;
    cudaMemcpyAsync(&nTrackCandidates, trackCandidatesInGPU->nTrackCandidates, sizeof(int), cudaMemcpyDeviceToHost,stream);
    cudaMemcpyAsync(&nTrackCandidatesT5, trackCandidatesInGPU->nTrackCandidatesT5, sizeof(int), cudaMemcpyDeviceToHost,stream);
    cudaStreamSynchronize(stream);

    return nTrackCandidates - nTrackCandidatesT5;
}

int SDL::Event::getNumberOfT5TrackCandidates()
{
    int nTrackCandidatesT5;
    cudaMemcpyAsync(&nTrackCandidatesT5, trackCandidatesInGPU->nTrackCandidatesT5, sizeof(int), cudaMemcpyDeviceToHost,stream);
    return nTrackCandidatesT5; 
}

SDL::hits* SDL::Event::getHits() //std::shared_ptr should take care of garbage collection
{
    if(hitsInCPU == nullptr)
    {
        hitsInCPU = new SDL::hits;
        hitsInCPU->nHits = new unsigned int;
        unsigned int nHits;
        cudaMemcpyAsync(&nHits, hitsInGPU->nHits, sizeof(unsigned int), cudaMemcpyDeviceToHost,stream);
        cudaStreamSynchronize(stream);
        *(hitsInCPU->nHits) = nHits;
        hitsInCPU->idxs = new unsigned int[nHits];
        hitsInCPU->detid = new unsigned int[nHits];
        hitsInCPU->xs = new float[nHits];
        hitsInCPU->ys = new float[nHits];
        hitsInCPU->zs = new float[nHits];
        hitsInCPU->moduleIndices = new uint16_t[nHits];
        cudaMemcpyAsync(hitsInCPU->idxs, hitsInGPU->idxs,sizeof(unsigned int) * nHits, cudaMemcpyDeviceToHost,stream);
        cudaMemcpyAsync(hitsInCPU->detid, hitsInGPU->detid, sizeof(unsigned int) * nHits, cudaMemcpyDeviceToHost,stream);
        cudaMemcpyAsync(hitsInCPU->xs, hitsInGPU->xs, sizeof(float) * nHits, cudaMemcpyDeviceToHost,stream);
        cudaMemcpyAsync(hitsInCPU->ys, hitsInGPU->ys, sizeof(float) * nHits, cudaMemcpyDeviceToHost,stream);
        cudaMemcpyAsync(hitsInCPU->zs, hitsInGPU->zs, sizeof(float) * nHits, cudaMemcpyDeviceToHost,stream);
        cudaMemcpyAsync(hitsInCPU->moduleIndices, hitsInGPU->moduleIndices, sizeof(uint16_t) * nHits, cudaMemcpyDeviceToHost,stream);
        cudaStreamSynchronize(stream);
    }
    return hitsInCPU;
}

SDL::hits* SDL::Event::getHitsInCMSSW()
{
    if(hitsInCPU == nullptr)
    {
        hitsInCPU = new SDL::hits;
        hitsInCPU->nHits = new unsigned int;
        unsigned int nHits;
        cudaMemcpyAsync(&nHits, hitsInGPU->nHits, sizeof(unsigned int), cudaMemcpyDeviceToHost,stream);
        cudaStreamSynchronize(stream);
        hitsInCPU->idxs = new unsigned int[nHits];
        cudaMemcpyAsync(hitsInCPU->idxs, hitsInGPU->idxs,sizeof(unsigned int) * nHits, cudaMemcpyDeviceToHost,stream);
        cudaStreamSynchronize(stream);
    }
    return hitsInCPU;
}

SDL::objectRanges* SDL::Event::getRanges()
{
    if(rangesInCPU == nullptr)
    {
        rangesInCPU = new SDL::objectRanges;
        rangesInCPU->hitRanges = new int[2*nModules];
        rangesInCPU->quintupletModuleIndices = new int[nLowerModules];
        cudaMemcpyAsync(rangesInCPU->hitRanges, hitsInGPU->hitRanges, 2*nModules * sizeof(int), cudaMemcpyDeviceToHost,stream);
        rangesInCPU->miniDoubletModuleIndices = new int[nLowerModules+1];
        rangesInCPU->segmentModuleIndices = new int[nLowerModules + 1];
        rangesInCPU->tripletModuleIndices = new int[nLowerModules];
        cudaMemcpyAsync(rangesInCPU->quintupletModuleIndices, rangesInGPU->quintupletModuleIndices, nLowerModules * sizeof(int), cudaMemcpyDeviceToHost,stream);
        cudaMemcpyAsync(rangesInCPU->miniDoubletModuleIndices, rangesInGPU->miniDoubletModuleIndices, (nLowerModules + 1) * sizeof(int), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(rangesInCPU->segmentModuleIndices, rangesInGPU->segmentModuleIndices, (nLowerModules + 1) * sizeof(int), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(rangesInCPU->tripletModuleIndices, rangesInGPU->tripletModuleIndices, nLowerModules * sizeof(int), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
    }
    return rangesInCPU;
}

SDL::miniDoublets* SDL::Event::getMiniDoublets()
{
    if(mdsInCPU == nullptr)
    {
        mdsInCPU = new SDL::miniDoublets;
        mdsInCPU->nMDs = new int[nLowerModules+1];

        //compute memory locations
        mdsInCPU->nMemoryLocations = new unsigned int;
        cudaMemcpyAsync(mdsInCPU->nMemoryLocations, mdsInGPU->nMemoryLocations, sizeof(unsigned int), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        mdsInCPU->totOccupancyMDs = new int[nLowerModules+1];

        mdsInCPU->anchorHitIndices = new unsigned int[*(mdsInCPU->nMemoryLocations)];
        mdsInCPU->outerHitIndices = new unsigned int[*(mdsInCPU->nMemoryLocations)];
        mdsInCPU->dphichanges = new float[*(mdsInCPU->nMemoryLocations)];
        cudaMemcpyAsync(mdsInCPU->anchorHitIndices, mdsInGPU->anchorHitIndices, *(mdsInCPU->nMemoryLocations) * sizeof(unsigned int), cudaMemcpyDeviceToHost,stream);
        cudaMemcpyAsync(mdsInCPU->outerHitIndices, mdsInGPU->outerHitIndices, *(mdsInCPU->nMemoryLocations) * sizeof(unsigned int), cudaMemcpyDeviceToHost,stream);
        cudaMemcpyAsync(mdsInCPU->dphichanges, mdsInGPU->dphichanges, *(mdsInCPU->nMemoryLocations) * sizeof(float), cudaMemcpyDeviceToHost,stream);
        cudaMemcpyAsync(mdsInCPU->nMDs, mdsInGPU->nMDs, (nLowerModules+1) * sizeof(unsigned int), cudaMemcpyDeviceToHost,stream);
        cudaMemcpyAsync(mdsInCPU->totOccupancyMDs, mdsInGPU->totOccupancyMDs, (nLowerModules+1) * sizeof(unsigned int), cudaMemcpyDeviceToHost,stream);
        cudaStreamSynchronize(stream);
    }
    return mdsInCPU;
}

SDL::segments<alpaka::DevCpu>* SDL::Event::getSegments()
{
    if(segmentsInCPU == nullptr)
    {
        // Get nMemoryLocations parameter to initilize host based segmentsInCPU
        auto nMemLocal_buf = allocBufWrapper<unsigned int>(devHost, 1);
        alpaka::memcpy(queue, nMemLocal_buf, segmentsInGPU->nMemoryLocations_buf, 1);
        alpaka::wait(queue);

        unsigned int nMemLocal = *alpaka::getPtrNative(nMemLocal_buf);
        segmentsInCPU = new SDL::segments<alpaka::DevCpu>(nMemLocal, nLowerModules, N_MAX_PIXEL_SEGMENTS_PER_MODULE, devHost, queue);

        *alpaka::getPtrNative(segmentsInCPU->nMemoryLocations_buf) = nMemLocal;
        alpaka::memcpy(queue, segmentsInCPU->nSegments_buf, segmentsInGPU->nSegments_buf, (nLowerModules+1));
        alpaka::memcpy(queue, segmentsInCPU->mdIndices_buf, segmentsInGPU->mdIndices_buf, 2 * nMemLocal);
        alpaka::memcpy(queue, segmentsInCPU->innerMiniDoubletAnchorHitIndices_buf, segmentsInGPU->innerMiniDoubletAnchorHitIndices_buf, nMemLocal);
        alpaka::memcpy(queue, segmentsInCPU->outerMiniDoubletAnchorHitIndices_buf, segmentsInGPU->outerMiniDoubletAnchorHitIndices_buf, nMemLocal);
        alpaka::memcpy(queue, segmentsInCPU->totOccupancySegments_buf, segmentsInGPU->totOccupancySegments_buf, (nLowerModules+1));
        alpaka::memcpy(queue, segmentsInCPU->ptIn_buf, segmentsInGPU->ptIn_buf, N_MAX_PIXEL_SEGMENTS_PER_MODULE);
        alpaka::memcpy(queue, segmentsInCPU->eta_buf, segmentsInGPU->eta_buf, N_MAX_PIXEL_SEGMENTS_PER_MODULE);
        alpaka::memcpy(queue, segmentsInCPU->phi_buf, segmentsInGPU->phi_buf, N_MAX_PIXEL_SEGMENTS_PER_MODULE);
        alpaka::memcpy(queue, segmentsInCPU->seedIdx_buf, segmentsInGPU->seedIdx_buf, N_MAX_PIXEL_SEGMENTS_PER_MODULE);
        alpaka::memcpy(queue, segmentsInCPU->isDup_buf, segmentsInGPU->isDup_buf, N_MAX_PIXEL_SEGMENTS_PER_MODULE);
        alpaka::memcpy(queue, segmentsInCPU->isQuad_buf, segmentsInGPU->isQuad_buf, N_MAX_PIXEL_SEGMENTS_PER_MODULE);
        alpaka::memcpy(queue, segmentsInCPU->score_buf, segmentsInGPU->score_buf, N_MAX_PIXEL_SEGMENTS_PER_MODULE);
        alpaka::wait(queue);
    }
    return segmentsInCPU;
}

SDL::triplets* SDL::Event::getTriplets()
{
    if(tripletsInCPU == nullptr)
    {
        tripletsInCPU = new SDL::triplets;
        tripletsInCPU->nMemoryLocations = new unsigned int;
        cudaMemcpyAsync(tripletsInCPU->nMemoryLocations, tripletsInGPU->nMemoryLocations, sizeof(unsigned int), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        tripletsInCPU->segmentIndices = new unsigned[2 * *(tripletsInCPU->nMemoryLocations)];
        tripletsInCPU->nTriplets = new int[nLowerModules];
        tripletsInCPU->betaIn  = new FPX[*(tripletsInCPU->nMemoryLocations)];
        tripletsInCPU->betaOut = new FPX[*(tripletsInCPU->nMemoryLocations)];
        tripletsInCPU->pt_beta = new FPX[*(tripletsInCPU->nMemoryLocations)];
        tripletsInCPU->hitIndices = new unsigned int[6 * *(tripletsInCPU->nMemoryLocations)];
        tripletsInCPU->logicalLayers = new uint8_t[3 * *(tripletsInCPU->nMemoryLocations)];
#ifdef CUT_VALUE_DEBUG
        tripletsInCPU->zOut = new float[4 * *(tripletsInCPU->nMemoryLocations)];
        tripletsInCPU->zLo = new float[*(tripletsInCPU->nMemoryLocations)];
        tripletsInCPU->zHi = new float[*(tripletsInCPU->nMemoryLocations)];
        tripletsInCPU->zLoPointed = new float[*(tripletsInCPU->nMemoryLocations)];
        tripletsInCPU->zHiPointed = new float[*(tripletsInCPU->nMemoryLocations)];
        tripletsInCPU->sdlCut = new float[*(tripletsInCPU->nMemoryLocations)];
        tripletsInCPU->betaInCut = new float[*(tripletsInCPU->nMemoryLocations)];
        tripletsInCPU->betaOutCut = new float[*(tripletsInCPU->nMemoryLocations)];
        tripletsInCPU->deltaBetaCut = new float[*(tripletsInCPU->nMemoryLocations)];
        tripletsInCPU->rtLo = new float[*(tripletsInCPU->nMemoryLocations)];
        tripletsInCPU->rtHi = new float[*(tripletsInCPU->nMemoryLocations)];
        tripletsInCPU->kZ = new float[*(tripletsInCPU->nMemoryLocations)];

        tripletsInCPU->rtOut = tripletsInCPU->zOut + *(tripletsInCPU->nMemoryLocations);
        tripletsInCPU->deltaPhiPos = tripletsInCPU->zOut + 2 * *(tripletsInCPU->nMemoryLocations);
        tripletsInCPU->deltaPhi = tripletsInCPU->zOut + 3 * *(tripletsInCPU->nMemoryLocations);

        cudaMemcpyAsync(tripletsInCPU->zOut, tripletsInGPU->zOut, 4 * * (tripletsInCPU->nMemoryLocations)* sizeof(unsigned int), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(tripletsInCPU->zLo, tripletsInGPU->zLo, * (tripletsInCPU->nMemoryLocations)* sizeof(unsigned int), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(tripletsInCPU->zHi, tripletsInGPU->zHi, * (tripletsInCPU->nMemoryLocations)* sizeof(unsigned int), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(tripletsInCPU->zLoPointed, tripletsInGPU->zLoPointed, 4 * * (tripletsInCPU->nMemoryLocations)* sizeof(unsigned int), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(tripletsInCPU->zHiPointed, tripletsInGPU->zHiPointed, * (tripletsInCPU->nMemoryLocations)* sizeof(unsigned int), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(tripletsInCPU->sdlCut, tripletsInGPU->sdlCut, *(tripletsInCPU->nMemoryLocations)* sizeof(unsigned int), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(tripletsInCPU->betaInCut, tripletsInGPU->betaInCut,  * (tripletsInCPU->nMemoryLocations)* sizeof(unsigned int), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(tripletsInCPU->betaOutCut, tripletsInGPU->betaOutCut,  * (tripletsInCPU->nMemoryLocations)* sizeof(unsigned int), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(tripletsInCPU->deltaBetaCut, tripletsInGPU->deltaBetaCut, *(tripletsInCPU->nMemoryLocations)*sizeof(unsigned int), cudaMemcpyDeviceToHost);
        cudaMemcpyAsync(tripletsInCPU->rtLo, tripletsInGPU->rtLo,  * (tripletsInCPU->nMemoryLocations)* sizeof(unsigned int), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(tripletsInCPU->rtHi, tripletsInGPU->rtHi,  * (tripletsInCPU->nMemoryLocations)* sizeof(unsigned int), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(tripletsInCPU->kZ, tripletsInGPU->kZ,  * (tripletsInCPU->nMemoryLocations) * sizeof(unsigned int), cudaMemcpyDeviceToHost, stream);
#endif
        cudaMemcpyAsync(tripletsInCPU->hitIndices, tripletsInGPU->hitIndices, 6 * *(tripletsInCPU->nMemoryLocations) * sizeof(unsigned int), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(tripletsInCPU->logicalLayers, tripletsInGPU->logicalLayers, 3 * *(tripletsInCPU->nMemoryLocations) * sizeof(uint8_t), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(tripletsInCPU->segmentIndices, tripletsInGPU->segmentIndices, 2 * *(tripletsInCPU->nMemoryLocations) * sizeof(unsigned int), cudaMemcpyDeviceToHost,stream);
        cudaMemcpyAsync(tripletsInCPU->betaIn, tripletsInGPU->betaIn,   *(tripletsInCPU->nMemoryLocations) * sizeof(FPX), cudaMemcpyDeviceToHost,stream);
        cudaMemcpyAsync(tripletsInCPU->betaOut, tripletsInGPU->betaOut, *(tripletsInCPU->nMemoryLocations) * sizeof(FPX), cudaMemcpyDeviceToHost,stream);
        cudaMemcpyAsync(tripletsInCPU->pt_beta, tripletsInGPU->pt_beta, *(tripletsInCPU->nMemoryLocations) * sizeof(FPX), cudaMemcpyDeviceToHost,stream);
        tripletsInCPU->totOccupancyTriplets = new int[nLowerModules];
        cudaMemcpyAsync(tripletsInCPU->nTriplets, tripletsInGPU->nTriplets, nLowerModules * sizeof(unsigned int), cudaMemcpyDeviceToHost,stream);
        cudaMemcpyAsync(tripletsInCPU->totOccupancyTriplets, tripletsInGPU->totOccupancyTriplets, nLowerModules * sizeof(unsigned int), cudaMemcpyDeviceToHost,stream);
        cudaStreamSynchronize(stream);
    }
    return tripletsInCPU;
}

SDL::quintuplets* SDL::Event::getQuintuplets()
{
    if(quintupletsInCPU == nullptr)
    {
        quintupletsInCPU = new SDL::quintuplets;
        uint16_t nEligibleT5Modules;
        cudaMemcpyAsync(&nEligibleT5Modules, rangesInGPU->nEligibleT5Modules, sizeof(uint16_t), cudaMemcpyDeviceToHost,stream);
        cudaStreamSynchronize(stream);
        unsigned int nMemoryLocations;
        cudaMemcpyAsync(&nMemoryLocations, quintupletsInGPU->nMemoryLocations, sizeof(unsigned int), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        quintupletsInCPU->nQuintuplets = new int[nLowerModules];
        quintupletsInCPU->totOccupancyQuintuplets = new int[nLowerModules];
        quintupletsInCPU->tripletIndices = new unsigned int[2 * nMemoryLocations];
        quintupletsInCPU->lowerModuleIndices = new uint16_t[5 * nMemoryLocations];
        quintupletsInCPU->innerRadius = new FPX[nMemoryLocations];
        quintupletsInCPU->outerRadius = new FPX[nMemoryLocations];
        quintupletsInCPU->bridgeRadius = new FPX[nMemoryLocations];

        quintupletsInCPU->isDup = new bool[nMemoryLocations];
        quintupletsInCPU->score_rphisum = new FPX[nMemoryLocations];
        quintupletsInCPU->eta = new FPX[nMemoryLocations];
        quintupletsInCPU->phi = new FPX[nMemoryLocations];

        quintupletsInCPU->chiSquared = new float[nMemoryLocations];
        quintupletsInCPU->nonAnchorChiSquared = new float[nMemoryLocations];
        quintupletsInCPU->rzChiSquared = new float[nMemoryLocations];

        cudaMemcpyAsync(quintupletsInCPU->nQuintuplets, quintupletsInGPU->nQuintuplets,  nLowerModules * sizeof(int), cudaMemcpyDeviceToHost,stream);
        cudaMemcpyAsync(quintupletsInCPU->totOccupancyQuintuplets, quintupletsInGPU->totOccupancyQuintuplets,  nLowerModules * sizeof(int), cudaMemcpyDeviceToHost,stream);
        cudaMemcpyAsync(quintupletsInCPU->tripletIndices, quintupletsInGPU->tripletIndices, 2 * nMemoryLocations * sizeof(unsigned int), cudaMemcpyDeviceToHost,stream);
        cudaMemcpyAsync(quintupletsInCPU->lowerModuleIndices, quintupletsInGPU->lowerModuleIndices, 5 * nMemoryLocations * sizeof(uint16_t), cudaMemcpyDeviceToHost,stream);
        cudaMemcpyAsync(quintupletsInCPU->innerRadius, quintupletsInGPU->innerRadius, nMemoryLocations * sizeof(FPX), cudaMemcpyDeviceToHost,stream);
        cudaMemcpyAsync(quintupletsInCPU->bridgeRadius, quintupletsInGPU->bridgeRadius, nMemoryLocations * sizeof(FPX), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(quintupletsInCPU->outerRadius, quintupletsInGPU->outerRadius, nMemoryLocations * sizeof(FPX), cudaMemcpyDeviceToHost,stream);
        cudaMemcpyAsync(quintupletsInCPU->isDup, quintupletsInGPU->isDup, nMemoryLocations * sizeof(bool), cudaMemcpyDeviceToHost,stream);
        cudaMemcpyAsync(quintupletsInCPU->score_rphisum, quintupletsInGPU->score_rphisum, nMemoryLocations * sizeof(FPX), cudaMemcpyDeviceToHost,stream);
        cudaMemcpyAsync(quintupletsInCPU->eta, quintupletsInGPU->eta, nMemoryLocations * sizeof(FPX), cudaMemcpyDeviceToHost,stream);
        cudaMemcpyAsync(quintupletsInCPU->phi, quintupletsInGPU->phi, nMemoryLocations * sizeof(FPX), cudaMemcpyDeviceToHost,stream);
        cudaMemcpyAsync(quintupletsInCPU->chiSquared, quintupletsInGPU->chiSquared, nMemoryLocations * sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(quintupletsInCPU->rzChiSquared, quintupletsInGPU->rzChiSquared, nMemoryLocations * sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(quintupletsInCPU->nonAnchorChiSquared, quintupletsInGPU->nonAnchorChiSquared, nMemoryLocations * sizeof(float), cudaMemcpyDeviceToHost, stream);

        cudaStreamSynchronize(stream);
    }
    return quintupletsInCPU;
}

SDL::pixelTriplets* SDL::Event::getPixelTriplets()
{
    if(pixelTripletsInCPU == nullptr)
    {
        pixelTripletsInCPU = new SDL::pixelTriplets;

        pixelTripletsInCPU->nPixelTriplets = new int;
        pixelTripletsInCPU->totOccupancyPixelTriplets = new int;
        cudaMemcpyAsync(pixelTripletsInCPU->nPixelTriplets, pixelTripletsInGPU->nPixelTriplets, sizeof(int), cudaMemcpyDeviceToHost,stream);
        cudaMemcpyAsync(pixelTripletsInCPU->totOccupancyPixelTriplets, pixelTripletsInGPU->totOccupancyPixelTriplets, sizeof(int), cudaMemcpyDeviceToHost,stream);
        cudaStreamSynchronize(stream);
        unsigned int nPixelTriplets = *(pixelTripletsInCPU->nPixelTriplets);
        pixelTripletsInCPU->tripletIndices = new unsigned int[nPixelTriplets];
        pixelTripletsInCPU->pixelSegmentIndices = new unsigned int[nPixelTriplets];
        pixelTripletsInCPU->pixelRadius = new FPX[nPixelTriplets];
        pixelTripletsInCPU->tripletRadius = new FPX[nPixelTriplets];
        pixelTripletsInCPU->isDup = new bool[nPixelTriplets];
        pixelTripletsInCPU->eta = new  FPX[nPixelTriplets];
        pixelTripletsInCPU->phi = new  FPX[nPixelTriplets];
        pixelTripletsInCPU->score =new FPX[nPixelTriplets];
        pixelTripletsInCPU->rzChiSquared = new float[nPixelTriplets];
        pixelTripletsInCPU->rPhiChiSquared = new float[nPixelTriplets];
        pixelTripletsInCPU->rPhiChiSquaredInwards = new float[nPixelTriplets];

        cudaMemcpyAsync(pixelTripletsInCPU->rzChiSquared, pixelTripletsInGPU->rzChiSquared, nPixelTriplets * sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(pixelTripletsInCPU->rPhiChiSquared, pixelTripletsInGPU->rPhiChiSquared, nPixelTriplets * sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(pixelTripletsInCPU->rPhiChiSquaredInwards, pixelTripletsInGPU->rPhiChiSquaredInwards, nPixelTriplets * sizeof(float), cudaMemcpyDeviceToHost, stream);

        cudaMemcpyAsync(pixelTripletsInCPU->tripletIndices, pixelTripletsInGPU->tripletIndices, nPixelTriplets * sizeof(unsigned int), cudaMemcpyDeviceToHost,stream);
        cudaMemcpyAsync(pixelTripletsInCPU->pixelSegmentIndices, pixelTripletsInGPU->pixelSegmentIndices, nPixelTriplets * sizeof(unsigned int), cudaMemcpyDeviceToHost,stream);
        cudaMemcpyAsync(pixelTripletsInCPU->pixelRadius, pixelTripletsInGPU->pixelRadius, nPixelTriplets * sizeof(FPX), cudaMemcpyDeviceToHost,stream);
        cudaMemcpyAsync(pixelTripletsInCPU->tripletRadius, pixelTripletsInGPU->tripletRadius, nPixelTriplets * sizeof(FPX), cudaMemcpyDeviceToHost,stream);
        cudaMemcpyAsync(pixelTripletsInCPU->isDup, pixelTripletsInGPU->isDup, nPixelTriplets * sizeof(bool), cudaMemcpyDeviceToHost,stream);
        cudaMemcpyAsync(pixelTripletsInCPU->eta, pixelTripletsInGPU->eta, nPixelTriplets * sizeof(FPX), cudaMemcpyDeviceToHost,stream);
        cudaMemcpyAsync(pixelTripletsInCPU->phi, pixelTripletsInGPU->phi, nPixelTriplets * sizeof(FPX), cudaMemcpyDeviceToHost,stream);
        cudaMemcpyAsync(pixelTripletsInCPU->score, pixelTripletsInGPU->score, nPixelTriplets * sizeof(FPX), cudaMemcpyDeviceToHost,stream);
        cudaStreamSynchronize(stream);
    }
    return pixelTripletsInCPU;
}

SDL::pixelQuintuplets* SDL::Event::getPixelQuintuplets()
{
    if(pixelQuintupletsInCPU == nullptr)
    {
        pixelQuintupletsInCPU = new SDL::pixelQuintuplets;

        pixelQuintupletsInCPU->nPixelQuintuplets = new int;
        pixelQuintupletsInCPU->totOccupancyPixelQuintuplets = new int;
        cudaMemcpyAsync(pixelQuintupletsInCPU->nPixelQuintuplets, pixelQuintupletsInGPU->nPixelQuintuplets, sizeof(int), cudaMemcpyDeviceToHost,stream);
        cudaMemcpyAsync(pixelQuintupletsInCPU->totOccupancyPixelQuintuplets, pixelQuintupletsInGPU->totOccupancyPixelQuintuplets, sizeof(int), cudaMemcpyDeviceToHost,stream);
        cudaStreamSynchronize(stream);
        int nPixelQuintuplets = *(pixelQuintupletsInCPU->nPixelQuintuplets);

        pixelQuintupletsInCPU->pixelIndices = new unsigned int[nPixelQuintuplets];
        pixelQuintupletsInCPU->T5Indices = new unsigned int[nPixelQuintuplets];
        pixelQuintupletsInCPU->isDup = new bool[nPixelQuintuplets];
        pixelQuintupletsInCPU->score = new FPX[nPixelQuintuplets];
        pixelQuintupletsInCPU->rzChiSquared = new float[nPixelQuintuplets];
        pixelQuintupletsInCPU->rPhiChiSquared = new float[nPixelQuintuplets];
        pixelQuintupletsInCPU->rPhiChiSquaredInwards = new float[nPixelQuintuplets];

        cudaMemcpyAsync(pixelQuintupletsInCPU->rzChiSquared, pixelQuintupletsInGPU->rzChiSquared, nPixelQuintuplets * sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(pixelQuintupletsInCPU->rPhiChiSquared, pixelQuintupletsInGPU->rPhiChiSquared, nPixelQuintuplets * sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(pixelQuintupletsInCPU->rPhiChiSquaredInwards, pixelQuintupletsInGPU->rPhiChiSquaredInwards, nPixelQuintuplets * sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(pixelQuintupletsInCPU->pixelIndices, pixelQuintupletsInGPU->pixelIndices, nPixelQuintuplets * sizeof(unsigned int), cudaMemcpyDeviceToHost,stream);
        cudaMemcpyAsync(pixelQuintupletsInCPU->T5Indices, pixelQuintupletsInGPU->T5Indices, nPixelQuintuplets * sizeof(unsigned int), cudaMemcpyDeviceToHost,stream);
        cudaMemcpyAsync(pixelQuintupletsInCPU->isDup, pixelQuintupletsInGPU->isDup, nPixelQuintuplets * sizeof(bool), cudaMemcpyDeviceToHost,stream);
        cudaMemcpyAsync(pixelQuintupletsInCPU->score, pixelQuintupletsInGPU->score, nPixelQuintuplets * sizeof(FPX), cudaMemcpyDeviceToHost,stream);
        cudaStreamSynchronize(stream);
    }
    return pixelQuintupletsInCPU;
}

SDL::trackCandidates* SDL::Event::getTrackCandidates()
{
    if(trackCandidatesInCPU == nullptr)
    {
        trackCandidatesInCPU = new SDL::trackCandidates;
        trackCandidatesInCPU->nTrackCandidates = new int;
        cudaMemcpyAsync(trackCandidatesInCPU->nTrackCandidates, trackCandidatesInGPU->nTrackCandidates, sizeof(int), cudaMemcpyDeviceToHost,stream);
        cudaStreamSynchronize(stream);
        int nTrackCandidates = *(trackCandidatesInCPU->nTrackCandidates);

        trackCandidatesInCPU->directObjectIndices = new unsigned int[nTrackCandidates];
        trackCandidatesInCPU->objectIndices = new unsigned int[2 * nTrackCandidates];
        trackCandidatesInCPU->trackCandidateType = new short[nTrackCandidates];
        trackCandidatesInCPU->hitIndices = new unsigned int[14 * nTrackCandidates];
        trackCandidatesInCPU->pixelSeedIndex = new int[nTrackCandidates];
        trackCandidatesInCPU->logicalLayers = new uint8_t[7 * nTrackCandidates];

        cudaMemcpyAsync(trackCandidatesInCPU->hitIndices, trackCandidatesInGPU->hitIndices, 14 * nTrackCandidates * sizeof(unsigned int), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(trackCandidatesInCPU->pixelSeedIndex, trackCandidatesInGPU->pixelSeedIndex, nTrackCandidates * sizeof(int), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(trackCandidatesInCPU->logicalLayers, trackCandidatesInGPU->logicalLayers, 7 * nTrackCandidates * sizeof(uint8_t), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(trackCandidatesInCPU->directObjectIndices, trackCandidatesInGPU->directObjectIndices, nTrackCandidates * sizeof(unsigned int), cudaMemcpyDeviceToHost,stream);                                                                                    
        cudaMemcpyAsync(trackCandidatesInCPU->objectIndices, trackCandidatesInGPU->objectIndices, 2 * nTrackCandidates * sizeof(unsigned int), cudaMemcpyDeviceToHost,stream);                                                                                    
        cudaMemcpyAsync(trackCandidatesInCPU->trackCandidateType, trackCandidatesInGPU->trackCandidateType, nTrackCandidates * sizeof(short), cudaMemcpyDeviceToHost,stream);                                                                                                                
        cudaStreamSynchronize(stream);
    }
    return trackCandidatesInCPU;
}

SDL::trackCandidates* SDL::Event::getTrackCandidatesInCMSSW()
{
    if(trackCandidatesInCPU == nullptr)
    {
        trackCandidatesInCPU = new SDL::trackCandidates;
        trackCandidatesInCPU->nTrackCandidates = new int;
        cudaMemcpyAsync(trackCandidatesInCPU->nTrackCandidates, trackCandidatesInGPU->nTrackCandidates, sizeof(unsigned int), cudaMemcpyDeviceToHost,stream);
        cudaStreamSynchronize(stream);
        unsigned int nTrackCandidates = *(trackCandidatesInCPU->nTrackCandidates);

        trackCandidatesInCPU->trackCandidateType = new short[nTrackCandidates];
        trackCandidatesInCPU->hitIndices = new unsigned int[14 * nTrackCandidates];
        trackCandidatesInCPU->pixelSeedIndex = new int[nTrackCandidates];

        cudaMemcpyAsync(trackCandidatesInCPU->hitIndices, trackCandidatesInGPU->hitIndices, 14 * nTrackCandidates * sizeof(unsigned int), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(trackCandidatesInCPU->pixelSeedIndex, trackCandidatesInGPU->pixelSeedIndex, nTrackCandidates * sizeof(int), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(trackCandidatesInCPU->trackCandidateType, trackCandidatesInGPU->trackCandidateType, nTrackCandidates * sizeof(short), cudaMemcpyDeviceToHost,stream);
        cudaStreamSynchronize(stream);
    }
    return trackCandidatesInCPU;
}

SDL::modules* SDL::Event::getFullModules()
{
    if(modulesInCPUFull == nullptr)
    {
        modulesInCPUFull = new SDL::modules;

        modulesInCPUFull->detIds = new unsigned int[nModules];
        modulesInCPUFull->moduleMap = new uint16_t[40*nModules];
        modulesInCPUFull->nConnectedModules = new uint16_t[nModules];
        modulesInCPUFull->drdzs = new float[nModules];
        modulesInCPUFull->slopes = new float[nModules];
        modulesInCPUFull->nModules = new uint16_t[1];
        modulesInCPUFull->nLowerModules = new uint16_t[1];
        modulesInCPUFull->layers = new short[nModules];
        modulesInCPUFull->rings = new short[nModules];
        modulesInCPUFull->modules = new short[nModules];
        modulesInCPUFull->rods = new short[nModules];
        modulesInCPUFull->subdets = new short[nModules];
        modulesInCPUFull->sides = new short[nModules];
        modulesInCPUFull->isInverted = new bool[nModules];
        modulesInCPUFull->isLower = new bool[nModules];

        modulesInCPUFull->moduleType = new ModuleType[nModules];
        modulesInCPUFull->moduleLayerType = new ModuleLayerType[nModules];
        cudaMemcpyAsync(modulesInCPUFull->detIds,modulesInGPU->detIds,nModules*sizeof(unsigned int),cudaMemcpyDeviceToHost,stream);
        cudaMemcpyAsync(modulesInCPUFull->moduleMap,modulesInGPU->moduleMap,40*nModules*sizeof(unsigned int),cudaMemcpyDeviceToHost,stream);
        cudaMemcpyAsync(modulesInCPUFull->nConnectedModules,modulesInGPU->nConnectedModules,nModules*sizeof(unsigned int),cudaMemcpyDeviceToHost,stream);
        cudaMemcpyAsync(modulesInCPUFull->drdzs,modulesInGPU->drdzs,sizeof(float)*nModules,cudaMemcpyDeviceToHost,stream);
        cudaMemcpyAsync(modulesInCPUFull->slopes,modulesInGPU->slopes,sizeof(float)*nModules,cudaMemcpyDeviceToHost,stream);
        cudaMemcpyAsync(modulesInCPUFull->nLowerModules,modulesInGPU->nLowerModules,sizeof(unsigned int),cudaMemcpyDeviceToHost,stream);
        cudaMemcpyAsync(modulesInCPUFull->layers,modulesInGPU->layers,nModules*sizeof(short),cudaMemcpyDeviceToHost,stream);
        cudaMemcpyAsync(modulesInCPUFull->rings,modulesInGPU->rings,sizeof(short)*nModules,cudaMemcpyDeviceToHost,stream);
        cudaMemcpyAsync(modulesInCPUFull->modules,modulesInGPU->modules,sizeof(short)*nModules,cudaMemcpyDeviceToHost,stream);
        cudaMemcpyAsync(modulesInCPUFull->rods,modulesInGPU->rods,sizeof(short)*nModules,cudaMemcpyDeviceToHost,stream);
        cudaMemcpyAsync(modulesInCPUFull->subdets,modulesInGPU->subdets,sizeof(short)*nModules,cudaMemcpyDeviceToHost,stream);
        cudaMemcpyAsync(modulesInCPUFull->sides,modulesInGPU->sides,sizeof(short)*nModules,cudaMemcpyDeviceToHost,stream);
        cudaMemcpyAsync(modulesInCPUFull->isInverted,modulesInGPU->isInverted,sizeof(bool)*nModules,cudaMemcpyDeviceToHost,stream);
        cudaMemcpyAsync(modulesInCPUFull->isLower,modulesInGPU->isLower,sizeof(bool)*nModules,cudaMemcpyDeviceToHost,stream);
        cudaMemcpyAsync(modulesInCPUFull->moduleType,modulesInGPU->moduleType,sizeof(ModuleType)*nModules,cudaMemcpyDeviceToHost,stream);
        cudaMemcpyAsync(modulesInCPUFull->moduleLayerType,modulesInGPU->moduleLayerType,sizeof(ModuleLayerType)*nModules,cudaMemcpyDeviceToHost,stream);
        cudaStreamSynchronize(stream);
    }
    return modulesInCPUFull;
}

SDL::modules* SDL::Event::getModules()
{
    if(modulesInCPU == nullptr)
    {
        modulesInCPU = new SDL::modules;
        modulesInCPU->nLowerModules = new uint16_t[1];
        modulesInCPU->nModules = new uint16_t[1];
        modulesInCPU->detIds = new unsigned int[nModules];
        modulesInCPU->isLower = new bool[nModules];
        modulesInCPU->layers = new short[nModules];
        modulesInCPU->subdets = new short[nModules];
        modulesInCPU->rings = new short[nModules];
        modulesInCPU->rods = new short[nModules];
        modulesInCPU->modules = new short[nModules];
        modulesInCPU->sides = new short[nModules];
        modulesInCPU->eta = new float[nModules];
        modulesInCPU->r = new float[nModules];
        modulesInCPU->moduleType = new ModuleType[nModules];

        cudaMemcpyAsync(modulesInCPU->nLowerModules, modulesInGPU->nLowerModules, sizeof(uint16_t), cudaMemcpyDeviceToHost,stream);
        cudaMemcpyAsync(modulesInCPU->nModules, modulesInGPU->nModules, sizeof(uint16_t), cudaMemcpyDeviceToHost,stream);
        cudaMemcpyAsync(modulesInCPU->detIds, modulesInGPU->detIds, nModules * sizeof(unsigned int), cudaMemcpyDeviceToHost,stream);
        cudaMemcpyAsync(modulesInCPU->isLower, modulesInGPU->isLower, nModules * sizeof(bool), cudaMemcpyDeviceToHost,stream);
        cudaMemcpyAsync(modulesInCPU->layers, modulesInGPU->layers, nModules * sizeof(short), cudaMemcpyDeviceToHost,stream);
        cudaMemcpyAsync(modulesInCPU->subdets, modulesInGPU->subdets, nModules * sizeof(short), cudaMemcpyDeviceToHost,stream);
        cudaMemcpyAsync(modulesInCPU->rings, modulesInGPU->rings, nModules * sizeof(short), cudaMemcpyDeviceToHost,stream);
        cudaMemcpyAsync(modulesInCPU->rods, modulesInGPU->rods, nModules * sizeof(short), cudaMemcpyDeviceToHost,stream);
        cudaMemcpyAsync(modulesInCPU->modules, modulesInGPU->modules, nModules * sizeof(short), cudaMemcpyDeviceToHost,stream);
        cudaMemcpyAsync(modulesInCPU->sides, modulesInGPU->sides, nModules * sizeof(short), cudaMemcpyDeviceToHost,stream);
        cudaMemcpyAsync(modulesInCPU->eta, modulesInGPU->eta, nModules * sizeof(short), cudaMemcpyDeviceToHost,stream);
        cudaMemcpyAsync(modulesInCPU->r, modulesInGPU->r, nModules * sizeof(short), cudaMemcpyDeviceToHost,stream);
        cudaMemcpyAsync(modulesInCPU->moduleType, modulesInGPU->moduleType, nModules * sizeof(ModuleType), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
    }
    return modulesInCPU;
}

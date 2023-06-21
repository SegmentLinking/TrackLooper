#include "Event.cuh"
#include "allocate.h"

struct SDL::modules* SDL::modulesInGPU = nullptr;
struct SDL::pixelMap* SDL::pixelMapping = nullptr;
uint16_t SDL::nModules;
uint16_t SDL::nLowerModules;

SDL::Event::Event(cudaStream_t estream,bool verbose)
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
    //resetObjectsInModule();
}

SDL::Event::~Event()
{
#ifdef CACHE_ALLOC
    if(rangesInGPU){rangesInGPU->freeMemoryCache();}
    if(hitsInGPU){hitsInGPU->freeMemoryCache();}
    if(mdsInGPU){mdsInGPU->freeMemoryCache();}
    if(segmentsInGPU){segmentsInGPU->freeMemoryCache();}
    if(tripletsInGPU){tripletsInGPU->freeMemoryCache();}
    if(quintupletsInGPU){quintupletsInGPU->freeMemoryCache();}
    if(pixelQuintupletsInGPU){pixelQuintupletsInGPU->freeMemoryCache();}
    if(pixelTripletsInGPU){pixelTripletsInGPU->freeMemoryCache();}
    if(trackCandidatesInGPU){trackCandidatesInGPU->freeMemoryCache();}
#else

    if(rangesInGPU){rangesInGPU->freeMemory();}
    if(hitsInGPU){hitsInGPU->freeMemory();}
    if(mdsInGPU){mdsInGPU->freeMemory(stream);}
    if(segmentsInGPU){segmentsInGPU->freeMemory(stream);}
    if(tripletsInGPU){tripletsInGPU->freeMemory(stream);}
    if(quintupletsInGPU){quintupletsInGPU->freeMemory(stream);}
    if(pixelQuintupletsInGPU){pixelQuintupletsInGPU->freeMemory(stream);}
    if(pixelTripletsInGPU){pixelTripletsInGPU->freeMemory(stream);}
    if(trackCandidatesInGPU){trackCandidatesInGPU->freeMemory(stream);}
#endif
    if(rangesInGPU != nullptr){cms::cuda::free_host(rangesInGPU);}
    if(mdsInGPU != nullptr){cms::cuda::free_host(mdsInGPU);}
    if(segmentsInGPU!= nullptr){cms::cuda::free_host(segmentsInGPU);}
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
        delete[] segmentsInCPU->mdIndices;
        delete[] segmentsInCPU->nSegments;
        delete[] segmentsInCPU->totOccupancySegments;
        delete[] segmentsInCPU->innerMiniDoubletAnchorHitIndices;
        delete[] segmentsInCPU->outerMiniDoubletAnchorHitIndices;
        delete[] segmentsInCPU->ptIn;
        delete[] segmentsInCPU->eta;
        delete[] segmentsInCPU->phi;
        delete segmentsInCPU->nMemoryLocations;
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
    if(segmentsInGPU){segmentsInGPU->freeMemoryCache();}
    if(tripletsInGPU){tripletsInGPU->freeMemoryCache();}
    if(pixelQuintupletsInGPU){pixelQuintupletsInGPU->freeMemoryCache();}
    if(pixelTripletsInGPU){pixelTripletsInGPU->freeMemoryCache();}
    if(trackCandidatesInGPU){trackCandidatesInGPU->freeMemoryCache();}

#else
    if(hitsInGPU){hitsInGPU->freeMemory();}
    if(quintupletsInGPU){quintupletsInGPU->freeMemory(stream);}
    if(rangesInGPU){rangesInGPU->freeMemory();}
    if(mdsInGPU){mdsInGPU->freeMemory(stream);}
    if(segmentsInGPU){segmentsInGPU->freeMemory(stream);}
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
    if(segmentsInGPU){cms::cuda::free_host(segmentsInGPU);
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
        delete[] segmentsInCPU->mdIndices;
        delete[] segmentsInCPU->nSegments;
        delete[] segmentsInCPU->totOccupancySegments;
        delete[] segmentsInCPU->innerMiniDoubletAnchorHitIndices;
        delete[] segmentsInCPU->outerMiniDoubletAnchorHitIndices;
        delete[] segmentsInCPU->ptIn;
        delete[] segmentsInCPU->eta;
        delete[] segmentsInCPU->phi;
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

__device__ int binary_search(
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

__global__ void moduleRangesKernel(uint16_t nLower, struct SDL::modules *modulesInGPU, struct SDL::hits *hitsInGPU)
{
    for ( int lowerIndex = blockIdx.x * blockDim.x + threadIdx.x; lowerIndex < nLower; lowerIndex += blockDim.x*gridDim.x )
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

__global__ void hitLoopKernel(
                            uint16_t Endcap, // Integer corresponding to endcap in module subdets
                            uint16_t TwoS, // Integer corresponding to TwoS in moduleType
                            int nHits, // Total number of hits in event
                            unsigned int nModules, // Number of modules
                            unsigned int nEndCapMap, // Number of elements in endcap map
                            unsigned int* geoMapDetId, // DetId's from endcap map
                            float* geoMapPhi, // Phi values from endcap map
                            struct SDL::modules *modulesInGPU,
                            struct SDL::hits *hitsInGPU)
{
    for( int ihit = blockIdx.x * blockDim.x + threadIdx.x; ihit < nHits; ihit += blockDim.x * gridDim.x )
    {
        float ihit_x = hitsInGPU->xs[ihit];
        float ihit_y = hitsInGPU->ys[ihit];
        float ihit_z = hitsInGPU->zs[ihit];
        int iDetId = hitsInGPU->detid[ihit];

        hitsInGPU->rts[ihit] = sqrt(ihit_x*ihit_x + ihit_y*ihit_y);
        hitsInGPU->phis[ihit] = SDL::phi(ihit_x,ihit_y);
        hitsInGPU->etas[ihit] = ((ihit_z>0)-(ihit_z<0))*acosh(sqrt(ihit_x*ihit_x+ihit_y*ihit_y+ihit_z*ihit_z)/hitsInGPU->rts[ihit]);

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
            float cos_phi = cosf(phi);
            hitsInGPU->highEdgeXs[ihit] = ihit_x + 2.5f * cos_phi;
            hitsInGPU->lowEdgeXs[ihit] = ihit_x - 2.5f * cos_phi;
            float sin_phi = sinf(phi);
            hitsInGPU->highEdgeYs[ihit] = ihit_y + 2.5f * sin_phi;
            hitsInGPU->lowEdgeYs[ihit] = ihit_y - 2.5f * sin_phi;
        }
        // Need to set initial value if index hasn't been seen before.
        int old = atomicCAS(&(hitsInGPU->hitRanges[lastModuleIndex * 2]), -1, ihit);
        // For subsequent visits, stores the min value.
        if (old != -1)
            atomicMin(&hitsInGPU->hitRanges[lastModuleIndex * 2], ihit);

        atomicMax(&hitsInGPU->hitRanges[lastModuleIndex * 2 + 1], ihit);
    }
}

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
    // Calculate secondary variables on the GPU.
    hitLoopKernel<<<MAX_BLOCKS,256,0,stream>>>(
                                            Endcap,
                                            TwoS,
                                            nHits,
                                            nModules,
                                            SDL::endcapGeometry.nEndCapMap,
                                            SDL::endcapGeometry.geoMapDetId,
                                            SDL::endcapGeometry.geoMapPhi,
                                            modulesInGPU,
                                            hitsInGPU);
    //std::cout << cudaGetLastError() << std::endl;
    cudaStreamSynchronize(stream);

    // No stream synchronize needed after second kernel call. Saves ~100 us.
    // This is because addPixelSegmentToEvent (which is run next) doesn't rely on hitsinGPU->hitrange variables.
    // Also, modulesInGPU->partnerModuleIndices is not alterned in addPixelSegmentToEvent.
    moduleRangesKernel<<<MAX_BLOCKS,256,0,stream>>>(nLowerModules, modulesInGPU, hitsInGPU);
    //std::cout << cudaGetLastError() << std::endl;
}

__global__ void addPixelSegmentToEventKernel(unsigned int* hitIndices0,unsigned int* hitIndices1,unsigned int* hitIndices2,unsigned int* hitIndices3, float* dPhiChange, float* ptIn, float* ptErr, float* px, float* py, float* pz, float* eta, float* etaErr,float* phi, int* charge, unsigned int* seedIdx, uint16_t pixelModuleIndex, struct SDL::modules& modulesInGPU, struct SDL::objectRanges& rangesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU,const int size, int* superbin, int8_t* pixelType, short* isQuad)
{
    for( int tid = blockIdx.x * blockDim.x + threadIdx.x; tid < size; tid += blockDim.x*gridDim.x)
    {

      unsigned int innerMDIndex = rangesInGPU.miniDoubletModuleIndices[pixelModuleIndex] + 2*(tid);
      unsigned int outerMDIndex = rangesInGPU.miniDoubletModuleIndices[pixelModuleIndex] + 2*(tid) +1;
      unsigned int pixelSegmentIndex = rangesInGPU.segmentModuleIndices[pixelModuleIndex] + tid;

      addMDToMemory(mdsInGPU, hitsInGPU, modulesInGPU, hitIndices0[tid], hitIndices1[tid], pixelModuleIndex, 0,0,0,0,0,0,0,0,0,innerMDIndex);
      addMDToMemory(mdsInGPU, hitsInGPU, modulesInGPU, hitIndices2[tid], hitIndices3[tid], pixelModuleIndex, 0,0,0,0,0,0,0,0,0,outerMDIndex);

    //in outer hits - pt, eta, phi
    float slope = sinhf(hitsInGPU.ys[mdsInGPU.outerHitIndices[innerMDIndex]]);
    float intercept = hitsInGPU.zs[mdsInGPU.anchorHitIndices[innerMDIndex]] - slope * hitsInGPU.rts[mdsInGPU.anchorHitIndices[innerMDIndex]];
    float score_lsq=(hitsInGPU.rts[mdsInGPU.anchorHitIndices[outerMDIndex]] * slope + intercept) - (hitsInGPU.zs[mdsInGPU.anchorHitIndices[outerMDIndex]]);
    score_lsq = score_lsq * score_lsq;

    unsigned int hits1[4];
    hits1[0] = hitsInGPU.idxs[mdsInGPU.anchorHitIndices[innerMDIndex]];
    hits1[1] = hitsInGPU.idxs[mdsInGPU.anchorHitIndices[outerMDIndex]];
    hits1[2] = hitsInGPU.idxs[mdsInGPU.outerHitIndices[innerMDIndex]];
    hits1[3] = hitsInGPU.idxs[mdsInGPU.outerHitIndices[outerMDIndex]];
    addPixelSegmentToMemory(segmentsInGPU, mdsInGPU, modulesInGPU, innerMDIndex, outerMDIndex, pixelModuleIndex, hits1, hitIndices0[tid], hitIndices2[tid], dPhiChange[tid], ptIn[tid], ptErr[tid], px[tid], py[tid], pz[tid], etaErr[tid], eta[tid], phi[tid], charge[tid], seedIdx[tid], pixelSegmentIndex, tid, superbin[tid], pixelType[tid],isQuad[tid],score_lsq);
    }
}
void SDL::Event::addPixelSegmentToEvent(std::vector<unsigned int> hitIndices0,std::vector<unsigned int> hitIndices1,std::vector<unsigned int> hitIndices2,std::vector<unsigned int> hitIndices3, std::vector<float> dPhiChange, std::vector<float> ptIn, std::vector<float> ptErr, std::vector<float> px, std::vector<float> py, std::vector<float> pz, std::vector<float> eta, std::vector<float> etaErr, std::vector<float> phi, std::vector<int> charge, std::vector<unsigned int> seedIdx, std::vector<int> superbin, std::vector<int8_t> pixelType, std::vector<short> isQuad)
{
    if(mdsInGPU == nullptr)
    {
        mdsInGPU = (SDL::miniDoublets*)cms::cuda::allocate_host(sizeof(SDL::miniDoublets), stream);
        unsigned int nTotalMDs;
        cudaMemsetAsync(&rangesInGPU->miniDoubletModuleOccupancy[nLowerModules],N_MAX_PIXEL_MD_PER_MODULES, sizeof(unsigned int),stream);
        createMDArrayRangesGPU<<<1,1024,0,stream>>>(*modulesInGPU, *rangesInGPU);
        cudaMemcpyAsync(&nTotalMDs,rangesInGPU->device_nTotalMDs,sizeof(unsigned int),cudaMemcpyDeviceToHost,stream);
        cudaStreamSynchronize(stream);
        nTotalMDs+= N_MAX_PIXEL_MD_PER_MODULES;
        createMDsInExplicitMemory(*mdsInGPU, nTotalMDs, nLowerModules, N_MAX_PIXEL_MD_PER_MODULES,stream);
        cudaMemcpyAsync(mdsInGPU->nMemoryLocations, &nTotalMDs, sizeof(unsigned int), cudaMemcpyHostToDevice, stream);
        cudaStreamSynchronize(stream);

    }
    if(segmentsInGPU == nullptr)
    {
        segmentsInGPU = (SDL::segments*)cms::cuda::allocate_host(sizeof(SDL::segments), stream);
        //hardcoded range numbers for this will come from studies!
        // can be optimized here: because we didn't distinguish pixel segments and outer-tracker segments and call them both "segments", so they use the index continuously.
        // If we want to further study the memory footprint in detail, we can separate the two and allocate different memories to them
        createSegmentArrayRanges<<<1,1024,0,stream>>>(*modulesInGPU, *rangesInGPU, *mdsInGPU);
        cudaMemcpyAsync(&nTotalSegments,rangesInGPU->device_nTotalSegs,sizeof(unsigned int),cudaMemcpyDeviceToHost,stream);
        cudaStreamSynchronize(stream);
        nTotalSegments += N_MAX_PIXEL_SEGMENTS_PER_MODULE;
        createSegmentsInExplicitMemory(*segmentsInGPU, nTotalSegments, nLowerModules, N_MAX_PIXEL_SEGMENTS_PER_MODULE,stream);

        cudaMemcpyAsync(segmentsInGPU->nMemoryLocations, &nTotalSegments, sizeof(unsigned int), cudaMemcpyHostToDevice, stream);;
        cudaStreamSynchronize(stream);

    }
    cudaStreamSynchronize(stream);
    const int size = ptIn.size();
    uint16_t pixelModuleIndex = (*detIdToIndex)[1];
    unsigned int* hitIndices0_host = &hitIndices0[0];
    unsigned int* hitIndices1_host = &hitIndices1[0];
    unsigned int* hitIndices2_host = &hitIndices2[0];
    unsigned int* hitIndices3_host = &hitIndices3[0];
    float* dPhiChange_host = &dPhiChange[0];
    float* ptIn_host = &ptIn[0];
    float* ptErr_host = &ptErr[0];
    float* px_host = &px[0];
    float* py_host = &py[0];
    float* pz_host = &pz[0];
    float* etaErr_host = &etaErr[0];
    float* eta_host = &eta[0];
    float* phi_host = &phi[0];
    int* charge_host = &charge[0];
    unsigned int* seedIdx_host = &seedIdx[0];
    int* superbin_host = &superbin[0];
    int8_t* pixelType_host = &pixelType[0];
    short* isQuad_host = &isQuad[0];

    unsigned int* hitIndices0_dev;
    unsigned int* hitIndices1_dev;
    unsigned int* hitIndices2_dev;
    unsigned int* hitIndices3_dev;
    float* dPhiChange_dev;
    float* ptIn_dev;
    float* ptErr_dev;
    float* px_dev;
    float* py_dev;
    float* pz_dev;
    float* etaErr_dev;
    float* eta_dev;
    float* phi_dev;
    int* charge_dev;
    unsigned int* seedIdx_dev;
    int* superbin_dev;
    int8_t* pixelType_dev;
    short* isQuad_dev;
    hitIndices0_dev = (unsigned int*)cms::cuda::allocate_device(dev, size*sizeof(unsigned int), stream);
    hitIndices1_dev = (unsigned int*)cms::cuda::allocate_device(dev, size*sizeof(unsigned int), stream);
    hitIndices2_dev = (unsigned int*)cms::cuda::allocate_device(dev, size*sizeof(unsigned int), stream);
    hitIndices3_dev = (unsigned int*)cms::cuda::allocate_device(dev, size*sizeof(unsigned int), stream);
    dPhiChange_dev = (float*)cms::cuda::allocate_device(dev, size*sizeof(float), stream);
    ptIn_dev = (float*)cms::cuda::allocate_device(dev, size*sizeof(float), stream);
    ptErr_dev = (float*)cms::cuda::allocate_device(dev, size*sizeof(float), stream);
    px_dev = (float*)cms::cuda::allocate_device(dev, size*sizeof(float), stream);
    py_dev = (float*)cms::cuda::allocate_device(dev, size*sizeof(float), stream);
    pz_dev = (float*)cms::cuda::allocate_device(dev, size*sizeof(float), stream);
    etaErr_dev = (float*)cms::cuda::allocate_device(dev, size*sizeof(float), stream);
    eta_dev = (float*)cms::cuda::allocate_device(dev, size*sizeof(float), stream);
    phi_dev = (float*)cms::cuda::allocate_device(dev, size*sizeof(float), stream);
    charge_dev = (int*)cms::cuda::allocate_device(dev, size*sizeof(int), stream);
    seedIdx_dev = (unsigned int*)cms::cuda::allocate_device(dev, size*sizeof(unsigned int), stream);
    superbin_dev = (int*)cms::cuda::allocate_device(dev, size*sizeof(int), stream);
    pixelType_dev = (int8_t*)cms::cuda::allocate_device(dev, size*sizeof(int8_t), stream);
    isQuad_dev = (short*)cms::cuda::allocate_device(dev, size*sizeof(short), stream);

    cudaMemcpyAsync(hitIndices0_dev,hitIndices0_host,size*sizeof(unsigned int),cudaMemcpyHostToDevice,stream);
    cudaMemcpyAsync(hitIndices1_dev,hitIndices1_host,size*sizeof(unsigned int),cudaMemcpyHostToDevice,stream);
    cudaMemcpyAsync(hitIndices2_dev,hitIndices2_host,size*sizeof(unsigned int),cudaMemcpyHostToDevice,stream);
    cudaMemcpyAsync(hitIndices3_dev,hitIndices3_host,size*sizeof(unsigned int),cudaMemcpyHostToDevice,stream);
    cudaMemcpyAsync(dPhiChange_dev,dPhiChange_host,size*sizeof(float),cudaMemcpyHostToDevice,stream);
    cudaMemcpyAsync(ptIn_dev,ptIn_host,size*sizeof(float),cudaMemcpyHostToDevice,stream);
    cudaMemcpyAsync(ptErr_dev,ptErr_host,size*sizeof(float),cudaMemcpyHostToDevice,stream);
    cudaMemcpyAsync(px_dev,px_host,size*sizeof(float),cudaMemcpyHostToDevice,stream);
    cudaMemcpyAsync(py_dev,py_host,size*sizeof(float),cudaMemcpyHostToDevice,stream);
    cudaMemcpyAsync(pz_dev,pz_host,size*sizeof(float),cudaMemcpyHostToDevice,stream);
    cudaMemcpyAsync(etaErr_dev,etaErr_host,size*sizeof(float),cudaMemcpyHostToDevice,stream);
    cudaMemcpyAsync(eta_dev, eta_host, size*sizeof(float),cudaMemcpyHostToDevice,stream);
    cudaMemcpyAsync(phi_dev, phi_host, size*sizeof(float),cudaMemcpyHostToDevice,stream);
    cudaMemcpyAsync(charge_dev, charge_host, size*sizeof(int),cudaMemcpyHostToDevice,stream);
    cudaMemcpyAsync(seedIdx_dev, seedIdx_host, size*sizeof(unsigned int),cudaMemcpyHostToDevice,stream);
    cudaMemcpyAsync(superbin_dev,superbin_host,size*sizeof(int),cudaMemcpyHostToDevice,stream);
    cudaMemcpyAsync(pixelType_dev,pixelType_host,size*sizeof(int8_t),cudaMemcpyHostToDevice,stream);
    cudaMemcpyAsync(isQuad_dev,isQuad_host,size*sizeof(short),cudaMemcpyHostToDevice,stream);

    cudaStreamSynchronize(stream);
    unsigned int nThreads = 256;
    unsigned int nBlocks =  MAX_BLOCKS;//size % nThreads == 0 ? size/nThreads : size/nThreads + 1;

    addPixelSegmentToEventKernel<<<nBlocks,nThreads,0,stream>>>(hitIndices0_dev,hitIndices1_dev,hitIndices2_dev,hitIndices3_dev,dPhiChange_dev,ptIn_dev,ptErr_dev,px_dev,py_dev,pz_dev,eta_dev, etaErr_dev, phi_dev, charge_dev, seedIdx_dev, pixelModuleIndex, *modulesInGPU, *rangesInGPU, *hitsInGPU,*mdsInGPU,*segmentsInGPU,size, superbin_dev, pixelType_dev,isQuad_dev);

   //cudaDeviceSynchronize();
   cudaStreamSynchronize(stream);
   cudaMemcpyAsync(&(segmentsInGPU->nSegments)[pixelModuleIndex], &size, sizeof(unsigned int), cudaMemcpyHostToDevice,stream);
   cudaMemcpyAsync(&(segmentsInGPU->totOccupancySegments)[pixelModuleIndex], &size, sizeof(unsigned int), cudaMemcpyHostToDevice,stream);
   unsigned int mdSize = 2 * size;
   cudaMemcpyAsync(&(mdsInGPU->nMDs)[pixelModuleIndex], &mdSize, sizeof(unsigned int), cudaMemcpyHostToDevice,stream);
   cudaMemcpyAsync(&(mdsInGPU->totOccupancyMDs)[pixelModuleIndex], &mdSize, sizeof(unsigned int), cudaMemcpyHostToDevice,stream);
   cudaStreamSynchronize(stream);
  
    cms::cuda::free_device(dev, hitIndices0_dev);
    cms::cuda::free_device(dev, hitIndices1_dev);
    cms::cuda::free_device(dev, hitIndices2_dev);
    cms::cuda::free_device(dev, hitIndices3_dev);
    cms::cuda::free_device(dev, dPhiChange_dev);
    cms::cuda::free_device(dev, ptIn_dev);
    cms::cuda::free_device(dev, ptErr_dev);
    cms::cuda::free_device(dev, px_dev);
    cms::cuda::free_device(dev, py_dev);
    cms::cuda::free_device(dev, pz_dev);
    cms::cuda::free_device(dev, etaErr_dev);
    cms::cuda::free_device(dev, eta_dev);
    cms::cuda::free_device(dev, phi_dev);
    cms::cuda::free_device(dev, charge_dev);
    cms::cuda::free_device(dev, seedIdx_dev);
    cms::cuda::free_device(dev, superbin_dev);
    cms::cuda::free_device(dev, pixelType_dev);
    cms::cuda::free_device(dev, isQuad_dev);
    cudaStreamSynchronize(stream);
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
    createMDArrayRangesGPU<<<1,1024,0,stream>>>(*modulesInGPU, *rangesInGPU); 
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

    dim3 nThreads(32,16,1);
    //dim3 nThreads(64,16,1);
    dim3 nBlocks(1,nLowerModules/nThreads.y,1); // max parallelization

    SDL::createMiniDoubletsInGPUv2<<<nBlocks,nThreads,0,stream>>>(*modulesInGPU,*hitsInGPU,*mdsInGPU,*rangesInGPU);
    addMiniDoubletRangesToEventExplicit<<<1,1024,0,stream>>>(*modulesInGPU,*mdsInGPU, *rangesInGPU,*hitsInGPU);

    cudaError_t cudaerr = cudaGetLastError(); 
    if(cudaerr != cudaSuccess)
    {
        std::cout<<"sync failed with error : "<<cudaGetErrorString(cudaerr)<<std::endl;
    }
    cudaStreamSynchronize(stream);

    if(addObjects){
      addMiniDoubletsToEventExplicit();
    }

}

void SDL::Event::createSegmentsWithModuleMap()
{
    if(segmentsInGPU == nullptr)
    {
        segmentsInGPU = (SDL::segments*)cms::cuda::allocate_host(sizeof(SDL::segments), stream);
        createSegmentsInExplicitMemory(*segmentsInGPU, nTotalSegments, nLowerModules, N_MAX_PIXEL_SEGMENTS_PER_MODULE,stream);
    }

//HERE
    dim3 cSnThreads(64,1,1);
    uint32_t blks = nLowerModules;
    dim3 cSnBlocks(blks,1,1);
    SDL::createSegmentsInGPUv2<<<cSnBlocks,cSnThreads,0,stream>>>(*modulesInGPU, *mdsInGPU, *segmentsInGPU, *rangesInGPU);
    addSegmentRangesToEventExplicit<<<1,1024,0,stream>>>(*modulesInGPU,*segmentsInGPU, *rangesInGPU);
    cudaError_t cudaerr = cudaGetLastError();
    if(cudaerr != cudaSuccess)
    {
        std::cout<<"sync failed with error : "<<cudaGetErrorString(cudaerr)<<std::endl;
    }
    cudaStreamSynchronize(stream);
    if(addObjects){
      addSegmentsToEventExplicit();
    }
}


void SDL::Event::createTriplets()
{
    if(tripletsInGPU == nullptr)
    {
        tripletsInGPU = (SDL::triplets*)cms::cuda::allocate_host(sizeof(SDL::triplets), stream);
        unsigned int maxTriplets;
        createTripletArrayRanges<<<1,1024,0,stream>>>(*modulesInGPU, *rangesInGPU, *segmentsInGPU);
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

    dim3 nThreads(16,16,1);
    dim3 nBlocks(1,1,MAX_BLOCKS);
    //createTripletsInGPU<<<nBlocks,nThreads,0,stream>>>(*modulesInGPU, *mdsInGPU, *segmentsInGPU, *tripletsInGPU, *rangesInGPU, index_gpu,nonZeroModules);
    SDL::createTripletsInGPUv2<<<nBlocks,nThreads,0,stream>>>(*modulesInGPU, *mdsInGPU, *segmentsInGPU, *tripletsInGPU, *rangesInGPU, index_gpu,nonZeroModules);
    addTripletRangesToEventExplicit<<<1,1024,0,stream>>>(*modulesInGPU,*tripletsInGPU,*rangesInGPU);
    cudaError_t cudaerr =cudaGetLastError();
    if(cudaerr != cudaSuccess)
    {
	    std::cout<<"sync failed with error : "<<cudaGetErrorString(cudaerr)<<std::endl;
    } 
    cudaStreamSynchronize(stream);
    free(nSegments);
    free(index);
    cms::cuda::free_device(dev, index_gpu);

    if(addObjects){
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

    //printf("running final state pT3\n");
    dim3 nThreadsT3(64,16,1);
    dim3 nBlocksT3(20,4,1);
    SDL::crossCleanpT3<<<nBlocksT3, nThreadsT3,0,stream>>>(*modulesInGPU, *rangesInGPU, *pixelTripletsInGPU, *segmentsInGPU, *pixelQuintupletsInGPU);
    cudaError_t cudaerr_pT3 = cudaGetLastError();
    if(cudaerr_pT3 != cudaSuccess)
    {
        std::cout<<"sync failed with error : "<<cudaGetErrorString(cudaerr_pT3)<<std::endl;
    }cudaStreamSynchronize(stream);

    //adding objects
    SDL::addpT3asTrackCandidatesInGPU<<<1,512,0,stream>>>(nLowerModules, *pixelTripletsInGPU, *trackCandidatesInGPU, *segmentsInGPU, *rangesInGPU);
    cudaError_t cudaerr_pT3TC = cudaGetLastError();
    if(cudaerr_pT3TC != cudaSuccess)
    {
        std::cout<<"sync failed with error : "<<cudaGetErrorString(cudaerr_pT3TC)<<std::endl;
    }cudaStreamSynchronize(stream);

    //dim3 dupThreads(32,16,2);
    //dim3 dupBlocks(1,1,MAX_BLOCKS);
    dim3 dupThreads(32,16,1);
    dim3 dupBlocks(max(nEligibleModules/32,1),max(nEligibleModules/16,1),1);

    removeDupQuintupletsInGPUBeforeTC<<<dupBlocks,dupThreads,0,stream>>>(*quintupletsInGPU,*rangesInGPU);
    cudaStreamSynchronize(stream);

    dim3 nThreads(32,1,32);
    dim3 nBlocks(MAX_BLOCKS,1,(13296/32) + 1);
    crossCleanT5<<<nBlocks,nThreads,0,stream>>>(*modulesInGPU, *quintupletsInGPU, *pixelQuintupletsInGPU,*pixelTripletsInGPU,*rangesInGPU);
    cudaError_t cudaerr_T5 =cudaGetLastError(); 
    if(cudaerr_T5 != cudaSuccess)
    {
        std::cout<<"sync failed with error : "<<cudaGetErrorString(cudaerr_T5)<<std::endl;
    }cudaStreamSynchronize(stream);

    dim3 nThreadsAddT5(128,8,1);
    dim3 nBlocksAddT5(10,8,1);
    addT5asTrackCandidateInGPU<<<nBlocksAddT5, nThreadsAddT5, 0, stream>>>(nLowerModules, *quintupletsInGPU, *trackCandidatesInGPU, *rangesInGPU);
    cudaError_t cudaerr_T5TC =cudaGetLastError(); 
    if(cudaerr_T5TC != cudaSuccess)
    {
        std::cout<<"sync failed with error : "<<cudaGetErrorString(cudaerr_T5TC)<<std::endl;
    }cudaStreamSynchronize(stream);
    dim3 nThreadspLS(32,32,1);
    dim3 nBlockspLS(MAX_BLOCKS/4, MAX_BLOCKS*4, 1);
    checkHitspLS<<<nBlockspLS, nThreadspLS, 0,stream>>>(*modulesInGPU, *rangesInGPU, *mdsInGPU, *segmentsInGPU, *hitsInGPU, true);
    cudaError_t cudaerrpix = cudaGetLastError();
    if(cudaerrpix != cudaSuccess)
    {
        std::cout<<"sync failed with error : "<<cudaGetErrorString(cudaerrpix)<<std::endl;

    }cudaStreamSynchronize(stream);

    dim3 nThreads_pLS(32,16,1);
    dim3 nBlocks_pLS(20,4,1);
    SDL::crossCleanpLS<<<nBlocks_pLS, nThreads_pLS, 0, stream>>>(*modulesInGPU, *rangesInGPU, *pixelTripletsInGPU, *trackCandidatesInGPU, *segmentsInGPU, *mdsInGPU,*hitsInGPU, *quintupletsInGPU);
    cudaError_t cudaerr_pLS = cudaGetLastError();
    if(cudaerr_pLS != cudaSuccess)
    {
        std::cout<<"sync failed with error : "<<cudaGetErrorString(cudaerr_pLS)<<std::endl;
    }cudaStreamSynchronize(stream);

    unsigned int nThreadsx_pLS = 384;
    unsigned int nBlocksx_pLS = MAX_BLOCKS;//(20000) % nThreadsx_pLS == 0 ? 20000 / nThreadsx_pLS : 20000 / nThreadsx_pLS + 1;
    SDL::addpLSasTrackCandidateInGPU<<<nBlocksx_pLS, nThreadsx_pLS, 0, stream>>>(nLowerModules, *trackCandidatesInGPU, *segmentsInGPU);
    cudaError_t cudaerr_pLSTC = cudaGetLastError();
    if(cudaerr_pLSTC != cudaSuccess)
    {
        std::cout<<"sync failed with error : "<<cudaGetErrorString(cudaerr_pLS)<<std::endl;
    }cudaStreamSynchronize(stream);
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
    cudaMemcpyAsync(&nInnerSegments, &(segmentsInGPU->nSegments[pixelModuleIndex]), sizeof(unsigned int), cudaMemcpyDeviceToHost,stream);
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

    // unsigned int max_size =0;
    cudaStreamSynchronize(stream);
    int pixelIndexOffsetPos = pixelMapping->connectedPixelsIndex[44999] + pixelMapping->connectedPixelsSizes[44999];
    int pixelIndexOffsetNeg = pixelMapping->connectedPixelsIndexPos[44999] + pixelMapping->connectedPixelsSizes[44999] + pixelIndexOffsetPos;

    // TODO: check if a map/reduction to just eligible pLSs would speed up the kernel
    //   the current selection still leaves a significant fraction of unmatchable pLSs
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

    dim3 nThreads(32,4,1);
    dim3 nBlocks(1,4096,16 /* above median of connected modules*/);

    SDL::createPixelTripletsInGPUFromMapv2<<<nBlocks, nThreads,0,stream>>>(*modulesInGPU, *rangesInGPU, *mdsInGPU, *segmentsInGPU, *tripletsInGPU, *pixelTripletsInGPU, connectedPixelSize_dev,connectedPixelIndex_dev,nInnerSegments);

    cudaError_t cudaerr = cudaGetLastError();
    if(cudaerr != cudaSuccess)
    {
	    std::cout<<"sync failed with error : "<<cudaGetErrorString(cudaerr)<<std::endl;

    }
    cudaStreamSynchronize(stream);
    //}cudaDeviceSynchronize();
    cms::cuda::free_device(dev, connectedPixelSize_dev);
    cms::cuda::free_device(dev, connectedPixelIndex_dev);


#ifdef Warnings
    unsigned int nPixelTriplets;
    cudaMemcpyAsync(&nPixelTriplets, pixelTripletsInGPU->nPixelTriplets,  sizeof(unsigned int), cudaMemcpyDeviceToHost,stream);
    cudaStreamSynchronize(stream);
    std::cout<<"number of pixel triplets = "<<nPixelTriplets<<std::endl;
#endif

    //pT3s can be cleaned here because they're not used in making pT5s!
    //dim3 nThreads_dup(160,1,1);
    dim3 nThreads_dup(32,32,1);
    dim3 nBlocks_dup(1,40,1); //seems like more blocks lead to conflicting writes
    removeDupPixelTripletsInGPUFromMap<<<nBlocks_dup,nThreads_dup,0,stream>>>(*pixelTripletsInGPU,false);
    cudaStreamSynchronize(stream);
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
    createEligibleModulesListForQuintupletsGPU<<<1,1024,0,stream>>>(*modulesInGPU, *tripletsInGPU, *rangesInGPU);
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
cudaStreamSynchronize(stream);



    dim3 nThreads(32, 8, 1);
    dim3 nBlocks(1,1,max(nEligibleT5Modules,1));

    SDL::createQuintupletsInGPUv2<<<nBlocks,nThreads,0,stream>>>(*modulesInGPU, *mdsInGPU, *segmentsInGPU, *tripletsInGPU, *quintupletsInGPU, *rangesInGPU,nEligibleT5Modules);
    cudaError_t cudaerr = cudaGetLastError();
    if(cudaerr != cudaSuccess)
    {
	    std::cout<<"sync failed with error : "<<cudaGetErrorString(cudaerr)<<std::endl;
    }
    cudaStreamSynchronize(stream);
    //free(indicesOfEligibleModules);

    dim3 dupThreads(32,32,1);
    dim3 dupBlocks(1,1,MAX_BLOCKS);
    removeDupQuintupletsInGPUAfterBuild<<<dupBlocks,dupThreads,0,stream>>>(*modulesInGPU, *quintupletsInGPU,*rangesInGPU);
    addQuintupletRangesToEventExplicit<<<1,1024,0,stream>>>(*modulesInGPU, *quintupletsInGPU,*rangesInGPU);
    cudaStreamSynchronize(stream);

    if(addObjects){
      addQuintupletsToEventExplicit();
    }

}
void SDL::Event::pixelLineSegmentCleaning()
{
    dim3 nThreadspLS(32,32,1);
    dim3 nBlockspLS(MAX_BLOCKS/4, MAX_BLOCKS*4, 1);

    checkHitspLS<<<nBlockspLS, nThreadspLS, 0,stream>>>(*modulesInGPU, *rangesInGPU, *mdsInGPU, *segmentsInGPU, *hitsInGPU, false);
    cudaError_t cudaerrpix = cudaGetLastError();
    if(cudaerrpix != cudaSuccess)
    {
        std::cout<<"sync failed with error : "<<cudaGetErrorString(cudaerrpix)<<std::endl;

    }cudaStreamSynchronize(stream);
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
    unsigned int *nQuintuplets;

    unsigned int* connectedPixelSize_host;
    unsigned int* connectedPixelIndex_host;
    unsigned int* connectedPixelSize_dev;
    unsigned int* connectedPixelIndex_dev;

    nQuintuplets = (unsigned int*)cms::cuda::allocate_host(nLowerModules * sizeof(unsigned int), stream);
    cudaMemcpyAsync(nQuintuplets, quintupletsInGPU->nQuintuplets, nLowerModules * sizeof(unsigned int), cudaMemcpyDeviceToHost,stream);

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

    //less cheap method to estimate max_size for y axis
    unsigned int max_size = *std::max_element(nQuintuplets, nQuintuplets + nLowerModules);
    dim3 nThreads(16,16,1);
    dim3 nBlocks(1,MAX_BLOCKS,16);
                  
    SDL::createPixelQuintupletsInGPUFromMapv2<<<nBlocks, nThreads,0,stream>>>(*modulesInGPU, *mdsInGPU, *segmentsInGPU, *tripletsInGPU, *quintupletsInGPU, *pixelQuintupletsInGPU, connectedPixelSize_dev, connectedPixelIndex_dev, nInnerSegments,*rangesInGPU);

    cudaError_t cudaerr = cudaGetLastError();
    if(cudaerr != cudaSuccess)
    {
	    std::cout<<"sync failed with error : "<<cudaGetErrorString(cudaerr)<<std::endl;

    }
    cudaStreamSynchronize(stream);
    cms::cuda::free_host(connectedPixelSize_host);
    cms::cuda::free_host(connectedPixelIndex_host);
    cms::cuda::free_device(dev, connectedPixelSize_dev);
    cms::cuda::free_device(dev, connectedPixelIndex_dev);
    cms::cuda::free_host(superbins);
    cms::cuda::free_host(pixelTypes);
    cms::cuda::free_host(nQuintuplets);
    //free(segs_pix);
    //cudaFree(segs_pix_gpu);

    dim3 nThreads_dup(32,32,1);
    dim3 nBlocks_dup(1,MAX_BLOCKS,1);
    //printf("run dup pT5\n");
    removeDupPixelQuintupletsInGPUFromMap<<<nBlocks_dup,nThreads_dup,0,stream>>>(*pixelQuintupletsInGPU, false);
    cudaError_t cudaerr2 = cudaGetLastError(); 
    if(cudaerr2 != cudaSuccess)
    {
        std::cout<<"sync failed with error : "<<cudaGetErrorString(cudaerr2)<<std::endl;
    }cudaStreamSynchronize(stream);
    unsigned int nThreadsx_pT5 = 256;
    unsigned int nBlocksx_pT5 = 1;//(N_MAX_PIXEL_QUINTUPLETS) % nThreadsx_pT5 == 0 ? N_MAX_PIXEL_QUINTUPLETS / nThreadsx_pT5 : N_MAX_PIXEL_QUINTUPLETS / nThreadsx_pT5 + 1;
    SDL::addpT5asTrackCandidateInGPU<<<nBlocksx_pT5, nThreadsx_pT5,0,stream>>>(nLowerModules, *pixelQuintupletsInGPU, *trackCandidatesInGPU, *segmentsInGPU, *rangesInGPU);

    cudaError_t cudaerr_pT5 = cudaGetLastError();
    if(cudaerr_pT5 != cudaSuccess)
    {
        std::cout<<"sync failed with error : "<<cudaGetErrorString(cudaerr_pT5)<<std::endl;
    }
    cudaStreamSynchronize(stream);
#ifdef Warnings
    unsigned int nPixelQuintuplets;
    cudaMemcpyAsync(&nPixelQuintuplets, &(pixelQuintupletsInGPU->nPixelQuintuplets), sizeof(unsigned int), cudaMemcpyDeviceToHost,stream);
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

unsigned int SDL::Event::getNumberOfPixelTriplets()
{
    unsigned int nPixelTriplets;
    cudaMemcpyAsync(&nPixelTriplets, pixelTripletsInGPU->nPixelTriplets, sizeof(unsigned int), cudaMemcpyDeviceToHost,stream);
    cudaStreamSynchronize(stream);
    return nPixelTriplets;
}

unsigned int SDL::Event::getNumberOfPixelQuintuplets()
{
    unsigned int nPixelQuintuplets;
    cudaMemcpyAsync(&nPixelQuintuplets, pixelQuintupletsInGPU->nPixelQuintuplets, sizeof(unsigned int), cudaMemcpyDeviceToHost,stream);
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

unsigned int SDL::Event::getNumberOfTrackCandidates()
{    
    unsigned int nTrackCandidates;
    cudaMemcpyAsync(&nTrackCandidates, trackCandidatesInGPU->nTrackCandidates, sizeof(unsigned int), cudaMemcpyDeviceToHost,stream);
cudaStreamSynchronize(stream);

    return nTrackCandidates;
}

unsigned int SDL::Event::getNumberOfPT5TrackCandidates()
{
    unsigned int nTrackCandidatesPT5;
    cudaMemcpyAsync(&nTrackCandidatesPT5, trackCandidatesInGPU->nTrackCandidatespT5, sizeof(unsigned int), cudaMemcpyDeviceToHost,stream);
cudaStreamSynchronize(stream);

    return nTrackCandidatesPT5;
}

unsigned int SDL::Event::getNumberOfPT3TrackCandidates()
{
    unsigned int nTrackCandidatesPT3;
    cudaMemcpyAsync(&nTrackCandidatesPT3, trackCandidatesInGPU->nTrackCandidatespT3, sizeof(unsigned int), cudaMemcpyDeviceToHost,stream);
cudaStreamSynchronize(stream);

    return nTrackCandidatesPT3;
}

unsigned int SDL::Event::getNumberOfPLSTrackCandidates()
{
    unsigned int nTrackCandidatesPLS;
    cudaMemcpyAsync(&nTrackCandidatesPLS, trackCandidatesInGPU->nTrackCandidatespLS, sizeof(unsigned int), cudaMemcpyDeviceToHost,stream);
cudaStreamSynchronize(stream);

    return nTrackCandidatesPLS;
}

unsigned int SDL::Event::getNumberOfPixelTrackCandidates()
{
    unsigned int nTrackCandidates;
    unsigned int nTrackCandidatesT5;
    cudaMemcpyAsync(&nTrackCandidates, trackCandidatesInGPU->nTrackCandidates, sizeof(unsigned int), cudaMemcpyDeviceToHost,stream);
    cudaMemcpyAsync(&nTrackCandidatesT5, trackCandidatesInGPU->nTrackCandidatesT5, sizeof(unsigned int), cudaMemcpyDeviceToHost,stream);
cudaStreamSynchronize(stream);

    return nTrackCandidates - nTrackCandidatesT5;
}

unsigned int SDL::Event::getNumberOfT5TrackCandidates()
{
    unsigned int nTrackCandidatesT5;
    cudaMemcpyAsync(&nTrackCandidatesT5, trackCandidatesInGPU->nTrackCandidatesT5, sizeof(unsigned int), cudaMemcpyDeviceToHost,stream);
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
        mdsInCPU->nMDs = new unsigned int[nLowerModules+1];

        //compute memory locations
        mdsInCPU->nMemoryLocations = new unsigned int;
        cudaMemcpyAsync(mdsInCPU->nMemoryLocations, mdsInGPU->nMemoryLocations, sizeof(unsigned int), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        mdsInCPU->totOccupancyMDs = new unsigned int[nLowerModules+1];

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

SDL::segments* SDL::Event::getSegments()
{
    if(segmentsInCPU == nullptr)
    {
        segmentsInCPU = new SDL::segments;
        
        segmentsInCPU->nSegments = new unsigned int[nLowerModules+1];
        cudaMemcpyAsync(segmentsInCPU->nSegments, segmentsInGPU->nSegments, (nLowerModules+1) * sizeof(unsigned int), cudaMemcpyDeviceToHost,stream);
        
        segmentsInCPU->nMemoryLocations = new unsigned int;
        cudaMemcpyAsync(segmentsInCPU->nMemoryLocations, segmentsInGPU->nMemoryLocations, sizeof(unsigned int), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        segmentsInCPU->mdIndices = new unsigned int[2 * *(segmentsInCPU->nMemoryLocations)];
        segmentsInCPU->innerMiniDoubletAnchorHitIndices = new unsigned int[*(segmentsInCPU->nMemoryLocations)];
        segmentsInCPU->outerMiniDoubletAnchorHitIndices = new unsigned int[*(segmentsInCPU->nMemoryLocations)];
        segmentsInCPU->totOccupancySegments = new unsigned int[nLowerModules+1];

        segmentsInCPU->ptIn = new float[N_MAX_PIXEL_SEGMENTS_PER_MODULE];
        segmentsInCPU->eta = new float[N_MAX_PIXEL_SEGMENTS_PER_MODULE];
        segmentsInCPU->phi = new float[N_MAX_PIXEL_SEGMENTS_PER_MODULE];
        segmentsInCPU->seedIdx = new unsigned int[N_MAX_PIXEL_SEGMENTS_PER_MODULE];
        segmentsInCPU->isDup = new bool[N_MAX_PIXEL_SEGMENTS_PER_MODULE];
        segmentsInCPU->isQuad = new bool[N_MAX_PIXEL_SEGMENTS_PER_MODULE];
        segmentsInCPU->score = new float[N_MAX_PIXEL_SEGMENTS_PER_MODULE];

        cudaMemcpyAsync(segmentsInCPU->mdIndices, segmentsInGPU->mdIndices, 2 * *(segmentsInCPU->nMemoryLocations) * sizeof(unsigned int), cudaMemcpyDeviceToHost,stream);
        cudaMemcpyAsync(segmentsInCPU->innerMiniDoubletAnchorHitIndices, segmentsInGPU->innerMiniDoubletAnchorHitIndices, *(segmentsInCPU->nMemoryLocations) * sizeof(unsigned int), cudaMemcpyDeviceToHost,stream);
        cudaMemcpyAsync(segmentsInCPU->outerMiniDoubletAnchorHitIndices, segmentsInGPU->outerMiniDoubletAnchorHitIndices, *(segmentsInCPU->nMemoryLocations) * sizeof(unsigned int), cudaMemcpyDeviceToHost,stream);
        cudaMemcpyAsync(segmentsInCPU->totOccupancySegments, segmentsInGPU->totOccupancySegments, (nLowerModules+1) * sizeof(unsigned int), cudaMemcpyDeviceToHost,stream);
        cudaMemcpyAsync(segmentsInCPU->ptIn, segmentsInGPU->ptIn, N_MAX_PIXEL_SEGMENTS_PER_MODULE * sizeof(float), cudaMemcpyDeviceToHost,stream);
        cudaMemcpyAsync(segmentsInCPU->eta, segmentsInGPU->eta, N_MAX_PIXEL_SEGMENTS_PER_MODULE * sizeof(float), cudaMemcpyDeviceToHost,stream);
        cudaMemcpyAsync(segmentsInCPU->phi, segmentsInGPU->phi, N_MAX_PIXEL_SEGMENTS_PER_MODULE * sizeof(float), cudaMemcpyDeviceToHost,stream);
        cudaMemcpyAsync(segmentsInCPU->seedIdx, segmentsInGPU->seedIdx, N_MAX_PIXEL_SEGMENTS_PER_MODULE * sizeof(unsigned int), cudaMemcpyDeviceToHost,stream);
        cudaMemcpyAsync(segmentsInCPU->isDup, segmentsInGPU->isDup, N_MAX_PIXEL_SEGMENTS_PER_MODULE * sizeof(bool), cudaMemcpyDeviceToHost,stream);
        cudaMemcpyAsync(segmentsInCPU->isQuad, segmentsInGPU->isQuad, N_MAX_PIXEL_SEGMENTS_PER_MODULE * sizeof(bool), cudaMemcpyDeviceToHost,stream);
        cudaMemcpyAsync(segmentsInCPU->score, segmentsInGPU->score, N_MAX_PIXEL_SEGMENTS_PER_MODULE * sizeof(float), cudaMemcpyDeviceToHost,stream);
        cudaStreamSynchronize(stream);
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
        tripletsInCPU->nTriplets = new unsigned int[nLowerModules];
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
        tripletsInCPU->totOccupancyTriplets = new unsigned int[nLowerModules];
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

        quintupletsInCPU->nQuintuplets = new unsigned int[nLowerModules];
        quintupletsInCPU->totOccupancyQuintuplets = new unsigned int[nLowerModules];
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

        cudaMemcpyAsync(quintupletsInCPU->nQuintuplets, quintupletsInGPU->nQuintuplets,  nLowerModules * sizeof(unsigned int), cudaMemcpyDeviceToHost,stream);
        cudaMemcpyAsync(quintupletsInCPU->totOccupancyQuintuplets, quintupletsInGPU->totOccupancyQuintuplets,  nLowerModules * sizeof(unsigned int), cudaMemcpyDeviceToHost,stream);
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

        pixelTripletsInCPU->nPixelTriplets = new unsigned int;
        pixelTripletsInCPU->totOccupancyPixelTriplets = new unsigned int;
        cudaMemcpyAsync(pixelTripletsInCPU->nPixelTriplets, pixelTripletsInGPU->nPixelTriplets, sizeof(unsigned int), cudaMemcpyDeviceToHost,stream);
        cudaMemcpyAsync(pixelTripletsInCPU->totOccupancyPixelTriplets, pixelTripletsInGPU->totOccupancyPixelTriplets, sizeof(unsigned int), cudaMemcpyDeviceToHost,stream);
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

        pixelQuintupletsInCPU->nPixelQuintuplets = new unsigned int;
        pixelQuintupletsInCPU->totOccupancyPixelQuintuplets = new unsigned int;
        cudaMemcpyAsync(pixelQuintupletsInCPU->nPixelQuintuplets, pixelQuintupletsInGPU->nPixelQuintuplets, sizeof(unsigned int), cudaMemcpyDeviceToHost,stream);
        cudaMemcpyAsync(pixelQuintupletsInCPU->totOccupancyPixelQuintuplets, pixelQuintupletsInGPU->totOccupancyPixelQuintuplets, sizeof(unsigned int), cudaMemcpyDeviceToHost,stream);
cudaStreamSynchronize(stream);
        unsigned int nPixelQuintuplets = *(pixelQuintupletsInCPU->nPixelQuintuplets);

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
        trackCandidatesInCPU->nTrackCandidates = new unsigned int;
        cudaMemcpyAsync(trackCandidatesInCPU->nTrackCandidates, trackCandidatesInGPU->nTrackCandidates, sizeof(unsigned int), cudaMemcpyDeviceToHost,stream);
        cudaStreamSynchronize(stream);
        unsigned int nTrackCandidates = *(trackCandidatesInCPU->nTrackCandidates);

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
        trackCandidatesInCPU->nTrackCandidates = new unsigned int;
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


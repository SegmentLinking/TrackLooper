# include "Event.cuh"
#include "allocate.h"

struct SDL::modules* SDL::modulesInGPU = nullptr;
struct SDL::pixelMap* SDL::pixelMapping = nullptr;
unsigned int SDL::nModules;
unsigned int SDL::nLowerModules;

SDL::Event::Event(cudaStream_t estream)
{
    int version;
    int driver;
            cudaRuntimeGetVersion(&version);
            cudaDriverGetVersion(&driver);
    //printf("version: %d Driver %d\n",version, driver);
    stream = estream;
    hitsInGPU = nullptr;
    mdsInGPU = nullptr;
    segmentsInGPU = nullptr;
    tripletsInGPU = nullptr;
    quintupletsInGPU = nullptr;
    trackCandidatesInGPU = nullptr;
    pixelTripletsInGPU = nullptr;
    pixelQuintupletsInGPU = nullptr;
    trackExtensionsInGPU = nullptr;
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
    trackExtensionsInCPU = nullptr;

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
    if(trackExtensionsInGPU){trackExtensionsInGPU->freeMemoryCache();}
#else

    if(rangesInGPU){rangesInGPU->freeMemory();}
    if(hitsInGPU){hitsInGPU->freeMemory(stream);}
    if(mdsInGPU){mdsInGPU->freeMemory(stream);}
    if(segmentsInGPU){segmentsInGPU->freeMemory(stream);}
    if(tripletsInGPU){tripletsInGPU->freeMemory(stream);}
    if(quintupletsInGPU){quintupletsInGPU->freeMemory(stream);}
    if(pixelQuintupletsInGPU){pixelQuintupletsInGPU->freeMemory(stream);}
    if(pixelTripletsInGPU){pixelTripletsInGPU->freeMemory(stream);}
    if(trackCandidatesInGPU){trackCandidatesInGPU->freeMemory(stream);}
    if(trackExtensionsInGPU){trackExtensionsInGPU->freeMemory(stream);}
#endif
    if(rangesInGPU != nullptr){cudaFreeHost(rangesInGPU);}
    if(mdsInGPU != nullptr){cudaFreeHost(mdsInGPU);}
    if(segmentsInGPU!= nullptr){cudaFreeHost(segmentsInGPU);}
    if(tripletsInGPU!= nullptr){cudaFreeHost(tripletsInGPU);}
    if(trackCandidatesInGPU!= nullptr){cudaFreeHost(trackCandidatesInGPU);}
    if(hitsInGPU!= nullptr){cudaFreeHost(hitsInGPU);}

    if(pixelTripletsInGPU!= nullptr){cudaFreeHost(pixelTripletsInGPU);}
    if(pixelQuintupletsInGPU!= nullptr){cudaFreeHost(pixelQuintupletsInGPU);}

    if(quintupletsInGPU!= nullptr){cudaFreeHost(quintupletsInGPU);}
    if(trackExtensionsInGPU != nullptr){cudaFreeHost(trackExtensionsInGPU);}

#ifdef Explicit_Hit
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
#endif
#ifdef Explicit_MD
    if(mdsInCPU != nullptr)
    {
        delete[] mdsInCPU->hitIndices;
        delete[] mdsInCPU->nMDs;
        delete mdsInCPU;
    }
#endif
#ifdef Explicit_Seg
    if(segmentsInCPU != nullptr)
    {
        delete[] segmentsInCPU->mdIndices;
        delete[] segmentsInCPU->nSegments;
        delete[] segmentsInCPU->innerMiniDoubletAnchorHitIndices;
        delete[] segmentsInCPU->outerMiniDoubletAnchorHitIndices;
        delete[] segmentsInCPU->ptIn;
        delete[] segmentsInCPU->eta;
        delete[] segmentsInCPU->phi;
        delete segmentsInCPU;
    }
#endif
#ifdef Explicit_Trips
    if(tripletsInCPU != nullptr)
    {
        delete[] tripletsInCPU->segmentIndices;
        delete[] tripletsInCPU->nTriplets;
        delete[] tripletsInCPU->betaIn;
        delete[] tripletsInCPU->betaOut;
        delete[] tripletsInCPU->pt_beta;
#ifdef TRACK_EXTENSIONS
        delete[] tripletsInCPU->hitIndices;
        delete[] tripletsInCPU->logicalLayers;
        delete[] tripletsInCPU->lowerModuleIndices;
#endif
        delete tripletsInCPU;
    }
#endif
#ifdef Explicit_T5
#ifdef FINAL_T5
    if(quintupletsInCPU != nullptr)
    {
        delete[] quintupletsInCPU->tripletIndices;
        delete[] quintupletsInCPU->nQuintuplets;
        delete[] quintupletsInCPU->lowerModuleIndices;
        delete[] quintupletsInCPU->innerRadius;
        delete[] quintupletsInCPU->outerRadius;
        delete[] quintupletsInCPU->regressionRadius;
        delete quintupletsInCPU;
    }
#endif
#endif

#ifdef Explicit_PT3
    if(pixelTripletsInCPU != nullptr)
    {
        delete[] pixelTripletsInCPU->tripletIndices;
        delete[] pixelTripletsInCPU->pixelSegmentIndices;
        delete[] pixelTripletsInCPU->pixelRadius;
        delete[] pixelTripletsInCPU->tripletRadius;
        delete pixelTripletsInCPU->nPixelTriplets;
        delete pixelTripletsInCPU;
    }
#endif
#ifdef Explicit_PT5
    if(pixelQuintupletsInCPU != nullptr)
    {
        delete[] pixelQuintupletsInCPU->pixelIndices;
        delete[] pixelQuintupletsInCPU->T5Indices;
        delete[] pixelQuintupletsInCPU->isDup;
        delete[] pixelQuintupletsInCPU->score;
        delete pixelQuintupletsInCPU->nPixelQuintuplets;
        delete pixelQuintupletsInCPU;
    }
#endif

#ifdef Explicit_Track
    if(trackCandidatesInCPU != nullptr)
    {
        delete[] trackCandidatesInCPU->objectIndices;
        delete[] trackCandidatesInCPU->trackCandidateType;
        delete[] trackCandidatesInCPU->nTrackCandidates;
#ifdef TRACK_EXTENSIONS
        delete[] trackCandidatesInCPU->hitIndices;
        delete[] trackCandidatesInCPU->logicalLayers;
        delete[] trackCandidatesInCPU->partOfExtension;
#endif
        delete trackCandidatesInCPU;
    }
#endif
#ifdef Explicit_Extensions
    if(trackExtensionsInCPU != nullptr)
    {
        delete[] trackExtensionsInCPU->nTrackExtensions;
        delete[] trackExtensionsInCPU->constituentTCTypes;
        delete[] trackExtensionsInCPU->constituentTCIndices;
        delete[] trackExtensionsInCPU->nLayerOverlaps;
        delete[] trackExtensionsInCPU->nHitOverlaps;
        delete[] trackExtensionsInCPU->isDup;
        delete[] trackExtensionsInCPU->regressionRadius;

        delete trackExtensionsInCPU;
    }
#endif
#ifdef Explicit_Module
    if(modulesInCPU != nullptr)
    {
        delete[] modulesInCPU->nLowerModules;
        delete[] modulesInCPU->nModules;
        delete[] modulesInCPU->lowerModuleIndices;
        delete[] modulesInCPU->detIds;
        delete[] modulesInCPU->isLower;
        delete[] modulesInCPU->layers;
        delete[] modulesInCPU->subdets;
        delete[] modulesInCPU->rings;
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
        delete[] modulesInCPUFull->isInverted;
        delete[] modulesInCPUFull->isLower;


        delete[] modulesInCPUFull->moduleType;
        delete[] modulesInCPUFull->moduleLayerType;

        delete[] modulesInCPUFull->lowerModuleIndices;
        delete[] modulesInCPUFull->reverseLookupLowerModuleIndices;
        delete[] modulesInCPUFull;
    }
#endif
}
void SDL::Event::resetEvent()
{
#ifdef CACHE_ALLOC
    if(hitsInGPU){hitsInGPU->freeMemoryCache();}
    if(quintupletsInGPU){quintupletsInGPU->freeMemoryCache();}
    if(rangesInGPU){rangesInGPU->freeMemoryCache();}
    if(mdsInGPU){mdsInGPU->freeMemoryCache();}
    if(segmentsInGPU){segmentsInGPU->freeMemoryCache();}
    if(tripletsInGPU){tripletsInGPU->freeMemoryCache();}
    if(pixelQuintupletsInGPU){pixelQuintupletsInGPU->freeMemoryCache();}
    if(pixelTripletsInGPU){pixelTripletsInGPU->freeMemoryCache();}
    if(trackCandidatesInGPU){trackCandidatesInGPU->freeMemoryCache();}
    if(trackExtensionsInGPU){trackExtensionsInGPU->freeMemoryCache();}
#else
    if(hitsInGPU){hitsInGPU->freeMemory(stream);}
    if(quintupletsInGPU){quintupletsInGPU->freeMemory(stream);}
    if(rangesInGPU){rangesInGPU->freeMemory();}
    if(mdsInGPU){mdsInGPU->freeMemory(stream);}
    if(segmentsInGPU){segmentsInGPU->freeMemory(stream);}
    if(tripletsInGPU){tripletsInGPU->freeMemory(stream);}
    if(pixelQuintupletsInGPU){pixelQuintupletsInGPU->freeMemory(stream);}
    if(pixelTripletsInGPU){pixelTripletsInGPU->freeMemory(stream);}
    if(trackCandidatesInGPU){trackCandidatesInGPU->freeMemory(stream);}
    if(trackExtensionsInGPU){trackExtensionsInGPU->freeMemory(stream);}
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
    if(hitsInGPU){cudaFreeHost(hitsInGPU);
    hitsInGPU = nullptr;}
    if(mdsInGPU){cudaFreeHost(mdsInGPU);
    mdsInGPU = nullptr;}
    if(rangesInGPU){cudaFreeHost(rangesInGPU);
    rangesInGPU = nullptr;}
    if(segmentsInGPU){cudaFreeHost(segmentsInGPU);
    segmentsInGPU = nullptr;}
    if(tripletsInGPU){cudaFreeHost(tripletsInGPU);
    tripletsInGPU = nullptr;}
      if(quintupletsInGPU){cudaFreeHost(quintupletsInGPU);
      quintupletsInGPU = nullptr;}
    if(trackCandidatesInGPU){cudaFreeHost(trackCandidatesInGPU);
    trackCandidatesInGPU = nullptr;}
    if(pixelTripletsInGPU){cudaFreeHost(pixelTripletsInGPU);
    pixelTripletsInGPU = nullptr;}
    if(pixelQuintupletsInGPU){cudaFreeHost(pixelQuintupletsInGPU);
    pixelQuintupletsInGPU = nullptr;}
    if(trackExtensionsInGPU){cudaFreeHost(trackExtensionsInGPU);
    trackExtensionsInGPU = nullptr;}
#ifdef Explicit_Hit
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
#endif
#ifdef Explicit_MD
    if(mdsInCPU != nullptr)
    {
        delete[] mdsInCPU->hitIndices;
        delete[] mdsInCPU->nMDs;
        delete mdsInCPU;
        mdsInCPU = nullptr;
    }
#endif
#ifdef Explicit_Seg
    if(segmentsInCPU != nullptr)
    {
        delete[] segmentsInCPU->mdIndices;
        delete[] segmentsInCPU->nSegments;
        delete[] segmentsInCPU->innerMiniDoubletAnchorHitIndices;
        delete[] segmentsInCPU->outerMiniDoubletAnchorHitIndices;
        delete[] segmentsInCPU->ptIn;
        delete[] segmentsInCPU->eta;
        delete[] segmentsInCPU->phi;
        delete segmentsInCPU;
        segmentsInCPU = nullptr;
    }
#endif
#ifdef Explicit_Trips
    if(tripletsInCPU != nullptr)
    {
        delete[] tripletsInCPU->segmentIndices;
        delete[] tripletsInCPU->nTriplets;
        delete[] tripletsInCPU->betaIn;
        delete[] tripletsInCPU->betaOut;
        delete[] tripletsInCPU->pt_beta;
#ifdef TRACK_EXTENSIONS
        delete[] tripletsInCPU->logicalLayers;
        delete[] tripletsInCPU->lowerModuleIndices;
        delete[] tripletsInCPU->hitIndices;
#endif
        delete tripletsInCPU;
        tripletsInCPU = nullptr;
    }
#endif
#ifdef Explicit_T5
#ifdef FINAL_T5
    if(quintupletsInCPU != nullptr)
    {
        delete[] quintupletsInCPU->tripletIndices;
        delete[] quintupletsInCPU->nQuintuplets;
        delete[] quintupletsInCPU->lowerModuleIndices;
        delete[] quintupletsInCPU->innerRadius;
        delete[] quintupletsInCPU->outerRadius;
        delete[] quintupletsInCPU->regressionRadius;
        delete quintupletsInCPU;
        quintupletsInCPU = nullptr;
    }
#endif
#endif

#ifdef Explicit_PT3
    if(pixelTripletsInCPU != nullptr)
    {
        delete[] pixelTripletsInCPU->tripletIndices;
        delete[] pixelTripletsInCPU->pixelSegmentIndices;
        delete[] pixelTripletsInCPU->pixelRadius;
        delete[] pixelTripletsInCPU->tripletRadius;
        delete pixelTripletsInCPU->nPixelTriplets;
        delete pixelTripletsInCPU;
        pixelTripletsInCPU = nullptr;
    }
#endif
#ifdef Explicit_PT5
    if(pixelQuintupletsInCPU != nullptr)
    {
        delete[] pixelQuintupletsInCPU->pixelIndices;
        delete[] pixelQuintupletsInCPU->T5Indices;
        delete[] pixelQuintupletsInCPU->isDup;
        delete[] pixelQuintupletsInCPU->score;
        delete pixelQuintupletsInCPU->nPixelQuintuplets;
        delete pixelQuintupletsInCPU;
        pixelQuintupletsInCPU = nullptr;
    }
#endif
#ifdef Explicit_Track
    if(trackCandidatesInCPU != nullptr)
    {
        delete[] trackCandidatesInCPU->objectIndices;
        delete[] trackCandidatesInCPU->trackCandidateType;
        delete[] trackCandidatesInCPU->nTrackCandidates;
#ifdef TRACK_EXTENSIONS
        delete[] trackCandidatesInCPU->logicalLayers;
        delete[] trackCandidatesInCPU->hitIndices;
        delete[] trackCandidatesInCPU->lowerModuleIndices;
#endif
        delete trackCandidatesInCPU;
        trackCandidatesInCPU = nullptr;
    }
#endif
#ifdef TRACK_EXTENSIONS
#ifdef Explicit_Extensions
    if(trackExtensionsInCPU != nullptr)
    {
        delete[] trackExtensionsInCPU->nTrackExtensions;
        delete[] trackExtensionsInCPU->constituentTCTypes;
        delete[] trackExtensionsInCPU->constituentTCIndices;
        delete[] trackExtensionsInCPU->nLayerOverlaps;
        delete[] trackExtensionsInCPU->nHitOverlaps;
        delete[] trackExtensionsInCPU->isDup;
        delete[] trackExtensionsInCPU->regressionRadius;

        delete trackExtensionsInCPU;
        trackExtensionsInCPU = nullptr;
    }
#endif
#endif
#ifdef Explicit_Module
    if(modulesInCPU != nullptr)
    {
        delete[] modulesInCPU->nLowerModules;
        delete[] modulesInCPU->nModules;
        delete[] modulesInCPU->lowerModuleIndices;
        delete[] modulesInCPU->detIds;
        delete[] modulesInCPU->isLower;
        delete[] modulesInCPU->layers;
        delete[] modulesInCPU->subdets;
        delete[] modulesInCPU->rings;
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
        delete[] modulesInCPUFull->subdets;
        delete[] modulesInCPUFull->sides;
        delete[] modulesInCPUFull->isInverted;
        delete[] modulesInCPUFull->isLower;


        delete[] modulesInCPUFull->moduleType;
        delete[] modulesInCPUFull->moduleLayerType;

        delete[] modulesInCPUFull->lowerModuleIndices;
        delete[] modulesInCPUFull->reverseLookupLowerModuleIndices;
        delete[] modulesInCPUFull;
        modulesInCPUFull = nullptr;
    }
#endif


}

void SDL::initModules(const char* moduleMetaDataFilePath)
{
    cudaStream_t modStream;
    cudaStreamCreate(&modStream);
    if(modulesInGPU == nullptr)
    {
        cudaMallocHost(&modulesInGPU, sizeof(struct SDL::modules));
        //pixelMapping = new pixelMap;
        cudaMallocHost(&pixelMapping, sizeof(struct SDL::pixelMap));
        loadModulesFromFile(*modulesInGPU,nModules,nLowerModules,*pixelMapping,modStream,moduleMetaDataFilePath); //nModules gets filled here
    }
    //resetObjectRanges(*modulesInGPU,nModules,modStream);
    cudaStreamSynchronize(modStream);
    cudaStreamDestroy(modStream);
}

void SDL::cleanModules()
{
  //#ifdef CACHE_ALLOC
  //freeModulesCache(*modulesInGPU,*pixelMapping); //bug in freeing cached modules. Decided to remove module caching since it doesn't change by event.
  //#else
    cudaStream_t modStream;
    cudaStreamCreate(&modStream);
  freeModules(*modulesInGPU,*pixelMapping,modStream);
    cudaStreamSynchronize(modStream);
    cudaStreamDestroy(modStream);
  //#endif
  cudaFreeHost(modulesInGPU);
  cudaFreeHost(pixelMapping);
//cudaDeviceReset(); // uncomment for leak check "cuda-memcheck --leak-check full --show-backtrace yes" does not work with caching.
}

void SDL::Event::resetObjectsInModule()
{
    resetObjectRanges(*rangesInGPU,nModules,stream);
}

// Best working hit loading method. Previously named OMP
void SDL::Event::addHitToEvent(std::vector<float> x, std::vector<float> y, std::vector<float> z, std::vector<unsigned int> detId, std::vector<unsigned int> idxInNtuple)
{
    const int loopsize = x.size();// use the actual number of hits instead of a "max"
unsigned int nLowerModules;
cudaMemcpyAsync(&nLowerModules,modulesInGPU->nLowerModules,sizeof(unsigned int),cudaMemcpyDeviceToHost,stream);
cudaStreamSynchronize(stream);
    //printf("loopsize %i\n",loopsize);

    if(rangesInGPU == nullptr)
    {

        cudaMallocHost(&rangesInGPU, sizeof(SDL::objectRanges));
        #ifdef Explicit_Hit
    	  createRangesInExplicitMemory(*rangesInGPU, nModules,stream,nLowerModules); //unclear why but this has to be 2*loopsize to avoid crashing later (reported in tracklet allocation). seems to do with nHits values as well. this allows nhits to be set to the correct value of loopsize to get correct results without crashing. still beats the "max hits" so i think this is fine.
        #else
        createRangesInUnifiedMemory(*rangesInGPU,nModules,stream,nLowerModules);
        #endif
    resetObjectsInModule();
    }
    if(hitsInGPU == nullptr)
    {

        cudaMallocHost(&hitsInGPU, sizeof(SDL::hits));
        #ifdef Explicit_Hit
    	  createHitsInExplicitMemory(*hitsInGPU, 2*loopsize,stream); //unclear why but this has to be 2*loopsize to avoid crashing later (reported in tracklet allocation). seems to do with nHits values as well. this allows nhits to be set to the correct value of loopsize to get correct results without crashing. still beats the "max hits" so i think this is fine.
        #else
        createHitsInUnifiedMemory(*hitsInGPU,2*loopsize,0,stream);
        #endif
    }
cudaStreamSynchronize(stream);


    float* host_x;// = &x[0]; // convert from std::vector to host array easily since vectors are ordered
    float* host_y;// = &y[0];
    float* host_z;// = &z[0];
    unsigned int* host_detId;// = &detId[0];
    unsigned int* host_idxs;// = &idxInNtuple[0];
    cudaMallocHost(&host_x,sizeof(float)*loopsize);
    cudaMallocHost(&host_y,sizeof(float)*loopsize);
    cudaMallocHost(&host_z,sizeof(float)*loopsize);
    cudaMallocHost(&host_detId,sizeof(unsigned int)*loopsize);
    cudaMallocHost(&host_idxs,sizeof(unsigned int)*loopsize);

    //float* host_x = &x[0]; // convert from std::vector to host array easily since vectors are ordered
    //float* host_y = &y[0];
    //float* host_z = &z[0];
    //unsigned int* host_detId = &detId[0];
    //unsigned int* host_idxs = &idxInNtuple[0];

    float* host_phis;
    float* host_etas;
    unsigned int* host_moduleIndex;
    float* host_rts;
    //float* host_idxs;
    float* host_highEdgeXs;
    float* host_highEdgeYs;
    float* host_lowEdgeXs;
    float* host_lowEdgeYs;
    cudaMallocHost(&host_moduleIndex,sizeof(unsigned int)*loopsize);
    cudaMallocHost(&host_phis,sizeof(float)*loopsize);
    cudaMallocHost(&host_etas,sizeof(float)*loopsize);
    cudaMallocHost(&host_rts,sizeof(float)*loopsize);
    //cudaMallocHost(&host_idxs,sizeof(unsigned int)*loopsize);
    cudaMallocHost(&host_highEdgeXs,sizeof(float)*loopsize);
    cudaMallocHost(&host_highEdgeYs,sizeof(float)*loopsize);
    cudaMallocHost(&host_lowEdgeXs,sizeof(float)*loopsize);
    cudaMallocHost(&host_lowEdgeYs,sizeof(float)*loopsize);


    short* module_layers;
    short* module_subdet;
    int* module_hitRanges;
    ModuleType* module_moduleType;
    cudaMallocHost(&module_layers,sizeof(short)*nModules);
    cudaMallocHost(&module_subdet,sizeof(short)*nModules);
    cudaMallocHost(&module_hitRanges,sizeof(int)*2*nModules);
    cudaMallocHost(&module_moduleType,sizeof(ModuleType)*nModules);
    cudaMemcpyAsync(module_layers,modulesInGPU->layers,nModules*sizeof(short),cudaMemcpyDeviceToHost,stream);
    cudaMemcpyAsync(module_subdet,modulesInGPU->subdets,nModules*sizeof(short),cudaMemcpyDeviceToHost,stream);
    cudaMemcpyAsync(module_hitRanges,rangesInGPU->hitRanges,nModules*2*sizeof(int),cudaMemcpyDeviceToHost,stream);
    cudaMemcpyAsync(module_moduleType,modulesInGPU->moduleType,nModules*sizeof(ModuleType),cudaMemcpyDeviceToHost,stream);
cudaStreamSynchronize(stream);


  for (int ihit=0; ihit<loopsize;ihit++){
    host_x[ihit] = x.at(ihit); // convert from std::vector to host array easily since vectors are ordered
    host_y[ihit] = y.at(ihit);
    host_z[ihit] = z.at(ihit);
    host_detId[ihit] = detId.at(ihit);
    host_idxs[ihit] = idxInNtuple.at(ihit);

    unsigned int moduleLayer = module_layers[(*detIdToIndex)[host_detId[ihit]]];
    unsigned int subdet = module_subdet[(*detIdToIndex)[host_detId[ihit]]];
    host_moduleIndex[ihit] = (*detIdToIndex)[host_detId[ihit]];


      host_rts[ihit] = sqrt(host_x[ihit]*host_x[ihit] + host_y[ihit]*host_y[ihit]);
      host_phis[ihit] = phi(host_x[ihit],host_y[ihit],host_z[ihit]);
      host_etas[ihit] = ((host_z[ihit]>0)-(host_z[ihit]<0))* std::acosh(sqrt(host_x[ihit]*host_x[ihit]+host_y[ihit]*host_y[ihit]+host_z[ihit]*host_z[ihit])/host_rts[ihit]);
//// This part i think has a race condition. so this is not run in parallel.
      unsigned int this_index = host_moduleIndex[ihit];
      if(module_subdet[this_index] == Endcap && module_moduleType[this_index] == TwoS)
      {
          float xhigh, yhigh, xlow, ylow;
          getEdgeHits(host_detId[ihit],host_x[ihit],host_y[ihit],xhigh,yhigh,xlow,ylow);
          host_highEdgeXs[ihit] = xhigh;
          host_highEdgeYs[ihit] = yhigh;
          host_lowEdgeXs[ihit] = xlow;
          host_lowEdgeYs[ihit] = ylow;

      }

      //set the hit ranges appropriately in the modules struct

      ////start the index rolling if the module is encountered for the first time
      ////always update the end index
      //modulesInGPU->hitRanges[this_index * 2 + 1] = ihit;
      //start the index rolling if the module is encountered for the first time
      if(module_hitRanges[this_index * 2] == -1)
      {
          module_hitRanges[this_index * 2] = ihit;
      }
      //always update the end index
      module_hitRanges[this_index * 2 + 1] = ihit;

  }
//simply copy the host arrays to the hitsInGPU struct
    cudaMemcpyAsync(hitsInGPU->xs,host_x,loopsize*sizeof(float),cudaMemcpyHostToDevice,stream);
    cudaMemcpyAsync(hitsInGPU->ys,host_y,loopsize*sizeof(float),cudaMemcpyHostToDevice,stream);
    cudaMemcpyAsync(hitsInGPU->zs,host_z,loopsize*sizeof(float),cudaMemcpyHostToDevice,stream);
    cudaMemcpyAsync(hitsInGPU->rts,host_rts,loopsize*sizeof(float),cudaMemcpyHostToDevice,stream);
    cudaMemcpyAsync(hitsInGPU->phis,host_phis,loopsize*sizeof(float),cudaMemcpyHostToDevice,stream);
    cudaMemcpyAsync(hitsInGPU->etas,host_etas,loopsize*sizeof(float),cudaMemcpyHostToDevice,stream);
    cudaMemcpyAsync(hitsInGPU->idxs,host_idxs,loopsize*sizeof(unsigned int),cudaMemcpyHostToDevice,stream);
    cudaMemcpyAsync(hitsInGPU->moduleIndices,host_moduleIndex,loopsize*sizeof(unsigned int),cudaMemcpyHostToDevice,stream);
    cudaMemcpyAsync(hitsInGPU->highEdgeXs,host_highEdgeXs,loopsize*sizeof(float),cudaMemcpyHostToDevice,stream);
    cudaMemcpyAsync(hitsInGPU->highEdgeYs,host_highEdgeYs,loopsize*sizeof(float),cudaMemcpyHostToDevice,stream);
    cudaMemcpyAsync(hitsInGPU->lowEdgeXs,host_lowEdgeXs,loopsize*sizeof(float),cudaMemcpyHostToDevice,stream);
    cudaMemcpyAsync(hitsInGPU->lowEdgeYs,host_lowEdgeYs,loopsize*sizeof(float),cudaMemcpyHostToDevice,stream);
    cudaMemcpyAsync(hitsInGPU->nHits,&loopsize,sizeof(unsigned int),cudaMemcpyHostToDevice,stream);// value can't correctly be set in hit allocation
    cudaMemcpyAsync(rangesInGPU->hitRanges,module_hitRanges,nModules*2*sizeof(int),cudaMemcpyHostToDevice,stream);// value can't correctly be set in hit allocation
cudaStreamSynchronize(stream);

    cudaFreeHost(host_rts);
    //cudaFreeHost(host_idxs);
    cudaFreeHost(host_phis);
    cudaFreeHost(host_etas);
    cudaFreeHost(host_moduleIndex);
    cudaFreeHost(host_highEdgeXs);
    cudaFreeHost(host_highEdgeYs);
    cudaFreeHost(host_lowEdgeXs);
    cudaFreeHost(host_lowEdgeYs);
    cudaFreeHost(module_layers);
    cudaFreeHost(module_subdet);
    cudaFreeHost(module_hitRanges);
    cudaFreeHost(module_moduleType);
    cudaFreeHost(host_x);
    cudaFreeHost(host_y);
    cudaFreeHost(host_z);
    cudaFreeHost(host_detId);
    cudaFreeHost(host_idxs);

}
__global__ void addPixelSegmentToEventKernel(unsigned int* hitIndices0,unsigned int* hitIndices1,unsigned int* hitIndices2,unsigned int* hitIndices3, float* dPhiChange, float* ptIn, float* ptErr, float* px, float* py, float* pz, float* eta, float* etaErr,float* phi, unsigned int pixelModuleIndex, struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU,const int size, int* superbin, int* pixelType, short* isQuad)
{

    for( int tid = blockIdx.x * blockDim.x + threadIdx.x; tid < size; tid += blockDim.x*gridDim.x)
    {

      unsigned int innerMDIndex = pixelModuleIndex * N_MAX_MD_PER_MODULES + 2*(tid);
      unsigned int outerMDIndex = pixelModuleIndex * N_MAX_MD_PER_MODULES + 2*(tid) +1;
      unsigned int pixelSegmentIndex = pixelModuleIndex * N_MAX_SEGMENTS_PER_MODULE + tid;

#ifdef CUT_VALUE_DEBUG
      addMDToMemory(mdsInGPU, hitsInGPU, modulesInGPU, hitIndices0[tid], hitIndices1[tid], pixelModuleIndex, 0,0,0,0,0,0,0,0,0,0,0,0,0,innerMDIndex);
      addMDToMemory(mdsInGPU, hitsInGPU, modulesInGPU, hitIndices2[tid], hitIndices3[tid], pixelModuleIndex, 0,0,0,0,0,0,0,0,0,0,0,0,0,outerMDIndex);
#else
      addMDToMemory(mdsInGPU, hitsInGPU, modulesInGPU, hitIndices0[tid], hitIndices1[tid], pixelModuleIndex, 0,0,0,0,0,0,0,0,0,innerMDIndex);
      addMDToMemory(mdsInGPU, hitsInGPU, modulesInGPU, hitIndices2[tid], hitIndices3[tid], pixelModuleIndex, 0,0,0,0,0,0,0,0,0,outerMDIndex);
#endif

    int hits1[4];
    hits1[0] = hitsInGPU.idxs[mdsInGPU.hitIndices[2*innerMDIndex]];
    hits1[1] = hitsInGPU.idxs[mdsInGPU.hitIndices[2*outerMDIndex]];
    hits1[2] = hitsInGPU.idxs[mdsInGPU.hitIndices[2*innerMDIndex+1]];
    hits1[3] = hitsInGPU.idxs[mdsInGPU.hitIndices[2*outerMDIndex+1]];
    float rsum=0, zsum=0, r2sum=0,rzsum=0;
    for(int i =0; i < 4; i++)
    {
        rsum += hitsInGPU.rts[hits1[i]];
        zsum += hitsInGPU.zs[hits1[i]];
        r2sum += hitsInGPU.rts[hits1[i]]*hitsInGPU.rts[hits1[i]];
        rzsum += hitsInGPU.rts[hits1[i]]*hitsInGPU.zs[hits1[i]];
    }
    float slope_lsq = (4*rzsum - rsum*zsum)/(4*r2sum-rsum*rsum);
    float b = (r2sum*zsum-rsum*rzsum)/(r2sum*4-rsum*rsum);
    float score_lsq=0;
    for( int i=0; i <4; i++)
    {
        float z = hitsInGPU.zs[hits1[i]];
        float r = hitsInGPU.rts[hits1[i]];
        float var_lsq = slope_lsq*(r)+b - z;
        score_lsq += abs(var_lsq);//(var_lsq*var_lsq) / (err*err);
    }
      addPixelSegmentToMemory(segmentsInGPU, mdsInGPU, hitsInGPU, modulesInGPU, innerMDIndex, outerMDIndex, pixelModuleIndex, hitIndices0[tid], hitIndices2[tid], dPhiChange[tid], ptIn[tid], ptErr[tid], px[tid], py[tid], pz[tid], etaErr[tid], eta[tid], phi[tid], pixelSegmentIndex, tid, superbin[tid], pixelType[tid],isQuad[tid],score_lsq);
    }
}
void SDL::Event::addPixelSegmentToEvent(std::vector<unsigned int> hitIndices0,std::vector<unsigned int> hitIndices1,std::vector<unsigned int> hitIndices2,std::vector<unsigned int> hitIndices3, std::vector<float> dPhiChange, std::vector<float> ptIn, std::vector<float> ptErr, std::vector<float> px, std::vector<float> py, std::vector<float> pz, std::vector<float> eta, std::vector<float> etaErr, std::vector<float> phi, std::vector<int> superbin, std::vector<int> pixelType, std::vector<short> isQuad)
{
    if(mdsInGPU == nullptr)
    {
        cudaMallocHost(&mdsInGPU, sizeof(SDL::miniDoublets));
#ifdef Explicit_MD
    	createMDsInExplicitMemory(*mdsInGPU, N_MAX_MD_PER_MODULES, nModules, N_MAX_PIXEL_MD_PER_MODULES,stream);
#else
    	createMDsInUnifiedMemory(*mdsInGPU, N_MAX_MD_PER_MODULES, nModules, N_MAX_PIXEL_MD_PER_MODULES,stream);
#endif
    }
    if(segmentsInGPU == nullptr)
    {
        cudaMallocHost(&segmentsInGPU, sizeof(SDL::segments));
#ifdef Explicit_Seg
        createSegmentsInExplicitMemory(*segmentsInGPU, N_MAX_SEGMENTS_PER_MODULE, nModules, N_MAX_PIXEL_SEGMENTS_PER_MODULE,stream);
#else
        createSegmentsInUnifiedMemory(*segmentsInGPU, N_MAX_SEGMENTS_PER_MODULE, nModules, N_MAX_PIXEL_SEGMENTS_PER_MODULE,stream);
#endif
    }
    cudaStreamSynchronize(stream);
    const int size = ptIn.size();
    unsigned int pixelModuleIndex = (*detIdToIndex)[1];
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
    int* superbin_host = &superbin[0];
    int* pixelType_host = &pixelType[0];
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
    int* superbin_dev;
    int* pixelType_dev;
    short* isQuad_dev;

    //cudaMallocAsync(&hitIndices0_dev,size*sizeof(unsigned int),stream);
    //cudaMallocAsync(&hitIndices1_dev,size*sizeof(unsigned int),stream);
    //cudaMallocAsync(&hitIndices2_dev,size*sizeof(unsigned int),stream);
    //cudaMallocAsync(&hitIndices3_dev,size*sizeof(unsigned int),stream);
    //cudaMallocAsync(&dPhiChange_dev,size*sizeof(unsigned int),stream);
    //cudaMallocAsync(&ptIn_dev,size*sizeof(unsigned int),stream);
    //cudaMallocAsync(&ptErr_dev,size*sizeof(unsigned int),stream);
    //cudaMallocAsync(&px_dev,size*sizeof(unsigned int),stream);
    //cudaMallocAsync(&py_dev,size*sizeof(unsigned int),stream);
    //cudaMallocAsync(&pz_dev,size*sizeof(unsigned int),stream);
    //cudaMallocAsync(&etaErr_dev,size*sizeof(unsigned int),stream);
    //cudaMallocAsync(&eta_dev, size*sizeof(unsigned int),stream);
    //cudaMallocAsync(&phi_dev, size*sizeof(unsigned int),stream);
    //cudaMallocAsync(&superbin_dev,size*sizeof(int),stream);
    //cudaMallocAsync(&pixelType_dev,size*sizeof(int),stream);
    //cudaMallocAsync(&isQuad_dev,size*sizeof(short),stream);
    cudaMalloc(&hitIndices0_dev,size*sizeof(unsigned int));
    cudaMalloc(&hitIndices1_dev,size*sizeof(unsigned int));
    cudaMalloc(&hitIndices2_dev,size*sizeof(unsigned int));
    cudaMalloc(&hitIndices3_dev,size*sizeof(unsigned int));
    cudaMalloc(&dPhiChange_dev,size*sizeof(unsigned int));
    cudaMalloc(&ptIn_dev,size*sizeof(unsigned int));
    cudaMalloc(&ptErr_dev,size*sizeof(unsigned int));
    cudaMalloc(&px_dev,size*sizeof(unsigned int));
    cudaMalloc(&py_dev,size*sizeof(unsigned int));
    cudaMalloc(&pz_dev,size*sizeof(unsigned int));
    cudaMalloc(&etaErr_dev,size*sizeof(unsigned int));
    cudaMalloc(&eta_dev, size*sizeof(unsigned int));
    cudaMalloc(&phi_dev, size*sizeof(unsigned int));
    cudaMalloc(&superbin_dev,size*sizeof(int));
    cudaMalloc(&pixelType_dev,size*sizeof(int));
    cudaMalloc(&isQuad_dev,size*sizeof(short));

    cudaMemcpyAsync(hitIndices0_dev,hitIndices0_host,size*sizeof(unsigned int),cudaMemcpyHostToDevice,stream);
    cudaMemcpyAsync(hitIndices1_dev,hitIndices1_host,size*sizeof(unsigned int),cudaMemcpyHostToDevice,stream);
    cudaMemcpyAsync(hitIndices2_dev,hitIndices2_host,size*sizeof(unsigned int),cudaMemcpyHostToDevice,stream);
    cudaMemcpyAsync(hitIndices3_dev,hitIndices3_host,size*sizeof(unsigned int),cudaMemcpyHostToDevice,stream);
    cudaMemcpyAsync(dPhiChange_dev,dPhiChange_host,size*sizeof(unsigned int),cudaMemcpyHostToDevice,stream);
    cudaMemcpyAsync(ptIn_dev,ptIn_host,size*sizeof(unsigned int),cudaMemcpyHostToDevice,stream);
    cudaMemcpyAsync(ptErr_dev,ptErr_host,size*sizeof(unsigned int),cudaMemcpyHostToDevice,stream);
    cudaMemcpyAsync(px_dev,px_host,size*sizeof(unsigned int),cudaMemcpyHostToDevice,stream);
    cudaMemcpyAsync(py_dev,py_host,size*sizeof(unsigned int),cudaMemcpyHostToDevice,stream);
    cudaMemcpyAsync(pz_dev,pz_host,size*sizeof(unsigned int),cudaMemcpyHostToDevice,stream);
    cudaMemcpyAsync(etaErr_dev,etaErr_host,size*sizeof(unsigned int),cudaMemcpyHostToDevice,stream);
    cudaMemcpyAsync(eta_dev, eta_host, size*sizeof(unsigned int),cudaMemcpyHostToDevice,stream);
    cudaMemcpyAsync(phi_dev, phi_host, size*sizeof(unsigned int),cudaMemcpyHostToDevice,stream);
    cudaMemcpyAsync(superbin_dev,superbin_host,size*sizeof(int),cudaMemcpyHostToDevice,stream);
    cudaMemcpyAsync(pixelType_dev,pixelType_host,size*sizeof(int),cudaMemcpyHostToDevice,stream);
    cudaMemcpyAsync(isQuad_dev,isQuad_host,size*sizeof(short),cudaMemcpyHostToDevice,stream);
cudaStreamSynchronize(stream);

    unsigned int nThreads = 256;
    unsigned int nBlocks =  MAX_BLOCKS;//size % nThreads == 0 ? size/nThreads : size/nThreads + 1;

    addPixelSegmentToEventKernel<<<nBlocks,nThreads,0,stream>>>(hitIndices0_dev,hitIndices1_dev,hitIndices2_dev,hitIndices3_dev,dPhiChange_dev,ptIn_dev,ptErr_dev,px_dev,py_dev,pz_dev,eta_dev, etaErr_dev, phi_dev, pixelModuleIndex, *modulesInGPU,*hitsInGPU,*mdsInGPU,*segmentsInGPU,size, superbin_dev, pixelType_dev,isQuad_dev);

   //cudaDeviceSynchronize();
   cudaStreamSynchronize(stream);

   cudaMemcpyAsync(&(segmentsInGPU->nSegments)[pixelModuleIndex], &size, sizeof(unsigned int), cudaMemcpyHostToDevice,stream);
   unsigned int mdSize = 2 * size;
   cudaMemcpyAsync(&(mdsInGPU->nMDs)[pixelModuleIndex], &mdSize, sizeof(unsigned int), cudaMemcpyHostToDevice,stream);
cudaStreamSynchronize(stream);

    //cudaFreeAsync(hitIndices0_dev,stream);
    //cudaFreeAsync(hitIndices1_dev,stream);
    //cudaFreeAsync(hitIndices2_dev,stream);
    //cudaFreeAsync(hitIndices3_dev,stream);
    //cudaFreeAsync(dPhiChange_dev,stream);
    //cudaFreeAsync(ptIn_dev,stream);
    //cudaFreeAsync(ptErr_dev,stream);
    //cudaFreeAsync(px_dev,stream);
    //cudaFreeAsync(py_dev,stream);
    //cudaFreeAsync(pz_dev,stream);
    //cudaFreeAsync(etaErr_dev,stream);
    //cudaFreeAsync(eta_dev,stream);
    //cudaFreeAsync(phi_dev,stream);
    //cudaFreeAsync(superbin_dev,stream);
    //cudaFreeAsync(pixelType_dev,stream);
    //cudaFreeAsync(isQuad_dev,stream);
  
    cudaFree(hitIndices0_dev);
    cudaFree(hitIndices1_dev);
    cudaFree(hitIndices2_dev);
    cudaFree(hitIndices3_dev);
    cudaFree(dPhiChange_dev);
    cudaFree(ptIn_dev);
    cudaFree(ptErr_dev);
    cudaFree(px_dev);
    cudaFree(py_dev);
    cudaFree(pz_dev);
    cudaFree(etaErr_dev);
    cudaFree(eta_dev);
    cudaFree(phi_dev);
    cudaFree(superbin_dev);
    cudaFree(pixelType_dev);
    cudaFree(isQuad_dev);
  
cudaStreamSynchronize(stream);
}

void SDL::Event::addMiniDoubletsToEvent()
{
    unsigned int idx;
    for(unsigned int i = 0; i<*(SDL::modulesInGPU->nLowerModules); i++)
    {
        idx = SDL::modulesInGPU->lowerModuleIndices[i];
        if(mdsInGPU->nMDs[idx] == 0 or rangesInGPU->hitRanges[idx * 2] == -1)
        {
            rangesInGPU->mdRanges[idx * 2] = -1;
            rangesInGPU->mdRanges[idx * 2 + 1] = -1;
        }
        else
        {
            rangesInGPU->mdRanges[idx * 2] = idx * N_MAX_MD_PER_MODULES;
            rangesInGPU->mdRanges[idx * 2 + 1] = (idx * N_MAX_MD_PER_MODULES) + mdsInGPU->nMDs[idx] - 1;

            if(modulesInGPU->subdets[idx] == Barrel)
            {
                n_minidoublets_by_layer_barrel_[modulesInGPU->layers[idx] -1] += mdsInGPU->nMDs[idx];
            }
            else
            {
                n_minidoublets_by_layer_endcap_[modulesInGPU->layers[idx] - 1] += mdsInGPU->nMDs[idx];
            }

        }
    }
}
void SDL::Event::addMiniDoubletsToEventExplicit()
{
unsigned int nLowerModules;
cudaMemcpyAsync(&nLowerModules,modulesInGPU->nLowerModules,sizeof(unsigned int),cudaMemcpyDeviceToHost,stream);
unsigned int* nMDsCPU;
cudaMallocHost(&nMDsCPU, nModules * sizeof(unsigned int));
cudaMemcpyAsync(nMDsCPU,mdsInGPU->nMDs,nModules*sizeof(unsigned int),cudaMemcpyDeviceToHost,stream);

short* module_subdets;
cudaMallocHost(&module_subdets, nModules* sizeof(short));
cudaMemcpyAsync(module_subdets,modulesInGPU->subdets,nModules*sizeof(short),cudaMemcpyDeviceToHost,stream);
unsigned int* module_lowerModuleIndices;
cudaMallocHost(&module_lowerModuleIndices, (nLowerModules +1)* sizeof(unsigned int));
cudaMemcpyAsync(module_lowerModuleIndices,modulesInGPU->lowerModuleIndices,(nLowerModules+1)*sizeof(unsigned int),cudaMemcpyDeviceToHost,stream);
int* module_mdRanges;
cudaMallocHost(&module_mdRanges, nModules* 2*sizeof(int));
cudaMemcpyAsync(module_mdRanges,rangesInGPU->mdRanges,nModules*2*sizeof(int),cudaMemcpyDeviceToHost,stream);
short* module_layers;
cudaMallocHost(&module_layers, nModules * sizeof(short));
cudaMemcpyAsync(module_layers,modulesInGPU->layers,nModules*sizeof(short),cudaMemcpyDeviceToHost,stream);
int* module_hitRanges;
cudaMallocHost(&module_hitRanges, nModules* 2*sizeof(int));
cudaMemcpyAsync(module_hitRanges,rangesInGPU->hitRanges,nModules*2*sizeof(int),cudaMemcpyDeviceToHost,stream);
cudaStreamSynchronize(stream);

    unsigned int idx;
    for(unsigned int i = 0; i<nLowerModules; i++)
    {
        idx = module_lowerModuleIndices[i];
        if(nMDsCPU[idx] == 0 or module_hitRanges[idx * 2] == -1)
        {
            module_mdRanges[idx * 2] = -1;
            module_mdRanges[idx * 2 + 1] = -1;
        }
        else
        {
            module_mdRanges[idx * 2] = idx * N_MAX_MD_PER_MODULES;
            module_mdRanges[idx * 2 + 1] = (idx * N_MAX_MD_PER_MODULES) + nMDsCPU[idx] - 1;

            if(module_subdets[idx] == Barrel)
            {
                n_minidoublets_by_layer_barrel_[module_layers[idx] -1] += nMDsCPU[idx];
            }
            else
            {
                n_minidoublets_by_layer_endcap_[module_layers[idx] - 1] += nMDsCPU[idx];
            }

        }
    }
cudaMemcpyAsync(rangesInGPU->mdRanges,module_mdRanges,nModules*2*sizeof(int),cudaMemcpyHostToDevice,stream);
cudaStreamSynchronize(stream);
cudaFreeHost(nMDsCPU);
cudaFreeHost(module_subdets);
cudaFreeHost(module_lowerModuleIndices);
cudaFreeHost(module_mdRanges);
cudaFreeHost(module_layers);
cudaFreeHost(module_hitRanges);
}

void SDL::Event::addSegmentsToEvent()
{
    unsigned int idx;
    for(unsigned int i = 0; i<*(SDL::modulesInGPU->nLowerModules); i++)
    {
        idx = SDL::modulesInGPU->lowerModuleIndices[i];
        if(segmentsInGPU->nSegments[idx] == 0)
        {
            rangesInGPU->segmentRanges[idx * 2] = -1;
            rangesInGPU->segmentRanges[idx * 2 + 1] = -1;
        }
        else
        {
            rangesInGPU->segmentRanges[idx * 2] = idx * N_MAX_SEGMENTS_PER_MODULE;
            rangesInGPU->segmentRanges[idx * 2 + 1] = idx * N_MAX_SEGMENTS_PER_MODULE + segmentsInGPU->nSegments[idx] - 1;

            if(modulesInGPU->subdets[idx] == Barrel)
            {

                n_segments_by_layer_barrel_[modulesInGPU->layers[idx] - 1] += segmentsInGPU->nSegments[idx];
            }
            else
            {
                n_segments_by_layer_endcap_[modulesInGPU->layers[idx] -1] += segmentsInGPU->nSegments[idx];
            }
        }
    }
}
void SDL::Event::addSegmentsToEventExplicit()
{
unsigned int nLowerModules;
cudaMemcpyAsync(&nLowerModules,modulesInGPU->nLowerModules,sizeof(unsigned int),cudaMemcpyDeviceToHost,stream);

unsigned int* nSegmentsCPU;
cudaMallocHost(&nSegmentsCPU, nModules * sizeof(unsigned int));
cudaMemcpyAsync(nSegmentsCPU,segmentsInGPU->nSegments,nModules*sizeof(unsigned int),cudaMemcpyDeviceToHost,stream);

short* module_subdets;
cudaMallocHost(&module_subdets, nModules* sizeof(short));
cudaMemcpyAsync(module_subdets,modulesInGPU->subdets,nModules*sizeof(short),cudaMemcpyDeviceToHost,stream);
unsigned int* module_lowerModuleIndices;
cudaMallocHost(&module_lowerModuleIndices, (nLowerModules +1)* sizeof(unsigned int));
cudaMemcpyAsync(module_lowerModuleIndices,modulesInGPU->lowerModuleIndices,(nLowerModules+1)*sizeof(unsigned int),cudaMemcpyDeviceToHost,stream);
int* module_segmentRanges;
cudaMallocHost(&module_segmentRanges, nModules* 2*sizeof(int));
cudaMemcpyAsync(module_segmentRanges,rangesInGPU->segmentRanges,nModules*2*sizeof(int),cudaMemcpyDeviceToHost,stream);
short* module_layers;
cudaMallocHost(&module_layers, nModules * sizeof(short));
cudaMemcpyAsync(module_layers,modulesInGPU->layers,nModules*sizeof(short),cudaMemcpyDeviceToHost,stream);
cudaStreamSynchronize(stream);
    unsigned int idx;
    for(unsigned int i = 0; i<nLowerModules; i++)
    {
        printf("i: %d", i);
        idx = module_lowerModuleIndices[i];
        if(nSegmentsCPU[idx] == 0)
        {
            module_segmentRanges[idx * 2] = -1;
            module_segmentRanges[idx * 2 + 1] = -1;
        }
        else
        {
            module_segmentRanges[idx * 2] = idx * N_MAX_SEGMENTS_PER_MODULE;
            module_segmentRanges[idx * 2 + 1] = idx * N_MAX_SEGMENTS_PER_MODULE + nSegmentsCPU[idx] - 1;

            if(module_subdets[idx] == Barrel)
            {

                n_segments_by_layer_barrel_[module_layers[idx] - 1] += nSegmentsCPU[idx];
            }
            else
            {
                n_segments_by_layer_endcap_[module_layers[idx] -1] += nSegmentsCPU[idx];
            }
        }
    }
cudaFreeHost(nSegmentsCPU);
cudaFreeHost(module_subdets);
cudaFreeHost(module_lowerModuleIndices);
cudaFreeHost(module_segmentRanges);
cudaFreeHost(module_layers);
}

void SDL::Event::createMiniDoublets()
{
    unsigned int nModules;
    cudaMemcpyAsync(&nModules,modulesInGPU->nModules,sizeof(unsigned int),cudaMemcpyDeviceToHost,stream);
    unsigned int nLowerModules;
    cudaMemcpyAsync(&nLowerModules,modulesInGPU->nLowerModules,sizeof(unsigned int),cudaMemcpyDeviceToHost,stream);
    cudaStreamSynchronize(stream);
    if(mdsInGPU == nullptr)
    {
        cudaMallocHost(&mdsInGPU, sizeof(SDL::miniDoublets));
#ifdef Explicit_MD
        //FIXME: Add memory locations for pixel MDs
    	createMDsInExplicitMemory(*mdsInGPU, N_MAX_MD_PER_MODULES, nModules, N_MAX_PIXEL_MD_PER_MODULES,stream);
#else
    	createMDsInUnifiedMemory(*mdsInGPU, N_MAX_MD_PER_MODULES, nModules, N_MAX_PIXEL_MD_PER_MODULES,stream);
#endif
    }

    //cudaDeviceSynchronize();

    cudaStreamSynchronize(stream);

    int maxThreadsPerModule=0;
    #ifdef Explicit_Module
    unsigned int* module_lowerModuleIndices;
    cudaMallocHost(&module_lowerModuleIndices, (nLowerModules +1)* sizeof(unsigned int));
    cudaMemcpyAsync(module_lowerModuleIndices,modulesInGPU->lowerModuleIndices,(nLowerModules+1)*sizeof(unsigned int),cudaMemcpyDeviceToHost,stream);
    int* module_hitRanges;
    cudaMallocHost(&module_hitRanges, nModules* 2*sizeof(int));
    cudaMemcpyAsync(module_hitRanges,rangesInGPU->hitRanges,nModules*2*sizeof(int),cudaMemcpyDeviceToHost,stream);
    bool* module_isLower;
    cudaMallocHost(&module_isLower, nModules*sizeof(bool));
    cudaMemcpyAsync(module_isLower,modulesInGPU->isLower,nModules*sizeof(bool),cudaMemcpyDeviceToHost,stream);
    bool* module_isInverted;
    cudaMallocHost(&module_isInverted, nModules*sizeof(bool));
    cudaMemcpyAsync(module_isInverted,modulesInGPU->isInverted,nModules*sizeof(bool),cudaMemcpyDeviceToHost,stream);
cudaStreamSynchronize(stream);

    for (int i=0; i<nLowerModules; i++) 
    {
        int lowerModuleIndex = module_lowerModuleIndices[i];
        int upperModuleIndex = modulesInGPU->partnerModuleIndexExplicit(lowerModuleIndex,module_isLower[lowerModuleIndex],module_isInverted[lowerModuleIndex]);
        int lowerHitRanges = module_hitRanges[lowerModuleIndex*2];
        int upperHitRanges = module_hitRanges[upperModuleIndex*2];
        if(lowerHitRanges!=-1 && upperHitRanges!=-1) 
        {
            unsigned int nLowerHits = module_hitRanges[lowerModuleIndex * 2 + 1] - lowerHitRanges + 1;
            unsigned int nUpperHits = module_hitRanges[upperModuleIndex * 2 + 1] - upperHitRanges + 1;
            maxThreadsPerModule = maxThreadsPerModule > (nLowerHits*nUpperHits) ? maxThreadsPerModule : nLowerHits*nUpperHits;
        }
    }
    cudaFreeHost(module_lowerModuleIndices);
    cudaFreeHost(module_hitRanges);
    cudaFreeHost(module_isLower);
    cudaFreeHost(module_isInverted);
    #else
    for (int i=0; i<nLowerModules; i++) 
    {
        int lowerModuleIndex = modulesInGPU->lowerModuleIndices[i];
        int upperModuleIndex = modulesInGPU->partnerModuleIndex(lowerModuleIndex);
        int lowerHitRanges = rangesInGPU->hitRanges[lowerModuleIndex*2];
        int upperHitRanges = rangesInGPU->hitRanges[upperModuleIndex*2];
        if(lowerHitRanges!=-1&&upperHitRanges!=-1) 
        {
            unsigned int nLowerHits = rangesInGPU->hitRanges[lowerModuleIndex * 2 + 1] - lowerHitRanges + 1;
            unsigned int nUpperHits = rangesInGPU->hitRanges[upperModuleIndex * 2 + 1] - upperHitRanges + 1;
            maxThreadsPerModule = maxThreadsPerModule > (nLowerHits*nUpperHits) ? maxThreadsPerModule : nLowerHits*nUpperHits;
        }
    }
    #endif
    //dim3 nThreads(1,128);
    //dim3 nBlocks((nLowerModules % nThreads.x == 0 ? nLowerModules/nThreads.x : nLowerModules/nThreads.x + 1), (maxThreadsPerModule % nThreads.y == 0 ? maxThreadsPerModule/nThreads.y : maxThreadsPerModule/nThreads.y + 1));
    dim3 nThreads(32,32,1);
    dim3 nBlocks(1,MAX_BLOCKS,1);

    createMiniDoubletsInGPU<<<nBlocks,nThreads,0,stream>>>(*modulesInGPU,*hitsInGPU,*mdsInGPU,*rangesInGPU);

    cudaError_t cudaerr = cudaGetLastError(); 
    if(cudaerr != cudaSuccess)
    {
        std::cout<<"sync failed with error : "<<cudaGetErrorString(cudaerr)<<std::endl;
    }
    //cudaDeviceSynchronize();
    cudaStreamSynchronize(stream);

#if defined(AddObjects)
#ifdef Explicit_MD
    addMiniDoubletsToEventExplicit();
#else
    addMiniDoubletsToEvent();
#endif
#endif

}

void SDL::Event::createSegmentsWithModuleMap()
{
    unsigned int nModules;
    cudaMemcpyAsync(&nModules,modulesInGPU->nModules,sizeof(unsigned int),cudaMemcpyDeviceToHost,stream);
cudaStreamSynchronize(stream);
    if(segmentsInGPU == nullptr)
    {
        cudaMallocHost(&segmentsInGPU, sizeof(SDL::segments));
#ifdef Explicit_Seg
        createSegmentsInExplicitMemory(*segmentsInGPU, N_MAX_SEGMENTS_PER_MODULE, nModules, N_MAX_PIXEL_SEGMENTS_PER_MODULE,stream);
#else
        createSegmentsInUnifiedMemory(*segmentsInGPU, N_MAX_SEGMENTS_PER_MODULE, nModules, N_MAX_PIXEL_SEGMENTS_PER_MODULE,stream);
#endif
    }
//    cudaStreamSynchronize(stream);
    unsigned int nLowerModules;
    cudaMemcpyAsync(&nLowerModules,modulesInGPU->nLowerModules,sizeof(unsigned int),cudaMemcpyDeviceToHost,stream);
cudaStreamSynchronize(stream);

    int max_cModules=0;
    int sq_max_nMDs = 0;
    int nonZeroModules = 0;
  #ifdef Explicit_Module
    unsigned int* nMDs;// = (unsigned int*)malloc(nModules*sizeof(unsigned int));
    cudaMallocHost(&nMDs, (nModules +1)* sizeof(unsigned int));
    cudaMemcpyAsync((void *)nMDs, mdsInGPU->nMDs, nModules*sizeof(unsigned int), cudaMemcpyDeviceToHost,stream);

    unsigned int* module_lowerModuleIndices;
    cudaMallocHost(&module_lowerModuleIndices, (nLowerModules +1)* sizeof(unsigned int));
    cudaMemcpyAsync(module_lowerModuleIndices,modulesInGPU->lowerModuleIndices,(nLowerModules+1)*sizeof(unsigned int),cudaMemcpyDeviceToHost,stream);
    unsigned int* module_nConnectedModules;
    cudaMallocHost(&module_nConnectedModules, nModules* sizeof(unsigned int));
    cudaMemcpyAsync(module_nConnectedModules,modulesInGPU->nConnectedModules,nModules*sizeof(unsigned int),cudaMemcpyDeviceToHost,stream);
    unsigned int* module_moduleMap;
    cudaMallocHost(&module_moduleMap, nModules*40* sizeof(unsigned int));
    cudaMemcpyAsync(module_moduleMap,modulesInGPU->moduleMap,nModules*40*sizeof(unsigned int),cudaMemcpyDeviceToHost,stream);
cudaStreamSynchronize(stream);
    for (int i=0; i<nLowerModules; i++) 
    {
        unsigned int innerLowerModuleIndex = module_lowerModuleIndices[i];
        unsigned int nConnectedModules = module_nConnectedModules[innerLowerModuleIndex];
        unsigned int nInnerMDs = nMDs[innerLowerModuleIndex];
      max_cModules = max_cModules > nConnectedModules ? max_cModules : nConnectedModules;
        int limit_local = 0;
        if (nConnectedModules!=0) nonZeroModules++;
        for (int j=0; j<nConnectedModules; j++) 
        {
            int outerLowerModuleIndex = module_moduleMap[innerLowerModuleIndex * MAX_CONNECTED_MODULES + j];
            int nOuterMDs = nMDs[outerLowerModuleIndex];
            int total = nInnerMDs*nOuterMDs;
            limit_local = limit_local > total ? limit_local : total;
        }
        sq_max_nMDs = sq_max_nMDs > limit_local ? sq_max_nMDs : limit_local;
    }
    cudaFreeHost(module_lowerModuleIndices);
    cudaFreeHost(module_nConnectedModules);
    cudaFreeHost(module_moduleMap);
  #else

    //unsigned int nModules = *modulesInGPU->nModules;
    //unsigned int* nMDs = (unsigned int*)malloc(nModules*sizeof(unsigned int));
    unsigned int* nMDs;// = (unsigned int*)malloc(nModules*sizeof(unsigned int));
    cudaMallocHost(&nMDs, (nModules +1)* sizeof(unsigned int));
    cudaMemcpyAsync((void *)nMDs, mdsInGPU->nMDs, nModules*sizeof(unsigned int), cudaMemcpyDeviceToHost,stream);
cudaStreamSynchronize(stream);
    for (int i=0; i<nLowerModules; i++) 
    {
        unsigned int innerLowerModuleIndex = modulesInGPU->lowerModuleIndices[i];
        unsigned int nConnectedModules = modulesInGPU->nConnectedModules[innerLowerModuleIndex];
        unsigned int nInnerMDs = nMDs[innerLowerModuleIndex] > N_MAX_MD_PER_MODULES ? N_MAX_MD_PER_MODULES : nMDs[innerLowerModuleIndex];
      max_cModules = max_cModules > nConnectedModules ? max_cModules : nConnectedModules;
        int limit_local = 0;
        if (nConnectedModules!=0) nonZeroModules++;
        for (int j=0; j<nConnectedModules; j++) 
        {
            int outerLowerModuleIndex = modulesInGPU->moduleMap[innerLowerModuleIndex * MAX_CONNECTED_MODULES + j];
            int nOuterMDs = nMDs[outerLowerModuleIndex] > N_MAX_MD_PER_MODULES ? N_MAX_MD_PER_MODULES : nMDs[outerLowerModuleIndex];
            int total = nInnerMDs*nOuterMDs;
            limit_local = limit_local > total ? limit_local : total;
        }
        sq_max_nMDs = sq_max_nMDs > limit_local ? sq_max_nMDs : limit_local;
    }
  #endif
    //dim3 nThreads(256,1,1);
    //dim3 nBlocks((sq_max_nMDs%nThreads.x==0 ? sq_max_nMDs/nThreads.x : sq_max_nMDs/nThreads.x + 1), (max_cModules%nThreads.y==0 ? max_cModules/nThreads.y : max_cModules/nThreads.y + 1), (nLowerModules%nThreads.z==0 ? nLowerModules/nThreads.z : nLowerModules/nThreads.z + 1));
    dim3 nThreads(32,32,1);
    dim3 nBlocks(1,1,MAX_BLOCKS);

    createSegmentsInGPU<<<nBlocks,nThreads,0,stream>>>(*modulesInGPU, *hitsInGPU, *mdsInGPU, *segmentsInGPU, *rangesInGPU);
    cudaFreeHost(nMDs);
    //free(nMDs);

    cudaError_t cudaerr = cudaGetLastError();
    if(cudaerr != cudaSuccess)
    {
        std::cout<<"sync failed with error : "<<cudaGetErrorString(cudaerr)<<std::endl;
    }
    //cudaDeviceSynchronize();
    cudaStreamSynchronize(stream);
#if defined(AddObjects)
#ifdef Explicit_Seg
    addSegmentsToEventExplicit();
#else
    addSegmentsToEvent();
#endif
#endif

}


void SDL::Event::createTriplets()
{
    unsigned int nLowerModules;
    cudaMemcpyAsync(&nLowerModules,modulesInGPU->nLowerModules,sizeof(unsigned int),cudaMemcpyDeviceToHost,stream);

    if(tripletsInGPU == nullptr)
    {
        cudaMallocHost(&tripletsInGPU, sizeof(SDL::triplets));
#ifdef Explicit_Trips
        createTripletsInExplicitMemory(*tripletsInGPU, N_MAX_TRIPLETS_PER_MODULE, nLowerModules,stream);
#else
        createTripletsInUnifiedMemory(*tripletsInGPU, N_MAX_TRIPLETS_PER_MODULE, nLowerModules,stream);
#endif
    }
    cudaStreamSynchronize(stream);

  #ifdef Explicit_Module
    unsigned int nonZeroModules=0;
    unsigned int max_InnerSeg=0;
    unsigned int *index = (unsigned int*)malloc(nLowerModules*sizeof(unsigned int));
    unsigned int *index_gpu;
    //cudaMallocAsync((void **)&index_gpu, nLowerModules*sizeof(unsigned int),stream);
    cudaMalloc((void **)&index_gpu, nLowerModules*sizeof(unsigned int));
    //unsigned int nModules = *modulesInGPU->nModules;
    unsigned int *nSegments = (unsigned int*)malloc(nModules*sizeof(unsigned int));
    cudaMemcpyAsync((void *)nSegments, segmentsInGPU->nSegments, nModules*sizeof(unsigned int), cudaMemcpyDeviceToHost,stream);
    unsigned int* module_lowerModuleIndices;
    cudaMallocHost(&module_lowerModuleIndices, (nLowerModules +1)* sizeof(unsigned int));
    cudaMemcpyAsync(module_lowerModuleIndices,modulesInGPU->lowerModuleIndices,(nLowerModules+1)*sizeof(unsigned int),cudaMemcpyDeviceToHost,stream);
    unsigned int* module_nConnectedModules;
    cudaMallocHost(&module_nConnectedModules, nModules* sizeof(unsigned int));
    cudaMemcpyAsync(module_nConnectedModules,modulesInGPU->nConnectedModules,nModules*sizeof(unsigned int),cudaMemcpyDeviceToHost,stream);
cudaStreamSynchronize(stream);
    for (int i=0; i<nLowerModules; i++) {
      unsigned int innerLowerModuleIndex = module_lowerModuleIndices[i];
      unsigned int nConnectedModules = module_nConnectedModules[innerLowerModuleIndex];
      unsigned int nInnerSegments = nSegments[innerLowerModuleIndex];
      if (nConnectedModules!=0&&nInnerSegments!=0) {
        index[nonZeroModules] = i;
        nonZeroModules++;
      }
      max_InnerSeg = max_InnerSeg > nInnerSegments ? max_InnerSeg : nInnerSegments;
    }
    cudaFreeHost(module_lowerModuleIndices);
    cudaFreeHost(module_nConnectedModules);
  #else
    unsigned int nonZeroModules=0;
    unsigned int max_InnerSeg=0;
    unsigned int *index = (unsigned int*)malloc(nLowerModules*sizeof(unsigned int));
    unsigned int *index_gpu;
    cudaMalloc((void **)&index_gpu, nLowerModules*sizeof(unsigned int));
    //cudaMallocAsync((void **)&index_gpu, nLowerModules*sizeof(unsigned int),stream);
    unsigned int nModules = *modulesInGPU->nModules;
    unsigned int *nSegments = (unsigned int*)malloc(nModules*sizeof(unsigned int));
    cudaMemcpyAsync((void *)nSegments, segmentsInGPU->nSegments, nModules*sizeof(unsigned int), cudaMemcpyDeviceToHost,stream);
cudaStreamSynchronize(stream);
    for (int i=0; i<nLowerModules; i++) {
      unsigned int innerLowerModuleIndex = modulesInGPU->lowerModuleIndices[i];
      unsigned int nConnectedModules = modulesInGPU->nConnectedModules[innerLowerModuleIndex];
      unsigned int nInnerSegments = nSegments[innerLowerModuleIndex] > N_MAX_SEGMENTS_PER_MODULE ? N_MAX_SEGMENTS_PER_MODULE : nSegments[innerLowerModuleIndex];
      if (nConnectedModules!=0&&nInnerSegments!=0) {
        index[nonZeroModules] = i;
        nonZeroModules++;
      }
      max_InnerSeg = max_InnerSeg > nInnerSegments ? max_InnerSeg : nInnerSegments;
    }
  #endif
    cudaMemcpyAsync(index_gpu, index, nonZeroModules*sizeof(unsigned int), cudaMemcpyHostToDevice,stream);
cudaStreamSynchronize(stream);
    int max_OuterSeg = 0;
    max_OuterSeg = N_MAX_SEGMENTS_PER_MODULE;
    //dim3 nThreads(16,16,1);
    //dim3 nBlocks((max_OuterSeg % nThreads.x == 0 ? max_OuterSeg / nThreads.x : max_OuterSeg / nThreads.x + 1),(max_InnerSeg % nThreads.y == 0 ? max_InnerSeg/nThreads.y : max_InnerSeg/nThreads.y + 1), (nonZeroModules % nThreads.z == 0 ? nonZeroModules/nThreads.z : nonZeroModules/nThreads.z + 1));
    dim3 nThreads(16,64,1);
    dim3 nBlocks(1,1,MAX_BLOCKS);
    createTripletsInGPU<<<nBlocks,nThreads,0,stream>>>(*modulesInGPU, *hitsInGPU, *mdsInGPU, *segmentsInGPU, *tripletsInGPU, index_gpu,nonZeroModules);
    cudaError_t cudaerr =cudaGetLastError();
    if(cudaerr != cudaSuccess)
    {
	    std::cout<<"sync failed with error : "<<cudaGetErrorString(cudaerr)<<std::endl;
    } 
    cudaStreamSynchronize(stream);
    //cudaDeviceSynchronize();
    //cudaThreadSynchronize();
    free(nSegments);
    free(index);
    cudaFree(index_gpu);
    //cudaFreeAsync(index_gpu,stream);

#if defined(AddObjects)
#ifdef Explicit_Trips
    addTripletsToEventExplicit();
#else
    addTripletsToEvent();
#endif
#endif
}

void SDL::Event::createTrackCandidates()
{
    unsigned int nLowerModules;    
    cudaMemcpyAsync(&nLowerModules,modulesInGPU->nLowerModules,sizeof(unsigned int),cudaMemcpyDeviceToHost,stream);

    if(trackCandidatesInGPU == nullptr)
    {
        //printf("did this run twice?\n");
        cudaMallocHost(&trackCandidatesInGPU, sizeof(SDL::trackCandidates));
#ifdef Explicit_Track
        createTrackCandidatesInExplicitMemory(*trackCandidatesInGPU, N_MAX_TRACK_CANDIDATES + N_MAX_PIXEL_TRACK_CANDIDATES,stream);
#else
        createTrackCandidatesInUnifiedMemory(*trackCandidatesInGPU, N_MAX_TRACK_CANDIDATES + N_MAX_PIXEL_TRACK_CANDIDATES,stream);

#endif
    }

#ifdef FINAL_pT3
    //printf("running final state pT3\n");
    unsigned int nThreadsx = 1024;
    unsigned int nBlocksx = MAX_BLOCKS;//(N_MAX_PIXEL_TRIPLETS) % nThreadsx == 0 ? N_MAX_PIXEL_TRIPLETS / nThreadsx : N_MAX_PIXEL_TRIPLETS / nThreadsx + 1;
    addpT3asTrackCandidateInGPU<<<nBlocksx, nThreadsx,0,stream>>>(*modulesInGPU, *pixelTripletsInGPU, *trackCandidatesInGPU, *segmentsInGPU, *pixelQuintupletsInGPU);
    cudaError_t cudaerr_pT3 = cudaGetLastError();
    if(cudaerr_pT3 != cudaSuccess)
    {
        std::cout<<"sync failed with error : "<<cudaGetErrorString(cudaerr_pT3)<<std::endl;
    }cudaStreamSynchronize(stream);
#endif

#ifdef FINAL_T5
    dim3 dupThreads(64,16,1);
    dim3 dupBlocks(1,MAX_BLOCKS,1);
    dim3 nThreads(32,32,1);
    dim3 nBlocks(1,MAX_BLOCKS,1);
    removeDupQuintupletsInGPU<<<dupBlocks,dupThreads,0,stream>>>(*modulesInGPU, *hitsInGPU, *mdsInGPU, *segmentsInGPU, *tripletsInGPU, *quintupletsInGPU,true,*rangesInGPU);
    //cudaDeviceSynchronize();
    cudaStreamSynchronize(stream);
    addT5asTrackCandidateInGPU<<<nBlocks,nThreads,0,stream>>>(*modulesInGPU, *quintupletsInGPU,*trackCandidatesInGPU,*pixelQuintupletsInGPU,*pixelTripletsInGPU,*rangesInGPU);

    cudaError_t cudaerr_T5 =cudaGetLastError(); 
    if(cudaerr_T5 != cudaSuccess)
    {
        std::cout<<"sync failed with error : "<<cudaGetErrorString(cudaerr_T5)<<std::endl;
    }cudaStreamSynchronize(stream);
#endif // final state T5
#ifdef FINAL_pLS
    //printf("Adding pLSs to TC collection\n");
#ifdef DUP_pLS
    //printf("cleaning pixels\n");
    checkHitspLS<<<MAX_BLOCKS,1024,0,stream>>>(*modulesInGPU,*mdsInGPU, *segmentsInGPU, *hitsInGPU, true);
    cudaError_t cudaerrpix = cudaGetLastError();
    if(cudaerrpix != cudaSuccess)
    {
        std::cout<<"sync failed with error : "<<cudaGetErrorString(cudaerrpix)<<std::endl;

    }cudaStreamSynchronize(stream);
#endif  
    unsigned int nThreadsx_pLS = 1024;
    unsigned int nBlocksx_pLS = MAX_BLOCKS;//(20000) % nThreadsx_pLS == 0 ? 20000 / nThreadsx_pLS : 20000 / nThreadsx_pLS + 1;
    addpLSasTrackCandidateInGPU<<<nBlocksx, nThreadsx,0,stream>>>(*modulesInGPU, *pixelTripletsInGPU, *trackCandidatesInGPU, *segmentsInGPU, *pixelQuintupletsInGPU,*mdsInGPU,*hitsInGPU,*quintupletsInGPU);
    cudaError_t cudaerr_pLS = cudaGetLastError();
    if(cudaerr_pLS != cudaSuccess)
    {
        std::cout<<"sync failed with error : "<<cudaGetErrorString(cudaerr_pLS)<<std::endl;
    }cudaStreamSynchronize(stream);
#endif
}

#ifdef TRACK_EXTENSIONS
void SDL::Event::createExtendedTracks()
{
    if(trackExtensionsInGPU == nullptr)
    {
        cudaMallocHost(&trackExtensionsInGPU, sizeof(SDL::trackExtensions));
    }

    unsigned int nTrackCandidates;
    cudaMemcpy(&nTrackCandidates, trackCandidatesInGPU->nTrackCandidates, sizeof(unsigned int), cudaMemcpyDeviceToHost);

#ifdef T3T3_EXTENSIONS
#ifdef Explicit_Extensions
    createTrackExtensionsInExplicitMemory(*trackExtensionsInGPU, nTrackCandidates * N_MAX_TRACK_EXTENSIONS_PER_TC + N_MAX_T3T3_TRACK_EXTENSIONS, nTrackCandidates + 1, stream); 
#else
    createTrackExtensionsInUnifiedMemory(*trackExtensionsInGPU, nTrackCandidates * N_MAX_TRACK_EXTENSIONS_PER_TC + N_MAX_T3T3_TRACK_EXTENSIONS, nTrackCandidates + 1, stream);
#endif
#else
#ifdef Explicit_Extensions
    createTrackExtensionsInExplicitMemory(*trackExtensionsInGPU, nTrackCandidates * N_MAX_TRACK_EXTENSIONS_PER_TC, nTrackCandidates, stream); 
#else
    createTrackExtensionsInUnifiedMemory(*trackExtensionsInGPU, nTrackCandidates * N_MAX_TRACK_EXTENSIONS_PER_TC, nTrackCandidates, stream);
#endif
#endif

    unsigned int nLowerModules;    
    cudaMemcpy(&nLowerModules,modulesInGPU->nLowerModules,sizeof(unsigned int),cudaMemcpyDeviceToHost);
    
    unsigned int *nTriplets;
    cudaMallocHost(&nTriplets, nLowerModules * sizeof(unsigned int));
    cudaMemcpy(nTriplets, tripletsInGPU->nTriplets, nLowerModules * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    /* extremely naive way - 3D grid
     * most of the threads launched here will exit without running
     */
    dim3 nThreads(16,4,4);
    unsigned int maxT3s = *std::max_element(nTriplets, nTriplets + nLowerModules); 
    unsigned int nOverlaps = 3;
    dim3 nBlocks(nTrackCandidates % nThreads.x == 0 ? nTrackCandidates / nThreads.x : nTrackCandidates / nThreads.x + 1, maxT3s % nThreads.y == 0 ? maxT3s / nThreads.y : maxT3s / nThreads.y + 1, nOverlaps % nThreads.z == 0 ? nOverlaps / nThreads.z : nOverlaps / nThreads.z + 1);
    createExtendedTracksInGPU<<<nBlocks,nThreads>>>(*modulesInGPU, *hitsInGPU, *mdsInGPU, *segmentsInGPU, *tripletsInGPU, *pixelTripletsInGPU, *quintupletsInGPU, *pixelQuintupletsInGPU, *trackCandidatesInGPU, *trackExtensionsInGPU);

    cudaError_t cudaerr = cudaDeviceSynchronize();
    if(cudaerr != cudaSuccess)
    {
	    std::cout<<"sync failed with error : "<<cudaGetErrorString(cudaerr)<<std::endl;
    }

#ifdef T3T3_EXTENSIONS
    dim3 nThreadsT3T3(1,16,16);
    dim3 nBlocksT3T3(nLowerModules % nThreadsT3T3.x == 0 ? nLowerModules / nThreadsT3T3.x: nLowerModules / nThreadsT3T3.x + 1, maxT3s % nThreadsT3T3.y == 0 ? maxT3s / nThreadsT3T3.y : maxT3s / nThreadsT3T3.y + 1, maxT3s % nThreadsT3T3.z == 0 ? maxT3s / nThreadsT3T3.z : maxT3s / nThreadsT3T3.z + 1);

    createT3T3ExtendedTracksInGPU<<<nBlocksT3T3, nThreadsT3T3>>>(*modulesInGPU, *hitsInGPU, *mdsInGPU, *segmentsInGPU, *tripletsInGPU, *quintupletsInGPU, *pixelTripletsInGPU, *pixelQuintupletsInGPU, *trackCandidatesInGPU, *trackExtensionsInGPU, nTrackCandidates);

    cudaerr = cudaDeviceSynchronize();
    if(cudaerr != cudaSuccess)
    {
	    std::cout<<"sync failed with error : "<<cudaGetErrorString(cudaerr)<<std::endl;
    }
#endif

    int nThreadsDupCleaning = 512;
    int nBlocksDupCleaning = (nTrackCandidates % nThreadsDupCleaning == 0) ? nTrackCandidates / nThreadsDupCleaning : nTrackCandidates / nThreadsDupCleaning + 1;

    cleanDuplicateExtendedTracks<<<nThreadsDupCleaning, nBlocksDupCleaning>>>(*trackExtensionsInGPU, nTrackCandidates);

    cudaerr = cudaDeviceSynchronize();
    if(cudaerr != cudaSuccess)
    {
	    std::cout<<"sync failed with error : "<<cudaGetErrorString(cudaerr)<<std::endl;
    }

    cudaFreeHost(nTriplets); 
}
#endif

void SDL::Event::createPixelTriplets()
{
    unsigned int nLowerModules;
    cudaMemcpyAsync(&nLowerModules, modulesInGPU->nLowerModules, sizeof(unsigned int), cudaMemcpyDeviceToHost,stream);
    unsigned int nModules;
    cudaMemcpyAsync(&nModules,modulesInGPU->nModules,sizeof(unsigned int),cudaMemcpyDeviceToHost,stream);
    cudaStreamSynchronize(stream);

    if(pixelTripletsInGPU == nullptr)
    {
        cudaMallocHost(&pixelTripletsInGPU, sizeof(SDL::pixelTriplets));
    }
#ifdef Explicit_PT3
    createPixelTripletsInExplicitMemory(*pixelTripletsInGPU, N_MAX_PIXEL_TRIPLETS,stream);
#else
    createPixelTripletsInUnifiedMemory(*pixelTripletsInGPU, N_MAX_PIXEL_TRIPLETS,stream);
#endif

    unsigned int pixelModuleIndex;
    int* superbins;
    int* pixelTypes;
    unsigned int *nTriplets;
    pixelModuleIndex = nModules-1;
    unsigned int nInnerSegments = 0;
    cudaMemcpyAsync(&nInnerSegments, &(segmentsInGPU->nSegments[pixelModuleIndex]), sizeof(unsigned int), cudaMemcpyDeviceToHost,stream);
    nInnerSegments = std::min(nInnerSegments, N_MAX_PIXEL_SEGMENTS_PER_MODULE);

    cudaMallocHost(&nTriplets, nLowerModules * sizeof(unsigned int));
    cudaMemcpyAsync(nTriplets, tripletsInGPU->nTriplets, nLowerModules * sizeof(unsigned int), cudaMemcpyDeviceToHost,stream);

    cudaMallocHost(&superbins,N_MAX_PIXEL_SEGMENTS_PER_MODULE*sizeof(int));
    cudaMallocHost(&pixelTypes,N_MAX_PIXEL_SEGMENTS_PER_MODULE*sizeof(int));

    cudaMemcpyAsync(superbins,segmentsInGPU->superbin,N_MAX_PIXEL_SEGMENTS_PER_MODULE*sizeof(int),cudaMemcpyDeviceToHost,stream);
    cudaMemcpyAsync(pixelTypes,segmentsInGPU->pixelType,N_MAX_PIXEL_SEGMENTS_PER_MODULE*sizeof(int),cudaMemcpyDeviceToHost,stream);

    unsigned int* connectedPixelSize_host;
    unsigned int* connectedPixelIndex_host;
    cudaMallocHost(&connectedPixelSize_host, nInnerSegments* sizeof(unsigned int));
    cudaMallocHost(&connectedPixelIndex_host, nInnerSegments* sizeof(unsigned int));
    unsigned int* connectedPixelSize_dev;
    unsigned int* connectedPixelIndex_dev;
    cudaMalloc(&connectedPixelSize_dev, nInnerSegments* sizeof(unsigned int));
    cudaMalloc(&connectedPixelIndex_dev, nInnerSegments* sizeof(unsigned int));
    //cudaMallocAsync(&connectedPixelSize_dev, nInnerSegments* sizeof(unsigned int),stream);
    //cudaMallocAsync(&connectedPixelIndex_dev, nInnerSegments* sizeof(unsigned int),stream);
    unsigned int max_size =0;
    int threadSize = 1000000;
    unsigned int *segs_pix = (unsigned int*)malloc(2*threadSize*sizeof(unsigned int));
    unsigned int *segs_pix_offset = segs_pix+threadSize;
    unsigned int *segs_pix_gpu;
    unsigned int *segs_pix_gpu_offset;
    cudaMalloc((void **)&segs_pix_gpu, 2*threadSize*sizeof(unsigned int));
    //cudaMallocAsync((void **)&segs_pix_gpu, 2*threadSize*sizeof(unsigned int),stream);
    segs_pix_gpu_offset = segs_pix_gpu + threadSize;
    cudaMemsetAsync(segs_pix_gpu, nInnerSegments, threadSize*sizeof(unsigned int),stream); // so if not set, it will pass in the kernel
cudaStreamSynchronize(stream);
    unsigned int totalSegs=0;
    int pixelIndexOffsetPos = pixelMapping->connectedPixelsIndex[44999] + pixelMapping->connectedPixelsSizes[44999];
    int pixelIndexOffsetNeg = pixelMapping->connectedPixelsIndexPos[44999] + pixelMapping->connectedPixelsSizes[44999] + pixelIndexOffsetPos;

    for (int i = 0; i < nInnerSegments; i++)
    {// loop over # pLS
        int pixelType = pixelTypes[i];// get pixel type for this pLS
        int superbin = superbins[i]; //get superbin for this pixel
        if((superbin < 0) or (superbin >= 45000) or (pixelType > 2) or (pixelType < 0))
        {
            continue;
        }

        if(pixelType ==0)
        { // used pixel type to select correct size-index arrays
            connectedPixelSize_host[i]  = pixelMapping->connectedPixelsSizes[superbin]; //number of connected modules to this pixel
            connectedPixelIndex_host[i] = pixelMapping->connectedPixelsIndex[superbin];// index to get start of connected modules for this superbin in map
            for (int j=0; j < pixelMapping->connectedPixelsSizes[superbin]; j++)
            { // loop over modules from the size
                segs_pix[totalSegs+j] = i; // save the pixel index in array to be transfered to kernel
              segs_pix_offset[totalSegs+j] = j; // save this segment in array to be transfered to kernel
            }
            totalSegs += connectedPixelSize_host[i]; // increment counter
        }
        else if(pixelType ==1)
        {
            connectedPixelSize_host[i] = pixelMapping->connectedPixelsSizesPos[superbin]; //number of pixel connected modules
            connectedPixelIndex_host[i] = pixelMapping->connectedPixelsIndexPos[superbin]+pixelIndexOffsetPos;// index to get start of connected pixel modules
            for (int j=0; j < pixelMapping->connectedPixelsSizesPos[superbin]; j++)
            {
                segs_pix[totalSegs+j] = i;
                segs_pix_offset[totalSegs+j] = j;
            }
            totalSegs += connectedPixelSize_host[i];
        }
        else if(pixelType ==2)
        {
            connectedPixelSize_host[i] = pixelMapping->connectedPixelsSizesNeg[superbin]; //number of pixel connected modules
            connectedPixelIndex_host[i] =pixelMapping->connectedPixelsIndexNeg[superbin] + pixelIndexOffsetNeg;// index to get start of connected pixel modules
            for (int j=0; j < pixelMapping->connectedPixelsSizesNeg[superbin]; j++)
            {
                segs_pix[totalSegs+j] = i;
                segs_pix_offset[totalSegs+j] = j;
            }
            totalSegs += connectedPixelSize_host[i];
        }
    }

    cudaMemcpyAsync(connectedPixelSize_dev, connectedPixelSize_host, nInnerSegments*sizeof(unsigned int), cudaMemcpyHostToDevice,stream);
    cudaMemcpyAsync(connectedPixelIndex_dev, connectedPixelIndex_host, nInnerSegments*sizeof(unsigned int), cudaMemcpyHostToDevice,stream);
    cudaMemcpyAsync(segs_pix_gpu,segs_pix,threadSize*sizeof(unsigned int), cudaMemcpyHostToDevice,stream);
    cudaMemcpyAsync(segs_pix_gpu_offset,segs_pix_offset,threadSize*sizeof(unsigned int), cudaMemcpyHostToDevice,stream);
cudaStreamSynchronize(stream);

    //less cheap method to estimate max_size for y axis
    max_size = *std::max_element(nTriplets, nTriplets + nLowerModules);
    //dim3 nThreads(16,16,1);
    //dim3 nBlocks((totalSegs % nThreads.x == 0 ? totalSegs / nThreads.x : totalSegs / nThreads.x + 1),
    //              (max_size % nThreads.y == 0 ? max_size/nThreads.y : max_size/nThreads.y + 1),1);
    //printf("%d %d\n",totalSegs,max_size);
    dim3 nThreads(16,64,1);
    dim3 nBlocks(1,MAX_BLOCKS,1);
    createPixelTripletsInGPUFromMap<<<nBlocks, nThreads,0,stream>>>(*modulesInGPU, *hitsInGPU, *mdsInGPU, *segmentsInGPU, *tripletsInGPU, *pixelTripletsInGPU, connectedPixelSize_dev,connectedPixelIndex_dev,nInnerSegments,segs_pix_gpu,segs_pix_gpu_offset, totalSegs);

    cudaError_t cudaerr = cudaGetLastError();
    if(cudaerr != cudaSuccess)
    {
	    std::cout<<"sync failed with error : "<<cudaGetErrorString(cudaerr)<<std::endl;

    }cudaStreamSynchronize(stream);
    //}cudaDeviceSynchronize();
    cudaFreeHost(connectedPixelSize_host);
    cudaFreeHost(connectedPixelIndex_host);
    cudaFree(connectedPixelSize_dev);
    cudaFree(connectedPixelIndex_dev);
    //cudaFreeAsync(connectedPixelSize_dev,stream);
    //cudaFreeAsync(connectedPixelIndex_dev,stream);
    cudaFreeHost(superbins);
    cudaFreeHost(pixelTypes);
    cudaFreeHost(nTriplets);
    free(segs_pix);
    //cudaFreeAsync(segs_pix_gpu,stream);
    cudaFree(segs_pix_gpu);
#ifdef Warnings
    unsigned int nPixelTriplets;
    cudaMemcpyAsync(&nPixelTriplets, &(pixelTripletsInGPU->nPixelTriplets),  sizeof(unsigned int), cudaMemcpyDeviceToHost,stream);
    cudaStreamSynhronize(stream);
    std::cout<<"number of pixel triplets = "<<nPixelTriplets<<std::endl;
#endif

    //pT3s can be cleaned here because they're not used in making pT5s!
#ifdef DUP_pT3
    dim3 nThreads_dup(1024,1,1);
    dim3 nBlocks_dup(MAX_BLOCKS,1,1);
    removeDupPixelTripletsInGPUFromMap<<<nBlocks_dup,nThreads_dup,0,stream>>>(*modulesInGPU, *hitsInGPU, *mdsInGPU, *segmentsInGPU, *pixelTripletsInGPU,*tripletsInGPU,false);
cudaStreamSynchronize(stream);
#endif

}

void SDL::Event::createQuintuplets()
{
    unsigned int nLowerModules;
    cudaMemcpyAsync(&nLowerModules,modulesInGPU->nLowerModules,sizeof(unsigned int),cudaMemcpyDeviceToHost,stream);
cudaStreamSynchronize(stream);

    unsigned int nEligibleT5Modules = 0;
    unsigned int *indicesOfEligibleModules = (unsigned int*)malloc(nLowerModules*sizeof(unsigned int));

    unsigned int maxTriplets;
    createEligibleModulesListForQuintuplets(*modulesInGPU, *tripletsInGPU, nEligibleT5Modules, indicesOfEligibleModules, N_MAX_QUINTUPLETS_PER_MODULE, maxTriplets,stream,*rangesInGPU);

    if(quintupletsInGPU == nullptr)
    {
        cudaMallocHost(&quintupletsInGPU, sizeof(SDL::quintuplets));
#ifdef Explicit_T5
        createQuintupletsInExplicitMemory(*quintupletsInGPU, N_MAX_QUINTUPLETS_PER_MODULE, nLowerModules, nEligibleT5Modules,stream);
#else
        createQuintupletsInUnifiedMemory(*quintupletsInGPU, N_MAX_QUINTUPLETS_PER_MODULE, nLowerModules, nEligibleT5Modules,stream);
#endif
    }
cudaStreamSynchronize(stream);


    int threadSize=N_MAX_TOTAL_TRIPLETS;
    unsigned int *threadIdx = (unsigned int*)malloc(2*threadSize*sizeof(unsigned int));
    unsigned int *threadIdx_offset = threadIdx+threadSize;
    unsigned int *threadIdx_gpu;
    unsigned int *threadIdx_gpu_offset;
    cudaMalloc((void **)&threadIdx_gpu, 2*threadSize*sizeof(unsigned int));
    //cudaMallocAsync((void **)&threadIdx_gpu, 2*threadSize*sizeof(unsigned int),stream);
    cudaMemsetAsync(threadIdx_gpu, nLowerModules, threadSize*sizeof(unsigned int),stream);

    unsigned int *nTriplets = (unsigned int*)malloc(nLowerModules*sizeof(unsigned int));
    cudaMemcpyAsync(nTriplets, tripletsInGPU->nTriplets, nLowerModules*sizeof(unsigned int), cudaMemcpyDeviceToHost,stream);
cudaStreamSynchronize(stream);
    threadIdx_gpu_offset = threadIdx_gpu + threadSize;

    int nTotalTriplets = 0;
    for (int i=0; i<nEligibleT5Modules; i++) 
    {
        int index = indicesOfEligibleModules[i];
        unsigned int nInnerTriplets = nTriplets[index];
        if (nInnerTriplets > N_MAX_TRIPLETS_PER_MODULE) nInnerTriplets = N_MAX_TRIPLETS_PER_MODULE;
        if (nInnerTriplets !=0) 
        {
            for (int j=0; j<nInnerTriplets; j++) 
            {
                threadIdx[nTotalTriplets + j] = index;
                threadIdx_offset[nTotalTriplets + j] = j;
            }
            nTotalTriplets += nInnerTriplets;
        }
    }
    //printf("T5: nTotalTriplets=%d nEligibleT5Modules=%d\n", nTotalTriplets, nEligibleT5Modules);
    if (threadSize < nTotalTriplets) 
    {
        printf("threadSize=%d nTotalTriplets=%d: Increase buffer size for threadIdx in createQuintuplets\n", threadSize, nTotalTriplets);
        exit(1);
    }
    cudaMemcpyAsync(threadIdx_gpu, threadIdx, threadSize*sizeof(unsigned int), cudaMemcpyHostToDevice,stream);
    cudaMemcpyAsync(threadIdx_gpu_offset, threadIdx_offset, threadSize*sizeof(unsigned int), cudaMemcpyHostToDevice,stream);
cudaStreamSynchronize(stream);

    dim3 nThreads(32, 32, 1);
    dim3 nBlocks(1,MAX_BLOCKS,1);

//    dim3 nThreads(16, 16, 1);
//    int max_outerTriplets = N_MAX_TRIPLETS_PER_MODULE;
//
//    int mp;
//    cudaDeviceGetAttribute(&mp, cudaDevAttrMultiProcessorCount, 0);
//    int m = (nTotalTriplets + nThreads.y*mp - 1)/(nThreads.y*mp);
//    int mPerThread=16;
//    m = (m + mPerThread -1)/mPerThread;
//    int nblocksY = mp*m;
//    printf("cuda multiprocessor #:%d mPerThreads=%d nBlocksY=%d\n", mp, mPerThread, nblocksY);
//    dim3 nBlocks((max_outerTriplets % nThreads.x == 0 ? max_outerTriplets/nThreads.x : max_outerTriplets/nThreads.x + 1), nblocksY, 1);

    createQuintupletsInGPU<<<nBlocks,nThreads,0,stream>>>(*modulesInGPU, *hitsInGPU, *mdsInGPU, *segmentsInGPU, *tripletsInGPU, *quintupletsInGPU, threadIdx_gpu, threadIdx_gpu_offset, nTotalTriplets,*rangesInGPU);
    cudaError_t cudaerr = cudaGetLastError();
    if(cudaerr != cudaSuccess)
    {
	    std::cout<<"sync failed with error : "<<cudaGetErrorString(cudaerr)<<std::endl;
    }
    //cudaDeviceSynchronize();
    cudaStreamSynchronize(stream);
    free(threadIdx);
    free(nTriplets);
    //cudaFreeAsync(threadIdx_gpu,stream);
    cudaFree(threadIdx_gpu);
    free(indicesOfEligibleModules);

#ifdef DUP_T5
    dim3 dupThreads(64,16,1);
    dim3 dupBlocks(1,MAX_BLOCKS,1);
    removeDupQuintupletsInGPU<<<dupBlocks,dupThreads,0,stream>>>(*modulesInGPU, *hitsInGPU, *mdsInGPU, *segmentsInGPU, *tripletsInGPU, *quintupletsInGPU,false,*rangesInGPU);
    //cudaDeviceSynchronize();
    cudaStreamSynchronize(stream);
#endif

#if defined(AddObjects)
#ifdef Explicit_T5
    addQuintupletsToEventExplicit();
#else
    addQuintupletsToEvent();
#endif
#endif

}
void SDL::Event::pixelLineSegmentCleaning()
{
#ifdef DUP_pLS
    //printf("cleaning pixels\n");
    checkHitspLS<<<MAX_BLOCKS,1024,0,stream>>>(*modulesInGPU,*mdsInGPU, *segmentsInGPU, *hitsInGPU, false);
    cudaError_t cudaerrpix = cudaGetLastError();
    if(cudaerrpix != cudaSuccess)
    {
        std::cout<<"sync failed with error : "<<cudaGetErrorString(cudaerrpix)<<std::endl;

    }cudaStreamSynchronize(stream);
    //}cudaDeviceSynchronize();
#endif  

}
void SDL::Event::createPixelQuintuplets()
{
    unsigned int nLowerModules;
    cudaMemcpyAsync(&nLowerModules, modulesInGPU->nLowerModules, sizeof(unsigned int), cudaMemcpyDeviceToHost,stream);

    if(pixelQuintupletsInGPU == nullptr)
    {
        cudaMallocHost(&pixelQuintupletsInGPU, sizeof(SDL::pixelQuintuplets));
#ifdef Explicit_PT5
    createPixelQuintupletsInExplicitMemory(*pixelQuintupletsInGPU, N_MAX_PIXEL_QUINTUPLETS,stream);
#else
    createPixelQuintupletsInUnifiedMemory(*pixelQuintupletsInGPU, N_MAX_PIXEL_QUINTUPLETS,stream);
#endif  
    }
   if(trackCandidatesInGPU == nullptr)
    {
        cudaMallocHost(&trackCandidatesInGPU, sizeof(SDL::trackCandidates));
#ifdef Explicit_Track
        createTrackCandidatesInExplicitMemory(*trackCandidatesInGPU, N_MAX_TRACK_CANDIDATES + N_MAX_PIXEL_TRACK_CANDIDATES,stream);
#else
        createTrackCandidatesInUnifiedMemory(*trackCandidatesInGPU, N_MAX_TRACK_CANDIDATES + N_MAX_PIXEL_TRACK_CANDIDATES,stream);

#endif
    } 

    unsigned int pixelModuleIndex;
    int* superbins;
    int* pixelTypes;
    unsigned int *nQuintuplets;
    unsigned int nModules;

    unsigned int* connectedPixelSize_host;
    unsigned int* connectedPixelIndex_host;
    unsigned int* connectedPixelSize_dev;
    unsigned int* connectedPixelIndex_dev;

    cudaMallocHost(&nQuintuplets, nLowerModules * sizeof(unsigned int));
    cudaMemcpyAsync(nQuintuplets, quintupletsInGPU->nQuintuplets, nLowerModules * sizeof(unsigned int), cudaMemcpyDeviceToHost,stream);

    cudaMallocHost(&superbins,N_MAX_PIXEL_SEGMENTS_PER_MODULE*sizeof(int));
    cudaMallocHost(&pixelTypes,N_MAX_PIXEL_SEGMENTS_PER_MODULE*sizeof(int));

    cudaMemcpyAsync(superbins,segmentsInGPU->superbin,N_MAX_PIXEL_SEGMENTS_PER_MODULE*sizeof(int),cudaMemcpyDeviceToHost,stream);
    cudaMemcpyAsync(pixelTypes,segmentsInGPU->pixelType,N_MAX_PIXEL_SEGMENTS_PER_MODULE*sizeof(int),cudaMemcpyDeviceToHost,stream);
    
cudaMemcpyAsync(&nModules,modulesInGPU->nModules,sizeof(unsigned int),cudaMemcpyDeviceToHost,stream);
cudaStreamSynchronize(stream);
    pixelModuleIndex = nModules-1;
    unsigned int nInnerSegments = 0;
    cudaMemcpyAsync(&nInnerSegments, &(segmentsInGPU->nSegments[pixelModuleIndex]), sizeof(unsigned int), cudaMemcpyDeviceToHost,stream);
    nInnerSegments = std::min(nInnerSegments, N_MAX_PIXEL_SEGMENTS_PER_MODULE);


    cudaMallocHost(&connectedPixelSize_host, nInnerSegments* sizeof(unsigned int));
    cudaMallocHost(&connectedPixelIndex_host, nInnerSegments* sizeof(unsigned int));
    cudaMalloc(&connectedPixelSize_dev, nInnerSegments* sizeof(unsigned int));
    cudaMalloc(&connectedPixelIndex_dev, nInnerSegments* sizeof(unsigned int));
    //cudaMallocAsync(&connectedPixelSize_dev, nInnerSegments* sizeof(unsigned int),stream);
    //cudaMallocAsync(&connectedPixelIndex_dev, nInnerSegments* sizeof(unsigned int),stream);

    unsigned int max_size =0;
    int threadSize = 1000000;
    unsigned int *segs_pix = (unsigned int*)malloc(2*threadSize*sizeof(unsigned int));
    unsigned int *segs_pix_offset = segs_pix+threadSize;
    unsigned int *segs_pix_gpu;
    unsigned int *segs_pix_gpu_offset;
    cudaMalloc((void **)&segs_pix_gpu, 2*threadSize*sizeof(unsigned int));
    //cudaMallocAsync((void **)&segs_pix_gpu, 2*threadSize*sizeof(unsigned int),stream);
cudaStreamSynchronize(stream);

    segs_pix_gpu_offset = segs_pix_gpu + threadSize;
    cudaMemsetAsync(segs_pix_gpu, nInnerSegments, threadSize*sizeof(unsigned int),stream); // so if not set, it will pass in the kernel
    unsigned int totalSegs=0;


    int pixelIndexOffsetPos = pixelMapping->connectedPixelsIndex[44999] + pixelMapping->connectedPixelsSizes[44999];
    int pixelIndexOffsetNeg = pixelMapping->connectedPixelsIndexPos[44999] + pixelMapping->connectedPixelsSizes[44999] + pixelIndexOffsetPos;

    for (int i = 0; i < nInnerSegments; i++)
    {// loop over # pLS
        int pixelType = pixelTypes[i];// get pixel type for this pLS
        int superbin = superbins[i]; //get superbin for this pixel
        if((superbin < 0) or (superbin >= 45000) or (pixelType > 2) or (pixelType < 0))
        {
            continue;
        }

        if(pixelType ==0)
        { // used pixel type to select correct size-index arrays
            connectedPixelSize_host[i]  = pixelMapping->connectedPixelsSizes[superbin]; //number of connected modules to this pixel
            connectedPixelIndex_host[i] = pixelMapping->connectedPixelsIndex[superbin];// index to get start of connected modules for this superbin in map
            for (int j=0; j < pixelMapping->connectedPixelsSizes[superbin]; j++)
            { // loop over modules from the size
                segs_pix[totalSegs+j] = i; // save the pixel index in array to be transfered to kernel
                segs_pix_offset[totalSegs+j] = j; // save this segment in array to be transfered to kernel
            }
            totalSegs += connectedPixelSize_host[i]; // increment counter
        }
        else if(pixelType ==1)
        {
            connectedPixelSize_host[i] = pixelMapping->connectedPixelsSizesPos[superbin]; //number of pixel connected modules
            connectedPixelIndex_host[i] = pixelMapping->connectedPixelsIndexPos[superbin]+pixelIndexOffsetPos;// index to get start of connected pixel modules
            for (int j=0; j < pixelMapping->connectedPixelsSizesPos[superbin]; j++)
            {
                segs_pix[totalSegs+j] = i;
                segs_pix_offset[totalSegs+j] = j;
            }
            totalSegs += connectedPixelSize_host[i];
        }
        else if(pixelType ==2)
        {
            connectedPixelSize_host[i] = pixelMapping->connectedPixelsSizesNeg[superbin]; //number of pixel connected modules
            connectedPixelIndex_host[i] =pixelMapping->connectedPixelsIndexNeg[superbin] + pixelIndexOffsetNeg;// index to get start of connected pixel modules
            for (int j=0; j < pixelMapping->connectedPixelsSizesNeg[superbin]; j++)
            {
                segs_pix[totalSegs+j] = i;
                segs_pix_offset[totalSegs+j] = j;
            }
            totalSegs += connectedPixelSize_host[i];
        }
    }

    cudaMemcpyAsync(connectedPixelSize_dev, connectedPixelSize_host, nInnerSegments*sizeof(unsigned int), cudaMemcpyHostToDevice,stream);
    cudaMemcpyAsync(connectedPixelIndex_dev, connectedPixelIndex_host, nInnerSegments*sizeof(unsigned int), cudaMemcpyHostToDevice,stream);
    cudaMemcpyAsync(segs_pix_gpu,segs_pix,threadSize*sizeof(unsigned int), cudaMemcpyHostToDevice,stream);
    cudaMemcpyAsync(segs_pix_gpu_offset,segs_pix_offset,threadSize*sizeof(unsigned int), cudaMemcpyHostToDevice,stream);
cudaStreamSynchronize(stream);

    //less cheap method to estimate max_size for y axis
    max_size = *std::max_element(nQuintuplets, nQuintuplets + nLowerModules);
    //dim3 nThreads(16,16,1);
    //dim3 nBlocks((totalSegs % nThreads.x == 0 ? totalSegs / nThreads.x : totalSegs / nThreads.x + 1),
    //              (max_size % nThreads.y == 0 ? max_size/nThreads.y : max_size/nThreads.y + 1),1);
    dim3 nThreads(16,64,1);
    dim3 nBlocks(1,MAX_BLOCKS,1);
                  
    createPixelQuintupletsInGPUFromMap<<<nBlocks, nThreads,0,stream>>>(*modulesInGPU, *hitsInGPU, *mdsInGPU, *segmentsInGPU, *tripletsInGPU, *quintupletsInGPU, *pixelQuintupletsInGPU, connectedPixelSize_dev, connectedPixelIndex_dev, nInnerSegments, segs_pix_gpu, segs_pix_gpu_offset, totalSegs,*rangesInGPU);

    cudaError_t cudaerr = cudaGetLastError();
    if(cudaerr != cudaSuccess)
    {
	    std::cout<<"sync failed with error : "<<cudaGetErrorString(cudaerr)<<std::endl;

    }cudaStreamSynchronize(stream);
    //}cudaDeviceSynchronize();
    cudaFreeHost(connectedPixelSize_host);
    cudaFreeHost(connectedPixelIndex_host);
    cudaFree(connectedPixelSize_dev);
    cudaFree(connectedPixelIndex_dev);
    //cudaFreeAsync(connectedPixelSize_dev,stream);
    //cudaFreeAsync(connectedPixelIndex_dev,stream);
    cudaFreeHost(superbins);
    cudaFreeHost(pixelTypes);
    cudaFreeHost(nQuintuplets);
    free(segs_pix);
    cudaFree(segs_pix_gpu);
    //cudaFreeAsync(segs_pix_gpu,stream);

    dim3 nThreads_dup(1024,1,1);
    dim3 nBlocks_dup(MAX_BLOCKS,1,1);
#ifdef DUP_pT5
    //printf("run dup pT5\n");
    removeDupPixelQuintupletsInGPUFromMap<<<nBlocks_dup,nThreads_dup,0,stream>>>(*modulesInGPU, *hitsInGPU, *mdsInGPU, *segmentsInGPU, *pixelTripletsInGPU,*tripletsInGPU, *pixelQuintupletsInGPU, *quintupletsInGPU,false);
    cudaError_t cudaerr2 = cudaGetLastError(); 
    if(cudaerr2 != cudaSuccess)
    {
        std::cout<<"sync failed with error : "<<cudaGetErrorString(cudaerr2)<<std::endl;
    }cudaStreamSynchronize(stream);
    //}cudaDeviceSynchronize();
#endif
#ifdef FINAL_pT5
    //printf("Adding pT5s to TC collection\n");
    unsigned int nThreadsx_pT5 = 256;
    unsigned int nBlocksx_pT5 = 1;//(N_MAX_PIXEL_QUINTUPLETS) % nThreadsx_pT5 == 0 ? N_MAX_PIXEL_QUINTUPLETS / nThreadsx_pT5 : N_MAX_PIXEL_QUINTUPLETS / nThreadsx_pT5 + 1;
    addpT5asTrackCandidateInGPU<<<nBlocksx_pT5, nThreadsx_pT5,0,stream>>>(*modulesInGPU, *pixelQuintupletsInGPU, *trackCandidatesInGPU, *segmentsInGPU, *tripletsInGPU,*quintupletsInGPU);

    cudaError_t cudaerr_pT5 = cudaGetLastError();
    if(cudaerr_pT5 != cudaSuccess)
    {
        std::cout<<"sync failed with error : "<<cudaGetErrorString(cudaerr_pT5)<<std::endl;
    }
    cudaStreamSynchronize(stream);
#endif
#ifdef Warnings
    unsigned int nPixelQuintuplets;
    cudaMemcpyAsync(&nPixelQuintuplets, &(pixelQuintupletsInGPU->nPixelQuintuplets), sizeof(unsigned int), cudaMemcpyDeviceToHost,stream);
    cudaStreamSynchronize(stream);
    std::cout<<"number of pixel quintuplets = "<<nPixelQuintuplets<<std::endl;
#endif   
}

void SDL::Event::addQuintupletsToEvent()
{
    unsigned int idx;
    for(unsigned int i = 0; i<*(SDL::modulesInGPU->nLowerModules); i++)
    {
        idx = SDL::modulesInGPU->lowerModuleIndices[i];
        //tracklets run only on lower modules!!!!!!
        if(quintupletsInGPU->nQuintuplets[i] == 0)
        {
            rangesInGPU->quintupletRanges[idx * 2] = -1;
            rangesInGPU->quintupletRanges[idx * 2 + 1] = -1;
        }
        else
        {
            rangesInGPU->quintupletRanges[idx * 2] = rangesInGPU->quintupletModuleIndices[i];
            rangesInGPU->quintupletRanges[idx * 2 + 1] = rangesInGPU->quintupletModuleIndices[i] + quintupletsInGPU->nQuintuplets[i] - 1;

            if(modulesInGPU->subdets[idx] == Barrel)
            {
                n_quintuplets_by_layer_barrel_[modulesInGPU->layers[idx] - 1] += quintupletsInGPU->nQuintuplets[i];
            }
            else
            {
                n_quintuplets_by_layer_endcap_[modulesInGPU->layers[idx] - 1] += quintupletsInGPU->nQuintuplets[i];
            }
        }
    }
}

void SDL::Event::addQuintupletsToEventExplicit()
{
    unsigned int nLowerModules;
    cudaMemcpyAsync(&nLowerModules,modulesInGPU->nLowerModules,sizeof(unsigned int),cudaMemcpyDeviceToHost,stream);

    unsigned int* nQuintupletsCPU;
    cudaMallocHost(&nQuintupletsCPU, nLowerModules * sizeof(unsigned int));
    cudaMemcpyAsync(nQuintupletsCPU,quintupletsInGPU->nQuintuplets,nLowerModules*sizeof(unsigned int),cudaMemcpyDeviceToHost,stream);

    short* module_subdets;
    cudaMallocHost(&module_subdets, nModules* sizeof(short));
    cudaMemcpyAsync(module_subdets,modulesInGPU->subdets,nModules*sizeof(short),cudaMemcpyDeviceToHost,stream);

    unsigned int* module_lowerModuleIndices;
    cudaMallocHost(&module_lowerModuleIndices, (nLowerModules +1)* sizeof(unsigned int));
    cudaMemcpyAsync(module_lowerModuleIndices,modulesInGPU->lowerModuleIndices,(nLowerModules+1)*sizeof(unsigned int),cudaMemcpyDeviceToHost,stream);

    int* module_quintupletRanges;
    cudaMallocHost(&module_quintupletRanges, nModules* 2*sizeof(int));
    cudaMemcpyAsync(module_quintupletRanges,rangesInGPU->quintupletRanges,nModules*2*sizeof(int),cudaMemcpyDeviceToHost,stream);
    short* module_layers;
    cudaMallocHost(&module_layers, nModules * sizeof(short));
    cudaMemcpyAsync(module_layers,modulesInGPU->layers,nModules*sizeof(short),cudaMemcpyDeviceToHost,stream);
    int* module_quintupletModuleIndices;
    cudaMallocHost(&module_quintupletModuleIndices, nLowerModules * sizeof(int));
    cudaMemcpyAsync(module_quintupletModuleIndices, rangesInGPU->quintupletModuleIndices, nLowerModules * sizeof(int), cudaMemcpyDeviceToHost,stream);
cudaStreamSynchronize(stream);
    unsigned int idx;
    for(unsigned int i = 0; i<nLowerModules; i++)
    {
        idx = module_lowerModuleIndices[i];
        if(nQuintupletsCPU[i] == 0 or module_quintupletModuleIndices[i] == -1)
        {
            module_quintupletRanges[idx * 2] = -1;
            module_quintupletRanges[idx * 2 + 1] = -1;
        }
       else
        {
            module_quintupletRanges[idx * 2] = module_quintupletModuleIndices[i];
            module_quintupletRanges[idx * 2 + 1] = module_quintupletModuleIndices[i] + nQuintupletsCPU[i] - 1;

            if(module_subdets[idx] == Barrel)
            {
                n_quintuplets_by_layer_barrel_[module_layers[idx] - 1] += nQuintupletsCPU[i];
            }
            else
            {
                n_quintuplets_by_layer_endcap_[module_layers[idx] - 1] += nQuintupletsCPU[i];
            }
        }
    }
    cudaFreeHost(nQuintupletsCPU);
    cudaFreeHost(module_lowerModuleIndices);
    cudaFreeHost(module_quintupletRanges);
    cudaFreeHost(module_layers);
    cudaFreeHost(module_subdets);
    cudaFreeHost(module_quintupletModuleIndices);

}

void SDL::Event::addTripletsToEvent()
{
    unsigned int idx;
    for(unsigned int i = 0; i<*(SDL::modulesInGPU->nLowerModules); i++)
    {
        idx = SDL::modulesInGPU->lowerModuleIndices[i];
        //tracklets run only on lower modules!!!!!!
        if(tripletsInGPU->nTriplets[i] == 0)
        {
            rangesInGPU->tripletRanges[idx * 2] = -1;
            rangesInGPU->tripletRanges[idx * 2 + 1] = -1;
        }
        else
        {
            rangesInGPU->tripletRanges[idx * 2] = idx * N_MAX_TRIPLETS_PER_MODULE;
            rangesInGPU->tripletRanges[idx * 2 + 1] = idx * N_MAX_TRIPLETS_PER_MODULE + tripletsInGPU->nTriplets[i] - 1;

            if(modulesInGPU->subdets[idx] == Barrel)
            {
                n_triplets_by_layer_barrel_[modulesInGPU->layers[idx] - 1] += tripletsInGPU->nTriplets[i];
            }
            else
            {
                n_triplets_by_layer_endcap_[modulesInGPU->layers[idx] - 1] += tripletsInGPU->nTriplets[i];
            }
        }
    }
}
void SDL::Event::addTripletsToEventExplicit()
{
    unsigned int nLowerModules;
    cudaMemcpyAsync(&nLowerModules,modulesInGPU->nLowerModules,sizeof(unsigned int),cudaMemcpyDeviceToHost,stream);

    unsigned int* nTripletsCPU;
    cudaMallocHost(&nTripletsCPU, nLowerModules * sizeof(unsigned int));
    cudaMemcpyAsync(nTripletsCPU,tripletsInGPU->nTriplets,nLowerModules*sizeof(unsigned int),cudaMemcpyDeviceToHost,stream);

    short* module_subdets;
    cudaMallocHost(&module_subdets, nModules* sizeof(short));
    cudaMemcpyAsync(module_subdets,modulesInGPU->subdets,nModules*sizeof(short),cudaMemcpyDeviceToHost,stream);
    unsigned int* module_lowerModuleIndices;
    cudaMallocHost(&module_lowerModuleIndices, (nLowerModules +1)* sizeof(unsigned int));
    cudaMemcpyAsync(module_lowerModuleIndices,modulesInGPU->lowerModuleIndices,(nLowerModules+1)*sizeof(unsigned int),cudaMemcpyDeviceToHost,stream);
    int* module_tripletRanges;
    cudaMallocHost(&module_tripletRanges, nModules* 2*sizeof(int));
    cudaMemcpyAsync(module_tripletRanges,rangesInGPU->tripletRanges,nModules*2*sizeof(int),cudaMemcpyDeviceToHost,stream);
    short* module_layers;
    cudaMallocHost(&module_layers, nModules * sizeof(short));
    cudaMemcpyAsync(module_layers,modulesInGPU->layers,nModules*sizeof(short),cudaMemcpyDeviceToHost,stream);
    unsigned int idx;
    cudaStreamSynchronize(stream);
    for(unsigned int i = 0; i<nLowerModules; i++)
    {
        idx = module_lowerModuleIndices[i];
        //tracklets run only on lower modules!!!!!!
        if(nTripletsCPU[i] == 0)
        {
            module_tripletRanges[idx * 2] = -1;
            module_tripletRanges[idx * 2 + 1] = -1;
        }
        else
        {
            module_tripletRanges[idx * 2] = idx * N_MAX_TRIPLETS_PER_MODULE;
            module_tripletRanges[idx * 2 + 1] = idx * N_MAX_TRIPLETS_PER_MODULE + nTripletsCPU[i] - 1;

            if(module_subdets[idx] == Barrel)
            {
                n_triplets_by_layer_barrel_[module_layers[idx] - 1] += nTripletsCPU[i];
            }
            else
            {
                n_triplets_by_layer_endcap_[module_layers[idx] - 1] += nTripletsCPU[i];
            }
        }
    }
    cudaFreeHost(nTripletsCPU);
    cudaFreeHost(module_lowerModuleIndices);
    cudaFreeHost(module_tripletRanges);
    cudaFreeHost(module_layers);
    cudaFreeHost(module_subdets);
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
#ifdef Explicit_PT3
    unsigned int nPixelTriplets;
    cudaMemcpyAsync(&nPixelTriplets, pixelTripletsInGPU->nPixelTriplets, sizeof(unsigned int), cudaMemcpyDeviceToHost,stream);
cudaStreamSynchronize(stream);
    return nPixelTriplets;
#else
    return *(pixelTripletsInGPU->nPixelTriplets);
#endif
}


unsigned int SDL::Event::getNumberOfExtendedTracks()
{
    unsigned int nTrackCandidates;
    cudaMemcpy(&nTrackCandidates, trackCandidatesInGPU->nTrackCandidates, sizeof(unsigned int), cudaMemcpyDeviceToHost);

    unsigned int *nTrackExtensionsCPU = new unsigned int[nTrackCandidates];
    cudaMemcpy(nTrackExtensionsCPU, trackExtensionsInGPU->nTrackExtensions, (nTrackCandidates)* sizeof(unsigned int), cudaMemcpyDeviceToHost);

    unsigned int nTrackExtensions = 0;
    for(size_t it = 0; it < nTrackCandidates; it++)    
    {
        nTrackExtensions += nTrackExtensionsCPU[it];

    }
#ifdef T3T3_EXTENSIONS
    unsigned int nT3T3Extensions;
    cudaMemcpy(&nT3T3Extensions,&(trackExtensionsInGPU->nTrackExtensions[nTrackCandidates]), sizeof(unsigned int), cudaMemcpyDeviceToHost);
    nTrackExtensions += nT3T3Extensions;
#endif
    delete[] nTrackExtensionsCPU;
    return nTrackExtensions;
}

unsigned int SDL::Event::getNumberOfT3T3ExtendedTracks()
{
    unsigned int nTrackCandidates;
    cudaMemcpy(&nTrackCandidates, trackCandidatesInGPU->nTrackCandidates, sizeof(unsigned int), cudaMemcpyDeviceToHost);

    unsigned int nT3T3Extensions;
    cudaMemcpy(&nT3T3Extensions, trackExtensionsInGPU->nTrackExtensions + nTrackCandidates, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    return nT3T3Extensions;
}

unsigned int SDL::Event::getNumberOfPixelQuintuplets()
{
#ifdef Explicit_PT5
    unsigned int nPixelQuintuplets;
    cudaMemcpyAsync(&nPixelQuintuplets, pixelQuintupletsInGPU->nPixelQuintuplets, sizeof(unsigned int), cudaMemcpyDeviceToHost,stream);
cudaStreamSynchronize(stream);
    return nPixelQuintuplets;

#else
    return *(pixelQuintupletsInGPU->nPixelQuintuplets);
#endif
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
#ifdef Explicit_Hit
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
        hitsInCPU->xs = new float[nHits];
        hitsInCPU->ys = new float[nHits];
        hitsInCPU->zs = new float[nHits];
        hitsInCPU->moduleIndices = new unsigned int[nHits];
        cudaMemcpyAsync(hitsInCPU->idxs, hitsInGPU->idxs,sizeof(unsigned int) * nHits, cudaMemcpyDeviceToHost,stream);
        cudaMemcpyAsync(hitsInCPU->xs, hitsInGPU->xs, sizeof(float) * nHits, cudaMemcpyDeviceToHost,stream);
        cudaMemcpyAsync(hitsInCPU->ys, hitsInGPU->ys, sizeof(float) * nHits, cudaMemcpyDeviceToHost,stream);
        cudaMemcpyAsync(hitsInCPU->zs, hitsInGPU->zs, sizeof(float) * nHits, cudaMemcpyDeviceToHost,stream);
        cudaMemcpyAsync(hitsInCPU->moduleIndices, hitsInGPU->moduleIndices, sizeof(unsigned int) * nHits, cudaMemcpyDeviceToHost,stream);
cudaStreamSynchronize(stream);
    }
    return hitsInCPU;
}
SDL::objectRanges* SDL::Event::getRanges()
{
        unsigned int nLowerModules;
        cudaMemcpyAsync(&nLowerModules, modulesInGPU->nLowerModules, sizeof(unsigned int), cudaMemcpyDeviceToHost,stream);
        cudaStreamSynchronize(stream);
    if(rangesInCPU == nullptr)
    {
        rangesInCPU = new SDL::objectRanges;
        rangesInCPU->hitRanges = new int[2*nModules];
        rangesInCPU->quintupletModuleIndices = new int[nLowerModules];
        cudaMemcpyAsync(rangesInCPU->hitRanges, rangesInGPU->hitRanges, 2*nModules * sizeof(int), cudaMemcpyDeviceToHost,stream);
        cudaMemcpyAsync(rangesInCPU->quintupletModuleIndices, rangesInGPU->quintupletModuleIndices, nLowerModules * sizeof(int), cudaMemcpyDeviceToHost,stream);
cudaStreamSynchronize(stream);
    }
    return rangesInCPU;
}
#else
SDL::hits* SDL::Event::getHits() //std::shared_ptr should take care of garbage collection
{
    return hitsInGPU;
}
SDL::objectRanges* SDL::Event::getRanges()
{
    return rangesInGPU;
}
#endif


#ifdef Explicit_MD
SDL::miniDoublets* SDL::Event::getMiniDoublets()
{
    if(mdsInCPU == nullptr)
    {
        mdsInCPU = new SDL::miniDoublets;
        unsigned int nMemoryLocations = (N_MAX_MD_PER_MODULES * (nModules - 1) + N_MAX_PIXEL_MD_PER_MODULES);
        mdsInCPU->hitIndices = new unsigned int[2 * nMemoryLocations];
        mdsInCPU->nMDs = new unsigned int[nModules];
        cudaMemcpyAsync(mdsInCPU->hitIndices, mdsInGPU->hitIndices, 2 * nMemoryLocations * sizeof(unsigned int), cudaMemcpyDeviceToHost,stream);
        cudaMemcpyAsync(mdsInCPU->nMDs, mdsInGPU->nMDs, nModules * sizeof(unsigned int), cudaMemcpyDeviceToHost,stream);
cudaStreamSynchronize(stream);
    }
    return mdsInCPU;
}
#else
SDL::miniDoublets* SDL::Event::getMiniDoublets()
{
    return mdsInGPU;
}
#endif


#ifdef Explicit_Seg
SDL::segments* SDL::Event::getSegments()
{
    if(segmentsInCPU == nullptr)
    {
        segmentsInCPU = new SDL::segments;
        unsigned int nMemoryLocations = (N_MAX_SEGMENTS_PER_MODULE) * (nModules - 1) + N_MAX_PIXEL_SEGMENTS_PER_MODULE;
        segmentsInCPU->mdIndices = new unsigned int[2 * nMemoryLocations];
        segmentsInCPU->nSegments = new unsigned int[nModules];
        segmentsInCPU->innerMiniDoubletAnchorHitIndices = new unsigned int[nMemoryLocations];
        segmentsInCPU->outerMiniDoubletAnchorHitIndices = new unsigned int[nMemoryLocations];
        segmentsInCPU->ptIn = new float[N_MAX_PIXEL_SEGMENTS_PER_MODULE];
        segmentsInCPU->eta = new float[N_MAX_PIXEL_SEGMENTS_PER_MODULE];
        segmentsInCPU->phi = new float[N_MAX_PIXEL_SEGMENTS_PER_MODULE];
        segmentsInCPU->isDup = new bool[N_MAX_PIXEL_SEGMENTS_PER_MODULE];
        segmentsInCPU->isQuad = new bool[N_MAX_PIXEL_SEGMENTS_PER_MODULE];
        segmentsInCPU->score = new float[N_MAX_PIXEL_SEGMENTS_PER_MODULE];
        cudaMemcpyAsync(segmentsInCPU->mdIndices, segmentsInGPU->mdIndices, 2 * nMemoryLocations * sizeof(unsigned int), cudaMemcpyDeviceToHost,stream);
        cudaMemcpyAsync(segmentsInCPU->nSegments, segmentsInGPU->nSegments, nModules * sizeof(unsigned int), cudaMemcpyDeviceToHost,stream);
        cudaMemcpyAsync(segmentsInCPU->innerMiniDoubletAnchorHitIndices, segmentsInGPU->innerMiniDoubletAnchorHitIndices, nMemoryLocations * sizeof(unsigned int), cudaMemcpyDeviceToHost,stream);
        cudaMemcpyAsync(segmentsInCPU->outerMiniDoubletAnchorHitIndices, segmentsInGPU->outerMiniDoubletAnchorHitIndices, nMemoryLocations * sizeof(unsigned int), cudaMemcpyDeviceToHost,stream);
        cudaMemcpyAsync(segmentsInCPU->ptIn, segmentsInGPU->ptIn, N_MAX_PIXEL_SEGMENTS_PER_MODULE * sizeof(float), cudaMemcpyDeviceToHost,stream);
        cudaMemcpyAsync(segmentsInCPU->eta, segmentsInGPU->eta, N_MAX_PIXEL_SEGMENTS_PER_MODULE * sizeof(float), cudaMemcpyDeviceToHost,stream);
        cudaMemcpyAsync(segmentsInCPU->phi, segmentsInGPU->phi, N_MAX_PIXEL_SEGMENTS_PER_MODULE * sizeof(float), cudaMemcpyDeviceToHost,stream);
        cudaMemcpyAsync(segmentsInCPU->isDup, segmentsInGPU->isDup, N_MAX_PIXEL_SEGMENTS_PER_MODULE * sizeof(bool), cudaMemcpyDeviceToHost,stream);
        cudaMemcpyAsync(segmentsInCPU->isQuad, segmentsInGPU->isQuad, N_MAX_PIXEL_SEGMENTS_PER_MODULE * sizeof(bool), cudaMemcpyDeviceToHost,stream);
        cudaMemcpyAsync(segmentsInCPU->score, segmentsInGPU->score, N_MAX_PIXEL_SEGMENTS_PER_MODULE * sizeof(float), cudaMemcpyDeviceToHost,stream);
cudaStreamSynchronize(stream);
    }
    return segmentsInCPU;
}
#else
SDL::segments* SDL::Event::getSegments()
{
    return segmentsInGPU;
}
#endif

#ifdef Explicit_Trips
SDL::triplets* SDL::Event::getTriplets()
{
    if(tripletsInCPU == nullptr)
    {
        unsigned int nLowerModules;
        tripletsInCPU = new SDL::triplets;
        cudaMemcpyAsync(&nLowerModules, modulesInGPU->nLowerModules, sizeof(unsigned int), cudaMemcpyDeviceToHost,stream);
cudaStreamSynchronize(stream);
        unsigned int nMemoryLocations = (N_MAX_TRIPLETS_PER_MODULE) * (nLowerModules);
        tripletsInCPU->segmentIndices = new unsigned[2 * nMemoryLocations];
        tripletsInCPU->nTriplets = new unsigned int[nLowerModules];
        tripletsInCPU->betaIn = new float[nMemoryLocations];
        tripletsInCPU->betaOut = new float[nMemoryLocations];
        tripletsInCPU->pt_beta = new float[nMemoryLocations];
#ifdef TRACK_EXTENSIONS
        tripletsInCPU->hitIndices = new unsigned int[6 * nMemoryLocations];
        tripletsInCPU->logicalLayers = new unsigned int[3 * nMemoryLocations];
        cudaMemcpyAsync(tripletsInCPU->hitIndices, tripletsInGPU->hitIndices, 6 * nMemoryLocations * sizeof(unsigned int), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(tripletsInCPU->logicalLayers, tripletsInGPU->logicalLayers, 3 * nMemoryLocations * sizeof(unsigned int), cudaMemcpyDeviceToHost, stream);
#endif
        cudaMemcpyAsync(tripletsInCPU->segmentIndices, tripletsInGPU->segmentIndices, 2 * nMemoryLocations * sizeof(unsigned int), cudaMemcpyDeviceToHost,stream);
        cudaMemcpyAsync(tripletsInCPU->betaIn, tripletsInGPU->betaIn, nMemoryLocations * sizeof(float), cudaMemcpyDeviceToHost,stream);
        cudaMemcpyAsync(tripletsInCPU->betaOut, tripletsInGPU->betaOut, nMemoryLocations * sizeof(float), cudaMemcpyDeviceToHost,stream);
        cudaMemcpyAsync(tripletsInCPU->pt_beta, tripletsInGPU->pt_beta, nMemoryLocations * sizeof(float), cudaMemcpyDeviceToHost,stream);
        cudaMemcpyAsync(tripletsInCPU->nTriplets, tripletsInGPU->nTriplets, nLowerModules * sizeof(unsigned int), cudaMemcpyDeviceToHost,stream);

cudaStreamSynchronize(stream);
    }
    return tripletsInCPU;
}
#else
SDL::triplets* SDL::Event::getTriplets()
{
    return tripletsInGPU;
}
#endif

#ifdef Explicit_T5
SDL::quintuplets* SDL::Event::getQuintuplets()
{
    if(quintupletsInCPU == nullptr)
    {
        quintupletsInCPU = new SDL::quintuplets;
        unsigned int nLowerModules;
        cudaMemcpyAsync(&nLowerModules, modulesInGPU->nLowerModules, sizeof(unsigned int), cudaMemcpyDeviceToHost,stream);
        unsigned int nEligibleT5Modules;
        cudaMemcpyAsync(&nEligibleT5Modules, rangesInGPU->nEligibleT5Modules, sizeof(unsigned int), cudaMemcpyDeviceToHost,stream);
cudaStreamSynchronize(stream);
        unsigned int nMemoryLocations = nEligibleT5Modules * N_MAX_QUINTUPLETS_PER_MODULE;

        quintupletsInCPU->nQuintuplets = new unsigned int[nLowerModules];
        quintupletsInCPU->tripletIndices = new unsigned int[2 * nMemoryLocations];
        quintupletsInCPU->lowerModuleIndices = new unsigned int[5 * nMemoryLocations];
        quintupletsInCPU->innerRadius = new float[nMemoryLocations];
        quintupletsInCPU->outerRadius = new float[nMemoryLocations];
        quintupletsInCPU->isDup = new bool[nMemoryLocations];
        quintupletsInCPU->score_rphisum = new float[nMemoryLocations];
        quintupletsInCPU->eta = new float[nMemoryLocations];
        quintupletsInCPU->phi = new float[nMemoryLocations];
        quintupletsInCPU->regressionRadius = new float[nMemoryLocations];
        cudaMemcpyAsync(quintupletsInCPU->nQuintuplets, quintupletsInGPU->nQuintuplets,  nLowerModules * sizeof(unsigned int), cudaMemcpyDeviceToHost,stream);
        cudaMemcpyAsync(quintupletsInCPU->tripletIndices, quintupletsInGPU->tripletIndices, 2 * nMemoryLocations * sizeof(unsigned int), cudaMemcpyDeviceToHost,stream);
        cudaMemcpyAsync(quintupletsInCPU->lowerModuleIndices, quintupletsInGPU->lowerModuleIndices, 5 * nMemoryLocations * sizeof(unsigned int), cudaMemcpyDeviceToHost,stream);
        cudaMemcpyAsync(quintupletsInCPU->innerRadius, quintupletsInGPU->innerRadius, nMemoryLocations * sizeof(float), cudaMemcpyDeviceToHost,stream);
        cudaMemcpyAsync(quintupletsInCPU->outerRadius, quintupletsInGPU->outerRadius, nMemoryLocations * sizeof(float), cudaMemcpyDeviceToHost,stream);
        cudaMemcpyAsync(quintupletsInCPU->isDup, quintupletsInGPU->isDup, nMemoryLocations * sizeof(bool), cudaMemcpyDeviceToHost,stream);
        cudaMemcpyAsync(quintupletsInCPU->score_rphisum, quintupletsInGPU->score_rphisum, nMemoryLocations * sizeof(float), cudaMemcpyDeviceToHost,stream);
        cudaMemcpyAsync(quintupletsInCPU->eta, quintupletsInGPU->eta, nMemoryLocations * sizeof(float), cudaMemcpyDeviceToHost,stream);
        cudaMemcpyAsync(quintupletsInCPU->phi, quintupletsInGPU->phi, nMemoryLocations * sizeof(float), cudaMemcpyDeviceToHost,stream);
        cudaMemcpyAsync(quintupletsInCPU->regressionRadius, quintupletsInGPU->regressionRadius, nMemoryLocations * sizeof(float), cudaMemcpyDeviceToHost,stream);
cudaStreamSynchronize(stream);
    }

    return quintupletsInCPU;
}
#else
SDL::quintuplets* SDL::Event::getQuintuplets()
{
    return quintupletsInGPU;
}
#endif

#ifdef Explicit_PT3
SDL::pixelTriplets* SDL::Event::getPixelTriplets()
{
    if(pixelTripletsInCPU == nullptr)
    {
        pixelTripletsInCPU = new SDL::pixelTriplets;

        pixelTripletsInCPU->nPixelTriplets = new unsigned int;
        cudaMemcpyAsync(pixelTripletsInCPU->nPixelTriplets, pixelTripletsInGPU->nPixelTriplets, sizeof(unsigned int), cudaMemcpyDeviceToHost,stream);
cudaStreamSynchronize(stream);
        unsigned int nPixelTriplets = *(pixelTripletsInCPU->nPixelTriplets);
        pixelTripletsInCPU->tripletIndices = new unsigned int[nPixelTriplets];
        pixelTripletsInCPU->pixelSegmentIndices = new unsigned int[nPixelTriplets];
        pixelTripletsInCPU->pixelRadius = new float[nPixelTriplets];
        pixelTripletsInCPU->pixelRadiusError = new float[nPixelTriplets];
        pixelTripletsInCPU->tripletRadius = new float[nPixelTriplets];
        pixelTripletsInCPU->isDup = new bool[nPixelTriplets];
        pixelTripletsInCPU->eta = new float[nPixelTriplets];
        pixelTripletsInCPU->phi = new float[nPixelTriplets];
        pixelTripletsInCPU->score = new float[nPixelTriplets];

        cudaMemcpyAsync(pixelTripletsInCPU->tripletIndices, pixelTripletsInGPU->tripletIndices, nPixelTriplets * sizeof(unsigned int), cudaMemcpyDeviceToHost,stream);
        cudaMemcpyAsync(pixelTripletsInCPU->pixelSegmentIndices, pixelTripletsInGPU->pixelSegmentIndices, nPixelTriplets * sizeof(unsigned int), cudaMemcpyDeviceToHost,stream);
        cudaMemcpyAsync(pixelTripletsInCPU->pixelRadius, pixelTripletsInGPU->pixelRadius, nPixelTriplets * sizeof(float), cudaMemcpyDeviceToHost,stream);
        cudaMemcpyAsync(pixelTripletsInCPU->tripletRadius, pixelTripletsInGPU->tripletRadius, nPixelTriplets * sizeof(float), cudaMemcpyDeviceToHost,stream);
        cudaMemcpyAsync(pixelTripletsInCPU->isDup, pixelTripletsInGPU->isDup, nPixelTriplets * sizeof(bool), cudaMemcpyDeviceToHost,stream);
        cudaMemcpyAsync(pixelTripletsInCPU->eta, pixelTripletsInGPU->eta, nPixelTriplets * sizeof(float), cudaMemcpyDeviceToHost,stream);
        cudaMemcpyAsync(pixelTripletsInCPU->phi, pixelTripletsInGPU->phi, nPixelTriplets * sizeof(float), cudaMemcpyDeviceToHost,stream);
        cudaMemcpyAsync(pixelTripletsInCPU->score, pixelTripletsInGPU->score, nPixelTriplets * sizeof(float), cudaMemcpyDeviceToHost,stream);
cudaStreamSynchronize(stream);
    }
    return pixelTripletsInCPU;
}
#else
SDL::pixelTriplets* SDL::Event::getPixelTriplets()
{
    return pixelTripletsInGPU;
}
#endif

#ifdef Explicit_PT5
SDL::pixelQuintuplets* SDL::Event::getPixelQuintuplets()
{
    if(pixelQuintupletsInCPU == nullptr)
    {
        pixelQuintupletsInCPU = new SDL::pixelQuintuplets;

        pixelQuintupletsInCPU->nPixelQuintuplets = new unsigned int;
        cudaMemcpyAsync(pixelQuintupletsInCPU->nPixelQuintuplets, pixelQuintupletsInGPU->nPixelQuintuplets, sizeof(unsigned int), cudaMemcpyDeviceToHost,stream);
cudaStreamSynchronize(stream);
        unsigned int nPixelQuintuplets = *(pixelQuintupletsInCPU->nPixelQuintuplets);

        pixelQuintupletsInCPU->pixelIndices = new unsigned int[nPixelQuintuplets];
        pixelQuintupletsInCPU->T5Indices = new unsigned int[nPixelQuintuplets];
        pixelQuintupletsInCPU->isDup = new bool[nPixelQuintuplets];
        pixelQuintupletsInCPU->score = new float[nPixelQuintuplets];

        cudaMemcpyAsync(pixelQuintupletsInCPU->pixelIndices, pixelQuintupletsInGPU->pixelIndices, nPixelQuintuplets * sizeof(unsigned int), cudaMemcpyDeviceToHost,stream);
        cudaMemcpyAsync(pixelQuintupletsInCPU->T5Indices, pixelQuintupletsInGPU->T5Indices, nPixelQuintuplets * sizeof(unsigned int), cudaMemcpyDeviceToHost,stream);
        cudaMemcpyAsync(pixelQuintupletsInCPU->isDup, pixelQuintupletsInGPU->isDup, nPixelQuintuplets * sizeof(bool), cudaMemcpyDeviceToHost,stream);
        cudaMemcpyAsync(pixelQuintupletsInCPU->score, pixelQuintupletsInGPU->score, nPixelQuintuplets * sizeof(float), cudaMemcpyDeviceToHost,stream);
cudaStreamSynchronize(stream);
    }
    return pixelQuintupletsInCPU;
}
#else
SDL::pixelQuintuplets* SDL::Event::getPixelQuintuplets()
{
    return pixelQuintupletsInGPU;
}
#endif

#ifdef Explicit_Track
SDL::trackCandidates* SDL::Event::getTrackCandidates()
{
    if(trackCandidatesInCPU == nullptr)
    {
        trackCandidatesInCPU = new SDL::trackCandidates;
        trackCandidatesInCPU->nTrackCandidates = new unsigned int;
        cudaMemcpyAsync(trackCandidatesInCPU->nTrackCandidates, trackCandidatesInGPU->nTrackCandidates, sizeof(unsigned int), cudaMemcpyDeviceToHost,stream);
cudaStreamSynchronize(stream);
        unsigned int nTrackCandidates = *(trackCandidatesInCPU->nTrackCandidates);

        trackCandidatesInCPU->objectIndices = new unsigned int[2 * nTrackCandidates];
        trackCandidatesInCPU->trackCandidateType = new short[nTrackCandidates];
#ifdef TRACK_EXTENSIONS
        trackCandidatesInCPU->partOfExtension = new bool[nTrackCandidates];
        trackCandidatesInCPU->hitIndices = new unsigned int[14 * nTrackCandidates];
        trackCandidatesInCPU->logicalLayers = new unsigned int[7 * nTrackCandidates];

        cudaMemcpyAsync(trackCandidatesInCPU->partOfExtension, trackCandidatesInGPU->partOfExtension, nTrackCandidates * sizeof(bool), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(trackCandidatesInCPU->hitIndices, trackCandidatesInGPU->hitIndices, 14 * nTrackCandidates * sizeof(unsigned int), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(trackCandidatesInCPU->logicalLayers, trackCandidatesInGPU->logicalLayers, 7 * nTrackCandidates * sizeof(unsigned int), cudaMemcpyDeviceToHost, stream);
#endif
        cudaMemcpyAsync(trackCandidatesInCPU->objectIndices, trackCandidatesInGPU->objectIndices, 2 * nTrackCandidates * sizeof(unsigned int), cudaMemcpyDeviceToHost,stream);                                                                                    
        cudaMemcpyAsync(trackCandidatesInCPU->trackCandidateType, trackCandidatesInGPU->trackCandidateType, nTrackCandidates * sizeof(short), cudaMemcpyDeviceToHost,stream);                                                                                                                
cudaStreamSynchronize(stream);
    }
    return trackCandidatesInCPU;
}
#else
SDL::trackCandidates* SDL::Event::getTrackCandidates()
{
    return trackCandidatesInGPU;
}
#endif
#ifdef Explicit_Module
SDL::modules* SDL::Event::getFullModules()
{
    if(modulesInCPUFull == nullptr)
    {
        modulesInCPUFull = new SDL::modules;
        unsigned int nLowerModules;
        cudaMemcpyAsync(&nLowerModules, modulesInGPU->nLowerModules, sizeof(unsigned int), cudaMemcpyDeviceToHost,stream);
cudaStreamSynchronize(stream);

    modulesInCPUFull->detIds = new unsigned int[nModules];
    modulesInCPUFull->moduleMap = new unsigned int[40*nModules];
    modulesInCPUFull->nConnectedModules = new unsigned int[nModules];
    modulesInCPUFull->drdzs = new float[nModules];
    modulesInCPUFull->slopes = new float[nModules];
    modulesInCPUFull->nModules = new unsigned int[1];
    modulesInCPUFull->nLowerModules = new unsigned int[1];
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

    modulesInCPUFull->lowerModuleIndices = new unsigned int[nLowerModules+1];
    modulesInCPUFull->reverseLookupLowerModuleIndices = new int[nModules];

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


    cudaMemcpyAsync(modulesInCPUFull->reverseLookupLowerModuleIndices, modulesInGPU->reverseLookupLowerModuleIndices, nModules * sizeof(int), cudaMemcpyDeviceToHost,stream);
    cudaMemcpyAsync(modulesInCPUFull->lowerModuleIndices, modulesInGPU->lowerModuleIndices, (nLowerModules+1) * sizeof(unsigned int), cudaMemcpyDeviceToHost,stream);

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
        unsigned int nLowerModules;
        cudaMemcpyAsync(&nLowerModules, modulesInGPU->nLowerModules, sizeof(unsigned int), cudaMemcpyDeviceToHost,stream);
cudaStreamSynchronize(stream);
        modulesInCPU->nLowerModules = new unsigned int[1];
        modulesInCPU->nModules = new unsigned int[1];
        modulesInCPU->lowerModuleIndices = new unsigned int[nLowerModules+1];
        modulesInCPU->detIds = new unsigned int[nModules];
        modulesInCPU->isLower = new bool[nModules];
        modulesInCPU->layers = new short[nModules];
        modulesInCPU->subdets = new short[nModules];
        modulesInCPU->rings = new short[nModules];


        cudaMemcpyAsync(modulesInCPU->nLowerModules, modulesInGPU->nLowerModules, sizeof(unsigned int), cudaMemcpyDeviceToHost,stream);
        cudaMemcpyAsync(modulesInCPU->nModules, modulesInGPU->nModules, sizeof(unsigned int), cudaMemcpyDeviceToHost,stream);
        cudaMemcpyAsync(modulesInCPU->lowerModuleIndices, modulesInGPU->lowerModuleIndices, (nLowerModules+1) * sizeof(unsigned int), cudaMemcpyDeviceToHost,stream);
        cudaMemcpyAsync(modulesInCPU->detIds, modulesInGPU->detIds, nModules * sizeof(unsigned int), cudaMemcpyDeviceToHost,stream);
        cudaMemcpyAsync(modulesInCPU->isLower, modulesInGPU->isLower, nModules * sizeof(bool), cudaMemcpyDeviceToHost,stream);
        cudaMemcpyAsync(modulesInCPU->layers, modulesInGPU->layers, nModules * sizeof(short), cudaMemcpyDeviceToHost,stream);
        cudaMemcpyAsync(modulesInCPU->subdets, modulesInGPU->subdets, nModules * sizeof(short), cudaMemcpyDeviceToHost,stream);
        cudaMemcpyAsync(modulesInCPU->rings, modulesInGPU->rings, nModules * sizeof(short), cudaMemcpyDeviceToHost,stream);
cudaStreamSynchronize(stream);
    }
    return modulesInCPU;
}
#else
SDL::modules* SDL::Event::getModules()
{
    return modulesInGPU;
}
SDL::modules* SDL::Event::getFullModules()
{
    return modulesInGPU;
}
#endif

#ifdef Explicit_Extensions
SDL::trackExtensions* SDL::Event::getTrackExtensions()
{
   if(trackExtensionsInCPU == nullptr)
   {
       trackExtensionsInCPU = new SDL::trackExtensions;
       unsigned int nTrackCandidates;
       cudaMemcpyAsync(&nTrackCandidates, trackCandidatesInGPU->nTrackCandidates, sizeof(unsigned int), cudaMemcpyDeviceToHost, stream);
       cudaStreamSynchronize(stream);
       unsigned int maxTrackExtensions = nTrackCandidates * N_MAX_TRACK_EXTENSIONS_PER_TC;
#ifdef T3T3_EXTENSIONS
       maxTrackExtensions += N_MAX_T3T3_TRACK_EXTENSIONS;
       nTrackCandidates++;
#endif
       std::cout<<"nTrackCandidates = "<<nTrackCandidates<<std::endl;
       trackExtensionsInCPU->nTrackExtensions = new unsigned int[nTrackCandidates];
       trackExtensionsInCPU->constituentTCTypes = new short[3 * maxTrackExtensions];
       trackExtensionsInCPU->constituentTCIndices = new unsigned int[3 * maxTrackExtensions];
       trackExtensionsInCPU->nLayerOverlaps = new unsigned int[2 * maxTrackExtensions];
       trackExtensionsInCPU->nHitOverlaps = new unsigned int[2 * maxTrackExtensions];
       trackExtensionsInCPU->isDup = new bool[maxTrackExtensions];
        trackExtensionsInCPU->regressionRadius = new float[maxTrackExtensions];

       cudaMemcpyAsync(trackExtensionsInCPU->nTrackExtensions, trackExtensionsInGPU->nTrackExtensions, nTrackCandidates * sizeof(unsigned int), cudaMemcpyDeviceToHost, stream);
       cudaMemcpy(trackExtensionsInCPU->constituentTCTypes, trackExtensionsInGPU->constituentTCTypes, 3 * maxTrackExtensions * sizeof(short), cudaMemcpyDeviceToHost);
       cudaMemcpyAsync(trackExtensionsInCPU->constituentTCIndices, trackExtensionsInGPU->constituentTCIndices, 3 * maxTrackExtensions * sizeof(unsigned int), cudaMemcpyDeviceToHost, stream);

       cudaMemcpyAsync(trackExtensionsInCPU->nLayerOverlaps, trackExtensionsInGPU->nLayerOverlaps, 2 * maxTrackExtensions * sizeof(unsigned int), cudaMemcpyDeviceToHost, stream);
       cudaMemcpyAsync(trackExtensionsInCPU->nHitOverlaps, trackExtensionsInGPU->nHitOverlaps, 2 * maxTrackExtensions * sizeof(unsigned int), cudaMemcpyDeviceToHost, stream);
       cudaMemcpyAsync(trackExtensionsInCPU->isDup, trackExtensionsInGPU->isDup, maxTrackExtensions * sizeof(bool), cudaMemcpyDeviceToHost, stream);
       cudaMemcpyAsync(trackExtensionsInCPU->regressionRadius, trackExtensionsInGPU->regressionRadius, maxTrackExtensions * sizeof(float), cudaMemcpyDeviceToHost, stream);
       cudaStreamSynchronize(stream);
   }

   return trackExtensionsInCPU;
}
#else
SDL::trackExtensions* SDL::Event::getTrackExtensions()
{
    return trackExtensionsInGPU;
}
#endif


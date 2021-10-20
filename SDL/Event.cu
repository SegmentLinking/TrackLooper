# include "Event.cuh"
#include "allocate.h"

struct SDL::modules* SDL::modulesInGPU = nullptr;
struct SDL::pixelMap* SDL::pixelMapping = nullptr;
unsigned int SDL::nModules;

SDL::Event::Event()
{
    hitsInGPU = nullptr;
    mdsInGPU = nullptr;
    segmentsInGPU = nullptr;
    tripletsInGPU = nullptr;
    quintupletsInGPU = nullptr;
    trackCandidatesInGPU = nullptr;
    pixelTripletsInGPU = nullptr;
    pixelQuintupletsInGPU = nullptr;
    trackExtensionsInGPU = nullptr;

    hitsInCPU = nullptr;
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
    resetObjectsInModule();

}

SDL::Event::~Event()
{

#ifdef CACHE_ALLOC
    mdsInGPU->freeMemoryCache();
    segmentsInGPU->freeMemoryCache();
    tripletsInGPU->freeMemoryCache();
    trackCandidatesInGPU->freeMemoryCache();
    hitsInGPU->freeMemoryCache();
#ifdef FINAL_T5
    quintupletsInGPU->freeMemoryCache();
#endif
#else
    hitsInGPU->freeMemory();
    mdsInGPU->freeMemory();
    segmentsInGPU->freeMemory();
    tripletsInGPU->freeMemory();
    trackCandidatesInGPU->freeMemory();
    trackExtensionsInGPU->freeMemory();
#ifdef FINAL_T5
    quintupletsInGPU->freeMemory();
#endif
#endif
    cudaFreeHost(mdsInGPU);
    cudaFreeHost(segmentsInGPU);
    cudaFreeHost(tripletsInGPU);
    cudaFreeHost(trackCandidatesInGPU);
    cudaFreeHost(trackExtensionsInGPU);
    cudaFreeHost(hitsInGPU);

    pixelTripletsInGPU->freeMemory();
    cudaFreeHost(pixelTripletsInGPU);
    pixelQuintupletsInGPU->freeMemory();
    cudaFreeHost(pixelQuintupletsInGPU);

#ifdef FINAL_T5
    cudaFreeHost(quintupletsInGPU);
#endif

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
        delete[] tripletsInCPU->logicalLayers;
        delete[] tripletsInCPU->lowerModuleIndices;
        delete[] tripletsInCPU->hitIndices;
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

#ifdef Explicit_Track
    if(trackCandidatesInCPU != nullptr)
    {
        delete[] trackCandidatesInCPU->objectIndices;
        delete[] trackCandidatesInCPU->trackCandidateType;
        delete[] trackCandidatesInCPU->nTrackCandidates;
        delete[] trackCandidatesInCPU->logicalLayers;
        delete[] trackCandidatesInCPU->hitIndices;
        delete[] trackCandidatesInCPU->lowerModuleIndices;
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
    }
#endif
#ifdef Explicit_Module
    if(modulesInCPU != nullptr)
    {
        delete[] modulesInCPU->nLowerModules;
        delete[] modulesInCPU->nModules;
        delete[] modulesInCPU->lowerModuleIndices;
        delete[] modulesInCPU->detIds;
        delete[] modulesInCPU->hitRanges;
        delete[] modulesInCPU->isLower;
        delete[] modulesInCPU->trackCandidateModuleIndices;
        delete[] modulesInCPU->quintupletModuleIndices;
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

        delete[] modulesInCPUFull->hitRanges;
        delete[] modulesInCPUFull->mdRanges;
        delete[] modulesInCPUFull->segmentRanges;
        delete[] modulesInCPUFull->tripletRanges;
        delete[] modulesInCPUFull->trackCandidateRanges;

        delete[] modulesInCPUFull->moduleType;
        delete[] modulesInCPUFull->moduleLayerType;

        delete[] modulesInCPUFull->lowerModuleIndices;
        delete[] modulesInCPUFull->reverseLookupLowerModuleIndices;
        delete[] modulesInCPUFull->trackCandidateModuleIndices;
        delete[] modulesInCPUFull->quintupletModuleIndices;
        delete[] modulesInCPUFull;
    }
#endif
}

void SDL::initModules(const char* moduleMetaDataFilePath)
{
    if(modulesInGPU == nullptr)
    {
        cudaMallocHost(&modulesInGPU, sizeof(struct SDL::modules));
        //pixelMapping = new pixelMap;
        cudaMallocHost(&pixelMapping, sizeof(struct SDL::pixelMap));
        loadModulesFromFile(*modulesInGPU,nModules,*pixelMapping,moduleMetaDataFilePath); //nModules gets filled here
    }
    resetObjectRanges(*modulesInGPU,nModules);
}

void SDL::cleanModules()
{
  #ifdef CACHE_ALLOC
  freeModulesCache(*modulesInGPU,*pixelMapping);
  #else
  freeModules(*modulesInGPU,*pixelMapping);
  #endif
  cudaFreeHost(modulesInGPU);
  cudaFreeHost(pixelMapping);
//cudaDeviceReset(); // uncomment for leak check "cuda-memcheck --leak-check full --show-backtrace yes" does not work with caching.
}

void SDL::Event::resetObjectsInModule()
{
    resetObjectRanges(*modulesInGPU,nModules);
}

// Best working hit loading method. Previously named OMP
void SDL::Event::addHitToEvent(std::vector<float> x, std::vector<float> y, std::vector<float> z, std::vector<unsigned int> detId, std::vector<unsigned int> idxInNtuple)
{
    const int loopsize = x.size();// use the actual number of hits instead of a "max"

    if(hitsInGPU == nullptr)
    {

        cudaMallocHost(&hitsInGPU, sizeof(SDL::hits));
        #ifdef Explicit_Hit
    	  createHitsInExplicitMemory(*hitsInGPU, 2*loopsize); //unclear why but this has to be 2*loopsize to avoid crashing later (reported in tracklet allocation). seems to do with nHits values as well. this allows nhits to be set to the correct value of loopsize to get correct results without crashing. still beats the "max hits" so i think this is fine.
        #else
        createHitsInUnifiedMemory(*hitsInGPU,2*loopsize,0);
        #endif
    }


    float* host_x = &x[0]; // convert from std::vector to host array easily since vectors are ordered
    float* host_y = &y[0];
    float* host_z = &z[0];
    float* host_phis;
    float* host_etas;
    unsigned int* host_detId = &detId[0];
    unsigned int* host_idxs = &idxInNtuple[0];
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
    cudaMemcpy(module_layers,modulesInGPU->layers,nModules*sizeof(short),cudaMemcpyDeviceToHost);
    cudaMemcpy(module_subdet,modulesInGPU->subdets,nModules*sizeof(short),cudaMemcpyDeviceToHost);
    cudaMemcpy(module_hitRanges,modulesInGPU->hitRanges,nModules*2*sizeof(int),cudaMemcpyDeviceToHost);
    cudaMemcpy(module_moduleType,modulesInGPU->moduleType,nModules*sizeof(ModuleType),cudaMemcpyDeviceToHost);


  for (int ihit=0; ihit<loopsize;ihit++){
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
    cudaMemcpy(hitsInGPU->xs,host_x,loopsize*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(hitsInGPU->ys,host_y,loopsize*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(hitsInGPU->zs,host_z,loopsize*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(hitsInGPU->rts,host_rts,loopsize*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(hitsInGPU->phis,host_phis,loopsize*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(hitsInGPU->etas,host_etas,loopsize*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(hitsInGPU->idxs,host_idxs,loopsize*sizeof(unsigned int),cudaMemcpyHostToDevice);
    cudaMemcpy(hitsInGPU->moduleIndices,host_moduleIndex,loopsize*sizeof(unsigned int),cudaMemcpyHostToDevice);
    cudaMemcpy(hitsInGPU->highEdgeXs,host_highEdgeXs,loopsize*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(hitsInGPU->highEdgeYs,host_highEdgeYs,loopsize*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(hitsInGPU->lowEdgeXs,host_lowEdgeXs,loopsize*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(hitsInGPU->lowEdgeYs,host_lowEdgeYs,loopsize*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(hitsInGPU->nHits,&loopsize,sizeof(unsigned int),cudaMemcpyHostToDevice);// value can't correctly be set in hit allocation
    cudaMemcpy(modulesInGPU->hitRanges,module_hitRanges,nModules*2*sizeof(int),cudaMemcpyHostToDevice);// value can't correctly be set in hit allocation
    cudaDeviceSynchronize(); //doesn't seem to make a difference

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
    	createMDsInExplicitMemory(*mdsInGPU, N_MAX_MD_PER_MODULES, nModules, N_MAX_PIXEL_MD_PER_MODULES);
#else
    	createMDsInUnifiedMemory(*mdsInGPU, N_MAX_MD_PER_MODULES, nModules, N_MAX_PIXEL_MD_PER_MODULES);
#endif
    }
    if(segmentsInGPU == nullptr)
    {
        cudaMallocHost(&segmentsInGPU, sizeof(SDL::segments));
#ifdef Explicit_Seg
        createSegmentsInExplicitMemory(*segmentsInGPU, N_MAX_SEGMENTS_PER_MODULE, nModules, N_MAX_PIXEL_SEGMENTS_PER_MODULE);
#else
        createSegmentsInUnifiedMemory(*segmentsInGPU, N_MAX_SEGMENTS_PER_MODULE, nModules, N_MAX_PIXEL_SEGMENTS_PER_MODULE);
#endif
    }
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

    cudaMemcpy(hitIndices0_dev,hitIndices0_host,size*sizeof(unsigned int),cudaMemcpyHostToDevice);
    cudaMemcpy(hitIndices1_dev,hitIndices1_host,size*sizeof(unsigned int),cudaMemcpyHostToDevice);
    cudaMemcpy(hitIndices2_dev,hitIndices2_host,size*sizeof(unsigned int),cudaMemcpyHostToDevice);
    cudaMemcpy(hitIndices3_dev,hitIndices3_host,size*sizeof(unsigned int),cudaMemcpyHostToDevice);
    cudaMemcpy(dPhiChange_dev,dPhiChange_host,size*sizeof(unsigned int),cudaMemcpyHostToDevice);
    cudaMemcpy(ptIn_dev,ptIn_host,size*sizeof(unsigned int),cudaMemcpyHostToDevice);
    cudaMemcpy(ptErr_dev,ptErr_host,size*sizeof(unsigned int),cudaMemcpyHostToDevice);
    cudaMemcpy(px_dev,px_host,size*sizeof(unsigned int),cudaMemcpyHostToDevice);
    cudaMemcpy(py_dev,py_host,size*sizeof(unsigned int),cudaMemcpyHostToDevice);
    cudaMemcpy(pz_dev,pz_host,size*sizeof(unsigned int),cudaMemcpyHostToDevice);
    cudaMemcpy(etaErr_dev,etaErr_host,size*sizeof(unsigned int),cudaMemcpyHostToDevice);
    cudaMemcpy(eta_dev, eta_host, size*sizeof(unsigned int),cudaMemcpyHostToDevice);
    cudaMemcpy(phi_dev, phi_host, size*sizeof(unsigned int),cudaMemcpyHostToDevice);
    cudaMemcpy(superbin_dev,superbin_host,size*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(pixelType_dev,pixelType_host,size*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(isQuad_dev,isQuad_host,size*sizeof(short),cudaMemcpyHostToDevice);

    unsigned int nThreads = 256;
    unsigned int nBlocks =  size % nThreads == 0 ? size/nThreads : size/nThreads + 1;

    addPixelSegmentToEventKernel<<<nBlocks,nThreads>>>(hitIndices0_dev,hitIndices1_dev,hitIndices2_dev,hitIndices3_dev,dPhiChange_dev,ptIn_dev,ptErr_dev,px_dev,py_dev,pz_dev,eta_dev, etaErr_dev, phi_dev, pixelModuleIndex, *modulesInGPU,*hitsInGPU,*mdsInGPU,*segmentsInGPU,size, superbin_dev, pixelType_dev,isQuad_dev);
   //std::cout<<"Number of pixel segments = "<<size<<std::endl;
   cudaDeviceSynchronize();
   cudaMemcpy(&(segmentsInGPU->nSegments)[pixelModuleIndex], &size, sizeof(unsigned int), cudaMemcpyHostToDevice);
   unsigned int mdSize = 2 * size;
   cudaMemcpy(&(mdsInGPU->nMDs)[pixelModuleIndex], &mdSize, sizeof(unsigned int), cudaMemcpyHostToDevice);

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
}

void SDL::Event::addMiniDoubletsToEvent()
{
    unsigned int idx;
    for(unsigned int i = 0; i<*(SDL::modulesInGPU->nLowerModules); i++)
    {
        idx = SDL::modulesInGPU->lowerModuleIndices[i];
        if(mdsInGPU->nMDs[idx] == 0 or modulesInGPU->hitRanges[idx * 2] == -1)
        {
            modulesInGPU->mdRanges[idx * 2] = -1;
            modulesInGPU->mdRanges[idx * 2 + 1] = -1;
        }
        else
        {
            modulesInGPU->mdRanges[idx * 2] = idx * N_MAX_MD_PER_MODULES;
            modulesInGPU->mdRanges[idx * 2 + 1] = (idx * N_MAX_MD_PER_MODULES) + mdsInGPU->nMDs[idx] - 1;

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
cudaMemcpy(&nLowerModules,modulesInGPU->nLowerModules,sizeof(unsigned int),cudaMemcpyDeviceToHost);
unsigned int* nMDsCPU;
cudaMallocHost(&nMDsCPU, nModules * sizeof(unsigned int));
cudaMemcpy(nMDsCPU,mdsInGPU->nMDs,nModules*sizeof(unsigned int),cudaMemcpyDeviceToHost);

short* module_subdets;
cudaMallocHost(&module_subdets, nModules* sizeof(short));
cudaMemcpy(module_subdets,modulesInGPU->subdets,nModules*sizeof(short),cudaMemcpyDeviceToHost);
unsigned int* module_lowerModuleIndices;
cudaMallocHost(&module_lowerModuleIndices, (nLowerModules +1)* sizeof(unsigned int));
cudaMemcpy(module_lowerModuleIndices,modulesInGPU->lowerModuleIndices,(nLowerModules+1)*sizeof(unsigned int),cudaMemcpyDeviceToHost);
int* module_mdRanges;
cudaMallocHost(&module_mdRanges, nModules* 2*sizeof(int));
cudaMemcpy(module_mdRanges,modulesInGPU->mdRanges,nModules*2*sizeof(int),cudaMemcpyDeviceToHost);
short* module_layers;
cudaMallocHost(&module_layers, nModules * sizeof(short));
cudaMemcpy(module_layers,modulesInGPU->layers,nModules*sizeof(short),cudaMemcpyDeviceToHost);
int* module_hitRanges;
cudaMallocHost(&module_hitRanges, nModules* 2*sizeof(int));
cudaMemcpy(module_hitRanges,modulesInGPU->hitRanges,nModules*2*sizeof(int),cudaMemcpyDeviceToHost);

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
cudaMemcpy(modulesInGPU->mdRanges,module_mdRanges,nModules*2*sizeof(int),cudaMemcpyHostToDevice);
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
            modulesInGPU->segmentRanges[idx * 2] = -1;
            modulesInGPU->segmentRanges[idx * 2 + 1] = -1;
        }
        else
        {
            modulesInGPU->segmentRanges[idx * 2] = idx * N_MAX_SEGMENTS_PER_MODULE;
            modulesInGPU->segmentRanges[idx * 2 + 1] = idx * N_MAX_SEGMENTS_PER_MODULE + segmentsInGPU->nSegments[idx] - 1;

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
cudaMemcpy(&nLowerModules,modulesInGPU->nLowerModules,sizeof(unsigned int),cudaMemcpyDeviceToHost);

unsigned int* nSegmentsCPU;
cudaMallocHost(&nSegmentsCPU, nModules * sizeof(unsigned int));
cudaMemcpy(nSegmentsCPU,segmentsInGPU->nSegments,nModules*sizeof(unsigned int),cudaMemcpyDeviceToHost);

short* module_subdets;
cudaMallocHost(&module_subdets, nModules* sizeof(short));
cudaMemcpy(module_subdets,modulesInGPU->subdets,nModules*sizeof(short),cudaMemcpyDeviceToHost);
unsigned int* module_lowerModuleIndices;
cudaMallocHost(&module_lowerModuleIndices, (nLowerModules +1)* sizeof(unsigned int));
cudaMemcpy(module_lowerModuleIndices,modulesInGPU->lowerModuleIndices,(nLowerModules+1)*sizeof(unsigned int),cudaMemcpyDeviceToHost);
int* module_segmentRanges;
cudaMallocHost(&module_segmentRanges, nModules* 2*sizeof(int));
cudaMemcpy(module_segmentRanges,modulesInGPU->segmentRanges,nModules*2*sizeof(int),cudaMemcpyDeviceToHost);
short* module_layers;
cudaMallocHost(&module_layers, nModules * sizeof(short));
cudaMemcpy(module_layers,modulesInGPU->layers,nModules*sizeof(short),cudaMemcpyDeviceToHost);
    unsigned int idx;
    for(unsigned int i = 0; i<nLowerModules; i++)
    {
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
    cudaDeviceSynchronize();
    auto memStart = std::chrono::high_resolution_clock::now();
    if(mdsInGPU == nullptr)
    {
        cudaMallocHost(&mdsInGPU, sizeof(SDL::miniDoublets));
#ifdef Explicit_MD
        //FIXME: Add memory locations for pixel MDs
    	createMDsInExplicitMemory(*mdsInGPU, N_MAX_MD_PER_MODULES, nModules, N_MAX_PIXEL_MD_PER_MODULES);
#else
    	createMDsInUnifiedMemory(*mdsInGPU, N_MAX_MD_PER_MODULES, nModules, N_MAX_PIXEL_MD_PER_MODULES);
#endif
    }
    cudaDeviceSynchronize();
    auto memStop = std::chrono::high_resolution_clock::now();
    auto memDuration = std::chrono::duration_cast<std::chrono::milliseconds>(memStop - memStart); //in milliseconds

    unsigned int nLowerModules;
    cudaMemcpy(&nLowerModules,modulesInGPU->nLowerModules,sizeof(unsigned int),cudaMemcpyDeviceToHost);

    int maxThreadsPerModule=0;
    #ifdef Explicit_Module
    unsigned int* module_lowerModuleIndices;
    cudaMallocHost(&module_lowerModuleIndices, (nLowerModules +1)* sizeof(unsigned int));
    cudaMemcpy(module_lowerModuleIndices,modulesInGPU->lowerModuleIndices,(nLowerModules+1)*sizeof(unsigned int),cudaMemcpyDeviceToHost);
    int* module_hitRanges;
    cudaMallocHost(&module_hitRanges, nModules* 2*sizeof(int));
    cudaMemcpy(module_hitRanges,modulesInGPU->hitRanges,nModules*2*sizeof(int),cudaMemcpyDeviceToHost);
    bool* module_isLower;
    cudaMallocHost(&module_isLower, nModules*sizeof(bool));
    cudaMemcpy(module_isLower,modulesInGPU->isLower,nModules*sizeof(bool),cudaMemcpyDeviceToHost);
    bool* module_isInverted;
    cudaMallocHost(&module_isInverted, nModules*sizeof(bool));
    cudaMemcpy(module_isInverted,modulesInGPU->isInverted,nModules*sizeof(bool),cudaMemcpyDeviceToHost);

    for (int i=0; i<nLowerModules; i++) {
      int lowerModuleIndex = module_lowerModuleIndices[i];
      int upperModuleIndex = modulesInGPU->partnerModuleIndexExplicit(lowerModuleIndex,module_isLower[lowerModuleIndex],module_isInverted[lowerModuleIndex]);
      int lowerHitRanges = module_hitRanges[lowerModuleIndex*2];
      int upperHitRanges = module_hitRanges[upperModuleIndex*2];
      if(lowerHitRanges!=-1&&upperHitRanges!=-1) {
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
    //int maxThreadsPerModule=0;
    for (int i=0; i<nLowerModules; i++) {
      int lowerModuleIndex = modulesInGPU->lowerModuleIndices[i];
      int upperModuleIndex = modulesInGPU->partnerModuleIndex(lowerModuleIndex);
      int lowerHitRanges = modulesInGPU->hitRanges[lowerModuleIndex*2];
      int upperHitRanges = modulesInGPU->hitRanges[upperModuleIndex*2];
      if(lowerHitRanges!=-1&&upperHitRanges!=-1) {
        unsigned int nLowerHits = modulesInGPU->hitRanges[lowerModuleIndex * 2 + 1] - lowerHitRanges + 1;
        unsigned int nUpperHits = modulesInGPU->hitRanges[upperModuleIndex * 2 + 1] - upperHitRanges + 1;
        maxThreadsPerModule = maxThreadsPerModule > (nLowerHits*nUpperHits) ? maxThreadsPerModule : nLowerHits*nUpperHits;
      }
    }
    #endif
    //printf("maxThreadsPerModule=%d\n", maxThreadsPerModule);
    dim3 nThreads(1,128);
    dim3 nBlocks((nLowerModules % nThreads.x == 0 ? nLowerModules/nThreads.x : nLowerModules/nThreads.x + 1), (maxThreadsPerModule % nThreads.y == 0 ? maxThreadsPerModule/nThreads.y : maxThreadsPerModule/nThreads.y + 1));

    cudaDeviceSynchronize();
    auto syncStart = std::chrono::high_resolution_clock::now();

    createMiniDoubletsInGPU<<<nBlocks,nThreads>>>(*modulesInGPU,*hitsInGPU,*mdsInGPU);

    cudaError_t cudaerr = cudaGetLastError(); 
    if(cudaerr != cudaSuccess)
    {
        std::cout<<"sync failed with error : "<<cudaGetErrorString(cudaerr)<<std::endl;
    }cudaDeviceSynchronize();
    auto syncStop = std::chrono::high_resolution_clock::now();

    auto syncDuration =  std::chrono::duration_cast<std::chrono::milliseconds>(syncStop - syncStart);


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
    if(segmentsInGPU == nullptr)
    {
        cudaMallocHost(&segmentsInGPU, sizeof(SDL::segments));
#ifdef Explicit_Seg
        createSegmentsInExplicitMemory(*segmentsInGPU, N_MAX_SEGMENTS_PER_MODULE, nModules, N_MAX_PIXEL_SEGMENTS_PER_MODULE);
#else
        createSegmentsInUnifiedMemory(*segmentsInGPU, N_MAX_SEGMENTS_PER_MODULE, nModules, N_MAX_PIXEL_SEGMENTS_PER_MODULE);
#endif
    }
    unsigned int nLowerModules;
    cudaMemcpy(&nLowerModules,modulesInGPU->nLowerModules,sizeof(unsigned int),cudaMemcpyDeviceToHost);

    int max_cModules=0;
    int sq_max_nMDs = 0;
    int nonZeroModules = 0;
  #ifdef Explicit_Module
    unsigned int nModules;
    cudaMemcpy(&nModules,modulesInGPU->nModules,sizeof(unsigned int),cudaMemcpyDeviceToHost);
    unsigned int* nMDs = (unsigned int*)malloc(nModules*sizeof(unsigned int));
    cudaMemcpy((void *)nMDs, mdsInGPU->nMDs, nModules*sizeof(unsigned int), cudaMemcpyDeviceToHost);

    unsigned int* module_lowerModuleIndices;
    cudaMallocHost(&module_lowerModuleIndices, (nLowerModules +1)* sizeof(unsigned int));
    cudaMemcpy(module_lowerModuleIndices,modulesInGPU->lowerModuleIndices,(nLowerModules+1)*sizeof(unsigned int),cudaMemcpyDeviceToHost);
    unsigned int* module_nConnectedModules;
    cudaMallocHost(&module_nConnectedModules, nModules* sizeof(unsigned int));
    cudaMemcpy(module_nConnectedModules,modulesInGPU->nConnectedModules,nModules*sizeof(unsigned int),cudaMemcpyDeviceToHost);
    unsigned int* module_moduleMap;
    cudaMallocHost(&module_moduleMap, nModules*40* sizeof(unsigned int));
    cudaMemcpy(module_moduleMap,modulesInGPU->moduleMap,nModules*40*sizeof(unsigned int),cudaMemcpyDeviceToHost);

    for (int i=0; i<nLowerModules; i++) {
      unsigned int innerLowerModuleIndex = module_lowerModuleIndices[i];
      unsigned int nConnectedModules = module_nConnectedModules[innerLowerModuleIndex];
      unsigned int nInnerMDs = nMDs[innerLowerModuleIndex];
      max_cModules = max_cModules > nConnectedModules ? max_cModules : nConnectedModules;
      int limit_local = 0;
      if (nConnectedModules!=0) nonZeroModules++;
      for (int j=0; j<nConnectedModules; j++) {
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

    unsigned int nModules = *modulesInGPU->nModules;
    unsigned int* nMDs = (unsigned int*)malloc(nModules*sizeof(unsigned int));
    cudaMemcpy((void *)nMDs, mdsInGPU->nMDs, nModules*sizeof(unsigned int), cudaMemcpyDeviceToHost);
    for (int i=0; i<nLowerModules; i++) {
      unsigned int innerLowerModuleIndex = modulesInGPU->lowerModuleIndices[i];
      unsigned int nConnectedModules = modulesInGPU->nConnectedModules[innerLowerModuleIndex];
      unsigned int nInnerMDs = nMDs[innerLowerModuleIndex] > N_MAX_MD_PER_MODULES ? N_MAX_MD_PER_MODULES : nMDs[innerLowerModuleIndex];
      max_cModules = max_cModules > nConnectedModules ? max_cModules : nConnectedModules;
      int limit_local = 0;
      if (nConnectedModules!=0) nonZeroModules++;
      for (int j=0; j<nConnectedModules; j++) {
        int outerLowerModuleIndex = modulesInGPU->moduleMap[innerLowerModuleIndex * MAX_CONNECTED_MODULES + j];
        int nOuterMDs = nMDs[outerLowerModuleIndex] > N_MAX_MD_PER_MODULES ? N_MAX_MD_PER_MODULES : nMDs[outerLowerModuleIndex];
        int total = nInnerMDs*nOuterMDs;
        limit_local = limit_local > total ? limit_local : total;
      }
      sq_max_nMDs = sq_max_nMDs > limit_local ? sq_max_nMDs : limit_local;
    }
  #endif
    //printf("max nConnectedModules=%d nonZeroModules=%d max sq_max_nMDs=%d\n", max_cModules, nonZeroModules, sq_max_nMDs);
    dim3 nThreads(256,1,1);
    dim3 nBlocks((sq_max_nMDs%nThreads.x==0 ? sq_max_nMDs/nThreads.x : sq_max_nMDs/nThreads.x + 1), (max_cModules%nThreads.y==0 ? max_cModules/nThreads.y : max_cModules/nThreads.y + 1), (nLowerModules%nThreads.z==0 ? nLowerModules/nThreads.z : nLowerModules/nThreads.z + 1));
    free(nMDs);

    createSegmentsInGPU<<<nBlocks,nThreads>>>(*modulesInGPU, *hitsInGPU, *mdsInGPU, *segmentsInGPU);

    cudaError_t cudaerr = cudaGetLastError();
    if(cudaerr != cudaSuccess)
    {
        std::cout<<"sync failed with error : "<<cudaGetErrorString(cudaerr)<<std::endl;
    }cudaDeviceSynchronize();
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
    cudaMemcpy(&nLowerModules,modulesInGPU->nLowerModules,sizeof(unsigned int),cudaMemcpyDeviceToHost);

    if(tripletsInGPU == nullptr)
    {
        cudaMallocHost(&tripletsInGPU, sizeof(SDL::triplets));
#ifdef Explicit_Trips
        createTripletsInExplicitMemory(*tripletsInGPU, N_MAX_TRIPLETS_PER_MODULE, nLowerModules);
#else
        createTripletsInUnifiedMemory(*tripletsInGPU, N_MAX_TRIPLETS_PER_MODULE, nLowerModules);
#endif
    }

  #ifdef Explicit_Module
    unsigned int nonZeroModules=0;
    unsigned int max_InnerSeg=0;
    unsigned int *index = (unsigned int*)malloc(nLowerModules*sizeof(unsigned int));
    unsigned int *index_gpu;
    cudaMalloc((void **)&index_gpu, nLowerModules*sizeof(unsigned int));
    //unsigned int nModules = *modulesInGPU->nModules;
    unsigned int *nSegments = (unsigned int*)malloc(nModules*sizeof(unsigned int));
    cudaMemcpy((void *)nSegments, segmentsInGPU->nSegments, nModules*sizeof(unsigned int), cudaMemcpyDeviceToHost);
    unsigned int* module_lowerModuleIndices;
    cudaMallocHost(&module_lowerModuleIndices, (nLowerModules +1)* sizeof(unsigned int));
    cudaMemcpy(module_lowerModuleIndices,modulesInGPU->lowerModuleIndices,(nLowerModules+1)*sizeof(unsigned int),cudaMemcpyDeviceToHost);
    unsigned int* module_nConnectedModules;
    cudaMallocHost(&module_nConnectedModules, nModules* sizeof(unsigned int));
    cudaMemcpy(module_nConnectedModules,modulesInGPU->nConnectedModules,nModules*sizeof(unsigned int),cudaMemcpyDeviceToHost);
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
    unsigned int nModules = *modulesInGPU->nModules;
    unsigned int *nSegments = (unsigned int*)malloc(nModules*sizeof(unsigned int));
    cudaMemcpy((void *)nSegments, segmentsInGPU->nSegments, nModules*sizeof(unsigned int), cudaMemcpyDeviceToHost);
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
    cudaMemcpy(index_gpu, index, nonZeroModules*sizeof(unsigned int), cudaMemcpyHostToDevice);
    int max_OuterSeg = 0;
    max_OuterSeg = N_MAX_SEGMENTS_PER_MODULE;
    dim3 nThreads(16,16,1);
    dim3 nBlocks((max_OuterSeg % nThreads.x == 0 ? max_OuterSeg / nThreads.x : max_OuterSeg / nThreads.x + 1),(max_InnerSeg % nThreads.y == 0 ? max_InnerSeg/nThreads.y : max_InnerSeg/nThreads.y + 1), (nonZeroModules % nThreads.z == 0 ? nonZeroModules/nThreads.z : nonZeroModules/nThreads.z + 1));
    createTripletsInGPU<<<nBlocks,nThreads>>>(*modulesInGPU, *hitsInGPU, *mdsInGPU, *segmentsInGPU, *tripletsInGPU, index_gpu);
    cudaError_t cudaerr =cudaGetLastError();
    if(cudaerr != cudaSuccess)
      {
	std::cout<<"sync failed with error : "<<cudaGetErrorString(cudaerr)<<std::endl;
      } cudaDeviceSynchronize();
    free(nSegments);
    free(index);
    cudaFree(index_gpu);

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
    cudaMemcpy(&nLowerModules,modulesInGPU->nLowerModules,sizeof(unsigned int),cudaMemcpyDeviceToHost);
    if(trackCandidatesInGPU == nullptr)
    {
        cudaMallocHost(&trackCandidatesInGPU, sizeof(SDL::trackCandidates));
#ifdef Explicit_Track
        createTrackCandidatesInExplicitMemory(*trackCandidatesInGPU, N_MAX_TRACK_CANDIDATES + N_MAX_PIXEL_TRACK_CANDIDATES);
#else
        createTrackCandidatesInUnifiedMemory(*trackCandidatesInGPU, N_MAX_TRACK_CANDIDATES +  N_MAX_PIXEL_TRACK_CANDIDATES);
#endif
    }

#ifdef FINAL_pT5
    printf("Adding pT5s to TC collection\n");
    unsigned int nThreadsx_pT5 = 1;
    unsigned int nBlocksx_pT5 = (N_MAX_PIXEL_QUINTUPLETS) % nThreadsx_pT5 == 0 ? N_MAX_PIXEL_QUINTUPLETS / nThreadsx_pT5 : N_MAX_PIXEL_QUINTUPLETS / nThreadsx_pT5 + 1;
    addpT5asTrackCandidateInGPU<<<nBlocksx_pT5, nThreadsx_pT5>>>(*modulesInGPU, *pixelQuintupletsInGPU, *trackCandidatesInGPU);
    cudaError_t cudaerr_pT5 = cudaGetLastError();
    if(cudaerr_pT5 != cudaSuccess)
    {
        std::cout<<"sync failed with error : "<<cudaGetErrorString(cudaerr_pT5)<<std::endl;
    }cudaDeviceSynchronize();
#endif
#ifdef FINAL_pT3
    printf("running final state pT3\n");
    unsigned int nThreadsx = 1;
    unsigned int nBlocksx = (N_MAX_PIXEL_TRIPLETS) % nThreadsx == 0 ? N_MAX_PIXEL_TRIPLETS / nThreadsx : N_MAX_PIXEL_TRIPLETS / nThreadsx + 1;
    cudaDeviceSynchronize();
    addpT3asTrackCandidateInGPU<<<nBlocksx, nThreadsx>>>(*modulesInGPU, *tripletsInGPU, *pixelTripletsInGPU, *trackCandidatesInGPU, *segmentsInGPU, *pixelQuintupletsInGPU);
    cudaError_t cudaerr_pT3 = cudaGetLastError();
    if(cudaerr_pT3 != cudaSuccess)
    {
        std::cout<<"sync failed with error : "<<cudaGetErrorString(cudaerr_pT3)<<std::endl;
    }cudaDeviceSynchronize();
#endif // final state pT2 and pT3

#ifdef FINAL_T5
    printf("running final state T5\n");
    dim3 nThreads(32,16,1);
    dim3 nBlocks(((nLowerModules) % nThreads.x == 0 ? (nLowerModules)/nThreads.x : (nLowerModules)/nThreads.x + 1),((N_MAX_QUINTUPLETS_PER_MODULE-1) % nThreads.y == 0 ? (N_MAX_QUINTUPLETS_PER_MODULE-1)/nThreads.y : (N_MAX_QUINTUPLETS_PER_MODULE-1)/nThreads.y + 1),1);
    dim3 dupThreads(32,32,1);
    dim3 dupBlocks(16,16,1);
    removeDupQuintupletsInGPU<<<dupBlocks,dupThreads>>>(*modulesInGPU, *hitsInGPU, *mdsInGPU, *segmentsInGPU, *tripletsInGPU, *quintupletsInGPU,true);
    cudaDeviceSynchronize();
    addT5asTrackCandidateInGPU<<<nBlocks,nThreads>>>(*modulesInGPU, *tripletsInGPU, *quintupletsInGPU,*trackCandidatesInGPU,*pixelQuintupletsInGPU,*pixelTripletsInGPU);

    cudaError_t cudaerr_T5 =cudaGetLastError(); 
    if(cudaerr_T5 != cudaSuccess)
    {
        std::cout<<"sync failed with error : "<<cudaGetErrorString(cudaerr_T5)<<std::endl;
    }cudaDeviceSynchronize();
#endif // final state T5
#ifdef FINAL_pLS
    printf("Adding pLSs to TC collection\n");
#ifdef DUP_pLS
    printf("cleaning pixels\n");
    checkHitspLS<<<64,1024>>>(*modulesInGPU,*mdsInGPU, *segmentsInGPU, *hitsInGPU, true);
    cudaError_t cudaerrpix = cudaGetLastError();
    if(cudaerrpix != cudaSuccess)
    {
        std::cout<<"sync failed with error : "<<cudaGetErrorString(cudaerrpix)<<std::endl;

    }cudaDeviceSynchronize();
#endif  
    unsigned int nThreadsx_pLS = 1;
    unsigned int nBlocksx_pLS = (20000) % nThreadsx_pLS == 0 ? 20000 / nThreadsx_pT5 : 20000 / nThreadsx_pT5 + 1;
    addpLSasTrackCandidateInGPU<<<nBlocksx, nThreadsx>>>(*modulesInGPU, *pixelTripletsInGPU, *trackCandidatesInGPU, *segmentsInGPU, *pixelQuintupletsInGPU,*mdsInGPU,*hitsInGPU,*quintupletsInGPU);
    cudaError_t cudaerr_pLS = cudaGetLastError();
    if(cudaerr_pLS != cudaSuccess)
    {
        std::cout<<"sync failed with error : "<<cudaGetErrorString(cudaerr_pLS)<<std::endl;
    }cudaDeviceSynchronize();
#endif
#if defined(AddObjects)
#endif
}

void SDL::Event::createExtendedTracks()
{
    if(trackExtensionsInGPU == nullptr)
    {
        cudaMallocHost(&trackExtensionsInGPU, sizeof(SDL::trackExtensions));
    }

    unsigned int nTrackCandidates;
    cudaMemcpy(&nTrackCandidates, trackCandidatesInGPU->nTrackCandidates, sizeof(unsigned int), cudaMemcpyDeviceToHost);

    //The +1 with nTrackCandidates is to store how many T3T3 extensions are produced
#ifdef Explicit_Extensions
    createTrackExtensionsInExplicitMemory(*trackExtensionsInGPU, nTrackCandidates * N_MAX_TRACK_EXTENSIONS_PER_TC + N_MAX_T3T3_TRACK_EXTENSIONS, nTrackCandidates + 1); 
#else
    createTrackExtensionsInUnifiedMemory(*trackExtensionsInGPU, nTrackCandidates * N_MAX_TRACK_EXTENSIONS_PER_TC + N_MAX_T3T3_TRACK_EXTENSIONS, nTrackCandidates + 1);
#endif
    unsigned int nLowerModules;    
    cudaMemcpy(&nLowerModules,modulesInGPU->nLowerModules,sizeof(unsigned int),cudaMemcpyDeviceToHost);
    std::cout<<"nLowerModules = "<<nLowerModules<<" nTrackCandidates = "<<nTrackCandidates<<std::endl;

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
   
/*    dim3 nThreadsT3T3(1,16,16);
    dim3 nBlocksT3T3(nLowerModules % nThreads.x == 0 ? nLowerModules / nThreads.x: nLowerModules / nThreads.x + 1, maxT3s % nThreads.y == 0 ? maxT3s / nThreads.y : maxT3s / nThreads.y + 1, maxT3s % nThreads.z == 0 ? maxT3s / nThreads.z : maxT3s / nThreads.z + 1);

    createT3T3ExtendedTracksInGPU<<<nBlocks, nThreads>>>(*modulesInGPU, *hitsInGPU, *mdsInGPU, *segmentsInGPU, *tripletsInGPU, *quintupletsInGPU, *pixelTripletsInGPU, *pixelQuintupletsInGPU, *trackCandidatesInGPU, *trackExtensionsInGPU, nTrackCandidates);
    cudaerr = cudaDeviceSynchronize();
    if(cudaerr != cudaSuccess)
    {
	    std::cout<<"sync failed with error : "<<cudaGetErrorString(cudaerr)<<std::endl;
    }


    int nThreadsDupCleaning = 512;
    int nBlocksDupCleaning = (nTrackCandidates % nThreadsDupCleaning == 0) ? nTrackCandidates / nThreadsDupCleaning : nTrackCandidates / nThreadsDupCleaning + 1;

    cleanDuplicateExtendedTracks<<<nThreadsDupCleaning, nBlocksDupCleaning>>>(*trackExtensionsInGPU, nTrackCandidates);
    cudaerr = cudaDeviceSynchronize();
    if(cudaerr != cudaSuccess)
    {
	    std::cout<<"sync failed with error : "<<cudaGetErrorString(cudaerr)<<std::endl;
    }*/

    cudaFreeHost(nTriplets);
}


void SDL::Event::createPixelTriplets()
{
    unsigned int nLowerModules;
    cudaMemcpy(&nLowerModules, modulesInGPU->nLowerModules, sizeof(unsigned int), cudaMemcpyDeviceToHost);

    if(pixelTripletsInGPU == nullptr)
    {
        cudaMallocHost(&pixelTripletsInGPU, sizeof(SDL::pixelTriplets));
    }
#ifdef Explicit_PT3
    createPixelTripletsInExplicitMemory(*pixelTripletsInGPU, N_MAX_PIXEL_TRIPLETS);
#else
    createPixelTripletsInUnifiedMemory(*pixelTripletsInGPU, N_MAX_PIXEL_TRIPLETS);
#endif

    unsigned int pixelModuleIndex;
    int* superbins;
    int* pixelTypes;
    unsigned int *nTriplets;
    unsigned int nModules;
    cudaMemcpy(&nModules,modulesInGPU->nModules,sizeof(unsigned int),cudaMemcpyDeviceToHost);
    pixelModuleIndex = nModules-1;
    unsigned int nInnerSegments = 0;
    cudaMemcpy(&nInnerSegments, &(segmentsInGPU->nSegments[pixelModuleIndex]), sizeof(unsigned int), cudaMemcpyDeviceToHost);
    nInnerSegments = std::min(nInnerSegments, N_MAX_PIXEL_SEGMENTS_PER_MODULE);

    cudaMallocHost(&nTriplets, nLowerModules * sizeof(unsigned int));
    cudaMemcpy(nTriplets, tripletsInGPU->nTriplets, nLowerModules * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    cudaMallocHost(&superbins,N_MAX_PIXEL_SEGMENTS_PER_MODULE*sizeof(int));
    cudaMallocHost(&pixelTypes,N_MAX_PIXEL_SEGMENTS_PER_MODULE*sizeof(int));

    cudaMemcpy(superbins,segmentsInGPU->superbin,N_MAX_PIXEL_SEGMENTS_PER_MODULE*sizeof(int),cudaMemcpyDeviceToHost);
    cudaMemcpy(pixelTypes,segmentsInGPU->pixelType,N_MAX_PIXEL_SEGMENTS_PER_MODULE*sizeof(int),cudaMemcpyDeviceToHost);

    unsigned int* connectedPixelSize_host;
    unsigned int* connectedPixelIndex_host;
    cudaMallocHost(&connectedPixelSize_host, nInnerSegments* sizeof(unsigned int));
    cudaMallocHost(&connectedPixelIndex_host, nInnerSegments* sizeof(unsigned int));
    unsigned int* connectedPixelSize_dev;
    unsigned int* connectedPixelIndex_dev;
    cudaMalloc(&connectedPixelSize_dev, nInnerSegments* sizeof(unsigned int));
    cudaMalloc(&connectedPixelIndex_dev, nInnerSegments* sizeof(unsigned int));
    unsigned int max_size =0;
    int threadSize = 1000000;
    unsigned int *segs_pix = (unsigned int*)malloc(2*threadSize*sizeof(unsigned int));
    unsigned int *segs_pix_offset = segs_pix+threadSize;
    unsigned int *segs_pix_gpu;
    unsigned int *segs_pix_gpu_offset;
    cudaMalloc((void **)&segs_pix_gpu, 2*threadSize*sizeof(unsigned int));
    segs_pix_gpu_offset = segs_pix_gpu + threadSize;
    cudaMemset(segs_pix_gpu, nInnerSegments, threadSize*sizeof(unsigned int)); // so if not set, it will pass in the kernel
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

    cudaMemcpy(connectedPixelSize_dev, connectedPixelSize_host, nInnerSegments*sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(connectedPixelIndex_dev, connectedPixelIndex_host, nInnerSegments*sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(segs_pix_gpu,segs_pix,threadSize*sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(segs_pix_gpu_offset,segs_pix_offset,threadSize*sizeof(unsigned int), cudaMemcpyHostToDevice);

    //less cheap method to estimate max_size for y axis
    max_size = *std::max_element(nTriplets, nTriplets + nLowerModules);
    dim3 nThreads(16,16,1);
    dim3 nBlocks((totalSegs % nThreads.x == 0 ? totalSegs / nThreads.x : totalSegs / nThreads.x + 1),
                  (max_size % nThreads.y == 0 ? max_size/nThreads.y : max_size/nThreads.y + 1),1);
    createPixelTripletsInGPUFromMap<<<nBlocks, nThreads>>>(*modulesInGPU, *hitsInGPU, *mdsInGPU, *segmentsInGPU, *tripletsInGPU, *pixelTripletsInGPU, connectedPixelSize_dev,connectedPixelIndex_dev,nInnerSegments,segs_pix_gpu,segs_pix_gpu_offset, totalSegs);

    cudaError_t cudaerr = cudaGetLastError();
    if(cudaerr != cudaSuccess)
    {
	    std::cout<<"sync failed with error : "<<cudaGetErrorString(cudaerr)<<std::endl;

    }cudaDeviceSynchronize();
    cudaFreeHost(connectedPixelSize_host);
    cudaFreeHost(connectedPixelIndex_host);
    cudaFree(connectedPixelSize_dev);
    cudaFree(connectedPixelIndex_dev);
    cudaFreeHost(superbins);
    cudaFreeHost(pixelTypes);
    cudaFreeHost(nTriplets);
    free(segs_pix);
    cudaFree(segs_pix_gpu);
    unsigned int nPixelTriplets;
    cudaMemcpy(&nPixelTriplets, &(pixelTripletsInGPU->nPixelTriplets),  sizeof(unsigned int), cudaMemcpyDeviceToHost);
#ifdef Warnings
    std::cout<<"number of pixel triplets = "<<nPixelTriplets<<std::endl;
#endif

    //pT3s can be cleaned here because they're not used in making pT5s!
#ifdef DUP_pT3
    printf("run dup pT3\n");
    dim3 nThreads_dup(512,1,1);
    dim3 nBlocks_dup(64,1,1);
    removeDupPixelTripletsInGPUFromMap<<<nBlocks_dup,nThreads_dup>>>(*modulesInGPU, *hitsInGPU, *mdsInGPU, *segmentsInGPU, *pixelTripletsInGPU,*tripletsInGPU,false);
#endif

}


void SDL::Event::createQuintuplets()
{
    unsigned int nLowerModules;
    cudaMemcpy(&nLowerModules,modulesInGPU->nLowerModules,sizeof(unsigned int),cudaMemcpyDeviceToHost);

    unsigned int nEligibleT5Modules = 0;
    unsigned int *indicesOfEligibleModules = (unsigned int*)malloc(nLowerModules*sizeof(unsigned int));

    unsigned int maxTriplets;
    createEligibleModulesListForQuintuplets(*modulesInGPU, *tripletsInGPU, nEligibleT5Modules, indicesOfEligibleModules, N_MAX_QUINTUPLETS_PER_MODULE, maxTriplets);

    if(quintupletsInGPU == nullptr)
    {
        cudaMallocHost(&quintupletsInGPU, sizeof(SDL::quintuplets));
#ifdef Explicit_T5
        createQuintupletsInExplicitMemory(*quintupletsInGPU, N_MAX_QUINTUPLETS_PER_MODULE, nLowerModules, nEligibleT5Modules);
#else
        createQuintupletsInUnifiedMemory(*quintupletsInGPU, N_MAX_QUINTUPLETS_PER_MODULE, nLowerModules, nEligibleT5Modules);
#endif
    }


    int threadSize=N_MAX_TOTAL_TRIPLETS;
    unsigned int *threadIdx = (unsigned int*)malloc(2*threadSize*sizeof(unsigned int));
    unsigned int *threadIdx_offset = threadIdx+threadSize;
    unsigned int *threadIdx_gpu;
    unsigned int *threadIdx_gpu_offset;
    cudaMalloc((void **)&threadIdx_gpu, 2*threadSize*sizeof(unsigned int));
    threadIdx_gpu_offset = threadIdx_gpu + threadSize;
    cudaMemset(threadIdx_gpu, nLowerModules, threadSize*sizeof(unsigned int));

    unsigned int *nTriplets = (unsigned int*)malloc(nLowerModules*sizeof(unsigned int));
    cudaMemcpy(nTriplets, tripletsInGPU->nTriplets, nLowerModules*sizeof(unsigned int), cudaMemcpyDeviceToHost);

    int nTotalTriplets = 0;
    for (int i=0; i<nEligibleT5Modules; i++) {
      int index = indicesOfEligibleModules[i];
      unsigned int nInnerTriplets = nTriplets[index];
      if (nInnerTriplets > N_MAX_TRIPLETS_PER_MODULE) nInnerTriplets = N_MAX_TRIPLETS_PER_MODULE;
      if (nInnerTriplets !=0) {
        for (int j=0; j<nInnerTriplets; j++) {
          threadIdx[nTotalTriplets + j] = index;
          threadIdx_offset[nTotalTriplets + j] = j;
        }
        nTotalTriplets += nInnerTriplets;
      }
    }
    printf("T5: nTotalTriplets=%d nEligibleT5Modules=%d\n", nTotalTriplets, nEligibleT5Modules);
    if (threadSize < nTotalTriplets) {
      printf("threadSize=%d nTotalTriplets=%d: Increase buffer size for threadIdx in createQuintuplets\n", threadSize, nTotalTriplets);
      exit(1);
    }
    cudaMemcpy(threadIdx_gpu, threadIdx, threadSize*sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(threadIdx_gpu_offset, threadIdx_offset, threadSize*sizeof(unsigned int), cudaMemcpyHostToDevice);

    dim3 nThreads(16, 16, 1);
    int max_outerTriplets = N_MAX_TRIPLETS_PER_MODULE;

    int mp;
    cudaDeviceGetAttribute(&mp, cudaDevAttrMultiProcessorCount, 0);
    int m = (nTotalTriplets + nThreads.y*mp - 1)/(nThreads.y*mp);
    int mPerThread=16;
    m = (m + mPerThread -1)/mPerThread;
    int nblocksY = mp*m;
    printf("cuda multiprocessor #:%d mPerThreads=%d nBlocksY=%d\n", mp, mPerThread, nblocksY);
    dim3 nBlocks((max_outerTriplets % nThreads.x == 0 ? max_outerTriplets/nThreads.x : max_outerTriplets/nThreads.x + 1), nblocksY, 1);

    //    dim3 nBlocks((max_outerTriplets % nThreads.x == 0 ? max_outerTriplets/nThreads.x : max_outerTriplets/nThreads.x + 1), (nTotalTriplets % nThreads.y == 0 ? nTotalTriplets/nThreads.y : nTotalTriplets/nThreads.y + 1), 1);
    createQuintupletsInGPU<<<nBlocks,nThreads>>>(*modulesInGPU, *hitsInGPU, *mdsInGPU, *segmentsInGPU, *tripletsInGPU, *quintupletsInGPU, threadIdx_gpu, threadIdx_gpu_offset, nTotalTriplets);
    cudaError_t cudaerr = cudaGetLastError();
    if(cudaerr != cudaSuccess)
      {
	std::cout<<"sync failed with error : "<<cudaGetErrorString(cudaerr)<<std::endl;
      }cudaDeviceSynchronize();
    free(threadIdx);
    free(nTriplets);
    cudaFree(threadIdx_gpu);
    free(indicesOfEligibleModules);

#ifdef DUP_T5
  printf("run dup T5\n");
    dim3 dupThreads(32,32,1);
    dim3 dupBlocks(16,16,1);
    removeDupQuintupletsInGPU<<<dupBlocks,dupThreads>>>(*modulesInGPU, *hitsInGPU, *mdsInGPU, *segmentsInGPU, *tripletsInGPU, *quintupletsInGPU,false);
    cudaDeviceSynchronize();
    //removeDupQuintupletsInGPU<<<dupBlocks,dupThreads>>>(*modulesInGPU, *hitsInGPU, *mdsInGPU, *segmentsInGPU, *tripletsInGPU, *quintupletsInGPU,true);
    //cudaDeviceSynchronize();
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
    printf("cleaning pixels\n");
    checkHitspLS<<<64,1024>>>(*modulesInGPU,*mdsInGPU, *segmentsInGPU, *hitsInGPU, false);
    cudaError_t cudaerrpix = cudaGetLastError();
    if(cudaerrpix != cudaSuccess)
    {
        std::cout<<"sync failed with error : "<<cudaGetErrorString(cudaerrpix)<<std::endl;

    }cudaDeviceSynchronize();
#endif  

}
void SDL::Event::createPixelQuintuplets()
{
    unsigned int nLowerModules;
    cudaMemcpy(&nLowerModules, modulesInGPU->nLowerModules, sizeof(unsigned int), cudaMemcpyDeviceToHost);

    if(pixelQuintupletsInGPU == nullptr)
    {
        cudaMallocHost(&pixelQuintupletsInGPU, sizeof(SDL::pixelQuintuplets));
    }
#ifdef Explicit_PT5
    createPixelQuintupletsInExplicitMemory(*pixelQuintupletsInGPU, N_MAX_PIXEL_QUINTUPLETS);
#else
    createPixelQuintupletsInUnifiedMemory(*pixelQuintupletsInGPU, N_MAX_PIXEL_QUINTUPLETS);
#endif  
    

    unsigned int pixelModuleIndex;
    int* superbins;
    int* pixelTypes;
    unsigned int *nQuintuplets;
    unsigned int nModules;
    cudaMemcpy(&nModules,modulesInGPU->nModules,sizeof(unsigned int),cudaMemcpyDeviceToHost);
    pixelModuleIndex = nModules-1;
    unsigned int nInnerSegments = 0;
    cudaMemcpy(&nInnerSegments, &(segmentsInGPU->nSegments[pixelModuleIndex]), sizeof(unsigned int), cudaMemcpyDeviceToHost);
    nInnerSegments = std::min(nInnerSegments, N_MAX_PIXEL_SEGMENTS_PER_MODULE);

    cudaMallocHost(&nQuintuplets, nLowerModules * sizeof(unsigned int));
    cudaMemcpy(nQuintuplets, quintupletsInGPU->nQuintuplets, nLowerModules * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    cudaMallocHost(&superbins,N_MAX_PIXEL_SEGMENTS_PER_MODULE*sizeof(int));
    cudaMallocHost(&pixelTypes,N_MAX_PIXEL_SEGMENTS_PER_MODULE*sizeof(int));

    cudaMemcpy(superbins,segmentsInGPU->superbin,N_MAX_PIXEL_SEGMENTS_PER_MODULE*sizeof(int),cudaMemcpyDeviceToHost);
    cudaMemcpy(pixelTypes,segmentsInGPU->pixelType,N_MAX_PIXEL_SEGMENTS_PER_MODULE*sizeof(int),cudaMemcpyDeviceToHost);

    unsigned int* connectedPixelSize_host;
    unsigned int* connectedPixelIndex_host;
    cudaMallocHost(&connectedPixelSize_host, nInnerSegments* sizeof(unsigned int));
    cudaMallocHost(&connectedPixelIndex_host, nInnerSegments* sizeof(unsigned int));
    unsigned int* connectedPixelSize_dev;
    unsigned int* connectedPixelIndex_dev;
    cudaMalloc(&connectedPixelSize_dev, nInnerSegments* sizeof(unsigned int));
    cudaMalloc(&connectedPixelIndex_dev, nInnerSegments* sizeof(unsigned int));
    unsigned int max_size =0;
    int threadSize = 1000000;
    unsigned int *segs_pix = (unsigned int*)malloc(2*threadSize*sizeof(unsigned int));
    unsigned int *segs_pix_offset = segs_pix+threadSize;
    unsigned int *segs_pix_gpu;
    unsigned int *segs_pix_gpu_offset;
    cudaMalloc((void **)&segs_pix_gpu, 2*threadSize*sizeof(unsigned int));
    segs_pix_gpu_offset = segs_pix_gpu + threadSize;
    cudaMemset(segs_pix_gpu, nInnerSegments, threadSize*sizeof(unsigned int)); // so if not set, it will pass in the kernel
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

    cudaMemcpy(connectedPixelSize_dev, connectedPixelSize_host, nInnerSegments*sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(connectedPixelIndex_dev, connectedPixelIndex_host, nInnerSegments*sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(segs_pix_gpu,segs_pix,threadSize*sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(segs_pix_gpu_offset,segs_pix_offset,threadSize*sizeof(unsigned int), cudaMemcpyHostToDevice);

    //less cheap method to estimate max_size for y axis
    max_size = *std::max_element(nQuintuplets, nQuintuplets + nLowerModules);
    dim3 nThreads(16,16,1);
    dim3 nBlocks((totalSegs % nThreads.x == 0 ? totalSegs / nThreads.x : totalSegs / nThreads.x + 1),
                  (max_size % nThreads.y == 0 ? max_size/nThreads.y : max_size/nThreads.y + 1),1);
    createPixelQuintupletsInGPUFromMap<<<nBlocks, nThreads>>>(*modulesInGPU, *hitsInGPU, *mdsInGPU, *segmentsInGPU, *tripletsInGPU, *quintupletsInGPU, *pixelQuintupletsInGPU, connectedPixelSize_dev, connectedPixelIndex_dev, nInnerSegments, segs_pix_gpu, segs_pix_gpu_offset, totalSegs);

    cudaError_t cudaerr = cudaGetLastError();
    if(cudaerr != cudaSuccess)
    {
	    std::cout<<"sync failed with error : "<<cudaGetErrorString(cudaerr)<<std::endl;

    }cudaDeviceSynchronize();
    cudaFreeHost(connectedPixelSize_host);
    cudaFreeHost(connectedPixelIndex_host);
    cudaFree(connectedPixelSize_dev);
    cudaFree(connectedPixelIndex_dev);
    cudaFreeHost(superbins);
    cudaFreeHost(pixelTypes);
    cudaFreeHost(nQuintuplets);
    free(segs_pix);
    cudaFree(segs_pix_gpu);

    unsigned int nPixelQuintuplets;
    cudaMemcpy(&nPixelQuintuplets, &(pixelQuintupletsInGPU->nPixelQuintuplets), sizeof(unsigned int), cudaMemcpyDeviceToHost);
    dim3 nThreads_dup(512,1,1);
    dim3 nBlocks_dup(128,1,1);
#ifdef DUP_pT5
    printf("run dup pT5\n");
    removeDupPixelQuintupletsInGPUFromMap<<<nBlocks_dup,nThreads_dup>>>(*modulesInGPU, *hitsInGPU, *mdsInGPU, *segmentsInGPU, *pixelTripletsInGPU,*tripletsInGPU, *pixelQuintupletsInGPU, *quintupletsInGPU,false);
    cudaError_t cudaerr2 = cudaGetLastError(); 
    if(cudaerr2 != cudaSuccess)
    {
        std::cout<<"sync failed with error : "<<cudaGetErrorString(cudaerr2)<<std::endl;
    }cudaDeviceSynchronize();
    //removeDupPixelQuintupletsInGPUFromMap<<<nBlocks_dup,nThreads_dup>>>(*modulesInGPU, *hitsInGPU, *mdsInGPU, *segmentsInGPU, *pixelTripletsInGPU,*tripletsInGPU, *pixelQuintupletsInGPU, *quintupletsInGPU,true);
    //cudaError_t cudaerr3 = cudaDeviceSynchronize();
#endif
    markUsedObjects<<<nBlocks_dup,nThreads_dup>>>(*modulesInGPU, *segmentsInGPU, *tripletsInGPU, *pixelQuintupletsInGPU, *quintupletsInGPU);
#ifdef Warnings
    std::cout<<"number of pixel quintuplets = "<<nPixelQuintuplets<<std::endl;
#endif   
}

void SDL::Event::addTrackCandidatesToEventExplicit()
{
    unsigned int nLowerModules;
    cudaMemcpy(&nLowerModules,modulesInGPU->nLowerModules,sizeof(unsigned int),cudaMemcpyDeviceToHost);

    unsigned int* nTrackCandidatesCPU;
    cudaMallocHost(&nTrackCandidatesCPU, (nLowerModules )* sizeof(unsigned int));
    cudaMemcpy(nTrackCandidatesCPU,trackCandidatesInGPU->nTrackCandidates,(nLowerModules)*sizeof(unsigned int),cudaMemcpyDeviceToHost);

    unsigned int* module_lowerModuleIndices;
    cudaMallocHost(&module_lowerModuleIndices, (nLowerModules +1)* sizeof(unsigned int));
    cudaMemcpy(module_lowerModuleIndices,modulesInGPU->lowerModuleIndices,(nLowerModules+1)*sizeof(unsigned int),cudaMemcpyDeviceToHost);
    int* module_trackCandidateRanges;
    cudaMallocHost(&module_trackCandidateRanges, nModules* 2*sizeof(int));
    cudaMemcpy(module_trackCandidateRanges,modulesInGPU->trackCandidateRanges,nModules*2*sizeof(int),cudaMemcpyDeviceToHost);
    short* module_layers;
    cudaMallocHost(&module_layers, nModules * sizeof(short));
    cudaMemcpy(module_layers,modulesInGPU->layers,nModules*sizeof(short),cudaMemcpyDeviceToHost);
    short* module_subdets;
    cudaMallocHost(&module_subdets, nModules* sizeof(short));
    cudaMemcpy(module_subdets,modulesInGPU->subdets,nModules*sizeof(short),cudaMemcpyDeviceToHost);

    int* module_trackCandidateModuleIndices;
    cudaMallocHost(&module_trackCandidateModuleIndices, (nLowerModules + 1) * sizeof(int));
    cudaMemcpy(module_trackCandidateModuleIndices, modulesInGPU->trackCandidateModuleIndices, sizeof(int) * (nLowerModules + 1), cudaMemcpyDeviceToHost);

    unsigned int idx;
    for(unsigned int i = 0; i<nLowerModules; i++)
    {
        idx = module_lowerModuleIndices[i];


        if(nTrackCandidatesCPU[i] == 0)
        {
            module_trackCandidateRanges[idx * 2] = -1;
            module_trackCandidateRanges[idx * 2 + 1] = -1;
        }
        else
        {
            module_trackCandidateRanges[idx * 2] = module_trackCandidateModuleIndices[i];
            module_trackCandidateRanges[idx * 2 + 1] = module_trackCandidateModuleIndices[i] + nTrackCandidatesCPU[i] - 1;

            if(module_subdets[idx] == Barrel)
            {
                n_trackCandidates_by_layer_barrel_[module_layers[idx] - 1] += nTrackCandidatesCPU[i];
            }
            else
            {
                n_trackCandidates_by_layer_endcap_[module_layers[idx] - 1] += nTrackCandidatesCPU[i];
            }
        }
    }
    cudaFreeHost(nTrackCandidatesCPU);
    cudaFreeHost(module_lowerModuleIndices);
    cudaFreeHost(module_trackCandidateRanges);
    cudaFreeHost(module_layers);
    cudaFreeHost(module_subdets);
    cudaFreeHost(module_trackCandidateModuleIndices);
}
void SDL::Event::addTrackCandidatesToEvent()
{

    unsigned int idx;
    for(unsigned int i = 0; i<*(SDL::modulesInGPU->nLowerModules); i++)
    {
        idx = SDL::modulesInGPU->lowerModuleIndices[i];


        if(trackCandidatesInGPU->nTrackCandidates[i] == 0 or SDL::modulesInGPU->trackCandidateModuleIndices[i] == -1)
        {
            modulesInGPU->trackCandidateRanges[idx * 2] = -1;
            modulesInGPU->trackCandidateRanges[idx * 2 + 1] = -1;
        }
        else
        {
            modulesInGPU->trackCandidateRanges[idx * 2] = SDL::modulesInGPU->trackCandidateModuleIndices[i];
            modulesInGPU->trackCandidateRanges[idx * 2 + 1] = SDL::modulesInGPU->trackCandidateModuleIndices[i] +  trackCandidatesInGPU->nTrackCandidates[i] - 1;

            if(modulesInGPU->subdets[idx] == Barrel)
            {
                n_trackCandidates_by_layer_barrel_[modulesInGPU->layers[idx] - 1] += trackCandidatesInGPU->nTrackCandidates[i];
            }
            else
            {
                n_trackCandidates_by_layer_endcap_[modulesInGPU->layers[idx] - 1] += trackCandidatesInGPU->nTrackCandidates[i];
            }
        }
    }
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
            modulesInGPU->quintupletRanges[idx * 2] = -1;
            modulesInGPU->quintupletRanges[idx * 2 + 1] = -1;
        }
        else
        {
            modulesInGPU->quintupletRanges[idx * 2] = SDL::modulesInGPU->quintupletModuleIndices[i];
            modulesInGPU->quintupletRanges[idx * 2 + 1] = SDL::modulesInGPU->quintupletModuleIndices[i] + quintupletsInGPU->nQuintuplets[i] - 1;

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
    cudaMemcpy(&nLowerModules,modulesInGPU->nLowerModules,sizeof(unsigned int),cudaMemcpyDeviceToHost);

    unsigned int* nQuintupletsCPU;
    cudaMallocHost(&nQuintupletsCPU, nLowerModules * sizeof(unsigned int));
    cudaMemcpy(nQuintupletsCPU,quintupletsInGPU->nQuintuplets,nLowerModules*sizeof(unsigned int),cudaMemcpyDeviceToHost);

    short* module_subdets;
    cudaMallocHost(&module_subdets, nModules* sizeof(short));
    cudaMemcpy(module_subdets,modulesInGPU->subdets,nModules*sizeof(short),cudaMemcpyDeviceToHost);

    unsigned int* module_lowerModuleIndices;
    cudaMallocHost(&module_lowerModuleIndices, (nLowerModules +1)* sizeof(unsigned int));
    cudaMemcpy(module_lowerModuleIndices,modulesInGPU->lowerModuleIndices,(nLowerModules+1)*sizeof(unsigned int),cudaMemcpyDeviceToHost);

    int* module_quintupletRanges;
    cudaMallocHost(&module_quintupletRanges, nModules* 2*sizeof(int));
    cudaMemcpy(module_quintupletRanges,modulesInGPU->quintupletRanges,nModules*2*sizeof(int),cudaMemcpyDeviceToHost);
    short* module_layers;
    cudaMallocHost(&module_layers, nModules * sizeof(short));
    cudaMemcpy(module_layers,modulesInGPU->layers,nModules*sizeof(short),cudaMemcpyDeviceToHost);
    int* module_quintupletModuleIndices;
    cudaMallocHost(&module_quintupletModuleIndices, nLowerModules * sizeof(int));
    cudaMemcpy(module_quintupletModuleIndices, modulesInGPU->quintupletModuleIndices, nLowerModules * sizeof(int), cudaMemcpyDeviceToHost);
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
            modulesInGPU->tripletRanges[idx * 2] = -1;
            modulesInGPU->tripletRanges[idx * 2 + 1] = -1;
        }
        else
        {
            modulesInGPU->tripletRanges[idx * 2] = idx * N_MAX_TRIPLETS_PER_MODULE;
            modulesInGPU->tripletRanges[idx * 2 + 1] = idx * N_MAX_TRIPLETS_PER_MODULE + tripletsInGPU->nTriplets[i] - 1;

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
    cudaMemcpy(&nLowerModules,modulesInGPU->nLowerModules,sizeof(unsigned int),cudaMemcpyDeviceToHost);

    unsigned int* nTripletsCPU;
    cudaMallocHost(&nTripletsCPU, nLowerModules * sizeof(unsigned int));
    cudaMemcpy(nTripletsCPU,tripletsInGPU->nTriplets,nLowerModules*sizeof(unsigned int),cudaMemcpyDeviceToHost);

    short* module_subdets;
    cudaMallocHost(&module_subdets, nModules* sizeof(short));
    cudaMemcpy(module_subdets,modulesInGPU->subdets,nModules*sizeof(short),cudaMemcpyDeviceToHost);
    unsigned int* module_lowerModuleIndices;
    cudaMallocHost(&module_lowerModuleIndices, (nLowerModules +1)* sizeof(unsigned int));
    cudaMemcpy(module_lowerModuleIndices,modulesInGPU->lowerModuleIndices,(nLowerModules+1)*sizeof(unsigned int),cudaMemcpyDeviceToHost);
    int* module_tripletRanges;
    cudaMallocHost(&module_tripletRanges, nModules* 2*sizeof(int));
    cudaMemcpy(module_tripletRanges,modulesInGPU->tripletRanges,nModules*2*sizeof(int),cudaMemcpyDeviceToHost);
    short* module_layers;
    cudaMallocHost(&module_layers, nModules * sizeof(short));
    cudaMemcpy(module_layers,modulesInGPU->layers,nModules*sizeof(short),cudaMemcpyDeviceToHost);
    unsigned int idx;
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
    cudaMemcpy(&nPixelTriplets, pixelTripletsInGPU->nPixelTriplets, sizeof(unsigned int), cudaMemcpyDeviceToHost);
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
    cudaMemcpy(nTrackExtensionsCPU, trackExtensionsInGPU->nTrackExtensions, nTrackCandidates * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    unsigned int nTrackExtensions = 0;
    for(size_t it = 0; it < nTrackCandidates; it++)    
    {
        nTrackExtensions += nTrackExtensionsCPU[it];

    }
    delete[] nTrackExtensionsCPU;
    return nTrackExtensions;
}

unsigned int SDL::Event::getNumberOfPixelQuintuplets()
{
#ifdef Explicit_PT5
    unsigned int nPixelQuintuplets;
    cudaMemcpy(&nPixelQuintuplets, pixelQuintupletsInGPU->nPixelQuintuplets, sizeof(unsigned int), cudaMemcpyDeviceToHost);
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

    cudaMemcpy(&nTrackCandidates, trackCandidatesInGPU->nTrackCandidates, sizeof(unsigned int), cudaMemcpyDeviceToHost);

    return nTrackCandidates;

}

unsigned int SDL::Event::getNumberOfPixelTrackCandidates()
{
    unsigned int nTrackCandidates;
    unsigned int nTrackCandidatesT5;
    return nTrackCandidates - nTrackCandidatesT5;

}

#ifdef Explicit_Hit
SDL::hits* SDL::Event::getHits() //std::shared_ptr should take care of garbage collection
{
    if(hitsInCPU == nullptr)
    {
        hitsInCPU = new SDL::hits;
        hitsInCPU->nHits = new unsigned int;
        unsigned int nHits;
        cudaMemcpy(&nHits, hitsInGPU->nHits, sizeof(unsigned int), cudaMemcpyDeviceToHost);
        *(hitsInCPU->nHits) = nHits;
        hitsInCPU->idxs = new unsigned int[nHits];
        hitsInCPU->xs = new float[nHits];
        hitsInCPU->ys = new float[nHits];
        hitsInCPU->zs = new float[nHits];
        hitsInCPU->moduleIndices = new unsigned int[nHits];
        cudaMemcpy(hitsInCPU->idxs, hitsInGPU->idxs,sizeof(unsigned int) * nHits, cudaMemcpyDeviceToHost);
        cudaMemcpy(hitsInCPU->xs, hitsInGPU->xs, sizeof(float) * nHits, cudaMemcpyDeviceToHost);
        cudaMemcpy(hitsInCPU->ys, hitsInGPU->ys, sizeof(float) * nHits, cudaMemcpyDeviceToHost);
        cudaMemcpy(hitsInCPU->zs, hitsInGPU->zs, sizeof(float) * nHits, cudaMemcpyDeviceToHost);
        cudaMemcpy(hitsInCPU->moduleIndices, hitsInGPU->moduleIndices, sizeof(unsigned int) * nHits, cudaMemcpyDeviceToHost);
    }
    return hitsInCPU;
}
#else
SDL::hits* SDL::Event::getHits() //std::shared_ptr should take care of garbage collection
{
    return hitsInGPU;
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
        cudaMemcpy(mdsInCPU->hitIndices, mdsInGPU->hitIndices, 2 * nMemoryLocations * sizeof(unsigned int), cudaMemcpyDeviceToHost);
        cudaMemcpy(mdsInCPU->nMDs, mdsInGPU->nMDs, nModules * sizeof(unsigned int), cudaMemcpyDeviceToHost);
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
        cudaMemcpy(segmentsInCPU->mdIndices, segmentsInGPU->mdIndices, 2 * nMemoryLocations * sizeof(unsigned int), cudaMemcpyDeviceToHost);
        cudaMemcpy(segmentsInCPU->nSegments, segmentsInGPU->nSegments, nModules * sizeof(unsigned int), cudaMemcpyDeviceToHost);
        cudaMemcpy(segmentsInCPU->innerMiniDoubletAnchorHitIndices, segmentsInGPU->innerMiniDoubletAnchorHitIndices, nMemoryLocations * sizeof(unsigned int), cudaMemcpyDeviceToHost);
        cudaMemcpy(segmentsInCPU->outerMiniDoubletAnchorHitIndices, segmentsInGPU->outerMiniDoubletAnchorHitIndices, nMemoryLocations * sizeof(unsigned int), cudaMemcpyDeviceToHost);
        cudaMemcpy(segmentsInCPU->ptIn, segmentsInGPU->ptIn, N_MAX_PIXEL_SEGMENTS_PER_MODULE * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(segmentsInCPU->eta, segmentsInGPU->eta, N_MAX_PIXEL_SEGMENTS_PER_MODULE * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(segmentsInCPU->phi, segmentsInGPU->phi, N_MAX_PIXEL_SEGMENTS_PER_MODULE * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(segmentsInCPU->isDup, segmentsInGPU->isDup, N_MAX_PIXEL_SEGMENTS_PER_MODULE * sizeof(bool), cudaMemcpyDeviceToHost);
        cudaMemcpy(segmentsInCPU->isQuad, segmentsInGPU->isQuad, N_MAX_PIXEL_SEGMENTS_PER_MODULE * sizeof(bool), cudaMemcpyDeviceToHost);
        cudaMemcpy(segmentsInCPU->score, segmentsInGPU->score, N_MAX_PIXEL_SEGMENTS_PER_MODULE * sizeof(float), cudaMemcpyDeviceToHost);


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
        cudaMemcpy(&nLowerModules, modulesInGPU->nLowerModules, sizeof(unsigned int), cudaMemcpyDeviceToHost);
        unsigned int nMemoryLocations = (N_MAX_TRIPLETS_PER_MODULE) * (nLowerModules);
        tripletsInCPU->segmentIndices = new unsigned[2 * nMemoryLocations];
        tripletsInCPU->nTriplets = new unsigned int[nLowerModules];
        tripletsInCPU->betaIn = new float[nMemoryLocations];
        tripletsInCPU->betaOut = new float[nMemoryLocations];
        tripletsInCPU->pt_beta = new float[nMemoryLocations];

        tripletsInCPU->logicalLayers = new unsigned int[3 * nMemoryLocations];
        tripletsInCPU->hitIndices = new unsigned int[6 * nMemoryLocations];
        tripletsInCPU->lowerModuleIndices = new unsigned int[3 * nMemoryLocations];

        cudaMemcpy(tripletsInCPU->segmentIndices, tripletsInGPU->segmentIndices, 2 * nMemoryLocations * sizeof(unsigned int), cudaMemcpyDeviceToHost);
        cudaMemcpy(tripletsInCPU->betaIn, tripletsInGPU->betaIn, nMemoryLocations * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(tripletsInCPU->betaOut, tripletsInGPU->betaOut, nMemoryLocations * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(tripletsInCPU->pt_beta, tripletsInGPU->pt_beta, nMemoryLocations * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(tripletsInCPU->nTriplets, tripletsInGPU->nTriplets, nLowerModules * sizeof(unsigned int), cudaMemcpyDeviceToHost);

        cudaMemcpy(tripletsInCPU->logicalLayers, tripletsInGPU->logicalLayers, 3 * nMemoryLocations * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(tripletsInCPU->lowerModuleIndices, tripletsInGPU->lowerModuleIndices, 3 * nMemoryLocations * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(tripletsInCPU->hitIndices, tripletsInGPU->hitIndices, 6 * nMemoryLocations * sizeof(float), cudaMemcpyDeviceToHost);

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
        cudaMemcpy(&nLowerModules, modulesInGPU->nLowerModules, sizeof(unsigned int), cudaMemcpyDeviceToHost);
        unsigned int nEligibleT5Modules;
        cudaMemcpy(&nEligibleT5Modules, modulesInGPU->nEligibleT5Modules, sizeof(unsigned int), cudaMemcpyDeviceToHost);
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
        cudaMemcpy(quintupletsInCPU->nQuintuplets, quintupletsInGPU->nQuintuplets,  nLowerModules * sizeof(unsigned int), cudaMemcpyDeviceToHost);
        cudaMemcpy(quintupletsInCPU->tripletIndices, quintupletsInGPU->tripletIndices, 2 * nMemoryLocations * sizeof(unsigned int), cudaMemcpyDeviceToHost);
        cudaMemcpy(quintupletsInCPU->lowerModuleIndices, quintupletsInGPU->lowerModuleIndices, 5 * nMemoryLocations * sizeof(unsigned int), cudaMemcpyDeviceToHost);
        cudaMemcpy(quintupletsInCPU->innerRadius, quintupletsInGPU->innerRadius, nMemoryLocations * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(quintupletsInCPU->outerRadius, quintupletsInGPU->outerRadius, nMemoryLocations * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(quintupletsInCPU->isDup, quintupletsInGPU->isDup, nMemoryLocations * sizeof(bool), cudaMemcpyDeviceToHost);
        cudaMemcpy(quintupletsInCPU->score_rphisum, quintupletsInGPU->score_rphisum, nMemoryLocations * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(quintupletsInCPU->eta, quintupletsInGPU->eta, nMemoryLocations * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(quintupletsInCPU->phi, quintupletsInGPU->phi, nMemoryLocations * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(quintupletsInCPU->regressionRadius, quintupletsInGPU->regressionRadius, nMemoryLocations * sizeof(float), cudaMemcpyDeviceToHost);
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
        cudaMemcpy(pixelTripletsInCPU->nPixelTriplets, pixelTripletsInGPU->nPixelTriplets, sizeof(unsigned int), cudaMemcpyDeviceToHost);
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

        cudaMemcpy(pixelTripletsInCPU->tripletIndices, pixelTripletsInGPU->tripletIndices, nPixelTriplets * sizeof(unsigned int), cudaMemcpyDeviceToHost);
        cudaMemcpy(pixelTripletsInCPU->pixelSegmentIndices, pixelTripletsInGPU->pixelSegmentIndices, nPixelTriplets * sizeof(unsigned int), cudaMemcpyDeviceToHost);
        cudaMemcpy(pixelTripletsInCPU->pixelRadius, pixelTripletsInGPU->pixelRadius, nPixelTriplets * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(pixelTripletsInCPU->tripletRadius, pixelTripletsInGPU->tripletRadius, nPixelTriplets * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(pixelTripletsInCPU->isDup, pixelTripletsInGPU->isDup, nPixelTriplets * sizeof(bool), cudaMemcpyDeviceToHost);
        cudaMemcpy(pixelTripletsInCPU->eta, pixelTripletsInGPU->eta, nPixelTriplets * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(pixelTripletsInCPU->phi, pixelTripletsInGPU->phi, nPixelTriplets * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(pixelTripletsInCPU->score, pixelTripletsInGPU->score, nPixelTriplets * sizeof(float), cudaMemcpyDeviceToHost);
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
        cudaMemcpy(pixelQuintupletsInCPU->nPixelQuintuplets, pixelQuintupletsInGPU->nPixelQuintuplets, sizeof(unsigned int), cudaMemcpyDeviceToHost);
        unsigned int nPixelQuintuplets = *(pixelQuintupletsInCPU->nPixelQuintuplets);

        pixelQuintupletsInCPU->pixelIndices = new unsigned int[nPixelQuintuplets];
        pixelQuintupletsInCPU->T5Indices = new unsigned int[nPixelQuintuplets];
        pixelQuintupletsInCPU->isDup = new bool[nPixelQuintuplets];
        pixelQuintupletsInCPU->score = new float[nPixelQuintuplets];

        cudaMemcpy(pixelQuintupletsInCPU->pixelIndices, pixelQuintupletsInGPU->pixelIndices, nPixelQuintuplets * sizeof(unsigned int), cudaMemcpyDeviceToHost);
        cudaMemcpy(pixelQuintupletsInCPU->T5Indices, pixelQuintupletsInGPU->T5Indices, nPixelQuintuplets * sizeof(unsigned int), cudaMemcpyDeviceToHost);
        cudaMemcpy(pixelQuintupletsInCPU->isDup, pixelQuintupletsInGPU->isDup, nPixelQuintuplets * sizeof(bool), cudaMemcpyDeviceToHost);
        cudaMemcpy(pixelQuintupletsInCPU->score, pixelQuintupletsInGPU->score, nPixelQuintuplets * sizeof(float), cudaMemcpyDeviceToHost);
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
        unsigned int nMemoryLocations = N_MAX_TRACK_CANDIDATES + N_MAX_PIXEL_TRACK_CANDIDATES;
        trackCandidatesInCPU->objectIndices = new unsigned int[2 * nMemoryLocations];
        trackCandidatesInCPU->trackCandidateType = new short[nMemoryLocations];
        trackCandidatesInCPU->nTrackCandidates = new unsigned int;

        trackCandidatesInCPU->logicalLayers = new unsigned int[7 * nMemoryLocations];
        trackCandidatesInCPU->lowerModuleIndices = new unsigned int[7 * nMemoryLocations];
        trackCandidatesInCPU->hitIndices = new unsigned int[14 * nMemoryLocations];
        trackCandidatesInCPU->partOfExtension = new bool[nMemoryLocations];

        cudaMemcpy(trackCandidatesInCPU->objectIndices, trackCandidatesInGPU->objectIndices, 2 * nMemoryLocations * sizeof(unsigned int), cudaMemcpyDeviceToHost);
        cudaMemcpy(trackCandidatesInCPU->trackCandidateType, trackCandidatesInGPU->trackCandidateType, nMemoryLocations * sizeof(short), cudaMemcpyDeviceToHost);
        cudaMemcpy(trackCandidatesInCPU->nTrackCandidates, trackCandidatesInGPU->nTrackCandidates, sizeof(unsigned int), cudaMemcpyDeviceToHost);

        cudaMemcpy(trackCandidatesInCPU->lowerModuleIndices, trackCandidatesInGPU->lowerModuleIndices, 7 * nMemoryLocations * sizeof(unsigned int), cudaMemcpyDeviceToHost);
        cudaMemcpy(trackCandidatesInCPU->logicalLayers, trackCandidatesInGPU->logicalLayers, 7 * nMemoryLocations * sizeof(unsigned int), cudaMemcpyDeviceToHost);
        cudaMemcpy(trackCandidatesInCPU->hitIndices, trackCandidatesInGPU->hitIndices, 14 * nMemoryLocations * sizeof(unsigned int), cudaMemcpyDeviceToHost);
        cudaMemcpy(trackCandidatesInCPU->partOfExtension, trackCandidatesInGPU->partOfExtension, nMemoryLocations * sizeof(bool), cudaMemcpyDeviceToHost);
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
        cudaMemcpy(&nLowerModules, modulesInGPU->nLowerModules, sizeof(unsigned int), cudaMemcpyDeviceToHost);

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

    modulesInCPUFull->hitRanges = new int[2*nModules];
    modulesInCPUFull->mdRanges = new int[2*nModules];
    modulesInCPUFull->segmentRanges = new int[2*nModules];
    modulesInCPUFull->tripletRanges = new int[2*nModules];
    modulesInCPUFull->trackCandidateRanges = new int[2*nModules];

    modulesInCPUFull->moduleType = new ModuleType[nModules];
    modulesInCPUFull->moduleLayerType = new ModuleLayerType[nModules];

    modulesInCPUFull->lowerModuleIndices = new unsigned int[nLowerModules+1];
    modulesInCPUFull->reverseLookupLowerModuleIndices = new int[nModules];
    modulesInCPUFull->trackCandidateModuleIndices = new int[nLowerModules+1];
    modulesInCPUFull->quintupletModuleIndices = new int[nLowerModules];

    cudaMemcpy(modulesInCPUFull->detIds,modulesInGPU->detIds,nModules*sizeof(unsigned int),cudaMemcpyDeviceToHost);
    cudaMemcpy(modulesInCPUFull->moduleMap,modulesInGPU->moduleMap,40*nModules*sizeof(unsigned int),cudaMemcpyDeviceToHost);
    cudaMemcpy(modulesInCPUFull->nConnectedModules,modulesInGPU->nConnectedModules,nModules*sizeof(unsigned int),cudaMemcpyDeviceToHost);
    cudaMemcpy(modulesInCPUFull->drdzs,modulesInGPU->drdzs,sizeof(float)*nModules,cudaMemcpyDeviceToHost);
    cudaMemcpy(modulesInCPUFull->slopes,modulesInGPU->slopes,sizeof(float)*nModules,cudaMemcpyDeviceToHost);
    cudaMemcpy(modulesInCPUFull->nLowerModules,modulesInGPU->nLowerModules,sizeof(unsigned int),cudaMemcpyDeviceToHost);
    cudaMemcpy(modulesInCPUFull->layers,modulesInGPU->layers,nModules*sizeof(short),cudaMemcpyDeviceToHost);
    cudaMemcpy(modulesInCPUFull->rings,modulesInGPU->rings,sizeof(short)*nModules,cudaMemcpyDeviceToHost);
    cudaMemcpy(modulesInCPUFull->modules,modulesInGPU->modules,sizeof(short)*nModules,cudaMemcpyDeviceToHost);
    cudaMemcpy(modulesInCPUFull->rods,modulesInGPU->rods,sizeof(short)*nModules,cudaMemcpyDeviceToHost);
    cudaMemcpy(modulesInCPUFull->subdets,modulesInGPU->subdets,sizeof(short)*nModules,cudaMemcpyDeviceToHost);
    cudaMemcpy(modulesInCPUFull->sides,modulesInGPU->sides,sizeof(short)*nModules,cudaMemcpyDeviceToHost);
    cudaMemcpy(modulesInCPUFull->isInverted,modulesInGPU->isInverted,sizeof(bool)*nModules,cudaMemcpyDeviceToHost);
    cudaMemcpy(modulesInCPUFull->isLower,modulesInGPU->isLower,sizeof(bool)*nModules,cudaMemcpyDeviceToHost);

    cudaMemcpy(modulesInCPUFull->hitRanges, modulesInGPU->hitRanges, 2*nModules * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(modulesInCPUFull->mdRanges, modulesInGPU->mdRanges, 2*nModules * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(modulesInCPUFull->segmentRanges, modulesInGPU->segmentRanges, 2*nModules * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(modulesInCPUFull->tripletRanges, modulesInGPU->tripletRanges, 2*nModules * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(modulesInCPUFull->trackCandidateRanges, modulesInGPU->trackCandidateRanges, 2*nModules * sizeof(int), cudaMemcpyDeviceToHost);

    cudaMemcpy(modulesInCPUFull->reverseLookupLowerModuleIndices, modulesInGPU->reverseLookupLowerModuleIndices, nModules * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(modulesInCPUFull->lowerModuleIndices, modulesInGPU->lowerModuleIndices, (nLowerModules+1) * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(modulesInCPUFull->trackCandidateModuleIndices, modulesInGPU->trackCandidateModuleIndices, (nLowerModules+1) * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(modulesInCPUFull->quintupletModuleIndices, modulesInGPU->quintupletModuleIndices, nLowerModules * sizeof(int), cudaMemcpyDeviceToHost);

    cudaMemcpy(modulesInCPUFull->moduleType,modulesInGPU->moduleType,sizeof(ModuleType)*nModules,cudaMemcpyDeviceToHost);
    cudaMemcpy(modulesInCPUFull->moduleLayerType,modulesInGPU->moduleLayerType,sizeof(ModuleLayerType)*nModules,cudaMemcpyDeviceToHost);
    }
    return modulesInCPUFull;
}
SDL::modules* SDL::Event::getModules()
{
    //if(modulesInCPU == nullptr)
    //{
        modulesInCPU = new SDL::modules;
        unsigned int nLowerModules;
        cudaMemcpy(&nLowerModules, modulesInGPU->nLowerModules, sizeof(unsigned int), cudaMemcpyDeviceToHost);
        modulesInCPU->nLowerModules = new unsigned int[1];
        modulesInCPU->nModules = new unsigned int[1];
        modulesInCPU->lowerModuleIndices = new unsigned int[nLowerModules+1];
        modulesInCPU->detIds = new unsigned int[nModules];
        modulesInCPU->hitRanges = new int[2*nModules];
        modulesInCPU->isLower = new bool[nModules];
        modulesInCPU->trackCandidateModuleIndices = new int[nLowerModules+1];
        modulesInCPU->quintupletModuleIndices = new int[nLowerModules];
        modulesInCPU->layers = new short[nModules];
        modulesInCPU->subdets = new short[nModules];
        modulesInCPU->rings = new short[nModules];


        cudaMemcpy(modulesInCPU->nLowerModules, modulesInGPU->nLowerModules, sizeof(unsigned int), cudaMemcpyDeviceToHost);
        cudaMemcpy(modulesInCPU->nModules, modulesInGPU->nModules, sizeof(unsigned int), cudaMemcpyDeviceToHost);
        cudaMemcpy(modulesInCPU->lowerModuleIndices, modulesInGPU->lowerModuleIndices, (nLowerModules+1) * sizeof(unsigned int), cudaMemcpyDeviceToHost);
        cudaMemcpy(modulesInCPU->detIds, modulesInGPU->detIds, nModules * sizeof(unsigned int), cudaMemcpyDeviceToHost);
        cudaMemcpy(modulesInCPU->hitRanges, modulesInGPU->hitRanges, 2*nModules * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(modulesInCPU->isLower, modulesInGPU->isLower, nModules * sizeof(bool), cudaMemcpyDeviceToHost);
        cudaMemcpy(modulesInCPU->trackCandidateModuleIndices, modulesInGPU->trackCandidateModuleIndices, (nLowerModules+1) * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(modulesInCPU->quintupletModuleIndices, modulesInGPU->quintupletModuleIndices, nLowerModules * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(modulesInCPU->layers, modulesInGPU->layers, nModules * sizeof(short), cudaMemcpyDeviceToHost);
        cudaMemcpy(modulesInCPU->subdets, modulesInGPU->subdets, nModules * sizeof(short), cudaMemcpyDeviceToHost);
        cudaMemcpy(modulesInCPU->rings, modulesInGPU->rings, nModules * sizeof(short), cudaMemcpyDeviceToHost);
    //}
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
       cudaMemcpy(&nTrackCandidates, trackCandidatesInCPU->nTrackCandidates, sizeof(unsigned int), cudaMemcpyDeviceToHost);
       unsigned int maxTrackExtensions = nTrackCandidates * 10;

       trackExtensionsInCPU->nTrackExtensions = new unsigned int[nTrackCandidates];
       trackExtensionsInCPU->constituentTCTypes = new short[3 * maxTrackExtensions];
       trackExtensionsInCPU->constituentTCIndices = new unsigned int[3 * maxTrackExtensions];
       trackExtensionsInCPU->nLayerOverlaps = new unsigned int[2 * maxTrackExtensions];
       trackExtensionsInCPU->nHitOverlaps = new unsigned int[2 * maxTrackExtensions];
       trackExtensionsInCPU->isDup = new bool[maxTrackExtensions];

       cudaMemcpy(trackExtensionsInCPU->nTrackExtensions, trackExtensionsInGPU->nTrackExtensions, nTrackCandidates * sizeof(unsigned int), cudaMemcpyDeviceToHost);
       cudaMemcpy(trackExtensionsInCPU->constituentTCTypes, trackExtensionsInGPU->constituentTCTypes, 3 * maxTrackExtensions * sizeof(short), cudaMemcpyDeviceToHost);
       cudaMemcpy(trackExtensionsInCPU->constituentTCIndices, trackExtensionsInGPU->constituentTCIndices, 3 * maxTrackExtensions * sizeof(unsigned int), cudaMemcpyDeviceToHost);

       cudaMemcpy(trackExtensionsInCPU->nLayerOverlaps, trackExtensionsInGPU->nLayerOverlaps, 2 * maxTrackExtensions * sizeof(unsigned int), cudaMemcpyDeviceToHost);
       cudaMemcpy(trackExtensionsInCPU->nHitOverlaps, trackExtensionsInGPU->nHitOverlaps, 2 * maxTrackExtensions * sizeof(unsigned int), cudaMemcpyDeviceToHost);
       cudaMemcpy(trackExtensionsInCPU->isDup, trackExtensionsInGPU->isDup, maxTrackExtensions * sizeof(bool), cudaMemcpyDeviceToHost);
   }

   return trackExtensionsInCPU;
}
#else
SDL::trackExtensions* SDL::Event::getTrackExtensions()
{
    return trackExtensionsInGPU;
}
#endif


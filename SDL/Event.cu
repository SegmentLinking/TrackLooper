# include "Event.cuh"
#include "allocate.h"


unsigned int N_MAX_HITS_PER_MODULE = 100;
const unsigned int N_MAX_MD_PER_MODULES = 100;
const unsigned int N_MAX_SEGMENTS_PER_MODULE = 600; //WHY!
const unsigned int MAX_CONNECTED_MODULES = 40;
const unsigned int N_MAX_TRACKLETS_PER_MODULE = 8000;//temporary
const unsigned int N_MAX_TRIPLETS_PER_MODULE = 5000;
const unsigned int N_MAX_TRACK_CANDIDATES_PER_MODULE = 50000;
const unsigned int N_MAX_PIXEL_MD_PER_MODULES = 100000;
const unsigned int N_MAX_PIXEL_SEGMENTS_PER_MODULE = 50000;
const unsigned int N_MAX_PIXEL_TRACKLETS_PER_MODULE = 3000000;
const unsigned int N_MAX_PIXEL_TRACK_CANDIDATES_PER_MODULE = 5000000;
const unsigned int N_MAX_QUINTUPLETS_PER_MODULE = 5000; 
struct SDL::modules* SDL::modulesInGPU = nullptr;
struct SDL::pixelMap* SDL::pixelMapping = nullptr;
unsigned int SDL::nModules;

SDL::Event::Event()
{
    hitsInGPU = nullptr;
    mdsInGPU = nullptr;
    segmentsInGPU = nullptr;
    trackletsInGPU = nullptr;
    tripletsInGPU = nullptr;
    quintupletsInGPU = nullptr;
    trackCandidatesInGPU = nullptr;


    hitsInCPU = nullptr;
    mdsInCPU = nullptr;
    segmentsInCPU = nullptr;
    trackletsInCPU = nullptr;
    tripletsInCPU = nullptr;
    trackCandidatesInCPU = nullptr;
    modulesInCPU = nullptr;
    modulesInCPUFull = nullptr;
    quintupletsInCPU = nullptr;

//    pixelMapping = nullptr;
    //reset the arrays
    for(int i = 0; i<6; i++)
    {
        n_hits_by_layer_barrel_[i] = 0;
        n_minidoublets_by_layer_barrel_[i] = 0;
        n_segments_by_layer_barrel_[i] = 0;
        n_tracklets_by_layer_barrel_[i] = 0;
        n_triplets_by_layer_barrel_[i] = 0;
        n_trackCandidates_by_layer_barrel_[i] = 0;
        n_quintuplets_by_layer_barrel_[i] = 0;
        if(i<5)
        {
            n_hits_by_layer_endcap_[i] = 0;
            n_minidoublets_by_layer_endcap_[i] = 0;
            n_segments_by_layer_endcap_[i] = 0;
            n_tracklets_by_layer_endcap_[i] = 0;
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
    trackletsInGPU->freeMemoryCache();
    trackCandidatesInGPU->freeMemoryCache();
#ifdef DO_QUINTUPLET
    quintupletsInGPU->freeMemoryCache();
#endif
#else
    mdsInGPU->freeMemory();
    segmentsInGPU->freeMemory();
    tripletsInGPU->freeMemory();
    trackletsInGPU->freeMemory();
    trackCandidatesInGPU->freeMemory();
#ifdef DO_QUINTUPLET
    quintupletsInGPU->freeMemory();
#endif
#endif
    cudaFreeHost(mdsInGPU);
    cudaFreeHost(segmentsInGPU);
    cudaFreeHost(tripletsInGPU);
    cudaFreeHost(trackletsInGPU);
    cudaFreeHost(trackCandidatesInGPU);
    hitsInGPU->freeMemory();
    cudaFreeHost(hitsInGPU);
#ifdef DO_QUINTUPLET
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
#ifdef Explicit_Tracklet
    if(trackletsInCPU != nullptr)
    {
        delete[] trackletsInCPU->segmentIndices;
        delete[] trackletsInCPU->nTracklets;
        delete[] trackletsInCPU->betaIn;
        delete[] trackletsInCPU->betaOut;
        delete trackletsInCPU;
    }
#endif
#ifdef Explicit_Trips
    if(tripletsInCPU != nullptr)
    {
        delete[] tripletsInCPU->segmentIndices;
        delete[] tripletsInCPU->nTriplets;
        delete[] tripletsInCPU->betaIn;
        delete[] tripletsInCPU->betaOut;
        delete tripletsInCPU;
    }
#endif
#ifdef Explicit_T5
#ifdef DO_QUINTUPLET
    if(quintupletsInCPU != nullptr)
    {
        delete[] quintupletsInCPU->tripletIndices;
        delete[] quintupletsInCPU->nQuintuplets;
        delete[] quintupletsInCPU->lowerModuleIndices;
        delete quintupletsInCPU;
    }
#endif
#endif

#ifdef Explicit_Track
    if(trackCandidatesInCPU != nullptr)
    {
        delete[] trackCandidatesInCPU->objectIndices;
        delete[] trackCandidatesInCPU->trackCandidateType;
        delete[] trackCandidatesInCPU->nTrackCandidates;
        delete trackCandidatesInCPU;
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
        delete[] modulesInCPUFull->trackletRanges;
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


//#pragma omp parallel for  // this part can be run in parallel.
  for (int ihit=0; ihit<loopsize;ihit++){
    unsigned int moduleLayer = module_layers[(*detIdToIndex)[host_detId[ihit]]];
    unsigned int subdet = module_subdet[(*detIdToIndex)[host_detId[ihit]]];
    //unsigned int moduleLayer = modulesInGPU->layers[(*detIdToIndex)[host_detId[ihit]]];
    //unsigned int subdet = modulesInGPU->subdets[(*detIdToIndex)[host_detId[ihit]]];
    host_moduleIndex[ihit] = (*detIdToIndex)[host_detId[ihit]];

//    if(subdet == Barrel) // this doesn't seem useful anymore
//    {
//        n_hits_by_layer_barrel_[moduleLayer-1]++;
//    }
//    else
//    {
//        n_hits_by_layer_endcap_[moduleLayer-1]++;
//    }


      host_rts[ihit] = sqrt(host_x[ihit]*host_x[ihit] + host_y[ihit]*host_y[ihit]);
      host_phis[ihit] = phi(host_x[ihit],host_y[ihit],host_z[ihit]);
      //host_idxs[ihit] = ihit;
//  }
//// This part i think has a race condition. so this is not run in parallel.
////#pragma omp parallel for
//  for (int ihit=0; ihit<loopsize;ihit++){
      unsigned int this_index = host_moduleIndex[ihit];
      //if(modulesInGPU->subdets[this_index] == Endcap && modulesInGPU->moduleType[this_index] == TwoS)
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
      //if(modulesInGPU->hitRanges[this_index * 2] == -1)
      //{
      //    modulesInGPU->hitRanges[this_index * 2] = ihit;
      //}
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
__global__ void addPixelSegmentToEventKernel(unsigned int* hitIndices0,unsigned int* hitIndices1,unsigned int* hitIndices2,unsigned int* hitIndices3, float* dPhiChange, float* ptIn, float* ptErr, float* px, float* py, float* pz, float* eta, float* etaErr,float* phi, unsigned int pixelModuleIndex, struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU,const int size, int* superbin, int* pixelType)
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
      addPixelSegmentToMemory(segmentsInGPU, mdsInGPU, hitsInGPU, modulesInGPU, innerMDIndex, outerMDIndex, pixelModuleIndex, hitIndices0[tid], hitIndices2[tid], dPhiChange[tid], ptIn[tid], ptErr[tid], px[tid], py[tid], pz[tid], etaErr[tid], eta[tid], phi[tid], pixelSegmentIndex, tid, superbin[tid], pixelType[tid]);
    }
}
void SDL::Event::addPixelSegmentToEvent(std::vector<unsigned int> hitIndices0,std::vector<unsigned int> hitIndices1,std::vector<unsigned int> hitIndices2,std::vector<unsigned int> hitIndices3, std::vector<float> dPhiChange, std::vector<float> ptIn, std::vector<float> ptErr, std::vector<float> px, std::vector<float> py, std::vector<float> pz, std::vector<float> eta, std::vector<float> etaErr, std::vector<float> phi, std::vector<int> superbin, std::vector<int> pixelType)
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
    //cudaMallocHost(&pixelMapping->superbin,superbin.size()*sizeof(int));
    //cudaMallocHost(&pixelMapping->pixelType,pixelType.size()*sizeof(int));
    //cudaMallocHost(&pixelMapping->superbin,N_MAX_PIXEL_SEGMENTS_PER_MODULE*sizeof(int));
    //cudaMallocHost(&pixelMapping->pixelType,N_MAX_PIXEL_SEGMENTS_PER_MODULE*sizeof(int));
    //pixelMapping->superbin = &superbin[0];
    //pixelMapping->pixelType = &pixelType[0];

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

    unsigned int nThreads = 256;
    unsigned int nBlocks =  size % nThreads == 0 ? size/nThreads : size/nThreads + 1;

  addPixelSegmentToEventKernel<<<nBlocks,nThreads>>>(hitIndices0_dev,hitIndices1_dev,hitIndices2_dev,hitIndices3_dev,dPhiChange_dev,ptIn_dev,ptErr_dev,px_dev,py_dev,pz_dev,eta_dev, etaErr_dev, phi_dev, pixelModuleIndex, *modulesInGPU,*hitsInGPU,*mdsInGPU,*segmentsInGPU,size, superbin_dev, pixelType_dev);
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

            //for(unsigned int jdx = 0; jdx < segmentsInGPU->nSegments[idx]; jdx++)
            //    printSegment(*segmentsInGPU, *mdsInGPU, *hitsInGPU, *modulesInGPU, idx * N_MAX_SEGMENTS_PER_MODULE + jdx);

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

            //for(unsigned int jdx = 0; jdx < segmentsInGPU->nSegments[idx]; jdx++)
            //    printSegment(*segmentsInGPU, *mdsInGPU, *hitsInGPU, *modulesInGPU, idx * N_MAX_SEGMENTS_PER_MODULE + jdx);

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

#ifdef NESTED_PARA
    int nThreads = 1;
    int nBlocks = nLowerModules % nThreads == 0 ? nLowerModules/nThreads : nLowerModules/nThreads + 1;
#else
#ifdef NEWGRID_MD
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
#else
    dim3 nThreads(1,16,16);
    dim3 nBlocks((nLowerModules % nThreads.x == 0 ? nLowerModules/nThreads.x : nLowerModules/nThreads.x + 1),(N_MAX_HITS_PER_MODULE % nThreads.y == 0 ? N_MAX_HITS_PER_MODULE/nThreads.y : N_MAX_HITS_PER_MODULE/nThreads.y + 1), (N_MAX_HITS_PER_MODULE % nThreads.z == 0 ? N_MAX_HITS_PER_MODULE/nThreads.z : N_MAX_HITS_PER_MODULE/nThreads.z + 1));
    //std::cout<<nBlocks.x<<" "<<nBlocks.y<<" "<<nBlocks.z<<" "<<std::endl;
#endif
#endif

    cudaDeviceSynchronize();
    auto syncStart = std::chrono::high_resolution_clock::now();

    createMiniDoubletsInGPU<<<nBlocks,nThreads>>>(*modulesInGPU,*hitsInGPU,*mdsInGPU);

    cudaError_t cudaerr = cudaDeviceSynchronize();
    auto syncStop = std::chrono::high_resolution_clock::now();

    auto syncDuration =  std::chrono::duration_cast<std::chrono::milliseconds>(syncStop - syncStart);

    if(cudaerr != cudaSuccess)
    {
        std::cout<<"sync failed with error : "<<cudaGetErrorString(cudaerr)<<std::endl;
    }

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
        //FIXME:Add memory locations for pixel segments
        //createSegmentsInExplicitMemory(*segmentsInGPU, N_MAX_SEGMENTS_PER_MODULE, nModules);
        createSegmentsInExplicitMemory(*segmentsInGPU, N_MAX_SEGMENTS_PER_MODULE, nModules, N_MAX_PIXEL_SEGMENTS_PER_MODULE);
#else
        createSegmentsInUnifiedMemory(*segmentsInGPU, N_MAX_SEGMENTS_PER_MODULE, nModules, N_MAX_PIXEL_SEGMENTS_PER_MODULE);
#endif
    }
    unsigned int nLowerModules;
    cudaMemcpy(&nLowerModules,modulesInGPU->nLowerModules,sizeof(unsigned int),cudaMemcpyDeviceToHost);

#ifdef NESTED_PARA
    unsigned int nThreads = 1;
    unsigned int nBlocks = nLowerModules % nThreads == 0 ? nLowerModules/nThreads : nLowerModules/nThreads + 1;
#else
#ifdef NEWGRID_Seg
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
    //int max_cModules=0;
    //int sq_max_nMDs = 0;
    //int nonZeroModules = 0;
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
#else
    dim3 nThreads(1,16,16);
    dim3 nBlocks(((nLowerModules * MAX_CONNECTED_MODULES)  % nThreads.x == 0 ? (nLowerModules * MAX_CONNECTED_MODULES)/nThreads.x : (nLowerModules * MAX_CONNECTED_MODULES)/nThreads.x + 1),(N_MAX_MD_PER_MODULES % nThreads.y == 0 ? N_MAX_MD_PER_MODULES/nThreads.y : N_MAX_MD_PER_MODULES/nThreads.y + 1), (N_MAX_MD_PER_MODULES % nThreads.z == 0  ? N_MAX_MD_PER_MODULES/nThreads.z : N_MAX_MD_PER_MODULES/nThreads.z + 1));
#endif
#endif

    createSegmentsInGPU<<<nBlocks,nThreads>>>(*modulesInGPU, *hitsInGPU, *mdsInGPU, *segmentsInGPU);

    cudaError_t cudaerr = cudaDeviceSynchronize();
    if(cudaerr != cudaSuccess)
    {
        std::cout<<"sync failed with error : "<<cudaGetErrorString(cudaerr)<<std::endl;
    }
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

#ifdef NESTED_PARA
    unsigned int nThreads = 1;
    unsigned int nBlocks = nLowerModules % nThreads == 0 ? nLowerModules/nThreads : nLowerModules/nThreads + 1;

    createTripletsInGPU<<<nBlocks,nThreads>>>(*modulesInGPU, *hitsInGPU, *mdsInGPU, *segmentsInGPU, *tripletsInGPU);
    cudaError_t cudaerr = cudaDeviceSynchronize();
    if(cudaerr != cudaSuccess)
      {
	std::cout<<"sync failed with error : "<<cudaGetErrorString(cudaerr)<<std::endl;
      }
#else
#ifdef NEWGRID_Trips
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
    /*
    for (int i=0; i<nModules; i++) {
      int nSeg = nSegments[i];
      max_OuterSeg = max_OuterSeg > nSeg ? max_OuterSeg : nSeg;
    }
    */
    max_OuterSeg = N_MAX_SEGMENTS_PER_MODULE;
    //printf("nonZeroModules=%d max_InnerSeg=%d max_OuterSeg=%d\n", nonZeroModules, max_InnerSeg, max_OuterSeg);
    dim3 nThreads(16,16,1);
    dim3 nBlocks((max_OuterSeg % nThreads.x == 0 ? max_OuterSeg / nThreads.x : max_OuterSeg / nThreads.x + 1),(max_InnerSeg % nThreads.y == 0 ? max_InnerSeg/nThreads.y : max_InnerSeg/nThreads.y + 1), (nonZeroModules % nThreads.z == 0 ? nonZeroModules/nThreads.z : nonZeroModules/nThreads.z + 1));
    createTripletsInGPU<<<nBlocks,nThreads>>>(*modulesInGPU, *hitsInGPU, *mdsInGPU, *segmentsInGPU, *tripletsInGPU, index_gpu);
    cudaError_t cudaerr = cudaDeviceSynchronize();
    if(cudaerr != cudaSuccess)
      {
	std::cout<<"sync failed with error : "<<cudaGetErrorString(cudaerr)<<std::endl;
      }
    free(nSegments);
    free(index);
    cudaFree(index_gpu);
#else
    printf("original 3D grid launching in createTriplets does not exist");
    exit(1);
#endif
#endif

#if defined(AddObjects)
#ifdef Explicit_Trips
    addTripletsToEventExplicit();
#else
    addTripletsToEvent();
#endif
#endif
}

void SDL::Event::createTrackletsWithModuleMap()
{
    unsigned int nLowerModules;// = *modulesInGPU->nLowerModules;
    cudaMemcpy(&nLowerModules,modulesInGPU->nLowerModules,sizeof(unsigned int),cudaMemcpyDeviceToHost);

    //TRCAKLETS - To conserve memory, we shall be only declaring nLowerModules amount of memory!!!!!!!
    if(trackletsInGPU == nullptr)
    {
        cudaMallocHost(&trackletsInGPU, sizeof(SDL::tracklets));
#ifdef Explicit_Tracklet
        //FIXME:Add memory locations for pixel tracklets
        createTrackletsInExplicitMemory(*trackletsInGPU, N_MAX_TRACKLETS_PER_MODULE , N_MAX_PIXEL_TRACKLETS_PER_MODULE, nLowerModules);
#else
        createTrackletsInUnifiedMemory(*trackletsInGPU, N_MAX_TRACKLETS_PER_MODULE , N_MAX_PIXEL_TRACKLETS_PER_MODULE, nLowerModules);
#endif
    }

#ifdef NESTED_PARA
    unsigned int nThreads = 1;
    unsigned int nBlocks = nLowerModules % nThreads == 0 ? nLowerModules/nThreads : nLowerModules/nThreads + 1;

    #ifdef T4FromT3
      createTrackletsFromTriplets<<<nBlocks,nThreads>>>(*modulesInGPU, *hitsInGPU, *mdsInGPU, *segmentsInGPU, *tripletsInGPU, *trackletsInGPU);
    #else
      createTrackletsInGPU<<<nBlocks,nThreads>>>(*modulesInGPU, *hitsInGPU, *mdsInGPU, *segmentsInGPU, *trackletsInGPU);
    #endif

    cudaError_t cudaerr = cudaDeviceSynchronize();
    if(cudaerr != cudaSuccess)
      {
	std::cout<<"sync failed with error : "<<cudaGetErrorString(cudaerr)<<std::endl;
      }
#else
#ifdef NEWGRID_Tracklet
  #ifdef T4FromT3
    int threadSize=230000;
    unsigned int *nTriplets = (unsigned int*)malloc((nLowerModules-1)*sizeof(unsigned int));
    unsigned int *threadIdx = (unsigned int*)malloc(2*threadSize*sizeof(unsigned int));
    unsigned int *threadIdx_offset = threadIdx+threadSize;
    unsigned int *threadIdx_gpu;
    unsigned int *threadIdx_gpu_offset;
    cudaMalloc((void **)&threadIdx_gpu, 2*threadSize*sizeof(unsigned int));
    threadIdx_gpu_offset = threadIdx_gpu + threadSize;
    cudaMemset(threadIdx_gpu, nLowerModules, threadSize*sizeof(unsigned int));
    cudaMemcpy(nTriplets, tripletsInGPU->nTriplets, (nLowerModules-1)*sizeof(unsigned int), cudaMemcpyDeviceToHost);
    unsigned int totalCand=0;
    for (int i=0; i< nLowerModules-1; i++) {
      unsigned int nInnerTriplets = nTriplets[i];
      if(nInnerTriplets > N_MAX_TRIPLETS_PER_MODULE)
        nInnerTriplets = N_MAX_TRIPLETS_PER_MODULE;
      if (nInnerTriplets !=0) {
        for (int k=0; k<nInnerTriplets; k++) {
          threadIdx[totalCand+k] = i;
          threadIdx_offset[totalCand+k] = k;
        }
        totalCand += nInnerTriplets;
      }
    }
    //printf("totalCand=%d\n", totalCand);
    cudaMemcpy(threadIdx_gpu, threadIdx, threadSize*sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(threadIdx_gpu_offset, threadIdx_offset, threadSize*sizeof(unsigned int), cudaMemcpyHostToDevice);

    dim3 nThreads(16, 32, 1);
    dim3 nBlocks((N_MAX_TRIPLETS_PER_MODULE % nThreads.x == 0 ? N_MAX_TRIPLETS_PER_MODULE/nThreads.x : N_MAX_TRIPLETS_PER_MODULE/nThreads.x + 1), (totalCand % nThreads.y == 0 ? totalCand/nThreads.y : totalCand/nThreads.y + 1), 1);

    createTrackletsFromTriplets<<<nBlocks,nThreads>>>(*modulesInGPU, *hitsInGPU, *mdsInGPU, *segmentsInGPU, *tripletsInGPU, *trackletsInGPU,threadIdx_gpu,threadIdx_gpu_offset);
    free(threadIdx);
    cudaFree(threadIdx_gpu);
    free(nTriplets);

  #else
      int max_cModules = 0;
      int sq_max_segments = 0;
      int nonZeroSegModules = 0;
      int inner_max_segments = 0;
      int outer_max_segments = 0;
      unsigned int *index_gpu;
      unsigned int *outerLowerModuleIndices = (unsigned int*)malloc(nModules*N_MAX_SEGMENTS_PER_MODULE*sizeof(unsigned int));
      unsigned int *nSegments = (unsigned int*)malloc(nModules*sizeof(unsigned int));
      unsigned int *index = (unsigned int*)malloc(nLowerModules*sizeof(unsigned int));
      cudaMalloc((void **)&index_gpu, nLowerModules*sizeof(unsigned int));
    #ifdef Explicit_Module
      //unsigned int nModules = *modulesInGPU->nModules;
      cudaMemcpy((void *)outerLowerModuleIndices, segmentsInGPU->outerLowerModuleIndices, nModules*N_MAX_SEGMENTS_PER_MODULE*sizeof(unsigned int), cudaMemcpyDeviceToHost);
      cudaMemcpy((void *)nSegments, segmentsInGPU->nSegments, nModules*sizeof(unsigned int), cudaMemcpyDeviceToHost);
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
        unsigned int innerInnerLowerModuleIndex = module_lowerModuleIndices[i];
        unsigned int nInnerSegments = nSegments[innerInnerLowerModuleIndex];
        if (nInnerSegments!=0) {
          index[nonZeroSegModules] = i;
          nonZeroSegModules++;
        }
        inner_max_segments = inner_max_segments > nInnerSegments ? inner_max_segments : nInnerSegments;

        for (int j=0; j<nInnerSegments; j++) {
          unsigned int innerSegmentIndex = innerInnerLowerModuleIndex * N_MAX_SEGMENTS_PER_MODULE + j;
          unsigned int innerOuterLowerModuleIndex = outerLowerModuleIndices[innerSegmentIndex];
          unsigned int nOuterInnerLowerModules = module_nConnectedModules[innerOuterLowerModuleIndex];
          max_cModules = max_cModules > nOuterInnerLowerModules ? max_cModules : nOuterInnerLowerModules;
          for (int k=0; k<nOuterInnerLowerModules; k++) {
            unsigned int outerInnerLowerModuleIndex = module_moduleMap[innerOuterLowerModuleIndex * MAX_CONNECTED_MODULES + k];
            unsigned int nOuterSegments = nSegments[outerInnerLowerModuleIndex];
            sq_max_segments = sq_max_segments > nInnerSegments*nOuterSegments ? sq_max_segments : nInnerSegments*nOuterSegments;
          }
        }
      }
      cudaFreeHost(module_lowerModuleIndices);
      cudaFreeHost(module_nConnectedModules);
      cudaFreeHost(module_moduleMap);
    #else
      //unsigned int nModules = *modulesInGPU->nModules;
      cudaMemcpy((void *)outerLowerModuleIndices, segmentsInGPU->outerLowerModuleIndices, nModules*N_MAX_SEGMENTS_PER_MODULE*sizeof(unsigned int), cudaMemcpyDeviceToHost);
      cudaMemcpy((void *)nSegments, segmentsInGPU->nSegments, nModules*sizeof(unsigned int), cudaMemcpyDeviceToHost);
      for (int i=0; i<nLowerModules; i++) {
        unsigned int innerInnerLowerModuleIndex = modulesInGPU->lowerModuleIndices[i];
        unsigned int nInnerSegments = nSegments[innerInnerLowerModuleIndex] > N_MAX_SEGMENTS_PER_MODULE  ? N_MAX_SEGMENTS_PER_MODULE : nSegments[innerInnerLowerModuleIndex];
        if (nInnerSegments!=0) {
          index[nonZeroSegModules] = i;
          nonZeroSegModules++;
        }
        inner_max_segments = inner_max_segments > nInnerSegments ? inner_max_segments : nInnerSegments;

        for (int j=0; j<nInnerSegments; j++) {
          unsigned int innerSegmentIndex = innerInnerLowerModuleIndex * N_MAX_SEGMENTS_PER_MODULE + j;
          unsigned int innerOuterLowerModuleIndex = outerLowerModuleIndices[innerSegmentIndex];
          unsigned int nOuterInnerLowerModules = modulesInGPU->nConnectedModules[innerOuterLowerModuleIndex];
          max_cModules = max_cModules > nOuterInnerLowerModules ? max_cModules : nOuterInnerLowerModules;
          for (int k=0; k<nOuterInnerLowerModules; k++) {
            unsigned int outerInnerLowerModuleIndex = modulesInGPU->moduleMap[innerOuterLowerModuleIndex * MAX_CONNECTED_MODULES + k];
            unsigned int nOuterSegments = nSegments[outerInnerLowerModuleIndex] > N_MAX_SEGMENTS_PER_MODULE ? N_MAX_SEGMENTS_PER_MODULE : nSegments[outerInnerLowerModuleIndex];
            sq_max_segments = sq_max_segments > nInnerSegments*nOuterSegments ? sq_max_segments : nInnerSegments*nOuterSegments;
          }
        }
      }
    #endif
    cudaMemcpy(index_gpu, index, nonZeroSegModules*sizeof(unsigned int), cudaMemcpyHostToDevice);
    //printf("max_cModules=%d sq_max_segments=%d nonZeroSegModules=%d\n", max_cModules, sq_max_segments, nonZeroSegModules);

    dim3 nThreads(128,1,1);
    dim3 nBlocks((sq_max_segments%nThreads.x==0 ? sq_max_segments/nThreads.x : sq_max_segments/nThreads.x + 1), (max_cModules%nThreads.y==0 ? max_cModules/nThreads.y : max_cModules/nThreads.y + 1), (nonZeroSegModules%nThreads.z==0 ? nonZeroSegModules/nThreads.z : nonZeroSegModules/nThreads.z + 1));

    createTrackletsInGPU<<<nBlocks,nThreads>>>(*modulesInGPU, *hitsInGPU, *mdsInGPU, *segmentsInGPU, *trackletsInGPU, index_gpu);
    free(outerLowerModuleIndices);
    free(nSegments);
    free(index);
    cudaFree(index_gpu);
  #endif
    cudaError_t cudaerr = cudaDeviceSynchronize();
    if(cudaerr != cudaSuccess)
      {
	std::cout<<"sync failed with error : "<<cudaGetErrorString(cudaerr)<<std::endl;
      }


#else
    printf("original 3D grid launching in createTracklets does not exist");
    exit(1);
#endif
#endif
    /*addTrackletsToEvent will be called in the createTrackletsWithAGapWithModuleMap function*/

#if defined(AddObjects)
#ifdef Explicit_Tracklet
    addTrackletsToEventExplicit();
#else
    addTrackletsToEvent();
#endif
#endif

}

void SDL::Event::createPixelTrackletsWithMap()
{
    unsigned int nLowerModules;// = *modulesInGPU->nLowerModules;
    cudaMemcpy(&nLowerModules,modulesInGPU->nLowerModules,sizeof(unsigned int),cudaMemcpyDeviceToHost);
    if(trackletsInGPU == nullptr)
    {
        cudaMallocHost(&trackletsInGPU, sizeof(SDL::tracklets));
#ifdef Explicit_Tracklet
        createTrackletsInExplicitMemory(*trackletsInGPU, N_MAX_TRACKLETS_PER_MODULE , N_MAX_PIXEL_TRACKLETS_PER_MODULE, nLowerModules);
#else
        createTrackletsInUnifiedMemory(*trackletsInGPU, N_MAX_TRACKLETS_PER_MODULE , N_MAX_PIXEL_TRACKLETS_PER_MODULE, nLowerModules);
#endif
    }

#ifdef NESTED_PARA
    unsigned int nThreads = 1;
    unsigned int nBlocks = nLowerModules % nThreads == 0 ? nLowerModules/nThreads : nLowerModules/nThreads + 1;

    createPixelTrackletsInGPU<<<nBlocks,nThreads>>>(*modulesInGPU, *hitsInGPU, *mdsInGPU, *segmentsInGPU, *trackletsInGPU);

    cudaError_t cudaerr = cudaDeviceSynchronize();
    if(cudaerr != cudaSuccess)
    {
        std::cout<<"sync failed with error : "<<cudaGetErrorString(cudaerr)<<std::endl;

    }
#else
#ifdef NEWGRID_Pixel
    unsigned int pixelModuleIndex;
    unsigned int nInnerSegments;
    int* superbins;
    int* pixelTypes;
#ifdef Explicit_Module
    unsigned int nModules; 
    cudaMemcpy(&nModules,modulesInGPU->nModules,sizeof(unsigned int),cudaMemcpyDeviceToHost);
    pixelModuleIndex = nModules-1;
    unsigned int* nSegments;
    cudaMallocHost(& nSegments,nModules*sizeof(unsigned int));
    cudaMemcpy(nSegments,segmentsInGPU->nSegments,nModules*sizeof(unsigned int),cudaMemcpyDeviceToHost);
    nInnerSegments = nSegments[pixelModuleIndex] > N_MAX_PIXEL_SEGMENTS_PER_MODULE ? N_MAX_PIXEL_SEGMENTS_PER_MODULE : nSegments[pixelModuleIndex]; // number of pLS
    cudaMallocHost(& superbins,N_MAX_PIXEL_SEGMENTS_PER_MODULE*sizeof(int));
    cudaMallocHost(& pixelTypes,N_MAX_PIXEL_SEGMENTS_PER_MODULE*sizeof(int));
    cudaMemcpy(superbins,segmentsInGPU->superbin,N_MAX_PIXEL_SEGMENTS_PER_MODULE*sizeof(int),cudaMemcpyDeviceToHost);
    cudaMemcpy(pixelTypes,segmentsInGPU->pixelType,N_MAX_PIXEL_SEGMENTS_PER_MODULE*sizeof(int),cudaMemcpyDeviceToHost);

#else
    pixelModuleIndex = *modulesInGPU->nModules - 1; // pixel module index
    nInnerSegments = segmentsInGPU->nSegments[pixelModuleIndex] > N_MAX_PIXEL_SEGMENTS_PER_MODULE ? N_MAX_PIXEL_SEGMENTS_PER_MODULE : segmentsInGPU->nSegments[pixelModuleIndex]; // number of pLS
    superbins = segmentsInGPU->superbin;
    pixelTypes = segmentsInGPU->pixelType;
#endif
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
    int pixelIndexOffsetPos = pixelMapping->connectedPixelsIndex[44999];
    int pixelIndexOffsetNeg = pixelMapping->connectedPixelsIndexPos[44999] + pixelIndexOffsetPos;
    //printf("indexOffset %u %u %u\n",pixelIndexOffsetPos,pixelIndexOffsetNeg,pixelIndexOffsetNeg+pixelMapping->connectedPixelsIndexNeg[44999]);
    int i =-1;
    for (int ix=0; ix < nInnerSegments;ix++){// loop over # pLS
      int pixelType = pixelTypes[ix];// get pixel type for this pLS
      int superbin = superbins[ix]; //get superbin for this pixel
      if(superbin <0) {/*printf("bad neg %d\n",ix);*/continue;}
      if(superbin >=45000) {/*printf("bad pos %d %d %d\n",ix,superbin,pixelType);*/continue;}// skip any weird out of range values
      if(pixelType >2 || pixelType < 0){/*printf("bad pixel type %d %d\n",ix,pixelType);*/continue;}
      i++;
      if(pixelType ==0){ // used pixel type to select correct size-index arrays
        connectedPixelSize_host[i]  = pixelMapping->connectedPixelsSizes[superbin]; //number of connected modules to this pixel
        connectedPixelIndex_host[i] = pixelMapping->connectedPixelsIndex[superbin];// index to get start of connected modules for this superbin in map
//        if(connectedPixelIndex_host[i] >= pixelIndexOffsetPos){printf("TESTX0 %d %d %u %d %u\n",ix,i,connectedPixelIndex_host[i],superbin,pixelMapping->connectedPixelsIndex[superbin]);}
        for (int j=0; j < pixelMapping->connectedPixelsSizes[superbin]; j++){ // loop over modules from the size
          segs_pix[totalSegs+j] = i; // save the pixel index in array to be transfered to kernel
          segs_pix_offset[totalSegs+j] = j; // save this segment in array to be transfered to kernel
        }
        totalSegs += connectedPixelSize_host[i]; // increment counter
      if (pixelMapping->connectedPixelsSizes[superbin] > max_size){ max_size = pixelMapping->connectedPixelsSizes[superbin];} // set the maximum number of modules in a row
      }
      else if(pixelType ==1){
        connectedPixelSize_host[i] = pixelMapping->connectedPixelsSizesPos[superbin]; //number of pixel connected modules
        connectedPixelIndex_host[i] = pixelMapping->connectedPixelsIndexPos[superbin]+pixelIndexOffsetPos;// index to get start of connected pixel modules
//        if(connectedPixelIndex_host[i] >= pixelIndexOffsetNeg){printf("TESTX1 %d %d %u %d %u\n",ix,i,connectedPixelIndex_host[i],superbin,pixelMapping->connectedPixelsIndexPos[superbin]);}
        for (int j=0; j < pixelMapping->connectedPixelsSizesPos[superbin]; j++){
          segs_pix[totalSegs+j] = i;
          segs_pix_offset[totalSegs+j] = j;
        }
        totalSegs += connectedPixelSize_host[i];
      if (pixelMapping->connectedPixelsSizesPos[superbin]> max_size){ max_size = pixelMapping->connectedPixelsSizesPos[superbin];}
      }
      else if(pixelType ==2){
        connectedPixelSize_host[i] = pixelMapping->connectedPixelsSizesNeg[superbin]; //number of pixel connected modules
        connectedPixelIndex_host[i] =pixelMapping->connectedPixelsIndexNeg[superbin] + pixelIndexOffsetNeg;// index to get start of connected pixel modules
//        if(connectedPixelIndex_host[i] >= pixelIndexOffsetNeg+ pixelMapping->connectedPixelsIndexNeg[44999]){printf("TESTX2 %d %d %u %d %u\n",ix,i,connectedPixelIndex_host[i],superbin,pixelMapping->connectedPixelsIndexNeg[superbin]);}
        for (int j=0; j < pixelMapping->connectedPixelsSizesNeg[superbin]; j++){
          segs_pix[totalSegs+j] = i;
          segs_pix_offset[totalSegs+j] = j;
        }
        totalSegs += connectedPixelSize_host[i];
      if (pixelMapping->connectedPixelsSizesNeg[superbin] > max_size){max_size = pixelMapping->connectedPixelsSizesNeg[superbin];}
      }
      //if(i==230){printf("255: %d %d %u\n",pixelType,superbin,connectedPixelIndex_host[i]);}
      else{continue;}
    }

//    for(int k =0; k< i;k++){
//      if(connectedPixelIndex_host[k] > 3735957){printf("This Bad %d %u\n",k,connectedPixelIndex_host[i]);}
//    }
    cudaMemcpy(connectedPixelSize_dev, connectedPixelSize_host, nInnerSegments*sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(connectedPixelIndex_dev, connectedPixelIndex_host, nInnerSegments*sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(segs_pix_gpu,segs_pix,threadSize*sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(segs_pix_gpu_offset,segs_pix_offset,threadSize*sizeof(unsigned int), cudaMemcpyHostToDevice);
    
    dim3 nThreads(32,16,1);
    dim3 nBlocks((totalSegs % nThreads.x == 0 ? totalSegs / nThreads.x : totalSegs / nThreads.x + 1),
                  (max_size % nThreads.y == 0 ? max_size/nThreads.y : max_size/nThreads.y + 1),1);
    createPixelTrackletsInGPUFromMap<<<nBlocks,nThreads>>>(*modulesInGPU, *hitsInGPU, *mdsInGPU, *segmentsInGPU, *trackletsInGPU, 
    connectedPixelSize_dev,connectedPixelIndex_dev,i,segs_pix_gpu,segs_pix_gpu_offset);
    //connectedPixelSize_dev,connectedPixelIndex_dev,nInnerSegments,segs_pix_gpu,segs_pix_gpu_offset);

    cudaError_t cudaerr = cudaDeviceSynchronize();
    if(cudaerr != cudaSuccess)
      {
	std::cout<<"sync failed with error : "<<cudaGetErrorString(cudaerr)<<std::endl;

      }

    cudaFreeHost(connectedPixelSize_host);
    cudaFreeHost(connectedPixelIndex_host);
    cudaFree(connectedPixelSize_dev);
    cudaFree(connectedPixelIndex_dev);
#ifdef Explicit_Module
    cudaFreeHost(nSegments);
    cudaFreeHost(superbins);
    cudaFreeHost(pixelTypes);
#endif
    free(segs_pix);
    cudaFree(segs_pix_gpu);

#else
    printf("original 3D grid launching in createPixelTracklets does not exist");
    exit(2);
#endif
#endif

    unsigned int nPixelTracklets;
    cudaMemcpy(&nPixelTracklets, &(trackletsInGPU->nTracklets[nLowerModules]), sizeof(unsigned int), cudaMemcpyDeviceToHost);
}
void SDL::Event::createPixelTracklets()
{
    unsigned int nLowerModules;// = *modulesInGPU->nLowerModules;
    cudaMemcpy(&nLowerModules,modulesInGPU->nLowerModules,sizeof(unsigned int),cudaMemcpyDeviceToHost);
    if(trackletsInGPU == nullptr)
    {
        cudaMallocHost(&trackletsInGPU, sizeof(SDL::tracklets));
#ifdef Explicit_Tracklet
//        //FIXME:Change this to look like the unified allocator below after pixels have been incorporated!
        //createTrackletsInExplicitMemory(*trackletsInGPU,N_MAX_TRACKLETS_PER_MODULE, nLowerModules);
        createTrackletsInExplicitMemory(*trackletsInGPU, N_MAX_TRACKLETS_PER_MODULE , N_MAX_PIXEL_TRACKLETS_PER_MODULE, nLowerModules);
#else
        createTrackletsInUnifiedMemory(*trackletsInGPU, N_MAX_TRACKLETS_PER_MODULE , N_MAX_PIXEL_TRACKLETS_PER_MODULE, nLowerModules);
#endif
    }

#ifdef NESTED_PARA
    unsigned int nThreads = 1;
    unsigned int nBlocks = nLowerModules % nThreads == 0 ? nLowerModules/nThreads : nLowerModules/nThreads + 1;

    createPixelTrackletsInGPU<<<nBlocks,nThreads>>>(*modulesInGPU, *hitsInGPU, *mdsInGPU, *segmentsInGPU, *trackletsInGPU);

    cudaError_t cudaerr = cudaDeviceSynchronize();
    if(cudaerr != cudaSuccess)
    {
        std::cout<<"sync failed with error : "<<cudaGetErrorString(cudaerr)<<std::endl;

    }
#else
#ifdef NEWGRID_Pixel
#ifdef Explicit_Module
    unsigned int nModules; //= *modulesInGPU->nModules;
    cudaMemcpy(&nModules,modulesInGPU->nModules,sizeof(unsigned int),cudaMemcpyDeviceToHost);
#else
    unsigned int nModules = *modulesInGPU->nModules;
#endif
    unsigned int *nSegments = (unsigned int*)malloc(nModules*sizeof(unsigned int));
    cudaMemcpy((void *)nSegments, segmentsInGPU->nSegments, nModules*sizeof(unsigned int), cudaMemcpyDeviceToHost);
    unsigned int pixelModuleIndex = nModules - 1;
    unsigned int nInnerSegments = nSegments[pixelModuleIndex] > N_MAX_PIXEL_SEGMENTS_PER_MODULE ? N_MAX_PIXEL_SEGMENTS_PER_MODULE : nSegments[pixelModuleIndex];
#ifdef Explicit_Module
    unsigned int* lowerModuleIndices;
    cudaMallocHost(&lowerModuleIndices, (nLowerModules +1)* sizeof(unsigned int));
    cudaMemcpy(lowerModuleIndices,modulesInGPU->lowerModuleIndices,(nLowerModules+1)*sizeof(unsigned int),cudaMemcpyDeviceToHost);
#endif
    int threadSize = 100000;
    unsigned int *threadIdx = (unsigned int*)malloc(2*threadSize*sizeof(unsigned int));
    unsigned int *threadIdx_offset = threadIdx+threadSize;
    unsigned int *threadIdx_gpu;
    unsigned int *threadIdx_gpu_offset;
    cudaMalloc((void **)&threadIdx_gpu, 2*threadSize*sizeof(unsigned int));
    threadIdx_gpu_offset = threadIdx_gpu + threadSize;
    cudaMemset(threadIdx_gpu, nLowerModules, threadSize*sizeof(unsigned int));
    unsigned int totalCand=0;
    for (int i=0; i<nLowerModules; i++) {
#ifdef Explicit_Module
      unsigned int outerInnerLowerModuleIndex = lowerModuleIndices[i];
#else
      unsigned int outerInnerLowerModuleIndex = modulesInGPU->lowerModuleIndices[i];
#endif
      unsigned int nOuterSegments = nSegments[outerInnerLowerModuleIndex] > N_MAX_SEGMENTS_PER_MODULE ? N_MAX_SEGMENTS_PER_MODULE : nSegments[outerInnerLowerModuleIndex];
      if (nOuterSegments!=0) {
	for (int k=0; k<nOuterSegments; k++) {
          threadIdx[totalCand+k] = i;
          threadIdx_offset[totalCand+k] = k;
        }
	totalCand += nOuterSegments;
      }
    }

    if (threadSize < totalCand) {
      printf("threadSize=%d totalCand=%d: increase buffer size for threadIdx in createPixelTracklets\n", threadSize, totalCand);
      exit(1);
    }
    //printf("createPixelTracklets: nInnerSeg=%d totalCand=%d\n", nInnerSegments, totalCand);

    cudaMemcpy(threadIdx_gpu, threadIdx, threadSize*sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(threadIdx_gpu_offset, threadIdx_offset, threadSize*sizeof(unsigned int), cudaMemcpyHostToDevice);

    dim3 nThreads(16,32,1);
    dim3 nBlocks((nInnerSegments % nThreads.x == 0 ? nInnerSegments / nThreads.x : nInnerSegments / nThreads.x + 1),(totalCand % nThreads.y == 0 ? totalCand/nThreads.y : totalCand/nThreads.y + 1), 1);
    createPixelTrackletsInGPU<<<nBlocks,nThreads>>>(*modulesInGPU, *hitsInGPU, *mdsInGPU, *segmentsInGPU, *trackletsInGPU, threadIdx_gpu, threadIdx_gpu_offset);

    cudaError_t cudaerr = cudaDeviceSynchronize();
    if(cudaerr != cudaSuccess)
      {
	std::cout<<"sync failed with error : "<<cudaGetErrorString(cudaerr)<<std::endl;

      }

    free(nSegments);
    free(threadIdx);
    cudaFree(threadIdx_gpu);
#ifdef Explicit_Module
    cudaFreeHost(lowerModuleIndices);
#endif

#else
    printf("original 3D grid launching in createPixelTracklets does not exist");
    exit(2);
#endif
#endif

    unsigned int nPixelTracklets;
    cudaMemcpy(&nPixelTracklets, &(trackletsInGPU->nTracklets[nLowerModules]), sizeof(unsigned int), cudaMemcpyDeviceToHost);
#ifdef Warnings
    std::cout<<"number of pixel tracklets = "<<nPixelTracklets<<std::endl;
#endif
}

void SDL::Event::createTrackCandidates()
{
    unsigned int nLowerModules;// = *modulesInGPU->nLowerModules + 1; //including the pixel module
    cudaMemcpy(&nLowerModules,modulesInGPU->nLowerModules,sizeof(unsigned int),cudaMemcpyDeviceToHost);
    nLowerModules += 1;// include the pixel module

    //construct the list of eligible modules
    unsigned int nEligibleModules = 0;
    createEligibleModulesListForTrackCandidates(*modulesInGPU, nEligibleModules, N_MAX_TRACK_CANDIDATES_PER_MODULE);

    if(trackCandidatesInGPU == nullptr)
    {
        cudaMallocHost(&trackCandidatesInGPU, sizeof(SDL::trackCandidates));
#ifdef Explicit_Track
        createTrackCandidatesInExplicitMemory(*trackCandidatesInGPU, N_MAX_TRACK_CANDIDATES_PER_MODULE, N_MAX_PIXEL_TRACK_CANDIDATES_PER_MODULE, nLowerModules, nEligibleModules);
#else
        createTrackCandidatesInUnifiedMemory(*trackCandidatesInGPU, N_MAX_TRACK_CANDIDATES_PER_MODULE, N_MAX_PIXEL_TRACK_CANDIDATES_PER_MODULE, nLowerModules, nEligibleModules);
#endif
    }

#ifdef NESTED_PARA
    //auto t0 = std::chrono::high_resolution_clock::now();
    unsigned int nThreads = 1;
    unsigned int nBlocks = (nLowerModules-1) % nThreads == 0 ? (nLowerModules-1)/nThreads : (nLowerModules-1)/nThreads + 1;

    createTrackCandidatesInGPU<<<nBlocks,nThreads>>>(*modulesInGPU, *hitsInGPU, *mdsInGPU, *segmentsInGPU, *trackletsInGPU, *tripletsInGPU, *trackCandidatesInGPU);

    cudaError_t cudaerr = cudaDeviceSynchronize();
    if(cudaerr != cudaSuccess)
    {
        std::cout<<"sync failed with error : "<<cudaGetErrorString(cudaerr)<<std::endl;
    }

    //auto t1 = std::chrono::high_resolution_clock::now();
    //auto trackTime = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0); //in milliseconds
    //Pixel Track Candidates created separately
    nThreads = 1;
    nBlocks = (nLowerModules - 1) % nThreads == 0 ? (nLowerModules - 1)/nThreads : (nLowerModules - 1)/nThreads + 1;

    createPixelTrackCandidatesInGPU<<<nBlocks, nThreads>>>(*modulesInGPU, *hitsInGPU, *mdsInGPU, *segmentsInGPU, *trackletsInGPU, *tripletsInGPU, *trackCandidatesInGPU);

    cudaerr = cudaDeviceSynchronize();
    if(cudaerr != cudaSuccess)
    {
        std::cout<<"sync failed with error : "<<cudaGetErrorString(cudaerr)<<std::endl;
    }
    //auto t2 = std::chrono::high_resolution_clock::now();
    //auto pixelTime = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1); //in milliseconds

    //std::cout<<"trackTime="<<trackTime.count()<<"pixelTime="<<pixelTime.count()<<std::endl;

#else
#ifdef NEWGRID_Track
    //auto t0 = std::chrono::high_resolution_clock::now();
    int maxOuterTr = max(N_MAX_TRACKLETS_PER_MODULE, N_MAX_TRIPLETS_PER_MODULE);
    unsigned int *nTriplets = (unsigned int*)malloc((2*nLowerModules-1)*sizeof(unsigned int));
    //unsigned int *nTracklets = (unsigned int*)malloc(nLowerModules*sizeof(unsigned int));
    unsigned int *nTracklets = nTriplets + nLowerModules -1;
    int threadSize=230000;
    unsigned int *threadIdx = (unsigned int*)malloc(2*threadSize*sizeof(unsigned int));
    unsigned int *threadIdx_offset = threadIdx+threadSize;
    unsigned int *threadIdx_gpu;
    unsigned int *threadIdx_gpu_offset;
    cudaMalloc((void **)&threadIdx_gpu, 2*threadSize*sizeof(unsigned int));
    threadIdx_gpu_offset = threadIdx_gpu + threadSize;
    cudaMemset(threadIdx_gpu, nLowerModules, threadSize*sizeof(unsigned int));
    cudaMemcpy(nTriplets, tripletsInGPU->nTriplets, (nLowerModules-1)*sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(nTracklets, trackletsInGPU->nTracklets, nLowerModules*sizeof(unsigned int), cudaMemcpyDeviceToHost);
    int nPixelTracklets= nTracklets[nLowerModules-1];
    if(nPixelTracklets > N_MAX_PIXEL_TRACKLETS_PER_MODULE)
      nPixelTracklets = N_MAX_PIXEL_TRACKLETS_PER_MODULE;
    unsigned int totalCand=0;
    for (int i=0; i< nLowerModules-1; i++) {
      unsigned int nInnerTracklets = nTracklets[i];
      if(nInnerTracklets > N_MAX_TRACKLETS_PER_MODULE)
	nInnerTracklets = N_MAX_TRACKLETS_PER_MODULE;
      unsigned int nInnerTriplets = nTriplets[i];
      if(nInnerTriplets > N_MAX_TRIPLETS_PER_MODULE)
        nInnerTriplets = N_MAX_TRIPLETS_PER_MODULE;
      unsigned int temp = max(nInnerTracklets, nInnerTriplets);
      if (temp !=0) {
        for (int k=0; k<temp; k++) {
          threadIdx[totalCand+k] = i;
          threadIdx_offset[totalCand+k] = k;
        }
	totalCand += temp;
      }
    }
    if (threadSize < totalCand) {
      printf("threadSize=%d totalCand=%d: Increase buffer size for threadIdx in createTrackCandidates\n", threadSize, totalCand);
      exit(2);
    }
    //printf("totalCand=%d\n", totalCand);
    cudaMemcpy(threadIdx_gpu, threadIdx, threadSize*sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(threadIdx_gpu_offset, threadIdx_offset, threadSize*sizeof(unsigned int), cudaMemcpyHostToDevice);
    //auto t1 = std::chrono::high_resolution_clock::now();
    //auto prepareTime = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0); //in milliseconds

    dim3 nThreads(16, 32, 1);
    dim3 nBlocks((maxOuterTr % nThreads.x == 0 ? maxOuterTr/nThreads.x : maxOuterTr/nThreads.x + 1), (totalCand % nThreads.y == 0 ? totalCand/nThreads.y : totalCand/nThreads.y + 1), 1);
    createTrackCandidatesInGPU<<<nBlocks,nThreads>>>(*modulesInGPU, *hitsInGPU, *mdsInGPU, *segmentsInGPU, *trackletsInGPU, *tripletsInGPU, *trackCandidatesInGPU, threadIdx_gpu, threadIdx_gpu_offset);
    cudaError_t cudaerr = cudaDeviceSynchronize();
    if(cudaerr != cudaSuccess)
      {
	std::cout<<"sync failed with error : "<<cudaGetErrorString(cudaerr)<<std::endl;
      }
    //auto t2 = std::chrono::high_resolution_clock::now();
    //auto trackTime = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1); //in milliseconds

    dim3 nThreads_p(16,16,1);
    dim3 nBlocks_p((nPixelTracklets % nThreads_p.x == 0 ? nPixelTracklets/nThreads_p.x : nPixelTracklets/nThreads_p.x + 1), (totalCand % nThreads_p.y == 0 ? totalCand/nThreads_p.y : totalCand/nThreads_p.y + 1), 1);
    createPixelTrackCandidatesInGPU<<<nBlocks_p, nThreads_p>>>(*modulesInGPU, *hitsInGPU, *mdsInGPU, *segmentsInGPU, *trackletsInGPU, *tripletsInGPU, *trackCandidatesInGPU, threadIdx_gpu, threadIdx_gpu_offset);
    cudaerr = cudaDeviceSynchronize();
    if(cudaerr != cudaSuccess)
      {
	std::cout<<"sync failed with error : "<<cudaGetErrorString(cudaerr)<<std::endl;
      }

    //auto t3 = std::chrono::high_resolution_clock::now();
    //auto pixelTime = std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2); //in milliseconds

    //std::cout<<"prepareTime="<<prepareTime.count()<<"trackTime="<<trackTime.count()<<"pixelTime="<<pixelTime.count()<<std::endl;

    free(threadIdx);
    free(nTriplets);
    //free(nTracklets);
    cudaFree(threadIdx_gpu);
#else
    printf("original 3D grid launching in createTrackCandidates does not exist");
    exit(3);
#endif
#endif

#if defined(AddObjects)
#ifdef Explicit_Track
    addTrackCandidatesToEventExplicit();
#else
    addTrackCandidatesToEvent();
#endif
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


#ifdef NESTED_PARA
    unsigned int nThreads = 1;
    unsigned int nBlocks = nLowerModules % nThreads == 0 ? nLowerModules/nThreads : nLowerModules/nThreads + 1;
    createQuintupletsInGPU<<<nBlocks,nThreads>>>(*modulesInGPU, *hitsInGPU, *mdsInGPU, *segmentsInGPU, *tripletsInGPU, *quintupletsInGPU);


    cudaError_t cudaerr = cudaDeviceSynchronize();
    if(cudaerr != cudaSuccess)
    {
	    std::cout<<"sync failed with error : "<<cudaGetErrorString(cudaerr)<<std::endl;
    }
#else
#ifdef NEWGRID_T5

    unsigned int *index_gpu;
    cudaMalloc((void **)&index_gpu, nLowerModules*sizeof(unsigned int));
    cudaMemcpy(index_gpu, indicesOfEligibleModules, nEligibleT5Modules * sizeof(unsigned int), cudaMemcpyHostToDevice);

    int max_InnerTriplets = min(maxTriplets, N_MAX_TRIPLETS_PER_MODULE); 
    int nonZeroModules = nEligibleT5Modules;
    int max_outerTriplets = N_MAX_TRIPLETS_PER_MODULE;
//    printf("nonZeroModules=%d max_InnerTriplets=%d max_outerTriplets=%d\n", nonZeroModules, max_InnerTriplets, max_outerTriplets);
    dim3 nThreads(16,16,1);

    dim3 nBlocks((max_outerTriplets % nThreads.x == 0 ? max_outerTriplets / nThreads.x : max_outerTriplets / nThreads.x + 1),(max_InnerTriplets % nThreads.y == 0 ? max_InnerTriplets/nThreads.y : max_InnerTriplets/nThreads.y + 1), (nonZeroModules % nThreads.z == 0 ? nonZeroModules/nThreads.z : nonZeroModules/nThreads.z + 1));

    createQuintupletsInGPU<<<nBlocks,nThreads>>>(*modulesInGPU, *hitsInGPU, *mdsInGPU, *segmentsInGPU, *tripletsInGPU, *quintupletsInGPU, index_gpu);
    cudaError_t cudaerr = cudaDeviceSynchronize();
    if(cudaerr != cudaSuccess)
    {
	    std::cout<<"sync failed with error : "<<cudaGetErrorString(cudaerr)<<std::endl;
    }
    cudaFree(index_gpu);

#else
    printf("original 3D grid launching in createQuintuplets does not exist");
    exit(3);
#endif
#endif
free(indicesOfEligibleModules);

#if defined(AddObjects)
#ifdef Explicit_T5
    addQuintupletsToEventExplicit();
#else
    addQuintupletsToEvent();
#endif
#endif

}

void SDL::Event::createTrackletsWithAGapWithModuleMap()
{
    //use the same trackletsInGPU as before if it exists
    unsigned int nLowerModules;// = *modulesInGPU->nLowerModules;
    cudaMemcpy(&nLowerModules,modulesInGPU->nLowerModules,sizeof(unsigned int),cudaMemcpyDeviceToHost);

    //TRCAKLETS - To conserve memory, we shall be only declaring nLowerModules amount of memory!!!!!!!
    if(trackletsInGPU == nullptr)
    {
        cudaMallocHost(&trackletsInGPU, sizeof(SDL::tracklets));
#ifdef Explicit_Tracklet
        //createTrackletsInExplicitMemory(*trackletsInGPU, N_MAX_TRACKLETS_PER_MODULE , nLowerModules);
        createTrackletsInExplicitMemory(*trackletsInGPU, N_MAX_TRACKLETS_PER_MODULE , N_MAX_PIXEL_TRACKLETS_PER_MODULE, nLowerModules);
#else
        createTrackletsInUnifiedMemory(*trackletsInGPU, N_MAX_TRACKLETS_PER_MODULE , N_MAX_PIXEL_TRACKLETS_PER_MODULE, nLowerModules);
#endif
    }

    unsigned int nThreads = 1;
    unsigned int nBlocks = nLowerModules % nThreads == 0 ? nLowerModules/nThreads : nLowerModules/nThreads + 1;

    createTrackletsWithAGapInGPU<<<nBlocks,nThreads>>>(*modulesInGPU, *hitsInGPU, *mdsInGPU, *segmentsInGPU, *trackletsInGPU);

    cudaError_t cudaerr = cudaDeviceSynchronize();
    if(cudaerr != cudaSuccess)
    {
        std::cout<<"sync failed with error : "<<cudaGetErrorString(cudaerr)<<std::endl;

    }

}


void SDL::Event::addTrackletsToEvent()
{
    unsigned int idx;
    for(unsigned int i = 0; i<*(SDL::modulesInGPU->nLowerModules); i++)
    {
        idx = SDL::modulesInGPU->lowerModuleIndices[i];
        //tracklets run only on lower modules!!!!!!
        if(trackletsInGPU->nTracklets[i] == 0)
        {
            modulesInGPU->trackletRanges[idx * 2] = -1;
            modulesInGPU->trackletRanges[idx * 2 + 1] = -1;
        }
        else
        {
            modulesInGPU->trackletRanges[idx * 2] = idx * N_MAX_TRACKLETS_PER_MODULE;
            modulesInGPU->trackletRanges[idx * 2 + 1] = idx * N_MAX_TRACKLETS_PER_MODULE + trackletsInGPU->nTracklets[i] - 1;


            if(modulesInGPU->subdets[idx] == Barrel)
            {
                n_tracklets_by_layer_barrel_[modulesInGPU->layers[idx] - 1] += trackletsInGPU->nTracklets[i];
            }
            else
            {
                n_tracklets_by_layer_endcap_[modulesInGPU->layers[idx] - 1] += trackletsInGPU->nTracklets[i];
            }
        }
    }
}
void SDL::Event::addTrackletsToEventExplicit()
{
unsigned int nLowerModules;
cudaMemcpy(&nLowerModules,modulesInGPU->nLowerModules,sizeof(unsigned int),cudaMemcpyDeviceToHost);

unsigned int* nTrackletsCPU;
cudaMallocHost(&nTrackletsCPU, nLowerModules * sizeof(unsigned int));
cudaMemcpy(nTrackletsCPU,trackletsInGPU->nTracklets,nLowerModules*sizeof(unsigned int),cudaMemcpyDeviceToHost);

short* module_subdets;
cudaMallocHost(&module_subdets, nModules* sizeof(short));
cudaMemcpy(module_subdets,modulesInGPU->subdets,nModules*sizeof(short),cudaMemcpyDeviceToHost);
unsigned int* module_lowerModuleIndices;
cudaMallocHost(&module_lowerModuleIndices, (nLowerModules +1)* sizeof(unsigned int));
cudaMemcpy(module_lowerModuleIndices,modulesInGPU->lowerModuleIndices,(nLowerModules+1)*sizeof(unsigned int),cudaMemcpyDeviceToHost);
int* module_trackletRanges;
cudaMallocHost(&module_trackletRanges, nModules* 2*sizeof(int));
cudaMemcpy(module_trackletRanges,modulesInGPU->trackletRanges,nModules*2*sizeof(int),cudaMemcpyDeviceToHost);
short* module_layers;
cudaMallocHost(&module_layers, nModules * sizeof(short));
cudaMemcpy(module_layers,modulesInGPU->layers,nModules*sizeof(short),cudaMemcpyDeviceToHost);
    unsigned int idx;
    for(unsigned int i = 0; i<nLowerModules; i++)
    {
        idx = module_lowerModuleIndices[i];
        //tracklets run only on lower modules!!!!!!
        if(nTrackletsCPU[i] == 0)
        {
            module_trackletRanges[idx * 2] = -1;
            module_trackletRanges[idx * 2 + 1] = -1;
        }
        else
        {
            module_trackletRanges[idx * 2] = idx * N_MAX_TRACKLETS_PER_MODULE;
            module_trackletRanges[idx * 2 + 1] = idx * N_MAX_TRACKLETS_PER_MODULE + nTrackletsCPU[i] - 1;


            if(module_subdets[idx] == Barrel)
            {
                n_tracklets_by_layer_barrel_[module_layers[idx] - 1] += nTrackletsCPU[i];
            }
            else
            {
                n_tracklets_by_layer_endcap_[module_layers[idx] - 1] += nTrackletsCPU[i];
            }
        }
    }
cudaFreeHost(nTrackletsCPU);
cudaFreeHost(module_subdets);
cudaFreeHost(module_lowerModuleIndices);
cudaFreeHost(module_trackletRanges);
cudaFreeHost(module_layers);
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
#ifndef NESTED_PARA
__global__ void createMiniDoubletsInGPU(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU)
{
    int lowerModuleArrayIndex = blockIdx.x * blockDim.x + threadIdx.x;
    //int lowerHitIndex = blockIdx.y * blockDim.y + threadIdx.y;
    //int upperHitIndex = blockIdx.z * blockDim.z + threadIdx.z;
    if(lowerModuleArrayIndex >= (*modulesInGPU.nLowerModules)) return; //extra precaution

    int lowerModuleIndex = modulesInGPU.lowerModuleIndices[lowerModuleArrayIndex];
    int upperModuleIndex = modulesInGPU.partnerModuleIndex(lowerModuleIndex);

    if(modulesInGPU.hitRanges[lowerModuleIndex * 2] == -1) return;
    if(modulesInGPU.hitRanges[upperModuleIndex * 2] == -1) return;
    unsigned int nLowerHits = modulesInGPU.hitRanges[lowerModuleIndex * 2 + 1] - modulesInGPU.hitRanges[lowerModuleIndex * 2] + 1;
    unsigned int nUpperHits = modulesInGPU.hitRanges[upperModuleIndex * 2 + 1] - modulesInGPU.hitRanges[upperModuleIndex * 2] + 1;

#ifdef NEWGRID_MD
    int lowerHitIndex =  (blockIdx.y * blockDim.y + threadIdx.y) / nUpperHits;
    int upperHitIndex =  (blockIdx.y * blockDim.y + threadIdx.y) % nUpperHits;
#else
    int lowerHitIndex = blockIdx.y * blockDim.y + threadIdx.y;
    int upperHitIndex = blockIdx.z * blockDim.z + threadIdx.z;
#endif

    //consider assigining a dummy computation function for these
    if(lowerHitIndex >= nLowerHits) return;
    if(upperHitIndex >= nUpperHits) return;

    unsigned int lowerHitArrayIndex = modulesInGPU.hitRanges[lowerModuleIndex * 2] + lowerHitIndex;
    unsigned int upperHitArrayIndex = modulesInGPU.hitRanges[upperModuleIndex * 2] + upperHitIndex;

    float dz, drt, dphi, dphichange, shiftedX, shiftedY, shiftedZ, noShiftedDz, noShiftedDphi, noShiftedDphiChange;

#ifdef CUT_VALUE_DEBUG
    float dzCut, drtCut, miniCut;
    bool success = runMiniDoubletDefaultAlgo(modulesInGPU, hitsInGPU, lowerModuleIndex, lowerHitArrayIndex, upperHitArrayIndex, dz,  drt, dphi, dphichange, shiftedX, shiftedY, shiftedZ, noShiftedDz, noShiftedDphi, noShiftedDphiChange, dzCut, drtCut, miniCut);
#else
    bool success = runMiniDoubletDefaultAlgo(modulesInGPU, hitsInGPU, lowerModuleIndex, lowerHitArrayIndex, upperHitArrayIndex, dz, dphi, dphichange, shiftedX, shiftedY, shiftedZ, noShiftedDz, noShiftedDphi, noShiftedDphiChange);
#endif

    if(success)
    {
        unsigned int mdModuleIndex = atomicAdd(&mdsInGPU.nMDs[lowerModuleIndex],1);
        if(mdModuleIndex >= N_MAX_MD_PER_MODULES)
        {
            #ifdef Warnings
            if(mdModuleIndex == N_MAX_MD_PER_MODULES)
                printf("Mini-doublet excess alert! Module index =  %d\n",lowerModuleIndex);
            #endif
        }
        else
        {
            unsigned int mdIndex = lowerModuleIndex * N_MAX_MD_PER_MODULES + mdModuleIndex;
#ifdef CUT_VALUE_DEBUG
            addMDToMemory(mdsInGPU,hitsInGPU, modulesInGPU, lowerHitArrayIndex, upperHitArrayIndex, lowerModuleIndex, dz,drt, dphi, dphichange, shiftedX, shiftedY, shiftedZ, noShiftedDz, noShiftedDphi, noShiftedDphiChange, dzCut, drtCut, miniCut, mdIndex);
#else
        addMDToMemory(mdsInGPU,hitsInGPU, modulesInGPU, lowerHitArrayIndex, upperHitArrayIndex, lowerModuleIndex, dz, dphi, dphichange, shiftedX, shiftedY, shiftedZ, noShiftedDz, noShiftedDphi, noShiftedDphiChange, mdIndex);
#endif

        }

    }
}
#else
__global__ void createMiniDoubletsFromLowerModule(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, unsigned int lowerModuleIndex, unsigned int upperModuleIndex, unsigned int nLowerHits, unsigned int nUpperHits)
{
    unsigned int lowerHitIndex = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int upperHitIndex = blockIdx.z * blockDim.z + threadIdx.z;

    //consider assigining a dummy computation function for these
    if(lowerHitIndex >= nLowerHits) return;
    if(upperHitIndex >= nUpperHits) return;

    unsigned int lowerHitArrayIndex = modulesInGPU.hitRanges[lowerModuleIndex * 2] + lowerHitIndex;
    unsigned int upperHitArrayIndex = modulesInGPU.hitRanges[upperModuleIndex * 2] + upperHitIndex;

    float dz, drt, dphi, dphichange, shiftedX, shiftedY, shiftedZ, noShiftedDz, noShiftedDphi, noShiftedDphiChange;

#ifdef CUT_VALUE_DEBUG
    float dzCut, drtCut, miniCut;
    bool success = runMiniDoubletDefaultAlgo(modulesInGPU, hitsInGPU, lowerModuleIndex, lowerHitArrayIndex, upperHitArrayIndex, dz,  drt, dphi, dphichange, shiftedX, shiftedY, shiftedZ, noShiftedDz, noShiftedDphi, noShiftedDphiChange, dzCut, drtCut, miniCut);
#else
    bool success = runMiniDoubletDefaultAlgo(modulesInGPU, hitsInGPU, lowerModuleIndex, lowerHitArrayIndex, upperHitArrayIndex, dz, dphi, dphichange, shiftedX, shiftedY, shiftedZ, noShiftedDz, noShiftedDphi, noShiftedDphiChange);
#endif

    if(success)
    {
        unsigned int mdModuleIndex = atomicAdd(&mdsInGPU.nMDs[lowerModuleIndex],1);

        if(mdModuleIndex >= N_MAX_MD_PER_MODULES)
        {
            #ifdef Warnings
            if(mdModuleIndex == N_MAX_MD_PER_MODULES)
                printf("Mini-doublet excess alert! Module index = %d\n",lowerModuleIndex);
            #endif
        }
        else
        {
            unsigned int mdIndex = lowerModuleIndex * N_MAX_MD_PER_MODULES + mdModuleIndex;
#ifdef CUT_VALUE_DEBUG
            addMDToMemory(mdsInGPU,hitsInGPU, modulesInGPU, lowerHitArrayIndex, upperHitArrayIndex, lowerModuleIndex, dz,drt, dphi, dphichange, shiftedX, shiftedY, shiftedZ, noShiftedDz, noShiftedDphi, noShiftedDphiChange, dzCut, drtCut, miniCut, mdIndex);
#else
            addMDToMemory(mdsInGPU,hitsInGPU, modulesInGPU, lowerHitArrayIndex, upperHitArrayIndex, lowerModuleIndex, dz, dphi, dphichange, shiftedX, shiftedY, shiftedZ, noShiftedDz, noShiftedDphi, noShiftedDphiChange, mdIndex);
#endif
        }

    }
}


__global__ void createMiniDoubletsInGPU(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU)
{
    int lowerModuleArrayIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if(lowerModuleArrayIndex >= (*modulesInGPU.nLowerModules)) return; //extra precaution

    int lowerModuleIndex = modulesInGPU.lowerModuleIndices[lowerModuleArrayIndex];
    int upperModuleIndex = modulesInGPU.partnerModuleIndex(lowerModuleIndex);

    if(modulesInGPU.hitRanges[lowerModuleIndex * 2] == -1) return;
    if(modulesInGPU.hitRanges[upperModuleIndex * 2] == -1) return;

    unsigned int nLowerHits = modulesInGPU.hitRanges[lowerModuleIndex * 2 + 1] - modulesInGPU.hitRanges[lowerModuleIndex * 2] + 1;
    unsigned int nUpperHits = modulesInGPU.hitRanges[upperModuleIndex * 2 + 1] - modulesInGPU.hitRanges[upperModuleIndex * 2] + 1;

    dim3 nThreads(1,16,16);
    dim3 nBlocks(1,nLowerHits % nThreads.y == 0 ? nLowerHits/nThreads.y : nLowerHits/nThreads.y + 1, nUpperHits % nThreads.z == 0 ? nUpperHits/nThreads.z : nUpperHits/nThreads.z + 1);

    createMiniDoubletsFromLowerModule<<<nBlocks,nThreads>>>(modulesInGPU, hitsInGPU, mdsInGPU, lowerModuleIndex, upperModuleIndex, nLowerHits, nUpperHits);


}
#endif

#ifndef NESTED_PARA
__global__ void createSegmentsInGPU(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU)
{
#ifdef NEWGRID_Seg
    int innerLowerModuleArrayIdx = blockIdx.z * blockDim.z + threadIdx.z;
    int outerLowerModuleArrayIdx = blockIdx.y * blockDim.y + threadIdx.y;
#else
    int xAxisIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int innerMDArrayIdx = blockIdx.y * blockDim.y + threadIdx.y;
    int outerMDArrayIdx = blockIdx.z * blockDim.z + threadIdx.z;

    int innerLowerModuleArrayIdx = xAxisIdx/MAX_CONNECTED_MODULES;
    int outerLowerModuleArrayIdx = xAxisIdx % MAX_CONNECTED_MODULES; //need this index from the connected module array
#endif
    if(innerLowerModuleArrayIdx >= *modulesInGPU.nLowerModules) return;

    unsigned int innerLowerModuleIndex = modulesInGPU.lowerModuleIndices[innerLowerModuleArrayIdx];

    unsigned int nConnectedModules = modulesInGPU.nConnectedModules[innerLowerModuleIndex];

    if(outerLowerModuleArrayIdx >= nConnectedModules) return;

    unsigned int outerLowerModuleIndex = modulesInGPU.moduleMap[innerLowerModuleIndex * MAX_CONNECTED_MODULES + outerLowerModuleArrayIdx];

    unsigned int nInnerMDs = mdsInGPU.nMDs[innerLowerModuleIndex] > N_MAX_MD_PER_MODULES ? N_MAX_MD_PER_MODULES : mdsInGPU.nMDs[innerLowerModuleIndex];
    unsigned int nOuterMDs = mdsInGPU.nMDs[outerLowerModuleIndex] > N_MAX_MD_PER_MODULES ? N_MAX_MD_PER_MODULES : mdsInGPU.nMDs[outerLowerModuleIndex];

#ifdef NEWGRID_Seg
    if (nInnerMDs*nOuterMDs == 0) return;
    int innerMDArrayIdx = (blockIdx.x * blockDim.x + threadIdx.x) / nOuterMDs;
    int outerMDArrayIdx = (blockIdx.x * blockDim.x + threadIdx.x) % nOuterMDs;
#endif

    if(innerMDArrayIdx >= nInnerMDs) return;
    if(outerMDArrayIdx >= nOuterMDs) return;

    unsigned int innerMDIndex = modulesInGPU.mdRanges[innerLowerModuleIndex * 2] + innerMDArrayIdx;
    unsigned int outerMDIndex = modulesInGPU.mdRanges[outerLowerModuleIndex * 2] + outerMDArrayIdx;

    float zIn, zOut, rtIn, rtOut, dPhi, dPhiMin, dPhiMax, dPhiChange, dPhiChangeMin, dPhiChangeMax, dAlphaInnerMDSegment, dAlphaOuterMDSegment, dAlphaInnerMDOuterMD;

    unsigned int innerMiniDoubletAnchorHitIndex, outerMiniDoubletAnchorHitIndex;

    dPhiMin = 0;
    dPhiMax = 0;
    dPhiChangeMin = 0;
    dPhiChangeMax = 0;
#ifdef CUT_VALUE_DEBUG
    float zLo, zHi, rtLo, rtHi, sdCut, dAlphaInnerMDSegmentThreshold, dAlphaOuterMDSegmentThreshold, dAlphaInnerMDOuterMDThreshold;

    bool success = runSegmentDefaultAlgo(modulesInGPU, hitsInGPU, mdsInGPU, innerLowerModuleIndex, outerLowerModuleIndex, innerMDIndex, outerMDIndex, zIn, zOut, rtIn, rtOut, dPhi, dPhiMin, dPhiMax, dPhiChange, dPhiChangeMin, dPhiChangeMax, dAlphaInnerMDSegment, dAlphaOuterMDSegment, dAlphaInnerMDOuterMD, zLo, zHi, rtLo, rtHi, sdCut, dAlphaInnerMDSegmentThreshold, dAlphaOuterMDSegmentThreshold,
            dAlphaInnerMDOuterMDThreshold, innerMiniDoubletAnchorHitIndex, outerMiniDoubletAnchorHitIndex);

#else
    bool success = runSegmentDefaultAlgo(modulesInGPU, hitsInGPU, mdsInGPU, innerLowerModuleIndex, outerLowerModuleIndex, innerMDIndex, outerMDIndex, zIn, zOut, rtIn, rtOut, dPhi, dPhiMin, dPhiMax, dPhiChange, dPhiChangeMin, dPhiChangeMax, dAlphaInnerMDSegment, dAlphaOuterMDSegment, dAlphaInnerMDOuterMD, innerMiniDoubletAnchorHitIndex, outerMiniDoubletAnchorHitIndex);
#endif

    if(success)
    {
        unsigned int segmentModuleIdx = atomicAdd(&segmentsInGPU.nSegments[innerLowerModuleIndex],1);
        if(segmentModuleIdx >= N_MAX_SEGMENTS_PER_MODULE)
        {
            #ifdef Warnings
            if(segmentModuleIdx == N_MAX_SEGMENTS_PER_MODULE)
                printf("Segment excess alert! Module index = %d\n",innerLowerModuleIndex);
            #endif
        }
        else
        {
            unsigned int segmentIdx = innerLowerModuleIndex * N_MAX_SEGMENTS_PER_MODULE + segmentModuleIdx;
#ifdef CUT_VALUE_DEBUG
            addSegmentToMemory(segmentsInGPU,innerMDIndex, outerMDIndex,innerLowerModuleIndex, outerLowerModuleIndex, innerMiniDoubletAnchorHitIndex, outerMiniDoubletAnchorHitIndex, dPhi, dPhiMin, dPhiMax, dPhiChange, dPhiChangeMin, dPhiChangeMax, zIn, zOut, rtIn, rtOut, dAlphaInnerMDSegment, dAlphaOuterMDSegment, dAlphaInnerMDOuterMD, zLo, zHi, rtLo, rtHi, sdCut, dAlphaInnerMDSegmentThreshold, dAlphaOuterMDSegmentThreshold,
                dAlphaInnerMDOuterMDThreshold, segmentIdx);
#else
            addSegmentToMemory(segmentsInGPU,innerMDIndex, outerMDIndex,innerLowerModuleIndex, outerLowerModuleIndex, innerMiniDoubletAnchorHitIndex, outerMiniDoubletAnchorHitIndex, dPhi, dPhiMin, dPhiMax, dPhiChange, dPhiChangeMin, dPhiChangeMax, zIn, zOut, rtIn, rtOut, dAlphaInnerMDSegment, dAlphaOuterMDSegment, dAlphaInnerMDOuterMD,segmentIdx);
#endif

        }
    }
}
#else

__global__ void createSegmentsFromInnerLowerModule(struct SDL::modules&modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, unsigned int innerLowerModuleIndex, unsigned int nInnerMDs)
{
    unsigned int outerLowerModuleArrayIndex = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int innerMDArrayIndex = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int outerMDArrayIndex = blockIdx.z * blockDim.z + threadIdx.z;

    unsigned int outerLowerModuleIndex = modulesInGPU.moduleMap[innerLowerModuleIndex * MAX_CONNECTED_MODULES + outerLowerModuleArrayIndex];

    unsigned int nOuterMDs = mdsInGPU.nMDs[outerLowerModuleIndex] > N_MAX_MD_PER_MODULES ? N_MAX_MD_PER_MODULES : mdsInGPU.nMDs[outerLowerModuleIndex];
    if(innerMDArrayIndex >= nInnerMDs) return;
    if(outerMDArrayIndex >= nOuterMDs) return;

    unsigned int innerMDIndex = innerLowerModuleIndex * N_MAX_MD_PER_MODULES + innerMDArrayIndex;
    unsigned int outerMDIndex = outerLowerModuleIndex * N_MAX_MD_PER_MODULES + outerMDArrayIndex;

    float zIn, zOut, rtIn, rtOut, dPhi, dPhiMin, dPhiMax, dPhiChange, dPhiChangeMin, dPhiChangeMax, dAlphaInnerMDSegment, dAlphaOuterMDSegment, dAlphaInnerMDOuterMD;

    unsigned int innerMiniDoubletAnchorHitIndex, outerMiniDoubletAnchorHitIndex;

    dPhiMin = 0;
    dPhiMax = 0;
    dPhiChangeMin = 0;
    dPhiChangeMax = 0;
#ifdef CUT_VALUE_DEBUG
    float zLo, zHi, rtLo, rtHi, sdCut, dAlphaInnerMDSegmentThreshold, dAlphaOuterMDSegmentThreshold, dAlphaInnerMDOuterMDThreshold;

    bool success = runSegmentDefaultAlgo(modulesInGPU, hitsInGPU, mdsInGPU, innerLowerModuleIndex, outerLowerModuleIndex, innerMDIndex, outerMDIndex, zIn, zOut, rtIn, rtOut, dPhi, dPhiMin, dPhiMax, dPhiChange, dPhiChangeMin, dPhiChangeMax, dAlphaInnerMDSegment, dAlphaOuterMDSegment, dAlphaInnerMDOuterMD, zLo, zHi, rtLo, rtHi, sdCut, dAlphaInnerMDSegmentThreshold, dAlphaOuterMDSegmentThreshold,
            dAlphaInnerMDOuterMDThreshold, innerMiniDoubletAnchorHitIndex, outerMiniDoubletAnchorHitIndex);

#else
    bool success = runSegmentDefaultAlgo(modulesInGPU, hitsInGPU, mdsInGPU, innerLowerModuleIndex, outerLowerModuleIndex, innerMDIndex, outerMDIndex, zIn, zOut, rtIn, rtOut, dPhi, dPhiMin, dPhiMax, dPhiChange, dPhiChangeMin, dPhiChangeMax, dAlphaInnerMDSegment, dAlphaOuterMDSegment, dAlphaInnerMDOuterMD, innerMiniDoubletAnchorHitIndex, outerMiniDoubletAnchorHitIndex);
#endif

    if(success)
    {
        unsigned int segmentModuleIdx = atomicAdd(&segmentsInGPU.nSegments[innerLowerModuleIndex],1);
        if(segmentModuleIdx >= N_MAX_SEGMENTS_PER_MODULE)
        {
            #ifdef Warnings
            if(segmentModuleIdx == N_MAX_SEGMENTS_PER_MODULE)
                printf("Segment excess alert! Module index = %d\n",innerLowerModuleIndex);
            #endif
        }
        else
        {
            unsigned int segmentIdx = innerLowerModuleIndex * N_MAX_SEGMENTS_PER_MODULE + segmentModuleIdx;
#ifdef CUT_VALUE_DEBUG
            addSegmentToMemory(segmentsInGPU,innerMDIndex, outerMDIndex,innerLowerModuleIndex, outerLowerModuleIndex, innerMiniDoubletAnchorHitIndex, outerMiniDoubletAnchorHitIndex, dPhi, dPhiMin, dPhiMax, dPhiChange, dPhiChangeMin, dPhiChangeMax, zIn, zOut, rtIn, rtOut, dAlphaInnerMDSegment, dAlphaOuterMDSegment, dAlphaInnerMDOuterMD, zLo, zHi, rtLo, rtHi, sdCut, dAlphaInnerMDSegmentThreshold, dAlphaOuterMDSegmentThreshold,
                dAlphaInnerMDOuterMDThreshold, segmentIdx);
#else
            addSegmentToMemory(segmentsInGPU,innerMDIndex, outerMDIndex,innerLowerModuleIndex, outerLowerModuleIndex, innerMiniDoubletAnchorHitIndex, outerMiniDoubletAnchorHitIndex, dPhi, dPhiMin, dPhiMax, dPhiChange, dPhiChangeMin, dPhiChangeMax, zIn, zOut, rtIn, rtOut, dAlphaInnerMDSegment, dAlphaOuterMDSegment, dAlphaInnerMDOuterMD,segmentIdx);
#endif

        }

    }

}

__global__ void createSegmentsInGPU(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU)
{
    int innerLowerModuleArrayIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if(innerLowerModuleArrayIndex >= *modulesInGPU.nLowerModules) return;
    unsigned int innerLowerModuleIndex = modulesInGPU.lowerModuleIndices[innerLowerModuleArrayIndex];
    unsigned int nConnectedModules = modulesInGPU.nConnectedModules[innerLowerModuleIndex];
    unsigned int nInnerMDs = mdsInGPU.nMDs[innerLowerModuleIndex] > N_MAX_MD_PER_MODULES ? N_MAX_MD_PER_MODULES : mdsInGPU.nMDs[innerLowerModuleIndex];

    if(nConnectedModules == 0) return;

    if(nInnerMDs == 0) return;
    dim3 nThreads(1,16,16);
    dim3 nBlocks((nConnectedModules % nThreads.x == 0 ? nConnectedModules/nThreads.x : nConnectedModules/nThreads.x + 1), (nInnerMDs % nThreads.y == 0 ? nInnerMDs/nThreads.y : nInnerMDs/nThreads.y + 1), (N_MAX_MD_PER_MODULES % nThreads.z == 0 ? N_MAX_MD_PER_MODULES/nThreads.z : N_MAX_MD_PER_MODULES/nThreads.z + 1));

    createSegmentsFromInnerLowerModule<<<nBlocks,nThreads>>>(modulesInGPU, hitsInGPU, mdsInGPU, segmentsInGPU, innerLowerModuleIndex,nInnerMDs);

}
#endif

#ifndef NESTED_PARA
#ifdef NEWGRID_Tracklet
__global__ void createTrackletsInGPU(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::tracklets& trackletsInGPU, unsigned int *index_gpu)
{
  //int innerInnerLowerModuleArrayIndex = blockIdx.z * blockDim.z + threadIdx.z;
  int innerInnerLowerModuleArrayIndex = index_gpu[blockIdx.z * blockDim.z + threadIdx.z];
  if(innerInnerLowerModuleArrayIndex >= *modulesInGPU.nLowerModules) return;
  unsigned int innerInnerLowerModuleIndex = modulesInGPU.lowerModuleIndices[innerInnerLowerModuleArrayIndex];
  unsigned int nInnerSegments = segmentsInGPU.nSegments[innerInnerLowerModuleIndex] > N_MAX_SEGMENTS_PER_MODULE ? N_MAX_SEGMENTS_PER_MODULE : segmentsInGPU.nSegments[innerInnerLowerModuleIndex];

  if(nInnerSegments == 0) return;

  int outerInnerLowerModuleArrayIndex = blockIdx.y * blockDim.y + threadIdx.y;
  int innerSegmentArrayIndex = (blockIdx.x * blockDim.x + threadIdx.x) % nInnerSegments;
  int outerSegmentArrayIndex = (blockIdx.x * blockDim.x + threadIdx.x) / nInnerSegments;

  if(innerSegmentArrayIndex >= nInnerSegments) return;

  //outer inner lower module array indices should be obtained from the partner module of the inner segment's outer lower module
  unsigned int innerSegmentIndex = innerInnerLowerModuleIndex * N_MAX_SEGMENTS_PER_MODULE + innerSegmentArrayIndex;

  unsigned int innerOuterLowerModuleIndex = segmentsInGPU.outerLowerModuleIndices[innerSegmentIndex];

  //number of possible outer segment inner MD lower modules
  unsigned int nOuterInnerLowerModules = modulesInGPU.nConnectedModules[innerOuterLowerModuleIndex];
  if(outerInnerLowerModuleArrayIndex >= nOuterInnerLowerModules) return;

  unsigned int outerInnerLowerModuleIndex = modulesInGPU.moduleMap[innerOuterLowerModuleIndex * MAX_CONNECTED_MODULES + outerInnerLowerModuleArrayIndex];

  unsigned int nOuterSegments = segmentsInGPU.nSegments[outerInnerLowerModuleIndex] > N_MAX_SEGMENTS_PER_MODULE ? N_MAX_SEGMENTS_PER_MODULE : segmentsInGPU.nSegments[outerInnerLowerModuleIndex];
  if(outerSegmentArrayIndex >= nOuterSegments) return;

  unsigned int outerSegmentIndex = outerInnerLowerModuleIndex * N_MAX_SEGMENTS_PER_MODULE + outerSegmentArrayIndex;

  //for completeness - outerOuterLowerModuleIndex
  unsigned int outerOuterLowerModuleIndex = segmentsInGPU.outerLowerModuleIndices[outerSegmentIndex];

  //with both segment indices obtained, run the tracklet algorithm
  float zOut,rtOut,deltaPhiPos,deltaPhi,betaIn,betaOut;

#ifdef CUT_VALUE_DEBUG
    float zLo, zHi, rtLo, rtHi, zLoPointed, zHiPointed, sdlCut, betaInCut, betaOutCut, deltaBetaCut, kZ;
    bool success = runTrackletDefaultAlgo(modulesInGPU, hitsInGPU, mdsInGPU, segmentsInGPU, innerInnerLowerModuleIndex, innerOuterLowerModuleIndex, outerInnerLowerModuleIndex, outerOuterLowerModuleIndex, innerSegmentIndex, outerSegmentIndex, zOut, rtOut, deltaPhiPos, deltaPhi, betaIn, betaOut, zLo, zHi, rtLo, rtHi, zLoPointed, zHiPointed, sdlCut, betaInCut, betaOutCut, deltaBetaCut, kZ, N_MAX_SEGMENTS_PER_MODULE); //might want to send the other two module indices and the anchor hits also to save memory accesses

#else

  bool success = runTrackletDefaultAlgo(modulesInGPU, hitsInGPU, mdsInGPU, segmentsInGPU, innerInnerLowerModuleIndex, innerOuterLowerModuleIndex, outerInnerLowerModuleIndex, outerOuterLowerModuleIndex, innerSegmentIndex, outerSegmentIndex, zOut, rtOut, deltaPhiPos, deltaPhi, betaIn, betaOut); //might want to send the other two module indices and the anchor hits also to save memory accesses
#endif

  if(success)
    {
      unsigned int trackletModuleIndex = atomicAdd(&trackletsInGPU.nTracklets[innerInnerLowerModuleArrayIndex],1);
      if(trackletModuleIndex >= N_MAX_TRACKLETS_PER_MODULE)
      {
          #ifdef Warnings
          if(trackletModuleIndex == N_MAX_TRACKLETS_PER_MODULE)
              printf("Tracklet excess alert! Module index = %d\n",innerInnerLowerModuleIndex);
          #endif
      }
      else
      {
          unsigned int trackletIndex = innerInnerLowerModuleArrayIndex * N_MAX_TRACKLETS_PER_MODULE + trackletModuleIndex;
#ifdef CUT_VALUE_DEBUG
          addTrackletToMemory(trackletsInGPU,innerSegmentIndex,outerSegmentIndex,innerInnerLowerModuleIndex,innerOuterLowerModuleIndex,outerInnerLowerModuleIndex,outerOuterLowerModuleIndex,zOut,rtOut,deltaPhiPos,deltaPhi,betaIn,betaOut,zLo, zHi, rtLo, rtHi, zLoPointed, zHiPointed, sdlCut, betaInCut, betaOutCut, deltaBetaCut, kZ, trackletIndex);

#else
          addTrackletToMemory(trackletsInGPU,innerSegmentIndex,outerSegmentIndex,innerInnerLowerModuleIndex,innerOuterLowerModuleIndex,outerInnerLowerModuleIndex,outerOuterLowerModuleIndex,zOut,rtOut,deltaPhiPos,deltaPhi,betaIn,betaOut,trackletIndex);

#endif

      }
    }
}
#endif
#else
__global__ void createTrackletsFromInnerInnerLowerModule(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::tracklets& trackletsInGPU, unsigned int innerInnerLowerModuleIndex, unsigned int nInnerSegments, unsigned int innerInnerLowerModuleArrayIndex)
{
    int outerInnerLowerModuleArrayIndex = blockIdx.x * blockDim.x + threadIdx.x;
    int innerSegmentArrayIndex = blockIdx.y * blockDim.y + threadIdx.y;
    int outerSegmentArrayIndex = blockIdx.z * blockDim.z + threadIdx.z;

    if(innerSegmentArrayIndex >= nInnerSegments) return;
        //outer inner lower module array indices should be obtained from the partner module of the inner segment's outer lower module
    unsigned int innerSegmentIndex = innerInnerLowerModuleIndex * N_MAX_SEGMENTS_PER_MODULE + innerSegmentArrayIndex;


    unsigned int innerOuterLowerModuleIndex = segmentsInGPU.outerLowerModuleIndices[innerSegmentIndex];

    //number of possible outer segment inner MD lower modules
    unsigned int nOuterInnerLowerModules = modulesInGPU.nConnectedModules[innerOuterLowerModuleIndex];
    if(outerInnerLowerModuleArrayIndex >= nOuterInnerLowerModules) return;

    unsigned int outerInnerLowerModuleIndex = modulesInGPU.moduleMap[innerOuterLowerModuleIndex * MAX_CONNECTED_MODULES + outerInnerLowerModuleArrayIndex];

    unsigned int nOuterSegments = segmentsInGPU.nSegments[outerInnerLowerModuleIndex] > N_MAX_SEGMENTS_PER_MODULE ? N_MAX_SEGMENTS_PER_MODULE : segmentsInGPU.nSegments[outerInnerLowerModuleIndex];
    if(outerSegmentArrayIndex >= nOuterSegments) return;

    unsigned int outerSegmentIndex = outerInnerLowerModuleIndex * N_MAX_SEGMENTS_PER_MODULE + outerSegmentArrayIndex;

    //for completeness - outerOuterLowerModuleIndex
    unsigned int outerOuterLowerModuleIndex = segmentsInGPU.outerLowerModuleIndices[outerSegmentIndex];

    //with both segment indices obtained, run the tracklet algorithm

   float zOut,rtOut,deltaPhiPos,deltaPhi,betaIn,betaOut;

#ifdef CUT_VALUE_DEBUG
    float zLo, zHi, rtLo, rtHi, zLoPointed, zHiPointed, sdlCut, betaInCut, betaOutCut, deltaBetaCut, kZ;
    bool success = runTrackletDefaultAlgo(modulesInGPU, hitsInGPU, mdsInGPU, segmentsInGPU, innerInnerLowerModuleIndex, innerOuterLowerModuleIndex, outerInnerLowerModuleIndex, outerOuterLowerModuleIndex, innerSegmentIndex, outerSegmentIndex, zOut, rtOut, deltaPhiPos, deltaPhi, betaIn, betaOut, zLo, zHi, rtLo, rtHi, zLoPointed, zHiPointed, sdlCut, betaInCut, betaOutCut, deltaBetaCut, kZ, N_MAX_SEGMENTS_PER_MODULE); //might want to send the other two module indices and the anchor hits also to save memory accesses

#else

  bool success = runTrackletDefaultAlgo(modulesInGPU, hitsInGPU, mdsInGPU, segmentsInGPU, innerInnerLowerModuleIndex, innerOuterLowerModuleIndex, outerInnerLowerModuleIndex, outerOuterLowerModuleIndex, innerSegmentIndex, outerSegmentIndex, zOut, rtOut, deltaPhiPos, deltaPhi, betaIn, betaOut); //might want to send the other two module indices and the anchor hits also to save memory accesses
#endif

   if(success)
   {
        unsigned int trackletModuleIndex = atomicAdd(&trackletsInGPU.nTracklets[innerInnerLowerModuleArrayIndex],1);
        if(trackletModuleIndex >= N_MAX_TRACKLETS_PER_MODULE)
        {
            #ifdef Warnings
            if(trackletModuleIndex == N_MAX_TRACKLETS_PER_MODULE)
                printf("Tracklet excess alert! Module index = %d\n",innerInnerLowerModuleIndex);
            #endif
        }
        else
        {
            unsigned int trackletIndex = innerInnerLowerModuleArrayIndex * N_MAX_TRACKLETS_PER_MODULE + trackletModuleIndex;
#ifdef CUT_VALUE_DEBUG
            addTrackletToMemory(trackletsInGPU,innerSegmentIndex,outerSegmentIndex,innerInnerLowerModuleIndex,innerOuterLowerModuleIndex,outerInnerLowerModuleIndex,outerOuterLowerModuleIndex,zOut,rtOut,deltaPhiPos,deltaPhi,betaIn,betaOut,zLo, zHi, rtLo, rtHi, zLoPointed, zHiPointed, sdlCut, betaInCut, betaOutCut, deltaBetaCut, kZ, trackletIndex);

#else
            addTrackletToMemory(trackletsInGPU,innerSegmentIndex,outerSegmentIndex,innerInnerLowerModuleIndex,innerOuterLowerModuleIndex,outerInnerLowerModuleIndex,outerOuterLowerModuleIndex,zOut,rtOut,deltaPhiPos,deltaPhi,betaIn,betaOut,trackletIndex);

#endif
        }
   }



}




__global__ void createTrackletsInGPU(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::tracklets& trackletsInGPU)
{
  int innerInnerLowerModuleArrayIndex = blockIdx.x * blockDim.x + threadIdx.x;
  if(innerInnerLowerModuleArrayIndex >= *modulesInGPU.nLowerModules) return;
  unsigned int innerInnerLowerModuleIndex = modulesInGPU.lowerModuleIndices[innerInnerLowerModuleArrayIndex];
  unsigned int nInnerSegments = segmentsInGPU.nSegments[innerInnerLowerModuleIndex] > N_MAX_SEGMENTS_PER_MODULE ? N_MAX_SEGMENTS_PER_MODULE : segmentsInGPU.nSegments[innerInnerLowerModuleIndex];
  if(nInnerSegments == 0) return;

  dim3 nThreads(1,16,16);
  dim3 nBlocks(MAX_CONNECTED_MODULES % nThreads.x  == 0 ? MAX_CONNECTED_MODULES / nThreads.x : MAX_CONNECTED_MODULES / nThreads.x + 1 ,nInnerSegments % nThreads.y == 0 ? nInnerSegments/nThreads.y : nInnerSegments/nThreads.y + 1,N_MAX_SEGMENTS_PER_MODULE % nThreads.z == 0 ? N_MAX_SEGMENTS_PER_MODULE/nThreads.z : N_MAX_SEGMENTS_PER_MODULE/nThreads.z + 1);

  createTrackletsFromInnerInnerLowerModule<<<nBlocks,nThreads>>>(modulesInGPU,hitsInGPU,mdsInGPU,segmentsInGPU,trackletsInGPU,innerInnerLowerModuleIndex,nInnerSegments,innerInnerLowerModuleArrayIndex);

}
#endif


#ifdef NEWGRID_Tracklet
__global__ void createTrackletsFromTriplets(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::triplets& tripletsInGPU, struct SDL::tracklets& trackletsInGPU,unsigned int *threadIdx_gpu, unsigned int *threadIdx_gpu_offset)
{

  int innerInnerLowerModuleArrayIndex = threadIdx_gpu[blockIdx.y * blockDim.y + threadIdx.y];
  if(innerInnerLowerModuleArrayIndex >= *modulesInGPU.nLowerModules) return;
  unsigned int nTriplets = tripletsInGPU.nTriplets[innerInnerLowerModuleArrayIndex] > N_MAX_TRIPLETS_PER_MODULE ? N_MAX_TRIPLETS_PER_MODULE : tripletsInGPU.nTriplets[innerInnerLowerModuleArrayIndex];

  if(nTriplets == 0) return;
  int innerTripletArrayIndex = threadIdx_gpu_offset[blockIdx.y * blockDim.y + threadIdx.y];
  int outerTripletArrayIndex = (blockIdx.x * blockDim.x + threadIdx.x);

//////////////////////////////////////////////////////////
  if(innerTripletArrayIndex >= nTriplets) return;

  //outer inner lower module array indices should be obtained from the partner module of the inner Triplet's outer lower module
  unsigned int innerTripletIndex = innerInnerLowerModuleArrayIndex * N_MAX_TRIPLETS_PER_MODULE + innerTripletArrayIndex;
  unsigned int outerInnerInnerLowerModuleIndex = modulesInGPU.reverseLookupLowerModuleIndices[tripletsInGPU.lowerModuleIndices[3 * innerTripletIndex + 1]];//same as innerOuterInnerLowerModuleIndex
        if(outerTripletArrayIndex < fminf(tripletsInGPU.nTriplets[outerInnerInnerLowerModuleIndex],N_MAX_TRIPLETS_PER_MODULE))
        {
            unsigned int outerTripletIndex = outerInnerInnerLowerModuleIndex * N_MAX_TRIPLETS_PER_MODULE + outerTripletArrayIndex;
            unsigned int innerOuterSegmentIndex = tripletsInGPU.segmentIndices[2 * innerTripletIndex + 1];
            unsigned int outerInnerSegmentIndex = tripletsInGPU.segmentIndices[2 * outerTripletIndex];

            if(innerOuterSegmentIndex == outerInnerSegmentIndex)
            {
              unsigned int innerSegmentIndex = tripletsInGPU.segmentIndices[2 * innerTripletIndex];
              unsigned int outerSegmentIndex = tripletsInGPU.segmentIndices[2 * outerTripletIndex + 1];
              unsigned int innerOuterLowerModuleIndex = segmentsInGPU.outerLowerModuleIndices[innerSegmentIndex];
              unsigned int outerInnerLowerModuleIndex = segmentsInGPU.innerLowerModuleIndices[outerSegmentIndex];
              unsigned int outerOuterLowerModuleIndex = segmentsInGPU.outerLowerModuleIndices[outerSegmentIndex];
              float zOut,rtOut,deltaPhiPos,deltaPhi,betaIn,betaOut;
              unsigned int innerInnerLowerModuleIndex = modulesInGPU.lowerModuleIndices[innerInnerLowerModuleArrayIndex];
#ifdef CUT_VALUE_DEBUG
              float zLo, zHi, rtLo, rtHi, zLoPointed, zHiPointed, sdlCut, betaInCut, betaOutCut, deltaBetaCut, kZ;
              bool success = runTrackletDefaultAlgo(modulesInGPU, hitsInGPU, mdsInGPU, segmentsInGPU, innerInnerLowerModuleIndex, innerOuterLowerModuleIndex, outerInnerLowerModuleIndex, outerOuterLowerModuleIndex, innerSegmentIndex, outerSegmentIndex, zOut, rtOut, deltaPhiPos, deltaPhi, betaIn, betaOut, zLo, zHi, rtLo, rtHi, zLoPointed, zHiPointed, sdlCut, betaInCut, betaOutCut, deltaBetaCut, kZ, N_MAX_SEGMENTS_PER_MODULE); //might want to send the other two module indices and the anchor hits also to save memory accesses

#else
              bool success = runTrackletDefaultAlgo(modulesInGPU, hitsInGPU, mdsInGPU, segmentsInGPU, innerInnerLowerModuleIndex, innerOuterLowerModuleIndex, outerInnerLowerModuleIndex, outerOuterLowerModuleIndex, innerSegmentIndex, outerSegmentIndex, zOut, rtOut, deltaPhiPos, deltaPhi, betaIn, betaOut); //might want to send the other two module indices and the anchor hits also to save memory accesses
#endif
              if(success)
              {
                   unsigned int trackletModuleIndex = atomicAdd(&trackletsInGPU.nTracklets[innerInnerLowerModuleArrayIndex],1);
                   if(trackletModuleIndex >= N_MAX_TRACKLETS_PER_MODULE)
                   {
                       #ifdef Warnings
                       if(trackletModuleIndex == N_MAX_TRACKLETS_PER_MODULE)
                           printf("Tracklet excess alert! Module index = %d\n",innerInnerLowerModuleIndex);
                       #endif
                   }
                   else
                   {
                       unsigned int trackletIndex = innerInnerLowerModuleArrayIndex * N_MAX_TRACKLETS_PER_MODULE + trackletModuleIndex;
                        #ifdef CUT_VALUE_DEBUG
                       addTrackletToMemory(trackletsInGPU,innerSegmentIndex,outerSegmentIndex,innerInnerLowerModuleIndex,innerOuterLowerModuleIndex,outerInnerLowerModuleIndex,outerOuterLowerModuleIndex,zOut,rtOut,deltaPhiPos,deltaPhi,betaIn,betaOut,zLo, zHi, rtLo, rtHi, zLoPointed, zHiPointed, sdlCut, betaInCut, betaOutCut, deltaBetaCut, kZ, trackletIndex);
                        #else
                       addTrackletToMemory(trackletsInGPU,innerSegmentIndex,outerSegmentIndex,innerInnerLowerModuleIndex,innerOuterLowerModuleIndex,outerInnerLowerModuleIndex,outerOuterLowerModuleIndex,zOut,rtOut,deltaPhiPos,deltaPhi,betaIn,betaOut,trackletIndex);
                        #endif
                   }
              }
            }
        }
}
#else
__global__ void createTrackletsFromTriplets(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::triplets& tripletsInGPU, struct SDL::tracklets& trackletsInGPU/*,unsigned int *index_gpu*/)
{
  int innerInnerLowerModuleArrayIndex = blockIdx.x * blockDim.x + threadIdx.x;
  if(innerInnerLowerModuleArrayIndex >= *modulesInGPU.nLowerModules) return;
  //unsigned int innerInnerLowerModuleIndex = modulesInGPU.lowerModuleIndices[innerInnerLowerModuleArrayIndex];
  unsigned int nTriplets = tripletsInGPU.nTriplets[innerInnerLowerModuleArrayIndex] > N_MAX_TRIPLETS_PER_MODULE ? N_MAX_TRIPLETS_PER_MODULE : tripletsInGPU.nTriplets[innerInnerLowerModuleArrayIndex];

  if(nTriplets == 0) return;
    dim3 nThreads(16,16,1);
    dim3 nBlocks(nTriplets % nThreads.x == 0 ? nTriplets / nThreads.x : nTriplets / nThreads.x + 1, N_MAX_TRIPLETS_PER_MODULE % nThreads.y == 0 ? N_MAX_TRIPLETS_PER_MODULE / nThreads.y : N_MAX_TRIPLETS_PER_MODULE / nThreads.y + 1, 1);
    createTrackletsFromTripletsP2<<<nBlocks,nThreads>>>(modulesInGPU,hitsInGPU,mdsInGPU,segmentsInGPU,tripletsInGPU,trackletsInGPU,innerInnerLowerModuleArrayIndex,nTriplets);

}
__global__ void createTrackletsFromTripletsP2(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::triplets& tripletsInGPU, struct SDL::tracklets& trackletsInGPU/*,unsigned int *index_gpu*/,unsigned int innerInnerLowerModuleArrayIndex, unsigned int nTriplets)
{
  int innerTripletArrayIndex = (blockIdx.x * blockDim.x + threadIdx.x);// % nTriplets;
  int outerTripletArrayIndex = (blockIdx.y * blockDim.y + threadIdx.y);// / nTriplets;
  if(innerTripletArrayIndex >= nTriplets) return;

  //outer inner lower module array indices should be obtained from the partner module of the inner Triplet's outer lower module
  unsigned int innerTripletIndex = innerInnerLowerModuleArrayIndex * N_MAX_TRIPLETS_PER_MODULE + innerTripletArrayIndex;
  unsigned int outerInnerInnerLowerModuleIndex = modulesInGPU.reverseLookupLowerModuleIndices[tripletsInGPU.lowerModuleIndices[3 * innerTripletIndex + 1]];//same as innerOuterInnerLowerModuleIndex
        if(outerTripletArrayIndex < fminf(tripletsInGPU.nTriplets[outerInnerInnerLowerModuleIndex],N_MAX_TRIPLETS_PER_MODULE))
        {
            unsigned int outerTripletIndex = outerInnerInnerLowerModuleIndex * N_MAX_TRIPLETS_PER_MODULE + outerTripletArrayIndex;
            unsigned int innerOuterSegmentIndex = tripletsInGPU.segmentIndices[2 * innerTripletIndex + 1];
            unsigned int outerInnerSegmentIndex = tripletsInGPU.segmentIndices[2 * outerTripletIndex];

            if(innerOuterSegmentIndex == outerInnerSegmentIndex)
            {
              unsigned int innerSegmentIndex = tripletsInGPU.segmentIndices[2 * innerTripletIndex];
              unsigned int outerSegmentIndex = tripletsInGPU.segmentIndices[2 * outerTripletIndex + 1];
              unsigned int innerOuterLowerModuleIndex = segmentsInGPU.outerLowerModuleIndices[innerSegmentIndex];
              unsigned int outerInnerLowerModuleIndex = segmentsInGPU.innerLowerModuleIndices[outerSegmentIndex];
              unsigned int outerOuterLowerModuleIndex = segmentsInGPU.outerLowerModuleIndices[outerSegmentIndex];
              float zOut,rtOut,deltaPhiPos,deltaPhi,betaIn,betaOut;
              unsigned int innerInnerLowerModuleIndex = modulesInGPU.lowerModuleIndices[innerInnerLowerModuleArrayIndex];
#ifdef CUT_VALUE_DEBUG
              float zLo, zHi, rtLo, rtHi, zLoPointed, zHiPointed, sdlCut, betaInCut, betaOutCut, deltaBetaCut, kZ;
              bool success = runTrackletDefaultAlgo(modulesInGPU, hitsInGPU, mdsInGPU, segmentsInGPU, innerInnerLowerModuleIndex, innerOuterLowerModuleIndex, outerInnerLowerModuleIndex, outerOuterLowerModuleIndex, innerSegmentIndex, outerSegmentIndex, zOut, rtOut, deltaPhiPos, deltaPhi, betaIn, betaOut, zLo, zHi, rtLo, rtHi, zLoPointed, zHiPointed, sdlCut, betaInCut, betaOutCut, deltaBetaCut, kZ, N_MAX_SEGMENTS_PER_MODULE); //might want to send the other two module indices and the anchor hits also to save memory accesses

#else
              bool success = runTrackletDefaultAlgo(modulesInGPU, hitsInGPU, mdsInGPU, segmentsInGPU, innerInnerLowerModuleIndex, innerOuterLowerModuleIndex, outerInnerLowerModuleIndex, outerOuterLowerModuleIndex, innerSegmentIndex, outerSegmentIndex, zOut, rtOut, deltaPhiPos, deltaPhi, betaIn, betaOut); //might want to send the other two module indices and the anchor hits also to save memory accesses
#endif
              if(success)
              {
                   unsigned int trackletModuleIndex = atomicAdd(&trackletsInGPU.nTracklets[innerInnerLowerModuleArrayIndex],1);
                   if(trackletModuleIndex >= N_MAX_TRACKLETS_PER_MODULE)
                   {
                       #ifdef Warnings
                       if(trackletModuleIndex == N_MAX_TRACKLETS_PER_MODULE)
                           printf("Tracklet excess alert! Module index = %d\n",innerInnerLowerModuleIndex);
                       #endif
                   }
                   else
                   {
                       unsigned int trackletIndex = innerInnerLowerModuleArrayIndex * N_MAX_TRACKLETS_PER_MODULE + trackletModuleIndex;
                        #ifdef CUT_VALUE_DEBUG
                       addTrackletToMemory(trackletsInGPU,innerSegmentIndex,outerSegmentIndex,innerInnerLowerModuleIndex,innerOuterLowerModuleIndex,outerInnerLowerModuleIndex,outerOuterLowerModuleIndex,zOut,rtOut,deltaPhiPos,deltaPhi,betaIn,betaOut,zLo, zHi, rtLo, rtHi, zLoPointed, zHiPointed, sdlCut, betaInCut, betaOutCut, deltaBetaCut, kZ, trackletIndex);
                        #else
                       addTrackletToMemory(trackletsInGPU,innerSegmentIndex,outerSegmentIndex,innerInnerLowerModuleIndex,innerOuterLowerModuleIndex,outerInnerLowerModuleIndex,outerOuterLowerModuleIndex,zOut,rtOut,deltaPhiPos,deltaPhi,betaIn,betaOut,trackletIndex);
                        #endif
                   }
              }
            }
        }
}
#endif
#ifndef NESTED_PARA
__global__ void createPixelTrackletsInGPU(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::tracklets& trackletsInGPU, unsigned int* threadIdx_gpu, unsigned int *threadIdx_gpu_offset)
{
  int outerInnerLowerModuleArrayIndex = threadIdx_gpu[blockIdx.y * blockDim.y + threadIdx.y];
  if(outerInnerLowerModuleArrayIndex >= *modulesInGPU.nLowerModules) return;

  unsigned int outerInnerLowerModuleIndex = modulesInGPU.lowerModuleIndices[outerInnerLowerModuleArrayIndex];
  unsigned int pixelModuleIndex = *modulesInGPU.nModules - 1; //last dude
  unsigned int pixelLowerModuleArrayIndex = modulesInGPU.reverseLookupLowerModuleIndices[pixelModuleIndex]; //should be the same as nLowerModules
  unsigned int nInnerSegments = segmentsInGPU.nSegments[pixelModuleIndex] > N_MAX_PIXEL_SEGMENTS_PER_MODULE ? N_MAX_PIXEL_SEGMENTS_PER_MODULE : segmentsInGPU.nSegments[pixelModuleIndex];
  unsigned int nOuterSegments = segmentsInGPU.nSegments[outerInnerLowerModuleIndex] > N_MAX_SEGMENTS_PER_MODULE ? N_MAX_SEGMENTS_PER_MODULE : segmentsInGPU.nSegments[outerInnerLowerModuleIndex];
  if(nOuterSegments == 0) return;
  if(nInnerSegments == 0) return;
  if(modulesInGPU.moduleType[outerInnerLowerModuleIndex] == SDL::TwoS) return; //REMOVES 2S-2S

  int innerSegmentArrayIndex = blockIdx.x * blockDim.x + threadIdx.x;
  int outerSegmentArrayIndex = threadIdx_gpu_offset[blockIdx.y * blockDim.y + threadIdx.y];
  if(innerSegmentArrayIndex >= nInnerSegments) return;
  if(outerSegmentArrayIndex >= nOuterSegments) return;
  unsigned int innerSegmentIndex = pixelModuleIndex * N_MAX_SEGMENTS_PER_MODULE + innerSegmentArrayIndex;
  unsigned int outerSegmentIndex = outerInnerLowerModuleIndex * N_MAX_SEGMENTS_PER_MODULE + outerSegmentArrayIndex;
  unsigned int outerOuterLowerModuleIndex = segmentsInGPU.outerLowerModuleIndices[outerSegmentIndex];
  if(modulesInGPU.moduleType[outerOuterLowerModuleIndex] == SDL::TwoS) return; //REMOVES PS-2S
  float zOut, rtOut, deltaPhiPos, deltaPhi, betaIn, betaOut;

#ifdef CUT_VALUE_DEBUG
  float zLo, zHi, rtLo, rtHi, zLoPointed, zHiPointed, sdlCut, betaInCut, betaOutCut, deltaBetaCut, kZ;
  bool success = runTrackletDefaultAlgo(modulesInGPU, hitsInGPU, mdsInGPU, segmentsInGPU, pixelModuleIndex, pixelModuleIndex, outerInnerLowerModuleIndex, outerOuterLowerModuleIndex, innerSegmentIndex, outerSegmentIndex, zOut, rtOut, deltaPhiPos, deltaPhi, betaIn, betaOut, zLo, zHi, rtLo, rtHi, zLoPointed, zHiPointed, sdlCut, betaInCut, betaOutCut, deltaBetaCut, kZ, N_MAX_SEGMENTS_PER_MODULE); //might want to send the other two module indices and the anchor hits also to save memory accesses
#else
  bool success = runTrackletDefaultAlgo(modulesInGPU, hitsInGPU, mdsInGPU, segmentsInGPU, pixelModuleIndex, pixelModuleIndex, outerInnerLowerModuleIndex, outerOuterLowerModuleIndex, innerSegmentIndex, outerSegmentIndex, zOut, rtOut, deltaPhiPos, deltaPhi, betaIn, betaOut, N_MAX_SEGMENTS_PER_MODULE); //might want to send the other two module indices and the anchor hits also to save memory accesses
#endif
  if(success)
    {
      unsigned int trackletModuleIndex = atomicAdd(&trackletsInGPU.nTracklets[pixelLowerModuleArrayIndex], 1);
      if(trackletModuleIndex >= N_MAX_PIXEL_TRACKLETS_PER_MODULE)
        {
            #ifdef Warnings
	  if(trackletModuleIndex == N_MAX_PIXEL_TRACKLETS_PER_MODULE)
	    printf("Pixel Tracklet excess alert! Module index = %d\n",pixelModuleIndex);
            #endif
        }
      else
        {
	  unsigned int trackletIndex = pixelLowerModuleArrayIndex * N_MAX_TRACKLETS_PER_MODULE + trackletModuleIndex;
#ifdef CUT_VALUE_DEBUG
	  addTrackletToMemory(trackletsInGPU,innerSegmentIndex,outerSegmentIndex,pixelModuleIndex,pixelModuleIndex,outerInnerLowerModuleIndex,outerOuterLowerModuleIndex,zOut,rtOut,deltaPhiPos,deltaPhi,betaIn,betaOut,zLo, zHi, rtLo, rtHi, zLoPointed, zHiPointed, sdlCut, betaInCut, betaOutCut, deltaBetaCut, kZ, trackletIndex);
#else
	  addTrackletToMemory(trackletsInGPU,innerSegmentIndex,outerSegmentIndex,pixelModuleIndex,pixelModuleIndex,outerInnerLowerModuleIndex,outerOuterLowerModuleIndex,zOut,rtOut,deltaPhiPos,deltaPhi,betaIn,betaOut,trackletIndex);
#endif
        }
    }
}
__global__ void createPixelTrackletsInGPUFromMap(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::tracklets& trackletsInGPU, unsigned int* connectedPixelSize, unsigned int* connectedPixelIndex,unsigned int nInnerSegs,unsigned int* seg_pix_gpu, unsigned int* seg_pix_gpu_offset)
{
  //newgrid with map
  int segmentArrayIndex = seg_pix_gpu_offset[blockIdx.x * blockDim.x + threadIdx.x];// segment loop; this segent
  int pixelArrayIndex = seg_pix_gpu[blockIdx.x * blockDim.x + threadIdx.x];// pixel loop; this pixel
  if(pixelArrayIndex >= nInnerSegs) return;// don't exceed # of pLS
  if( segmentArrayIndex >= connectedPixelSize[pixelArrayIndex]) return; // don't exceed # connected segment modules for this pixel

  unsigned int outerInnerLowerModuleArrayIndex;// This will be the index of the module that connects to this pixel. 
    unsigned int temp = connectedPixelIndex[pixelArrayIndex]+segmentArrayIndex; //gets module index for segment
    //unsigned int temp = connectedPixelIndex[255]+0;//gets module index for segment
    //outerInnerLowerModuleArrayIndex = modulesInGPU.connectedPixels[134378]; //gets module index for segment
    //printf("test %u %u %u %u\n",temp,pixelArrayIndex,segmentArrayIndex,blockIdx.x);
    //if(temp >3735957){printf("kernel %u %d %d %u\n",temp,pixelArrayIndex,segmentArrayIndex,nInnerSegs); return;}
    outerInnerLowerModuleArrayIndex = modulesInGPU.connectedPixels[temp]; //gets module index for segment
    //outerInnerLowerModuleArrayIndex = modulesInGPU.connectedPixels[connectedPixelIndex[pixelArrayIndex]+segmentArrayIndex]; //gets module index for segment
  if(outerInnerLowerModuleArrayIndex >= *modulesInGPU.nLowerModules) return;
  unsigned int outerInnerLowerModuleIndex = /*modulesInGPU.lowerModuleIndices[*/outerInnerLowerModuleArrayIndex;//];

  unsigned int pixelModuleIndex = *modulesInGPU.nModules - 1; //last dude
  unsigned int pixelLowerModuleArrayIndex = modulesInGPU.reverseLookupLowerModuleIndices[pixelModuleIndex]; //should be the same as nLowerModules
  unsigned int nOuterSegments = segmentsInGPU.nSegments[outerInnerLowerModuleIndex] > N_MAX_SEGMENTS_PER_MODULE ? N_MAX_SEGMENTS_PER_MODULE : segmentsInGPU.nSegments[outerInnerLowerModuleIndex];
  if(nOuterSegments == 0) return;
  if(modulesInGPU.moduleType[outerInnerLowerModuleIndex] == SDL::TwoS) return; //REMOVES 2S-2S

//  int outerSegmentArrayIndex = blockIdx.z * blockDim.z + threadIdx.z;
  int outerSegmentArrayIndex = blockIdx.y * blockDim.y + threadIdx.y;
  if(outerSegmentArrayIndex >= nOuterSegments) return;
  unsigned int innerSegmentIndex = pixelModuleIndex * N_MAX_SEGMENTS_PER_MODULE + pixelArrayIndex;
  unsigned int outerSegmentIndex = outerInnerLowerModuleIndex * N_MAX_SEGMENTS_PER_MODULE + outerSegmentArrayIndex;
  unsigned int outerOuterLowerModuleIndex = segmentsInGPU.outerLowerModuleIndices[outerSegmentIndex];
  if(modulesInGPU.moduleType[outerOuterLowerModuleIndex] == SDL::TwoS) return; //REMOVES PS-2S
  float zOut, rtOut, deltaPhiPos, deltaPhi, betaIn, betaOut;
#ifdef CUT_VALUE_DEBUG
  float zLo, zHi, rtLo, rtHi, zLoPointed, zHiPointed, sdlCut, betaInCut, betaOutCut, deltaBetaCut, kZ;
  bool success = runTrackletDefaultAlgo(modulesInGPU, hitsInGPU, mdsInGPU, segmentsInGPU, pixelModuleIndex, pixelModuleIndex, outerInnerLowerModuleIndex, outerOuterLowerModuleIndex, innerSegmentIndex, outerSegmentIndex, zOut, rtOut, deltaPhiPos, deltaPhi, betaIn, betaOut, zLo, zHi, rtLo, rtHi, zLoPointed, zHiPointed, sdlCut, betaInCut, betaOutCut, deltaBetaCut, kZ, N_MAX_SEGMENTS_PER_MODULE); //might want to send the other two module indices and the anchor hits also to save memory accesses
#else
  bool success = runTrackletDefaultAlgo(modulesInGPU, hitsInGPU, mdsInGPU, segmentsInGPU, pixelModuleIndex, pixelModuleIndex, outerInnerLowerModuleIndex, outerOuterLowerModuleIndex, innerSegmentIndex, outerSegmentIndex, zOut, rtOut, deltaPhiPos, deltaPhi, betaIn, betaOut, N_MAX_SEGMENTS_PER_MODULE); //might want to send the other two module indices and the anchor hits also to save memory accesses
#endif
  if(success)
    {
      unsigned int trackletModuleIndex = atomicAdd(&trackletsInGPU.nTracklets[pixelLowerModuleArrayIndex], 1);
      if(trackletModuleIndex >= N_MAX_PIXEL_TRACKLETS_PER_MODULE)
        {
            #ifdef Warnings
	  if(trackletModuleIndex == N_MAX_PIXEL_TRACKLETS_PER_MODULE)
	    printf("Pixel Tracklet excess alert! Module index = %d\n",pixelModuleIndex);
            #endif
        }
      else
        {
	  unsigned int trackletIndex = pixelLowerModuleArrayIndex * N_MAX_TRACKLETS_PER_MODULE + trackletModuleIndex;
#ifdef CUT_VALUE_DEBUG
	  addTrackletToMemory(trackletsInGPU,innerSegmentIndex,outerSegmentIndex,pixelModuleIndex,pixelModuleIndex,outerInnerLowerModuleIndex,outerOuterLowerModuleIndex,zOut,rtOut,deltaPhiPos,deltaPhi,betaIn,betaOut,zLo, zHi, rtLo, rtHi, zLoPointed, zHiPointed, sdlCut, betaInCut, betaOutCut, deltaBetaCut, kZ, trackletIndex);
#else
	  addTrackletToMemory(trackletsInGPU,innerSegmentIndex,outerSegmentIndex,pixelModuleIndex,pixelModuleIndex,outerInnerLowerModuleIndex,outerOuterLowerModuleIndex,zOut,rtOut,deltaPhiPos,deltaPhi,betaIn,betaOut,trackletIndex);
#endif
        }
    }
}

#else
__global__ void createPixelTrackletsInGPU(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::tracklets& trackletsInGPU)
{
    //printf("pixel nested kernel\n");
    int outerInnerLowerModuleArrayIndex = blockIdx.x * blockDim.x + threadIdx.x; // loop for modules for segments lower hit
    if(outerInnerLowerModuleArrayIndex >= *modulesInGPU.nLowerModules) return; // don't exceed number of modules

    unsigned int outerInnerLowerModuleIndex = modulesInGPU.lowerModuleIndices[outerInnerLowerModuleArrayIndex]; // correspond to module number index
    unsigned int pixelModuleIndex = *modulesInGPU.nModules - 1; // pixel module index
    unsigned int pixelLowerModuleArrayIndex = modulesInGPU.reverseLookupLowerModuleIndices[pixelModuleIndex]; //should be the same as nLowerModules 
    unsigned int nInnerSegments = segmentsInGPU.nSegments[pixelModuleIndex] > N_MAX_PIXEL_SEGMENTS_PER_MODULE ? N_MAX_PIXEL_SEGMENTS_PER_MODULE : segmentsInGPU.nSegments[pixelModuleIndex]; // number of pLS
    unsigned int nOuterSegments = segmentsInGPU.nSegments[outerInnerLowerModuleIndex] > N_MAX_SEGMENTS_PER_MODULE ? N_MAX_SEGMENTS_PER_MODULE : segmentsInGPU.nSegments[outerInnerLowerModuleIndex]; // number of segments from module corresponding to each module. 
    if(nOuterSegments == 0) return;
    if(nInnerSegments == 0) return;
    if(modulesInGPU.moduleType[outerInnerLowerModuleIndex] == SDL::TwoS) return; //REMOVES 2S-2S

    dim3 nThreads(16,16,1);
    dim3 nBlocks(nInnerSegments % nThreads.x == 0 ? nInnerSegments / nThreads.x : nInnerSegments / nThreads.x + 1, nOuterSegments % nThreads.y == 0 ? nOuterSegments / nThreads.y : nOuterSegments / nThreads.y + 1, 1);

    createPixelTrackletsFromOuterInnerLowerModule<<<nBlocks,nThreads>>>(modulesInGPU, hitsInGPU, mdsInGPU, segmentsInGPU, trackletsInGPU, outerInnerLowerModuleIndex, nInnerSegments, nOuterSegments, pixelModuleIndex, pixelLowerModuleArrayIndex);

}
__global__ void createPixelTrackletsFromOuterInnerLowerModule(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::tracklets& trackletsInGPU, unsigned int outerInnerLowerModuleIndex, unsigned int nInnerSegments, unsigned int nOuterSegments, unsigned int pixelModuleIndex, unsigned int pixelLowerModuleArrayIndex)
{
    //printf("pixel nested kernel inner\n");
    int innerSegmentArrayIndex = blockIdx.x * blockDim.x + threadIdx.x;// looping over pixels
    int outerSegmentArrayIndex = blockIdx.y * blockDim.y + threadIdx.y;//looping over segments
    if(innerSegmentArrayIndex >= nInnerSegments) return; // not over # of pLS
    if(outerSegmentArrayIndex >= nOuterSegments) return; // not over # of segments for this module
    unsigned int innerSegmentIndex = pixelModuleIndex * N_MAX_SEGMENTS_PER_MODULE + innerSegmentArrayIndex; // get this pixel index Just innerSegmentArrayIndex'th value (1-pLS)
    unsigned int outerSegmentIndex = outerInnerLowerModuleIndex * N_MAX_SEGMENTS_PER_MODULE + outerSegmentArrayIndex; // get this segment Index for this this module
    unsigned int outerOuterLowerModuleIndex = segmentsInGPU.outerLowerModuleIndices[outerSegmentIndex]; // get corresponding outer module index for this segment 
    if(modulesInGPU.moduleType[outerOuterLowerModuleIndex] == SDL::TwoS) return; //REMOVES PS-2S
    float zOut, rtOut, deltaPhiPos, deltaPhi, betaIn, betaOut;
#ifdef CUT_VALUE_DEBUG
    float zLo, zHi, rtLo, rtHi, zLoPointed, zHiPointed, sdlCut, betaInCut, betaOutCut, deltaBetaCut, kZ;
    bool success = runTrackletDefaultAlgo(modulesInGPU, hitsInGPU, mdsInGPU, segmentsInGPU, pixelModuleIndex, pixelModuleIndex, outerInnerLowerModuleIndex, outerOuterLowerModuleIndex, innerSegmentIndex, outerSegmentIndex, zOut, rtOut, deltaPhiPos, deltaPhi, betaIn, betaOut, zLo, zHi, rtLo, rtHi, zLoPointed, zHiPointed, sdlCut, betaInCut, betaOutCut, deltaBetaCut, kZ, N_MAX_SEGMENTS_PER_MODULE); //might want to send the other two module indices and the anchor hits also to save memory accesses
#else
    bool success = runTrackletDefaultAlgo(modulesInGPU, hitsInGPU, mdsInGPU, segmentsInGPU, pixelModuleIndex, pixelModuleIndex, outerInnerLowerModuleIndex, outerOuterLowerModuleIndex, innerSegmentIndex, outerSegmentIndex, zOut, rtOut, deltaPhiPos, deltaPhi, betaIn, betaOut, N_MAX_SEGMENTS_PER_MODULE); //might want to send the other two module indices and the anchor hits also to save memory accesses
#endif

    if(success)
    {
        unsigned int trackletModuleIndex = atomicAdd(&trackletsInGPU.nTracklets[pixelLowerModuleArrayIndex], 1);
        if(trackletModuleIndex >= N_MAX_PIXEL_TRACKLETS_PER_MODULE)
        {
            #ifdef Warnings
            if(trackletModuleIndex == N_MAX_PIXEL_TRACKLETS_PER_MODULE)
                printf("Pixel Tracklet excess alert! Module index = %d\n",pixelModuleIndex);
            #endif
        }
        else
        {
            unsigned int trackletIndex = pixelLowerModuleArrayIndex * N_MAX_TRACKLETS_PER_MODULE + trackletModuleIndex;
#ifdef CUT_VALUE_DEBUG
                addTrackletToMemory(trackletsInGPU,innerSegmentIndex,outerSegmentIndex,pixelModuleIndex,pixelModuleIndex,outerInnerLowerModuleIndex,outerOuterLowerModuleIndex,zOut,rtOut,deltaPhiPos,deltaPhi,betaIn,betaOut,zLo, zHi, rtLo, rtHi, zLoPointed, zHiPointed, sdlCut, betaInCut, betaOutCut, deltaBetaCut, kZ, trackletIndex);

#else
            addTrackletToMemory(trackletsInGPU,innerSegmentIndex,outerSegmentIndex,pixelModuleIndex,pixelModuleIndex,outerInnerLowerModuleIndex,outerOuterLowerModuleIndex,zOut,rtOut,deltaPhiPos,deltaPhi,betaIn,betaOut,trackletIndex);
#endif
        }
    }
}
#endif

__global__ void createTrackletsWithAGapFromInnerInnerLowerModule(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::tracklets& trackletsInGPU, unsigned int innerInnerLowerModuleIndex, unsigned int nInnerSegments, unsigned int innerInnerLowerModuleArrayIndex)
{
    //Proposal 1 : Inner kernel takes care of both loops
    int xAxisIndex = blockIdx.x * blockDim.x + threadIdx.x;
    int innerSegmentArrayIndex =  blockIdx.y * blockDim.y + threadIdx.y;
    int outerSegmentArrayIndex = blockIdx.z * blockDim.z + threadIdx.z;

    if(innerSegmentArrayIndex >= nInnerSegments) return;

    int middleLowerModuleArrayIndex = xAxisIndex / MAX_CONNECTED_MODULES;
    int outerInnerLowerModuleArrayIndex = xAxisIndex % MAX_CONNECTED_MODULES;

    unsigned int innerSegmentIndex = innerInnerLowerModuleIndex * N_MAX_SEGMENTS_PER_MODULE + innerSegmentArrayIndex;
    unsigned int innerOuterLowerModuleIndex = segmentsInGPU.outerLowerModuleIndices[innerSegmentIndex];

    //first check for middle modules
    unsigned int nMiddleLowerModules = modulesInGPU.nConnectedModules[innerOuterLowerModuleIndex];
    if(middleLowerModuleArrayIndex >= nMiddleLowerModules) return;

    unsigned int middleLowerModuleIndex = modulesInGPU.moduleMap[innerOuterLowerModuleIndex * MAX_CONNECTED_MODULES + middleLowerModuleArrayIndex];

    //second check for outerInnerLowerMoules
    unsigned int nOuterInnerLowerModules = modulesInGPU.nConnectedModules[middleLowerModuleIndex];
    if(outerInnerLowerModuleArrayIndex >= nOuterInnerLowerModules) return;

    unsigned int outerInnerLowerModuleIndex = modulesInGPU.moduleMap[middleLowerModuleIndex * MAX_CONNECTED_MODULES + outerInnerLowerModuleArrayIndex];

    unsigned int nOuterSegments = segmentsInGPU.nSegments[outerInnerLowerModuleIndex] > N_MAX_SEGMENTS_PER_MODULE ? N_MAX_SEGMENTS_PER_MODULE : segmentsInGPU.nSegments[outerInnerLowerModuleIndex];
    if(outerSegmentArrayIndex >= nOuterSegments) return;

    unsigned int outerSegmentIndex = outerInnerLowerModuleIndex * N_MAX_SEGMENTS_PER_MODULE + outerSegmentArrayIndex;

    //for completeness - outerOuterLowerModuleIndex
    unsigned int outerOuterLowerModuleIndex = segmentsInGPU.outerLowerModuleIndices[outerSegmentIndex];

    //with both segment indices obtained, run the tracklet algorithm

   float zOut,rtOut,deltaPhiPos,deltaPhi,betaIn,betaOut;
#ifdef CUT_VALUE_DEBUG
    float zLo, zHi, rtLo, rtHi, zLoPointed, zHiPointed, sdlCut, betaInCut, betaOutCut, deltaBetaCut, kZ;
    bool success = runTrackletDefaultAlgo(modulesInGPU, hitsInGPU, mdsInGPU, segmentsInGPU, innerInnerLowerModuleIndex, innerOuterLowerModuleIndex, outerInnerLowerModuleIndex, outerOuterLowerModuleIndex, innerSegmentIndex, outerSegmentIndex, zOut, rtOut, deltaPhiPos, deltaPhi, betaIn, betaOut, zLo, zHi, rtLo, rtHi, zLoPointed, zHiPointed, sdlCut, betaInCut, betaOutCut, deltaBetaCut, kZ, N_MAX_SEGMENTS_PER_MODULE); //might want to send the other two module indices and the anchor hits also to save memory accesses
#else
   bool success = runTrackletDefaultAlgo(modulesInGPU, hitsInGPU, mdsInGPU, segmentsInGPU, innerInnerLowerModuleIndex, innerOuterLowerModuleIndex, outerInnerLowerModuleIndex, outerOuterLowerModuleIndex, innerSegmentIndex, outerSegmentIndex, zOut, rtOut, deltaPhiPos, deltaPhi, betaIn, betaOut); //might want to send the other two module indices and the anchor hits also to save memory accesses
#endif
   if(success)
   {
        unsigned int trackletModuleIndex = atomicAdd(&trackletsInGPU.nTracklets[innerInnerLowerModuleArrayIndex],1);
        if(trackletModuleIndex >= N_MAX_TRACKLETS_PER_MODULE)
        {
            #ifdef Warnings
            if(trackletModuleIndex == N_MAX_TRACKLETS_PER_MODULE)
                 printf("T4x excess alert! Module index = %d\n",innerInnerLowerModuleIndex);
            #endif
        }
        else
        {

            unsigned int trackletIndex = innerInnerLowerModuleArrayIndex * N_MAX_TRACKLETS_PER_MODULE + trackletModuleIndex;
#ifdef CUT_VALUE_DEBUG
            addTrackletToMemory(trackletsInGPU,innerSegmentIndex,outerSegmentIndex,innerInnerLowerModuleIndex,innerOuterLowerModuleIndex,outerInnerLowerModuleIndex,outerOuterLowerModuleIndex,zOut,rtOut,deltaPhiPos,deltaPhi,betaIn,betaOut,zLo, zHi, rtLo, rtHi, zLoPointed, zHiPointed, sdlCut, betaInCut, betaOutCut, deltaBetaCut, kZ, trackletIndex);

#else
            addTrackletToMemory(trackletsInGPU,innerSegmentIndex,outerSegmentIndex,innerInnerLowerModuleIndex,innerOuterLowerModuleIndex,outerInnerLowerModuleIndex,outerOuterLowerModuleIndex,zOut,rtOut,deltaPhiPos,deltaPhi,betaIn,betaOut,trackletIndex);
#endif

        }
   }
}

__global__ void createTrackletsWithAGapInGPU(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::tracklets& trackletsInGPU)
{
    //outer kernel for proposal 1
    int innerInnerLowerModuleArrayIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if(innerInnerLowerModuleArrayIndex >= *modulesInGPU.nLowerModules) return;
    unsigned int innerInnerLowerModuleIndex = modulesInGPU.lowerModuleIndices[innerInnerLowerModuleArrayIndex];
    unsigned int nInnerSegments = segmentsInGPU.nSegments[innerInnerLowerModuleIndex] > N_MAX_SEGMENTS_PER_MODULE ? N_MAX_SEGMENTS_PER_MODULE : segmentsInGPU.nSegments[innerInnerLowerModuleIndex];
    if(nInnerSegments == 0) return;

    dim3 nThreads(1,16,16);
    dim3 nBlocks((MAX_CONNECTED_MODULES * MAX_CONNECTED_MODULES) % nThreads.x  == 0 ? (MAX_CONNECTED_MODULES * MAX_CONNECTED_MODULES) / nThreads.x : (MAX_CONNECTED_MODULES * MAX_CONNECTED_MODULES) / nThreads.x + 1 ,nInnerSegments % nThreads.y == 0 ? nInnerSegments/nThreads.y : nInnerSegments/nThreads.y + 1,N_MAX_SEGMENTS_PER_MODULE % nThreads.z == 0 ? N_MAX_SEGMENTS_PER_MODULE/nThreads.z : N_MAX_SEGMENTS_PER_MODULE/nThreads.z + 1);

    createTrackletsWithAGapFromInnerInnerLowerModule<<<nBlocks,nThreads>>>(modulesInGPU,hitsInGPU,mdsInGPU,segmentsInGPU,trackletsInGPU,innerInnerLowerModuleIndex,nInnerSegments,innerInnerLowerModuleArrayIndex);

}

/*__global__ void createTrackletsWithAGapFromMiddleLowerModule(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::tracklets& trackletsInGPU, unsigned int innerInnerLowerModuleIndex, unsigned int middleLowerModuleIndex, unsigned int innerSegmentIndex, unsigned int innerInnerLowerModuleArrayIndex, unsigned int nOuterInnerLowerModules,unsigned int innerOuterLowerModuleIndex)
{
    //Inner kernel of Proposal 2 : Inner kernel does middle->outer modoule mapping
    int outerInnerLowerModuleArrayIndex = blockIdx.x * blockDim.x + threadIdx.x;
    int outerSegmentArrayIndex = blockIdx.y * blockDim.y + threadIdx.y;

    //check for outerInnerLowerModules
    if(outerInnerLowerModuleArrayIndex >= nOuterInnerLowerModules) return;


    unsigned int outerInnerLowerModuleIndex = modulesInGPU.moduleMap[middleLowerModuleIndex * MAX_CONNECTED_MODULES + outerInnerLowerModuleArrayIndex];

    unsigned int nOuterSegments = segmentsInGPU.nSegments[outerInnerLowerModuleIndex];
    if(outerSegmentArrayIndex >= nOuterSegments) return;

    unsigned int outerSegmentIndex = outerInnerLowerModuleIndex * N_MAX_SEGMENTS_PER_MODULE + outerSegmentArrayIndex;

    //for completeness - outerOuterLowerModuleIndex and innerOuterLowerModuleIndex
    unsigned int outerOuterLowerModuleIndex = segmentsInGPU.outerLowerModuleIndices[outerSegmentIndex];

    //with both segment indices obtained, run the tracklet algorithm

   float zOut,rtOut,deltaPhiPos,deltaPhi,betaIn,betaOut;

   bool success = runTrackletDefaultAlgo(modulesInGPU, hitsInGPU, mdsInGPU, segmentsInGPU, innerInnerLowerModuleIndex, innerOuterLowerModuleIndex, outerInnerLowerModuleIndex, outerOuterLowerModuleIndex, innerSegmentIndex, outerSegmentIndex, zOut, rtOut, deltaPhiPos, deltaPhi, betaIn, betaOut); //might want to send the other two module indices and the anchor hits also to save memory accesses
   if(success)
   {
        unsigned int trackletModuleIndex = atomicAdd(&trackletsInGPU.nTracklets[innerInnerLowerModuleArrayIndex],1);
        unsigned int trackletIndex = innerInnerLowerModuleArrayIndex * N_MAX_TRACKLETS_PER_MODULE + trackletModuleIndex;

        addTrackletToMemory(trackletsInGPU,innerSegmentIndex,outerSegmentIndex,innerInnerLowerModuleIndex,innerOuterLowerModuleIndex,outerInnerLowerModuleIndex,outerOuterLowerModuleIndex,zOut,rtOut,deltaPhiPos,deltaPhi,betaIn,betaOut,trackletIndex);
   }
}

__global__ void createTrackletsWithAGapFromInnerInnerLowerModule(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::tracklets& trackletsInGPU, unsigned int innerInnerLowerModuleIndex, unsigned int nInnerSegments, unsigned int innerInnerLowerModuleArrayIndex)
{
    //Middle kernel of Proposal 2 : middle kernel does the inner->middle module mapping

    int middleLowerModuleArrayIndex = blockIdx.x * blockDim.x + threadIdx.x;
    int innerSegmentArrayIndex = blockIdx.y * blockDim.y + threadIdx.y;
    if(innerSegmentArrayIndex >= nInnerSegments) return;

    unsigned int innerSegmentIndex = innerInnerLowerModuleIndex * N_MAX_SEGMENTS_PER_MODULE + innerSegmentArrayIndex;

    //middle lower module - modules that are connected to outer lower module of inner segment
    unsigned int innerOuterLowerModuleIndex = segmentsInGPU.outerLowerModuleIndices[innerSegmentIndex];
    unsigned int nMiddleLowerModules = modulesInGPU.nConnectedModules[innerOuterLowerModuleIndex];

    if(middleLowerModuleArrayIndex >= nMiddleLowerModules) return;

    unsigned int middleLowerModuleIndex = modulesInGPU.moduleMap[innerOuterLowerModuleIndex * MAX_CONNECTED_MODULES + middleLowerModuleArrayIndex];

    unsigned int nOuterInnerLowerModules = modulesInGPU.nConnectedModules[middleLowerModuleIndex];

    dim3 nThreads(1,32,1);
    dim3 nBlocks(nOuterInnerLowerModules % nThreads.x  == 0 ? nOuterInnerLowerModules / nThreads.x : nOuterInnerLowerModules / nThreads.x + 1 ,N_MAX_SEGMENTS_PER_MODULE % nThreads.y == 0 ? N_MAX_SEGMENTS_PER_MODULE/nThreads.y : N_MAX_SEGMENTS_PER_MODULE/nThreads.y + 1,1);

    createTrackletsWithAGapFromMiddleLowerModule<<<nBlocks,nThreads>>>(modulesInGPU,hitsInGPU,mdsInGPU,segmentsInGPU,trackletsInGPU,innerInnerLowerModuleIndex, middleLowerModuleIndex,innerSegmentIndex,innerInnerLowerModuleArrayIndex,nOuterInnerLowerModules,innerOuterLowerModuleIndex);


}


__global__ void createTrackletsWithAGapInGPU(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::tracklets& trackletsInGPU)
{
    //outer kernel for proposal 2
    int innerInnerLowerModuleArrayIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if(innerInnerLowerModuleArrayIndex >= *modulesInGPU.nLowerModules) return;
    unsigned int innerInnerLowerModuleIndex = modulesInGPU.lowerModuleIndices[innerInnerLowerModuleArrayIndex];
    unsigned int nInnerSegments = segmentsInGPU.nSegments[innerInnerLowerModuleIndex];
    if(nInnerSegments == 0) return;

    dim3 nThreads(1,1,1);
    dim3 nBlocks(MAX_CONNECTED_MODULES % nThreads.x  == 0 ? MAX_CONNECTED_MODULES / nThreads.x : MAX_CONNECTED_MODULES / nThreads.x + 1 , nInnerSegments % nThreads.y == 0 ? nInnerSegments/nThreads.y : nInnerSegments/nThreads.y + 1,1);

    createTrackletsWithAGapFromInnerInnerLowerModule<<<nBlocks,nThreads>>>(modulesInGPU, hitsInGPU, mdsInGPU, segmentsInGPU, trackletsInGPU, innerInnerLowerModuleIndex, nInnerSegments, innerInnerLowerModuleArrayIndex);

}*/

#ifndef NESTED_PARA
#ifdef NEWGRID_Trips
__global__ void createTripletsInGPU(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::triplets& tripletsInGPU, unsigned int *index_gpu)
{
//int innerInnerLowerModuleArrayIndex = blockIdx.x * blockDim.x + threadIdx.x;
  int innerInnerLowerModuleArrayIndex = index_gpu[blockIdx.z * blockDim.z + threadIdx.z];
  if(innerInnerLowerModuleArrayIndex >= *modulesInGPU.nLowerModules) return;

  unsigned int innerInnerLowerModuleIndex = modulesInGPU.lowerModuleIndices[innerInnerLowerModuleArrayIndex];
  unsigned int nConnectedModules = modulesInGPU.nConnectedModules[innerInnerLowerModuleIndex];
  if(nConnectedModules == 0) return;

  unsigned int nInnerSegments = segmentsInGPU.nSegments[innerInnerLowerModuleIndex] > N_MAX_SEGMENTS_PER_MODULE ? N_MAX_SEGMENTS_PER_MODULE : segmentsInGPU.nSegments[innerInnerLowerModuleIndex];

  int innerSegmentArrayIndex = blockIdx.y * blockDim.y + threadIdx.y;
  int outerSegmentArrayIndex = blockIdx.x * blockDim.x + threadIdx.x;

  if(innerSegmentArrayIndex >= nInnerSegments) return;

  unsigned int innerSegmentIndex = innerInnerLowerModuleIndex * N_MAX_SEGMENTS_PER_MODULE + innerSegmentArrayIndex;

  //middle lower module - outer lower module of inner segment
  unsigned int middleLowerModuleIndex = segmentsInGPU.outerLowerModuleIndices[innerSegmentIndex];

  unsigned int nOuterSegments = segmentsInGPU.nSegments[middleLowerModuleIndex] > N_MAX_SEGMENTS_PER_MODULE ? N_MAX_SEGMENTS_PER_MODULE : segmentsInGPU.nSegments[middleLowerModuleIndex];
  if(outerSegmentArrayIndex >= nOuterSegments) return;

  unsigned int outerSegmentIndex = middleLowerModuleIndex * N_MAX_SEGMENTS_PER_MODULE + outerSegmentArrayIndex;
  unsigned int outerOuterLowerModuleIndex = segmentsInGPU.outerLowerModuleIndices[outerSegmentIndex];

  float zOut,rtOut,deltaPhiPos,deltaPhi,betaIn,betaOut;
#ifdef CUT_VALUE_DEBUG
  float zLo, zHi, rtLo, rtHi, zLoPointed, zHiPointed, sdlCut, betaInCut, betaOutCut, deltaBetaCut, kZ;

    bool success = runTripletDefaultAlgo(modulesInGPU, hitsInGPU, mdsInGPU, segmentsInGPU, innerInnerLowerModuleIndex, middleLowerModuleIndex, outerOuterLowerModuleIndex, innerSegmentIndex, outerSegmentIndex, zOut, rtOut, deltaPhiPos, deltaPhi, betaIn, betaOut, zLo, zHi, rtLo, rtHi, zLoPointed, zHiPointed, sdlCut, betaInCut, betaOutCut, deltaBetaCut, kZ);
#else

  bool success = runTripletDefaultAlgo(modulesInGPU, hitsInGPU, mdsInGPU, segmentsInGPU, innerInnerLowerModuleIndex, middleLowerModuleIndex, outerOuterLowerModuleIndex, innerSegmentIndex, outerSegmentIndex, zOut, rtOut, deltaPhiPos, deltaPhi, betaIn, betaOut);
#endif

  if(success)
    {
      unsigned int tripletModuleIndex = atomicAdd(&tripletsInGPU.nTriplets[innerInnerLowerModuleArrayIndex], 1);
      if(tripletModuleIndex >= N_MAX_TRIPLETS_PER_MODULE)
      {
          #ifdef Warnings
          if(tripletModuleIndex == N_MAX_TRIPLETS_PER_MODULE)
              printf("Triplet excess alert! Module index = %d\n",innerInnerLowerModuleIndex);
          #endif
      }
      unsigned int tripletIndex = innerInnerLowerModuleArrayIndex * N_MAX_TRIPLETS_PER_MODULE + tripletModuleIndex;
#ifdef CUT_VALUE_DEBUG

        addTripletToMemory(tripletsInGPU, innerSegmentIndex, outerSegmentIndex, innerInnerLowerModuleIndex, middleLowerModuleIndex, outerOuterLowerModuleIndex, zOut, rtOut, deltaPhiPos, deltaPhi, betaIn, betaOut, zLo,zHi, rtLo, rtHi, zLoPointed, zHiPointed, sdlCut, betaInCut, betaOutCut, deltaBetaCut, kZ, tripletIndex);

#else
      addTripletToMemory(tripletsInGPU, innerSegmentIndex, outerSegmentIndex, innerInnerLowerModuleIndex, middleLowerModuleIndex, outerOuterLowerModuleIndex, zOut, rtOut, deltaPhiPos, deltaPhi, betaIn, betaOut, tripletIndex);
#endif
    }
}
#endif
#else
__global__ void createTripletsFromInnerInnerLowerModule(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::triplets& tripletsInGPU, unsigned int innerInnerLowerModuleIndex, unsigned int nInnerSegments, unsigned int nConnectedModules, unsigned int innerInnerLowerModuleArrayIndex)
{
    int innerSegmentArrayIndex = blockIdx.x * blockDim.x + threadIdx.x;
    int outerSegmentArrayIndex = blockIdx.y * blockDim.y + threadIdx.y;

    if(innerSegmentArrayIndex >= nInnerSegments) return;

    unsigned int innerSegmentIndex = innerInnerLowerModuleIndex * N_MAX_SEGMENTS_PER_MODULE + innerSegmentArrayIndex;

    //middle lower module - outer lower module of inner segment
    unsigned int middleLowerModuleIndex = segmentsInGPU.outerLowerModuleIndices[innerSegmentIndex];

    unsigned int nOuterSegments = segmentsInGPU.nSegments[middleLowerModuleIndex] > N_MAX_SEGMENTS_PER_MODULE ? N_MAX_SEGMENTS_PER_MODULE : segmentsInGPU.nSegments[middleLowerModuleIndex];
    if(outerSegmentArrayIndex >= nOuterSegments) return;
    unsigned int outerSegmentIndex = middleLowerModuleIndex * N_MAX_SEGMENTS_PER_MODULE + outerSegmentArrayIndex;
    unsigned int outerOuterLowerModuleIndex = segmentsInGPU.outerLowerModuleIndices[outerSegmentIndex];

    float zOut,rtOut,deltaPhiPos,deltaPhi,betaIn,betaOut;
#ifdef CUT_VALUE_DEBUG
  float zLo, zHi, rtLo, rtHi, zLoPointed, zHiPointed, sdlCut, betaInCut, betaOutCut, deltaBetaCut, kZ;

    bool success = runTripletDefaultAlgo(modulesInGPU, hitsInGPU, mdsInGPU, segmentsInGPU, innerInnerLowerModuleIndex, middleLowerModuleIndex, outerOuterLowerModuleIndex, innerSegmentIndex, outerSegmentIndex, zOut, rtOut, deltaPhiPos, deltaPhi, betaIn, betaOut, zLo, zHi, rtLo, rtHi, zLoPointed, zHiPointed, sdlCut, betaInCut, betaOutCut, deltaBetaCut, kZ);
#else

  bool success = runTripletDefaultAlgo(modulesInGPU, hitsInGPU, mdsInGPU, segmentsInGPU, innerInnerLowerModuleIndex, middleLowerModuleIndex, outerOuterLowerModuleIndex, innerSegmentIndex, outerSegmentIndex, zOut, rtOut, deltaPhiPos, deltaPhi, betaIn, betaOut);
#endif

    if(success)
    {
        unsigned int tripletModuleIndex = atomicAdd(&tripletsInGPU.nTriplets[innerInnerLowerModuleArrayIndex], 1);
        if(tripletModuleIndex >= N_MAX_TRIPLETS_PER_MODULE)
        {
            #ifdef Warnings
            if(tripletModuleIndex == N_MAX_TRIPLETS_PER_MODULE)
                printf("Triplet excess alert! Module index = %d\n",innerInnerLowerModuleIndex);
            #endif
        }
        else
        {
            unsigned int tripletIndex = innerInnerLowerModuleArrayIndex * N_MAX_TRIPLETS_PER_MODULE + tripletModuleIndex;
#ifdef CUT_VALUE_DEBUG

            addTripletToMemory(tripletsInGPU, innerSegmentIndex, outerSegmentIndex, innerInnerLowerModuleIndex, middleLowerModuleIndex, outerOuterLowerModuleIndex, zOut, rtOut, deltaPhiPos, deltaPhi, betaIn, betaOut, zLo,zHi, rtLo, rtHi, zLoPointed, zHiPointed, sdlCut, betaInCut, betaOutCut, deltaBetaCut, kZ, tripletIndex);

#else
        addTripletToMemory(tripletsInGPU, innerSegmentIndex, outerSegmentIndex, innerInnerLowerModuleIndex, middleLowerModuleIndex, outerOuterLowerModuleIndex, zOut, rtOut, deltaPhiPos, deltaPhi, betaIn, betaOut, tripletIndex);
#endif

        }
    }
}

__global__ void createTripletsInGPU(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::triplets& tripletsInGPU)
{
    int innerInnerLowerModuleArrayIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if(innerInnerLowerModuleArrayIndex >= *modulesInGPU.nLowerModules) return;
    unsigned int innerInnerLowerModuleIndex = modulesInGPU.lowerModuleIndices[innerInnerLowerModuleArrayIndex];
    unsigned int nInnerSegments = segmentsInGPU.nSegments[innerInnerLowerModuleIndex] > N_MAX_SEGMENTS_PER_MODULE ? N_MAX_SEGMENTS_PER_MODULE : segmentsInGPU.nSegments[innerInnerLowerModuleIndex] ;
    if(nInnerSegments == 0) return;

    unsigned int nConnectedModules = modulesInGPU.nConnectedModules[innerInnerLowerModuleIndex];
    if(nConnectedModules == 0) return;

    dim3 nThreads(16,16,1);
    dim3 nBlocks(nInnerSegments % nThreads.x == 0 ? nInnerSegments / nThreads.x : nInnerSegments / nThreads.x + 1, N_MAX_SEGMENTS_PER_MODULE % nThreads.y == 0 ? N_MAX_SEGMENTS_PER_MODULE / nThreads.y : N_MAX_SEGMENTS_PER_MODULE / nThreads.y + 1);

    createTripletsFromInnerInnerLowerModule<<<nBlocks,nThreads>>>(modulesInGPU, hitsInGPU, mdsInGPU, segmentsInGPU, tripletsInGPU, innerInnerLowerModuleIndex, nInnerSegments, nConnectedModules, innerInnerLowerModuleArrayIndex);
}
#endif

#ifndef NESTED_PARA
__global__ void createPixelTrackCandidatesInGPU(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::tracklets& trackletsInGPU, struct SDL::triplets& tripletsInGPU, struct SDL::trackCandidates& trackCandidatesInGPU, unsigned int* threadIdx_gpu, unsigned int* threadIdx_gpu_offset)
{
  unsigned int outerInnerInnerLowerModuleArrayIndex = threadIdx_gpu[blockIdx.y * blockDim.y + threadIdx.y];
  if(outerInnerInnerLowerModuleArrayIndex >= *modulesInGPU.nLowerModules) return;
  //FIXME:Cheapo module map - We care about pT4s and pTCs Only if the outerInnerInnerLowerModule is "connected" to the pixel module

  int outerInnerInnerLowerModuleIndex = modulesInGPU.lowerModuleIndices[outerInnerInnerLowerModuleArrayIndex];
  if(modulesInGPU.moduleType[outerInnerInnerLowerModuleIndex] == SDL::TwoS) return;

  unsigned int pixelLowerModuleArrayIndex = *modulesInGPU.nLowerModules;

  unsigned int nPixelTracklets = trackletsInGPU.nTracklets[pixelLowerModuleArrayIndex];
  //capping
  if(nPixelTracklets > N_MAX_PIXEL_TRACKLETS_PER_MODULE)
    nPixelTracklets = N_MAX_PIXEL_TRACKLETS_PER_MODULE;

  unsigned int nOuterLayerTracklets = trackletsInGPU.nTracklets[outerInnerInnerLowerModuleArrayIndex];
  if(nOuterLayerTracklets > N_MAX_TRACKLETS_PER_MODULE)
    {
      nOuterLayerTracklets = N_MAX_TRACKLETS_PER_MODULE;
    }
  unsigned int nOuterLayerTriplets = tripletsInGPU.nTriplets[outerInnerInnerLowerModuleArrayIndex];
  if(nOuterLayerTriplets > N_MAX_TRIPLETS_PER_MODULE)
    {
      nOuterLayerTriplets = N_MAX_TRIPLETS_PER_MODULE;
    }

  unsigned int nThreadsForNestedKernel = max(nOuterLayerTracklets,nOuterLayerTriplets);
  if(nThreadsForNestedKernel == 0) return;

  int pixelTrackletArrayIndex = blockIdx.x * blockDim.x + threadIdx.x;

  int outerObjectArrayIndex = threadIdx_gpu_offset[blockIdx.y * blockDim.y+ threadIdx.y];
  if(pixelTrackletArrayIndex >= nPixelTracklets) return;

  int pixelTrackletIndex = pixelLowerModuleArrayIndex * N_MAX_TRACKLETS_PER_MODULE + pixelTrackletArrayIndex;
  int outerObjectIndex = 0;
  short trackCandidateType;
  bool success;

  //pT4-T4
  if(outerObjectArrayIndex < nOuterLayerTracklets)
    {
      outerObjectIndex = outerInnerInnerLowerModuleArrayIndex * N_MAX_TRACKLETS_PER_MODULE + outerObjectArrayIndex;

      //part 2 of cheapo module map : only considering tracklets with PS-PS inner segment
      if(modulesInGPU.moduleType[trackletsInGPU.lowerModuleIndices[4 * outerObjectIndex + 1]] == SDL::PS)
        {
	  success = runTrackCandidateDefaultAlgoTwoTracklets(trackletsInGPU, tripletsInGPU, pixelTrackletIndex, outerObjectIndex, trackCandidateType);
	  if(success)
            {
	      unsigned int trackCandidateModuleIdx = atomicAdd(&trackCandidatesInGPU.nTrackCandidates[pixelLowerModuleArrayIndex],1);
	      atomicAdd(&trackCandidatesInGPU.nTrackCandidatesT4T4[pixelLowerModuleArrayIndex],1);
	      if(trackCandidateModuleIdx >= N_MAX_PIXEL_TRACK_CANDIDATES_PER_MODULE)
                {
                    #ifdef Warnings
		  if(innerInnerInnerLowerModuleArrayIndex == *modulesInGPU.nLowerModules && trackCandidateModuleIdx == N_MAX_PIXEL_TRACK_CANDIDATES_PER_MODULE)
                    {

		      printf("Track Candidate excess alert! lower Module array index = %d\n",innerInnerInnerLowerModuleArrayIndex);
                    }
                    #endif
                }
	      else
                {
		  if(modulesInGPU.trackCandidateModuleIndices[pixelLowerModuleArrayIndex] == -1)
                    {
                       #ifdef Warnings
		      printf("Track candidates: no memory for pixel lower module index at %d\n",innerInnerInnerLowerModuleArrayIndex);
                       #endif

                    }
		  else
		    {
		      unsigned int trackCandidateIdx = modulesInGPU.trackCandidateModuleIndices[pixelLowerModuleArrayIndex] + trackCandidateModuleIdx;
		      addTrackCandidateToMemory(trackCandidatesInGPU, trackCandidateType, pixelTrackletIndex, outerObjectIndex, trackCandidateIdx);
                    }

                }
            }
        }
    }

  //pT4-T3
  if(outerObjectArrayIndex < nOuterLayerTriplets)
    {
      outerObjectIndex = outerInnerInnerLowerModuleArrayIndex * N_MAX_TRIPLETS_PER_MODULE + outerObjectArrayIndex;

      //part 2 of cheapo module map : only considering tracklets with PS-PS inner segment
      if(modulesInGPU.moduleType[tripletsInGPU.lowerModuleIndices[3 * outerObjectIndex + 1]] == SDL::PS)
        {
	  success = runTrackCandidateDefaultAlgoTrackletToTriplet(trackletsInGPU, tripletsInGPU, pixelTrackletIndex, outerObjectIndex, trackCandidateType);
	  if(success)
            {
	      unsigned int trackCandidateModuleIdx = atomicAdd(&trackCandidatesInGPU.nTrackCandidates[pixelLowerModuleArrayIndex],1);
	      atomicAdd(&trackCandidatesInGPU.nTrackCandidatesT4T4[pixelLowerModuleArrayIndex],1);
	      if(trackCandidateModuleIdx >= N_MAX_PIXEL_TRACK_CANDIDATES_PER_MODULE)
                {
                    #ifdef Warnings
		  if(innerInnerInnerLowerModuleArrayIndex == *modulesInGPU.nLowerModules && trackCandidateModuleIdx == N_MAX_PIXEL_TRACK_CANDIDATES_PER_MODULE)
                    {

		      printf("Track Candidate excess alert! lower Module array index = %d\n",innerInnerInnerLowerModuleArrayIndex);
                    }
                    #endif
                }
	      else
                {
		  if(modulesInGPU.trackCandidateModuleIndices[pixelLowerModuleArrayIndex] == -1)
                    {
                       #ifdef Warnings
		      printf("Track candidates: no memory for pixel lower module index at %d\n",innerInnerInnerLowerModuleArrayIndex);
                       #endif

                    }
		  else
		    {
		      unsigned int trackCandidateIdx = modulesInGPU.trackCandidateModuleIndices[pixelLowerModuleArrayIndex] + trackCandidateModuleIdx;
		      addTrackCandidateToMemory(trackCandidatesInGPU, trackCandidateType, pixelTrackletIndex, outerObjectIndex, trackCandidateIdx);
                    }

                }
            }
        }

    }
}

#else
__global__ void createPixelTrackCandidatesInGPU(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::tracklets& trackletsInGPU, struct SDL::triplets& tripletsInGPU, struct SDL::trackCandidates& trackCandidatesInGPU)
{
    unsigned int outerInnerInnerLowerModuleArrayIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if(outerInnerInnerLowerModuleArrayIndex >= *modulesInGPU.nLowerModules) return;
    //FIXME:Cheapo module map - We care about pT4s and pTCs Only if the outerInnerInnerLowerModule is "connected" to the pixel module

    int outerInnerInnerLowerModuleIndex = modulesInGPU.lowerModuleIndices[outerInnerInnerLowerModuleArrayIndex];
    if(modulesInGPU.moduleType[outerInnerInnerLowerModuleIndex] == SDL::TwoS) return;

    unsigned int pixelLowerModuleArrayIndex = *modulesInGPU.nLowerModules;

    unsigned int nPixelTracklets = trackletsInGPU.nTracklets[pixelLowerModuleArrayIndex];
    //capping
    if(nPixelTracklets > N_MAX_PIXEL_TRACKLETS_PER_MODULE)
        nPixelTracklets = N_MAX_PIXEL_TRACKLETS_PER_MODULE;

    unsigned int nOuterLayerTracklets = trackletsInGPU.nTracklets[outerInnerInnerLowerModuleArrayIndex];
    if(nOuterLayerTracklets > N_MAX_TRACKLETS_PER_MODULE)
    {
        nOuterLayerTracklets = N_MAX_TRACKLETS_PER_MODULE;
    }
    unsigned int nOuterLayerTriplets = tripletsInGPU.nTriplets[outerInnerInnerLowerModuleArrayIndex];
    if(nOuterLayerTriplets > N_MAX_TRIPLETS_PER_MODULE)
    {
        nOuterLayerTriplets = N_MAX_TRIPLETS_PER_MODULE;
    }

    unsigned int nThreadsForNestedKernel = max(nOuterLayerTracklets,nOuterLayerTriplets);
    if(nThreadsForNestedKernel == 0) return;

    dim3 nThreads(16,16,1);
    dim3 nBlocks( nPixelTracklets % nThreads.x == 0 ? nPixelTracklets/nThreads.x : nPixelTracklets/nThreads.x + 1, nThreadsForNestedKernel % nThreads.y == 0 ? nThreadsForNestedKernel/nThreads.y : nThreadsForNestedKernel/nThreads.y + 1, 1);

    createPixelTrackCandidatesFromOuterInnerInnerLowerModule<<<nBlocks,nThreads>>>(modulesInGPU, hitsInGPU, mdsInGPU, segmentsInGPU, trackletsInGPU, tripletsInGPU, trackCandidatesInGPU, pixelLowerModuleArrayIndex, outerInnerInnerLowerModuleArrayIndex, nPixelTracklets, nOuterLayerTracklets, nOuterLayerTriplets);
}


__global__ void createPixelTrackCandidatesFromOuterInnerInnerLowerModule(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::tracklets& trackletsInGPU, struct SDL::triplets& tripletsInGPU, struct SDL::trackCandidates& trackCandidatesInGPU, unsigned int pixelLowerModuleArrayIndex, unsigned int outerInnerInnerLowerModuleArrayIndex, unsigned int nPixelTracklets, unsigned int nOuterLayerTracklets, unsigned int nOuterLayerTriplets)
{
    int pixelTrackletArrayIndex = blockIdx.x * blockDim.x + threadIdx.x;
    int outerObjectArrayIndex = blockIdx.y * blockDim.y + threadIdx.y;

    if(pixelTrackletArrayIndex >= nPixelTracklets) return;

    int pixelTrackletIndex = pixelLowerModuleArrayIndex * N_MAX_TRACKLETS_PER_MODULE + pixelTrackletArrayIndex;
    int outerObjectIndex = 0;
    short trackCandidateType;
    bool success;

    //pT4-T4
    if(outerObjectArrayIndex < nOuterLayerTracklets)
    {
        outerObjectIndex = outerInnerInnerLowerModuleArrayIndex * N_MAX_TRACKLETS_PER_MODULE + outerObjectArrayIndex;

        //part 2 of cheapo module map : only considering tracklets with PS-PS inner segment
       if(modulesInGPU.moduleType[trackletsInGPU.lowerModuleIndices[4 * outerObjectIndex + 1]] == SDL::PS)
        {
            success = runTrackCandidateDefaultAlgoTwoTracklets(trackletsInGPU, tripletsInGPU, pixelTrackletIndex, outerObjectIndex, trackCandidateType);
            if(success)
            {
                unsigned int trackCandidateModuleIdx = atomicAdd(&trackCandidatesInGPU.nTrackCandidates[pixelLowerModuleArrayIndex],1);
                atomicAdd(&trackCandidatesInGPU.nTrackCandidatesT4T4[pixelLowerModuleArrayIndex],1);
                if(trackCandidateModuleIdx >= N_MAX_PIXEL_TRACK_CANDIDATES_PER_MODULE)
                {
                    #ifdef Warnings
                    if(innerInnerInnerLowerModuleArrayIndex == *modulesInGPU.nLowerModules && trackCandidateModuleIdx == N_MAX_PIXEL_TRACK_CANDIDATES_PER_MODULE)
                    {

                        printf("Track Candidate excess alert! lower Module array index = %d\n",innerInnerInnerLowerModuleArrayIndex);
                    }
                    #endif
                }
                else
                {
                    if(modulesInGPU.trackCandidateModuleIndices[pixelLowerModuleArrayIndex] == -1)
                    {
                       #ifdef Warnings
                       printf("Track candidates: no memory for pixel lower module index at %d\n",innerInnerInnerLowerModuleArrayIndex);
                       #endif

                    }
                    else
                   {
                        unsigned int trackCandidateIdx = modulesInGPU.trackCandidateModuleIndices[pixelLowerModuleArrayIndex] + trackCandidateModuleIdx;
                        addTrackCandidateToMemory(trackCandidatesInGPU, trackCandidateType, pixelTrackletIndex, outerObjectIndex, trackCandidateIdx);
                    }

                }
            }
        }
    }

    //pT4-T3
    if(outerObjectArrayIndex < nOuterLayerTriplets)
    {
        outerObjectIndex = outerInnerInnerLowerModuleArrayIndex * N_MAX_TRIPLETS_PER_MODULE + outerObjectArrayIndex;

        //part 2 of cheapo module map : only considering tracklets with PS-PS inner segment
        if(modulesInGPU.moduleType[tripletsInGPU.lowerModuleIndices[3 * outerObjectIndex + 1]] == SDL::PS)
        {
            success = runTrackCandidateDefaultAlgoTrackletToTriplet(trackletsInGPU, tripletsInGPU, pixelTrackletIndex, outerObjectIndex, trackCandidateType);
            if(success)
            {
                unsigned int trackCandidateModuleIdx = atomicAdd(&trackCandidatesInGPU.nTrackCandidates[pixelLowerModuleArrayIndex],1);
                atomicAdd(&trackCandidatesInGPU.nTrackCandidatesT4T4[pixelLowerModuleArrayIndex],1);
                if(trackCandidateModuleIdx >= N_MAX_PIXEL_TRACK_CANDIDATES_PER_MODULE)
                {
                    #ifdef Warnings
                    if(innerInnerInnerLowerModuleArrayIndex == *modulesInGPU.nLowerModules && trackCandidateModuleIdx == N_MAX_PIXEL_TRACK_CANDIDATES_PER_MODULE)
                    {

                        printf("Track Candidate excess alert! lower Module array index = %d\n",innerInnerInnerLowerModuleArrayIndex);
                    }
                    #endif
                }
                else
                {
                    if(modulesInGPU.trackCandidateModuleIndices[pixelLowerModuleArrayIndex] == -1)
                    {
                       #ifdef Warnings
                       printf("Track candidates: no memory for pixel lower module index at %d\n",innerInnerInnerLowerModuleArrayIndex);
                       #endif

                    }
                    else
                   {
                        unsigned int trackCandidateIdx = modulesInGPU.trackCandidateModuleIndices[pixelLowerModuleArrayIndex] + trackCandidateModuleIdx;
                        addTrackCandidateToMemory(trackCandidatesInGPU, trackCandidateType, pixelTrackletIndex, outerObjectIndex, trackCandidateIdx);
                    }

                }
            }
        }

    }
}
#endif

#ifndef NESTED_PARA
__global__ void createTrackCandidatesInGPU(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::tracklets& trackletsInGPU, struct SDL::triplets& tripletsInGPU, struct SDL::trackCandidates& trackCandidatesInGPU, unsigned int* threadIdx_gpu, unsigned int *threadIdx_gpu_offset)
{
  //inner tracklet/triplet inner segment inner MD lower module
  int innerInnerInnerLowerModuleArrayIndex = threadIdx_gpu[blockIdx.y * blockDim.y + threadIdx.y];
  //hack to include pixel detector
  if(innerInnerInnerLowerModuleArrayIndex >= *modulesInGPU.nLowerModules) return;

  unsigned int nTracklets = trackletsInGPU.nTracklets[innerInnerInnerLowerModuleArrayIndex];
  if(nTracklets > N_MAX_TRACKLETS_PER_MODULE)
    {
      nTracklets = N_MAX_TRACKLETS_PER_MODULE;
    }

  unsigned int nTriplets = tripletsInGPU.nTriplets[innerInnerInnerLowerModuleArrayIndex]; // should be zero for the pixels
  if(nTriplets > N_MAX_TRIPLETS_PER_MODULE)
    {
      nTriplets = N_MAX_TRIPLETS_PER_MODULE;
    }

  unsigned int temp = max(nTracklets,nTriplets);
  unsigned int MAX_OBJECTS = max(N_MAX_TRACKLETS_PER_MODULE, N_MAX_TRIPLETS_PER_MODULE);

  if(temp == 0) return;

  int innerObjectArrayIndex = threadIdx_gpu_offset[blockIdx.y * blockDim.y + threadIdx.y];
  int outerObjectArrayIndex = blockIdx.x * blockDim.x + threadIdx.x;

  int innerObjectIndex = 0;
  int outerObjectIndex = 0;
  short trackCandidateType;
  bool success;

  //step 1 tracklet-tracklet
  if(innerObjectArrayIndex < nTracklets)
    {
      innerObjectIndex = innerInnerInnerLowerModuleArrayIndex * N_MAX_TRACKLETS_PER_MODULE + innerObjectArrayIndex;
      unsigned int outerInnerInnerLowerModuleIndex = modulesInGPU.reverseLookupLowerModuleIndices[trackletsInGPU.lowerModuleIndices[4 * innerObjectIndex + 2]];/*same as innerOuterInnerLowerModuleIndex*/

      if(outerObjectArrayIndex < fminf(trackletsInGPU.nTracklets[outerInnerInnerLowerModuleIndex],N_MAX_TRACKLETS_PER_MODULE))
        {

	  outerObjectIndex = outerInnerInnerLowerModuleIndex * N_MAX_TRACKLETS_PER_MODULE + outerObjectArrayIndex;

	  success = runTrackCandidateDefaultAlgoTwoTracklets(trackletsInGPU, tripletsInGPU, innerObjectIndex, outerObjectIndex,trackCandidateType);

	  if(success)
            {
	      unsigned int trackCandidateModuleIdx = atomicAdd(&trackCandidatesInGPU.nTrackCandidates[innerInnerInnerLowerModuleArrayIndex],1);
	      atomicAdd(&trackCandidatesInGPU.nTrackCandidatesT4T4[innerInnerInnerLowerModuleArrayIndex],1);
	      if(trackCandidateModuleIdx >= N_MAX_TRACK_CANDIDATES_PER_MODULE)
                {
                    #ifdef Warnings
		  if(trackCandidateModuleIdx == N_MAX_TRACK_CANDIDATES_PER_MODULE)
                    {
		      printf("Track Candidate excess alert! lower Module array index = %d\n",innerInnerInnerLowerModuleArrayIndex);
                    }
                    #endif
                }
	      else
                {
		  if(modulesInGPU.trackCandidateModuleIndices[innerInnerInnerLowerModuleArrayIndex] == -1)
                    {
                       #ifdef Warnings
		      printf("Track candidates: no memory for module at module index = %d\n",innerInnerInnerLowerModuleArrayIndex);
                       #endif

                    }
		  else
		    {
		      unsigned int trackCandidateIdx = modulesInGPU.trackCandidateModuleIndices[innerInnerInnerLowerModuleArrayIndex] + trackCandidateModuleIdx;
		      addTrackCandidateToMemory(trackCandidatesInGPU, trackCandidateType, innerObjectIndex, outerObjectIndex, trackCandidateIdx);
                    }

                }
            }

        }
    }

  //step 2 tracklet-triplet
  if(innerObjectArrayIndex < nTracklets)
    {
      innerObjectIndex = innerInnerInnerLowerModuleArrayIndex * N_MAX_TRACKLETS_PER_MODULE + innerObjectArrayIndex;
      unsigned int outerInnerInnerLowerModuleIndex = modulesInGPU.reverseLookupLowerModuleIndices[trackletsInGPU.lowerModuleIndices[4 * innerObjectIndex + 2]];//same as innerOuterInnerLowerModuleIndex
      if(outerObjectArrayIndex < fminf(tripletsInGPU.nTriplets[outerInnerInnerLowerModuleIndex],N_MAX_TRIPLETS_PER_MODULE))
        {
	  outerObjectIndex = outerInnerInnerLowerModuleIndex * N_MAX_TRIPLETS_PER_MODULE + outerObjectArrayIndex;
	  success = runTrackCandidateDefaultAlgoTrackletToTriplet(trackletsInGPU, tripletsInGPU, innerObjectIndex, outerObjectIndex,trackCandidateType);
	  if(success)
            {
	      unsigned int trackCandidateModuleIdx = atomicAdd(&trackCandidatesInGPU.nTrackCandidates[innerInnerInnerLowerModuleArrayIndex],1);
	      atomicAdd(&trackCandidatesInGPU.nTrackCandidatesT4T3[innerInnerInnerLowerModuleArrayIndex],1);
	      if(trackCandidateModuleIdx >= N_MAX_TRACK_CANDIDATES_PER_MODULE)
                {
                    #ifdef Warnings
		  if(trackCandidateModuleIdx == N_MAX_TRACK_CANDIDATES_PER_MODULE)
                    {
		      printf("Track Candidate excess alert! lower Module array index = %d\n",innerInnerInnerLowerModuleArrayIndex);
                    }
                    #endif
                }
	      else
                {

		  if(modulesInGPU.trackCandidateModuleIndices[innerInnerInnerLowerModuleArrayIndex] == -1)
                    {
                        #ifdef Warnings
		      printf("Track candidates: no memory for module at module index = %d\n",innerInnerInnerLowerModuleArrayIndex);
                        #endif
                    }
		  else
                    {
		      unsigned int trackCandidateIdx = modulesInGPU.trackCandidateModuleIndices[innerInnerInnerLowerModuleArrayIndex] + trackCandidateModuleIdx;

		      addTrackCandidateToMemory(trackCandidatesInGPU, trackCandidateType, innerObjectIndex, outerObjectIndex, trackCandidateIdx);
                    }
                }
            }

        }
    }
  //step 3 triplet-tracklet
  if(innerObjectArrayIndex < nTriplets)
    {
      innerObjectIndex = innerInnerInnerLowerModuleArrayIndex * N_MAX_TRIPLETS_PER_MODULE + innerObjectArrayIndex;
      unsigned int outerInnerInnerLowerModuleIndex = modulesInGPU.reverseLookupLowerModuleIndices[tripletsInGPU.lowerModuleIndices[3 * innerObjectIndex + 1]];//same as innerOuterInnerLowerModuleIndex

      if(outerObjectArrayIndex < fminf(trackletsInGPU.nTracklets[outerInnerInnerLowerModuleIndex],N_MAX_TRACKLETS_PER_MODULE))
        {
	  outerObjectIndex = outerInnerInnerLowerModuleIndex * N_MAX_TRACKLETS_PER_MODULE + outerObjectArrayIndex;
	  success = runTrackCandidateDefaultAlgoTripletToTracklet(trackletsInGPU, tripletsInGPU, innerObjectIndex, outerObjectIndex,trackCandidateType);
	  if(success)
            {
	      unsigned int trackCandidateModuleIdx = atomicAdd(&trackCandidatesInGPU.nTrackCandidates[innerInnerInnerLowerModuleArrayIndex],1);
	      atomicAdd(&trackCandidatesInGPU.nTrackCandidatesT3T4[innerInnerInnerLowerModuleArrayIndex],1);
	      if(trackCandidateModuleIdx >= N_MAX_TRACK_CANDIDATES_PER_MODULE)
                {
                   #ifdef Warnings
		  if(trackCandidateModuleIdx == N_MAX_TRACK_CANDIDATES_PER_MODULE)
		    printf("Track Candidate excess alert! Module index = %d\n",innerInnerInnerLowerModuleArrayIndex);
                   #endif
                }
	      else
                {
		  if(modulesInGPU.trackCandidateModuleIndices[innerInnerInnerLowerModuleArrayIndex] == -1)
                    {
                        #ifdef Warnings
		      printf("Track candidates: no memory for module at module index = %d, outer T4 module index = %d\n",innerInnerInnerLowerModuleArrayIndex, outerInnerInnerLowerModuleIndex);
                        #endif
                    }
		  else
                    {
		      unsigned int trackCandidateIdx = modulesInGPU.trackCandidateModuleIndices[innerInnerInnerLowerModuleArrayIndex] + trackCandidateModuleIdx;
		      addTrackCandidateToMemory(trackCandidatesInGPU, trackCandidateType, innerObjectIndex, outerObjectIndex, trackCandidateIdx);

                    }
                }
            }

        }
    }
}

#else
__global__ void createTrackCandidatesInGPU(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::tracklets& trackletsInGPU, struct SDL::triplets& tripletsInGPU, struct SDL::trackCandidates& trackCandidatesInGPU)
{
    //inner tracklet/triplet inner segment inner MD lower module
    int innerInnerInnerLowerModuleArrayIndex = blockIdx.x * blockDim.x + threadIdx.x;
    //hack to include pixel detector
    if(innerInnerInnerLowerModuleArrayIndex >= *modulesInGPU.nLowerModules) return;

    unsigned int nTracklets = trackletsInGPU.nTracklets[innerInnerInnerLowerModuleArrayIndex];
    if(nTracklets > N_MAX_TRACKLETS_PER_MODULE)
    {
        nTracklets = N_MAX_TRACKLETS_PER_MODULE;
    }

    unsigned int nTriplets = tripletsInGPU.nTriplets[innerInnerInnerLowerModuleArrayIndex]; // should be zero for the pixels
    if(nTriplets > N_MAX_TRIPLETS_PER_MODULE)
    {
        nTriplets = N_MAX_TRIPLETS_PER_MODULE;
    }

    unsigned int temp = max(nTracklets,nTriplets);
    unsigned int MAX_OBJECTS = max(N_MAX_TRACKLETS_PER_MODULE, N_MAX_TRIPLETS_PER_MODULE);

    if(temp == 0) return;

    //triplets and tracklets are stored directly using lower module array index
    dim3 nThreads(16,16,1);
    dim3 nBlocks(temp % nThreads.x == 0 ? temp / nThreads.x : temp / nThreads.x + 1, MAX_OBJECTS % nThreads.y == 0 ? MAX_OBJECTS / nThreads.y : MAX_OBJECTS / nThreads.y + 1, 1);

    createTrackCandidatesFromInnerInnerInnerLowerModule<<<nBlocks, nThreads>>>(modulesInGPU, hitsInGPU, mdsInGPU, segmentsInGPU, trackletsInGPU, tripletsInGPU, trackCandidatesInGPU,innerInnerInnerLowerModuleArrayIndex,nTracklets,nTriplets);
}

__global__ void createTrackCandidatesFromInnerInnerInnerLowerModule(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::tracklets& trackletsInGPU, struct SDL::triplets& tripletsInGPU, struct SDL::trackCandidates& trackCandidatesInGPU, unsigned int innerInnerInnerLowerModuleArrayIndex, unsigned int nInnerTracklets, unsigned int nInnerTriplets)
{
    int innerObjectArrayIndex = blockIdx.x * blockDim.x + threadIdx.x;
    int outerObjectArrayIndex = blockIdx.y * blockDim.y + threadIdx.y;

    int innerObjectIndex = 0;
    int outerObjectIndex = 0;
    short trackCandidateType;
    bool success;
    //step 1 tracklet-tracklet
    if(innerObjectArrayIndex < nInnerTracklets)
    {
        innerObjectIndex = innerInnerInnerLowerModuleArrayIndex * N_MAX_TRACKLETS_PER_MODULE + innerObjectArrayIndex;
        unsigned int outerInnerInnerLowerModuleIndex = modulesInGPU.reverseLookupLowerModuleIndices[trackletsInGPU.lowerModuleIndices[4 * innerObjectIndex + 2]];/*same as innerOuterInnerLowerModuleIndex*/

        if(outerObjectArrayIndex < fminf(trackletsInGPU.nTracklets[outerInnerInnerLowerModuleIndex],N_MAX_TRACKLETS_PER_MODULE))
        {

            outerObjectIndex = outerInnerInnerLowerModuleIndex * N_MAX_TRACKLETS_PER_MODULE + outerObjectArrayIndex;

            success = runTrackCandidateDefaultAlgoTwoTracklets(trackletsInGPU, tripletsInGPU, innerObjectIndex, outerObjectIndex,trackCandidateType);

            if(success)
            {
                unsigned int trackCandidateModuleIdx = atomicAdd(&trackCandidatesInGPU.nTrackCandidates[innerInnerInnerLowerModuleArrayIndex],1);
                atomicAdd(&trackCandidatesInGPU.nTrackCandidatesT4T4[innerInnerInnerLowerModuleArrayIndex],1);
                if(trackCandidateModuleIdx >= N_MAX_TRACK_CANDIDATES_PER_MODULE)
                {
                    #ifdef Warnings
                    if(trackCandidateModuleIdx == N_MAX_TRACK_CANDIDATES_PER_MODULE)
                    {
                        printf("Track Candidate excess alert! lower Module array index = %d\n",innerInnerInnerLowerModuleArrayIndex);
                    }
                    #endif
                }
                else
                {
                    if(modulesInGPU.trackCandidateModuleIndices[innerInnerInnerLowerModuleArrayIndex] == -1)
                    {
                       #ifdef Warnings
                       printf("Track candidates: no memory for module at module index = %d\n",innerInnerInnerLowerModuleArrayIndex);
                       #endif

                    }
                    else
                   {
                        unsigned int trackCandidateIdx = modulesInGPU.trackCandidateModuleIndices[innerInnerInnerLowerModuleArrayIndex] + trackCandidateModuleIdx;
                        addTrackCandidateToMemory(trackCandidatesInGPU, trackCandidateType, innerObjectIndex, outerObjectIndex, trackCandidateIdx);
                    }

                }
            }

        }
    }
    //step 2 tracklet-triplet
    if(innerObjectArrayIndex < nInnerTracklets)
    {
        innerObjectIndex = innerInnerInnerLowerModuleArrayIndex * N_MAX_TRACKLETS_PER_MODULE + innerObjectArrayIndex;
        unsigned int outerInnerInnerLowerModuleIndex = modulesInGPU.reverseLookupLowerModuleIndices[trackletsInGPU.lowerModuleIndices[4 * innerObjectIndex + 2]];//same as innerOuterInnerLowerModuleIndex
        if(outerObjectArrayIndex < fminf(tripletsInGPU.nTriplets[outerInnerInnerLowerModuleIndex],N_MAX_TRIPLETS_PER_MODULE))
        {
            outerObjectIndex = outerInnerInnerLowerModuleIndex * N_MAX_TRIPLETS_PER_MODULE + outerObjectArrayIndex;
            success = runTrackCandidateDefaultAlgoTrackletToTriplet(trackletsInGPU, tripletsInGPU, innerObjectIndex, outerObjectIndex,trackCandidateType);
            if(success)
            {
                unsigned int trackCandidateModuleIdx = atomicAdd(&trackCandidatesInGPU.nTrackCandidates[innerInnerInnerLowerModuleArrayIndex],1);
                atomicAdd(&trackCandidatesInGPU.nTrackCandidatesT4T3[innerInnerInnerLowerModuleArrayIndex],1);
                if(trackCandidateModuleIdx >= N_MAX_TRACK_CANDIDATES_PER_MODULE)
                {
                    #ifdef Warnings
                    if(trackCandidateModuleIdx == N_MAX_TRACK_CANDIDATES_PER_MODULE)
                    {
                        printf("Track Candidate excess alert! lower Module array index = %d\n",innerInnerInnerLowerModuleArrayIndex);
                    }
                    #endif
                }
                else
                {

                    if(modulesInGPU.trackCandidateModuleIndices[innerInnerInnerLowerModuleArrayIndex] == -1)
                    {
                        #ifdef Warnings
                        printf("Track candidates: no memory for module at module index = %d\n",innerInnerInnerLowerModuleArrayIndex);
                        #endif
                    }
                    else
                    {
                        unsigned int trackCandidateIdx = modulesInGPU.trackCandidateModuleIndices[innerInnerInnerLowerModuleArrayIndex] + trackCandidateModuleIdx;

                        addTrackCandidateToMemory(trackCandidatesInGPU, trackCandidateType, innerObjectIndex, outerObjectIndex, trackCandidateIdx);
                    }
                }
            }

        }
    }

    //step 3 triplet-tracklet
    if(innerObjectArrayIndex < nInnerTriplets)
    {
        innerObjectIndex = innerInnerInnerLowerModuleArrayIndex * N_MAX_TRIPLETS_PER_MODULE + innerObjectArrayIndex;
        unsigned int outerInnerInnerLowerModuleIndex = modulesInGPU.reverseLookupLowerModuleIndices[tripletsInGPU.lowerModuleIndices[3 * innerObjectIndex + 1]];//same as innerOuterInnerLowerModuleIndex

        if(outerObjectArrayIndex < fminf(trackletsInGPU.nTracklets[outerInnerInnerLowerModuleIndex],N_MAX_TRACKLETS_PER_MODULE))
        {
            outerObjectIndex = outerInnerInnerLowerModuleIndex * N_MAX_TRACKLETS_PER_MODULE + outerObjectArrayIndex;
            success = runTrackCandidateDefaultAlgoTripletToTracklet(trackletsInGPU, tripletsInGPU, innerObjectIndex, outerObjectIndex,trackCandidateType);
            if(success)
            {
                unsigned int trackCandidateModuleIdx = atomicAdd(&trackCandidatesInGPU.nTrackCandidates[innerInnerInnerLowerModuleArrayIndex],1);
                atomicAdd(&trackCandidatesInGPU.nTrackCandidatesT3T4[innerInnerInnerLowerModuleArrayIndex],1);
	        if(trackCandidateModuleIdx >= N_MAX_TRACK_CANDIDATES_PER_MODULE)
                {
                   #ifdef Warnings
                   if(trackCandidateModuleIdx == N_MAX_TRACK_CANDIDATES_PER_MODULE)
                       printf("Track Candidate excess alert! Module index = %d\n",innerInnerInnerLowerModuleArrayIndex);
                   #endif
                }
                else
                {
                    if(modulesInGPU.trackCandidateModuleIndices[innerInnerInnerLowerModuleArrayIndex] == -1)
                    {
                        #ifdef Warnings
                        printf("Track candidates: no memory for module at module index = %d, outer T4 module index = %d\n",innerInnerInnerLowerModuleArrayIndex, outerInnerInnerLowerModuleIndex);
                        #endif
                    }
                    else
                    {
                        unsigned int trackCandidateIdx = modulesInGPU.trackCandidateModuleIndices[innerInnerInnerLowerModuleArrayIndex] + trackCandidateModuleIdx;
                        addTrackCandidateToMemory(trackCandidatesInGPU, trackCandidateType, innerObjectIndex, outerObjectIndex, trackCandidateIdx);

                    }
                }
            }

        }
    }
}
#endif


#ifndef NESTED_PARA
__global__ void createQuintupletsInGPU(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::triplets& tripletsInGPU, struct SDL::quintuplets& quintupletsInGPU, unsigned int* index_gpu)
{
    int lowerModuleArray1 = index_gpu[blockIdx.z * blockDim.z + threadIdx.z];

    //this if statement never gets executed!
    if(lowerModuleArray1  >= *modulesInGPU.nLowerModules) return;

    unsigned int nInnerTriplets = tripletsInGPU.nTriplets[lowerModuleArray1];

    unsigned int innerTripletArrayIndex = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int outerTripletArrayIndex = blockIdx.x * blockDim.x + threadIdx.x;

    if(innerTripletArrayIndex >= nInnerTriplets) return;

    unsigned int innerTripletIndex = lowerModuleArray1 * N_MAX_TRIPLETS_PER_MODULE + innerTripletArrayIndex;
    unsigned int lowerModule1 = modulesInGPU.lowerModuleIndices[lowerModuleArray1];
    //these are actual module indices!! not lower module indices!
    unsigned int lowerModule2 = tripletsInGPU.lowerModuleIndices[3 * innerTripletIndex + 1];
    unsigned int lowerModule3 = tripletsInGPU.lowerModuleIndices[3 * innerTripletIndex + 2];
    unsigned int lowerModuleArray3 = modulesInGPU.reverseLookupLowerModuleIndices[lowerModule3];
    unsigned int nOuterTriplets = min(tripletsInGPU.nTriplets[lowerModuleArray3], N_MAX_TRIPLETS_PER_MODULE);

    if(outerTripletArrayIndex >= nOuterTriplets) return;
    unsigned int outerTripletIndex = lowerModuleArray3 * N_MAX_TRIPLETS_PER_MODULE + outerTripletArrayIndex;
    //these are actual module indices!!
    unsigned int lowerModule4 = tripletsInGPU.lowerModuleIndices[3 * outerTripletIndex + 1];
    unsigned int lowerModule5 = tripletsInGPU.lowerModuleIndices[3 * outerTripletIndex + 2];
   
    float innerRadius, innerRadiusMin, innerRadiusMin2S, innerRadiusMax, innerRadiusMax2S, outerRadius, outerRadiusMin, outerRadiusMin2S, outerRadiusMax, outerRadiusMax2S, bridgeRadius, bridgeRadiusMin, bridgeRadiusMin2S, bridgeRadiusMax, bridgeRadiusMax2S; //required for making distributions
    bool success = runQuintupletDefaultAlgo(modulesInGPU, hitsInGPU, mdsInGPU, segmentsInGPU, tripletsInGPU, lowerModule1, lowerModule2, lowerModule3, lowerModule4, lowerModule5, innerTripletIndex, outerTripletIndex, innerRadius, innerRadiusMin, innerRadiusMax, outerRadius, outerRadiusMin, outerRadiusMax, bridgeRadius, bridgeRadiusMin, bridgeRadiusMax, innerRadiusMin2S, innerRadiusMax2S, bridgeRadiusMin2S, bridgeRadiusMax2S, outerRadiusMin2S,
            outerRadiusMax2S);

   if(success)
   {
       unsigned int quintupletModuleIndex = atomicAdd(&quintupletsInGPU.nQuintuplets[lowerModuleArray1], 1);
       if(quintupletModuleIndex >= N_MAX_QUINTUPLETS_PER_MODULE)
       {
#ifdef Warnings
           if(quintupletModuleIndex ==  N_MAX_QUINTUPLETS_PER_MODULE)
               printf("Quintuplet excess alert! Module index = %d\n", lowerModuleArray1);
#endif
       }
       else
       {
           //this if statement should never get executed!
           if(modulesInGPU.quintupletModuleIndices[lowerModuleArray1] == -1)
           {
                printf("Quintuplets : no memory for module at module index = %d\n", lowerModuleArray1);
           }
           else
           {
                unsigned int quintupletIndex = modulesInGPU.quintupletModuleIndices[lowerModuleArray1] +  quintupletModuleIndex;
#ifdef CUT_VALUE_DEBUG
                addQuintupletToMemory(quintupletsInGPU, innerTripletIndex, outerTripletIndex, lowerModule1, lowerModule2, lowerModule3, lowerModule4, lowerModule5, innerRadius, innerRadiusMin, innerRadiusMax, outerRadius, outerRadiusMin, outerRadiusMax, bridgeRadius, bridgeRadiusMin, bridgeRadiusMax, innerRadiusMin2S, innerRadiusMax2S, bridgeRadiusMin2S, bridgeRadiusMax2S, outerRadiusMin2S, outerRadiusMax2S, quintupletIndex);
#else
                addQuintupletToMemory(quintupletsInGPU, innerTripletIndex, outerTripletIndex, lowerModule1, lowerModule2, lowerModule3, lowerModule4, lowerModule5, quintupletIndex);
#endif
            }
        }
    }
}

#else
__global__ void createQuintupletsFromInnerInnerLowerModule(SDL::modules& modulesInGPU, SDL::hits& hitsInGPU, SDL::miniDoublets& mdsInGPU, SDL::segments& segmentsInGPU, SDL::triplets& tripletsInGPU, SDL::quintuplets& quintupletsInGPU, unsigned int lowerModuleArray1, unsigned int nInnerTriplets)
{
   int innerTripletArrayIndex = blockIdx.x * blockDim.x + threadIdx.x;
   int outerTripletArrayIndex = blockIdx.y * blockDim.y + threadIdx.y;

   if(innerTripletArrayIndex >= nInnerTriplets) return;

   unsigned int innerTripletIndex = lowerModuleArray1 * N_MAX_TRIPLETS_PER_MODULE + innerTripletArrayIndex;
   unsigned int lowerModule1 = modulesInGPU.lowerModuleIndices[lowerModuleArray1];
   //these are actual module indices!!! not lower module indices
   unsigned int lowerModule2 = tripletsInGPU.lowerModuleIndices[3 * innerTripletIndex + 1];
   unsigned int lowerModule3 = tripletsInGPU.lowerModuleIndices[3 * innerTripletIndex + 2];
   unsigned int lowerModuleArray3 = modulesInGPU.reverseLookupLowerModuleIndices[lowerModule3];

   unsigned int nOuterTriplets = min(tripletsInGPU.nTriplets[lowerModuleArray3], N_MAX_TRIPLETS_PER_MODULE);
  
   if(outerTripletArrayIndex >= nOuterTriplets) return;

   unsigned int outerTripletIndex = lowerModuleArray3 * N_MAX_TRIPLETS_PER_MODULE + outerTripletArrayIndex;
    //these are actual module indices!!
    unsigned int lowerModule4 = tripletsInGPU.lowerModuleIndices[3 * outerTripletIndex + 1];
    unsigned int lowerModule5 = tripletsInGPU.lowerModuleIndices[3 * outerTripletIndex + 2];
   
    float innerRadius, innerRadiusMin, innerRadiusMin2S, innerRadiusMax, innerRadiusMax2S, outerRadius, outerRadiusMin, outerRadiusMin2S, outerRadiusMax, outerRadiusMax2S, bridgeRadius, bridgeRadiusMin, bridgeRadiusMin2S, bridgeRadiusMax, bridgeRadiusMax2S; //required for making distributions
    bool success = runQuintupletDefaultAlgo(modulesInGPU, hitsInGPU, mdsInGPU, segmentsInGPU, tripletsInGPU, lowerModule1, lowerModule2, lowerModule3, lowerModule4, lowerModule5, innerTripletIndex, outerTripletIndex, innerRadius, innerRadiusMin, innerRadiusMax, outerRadius, outerRadiusMin, outerRadiusMax, bridgeRadius, bridgeRadiusMin, bridgeRadiusMax, innerRadiusMin2S, innerRadiusMax2S, bridgeRadiusMin2S, bridgeRadiusMax2S, outerRadiusMin2S,
            outerRadiusMax2S);

   if(success)
   {
       unsigned int quintupletModuleIndex = atomicAdd(&quintupletsInGPU.nQuintuplets[lowerModuleArray1], 1);
       if(quintupletModuleIndex >= N_MAX_QUINTUPLETS_PER_MODULE)
       {
#ifdef Warnings
           if(quintupletModuleIndex ==  N_MAX_QUINTUPLETS_PER_MODULE)
               printf("Quintuplet excess alert! Module index = %d\n", lowerModuleArray1);
#endif
       }
       else
       {
           if(modulesInGPU.quintupletModuleIndices[lowerModuleArray1] == -1)
           {
                printf("Quintuplets : no memory for module at module index = %d\n", lowerModuleArray1);
           }
           else
           {
                unsigned int quintupletIndex = modulesInGPU.quintupletModuleIndices[lowerModuleArray1] +  quintupletModuleIndex;
#ifdef CUT_VALUE_DEBUG
                addQuintupletToMemory(quintupletsInGPU, innerTripletIndex, outerTripletIndex, lowerModule1, lowerModule2, lowerModule3, lowerModule4, lowerModule5, innerRadius, innerRadiusMin, innerRadiusMax, outerRadius, outerRadiusMin, outerRadiusMax, bridgeRadius, bridgeRadiusMin, bridgeRadiusMax, innerRadiusMin2S, innerRadiusMax2S, bridgeRadiusMin2S, bridgeRadiusMax2S, outerRadiusMin2S, outerRadiusMax2S, quintupletIndex);
#else
                addQuintupletToMemory(quintupletsInGPU, innerTripletIndex, outerTripletIndex, lowerModule1, lowerModule2, lowerModule3, lowerModule4, lowerModule5, quintupletIndex);
#endif

            }
        }
    }
}

__global__ void createQuintupletsInGPU(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::triplets& tripletsInGPU, struct SDL::quintuplets& quintupletsInGPU)
{
    int innerInnerInnerLowerModuleArrayIndex = blockIdx.x * blockDim.x + threadIdx.x; //inner triplet inner segment inner MD

    //no quintuplets can be formed for these folks - no need to run inner kernels for them!

    if(innerInnerInnerLowerModuleArrayIndex >= *modulesInGPU.nLowerModules or modulesInGPU.quintupletModuleIndices[innerInnerInnerLowerModuleArrayIndex] == -1) return; 

    unsigned int nInnerTriplets = min(tripletsInGPU.nTriplets[innerInnerInnerLowerModuleArrayIndex], N_MAX_TRIPLETS_PER_MODULE);
    if(nInnerTriplets == 0) return;

    dim3 nThreads(16,16,1);
    dim3 nBlocks(nInnerTriplets % nThreads.x == 0 ? nInnerTriplets / nThreads.x : nInnerTriplets / nThreads.x + 1, N_MAX_TRIPLETS_PER_MODULE % nThreads.y == 0 ? N_MAX_TRIPLETS_PER_MODULE / nThreads.y : N_MAX_TRIPLETS_PER_MODULE / nThreads.y + 1);

    createQuintupletsFromInnerInnerLowerModule<<<nBlocks,nThreads>>>(modulesInGPU, hitsInGPU, mdsInGPU, segmentsInGPU, tripletsInGPU, quintupletsInGPU, innerInnerInnerLowerModuleArrayIndex, nInnerTriplets);
   
}

#endif

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

unsigned int SDL::Event::getNumberOfPixelTracklets()
{
#ifdef Explicit_Tracklet
    unsigned int nLowerModules;// = *(SDL::modulesInGPU->nLowerModules);
    cudaMemcpy(&nLowerModules,modulesInGPU->nLowerModules,sizeof(unsigned int),cudaMemcpyDeviceToHost);
    unsigned int nTrackletsInPixelModule;
    cudaMemcpy(&nTrackletsInPixelModule,&trackletsInGPU->nTracklets[nLowerModules],sizeof(unsigned int),cudaMemcpyDeviceToHost);
    return nTrackletsInPixelModule;
#else
    return trackletsInGPU->nTracklets[*(modulesInGPU->nLowerModules)];
#endif

}

unsigned int SDL::Event::getNumberOfTracklets()
{
    unsigned int tracklets = 0;
    for(auto &it:n_tracklets_by_layer_barrel_)
    {
        tracklets += it;
    }
    for(auto &it:n_tracklets_by_layer_endcap_)
    {
        tracklets += it;
    }

    return tracklets;

}

unsigned int SDL::Event::getNumberOfTrackletsByLayer(unsigned int layer)
{
    if(layer == 6)
        return n_tracklets_by_layer_barrel_[layer];
    else
        return n_tracklets_by_layer_barrel_[layer] + n_tracklets_by_layer_endcap_[layer];
}

unsigned int SDL::Event::getNumberOfTrackletsByLayerBarrel(unsigned int layer)
{
    return n_tracklets_by_layer_barrel_[layer];
}

unsigned int SDL::Event::getNumberOfTrackletsByLayerEndcap(unsigned int layer)
{
    return n_tracklets_by_layer_endcap_[layer];
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
    unsigned int trackCandidates = 0;
    for(auto &it:n_trackCandidates_by_layer_barrel_)
    {
        trackCandidates += it;
    }
    for(auto &it:n_trackCandidates_by_layer_endcap_)
    {
        trackCandidates += it;
    }

    //hack - add pixel track candidate multiplicity
    trackCandidates += getNumberOfPixelTrackCandidates();

    return trackCandidates;

}

unsigned int SDL::Event::getNumberOfPixelTrackCandidates()
{
#ifdef Explicit_Track
    unsigned int nLowerModules;// = *(SDL::modulesInGPU->nLowerModules);
    cudaMemcpy(&nLowerModules,modulesInGPU->nLowerModules,sizeof(unsigned int),cudaMemcpyDeviceToHost);
    unsigned int nTrackCandidatesInPixelModule;
    cudaMemcpy(&nTrackCandidatesInPixelModule,&trackCandidatesInGPU->nTrackCandidates[nLowerModules],sizeof(unsigned int),cudaMemcpyDeviceToHost);
    return nTrackCandidatesInPixelModule;
#else
    return trackCandidatesInGPU->nTrackCandidates[*(modulesInGPU->nLowerModules)];
#endif

}
unsigned int SDL::Event::getNumberOfTrackCandidatesByLayer(unsigned int layer)
{
    if(layer == 6)
        return n_trackCandidates_by_layer_barrel_[layer];
    else
        return n_trackCandidates_by_layer_barrel_[layer] + n_tracklets_by_layer_endcap_[layer];
}

unsigned int SDL::Event::getNumberOfTrackCandidatesByLayerBarrel(unsigned int layer)
{
    return n_trackCandidates_by_layer_barrel_[layer];
}

unsigned int SDL::Event::getNumberOfTrackCandidatesByLayerEndcap(unsigned int layer)
{
    return n_trackCandidates_by_layer_endcap_[layer];
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
        cudaMemcpy(segmentsInCPU->mdIndices, segmentsInGPU->mdIndices, 2 * nMemoryLocations * sizeof(unsigned int), cudaMemcpyDeviceToHost);
        cudaMemcpy(segmentsInCPU->nSegments, segmentsInGPU->nSegments, nModules * sizeof(unsigned int), cudaMemcpyDeviceToHost);
        cudaMemcpy(segmentsInCPU->innerMiniDoubletAnchorHitIndices, segmentsInGPU->innerMiniDoubletAnchorHitIndices, nMemoryLocations * sizeof(unsigned int), cudaMemcpyDeviceToHost);
        cudaMemcpy(segmentsInCPU->outerMiniDoubletAnchorHitIndices, segmentsInGPU->outerMiniDoubletAnchorHitIndices, nMemoryLocations * sizeof(unsigned int), cudaMemcpyDeviceToHost);
        cudaMemcpy(segmentsInCPU->ptIn, segmentsInGPU->ptIn, N_MAX_PIXEL_SEGMENTS_PER_MODULE * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(segmentsInCPU->eta, segmentsInGPU->eta, N_MAX_PIXEL_SEGMENTS_PER_MODULE * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(segmentsInCPU->phi, segmentsInGPU->phi, N_MAX_PIXEL_SEGMENTS_PER_MODULE * sizeof(float), cudaMemcpyDeviceToHost);


    }
    return segmentsInCPU;
}
#else
SDL::segments* SDL::Event::getSegments()
{
    return segmentsInGPU;
}
#endif

#ifdef Explicit_Tracklet
SDL::tracklets* SDL::Event::getTracklets()
{
    if(trackletsInCPU == nullptr)
    {
        unsigned int nLowerModules;
        trackletsInCPU = new SDL::tracklets;
        cudaMemcpy(&nLowerModules, modulesInGPU->nLowerModules, sizeof(unsigned int), cudaMemcpyDeviceToHost);
        unsigned int nMemoryLocations = (N_MAX_TRACKLETS_PER_MODULE) * nLowerModules + N_MAX_PIXEL_TRACKLETS_PER_MODULE;
        trackletsInCPU->segmentIndices = new unsigned int[2 * nMemoryLocations];
        trackletsInCPU->nTracklets = new unsigned int[nLowerModules];
        trackletsInCPU->betaIn = new float[nMemoryLocations];
        trackletsInCPU->betaOut = new float[nMemoryLocations];
        cudaMemcpy(trackletsInCPU->segmentIndices, trackletsInGPU->segmentIndices, 2 * nMemoryLocations * sizeof(unsigned int), cudaMemcpyDeviceToHost);
        cudaMemcpy(trackletsInCPU->betaIn, trackletsInGPU->betaIn, nMemoryLocations * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(trackletsInCPU->betaOut, trackletsInGPU->betaOut, nMemoryLocations * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(trackletsInCPU->nTracklets, trackletsInGPU->nTracklets, (nLowerModules + 1)* sizeof(unsigned int), cudaMemcpyDeviceToHost);
    }
    return trackletsInCPU;
}
#else
SDL::tracklets* SDL::Event::getTracklets()
{
    return trackletsInGPU;
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
        cudaMemcpy(tripletsInCPU->segmentIndices, tripletsInGPU->segmentIndices, 2 * nMemoryLocations * sizeof(unsigned int), cudaMemcpyDeviceToHost);
        cudaMemcpy(tripletsInCPU->betaIn, tripletsInGPU->betaIn, nMemoryLocations * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(tripletsInCPU->betaOut, tripletsInGPU->betaOut, nMemoryLocations * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(tripletsInCPU->nTriplets, tripletsInGPU->nTriplets, nLowerModules * sizeof(unsigned int), cudaMemcpyDeviceToHost);
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
        cudaMemcpy(quintupletsInCPU->nQuintuplets, quintupletsInGPU->nQuintuplets,  nLowerModules * sizeof(unsigned int), cudaMemcpyDeviceToHost);
        cudaMemcpy(quintupletsInCPU->tripletIndices, quintupletsInGPU->tripletIndices, 2 * nMemoryLocations * sizeof(unsigned int), cudaMemcpyDeviceToHost);    
        cudaMemcpy(quintupletsInCPU->lowerModuleIndices, quintupletsInGPU->lowerModuleIndices, 5 * nMemoryLocations * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    }  

    return quintupletsInCPU;
}
#else
SDL::quintuplets* SDL::Event::getQuintuplets()
{
    return quintupletsInGPU;
}
#endif

#ifdef Explicit_Track
SDL::trackCandidates* SDL::Event::getTrackCandidates()
{
    if(trackCandidatesInCPU == nullptr)
    {
        trackCandidatesInCPU = new SDL::trackCandidates;
        unsigned int nLowerModules;
        cudaMemcpy(&nLowerModules, modulesInGPU->nLowerModules, sizeof(unsigned int), cudaMemcpyDeviceToHost);
        unsigned int nEligibleModules;
        cudaMemcpy(&nEligibleModules, modulesInGPU->nEligibleModules, sizeof(unsigned int), cudaMemcpyDeviceToHost);
        unsigned int nMemoryLocations = (N_MAX_TRACK_CANDIDATES_PER_MODULE) * (nEligibleModules -1) + (N_MAX_PIXEL_TRACK_CANDIDATES_PER_MODULE);

        trackCandidatesInCPU->objectIndices = new unsigned int[2 * nMemoryLocations];
        trackCandidatesInCPU->trackCandidateType = new short[nMemoryLocations];
        trackCandidatesInCPU->nTrackCandidates = new unsigned int[nLowerModules+1];
        cudaMemcpy(trackCandidatesInCPU->objectIndices, trackCandidatesInGPU->objectIndices, 2 * nMemoryLocations * sizeof(unsigned int), cudaMemcpyDeviceToHost);
        cudaMemcpy(trackCandidatesInCPU->trackCandidateType, trackCandidatesInGPU->trackCandidateType, nMemoryLocations * sizeof(short), cudaMemcpyDeviceToHost);
        cudaMemcpy(trackCandidatesInCPU->nTrackCandidates, trackCandidatesInGPU->nTrackCandidates, (nLowerModules + 1) * sizeof(unsigned int), cudaMemcpyDeviceToHost);
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
    modulesInCPUFull->trackletRanges = new int[2*nModules];
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
    cudaMemcpy(modulesInCPUFull->trackletRanges, modulesInGPU->trackletRanges, 2*nModules * sizeof(int), cudaMemcpyDeviceToHost);
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

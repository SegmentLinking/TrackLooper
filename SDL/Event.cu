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
const unsigned int N_MAX_PIXEL_TRACK_CANDIDATES_PER_MODULE = 2000000;


struct SDL::modules* SDL::modulesInGPU = nullptr;
struct SDL::modules* SDL::modulesInHost = nullptr;//explicit
unsigned int SDL::nModules;

SDL::Event::Event()
{
    hitsInGPU = nullptr;
    mdsInGPU = nullptr;
    segmentsInGPU = nullptr;
    trackletsInGPU = nullptr;
    tripletsInGPU = nullptr;
    trackCandidatesInGPU = nullptr;
    //reset the arrays
    for(int i = 0; i<6; i++)
    {
        n_hits_by_layer_barrel_[i] = 0;
        n_minidoublets_by_layer_barrel_[i] = 0;
        n_segments_by_layer_barrel_[i] = 0;
        n_tracklets_by_layer_barrel_[i] = 0;
        n_triplets_by_layer_barrel_[i] = 0;
        n_trackCandidates_by_layer_barrel_[i] = 0;
        if(i<5)
        {
            n_hits_by_layer_endcap_[i] = 0;
            n_minidoublets_by_layer_endcap_[i] = 0;
    	    n_segments_by_layer_endcap_[i] = 0;
            n_tracklets_by_layer_endcap_[i] = 0;
            n_triplets_by_layer_endcap_[i] = 0;
            n_trackCandidates_by_layer_endcap_[i] = 0;
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
#else
    mdsInGPU->freeMemory();
    segmentsInGPU->freeMemory();
    tripletsInGPU->freeMemory();
    trackletsInGPU->freeMemory();
    trackCandidatesInGPU->freeMemory();
#endif
    cudaFreeHost(mdsInGPU);
    cudaFreeHost(segmentsInGPU);
    cudaFreeHost(tripletsInGPU);
    cudaFreeHost(trackletsInGPU);
    cudaFreeHost(trackCandidatesInGPU);
    hitsInGPU->freeMemory();
    cudaFreeHost(hitsInGPU);
}

void SDL::initModules()
{
    cudaMallocManaged(&modulesInGPU, sizeof(struct SDL::modules));
    if((modulesInGPU->detIds) == nullptr) //check for nullptr and create memory
    {
        loadModulesFromFile(*modulesInGPU,nModules); //nModules gets filled here
    }
    resetObjectRanges(*modulesInGPU,nModules);
}

void SDL::cleanModules()
{
  freeModulesInUnifiedMemory(*modulesInGPU);
  cudaFree(modulesInGPU);
}

void SDL::Event::resetObjectsInModule()
{
    resetObjectRanges(*modulesInGPU,nModules);
}
// add hits via kernel method. 
void SDL::Event::addHitToEventGPU(std::vector<float> x, std::vector<float> y, std::vector<float> z, std::vector<unsigned int> detId)
{

    const int loopsize = x.size();
    if(hitsInGPU == nullptr)
    {

        cudaMallocHost(&hitsInGPU, sizeof(SDL::hits));
    	  createHitsInExplicitMemory(*hitsInGPU, 2*loopsize);
    }

    //calls the addHitToMemory function
    ////Explicit
    unsigned int nThreads = 256;
    unsigned int nBlocks =  loopsize % nThreads == 0 ? loopsize/nThreads : loopsize/nThreads + 1;

    float* dev_x;
    float* dev_y;
    float* dev_z;
    float* dev_phi;
    float* host_x = &x[0];
    float* host_y = &y[0];
    float* host_z = &z[0];
    float* host_phi;
    float* host_highEdgeXs;
    float* host_highEdgeYs;
    float* host_lowEdgeXs;
    float* host_lowEdgeYs;
    cudaMallocHost(&host_highEdgeXs,sizeof(float)*loopsize);
    cudaMallocHost(&host_highEdgeYs,sizeof(float)*loopsize);
    cudaMallocHost(&host_lowEdgeXs,sizeof(float)*loopsize);
    cudaMallocHost(&host_lowEdgeYs,sizeof(float)*loopsize);
    unsigned int* host_detId = &detId[0];
    unsigned int* host_moduleIndex;
    unsigned int* dev_moduleIndex;
    cudaMalloc(&dev_x,loopsize*sizeof(float));
    cudaMalloc(&dev_y,loopsize*sizeof(float));
    cudaMalloc(&dev_z,loopsize*sizeof(float));
    cudaMalloc(&dev_moduleIndex,sizeof(unsigned int)*loopsize);
    cudaMallocHost(&host_moduleIndex,sizeof(unsigned int)*loopsize);
    cudaMalloc(&dev_phi,sizeof(float)*loopsize);
    cudaMallocHost(&host_phi,sizeof(float)*loopsize);
  for (int ihit=0; ihit<loopsize;ihit++){
    unsigned int moduleLayer = modulesInGPU->layers[(*detIdToIndex)[host_detId[ihit]]]; // I think detIdToIndex needs to be handled on host. this can be run in parallel otherwise
    unsigned int subdet = modulesInGPU->subdets[(*detIdToIndex)[host_detId[ihit]]];
    host_moduleIndex[ihit] = (*detIdToIndex)[host_detId[ihit]];
    host_phi[ihit] = endcapGeometry.getCentroidPhi(host_detId[ihit]);

    if(subdet == Barrel)
    {
        n_hits_by_layer_barrel_[moduleLayer-1]++;
    }
    else
    {
        n_hits_by_layer_endcap_[moduleLayer-1]++;
    }
      unsigned int this_index = host_moduleIndex[ihit];
      if(modulesInGPU->subdets[this_index] == Endcap && modulesInGPU->moduleType[this_index] == TwoS) // cannot be run in parallel
      {
          float xhigh, yhigh, xlow, ylow;
          getEdgeHits(host_detId[ihit],host_x[ihit],host_y[ihit],xhigh,yhigh,xlow,ylow);
          host_highEdgeXs[ihit] = xhigh;
          host_highEdgeYs[ihit] = yhigh;
          host_lowEdgeXs[ihit] = xlow;
          host_lowEdgeYs[ihit] = ylow;

      }

      //set the hit ranges appropriately in the modules struct

      //start the index rolling if the module is encountered for the first time
      if(modulesInGPU->hitRanges[this_index * 2] == -1) // cannot be run in parallel
      {
          modulesInGPU->hitRanges[this_index * 2] = ihit;
      }
      //always update the end index
      modulesInGPU->hitRanges[this_index * 2 + 1] = ihit;
  }
    cudaMemcpy(dev_x,host_x,loopsize*sizeof(float),cudaMemcpyHostToDevice); 
    cudaMemcpy(dev_y,host_y,loopsize*sizeof(float),cudaMemcpyHostToDevice); 
    cudaMemcpy(dev_z,host_z,loopsize*sizeof(float),cudaMemcpyHostToDevice); 
    cudaMemcpy(dev_moduleIndex,host_moduleIndex,loopsize*sizeof(unsigned int),cudaMemcpyHostToDevice); 
    cudaMemcpy(dev_phi,host_phi,loopsize*sizeof(float),cudaMemcpyHostToDevice); 
    //cudaDeviceSynchronize();
    addHitToMemoryKernel<<<nBlocks,nThreads>>>(*hitsInGPU, *modulesInGPU, dev_x, dev_y, dev_z, dev_moduleIndex,dev_phi,loopsize);
    cudaMemcpy(hitsInGPU->highEdgeXs,host_highEdgeXs,loopsize*sizeof(float),cudaMemcpyHostToDevice); 
    cudaMemcpy(hitsInGPU->highEdgeYs,host_highEdgeYs,loopsize*sizeof(float),cudaMemcpyHostToDevice); 
    cudaMemcpy(hitsInGPU->lowEdgeXs,host_lowEdgeXs,loopsize*sizeof(float),cudaMemcpyHostToDevice); 
    cudaMemcpy(hitsInGPU->lowEdgeYs,host_lowEdgeYs,loopsize*sizeof(float),cudaMemcpyHostToDevice); 
    cudaMemcpy(hitsInGPU->nHits,&loopsize,sizeof(unsigned int),cudaMemcpyHostToDevice);// value can't correctly be set in hit allocation 
    cudaDeviceSynchronize();
    cudaFree(dev_x);
    cudaFree(dev_y);
    cudaFree(dev_z);
    cudaFree(dev_moduleIndex);
    cudaFree(dev_phi);
    cudaFreeHost(host_phi);
    cudaFreeHost(host_moduleIndex);
    cudaFreeHost(host_highEdgeXs);
    cudaFreeHost(host_highEdgeYs);
    cudaFreeHost(host_lowEdgeXs);
    cudaFreeHost(host_lowEdgeYs);

}
//explicit method using omp
void SDL::Event::addHitToEventOMP(std::vector<float> x, std::vector<float> y, std::vector<float> z, std::vector<unsigned int> detId)
{
    const int loopsize = x.size();// use the actual number of hits instead of a "max"

    if(hitsInGPU == nullptr)
    {

        cudaMallocHost(&hitsInGPU, sizeof(SDL::hits));
    	  createHitsInExplicitMemory(*hitsInGPU, 2*loopsize); //unclear why but this has to be 2*loopsize to avoid crashing later (reported in tracklet allocation). seems to do with nHits values as well. this allows nhits to be set to the correct value of loopsize to get correct results without crashing. still beats the "max hits" so i think this is fine.  
    }


    float* host_x = &x[0]; // convert from std::vector to host array easily since vectors are ordered
    float* host_y = &y[0];
    float* host_z = &z[0];
    float* host_phis;
    unsigned int* host_detId = &detId[0];
    unsigned int* host_moduleIndex;
    float* host_rts;
    float* host_idxs;
    float* host_highEdgeXs;
    float* host_highEdgeYs;
    float* host_lowEdgeXs;
    float* host_lowEdgeYs;
    cudaMallocHost(&host_moduleIndex,sizeof(unsigned int)*loopsize);
    cudaMallocHost(&host_phis,sizeof(float)*loopsize);
    cudaMallocHost(&host_rts,sizeof(float)*loopsize);
    cudaMallocHost(&host_idxs,sizeof(unsigned int)*loopsize);
    cudaMallocHost(&host_highEdgeXs,sizeof(float)*loopsize);
    cudaMallocHost(&host_highEdgeYs,sizeof(float)*loopsize);
    cudaMallocHost(&host_lowEdgeXs,sizeof(float)*loopsize);
    cudaMallocHost(&host_lowEdgeYs,sizeof(float)*loopsize);
    
//#pragma omp parallel for  // this part can be run in parallel.
  for (int ihit=0; ihit<loopsize;ihit++){
    unsigned int moduleLayer = modulesInGPU->layers[(*detIdToIndex)[host_detId[ihit]]];
    unsigned int subdet = modulesInGPU->subdets[(*detIdToIndex)[host_detId[ihit]]];
    host_moduleIndex[ihit] = (*detIdToIndex)[host_detId[ihit]];

    if(subdet == Barrel)
    {
        n_hits_by_layer_barrel_[moduleLayer-1]++;
    }
    else
    {
        n_hits_by_layer_endcap_[moduleLayer-1]++;
    }
  

      host_rts[ihit] = sqrt(host_x[ihit]*host_x[ihit] + host_y[ihit]*host_y[ihit]);
      host_phis[ihit] = phi(host_x[ihit],host_y[ihit],host_z[ihit]);
      host_idxs[ihit] = ihit;
//  }
//// This part i think has a race condition. so this is not run in parallel. 
////#pragma omp parallel for
//  for (int ihit=0; ihit<loopsize;ihit++){
      unsigned int this_index = host_moduleIndex[ihit];
      if(modulesInGPU->subdets[this_index] == Endcap && modulesInGPU->moduleType[this_index] == TwoS)
      {
          float xhigh, yhigh, xlow, ylow;
          getEdgeHits(host_detId[ihit],host_x[ihit],host_y[ihit],xhigh,yhigh,xlow,ylow);
          host_highEdgeXs[ihit] = xhigh;
          host_highEdgeYs[ihit] = yhigh;
          host_lowEdgeXs[ihit] = xlow;
          host_lowEdgeYs[ihit] = ylow;

      }

      //set the hit ranges appropriately in the modules struct

      //start the index rolling if the module is encountered for the first time
      if(modulesInGPU->hitRanges[this_index * 2] == -1)
      {
          modulesInGPU->hitRanges[this_index * 2] = ihit;
      }
      //always update the end index
      modulesInGPU->hitRanges[this_index * 2 + 1] = ihit;

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
    cudaDeviceSynchronize(); //doesn't seem to make a difference

    cudaFreeHost(host_rts);
    cudaFreeHost(host_idxs);
    cudaFreeHost(host_phis);
    cudaFreeHost(host_moduleIndex);
    cudaFreeHost(host_highEdgeXs);
    cudaFreeHost(host_highEdgeYs);
    cudaFreeHost(host_lowEdgeXs);
    cudaFreeHost(host_lowEdgeYs);

}
// old method using unified memory
void SDL::Event::addHitToEvent(float x, float y, float z, unsigned int detId, unsigned int idx)
{
    const int HIT_MAX = 1000000;
    const int HIT_2S_MAX = 100000;

    if(hitsInGPU == nullptr)
    {

        cudaMallocHost(&hitsInGPU, sizeof(SDL::hits));
        createHitsInUnifiedMemory(*hitsInGPU,HIT_MAX,HIT_2S_MAX);
    }

    //calls the addHitToMemory function
    addHitToMemory(*hitsInGPU, *modulesInGPU, x, y, z, detId, idx);

    unsigned int moduleLayer = modulesInGPU->layers[(*detIdToIndex)[detId]];
    unsigned int subdet = modulesInGPU->subdets[(*detIdToIndex)[detId]];

    if(subdet == Barrel)
    {
        n_hits_by_layer_barrel_[moduleLayer-1]++;
    }
    else if(subdet == Endcap)
    {
        n_hits_by_layer_endcap_[moduleLayer-1]++;
    }

}
void /*unsigned int*/ SDL::Event::addPixToEvent(float x, float y, float z, unsigned int detId, unsigned int idx)
{
    const int HIT_MAX = 1000000;
    const int HIT_2S_MAX = 100000;

    if(hitsInGPU == nullptr)
    {

        cudaMallocHost(&hitsInGPU, sizeof(SDL::hits));
        createHitsInUnifiedMemory(*hitsInGPU,HIT_MAX,HIT_2S_MAX);
        //createHitsInExplicitMemory(*hitsInGPU,229866);
    }

    //calls the addHitToMemory function
    unsigned int moduleIndex = (*detIdToIndex)[detId];
    float phis = phi(x,y,z);
    addHitToMemoryGPU<<<1,1>>>(*hitsInGPU, *modulesInGPU, x, y, z, detId, idx, moduleIndex,phis);
    //addHitToMemory(*hitsInGPU, *modulesInGPU, x, y, z, detId, idx);

    unsigned int moduleLayer = modulesInGPU->layers[(*detIdToIndex)[detId]];
    unsigned int subdet = modulesInGPU->subdets[(*detIdToIndex)[detId]];

    if(subdet == Barrel)
    {
        n_hits_by_layer_barrel_[moduleLayer-1]++;
    }
    else if(subdet == Endcap)
    {
        n_hits_by_layer_endcap_[moduleLayer-1]++;
    }
//    unsigned int* hitIdx;
//    cudaMallocHost(&hitIdx,sizeof(unsigned int));
//    cudaMemcpy(&hitIdx,hitsInGPU->nHits,sizeof(unsigned int),cudaMemcpyDeviceToHost);
//    printf("hit %u\n",hitIdx[0]);
//   // unsigned int dummy = *hitIdx;//so i can return the value and still free?
//    //cudaFreeHost(hitIdx);
//    return 0;
}

__global__ void /*SDL::Event::*/addPixelSegmentToEventKernel(unsigned int* hitIndices, float dPhiChange, float ptIn, float ptErr, float px, float py, float pz, float etaErr,unsigned int pixelModuleIndex, struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU)
{

    //step 1 : Add pixel MDs
    unsigned int innerMDIndex = pixelModuleIndex * N_MAX_MD_PER_MODULES + mdsInGPU.nMDs[pixelModuleIndex];

    //FIXME:Fake Pixel MDs are being added to MD unified memory!
    addMDToMemory(mdsInGPU, hitsInGPU, modulesInGPU, hitIndices[0], hitIndices[1], pixelModuleIndex, 0,0,0,0,0,0,0,0,0,innerMDIndex);
    mdsInGPU.nMDs[pixelModuleIndex]++;
    unsigned int outerMDIndex = pixelModuleIndex * N_MAX_MD_PER_MODULES + mdsInGPU.nMDs[pixelModuleIndex];
    addMDToMemory(mdsInGPU, hitsInGPU, modulesInGPU, hitIndices[2], hitIndices[3], pixelModuleIndex, 0,0,0,0,0,0,0,0,0,outerMDIndex);
    mdsInGPU.nMDs[pixelModuleIndex]++;

    //step 2 : Add pixel segment
    unsigned int pixelSegmentIndex = pixelModuleIndex * N_MAX_SEGMENTS_PER_MODULE + segmentsInGPU.nSegments[pixelModuleIndex];
    //FIXME:Fake Pixel Segment gets added to Segment unified memory in a convoluted fashion!
    addPixelSegmentToMemory(segmentsInGPU, mdsInGPU, hitsInGPU, modulesInGPU, innerMDIndex, outerMDIndex, pixelModuleIndex, hitIndices[0], hitIndices[2], dPhiChange, ptIn, ptErr, px, py, pz, etaErr, pixelSegmentIndex);
    segmentsInGPU.nSegments[pixelModuleIndex]++;
}
void SDL::Event::addPixelSegmentToEvent(std::vector<unsigned int> hitIndices, float dPhiChange, float ptIn, float ptErr, float px, float py, float pz, float etaErr)
{
    assert(hitIndices.size() == 4);
    unsigned int pixelModuleIndex = (*detIdToIndex)[1] -1; //double check the -1?
  unsigned int* hitIndices_host = &hitIndices[0];
  unsigned int * hitIndices_dev;
  cudaMalloc(&hitIndices_dev,4*sizeof(unsigned int));
  cudaMemcpy(hitIndices_dev,hitIndices_host,4*sizeof(unsigned int),cudaMemcpyHostToDevice);

  addPixelSegmentToEventKernel<<<1,1>>>(hitIndices_dev,dPhiChange,ptIn,ptErr,px,py,pz,etaErr,pixelModuleIndex, *modulesInGPU,*hitsInGPU,*mdsInGPU,*segmentsInGPU);
  cudaDeviceSynchronize();
  cudaFree(hitIndices_dev); // added by tres
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
unsigned int nLowerModules = *(SDL::modulesInGPU->nModules);
unsigned int* nMDsCPU;
cudaMallocHost(&nMDsCPU, nLowerModules * sizeof(unsigned int));
cudaMemcpy(nMDsCPU,mdsInGPU->nMDs,nLowerModules*sizeof(unsigned int),cudaMemcpyDeviceToHost);
    unsigned int idx;
    for(unsigned int i = 0; i<*(SDL::modulesInGPU->nLowerModules); i++)
    {
        idx = SDL::modulesInGPU->lowerModuleIndices[i];
        if(nMDsCPU[idx] == 0 or modulesInGPU->hitRanges[idx * 2] == -1)
        {
            modulesInGPU->mdRanges[idx * 2] = -1;
            modulesInGPU->mdRanges[idx * 2 + 1] = -1;
        }
        else
        {
            modulesInGPU->mdRanges[idx * 2] = idx * N_MAX_MD_PER_MODULES;
            modulesInGPU->mdRanges[idx * 2 + 1] = (idx * N_MAX_MD_PER_MODULES) + nMDsCPU[idx] - 1;

            if(modulesInGPU->subdets[idx] == Barrel)
            {
                n_minidoublets_by_layer_barrel_[modulesInGPU->layers[idx] -1] += nMDsCPU[idx];
            }
            else
            {
                n_minidoublets_by_layer_endcap_[modulesInGPU->layers[idx] - 1] += nMDsCPU[idx];
            }

        }
    }
cudaFreeHost(nMDsCPU);
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
unsigned int nLowerModules = *(SDL::modulesInGPU->nModules);
unsigned int* nSegmentsCPU;
cudaMallocHost(&nSegmentsCPU, nLowerModules * sizeof(unsigned int));
cudaMemcpy(nSegmentsCPU,segmentsInGPU->nSegments,nLowerModules*sizeof(unsigned int),cudaMemcpyDeviceToHost);
    unsigned int idx;
    for(unsigned int i = 0; i<*(SDL::modulesInGPU->nLowerModules); i++)
    {
        idx = SDL::modulesInGPU->lowerModuleIndices[i];
        if(nSegmentsCPU[idx] == 0)
        {
            modulesInGPU->segmentRanges[idx * 2] = -1;
            modulesInGPU->segmentRanges[idx * 2 + 1] = -1;
        }
        else
        {
            modulesInGPU->segmentRanges[idx * 2] = idx * N_MAX_SEGMENTS_PER_MODULE;
            modulesInGPU->segmentRanges[idx * 2 + 1] = idx * N_MAX_SEGMENTS_PER_MODULE + nSegmentsCPU[idx] - 1;

            //for(unsigned int jdx = 0; jdx < segmentsInGPU->nSegments[idx]; jdx++)
            //    printSegment(*segmentsInGPU, *mdsInGPU, *hitsInGPU, *modulesInGPU, idx * N_MAX_SEGMENTS_PER_MODULE + jdx);

            if(modulesInGPU->subdets[idx] == Barrel)
            {

                n_segments_by_layer_barrel_[modulesInGPU->layers[idx] - 1] += nSegmentsCPU[idx];
            }
            else
            {
                n_segments_by_layer_endcap_[modulesInGPU->layers[idx] -1] += nSegmentsCPU[idx];
            }
        }
    }
cudaFreeHost(nSegmentsCPU);
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
//#if defined(Explicit_MD) && !defined(Full_Explicit)
//    cudaMemset(mdsInGPU->nMDs,0,nModules*sizeof(unsigned int));
//#endif
    cudaDeviceSynchronize();
    auto memStop = std::chrono::high_resolution_clock::now();
    auto memDuration = std::chrono::duration_cast<std::chrono::milliseconds>(memStop - memStart); //in milliseconds

    unsigned int nLowerModules = *modulesInGPU->nLowerModules;

    dim3 nThreads(1,16,16);
    dim3 nBlocks((nLowerModules % nThreads.x == 0 ? nLowerModules/nThreads.x : nLowerModules/nThreads.x + 1),(N_MAX_HITS_PER_MODULE % nThreads.y == 0 ? N_MAX_HITS_PER_MODULE/nThreads.y : N_MAX_HITS_PER_MODULE/nThreads.y + 1), (N_MAX_HITS_PER_MODULE % nThreads.z == 0 ? N_MAX_HITS_PER_MODULE/nThreads.z : N_MAX_HITS_PER_MODULE/nThreads.z + 1));
    //std::cout<<nBlocks.x<<" "<<nBlocks.y<<" "<<nBlocks.z<<" "<<std::endl;

//    int nThreads = 1;
//    int nBlocks = nLowerModules % nThreads == 0 ? nLowerModules/nThreads : nLowerModules/nThreads + 1;

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
//#if defined(Explicit_Seg) && !defined(Full_Explicit)
//    cudaMemset(segmentsInGPU->nSegments,0,nModules*sizeof(unsigned int));
//#endif
    unsigned int nLowerModules = *modulesInGPU->nLowerModules;

//    dim3 nThreads(1,16,16);
//    dim3 nBlocks(((nLowerModules * MAX_CONNECTED_MODULES)  % nThreads.x == 0 ? (nLowerModules * MAX_CONNECTED_MODULES)/nThreads.x : (nLowerModules * MAX_CONNECTED_MODULES)/nThreads.x + 1),(N_MAX_MD_PER_MODULES % nThreads.y == 0 ? N_MAX_MD_PER_MODULES/nThreads.y : N_MAX_MD_PER_MODULES/nThreads.y + 1), (N_MAX_MD_PER_MODULES % nThreads.z == 0 ? N_MAX_MD_PER_MODULES/nThreads.z : N_MAX_MD_PER_MODULES/nThreads.z + 1));

    unsigned int nThreads = 1;
    unsigned int nBlocks = nLowerModules % nThreads == 0 ? nLowerModules/nThreads : nLowerModules/nThreads + 1;

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
    unsigned int nLowerModules = *modulesInGPU->nLowerModules;

    if(tripletsInGPU == nullptr)
    {
        cudaMallocHost(&tripletsInGPU, sizeof(SDL::triplets));
#ifdef Explicit_Trips
        createTripletsInExplicitMemory(*tripletsInGPU, N_MAX_TRIPLETS_PER_MODULE, nLowerModules);
#else
        createTripletsInUnifiedMemory(*tripletsInGPU, N_MAX_TRIPLETS_PER_MODULE, nLowerModules);
#endif
    }
//#if defined(Explicit_Trips) && !defined(Full_Explicit)
//    cudaMemset(tripletsInGPU->nTriplets,0,nLowerModules*sizeof(unsigned int));
//#endif

    unsigned int nThreads = 1;
    unsigned int nBlocks = nLowerModules % nThreads == 0 ? nLowerModules/nThreads : nLowerModules/nThreads + 1;
    createTripletsInGPU<<<nBlocks,nThreads>>>(*modulesInGPU, *hitsInGPU, *mdsInGPU, *segmentsInGPU, *tripletsInGPU);

    cudaError_t cudaerr = cudaDeviceSynchronize();
    if(cudaerr != cudaSuccess)
    {
        std::cout<<"sync failed with error : "<<cudaGetErrorString(cudaerr)<<std::endl;
    }
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
    unsigned int nLowerModules = *modulesInGPU->nLowerModules;

    //TRCAKLETS - To conserve memory, we shall be only declaring nLowerModules amount of memory!!!!!!!
    if(trackletsInGPU == nullptr)
    {
        cudaMallocHost(&trackletsInGPU, sizeof(SDL::tracklets));
#ifdef Explicit_Tracklet
        //FIXME:Add memory locations for pixel tracklets
        //createTrackletsInExplicitMemory(*trackletsInGPU, N_MAX_TRACKLETS_PER_MODULE , nLowerModules);
        createTrackletsInExplicitMemory(*trackletsInGPU, N_MAX_TRACKLETS_PER_MODULE , N_MAX_PIXEL_TRACKLETS_PER_MODULE, nLowerModules);
#else
        createTrackletsInUnifiedMemory(*trackletsInGPU, N_MAX_TRACKLETS_PER_MODULE , N_MAX_PIXEL_TRACKLETS_PER_MODULE, nLowerModules);
#endif
    }

    unsigned int nThreads = 1;
    unsigned int nBlocks = nLowerModules % nThreads == 0 ? nLowerModules/nThreads : nLowerModules/nThreads + 1;

    createTrackletsInGPU<<<nBlocks,nThreads>>>(*modulesInGPU, *hitsInGPU, *mdsInGPU, *segmentsInGPU, *trackletsInGPU);

    cudaError_t cudaerr = cudaDeviceSynchronize();
    if(cudaerr != cudaSuccess)
    {
        std::cout<<"sync failed with error : "<<cudaGetErrorString(cudaerr)<<std::endl;

    }
    /*addTrackletsToEvent will be called in the createTrackletsWithAGapWithModuleMap function*/

#if defined(AddObjects)
#ifdef Explicit_Tracklet
    addTrackletsToEventExplicit();
#else
    addTrackletsToEvent();
#endif
#endif

}

void SDL::Event::createPixelTracklets()
{
    unsigned int nLowerModules = *modulesInGPU->nLowerModules;
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
    unsigned int nThreads = 1;
    unsigned int nBlocks = nLowerModules % nThreads == 0 ? nLowerModules/nThreads : nLowerModules/nThreads + 1;

    createPixelTrackletsInGPU<<<nBlocks,nThreads>>>(*modulesInGPU, *hitsInGPU, *mdsInGPU, *segmentsInGPU, *trackletsInGPU);

    cudaError_t cudaerr = cudaDeviceSynchronize();
    if(cudaerr != cudaSuccess)
    {
        std::cout<<"sync failed with error : "<<cudaGetErrorString(cudaerr)<<std::endl;    

    }
//#if defined(AddObjects) && !defined(Full_Explicit)
 //   std::cout<<"Number of pixel tracklets = "<<trackletsInGPU->nTracklets[nLowerModules]<<std::endl;
//#endif

}

void SDL::Event::createTrackCandidates()
{
    unsigned int nLowerModules = *modulesInGPU->nLowerModules + 1; //including the pixel module

    if(trackCandidatesInGPU == nullptr)
    {
        cudaMallocHost(&trackCandidatesInGPU, sizeof(SDL::trackCandidates));
#ifdef Explicit_Track
        createTrackCandidatesInExplicitMemory(*trackCandidatesInGPU, N_MAX_TRACK_CANDIDATES_PER_MODULE, N_MAX_PIXEL_TRACK_CANDIDATES_PER_MODULE, nLowerModules);
#else
        createTrackCandidatesInUnifiedMemory(*trackCandidatesInGPU, N_MAX_TRACK_CANDIDATES_PER_MODULE, N_MAX_PIXEL_TRACK_CANDIDATES_PER_MODULE, nLowerModules);
#endif
    }
//#if defined(Explicit_Track) //&& !defined(Full_Explicit)
//    cudaMemset(trackCandidatesInGPU->nTrackCandidates,0,nLowerModules*sizeof(unsigned int));
//    cudaMemset(trackCandidatesInGPU->nTrackCandidatesT4T4,0,nLowerModules*sizeof(unsigned int));
//    cudaMemset(trackCandidatesInGPU->nTrackCandidatesT4T3,0,nLowerModules*sizeof(unsigned int));
//    cudaMemset(trackCandidatesInGPU->nTrackCandidatesT3T4,0,nLowerModules*sizeof(unsigned int));
//#endif

    unsigned int nThreads = 1;
    unsigned int nBlocks = nLowerModules % nThreads == 0 ? nLowerModules/nThreads : nLowerModules/nThreads + 1;

    createTrackCandidatesInGPU<<<nBlocks,nThreads>>>(*modulesInGPU, *hitsInGPU, *mdsInGPU, *segmentsInGPU, *trackletsInGPU, *tripletsInGPU, *trackCandidatesInGPU);

    cudaError_t cudaerr = cudaDeviceSynchronize();
    if(cudaerr != cudaSuccess)
    {
        std::cout<<"sync failed with error : "<<cudaGetErrorString(cudaerr)<<std::endl;
    }
#if defined(AddObjects)
#ifdef Explicit_Track
    addTrackCandidatesToEventExplicit();
#else
    addTrackCandidatesToEvent();
#endif
#endif

}

void SDL::Event::createTrackletsWithAGapWithModuleMap()
{
    //use the same trackletsInGPU as before if it exists
    unsigned int nLowerModules = *modulesInGPU->nLowerModules;

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
//#if defined(Explicit_Tracklet) && !defined(Full_Explicit)
//    cudaMemset(trackletsInGPU->nTracklets,0,nLowerModules*sizeof(unsigned int));
//#endif

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
unsigned int nLowerModules = *(SDL::modulesInGPU->nLowerModules);
unsigned int* nTrackletsCPU;
cudaMallocHost(&nTrackletsCPU, nLowerModules * sizeof(unsigned int));
cudaMemcpy(nTrackletsCPU,trackletsInGPU->nTracklets,nLowerModules*sizeof(unsigned int),cudaMemcpyDeviceToHost);
    unsigned int idx;
    for(unsigned int i = 0; i<*(SDL::modulesInGPU->nLowerModules); i++)
    {
        idx = SDL::modulesInGPU->lowerModuleIndices[i];
        //tracklets run only on lower modules!!!!!!
        if(nTrackletsCPU[i] == 0)
        {
            modulesInGPU->trackletRanges[idx * 2] = -1;
            modulesInGPU->trackletRanges[idx * 2 + 1] = -1;
        }
        else
        {
            modulesInGPU->trackletRanges[idx * 2] = idx * N_MAX_TRACKLETS_PER_MODULE;
            modulesInGPU->trackletRanges[idx * 2 + 1] = idx * N_MAX_TRACKLETS_PER_MODULE + nTrackletsCPU[i] - 1;


            if(modulesInGPU->subdets[idx] == Barrel)
            {
                n_tracklets_by_layer_barrel_[modulesInGPU->layers[idx] - 1] += nTrackletsCPU[i];
            }
            else
            {
                n_tracklets_by_layer_endcap_[modulesInGPU->layers[idx] - 1] += nTrackletsCPU[i];
            }
        }
    }
cudaFreeHost(nTrackletsCPU);
}

void SDL::Event::addTrackCandidatesToEventExplicit()
{
unsigned int nLowerModules = *(SDL::modulesInGPU->nLowerModules);
unsigned int* nTrackCandidatesCPU;
cudaMallocHost(&nTrackCandidatesCPU, nLowerModules * sizeof(unsigned int));
cudaMemcpy(nTrackCandidatesCPU,trackCandidatesInGPU->nTrackCandidates,nLowerModules*sizeof(unsigned int),cudaMemcpyDeviceToHost);

    unsigned int idx;
    for(unsigned int i = 0; i<*(SDL::modulesInGPU->nLowerModules); i++)
    {
        idx = SDL::modulesInGPU->lowerModuleIndices[i];
    

        if(nTrackCandidatesCPU[i] == 0)
        {
            modulesInGPU->trackCandidateRanges[idx * 2] = -1;
            modulesInGPU->trackCandidateRanges[idx * 2 + 1] = -1;
        }
        else
        {
            modulesInGPU->trackCandidateRanges[idx * 2] = idx * N_MAX_TRACK_CANDIDATES_PER_MODULE;
            modulesInGPU->trackCandidateRanges[idx * 2 + 1] = idx * N_MAX_TRACK_CANDIDATES_PER_MODULE + nTrackCandidatesCPU[i] - 1;
 
            if(modulesInGPU->subdets[idx] == Barrel)
            {
                n_trackCandidates_by_layer_barrel_[modulesInGPU->layers[idx] - 1] += nTrackCandidatesCPU[i];
            }
            else
            {
                n_trackCandidates_by_layer_endcap_[modulesInGPU->layers[idx] - 1] += nTrackCandidatesCPU[i];
            }
        }
    }
cudaFreeHost(nTrackCandidatesCPU);
}
void SDL::Event::addTrackCandidatesToEvent()
{

    unsigned int idx;
    for(unsigned int i = 0; i<*(SDL::modulesInGPU->nLowerModules); i++)
    {
        idx = SDL::modulesInGPU->lowerModuleIndices[i];
    

        if(trackCandidatesInGPU->nTrackCandidates[i] == 0)
        {
            modulesInGPU->trackCandidateRanges[idx * 2] = -1;
            modulesInGPU->trackCandidateRanges[idx * 2 + 1] = -1;
        }
        else
        {
            modulesInGPU->trackCandidateRanges[idx * 2] = idx * N_MAX_TRACK_CANDIDATES_PER_MODULE;
            modulesInGPU->trackCandidateRanges[idx * 2 + 1] = idx * N_MAX_TRACK_CANDIDATES_PER_MODULE + trackCandidatesInGPU->nTrackCandidates[i] - 1;
 
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
unsigned int nLowerModules = *(SDL::modulesInGPU->nLowerModules);
unsigned int* nTripletsCPU;
cudaMallocHost(&nTripletsCPU, nLowerModules * sizeof(unsigned int));
cudaMemcpy(nTripletsCPU,tripletsInGPU->nTriplets,nLowerModules*sizeof(unsigned int),cudaMemcpyDeviceToHost);
    unsigned int idx;
    for(unsigned int i = 0; i<*(SDL::modulesInGPU->nLowerModules); i++)
    {
        idx = SDL::modulesInGPU->lowerModuleIndices[i];
        //tracklets run only on lower modules!!!!!!
        if(nTripletsCPU[i] == 0)
        {
            modulesInGPU->tripletRanges[idx * 2] = -1;
            modulesInGPU->tripletRanges[idx * 2 + 1] = -1;
        }
        else
        {
            modulesInGPU->tripletRanges[idx * 2] = idx * N_MAX_TRIPLETS_PER_MODULE;
            modulesInGPU->tripletRanges[idx * 2 + 1] = idx * N_MAX_TRIPLETS_PER_MODULE + nTripletsCPU[i] - 1;

            if(modulesInGPU->subdets[idx] == Barrel)
            {
                n_triplets_by_layer_barrel_[modulesInGPU->layers[idx] - 1] += nTripletsCPU[i];
            }
            else
            {
                n_triplets_by_layer_endcap_[modulesInGPU->layers[idx] - 1] += nTripletsCPU[i];
            }
        }
    }
cudaFreeHost(nTripletsCPU);
}
__global__ void createMiniDoubletsInGPU(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU)
{
    int lowerModuleArrayIndex = blockIdx.x * blockDim.x + threadIdx.x;
    int lowerHitIndex = blockIdx.y * blockDim.y + threadIdx.y;
    int upperHitIndex = blockIdx.z * blockDim.z + threadIdx.z;
    if(lowerModuleArrayIndex >= (*modulesInGPU.nLowerModules)) return; //extra precaution

    int lowerModuleIndex = modulesInGPU.lowerModuleIndices[lowerModuleArrayIndex];
    int upperModuleIndex = modulesInGPU.partnerModuleIndex(lowerModuleIndex);

    if(modulesInGPU.hitRanges[lowerModuleIndex * 2] == -1) return;
    if(modulesInGPU.hitRanges[upperModuleIndex * 2] == -1) return;
    unsigned int nLowerHits = modulesInGPU.hitRanges[lowerModuleIndex * 2 + 1] - modulesInGPU.hitRanges[lowerModuleIndex * 2] + 1;
    unsigned int nUpperHits = modulesInGPU.hitRanges[upperModuleIndex * 2 + 1] - modulesInGPU.hitRanges[upperModuleIndex * 2] + 1;

    //consider assigining a dummy computation function for these
    if(lowerHitIndex >= nLowerHits) return;
    if(upperHitIndex >= nUpperHits) return;

    unsigned int lowerHitArrayIndex = modulesInGPU.hitRanges[lowerModuleIndex * 2] + lowerHitIndex;
    unsigned int upperHitArrayIndex = modulesInGPU.hitRanges[upperModuleIndex * 2] + upperHitIndex;

    float dz, dphi, dphichange, shiftedX, shiftedY, shiftedZ, noShiftedDz, noShiftedDphi, noShiftedDphiChange;
    bool success = runMiniDoubletDefaultAlgo(modulesInGPU, hitsInGPU, lowerModuleIndex, lowerHitArrayIndex, upperHitArrayIndex, dz, dphi, dphichange, shiftedX, shiftedY, shiftedZ, noShiftedDz, noShiftedDphi, noShiftedDphiChange);

    if(success)
    {
        unsigned int mdModuleIndex = atomicAdd(&mdsInGPU.nMDs[lowerModuleIndex],1);
        unsigned int mdIndex = lowerModuleIndex * N_MAX_MD_PER_MODULES + mdModuleIndex;

        addMDToMemory(mdsInGPU,hitsInGPU, modulesInGPU, lowerHitArrayIndex, upperHitArrayIndex, lowerModuleIndex, dz, dphi, dphichange, shiftedX, shiftedY, shiftedZ, noShiftedDz, noShiftedDphi, noShiftedDphiChange, mdIndex);
    }
}

/*__global__ void createMiniDoubletsFromLowerModule(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, unsigned int lowerModuleIndex, unsigned int upperModuleIndex, unsigned int nLowerHits, unsigned int nUpperHits)
{
    unsigned int lowerHitIndex = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int upperHitIndex = blockIdx.z * blockDim.z + threadIdx.z;

    //consider assigining a dummy computation function for these
    if(lowerHitIndex >= nLowerHits) return;
    if(upperHitIndex >= nUpperHits) return;

    unsigned int lowerHitArrayIndex = modulesInGPU.hitRanges[lowerModuleIndex * 2] + lowerHitIndex;
    unsigned int upperHitArrayIndex = modulesInGPU.hitRanges[upperModuleIndex * 2] + upperHitIndex;

    float dz, dphi, dphichange, shiftedX, shiftedY, shiftedZ, noShiftedDz, noShiftedDphi, noShiftedDphiChange;

    bool success = runMiniDoubletDefaultAlgo(modulesInGPU, hitsInGPU, lowerModuleIndex, lowerHitArrayIndex, upperHitArrayIndex, dz, dphi, dphichange, shiftedX, shiftedY, shiftedZ, noShiftedDz, noShiftedDphi, noShiftedDphiChange);

    if(success)
    {
        unsigned int mdModuleIdx = atomicAdd(&mdsInGPU.nMDs[lowerModuleIndex],1);
        unsigned int mdIdx = lowerModuleIndex * N_MAX_MD_PER_MODULES + mdModuleIdx;

        addMDToMemory(mdsInGPU,hitsInGPU, modulesInGPU, lowerHitArrayIndex, upperHitArrayIndex, lowerModuleIndex, dz, dphi, dphichange, shiftedX, shiftedY, shiftedZ, noShiftedDz, noShiftedDphi, noShiftedDphiChange, mdIdx);
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


}*/

/*__global__ void createSegmentsInGPU(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU)
{
    int xAxisIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int innerMDArrayIdx = blockIdx.y * blockDim.y + threadIdx.y;
    int outerMDArrayIdx = blockIdx.z * blockDim.z + threadIdx.z;

    int innerLowerModuleArrayIdx = xAxisIdx/MAX_CONNECTED_MODULES;
    int outerLowerModuleArrayIdx = xAxisIdx % MAX_CONNECTED_MODULES; //need this index from the connected module array

    unsigned int innerLowerModuleIndex = modulesInGPU.lowerModuleIndices[innerLowerModuleArrayIdx];

    unsigned int nConnectedModules = modulesInGPU.nConnectedModules[innerLowerModuleIndex];

    if(outerLowerModuleArrayIdx >= nConnectedModules) return;

    unsigned int outerLowerModuleIndex = modulesInGPU.moduleMap[innerLowerModuleIndex * MAX_CONNECTED_MODULES + outerLowerModuleArrayIdx];

    unsigned int nInnerMDs = mdsInGPU.nMDs[innerLowerModuleIndex];
    unsigned int nOuterMDs = mdsInGPU.nMDs[outerLowerModuleIndex];

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

    bool success = runSegmentDefaultAlgo(modulesInGPU, hitsInGPU, mdsInGPU, innerLowerModuleIndex, outerLowerModuleIndex, innerMDIndex, outerMDIndex, zIn, zOut, rtIn, rtOut, dPhi, dPhiMin, dPhiMax, dPhiChange, dPhiChangeMin, dPhiChangeMax, dAlphaInnerMDSegment, dAlphaOuterMDSegment, dAlphaInnerMDOuterMD, innerMiniDoubletAnchorHitIndex, outerMiniDoubletAnchorHitIndex);

    if(success)
    {
        unsigned int segmentModuleIdx = atomicAdd(&segmentsInGPU.nSegments[innerLowerModuleIndex],1);
        unsigned int segmentIdx = innerLowerModuleIndex * N_MAX_SEGMENTS_PER_MODULE + segmentModuleIdx;

        addSegmentToMemory(segmentsInGPU,innerMDIndex, outerMDIndex,innerLowerModuleIndex, outerLowerModuleIndex, innerMiniDoubletAnchorHitIndex, outerMiniDoubletAnchorHitIndex, dPhi, dPhiMin, dPhiMax, dPhiChange, dPhiChangeMin, dPhiChangeMax, zIn, zOut, rtIn, rtOut, dAlphaInnerMDSegment, dAlphaOuterMDSegment, dAlphaInnerMDOuterMD,segmentIdx);
    }


}*/


__global__ void createSegmentsFromInnerLowerModule(struct SDL::modules&modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, unsigned int innerLowerModuleIndex, unsigned int nInnerMDs)
{
    unsigned int outerLowerModuleArrayIndex = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int innerMDArrayIndex = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int outerMDArrayIndex = blockIdx.z * blockDim.z + threadIdx.z;

    unsigned int outerLowerModuleIndex = modulesInGPU.moduleMap[innerLowerModuleIndex * MAX_CONNECTED_MODULES + outerLowerModuleArrayIndex];

    unsigned int nOuterMDs = mdsInGPU.nMDs[outerLowerModuleIndex];
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

    bool success = runSegmentDefaultAlgo(modulesInGPU, hitsInGPU, mdsInGPU, innerLowerModuleIndex, outerLowerModuleIndex, innerMDIndex, outerMDIndex, zIn, zOut, rtIn, rtOut, dPhi, dPhiMin, dPhiMax, dPhiChange, dPhiChangeMin, dPhiChangeMax, dAlphaInnerMDSegment, dAlphaOuterMDSegment, dAlphaInnerMDOuterMD, innerMiniDoubletAnchorHitIndex, outerMiniDoubletAnchorHitIndex);

    if(success)
    {
        unsigned int segmentModuleIdx = atomicAdd(&segmentsInGPU.nSegments[innerLowerModuleIndex],1);
        unsigned int segmentIdx = innerLowerModuleIndex * N_MAX_SEGMENTS_PER_MODULE + segmentModuleIdx;

        addSegmentToMemory(segmentsInGPU,innerMDIndex, outerMDIndex,innerLowerModuleIndex, outerLowerModuleIndex, innerMiniDoubletAnchorHitIndex, outerMiniDoubletAnchorHitIndex, dPhi, dPhiMin, dPhiMax, dPhiChange, dPhiChangeMin, dPhiChangeMax, zIn, zOut, rtIn, rtOut, dAlphaInnerMDSegment, dAlphaOuterMDSegment, dAlphaInnerMDOuterMD,segmentIdx);
    }

}

__global__ void createSegmentsInGPU(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU)
{
    int innerLowerModuleArrayIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if(innerLowerModuleArrayIndex >= *modulesInGPU.nLowerModules) return;
    unsigned int innerLowerModuleIndex = modulesInGPU.lowerModuleIndices[innerLowerModuleArrayIndex];
    unsigned int nConnectedModules = modulesInGPU.nConnectedModules[innerLowerModuleIndex];
    unsigned int nInnerMDs = mdsInGPU.nMDs[innerLowerModuleIndex];

    if(nConnectedModules == 0) return;

    if(nInnerMDs == 0) return;
    dim3 nThreads(1,16,16);
    dim3 nBlocks((nConnectedModules % nThreads.x == 0 ? nConnectedModules/nThreads.x : nConnectedModules/nThreads.x + 1), (nInnerMDs % nThreads.y == 0 ? nInnerMDs/nThreads.y : nInnerMDs/nThreads.y + 1), (N_MAX_MD_PER_MODULES % nThreads.z == 0 ? N_MAX_MD_PER_MODULES/nThreads.z : N_MAX_MD_PER_MODULES/nThreads.z + 1));

    createSegmentsFromInnerLowerModule<<<nBlocks,nThreads>>>(modulesInGPU, hitsInGPU, mdsInGPU, segmentsInGPU, innerLowerModuleIndex,nInnerMDs);

}


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

    unsigned int nOuterSegments = segmentsInGPU.nSegments[outerInnerLowerModuleIndex];
    if(outerSegmentArrayIndex >= nOuterSegments) return;

    unsigned int outerSegmentIndex = outerInnerLowerModuleIndex * N_MAX_SEGMENTS_PER_MODULE + outerSegmentArrayIndex;

    //for completeness - outerOuterLowerModuleIndex
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

__global__ void createPixelTrackletsInGPU(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::tracklets& trackletsInGPU)
{
    int outerInnerLowerModuleArrayIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if(outerInnerLowerModuleArrayIndex >= *modulesInGPU.nLowerModules) return;

    unsigned int outerInnerLowerModuleIndex = modulesInGPU.lowerModuleIndices[outerInnerLowerModuleArrayIndex];
    unsigned int pixelModuleIndex = *modulesInGPU.nModules - 1; //last dude
    unsigned int pixelLowerModuleArrayIndex = modulesInGPU.reverseLookupLowerModuleIndices[pixelModuleIndex]; //should be the same as nLowerModules
    unsigned int nInnerSegments = segmentsInGPU.nSegments[pixelModuleIndex];
    unsigned int nOuterSegments = segmentsInGPU.nSegments[outerInnerLowerModuleIndex];
    if(nOuterSegments == 0) return;
    if(nInnerSegments == 0) return;

    dim3 nThreads(16,16,1);
    dim3 nBlocks(nInnerSegments % nThreads.x == 0 ? nInnerSegments / nThreads.x : nInnerSegments / nThreads.x + 1, nOuterSegments % nThreads.y == 0 ? nOuterSegments / nThreads.y : nOuterSegments / nThreads.y + 1, 1);

    createPixelTrackletsFromOuterInnerLowerModule<<<nBlocks,nThreads>>>(modulesInGPU, hitsInGPU, mdsInGPU, segmentsInGPU, trackletsInGPU, outerInnerLowerModuleIndex, nInnerSegments, nOuterSegments, pixelModuleIndex, pixelLowerModuleArrayIndex);

}


__global__ void createPixelTrackletsFromOuterInnerLowerModule(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::tracklets& trackletsInGPU, unsigned int outerInnerLowerModuleIndex, unsigned int nInnerSegments, unsigned int nOuterSegments, unsigned int pixelModuleIndex, unsigned int pixelLowerModuleArrayIndex)
{
    int innerSegmentArrayIndex = blockIdx.x * blockDim.x + threadIdx.x;
    int outerSegmentArrayIndex = blockIdx.y * blockDim.y + threadIdx.y;
    if(innerSegmentArrayIndex >= nInnerSegments) return;
    if(outerSegmentArrayIndex >= nOuterSegments) return;
    unsigned int innerSegmentIndex = pixelModuleIndex * N_MAX_SEGMENTS_PER_MODULE + innerSegmentArrayIndex;
    unsigned int outerSegmentIndex = outerInnerLowerModuleIndex * N_MAX_SEGMENTS_PER_MODULE + outerSegmentArrayIndex;
    unsigned int outerOuterLowerModuleIndex = segmentsInGPU.outerLowerModuleIndices[outerSegmentIndex];

    float zOut, rtOut, deltaPhiPos, deltaPhi, betaIn, betaOut;
   
    bool success = runTrackletDefaultAlgo(modulesInGPU, hitsInGPU, mdsInGPU, segmentsInGPU, pixelModuleIndex, pixelModuleIndex, outerInnerLowerModuleIndex, outerOuterLowerModuleIndex, innerSegmentIndex, outerSegmentIndex, zOut, rtOut, deltaPhiPos, deltaPhi, betaIn, betaOut, N_MAX_SEGMENTS_PER_MODULE); //might want to send the other two module indices and the anchor hits also to save memory accesses

    if(success)
    {
        unsigned int trackletModuleIndex = atomicAdd(&trackletsInGPU.nTracklets[pixelLowerModuleArrayIndex], 1);
        unsigned int trackletIndex = pixelLowerModuleArrayIndex * N_MAX_TRACKLETS_PER_MODULE + trackletModuleIndex;

        addTrackletToMemory(trackletsInGPU,innerSegmentIndex,outerSegmentIndex,pixelModuleIndex,pixelModuleIndex,outerInnerLowerModuleIndex,outerOuterLowerModuleIndex,zOut,rtOut,deltaPhiPos,deltaPhi,betaIn,betaOut,trackletIndex);

    }
}

__global__ void createTrackletsInGPU(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::tracklets& trackletsInGPU)
{
    int innerInnerLowerModuleArrayIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if(innerInnerLowerModuleArrayIndex >= *modulesInGPU.nLowerModules) return;
    unsigned int innerInnerLowerModuleIndex = modulesInGPU.lowerModuleIndices[innerInnerLowerModuleArrayIndex];
    unsigned int nInnerSegments = segmentsInGPU.nSegments[innerInnerLowerModuleIndex];
    if(nInnerSegments == 0) return;

    dim3 nThreads(1,16,16);
    dim3 nBlocks(MAX_CONNECTED_MODULES % nThreads.x  == 0 ? MAX_CONNECTED_MODULES / nThreads.x : MAX_CONNECTED_MODULES / nThreads.x + 1 ,nInnerSegments % nThreads.y == 0 ? nInnerSegments/nThreads.y : nInnerSegments/nThreads.y + 1,N_MAX_SEGMENTS_PER_MODULE % nThreads.z == 0 ? N_MAX_SEGMENTS_PER_MODULE/nThreads.z : N_MAX_SEGMENTS_PER_MODULE/nThreads.z + 1);

    createTrackletsFromInnerInnerLowerModule<<<nBlocks,nThreads>>>(modulesInGPU,hitsInGPU,mdsInGPU,segmentsInGPU,trackletsInGPU,innerInnerLowerModuleIndex,nInnerSegments,innerInnerLowerModuleArrayIndex);

}

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

    unsigned int nOuterSegments = segmentsInGPU.nSegments[outerInnerLowerModuleIndex];
    if(outerSegmentArrayIndex >= nOuterSegments) return;

    unsigned int outerSegmentIndex = outerInnerLowerModuleIndex * N_MAX_SEGMENTS_PER_MODULE + outerSegmentArrayIndex;

    //for completeness - outerOuterLowerModuleIndex
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

__global__ void createTrackletsWithAGapInGPU(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::tracklets& trackletsInGPU)
{
    //outer kernel for proposal 1
    int innerInnerLowerModuleArrayIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if(innerInnerLowerModuleArrayIndex >= *modulesInGPU.nLowerModules) return;
    unsigned int innerInnerLowerModuleIndex = modulesInGPU.lowerModuleIndices[innerInnerLowerModuleArrayIndex];
    unsigned int nInnerSegments = segmentsInGPU.nSegments[innerInnerLowerModuleIndex];
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

__global__ void createTripletsFromInnerInnerLowerModule(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::triplets& tripletsInGPU, unsigned int innerInnerLowerModuleIndex, unsigned int nInnerSegments, unsigned int nConnectedModules, unsigned int innerInnerLowerModuleArrayIndex)
{
    int innerSegmentArrayIndex = blockIdx.x * blockDim.x + threadIdx.x;
    int outerSegmentArrayIndex = blockIdx.y * blockDim.y + threadIdx.y;

    if(innerSegmentArrayIndex >= nInnerSegments) return;

    unsigned int innerSegmentIndex = innerInnerLowerModuleIndex * N_MAX_SEGMENTS_PER_MODULE + innerSegmentArrayIndex;

    //middle lower module - outer lower module of inner segment
    unsigned int middleLowerModuleIndex = segmentsInGPU.outerLowerModuleIndices[innerSegmentIndex];

    unsigned int nOuterSegments = segmentsInGPU.nSegments[middleLowerModuleIndex];
    if(outerSegmentArrayIndex >= nOuterSegments) return;
    unsigned int outerSegmentIndex = middleLowerModuleIndex * N_MAX_SEGMENTS_PER_MODULE + outerSegmentArrayIndex;
    unsigned int outerOuterLowerModuleIndex = segmentsInGPU.outerLowerModuleIndices[outerSegmentIndex];

    float zOut,rtOut,deltaPhiPos,deltaPhi,betaIn,betaOut;

    bool success = runTripletDefaultAlgo(modulesInGPU, hitsInGPU, mdsInGPU, segmentsInGPU, innerInnerLowerModuleIndex, middleLowerModuleIndex, outerOuterLowerModuleIndex, innerSegmentIndex, outerSegmentIndex, zOut, rtOut, deltaPhiPos, deltaPhi, betaIn, betaOut);
    if(success)
    {
        unsigned int tripletModuleIndex = atomicAdd(&tripletsInGPU.nTriplets[innerInnerLowerModuleArrayIndex], 1);
        unsigned int tripletIndex = innerInnerLowerModuleArrayIndex * N_MAX_TRIPLETS_PER_MODULE + tripletModuleIndex;

        addTripletToMemory(tripletsInGPU, innerSegmentIndex, outerSegmentIndex, innerInnerLowerModuleIndex, middleLowerModuleIndex, outerOuterLowerModuleIndex, zOut, rtOut, deltaPhiPos, deltaPhi, betaIn, betaOut, tripletIndex);
    }
}

__global__ void createTripletsInGPU(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::triplets& tripletsInGPU)
{
    int innerInnerLowerModuleArrayIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if(innerInnerLowerModuleArrayIndex >= *modulesInGPU.nLowerModules) return;
    unsigned int innerInnerLowerModuleIndex = modulesInGPU.lowerModuleIndices[innerInnerLowerModuleArrayIndex];
    unsigned int nInnerSegments = segmentsInGPU.nSegments[innerInnerLowerModuleIndex];
    if(nInnerSegments == 0) return;

    unsigned int nConnectedModules = modulesInGPU.nConnectedModules[innerInnerLowerModuleIndex];
    if(nConnectedModules == 0) return;

    dim3 nThreads(16,16,1);
    dim3 nBlocks(nInnerSegments % nThreads.x == 0 ? nInnerSegments / nThreads.x : nInnerSegments / nThreads.x + 1, N_MAX_SEGMENTS_PER_MODULE % nThreads.y == 0 ? N_MAX_SEGMENTS_PER_MODULE / nThreads.y : N_MAX_SEGMENTS_PER_MODULE / nThreads.y + 1);

    createTripletsFromInnerInnerLowerModule<<<nBlocks,nThreads>>>(modulesInGPU, hitsInGPU, mdsInGPU, segmentsInGPU, tripletsInGPU, innerInnerLowerModuleIndex, nInnerSegments, nConnectedModules, innerInnerLowerModuleArrayIndex);
}

__global__ void createTrackCandidatesInGPU(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::tracklets& trackletsInGPU, struct SDL::triplets& tripletsInGPU, struct SDL::trackCandidates& trackCandidatesInGPU)
{
    //inner tracklet/triplet inner segment inner MD lower module
    int innerInnerInnerLowerModuleArrayIndex = blockIdx.x * blockDim.x + threadIdx.x;
    //hack to include pixel detector
    if(innerInnerInnerLowerModuleArrayIndex >= *modulesInGPU.nLowerModules + 1) return; 

    unsigned int nTracklets = trackletsInGPU.nTracklets[innerInnerInnerLowerModuleArrayIndex];
    unsigned int nTriplets = tripletsInGPU.nTriplets[innerInnerInnerLowerModuleArrayIndex]; // should be zero for the pixels

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
	
        if(outerObjectArrayIndex < trackletsInGPU.nTracklets[outerInnerInnerLowerModuleIndex])
        {

            outerObjectIndex = outerInnerInnerLowerModuleIndex * N_MAX_TRACKLETS_PER_MODULE + outerObjectArrayIndex;
            
            success = runTrackCandidateDefaultAlgoTwoTracklets(trackletsInGPU, tripletsInGPU, innerObjectIndex, outerObjectIndex,trackCandidateType);

            if(success)
            {
                unsigned int trackCandidateModuleIdx = atomicAdd(&trackCandidatesInGPU.nTrackCandidates[innerInnerInnerLowerModuleArrayIndex],1);
                unsigned int trackCandidateIdx = innerInnerInnerLowerModuleArrayIndex * N_MAX_TRACK_CANDIDATES_PER_MODULE + trackCandidateModuleIdx;
                atomicAdd(&trackCandidatesInGPU.nTrackCandidatesT4T4[innerInnerInnerLowerModuleArrayIndex],1);
                addTrackCandidateToMemory(trackCandidatesInGPU, trackCandidateType, innerObjectIndex, outerObjectIndex, trackCandidateIdx);
            }

        }
    }
    //step 2 tracklet-triplet
    if(innerObjectArrayIndex < nInnerTracklets)
    {
        innerObjectIndex = innerInnerInnerLowerModuleArrayIndex * N_MAX_TRACKLETS_PER_MODULE + innerObjectArrayIndex;
        unsigned int outerInnerInnerLowerModuleIndex = modulesInGPU.reverseLookupLowerModuleIndices[trackletsInGPU.lowerModuleIndices[4 * innerObjectIndex + 2]];//same as innerOuterInnerLowerModuleIndex
        if(outerObjectArrayIndex < tripletsInGPU.nTriplets[outerInnerInnerLowerModuleIndex])
        {
            outerObjectIndex = outerInnerInnerLowerModuleIndex * N_MAX_TRIPLETS_PER_MODULE + outerObjectArrayIndex;
            success = runTrackCandidateDefaultAlgoTrackletToTriplet(trackletsInGPU, tripletsInGPU, innerObjectIndex, outerObjectIndex,trackCandidateType);
            if(success)
            {
                unsigned int trackCandidateModuleIdx = atomicAdd(&trackCandidatesInGPU.nTrackCandidates[innerInnerInnerLowerModuleArrayIndex],1);
                unsigned int trackCandidateIdx = innerInnerInnerLowerModuleArrayIndex * N_MAX_TRACK_CANDIDATES_PER_MODULE + trackCandidateModuleIdx;

                atomicAdd(&trackCandidatesInGPU.nTrackCandidatesT4T3[innerInnerInnerLowerModuleArrayIndex],1);
                addTrackCandidateToMemory(trackCandidatesInGPU, trackCandidateType, innerObjectIndex, outerObjectIndex, trackCandidateIdx);
            }

        }
    }

    //step 3 triplet-tracklet
    if(innerObjectArrayIndex < nInnerTriplets)
    {
        innerObjectIndex = innerInnerInnerLowerModuleArrayIndex * N_MAX_TRIPLETS_PER_MODULE + innerObjectArrayIndex;
        unsigned int outerInnerInnerLowerModuleIndex = modulesInGPU.reverseLookupLowerModuleIndices[tripletsInGPU.lowerModuleIndices[3 * innerObjectIndex + 1]];//same as innerOuterInnerLowerModuleIndex

        if(outerObjectArrayIndex < trackletsInGPU.nTracklets[outerInnerInnerLowerModuleIndex])
        {
            outerObjectIndex = outerInnerInnerLowerModuleIndex * N_MAX_TRACKLETS_PER_MODULE + outerObjectArrayIndex;
            success = runTrackCandidateDefaultAlgoTripletToTracklet(trackletsInGPU, tripletsInGPU, innerObjectIndex, outerObjectIndex,trackCandidateType);
            if(success)
            {
                unsigned int trackCandidateModuleIdx = atomicAdd(&trackCandidatesInGPU.nTrackCandidates[innerInnerInnerLowerModuleArrayIndex],1);
unsigned int trackCandidateIdx = innerInnerInnerLowerModuleArrayIndex * N_MAX_TRACK_CANDIDATES_PER_MODULE + trackCandidateModuleIdx;
                atomicAdd(&trackCandidatesInGPU.nTrackCandidatesT3T4[innerInnerInnerLowerModuleArrayIndex],1);
                addTrackCandidateToMemory(trackCandidatesInGPU, trackCandidateType, innerObjectIndex, outerObjectIndex, trackCandidateIdx);
            }

 
        }

   } 
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
        return n_triplets_by_layer_barrel_[layer] + n_tracklets_by_layer_endcap_[layer];
}

unsigned int SDL::Event::getNumberOfTripletsByLayerBarrel(unsigned int layer)
{
    return n_triplets_by_layer_barrel_[layer];
}

unsigned int SDL::Event::getNumberOfTripletsByLayerEndcap(unsigned int layer)
{
    return n_triplets_by_layer_endcap_[layer];
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
#ifdef Explicit_Track
    unsigned int nLowerModules = *(SDL::modulesInGPU->nLowerModules);
    unsigned int* nTrackCandidatesCPU;
    cudaMallocHost(&nTrackCandidatesCPU, (nLowerModules+1) * sizeof(unsigned int));
    cudaMemcpy(nTrackCandidatesCPU,trackCandidatesInGPU->nTrackCandidates,(1+nLowerModules)*sizeof(unsigned int),cudaMemcpyDeviceToHost);
    trackCandidates += nTrackCandidatesCPU[nLowerModules];
    cudaFreeHost(nTrackCandidatesCPU);
#else
    trackCandidates += trackCandidatesInGPU->nTrackCandidates[*(modulesInGPU->nLowerModules)];
#endif

    return trackCandidates;
   
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

struct SDL::hits* SDL::Event::getHits()
{
    return hitsInGPU;
}

struct SDL::miniDoublets* SDL::Event::getMiniDoublets()
{
    return mdsInGPU;
}

struct SDL::segments* SDL::Event::getSegments()
{
    return segmentsInGPU;
}

struct SDL::tracklets* SDL::Event::getTracklets()
{
    return trackletsInGPU;
}

struct SDL::triplets* SDL::Event::getTriplets()
{
    return tripletsInGPU;
}

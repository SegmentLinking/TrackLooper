# include "Event.cuh"


const unsigned int N_MAX_HITS_PER_MODULE = 100;
const unsigned int N_MAX_MD_PER_MODULES = 100;
const unsigned int N_MAX_SEGMENTS_PER_MODULE = 600; //WHY!
const unsigned int MAX_CONNECTED_MODULES = 40;
const unsigned int N_MAX_TRACKLETS_PER_MODULE = 5000;//temporary
const unsigned int N_MAX_TRIPLETS_PER_MODULE = 1000;
struct SDL::modules* SDL::modulesInGPU = nullptr;
struct SDL::modules* SDL::modulesInHost = nullptr;//explicit
unsigned int SDL::nModules;

SDL::Event::Event()
{
    hitsInGPU = nullptr;
    mdsInGPU = nullptr;
    mdsInTemp = nullptr; //explicit
    segmentsInGPU = nullptr;
    segmentsInTemp = nullptr;
    trackletsInGPU = nullptr; 
    trackletsInTemp = nullptr; 
    tripletsInGPU = nullptr; 
    tripletsInTemp = nullptr; 
    //reset the arrays
    for(int i = 0; i<6; i++)
    {
        n_hits_by_layer_barrel_[i] = 0;
        n_minidoublets_by_layer_barrel_[i] = 0;
        n_segments_by_layer_barrel_[i] = 0;
        n_tracklets_by_layer_barrel_[i] = 0;
        n_triplets_by_layer_barrel_[i] = 0;
        if(i<5)
        {
            n_hits_by_layer_endcap_[i] = 0;
            n_minidoublets_by_layer_endcap_[i] = 0;
	    n_segments_by_layer_endcap_[i] = 0;
            n_tracklets_by_layer_endcap_[i] = 0;
            n_triplets_by_layer_endcap_[i] = 0;
        }
    }
    resetObjectsInModule();

}

SDL::Event::~Event()
{

#ifdef Explicit_MD
    mdsInTemp->freeMemory();
//    cudaFree(mdsInTemp);
#else
    mdsInGPU->freeMemory();
#endif
#ifdef Explicit_Seg
    segmentsInTemp->freeMemory(); 
//    cudaFree(segmentsInTemp);
#else
    segmentsInGPU->freeMemory(); 
#endif
#ifdef Explicit_Tracklet 
    trackletsInTemp->freeMemory(); 
//    cudaFree(trackletsInTemp);
#else
    trackletsInGPU->freeMemory();
#endif
#ifdef Explicit_Trips
    tripletsInTemp->freeMemory(); 
//    cudaFree(tripletsInTemp);
#else
    tripletsInGPU->freeMemory();
#endif
    hitsInGPU->freeMemory();
    cudaFree(hitsInGPU);
    cudaFree(mdsInGPU);
    cudaFree(segmentsInGPU);
    cudaFree(trackletsInGPU);
    cudaFree(tripletsInGPU);
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

void SDL::Event::resetObjectsInModule()
{
    resetObjectRanges(*modulesInGPU,nModules);
}

void SDL::Event::addHitToEvent(float x, float y, float z, unsigned int detId)
{
    const int HIT_MAX = 1000000;
    const int HIT_2S_MAX = 100000;

    if(hitsInGPU == nullptr)
    {
        cudaMallocManaged(&hitsInGPU, sizeof(SDL::hits));
        createHitsInUnifiedMemory(*hitsInGPU,HIT_MAX,HIT_2S_MAX);
    }

    //calls the addHitToMemory function
    addHitToMemory(*hitsInGPU, *modulesInGPU, x, y, z, detId);

    unsigned int moduleLayer = modulesInGPU->layers[(*detIdToIndex)[detId]];
    unsigned int subdet = modulesInGPU->subdets[(*detIdToIndex)[detId]];

    if(subdet == Barrel)
    {
        n_hits_by_layer_barrel_[moduleLayer-1]++;
    }
    else
    {
        n_hits_by_layer_endcap_[moduleLayer-1]++;
    } 

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

void SDL::Event::createMiniDoublets()
{
    cudaDeviceSynchronize();
    auto memStart = std::chrono::high_resolution_clock::now();
    if(mdsInGPU == nullptr)
    {
#ifdef Explicit_MD
#ifdef Full_Explicit
        cudaMalloc(&mdsInGPU, sizeof(SDL::miniDoublets));
#else
        cudaMallocManaged(&mdsInGPU, sizeof(SDL::miniDoublets));
#endif
        cudaMallocHost(&mdsInTemp, sizeof(SDL::miniDoublets));
    	  createMDsInExplicitMemory(*mdsInGPU,*mdsInTemp, N_MAX_MD_PER_MODULES, nModules);
#else
        cudaMallocManaged(&mdsInGPU, sizeof(SDL::miniDoublets));
    	  createMDsInUnifiedMemory(*mdsInGPU, N_MAX_MD_PER_MODULES, nModules);
#endif
    }
#if defined(Explicit_MD) && !defined(Full_Explicit)
    cudaMemset(mdsInGPU->nMDs,0,nModules*sizeof(unsigned int));
#endif
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
    
#if defined(AddObjects) && !defined(Full_Explicit)
    addMiniDoubletsToEvent();
#endif


}

void SDL::Event::createSegmentsWithModuleMap()
{
    if(segmentsInGPU == nullptr)
    {
#ifdef Explicit_Seg
#ifdef Full_Explicit
        cudaMalloc(&segmentsInGPU, sizeof(SDL::segments));
#else
        cudaMallocManaged(&segmentsInGPU, sizeof(SDL::segments));
#endif

        cudaMallocHost(&segmentsInTemp, sizeof(SDL::segments));
        createSegmentsInExplicitMemory(*segmentsInGPU, *segmentsInTemp,N_MAX_SEGMENTS_PER_MODULE, nModules);
#else
        cudaMallocManaged(&segmentsInGPU, sizeof(SDL::segments));
        createSegmentsInUnifiedMemory(*segmentsInGPU, N_MAX_SEGMENTS_PER_MODULE, nModules);
#endif
    }
#if defined(Explicit_Seg) && !defined(Full_Explicit)
    cudaMemset(segmentsInGPU->nSegments,0,nModules*sizeof(unsigned int));
#endif
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
#if defined(AddObjects) && !defined(Full_Explicit)
    addSegmentsToEvent();
#endif

}


void SDL::Event::createTriplets()
{
    unsigned int nLowerModules = *modulesInGPU->nLowerModules;

    if(tripletsInGPU == nullptr)
    {
#ifdef Explicit_Trips
#ifdef Full_Explicit
        cudaMalloc(&tripletsInGPU, sizeof(SDL::triplets));
#else
        cudaMallocManaged(&tripletsInGPU, sizeof(SDL::triplets));
#endif
        cudaMallocHost(&tripletsInTemp,sizeof(SDL::triplets));
        createTripletsInExplicitMemory(*tripletsInGPU,*tripletsInTemp, N_MAX_TRIPLETS_PER_MODULE, nLowerModules);
#else
        cudaMallocManaged(&tripletsInGPU, sizeof(SDL::triplets));
        createTripletsInUnifiedMemory(*tripletsInGPU, N_MAX_TRIPLETS_PER_MODULE, nLowerModules);
#endif
    }
#if defined(Explicit_Trips) && !defined(Full_Explicit)
    cudaMemset(tripletsInGPU->nTriplets,0,nLowerModules*sizeof(unsigned int));
#endif

    unsigned int nThreads = 1;
    unsigned int nBlocks = nLowerModules % nThreads == 0 ? nLowerModules/nThreads : nLowerModules/nThreads + 1;
    createTripletsInGPU<<<nBlocks,nThreads>>>(*modulesInGPU, *hitsInGPU, *mdsInGPU, *segmentsInGPU, *tripletsInGPU);

    cudaError_t cudaerr = cudaDeviceSynchronize();
    if(cudaerr != cudaSuccess)
    {
        std::cout<<"sync failed with error : "<<cudaGetErrorString(cudaerr)<<std::endl;
    }
#if defined(AddObjects) && !defined(Full_Explicit)
    addTripletsToEvent();
#endif
}


void SDL::Event::createTrackletsWithModuleMap()
{
    unsigned int nLowerModules = *modulesInGPU->nLowerModules;

    //TRCAKLETS - To conserve memory, we shall be only declaring nLowerModules amount of memory!!!!!!!
    if(trackletsInGPU == nullptr)
    {
#ifdef Explicit_Tracklet
#ifdef Full_Explicit
        cudaMalloc(&trackletsInGPU, sizeof(SDL::tracklets));
#else
        cudaMallocManaged(&trackletsInGPU, sizeof(SDL::tracklets));
#endif
        cudaMallocHost(&trackletsInTemp,sizeof(SDL::tracklets));
        createTrackletsInExplicitMemory(*trackletsInGPU,*trackletsInTemp, N_MAX_TRACKLETS_PER_MODULE , nLowerModules);
#else
        cudaMallocManaged(&trackletsInGPU, sizeof(SDL::tracklets));
        createTrackletsInUnifiedMemory(*trackletsInGPU, N_MAX_TRACKLETS_PER_MODULE , nLowerModules);
#endif
    }
#if defined(Explicit_Tracklet) && !defined(Full_Explicit)
    cudaMemset(trackletsInGPU->nTracklets,0,nLowerModules*sizeof(unsigned int));
#endif

    unsigned int nThreads = 1;
    unsigned int nBlocks = nLowerModules % nThreads == 0 ? nLowerModules/nThreads : nLowerModules/nThreads + 1;

    createTrackletsInGPU<<<nBlocks,nThreads>>>(*modulesInGPU, *hitsInGPU, *mdsInGPU, *segmentsInGPU, *trackletsInGPU);

    cudaError_t cudaerr = cudaDeviceSynchronize();
    if(cudaerr != cudaSuccess)
    {
        std::cout<<"sync failed with error : "<<cudaGetErrorString(cudaerr)<<std::endl;    

    }
    /*addTrackletsToEvent will be called in the createTrackletsWithAGapWithModuleMap function*/
#if defined(AddObjects) && !defined(Full_Explicit)
    addTrackletsToEvent();
#endif

}


void SDL::Event::createTrackletsWithAGapWithModuleMap()
{
    //use the same trackletsInGPU as before if it exists
    unsigned int nLowerModules = *modulesInGPU->nLowerModules;

    //TRCAKLETS - To conserve memory, we shall be only declaring nLowerModules amount of memory!!!!!!!
    if(trackletsInGPU == nullptr)
    {
#ifdef Explicit_Tracklet
#ifdef Full_Explicit
        cudaMalloc(&trackletsInGPU, sizeof(SDL::tracklets));
#else
        cudaMallocManaged(&trackletsInGPU, sizeof(SDL::tracklets));
#endif
        cudaMallocHost(&trackletsInTemp,sizeof(SDL::tracklets));
        createTrackletsInExplicitMemory(*trackletsInGPU,*trackletsInTemp, N_MAX_TRACKLETS_PER_MODULE , nLowerModules);
#else
        cudaMallocManaged(&trackletsInGPU, sizeof(SDL::tracklets));
        createTrackletsInUnifiedMemory(*trackletsInGPU, N_MAX_TRACKLETS_PER_MODULE , nLowerModules);
#endif
    }
#if defined(Explicit_Tracklet) && !defined(Full_Explicit)
    cudaMemset(trackletsInGPU->nTracklets,0,nLowerModules*sizeof(unsigned int));
#endif

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

            //for(unsigned int jdx = 0; jdx < trackletsInGPU->nTracklets[i]; jdx++)
            //{
            //    printTracklet(*trackletsInGPU, *segmentsInGPU, *mdsInGPU, *hitsInGPU, *modulesInGPU, i * N_MAX_TRACKLETS_PER_MODULE + jdx);
            //}
            
 
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

__global__ void testMiniDoublets(struct SDL::miniDoublets& mdsInGPU){
  int tid = threadIdx.x+(blockDim.x*blockIdx.x);
    if(tid<2000){
    printf("testmini %d: %d\n",tid,mdsInGPU.nMDs[tid]);
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

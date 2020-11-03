#include "Event.cuh"

//CUDA Kernel for Minidoublet creation
__global__ void createMiniDoubletsInGPU(int nModules,SDL::MiniDoublet* mdsInGPU,SDL::Module** lowerModulesInGPU,int* mdMemoryCounter,SDL::MDAlgo algo)
{
    int moduleIter = blockIdx.x * blockDim.x + threadIdx.x;
    int moduleStride = blockDim.x * gridDim.x;
    int lowerHitIter = blockIdx.y * blockDim.y + threadIdx.y;
    int lowerHitStride = blockDim.y * gridDim.y;
    int upperHitIter = blockIdx.z * blockDim.z + threadIdx.z;
    int upperHitStride = blockDim.z * gridDim.z;

    for(int i = moduleIter; i<nModules;i+=moduleStride)
    {
        SDL::Module* lowerModule = lowerModulesInGPU[i];
        SDL::Module* upperModule = (lowerModulesInGPU[i]->partnerModule());
        if(upperModule == nullptr) continue;
        int numberOfLowerHits = lowerModule->getNumberOfHits();
        int numberOfUpperHits = upperModule->getNumberOfHits();
        if(lowerHitIter >= numberOfLowerHits) continue;
        if(upperHitIter >= numberOfUpperHits) continue;

        SDL::Hit** lowerHits = lowerModule->getHitPtrs();
        SDL::Hit** upperHits = upperModule->getHitPtrs();

        for(int j = lowerHitIter; j<numberOfLowerHits;j+=lowerHitStride)
        {
            for(int k = upperHitIter;k<numberOfUpperHits;k+=upperHitStride)
            {
                SDL::MiniDoublet mdCand(lowerHits[j],upperHits[k]);
                mdCand.runMiniDoubletAlgo(algo);
                if(mdCand.passesMiniDoubletAlgo(algo))
                {
                    int idx = atomicAdd(&mdMemoryCounter[i],1);
                    mdsInGPU[i*100+idx] = mdCand;

                }
            }
        }

    }

}




__global__ void createSegmentsInGPU(int nModules, SDL::Segment* segmentsInGPU,SDL::Module** lowerModulesInGPU,SDL::Module* modulesInGPU,int* segmentMemoryCounter,int* connectedModuleArray, int* nConnectedModules,SDL::SGAlgo algo)
{
    int MAX_CONNECTED_MODULES = 40;
    int moduleIter = blockIdx.x * blockDim.x + threadIdx.x;
    int moduleStride = blockDim.x * gridDim.x;
    int lowerMDIter = blockIdx.y * blockDim.y + threadIdx.y;
//    int lowerMDStride = blockDim.y * gridDim.y;
    int upperMDIter = blockIdx.z * blockDim.z + threadIdx.z;
//    int upperMDStride = blockDim.z * gridDim.z;

    for(int i = moduleIter; i<nModules * MAX_CONNECTED_MODULES;i+=moduleStride) //runs only once
    {
        int lowerModuleIter = i/MAX_CONNECTED_MODULES;
        int upperModuleIter = i % MAX_CONNECTED_MODULES;
        if(lowerModuleIter >= nModules) continue;
        if(upperModuleIter >= nConnectedModules[lowerModuleIter]) continue;  //important dude
        int upperModuleIndex = connectedModuleArray[lowerModuleIter * MAX_CONNECTED_MODULES + upperModuleIter];
        SDL::Module* lowerModule = lowerModulesInGPU[lowerModuleIter];
        SDL::Module* upperModule = &modulesInGPU[upperModuleIndex];
//        printf("lowerModuleIter = %d,upperModuleIndex = %d\n",lowerModuleIter,upperModuleIndex);
        int numberOfLowerMDs = lowerModule->getNumberOfMiniDoublets();
        int numberOfUpperMDs = upperModule->getNumberOfMiniDoublets();

        if(lowerMDIter >= numberOfLowerMDs) continue;
        if(upperMDIter >= numberOfUpperMDs) continue;

        SDL::MiniDoublet** lowerMDs = lowerModule->getMiniDoubletPtrs();
        SDL::MiniDoublet** upperMDs = upperModule->getMiniDoubletPtrs();
        SDL::Segment sgCand(lowerMDs[lowerMDIter],upperMDs[upperMDIter]);
        sgCand.runSegmentAlgo(algo);
        if(sgCand.passesSegmentAlgo(algo))
        {
            int idx = atomicAdd(segmentMemoryCounter,1);
            segmentsInGPU[idx] = sgCand;
        }
    }

}

SDL::Event::Event() : logLevel_(SDL::Log_Nothing)
{
    //createLayers();
    n_hits_by_layer_barrel_.fill(0);
    n_hits_by_layer_endcap_.fill(0);
    n_hits_by_layer_barrel_upper_.fill(0);
    n_hits_by_layer_endcap_upper_.fill(0);
    n_miniDoublet_candidates_by_layer_barrel_.fill(0);
    n_segment_candidates_by_layer_barrel_.fill(0);
    n_tracklet_candidates_by_layer_barrel_.fill(0);
    n_triplet_candidates_by_layer_barrel_.fill(0);
    n_trackcandidate_candidates_by_layer_barrel_.fill(0);
    n_miniDoublet_by_layer_barrel_.fill(0);
    n_segment_by_layer_barrel_.fill(0);
    n_tracklet_by_layer_barrel_.fill(0);
    n_triplet_by_layer_barrel_.fill(0);
    n_trackcandidate_by_layer_barrel_.fill(0);
    n_miniDoublet_candidates_by_layer_endcap_.fill(0);
    n_segment_candidates_by_layer_endcap_.fill(0);
    n_tracklet_candidates_by_layer_endcap_.fill(0);
    n_triplet_candidates_by_layer_endcap_.fill(0);
    n_trackcandidate_candidates_by_layer_endcap_.fill(0);
    n_miniDoublet_by_layer_endcap_.fill(0);
    n_segment_by_layer_endcap_.fill(0);
    n_tracklet_by_layer_endcap_.fill(0);
    n_triplet_by_layer_endcap_.fill(0);
    n_trackcandidate_by_layer_endcap_.fill(0);
    moduleMemoryCounter = 0;
    lowerModuleMemoryCounter = 0;
    hitMemoryCounter = 0;
    hit2SEdgeMemoryCounter = 0;

}

SDL::Event::~Event()
{
    cudaFree(hitsInGPU);
    cudaFree(modulesInGPU);
    cudaFree(lowerModulesInGPU);
    cudaFree(mdsInGPU);
    cudaFree(mdMemoryCounter);

    cudaFree(segmentsInGPU);
    cudaFree(moduleConnectionMapArray);
    cudaFree(numberOfConnectedModules);
    cudaFree(segmentMemoryCounter);
}


bool SDL::Event::hasModule(unsigned int detId)
{
    if (modulesMapByDetId_.find(detId) == modulesMapByDetId_.end())
    {
        return false;
    }
    else
    {
        return true;
    }
}

void SDL::Event::setLogLevel(SDL::LogLevel logLevel)
{
    logLevel_ = logLevel;
}

void SDL::Event::initModulesInGPU()
{
    const int MODULE_MAX=50000;
    cudaMallocManaged(&modulesInGPU,MODULE_MAX * sizeof(SDL::Module));
    cudaMallocManaged(&lowerModulesInGPU, MODULE_MAX * sizeof(SDL::Module*));
}

SDL::Module* SDL::Event::getModule(unsigned int detId,bool addModule)
{
    // using std::map::emplace
    if(moduleMemoryCounter == 0)
    {
        initModulesInGPU();
    }
    std::pair<std::map<unsigned int, Module*>::iterator, bool> emplace_result = modulesMapByDetId_.emplace(detId,nullptr);
    // Retreive the module
    auto& inserted_or_existing = (*(emplace_result.first)).second;

    // If new was inserted, then insert to modulePtrs_ pointer list
    if (emplace_result.second and addModule) // if true, new was inserted
    {
        //cudaMallocManaged(&((*(emplace_result.first)).second),sizeof(SDL::Module));
         (*(emplace_result.first)).second = &modulesInGPU[moduleMemoryCounter];

        //*inserted_or_existing =SDL:: Module(detId);
        modulesInGPU[moduleMemoryCounter] = SDL::Module(detId);
        Module* module_ptr = inserted_or_existing;
        module_ptr->setDrDz(tiltedGeometry.getDrDz(detId));
        if(module_ptr->subdet() == SDL::Module::Endcap)
        {
            module_ptr->setSlope(SDL::endcapGeometry.getSlopeLower(detId));
        }
        else
        {
            module_ptr->setSlope(SDL::tiltedGeometry.getSlope(detId));
        }

        
        // Add the module pointer to the list of modules
        modulePtrs_.push_back(module_ptr);
        // If the module is lower module then add to list of lower modules
        if (module_ptr->isLower())
        {
            lowerModulesInGPU[lowerModuleMemoryCounter] = module_ptr;
            lowerModuleMemoryCounter++;
        }
       
       moduleMemoryCounter++;

    }

    return inserted_or_existing;
}

const std::vector<SDL::Module*> SDL::Event::getModulePtrs() const
{
    return modulePtrs_;
}

const std::vector<SDL::Module*> SDL::Event::getLowerModulePtrs() const
{
    return lowerModulePtrs_;
}

/*
void SDL::Event::createLayers()
{
    // Create barrel layers
    for (int ilayer = SDL::Layer::BarrelLayer0; ilayer < SDL::Layer::nBarrelLayer; ++ilayer)
    {
        barrelLayers_[ilayer] = SDL::Layer(ilayer, SDL::Layer::Barrel);
        layerPtrs_.push_back(&(barrelLayers_[ilayer]));
    }

    // Create endcap layers
    for (int ilayer = SDL::Layer::EndcapLayer0; ilayer < SDL::Layer::nEndcapLayer; ++ilayer)
    {
        endcapLayers_[ilayer] = SDL::Layer(ilayer, SDL::Layer::Endcap);
        layerPtrs_.push_back(&(endcapLayers_[ilayer]));
    }
}

SDL::Layer& SDL::Event::getLayer(int ilayer, SDL::Layer::SubDet subdet)
{
    if (subdet == SDL::Layer::Barrel)
        return barrelLayers_[ilayer];
    else // if (subdet == SDL::Layer::Endcap)
        return endcapLayers_[ilayer];
}

const std::vector<SDL::Layer*> SDL::Event::getLayerPtrs() const
{
    return layerPtrs_;
}*/

void SDL::Event::initHitsInGPU()
{
    const int HIT_MAX = 1000000;
    cudaMallocManaged(&hitsInGPU,HIT_MAX * sizeof(SDL::Hit));
    const int HIT_2S_MAX = 100000;
    cudaMallocManaged(&hits2sEdgeInGPU,HIT_2S_MAX * sizeof(SDL::Hit));
}

void SDL::Event::addHitToModule(SDL::Hit hit, unsigned int detId)
{
    // Add to global list of hits, where it will hold the object's instance
    // And get the module (if not exists, then create), and add the address to Module.hits_
    //construct a cudaMallocManaged object and send that in, so that we won't have issues in the GPU
    if(hitMemoryCounter == 0)
    {
        initHitsInGPU();
    }
    hitsInGPU[hitMemoryCounter] = hit;
    hitsInGPU[hitMemoryCounter].setModule(getModule(detId));
    getModule(detId)->addHit(&hitsInGPU[hitMemoryCounter]);
    //hits_.push_back(hitsInGPU[hitMemoryCounter]);


    // Count number of hits in the event
    incrementNumberOfHits(*getModule(detId));

    // If the hit is 2S in the endcap then the hit boundary needs to be set
    if (getModule(detId)->subdet() == SDL::Module::Endcap and getModule(detId)->moduleType() == SDL::Module::TwoS)
    {
         
        hits2sEdgeInGPU[hit2SEdgeMemoryCounter] = SDL::GeometryUtil::stripHighEdgeHit(hitsInGPU[hitMemoryCounter]);
        hits2sEdgeInGPU[hit2SEdgeMemoryCounter+1] = SDL::GeometryUtil::stripLowEdgeHit(hitsInGPU[hitMemoryCounter]);
        hits_2s_edges_.push_back(hits2sEdgeInGPU[hit2SEdgeMemoryCounter]);
        hitsInGPU[hitMemoryCounter].setHitHighEdgePtr(&hits2sEdgeInGPU[hit2SEdgeMemoryCounter]);

        hits_2s_edges_.push_back(hits2sEdgeInGPU[hit2SEdgeMemoryCounter+1]);
        hitsInGPU[hitMemoryCounter].setHitLowEdgePtr(&hits2sEdgeInGPU[hit2SEdgeMemoryCounter+1]);

        hit2SEdgeMemoryCounter+= 2;
    }

    hitMemoryCounter++;
}

void SDL::Event::initMDsInGPU()
{
    const int MD_MAX = lowerModuleMemoryCounter * 100;
    cudaMallocManaged(&mdsInGPU,MD_MAX * sizeof(SDL::MiniDoublet));
    cudaMallocManaged(&mdMemoryCounter,lowerModuleMemoryCounter * sizeof(int));
#pragma omp parallel for
    for(int i = 0; i< lowerModuleMemoryCounter; i++)
    {
        mdMemoryCounter[i] = 0;
    }
}

void SDL::Event::addMiniDoubletToEvent(SDL::MiniDoublet* md, SDL::Module& module)//, int layerIdx, SDL::Layer::SubDet subdet)
{
    // Add to global list of mini doublets, where it will hold the object's instance

    // And get the module (if not exists, then create), and add the address to Module.hits_
    //construct a cudaMallocManaged object and send that in, so that we won't have issues in the GPU
    module.addMiniDoublet(md);
//    miniDoublets_.push_back(md);

    incrementNumberOfMiniDoublets(module);
    // And get the layer
//    getLayer(layerIdx, subdet).addMiniDoublet(&mdsInGPU[mdMemoryCounter]);
}



void SDL::Event::addSegmentToEvent(SDL::Segment* sg, SDL::Module& module)//, int layerIdx, SDL::Layer::SubDet subdet)
{
    // Add to global list of segments, where it will hold the object's instance
//    segments_.push_back(sg);

    // And get the module (if not exists, then create), and add the address to Module.hits_
    module.addSegment(sg);
    incrementNumberOfSegments(module);

    // And get the layer andd the segment to it
//    getLayer(layerIdx, subdet).addSegment(&(segments_.back()));

    // Link segments to mini-doublets
//    segments_.back().addSelfPtrToMiniDoublets();

}

/*void SDL::Event::addTrackletToEvent(SDL::Tracklet tl, unsigned int detId, int layerIdx, SDL::Layer::SubDet subdet)
{
    // Add to global list of segments, where it will hold the object's instance
    tracklets_.push_back(tl);

    // And get the module (if not exists, then create), and add the address to Module.hits_
    getModule(detId)->addTracklet(&(tracklets_.back()));

    // And get the layer andd the segment to it
    getLayer(layerIdx, subdet).addTracklet(&(tracklets_.back()));

    // Link segments to mini-doublets
    tracklets_.back().addSelfPtrToSegments();

}

void SDL::Event::addTripletToEvent(SDL::Triplet tp, unsigned int detId, int layerIdx, SDL::Layer::SubDet subdet)
{
    // Add to global list of segments, where it will hold the object's instance
    triplets_.push_back(tp);

    // And get the module (if not exists, then create), and add the address to Module.hits_
    getModule(detId)->addTriplet(&(triplets_.back()));

    // And get the layer andd the triplet to it
    getLayer(layerIdx, subdet).addTriplet(&(triplets_.back()));
}

[[deprecated("SDL:: addSegmentToLowerModule() is deprecated. Use addSegmentToEvent")]]
void SDL::Event::addSegmentToLowerModule(SDL::Segment sg, unsigned int detId)
{
    // Add to global list of segments, where it will hold the object's instance
    segments_.push_back(sg);

    // And get the module (if not exists, then create), and add the address to Module.hits_
    getModule(detId)->addSegment(&(segments_.back()));
}

[[deprecated("SDL:: addSegmentToLowerLayer() is deprecated. Use addSegmentToEvent")]]
void SDL::Event::addSegmentToLowerLayer(SDL::Segment sg, int layerIdx, SDL::Layer::SubDet subdet)
{
    // Add to global list of segments, where it will hold the object's instance
    segments_.push_back(sg);

    // And get the layer
    getLayer(layerIdx, subdet).addSegment(&(segments_.back()));
}

void SDL::Event::addTrackletToLowerLayer(SDL::Tracklet tl, int layerIdx, SDL::Layer::SubDet subdet)
{
    // Add to global list of tracklets, where it will hold the object's instance
    tracklets_.push_back(tl);

    // And get the layer
    getLayer(layerIdx, subdet).addTracklet(&(tracklets_.back()));
}

void SDL::Event::addTrackCandidateToLowerLayer(SDL::TrackCandidate tc, int layerIdx, SDL::Layer::SubDet subdet)
{
    // Add to global list of trackcandidates, where it will hold the object's instance
    trackcandidates_.push_back(tc);

    // And get the layer
    getLayer(layerIdx, subdet).addTrackCandidate(&(trackcandidates_.back()));
}

*/

//This dude needs to get into the GPU

void SDL::Event::createMiniDoublets(MDAlgo algo)
{

    for(int i = 0; i < moduleMemoryCounter; i++)
    {
        modulesInGPU[i].setPartnerModule(getModule(modulesInGPU[i].partnerDetId()));
    }

    initMDsInGPU();

    int nModules = lowerModuleMemoryCounter;
    int MAX_HITS = 100;
    dim3 nThreads(1,16,16);
    dim3 nBlocks((nModules % nThreads.x == 0 ? nModules/nThreads.x : nModules/nThreads.x + 1),(MAX_HITS % nThreads.y == 0 ? MAX_HITS/nThreads.y : MAX_HITS/nThreads.y + 1), (MAX_HITS % nThreads.z == 0 ? MAX_HITS/nThreads.z : MAX_HITS/nThreads.z + 1));
      std::cout<<nBlocks.x<<" " <<nBlocks.y<<" "<<nBlocks.z<<" "<<std::endl;
//    int nBlocks = (mdGPUCounter % nThreads == 0) ? mdGPUCounter/nThreads : mdGPUCounter/nThreads + 1;
    cudaError_t prefetchErr = cudaMemPrefetchAsync(hitsInGPU,sizeof(SDL::Hit) * (hitMemoryCounter+1),0);
    if(prefetchErr != cudaSuccess)
    {
        std::cout<<"prefetch failed with error : "<<cudaGetErrorString(prefetchErr)<<std::endl;
    }
    prefetchErr = cudaMemPrefetchAsync(modulesInGPU,sizeof(SDL::Module) * (moduleMemoryCounter+1),0);
    if(prefetchErr != cudaSuccess)
    {
        std::cout<<"prefetch failed with error : "<<cudaGetErrorString(prefetchErr)<<std::endl;
    } 
    createMiniDoubletsInGPU<<<nBlocks,nThreads>>>(nModules,mdsInGPU,lowerModulesInGPU,mdMemoryCounter,algo);
    
    cudaError_t cudaerr = cudaDeviceSynchronize();
//    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
    {          
        std::cout<<"sync failed with error : "<<cudaGetErrorString(cudaerr)<<std::endl;    
    }
    //add mini-doublets to the module arrays for other stuff outside

    for(int i=0; i<lowerModuleMemoryCounter;i++)
    {
        SDL::Module& lowerModule = (Module&)(*lowerModulesInGPU[i]);

        for(int j = 0; j<mdMemoryCounter[i];j++)
        {
            if(lowerModule.subdet() == SDL::Module::Barrel)
            {
                addMiniDoubletToEvent(&mdsInGPU[i*100 + j],lowerModule);//,lowerModule.layer(),SDL::Layer::Barrel);    
            }
            else
            {
                addMiniDoubletToEvent(&mdsInGPU[i*100+j],lowerModule);//,lowerModule.layer(),SDL::Layer::Barrel);
            }
        }
    }
}

void SDL::Event::getConnectedModuleArray()
{
    
    /* Create an index based module map array. First get the detIds of the lower modules, then search the modulesInGPU
     * array for the corresponding module indices, and fill the moduleConnectionMap array
     */
    int N_MAX_CONNECTED_MODULES = 40;
    for(int i = 0; i<lowerModuleMemoryCounter;i++)
    {
        unsigned int detId = lowerModulesInGPU[i]->detId();
        const std::vector<unsigned int>& connections = moduleConnectionMap.getConnectedModuleDetIds(detId);
        int j = 0;
        for(auto &connectedModuleId:connections)
        {
            if( not getModule(connectedModuleId,false)) continue;
            unsigned int index = getModule(connectedModuleId,false) - modulesInGPU;
            moduleConnectionMapArray[i * N_MAX_CONNECTED_MODULES + j] = index;
            j++;
        }
        if(j > N_MAX_CONNECTED_MODULES) std::cout<<"WARNING : increase max connected modules to "<<j<<std::endl;
        numberOfConnectedModules[i] = j;
    }
}

void SDL::Event::initSegmentsInGPU()
{
    int N_MAX_CONNECTED_MODULES = 40;
    int SEGMENTS_MAX = 100000;
    cudaMallocManaged(&moduleConnectionMapArray,sizeof(int) * N_MAX_CONNECTED_MODULES * lowerModuleMemoryCounter);
    cudaMallocManaged(&numberOfConnectedModules,sizeof(int) * lowerModuleMemoryCounter);
    cudaMallocManaged(&segmentsInGPU,SEGMENTS_MAX * sizeof(SDL::Segment));
    cudaMallocManaged(&segmentMemoryCounter,sizeof(int));
    getConnectedModuleArray();
    
}

void SDL::Event::createSegmentsWithModuleMap(SGAlgo algo)
{
    initSegmentsInGPU();
    getConnectedModuleArray();
    int N_MAX_CONNECTED_MODULES = 40;
    int N_MAX_MD = 200;
    int nModules = lowerModuleMemoryCounter;
    int dimX = N_MAX_CONNECTED_MODULES * nModules;
    dim3 nThreads(1,16,16);
    dim3 nBlocks((dimX % nThreads.x == 0 ? dimX/nThreads.x : dimX/nThreads.x + 1),(N_MAX_MD % nThreads.y == 0 ? N_MAX_MD / nThreads.y : N_MAX_MD/nThreads.y + 1),(N_MAX_MD % nThreads.z == 0 ? N_MAX_MD/nThreads.z : N_MAX_MD/nThreads.z + 1 ));
    createSegmentsInGPU<<<nBlocks,nThreads>>>(nModules,segmentsInGPU,lowerModulesInGPU,modulesInGPU,segmentMemoryCounter,moduleConnectionMapArray,numberOfConnectedModules,algo);
    cudaError_t cudaerr = cudaDeviceSynchronize();
//    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
    {          
        std::cout<<"sync failed with error : "<<cudaGetErrorString(cudaerr)<<std::endl;    
    }
    else
    {
        std::cout<<"sync successful!"<<std::endl;
    }

    for(int i = 0; i<*segmentMemoryCounter; i++)
    {
    //    SDL::cout<<segmentsInGPU[i]<<std::endl;
        SDL::Module& lowerModule = (SDL::Module&)segmentsInGPU[i].innerMiniDoubletPtr()->lowerHitPtr()->getModule();
        addSegmentToEvent(&segmentsInGPU[i],lowerModule);
    }

}


/*
void SDL::Event::createSegments(SGAlgo algo)
{

    for (auto& segment_compatible_layer_pair : SDL::Layer::getListOfSegmentCompatibleLayerPairs())
    {
        int innerLayerIdx = segment_compatible_layer_pair.first.first;
        SDL::Layer::SubDet innerLayerSubDet = segment_compatible_layer_pair.first.second;
        int outerLayerIdx = segment_compatible_layer_pair.second.first;
        SDL::Layer::SubDet outerLayerSubDet = segment_compatible_layer_pair.second.second;
        createSegmentsFromTwoLayers(innerLayerIdx, innerLayerSubDet, outerLayerIdx, outerLayerSubDet, algo);
    }
}

void SDL::Event::createSegmentsWithModuleMap(SGAlgo algo)
{
    // Loop over lower modules
    for (auto& lowerModulePtr : getLowerModulePtrs())
    {

        // Create mini doublets
        createSegmentsFromInnerLowerModule(lowerModulePtr->detId(), algo);

    }
}

void SDL::Event::createSegmentsFromInnerLowerModule(unsigned int detId, SDL::SGAlgo algo)
{

    // x's and y's are mini doublets
    // -------x--------
    // --------x------- <--- outer lower module
    //
    // --------y-------
    // -------y-------- <--- inner lower module

    // Get reference to the inner lower Module
    Module& innerLowerModule = *getModule(detId);

    // Triple nested loops
    // Loop over inner lower module mini-doublets
    for (auto& innerMiniDoubletPtr : innerLowerModule.getMiniDoubletPtrs())
    {

        // Get reference to mini-doublet in inner lower module
        SDL::MiniDoublet& innerMiniDoublet = *innerMiniDoubletPtr;

        // Get connected outer lower module detids
        const std::vector<unsigned int>& connectedModuleDetIds = moduleConnectionMap.getConnectedModuleDetIds(detId);

        // Loop over connected outer lower modules
        for (auto& outerLowerModuleDetId : connectedModuleDetIds)
        {

            if (not hasModule(outerLowerModuleDetId))
                continue;

            // Get reference to the outer lower module
            Module& outerLowerModule = *getModule(outerLowerModuleDetId);

            // Loop over outer lower module mini-doublets
            for (auto& outerMiniDoubletPtr : outerLowerModule.getMiniDoubletPtrs())
            {

                // Get reference to mini-doublet in outer lower module
                SDL::MiniDoublet& outerMiniDoublet = *outerMiniDoubletPtr;

                // Create a segment candidate
                SDL::Segment sgCand(innerMiniDoubletPtr, outerMiniDoubletPtr);

                // Run segment algorithm on sgCand (segment candidate)
                sgCand.runSegmentAlgo(algo, logLevel_);

                // Count the # of sgCands considered by layer
                incrementNumberOfSegmentCandidates(innerLowerModule);

                if (sgCand.passesSegmentAlgo(algo))
                {

                    // Count the # of sg formed by layer
                    incrementNumberOfSegments(innerLowerModule);

                    if (innerLowerModule.subdet() == SDL::Module::Barrel)
                        addSegmentToEvent(sgCand, innerLowerModule.detId(), innerLowerModule.layer(), SDL::Layer::Barrel);
                    else
                        addSegmentToEvent(sgCand, innerLowerModule.detId(), innerLowerModule.layer(), SDL::Layer::Endcap);
                }

            }
        }
    }
}

void SDL::Event::createTriplets(TPAlgo algo)
{
    // Loop over lower modules
    for (auto& lowerModulePtr : getLowerModulePtrs())
    {

        // if (lowerModulePtr->layer() != 4 and lowerModulePtr->layer() != 3)
        //     continue;

        // Create mini doublets
        createTripletsFromInnerLowerModule(lowerModulePtr->detId(), algo);

    }
}

void SDL::Event::createTripletsFromInnerLowerModule(unsigned int detId, SDL::TPAlgo algo)
{

    // Get reference to the inner lower Module
    Module& innerLowerModule = *getModule(detId);

    // Triple nested loops
    // Loop over inner lower module for segments
    for (auto& innerSegmentPtr : innerLowerModule.getSegmentPtrs())
    {

        // Get reference to segment in inner lower module
        SDL::Segment& innerSegment = *innerSegmentPtr;

        // Get connected outer lower module detids
        const std::vector<unsigned int>& connectedModuleDetIds = moduleConnectionMap.getConnectedModuleDetIds(detId);

        // Loop over connected outer lower modules
        for (auto& outerLowerModuleDetId : connectedModuleDetIds)
        {

            if (not hasModule(outerLowerModuleDetId))
                continue;

            // Get reference to the outer lower module
            Module& outerLowerModule = *getModule(outerLowerModuleDetId);

            // Loop over outer lower module mini-doublets
            for (auto& outerSegmentPtr : outerLowerModule.getSegmentPtrs())
            {

                // Get reference to mini-doublet in outer lower module
                SDL::Segment& outerSegment = *outerSegmentPtr;

                // Create a segment candidate
                SDL::Triplet tpCand(innerSegmentPtr, outerSegmentPtr);

                // Run segment algorithm on tpCand (segment candidate)
                tpCand.runTripletAlgo(algo, logLevel_);

                // Count the # of tpCands considered by layer
                incrementNumberOfTripletCandidates(innerLowerModule);

                if (tpCand.passesTripletAlgo(algo))
                {

                    // Count the # of sg formed by layer
                    incrementNumberOfTriplets(innerLowerModule);

                    if (innerLowerModule.subdet() == SDL::Module::Barrel)
                        addTripletToEvent(tpCand, innerLowerModule.detId(), innerLowerModule.layer(), SDL::Layer::Barrel);
                    else
                        addTripletToEvent(tpCand, innerLowerModule.detId(), innerLowerModule.layer(), SDL::Layer::Endcap);
                }

            }
        }
    }
}

void SDL::Event::createTracklets(TLAlgo algo)
{
    for (auto& tracklet_compatible_layer_pair : SDL::Layer::getListOfTrackletCompatibleLayerPairs())
    {
        int innerLayerIdx = tracklet_compatible_layer_pair.first.first;
        SDL::Layer::SubDet innerLayerSubDet = tracklet_compatible_layer_pair.first.second;
        int outerLayerIdx = tracklet_compatible_layer_pair.second.first;
        SDL::Layer::SubDet outerLayerSubDet = tracklet_compatible_layer_pair.second.second;
        createTrackletsFromTwoLayers(innerLayerIdx, innerLayerSubDet, outerLayerIdx, outerLayerSubDet, algo);
    }
}

void SDL::Event::createTrackletsWithModuleMap(TLAlgo algo)
{
    // Loop over lower modules
    for (auto& lowerModulePtr : getLowerModulePtrs())
    {

        // if (lowerModulePtr->layer() != 1)
        //     continue;

        // Create mini doublets
        createTrackletsFromInnerLowerModule(lowerModulePtr->detId(), algo);

    }
}

// Create tracklets from inner modules
void SDL::Event::createTrackletsFromInnerLowerModule(unsigned int detId, SDL::TLAlgo algo)
{

    // Get reference to the inner lower Module
    Module& innerLowerModule = *getModule(detId);

    // Triple nested loops
    // Loop over inner lower module for segments
    for (auto& innerSegmentPtr : innerLowerModule.getSegmentPtrs())
    {

        // Get reference to segment in inner lower module
        SDL::Segment& innerSegment = *innerSegmentPtr;

        // Get the outer mini-doublet module detId
        const SDL::Module& innerSegmentOuterModule = innerSegment.outerMiniDoubletPtr()->lowerHitPtr()->getModule();

        unsigned int innerSegmentOuterModuleDetId = innerSegmentOuterModule.detId();

        // Get connected outer lower module detids
        const std::vector<unsigned int>& connectedModuleDetIds = moduleConnectionMap.getConnectedModuleDetIds(innerSegmentOuterModuleDetId);

        // Loop over connected outer lower modules
        for (auto& outerLowerModuleDetId : connectedModuleDetIds)
        {

            if (not hasModule(outerLowerModuleDetId))
                continue;

            // Get reference to the outer lower module
            Module& outerLowerModule = *getModule(outerLowerModuleDetId);

            // Loop over outer lower module mini-doublets
            for (auto& outerSegmentPtr : outerLowerModule.getSegmentPtrs())
            {

                // Count the # of tlCands considered by layer
                incrementNumberOfTrackletCandidates(innerLowerModule);

                // Get reference to mini-doublet in outer lower module
                SDL::Segment& outerSegment = *outerSegmentPtr;

                // Create a tracklet candidate
                SDL::Tracklet tlCand(innerSegmentPtr, outerSegmentPtr);

                // Run segment algorithm on tlCand (tracklet candidate)
                tlCand.runTrackletAlgo(algo, logLevel_);

                if (tlCand.passesTrackletAlgo(algo))
                {

                    // Count the # of sg formed by layer
                    incrementNumberOfTracklets(innerLowerModule);

                    if (innerLowerModule.subdet() == SDL::Module::Barrel)
                        addTrackletToEvent(tlCand, innerLowerModule.detId(), innerLowerModule.layer(), SDL::Layer::Barrel);
                    else
                        addTrackletToEvent(tlCand, innerLowerModule.detId(), innerLowerModule.layer(), SDL::Layer::Endcap);
                }

            }

        }

    }

}

// Create tracklets via navigation
void SDL::Event::createTrackletsViaNavigation(SDL::TLAlgo algo)
{
    // Loop over lower modules
    for (auto& lowerModulePtr : getLowerModulePtrs())
    {

        // Get reference to the inner lower Module
        Module& innerLowerModule = *getModule(lowerModulePtr->detId());

        // Triple nested loops
        // Loop over inner lower module for segments
        for (auto& innerSegmentPtr : innerLowerModule.getSegmentPtrs())
        {

            // Get reference to segment in inner lower module
            SDL::Segment& innerSegment = *innerSegmentPtr;

            // Get the connecting segment ptrs
            for (auto& connectingSegmentPtr : innerSegmentPtr->outerMiniDoubletPtr()->getListOfOutwardSegmentPtrs())
            {

                for (auto& outerSegmentPtr : connectingSegmentPtr->outerMiniDoubletPtr()->getListOfOutwardSegmentPtrs())
                {

                    // Count the # of tlCands considered by layer
                    incrementNumberOfTrackletCandidates(innerLowerModule);

                    // Get reference to mini-doublet in outer lower module
                    SDL::Segment& outerSegment = *outerSegmentPtr;

                    // Create a tracklet candidate
                    SDL::Tracklet tlCand(innerSegmentPtr, outerSegmentPtr);

                    // Run segment algorithm on tlCand (tracklet candidate)
                    tlCand.runTrackletAlgo(algo, logLevel_);

                    if (tlCand.passesTrackletAlgo(algo))
                    {

                        // Count the # of sg formed by layer
                        incrementNumberOfTracklets(innerLowerModule);

                        if (innerLowerModule.subdet() == SDL::Module::Barrel)
                            addTrackletToEvent(tlCand, innerLowerModule.detId(), innerLowerModule.layer(), SDL::Layer::Barrel);
                        else
                            addTrackletToEvent(tlCand, innerLowerModule.detId(), innerLowerModule.layer(), SDL::Layer::Endcap);
                    }

                }

            }

        }
    }

}


// Create tracklets from two layers (inefficient way)
void SDL::Event::createTrackletsFromTwoLayers(int innerLayerIdx, SDL::Layer::SubDet innerLayerSubDet, int outerLayerIdx, SDL::Layer::SubDet outerLayerSubDet, TLAlgo algo)
{
    Layer& innerLayer = getLayer(innerLayerIdx, innerLayerSubDet);
    Layer& outerLayer = getLayer(outerLayerIdx, outerLayerSubDet);

    for (auto& innerSegmentPtr : innerLayer.getSegmentPtrs())
    {
        SDL::Segment& innerSegment = *innerSegmentPtr;
        for (auto& outerSegmentPtr : outerLayer.getSegmentPtrs())
        {
            // SDL::Segment& outerSegment = *outerSegmentPtr;

            // if (SDL::Tracklet::isSegmentPairATracklet(innerSegment, outerSegment, algo, logLevel_))
            //     addTrackletToLowerLayer(SDL::Tracklet(innerSegmentPtr, outerSegmentPtr), innerLayerIdx, innerLayerSubDet);

            SDL::Segment& outerSegment = *outerSegmentPtr;

            SDL::Tracklet tlCand(innerSegmentPtr, outerSegmentPtr);

            tlCand.runTrackletAlgo(algo, logLevel_);

            // Count the # of tracklet candidate considered by layer
            incrementNumberOfTrackletCandidates(innerLayer);

            if (tlCand.passesTrackletAlgo(algo))
            {

                // Count the # of tracklet formed by layer
                incrementNumberOfTracklets(innerLayer);

                addTrackletToLowerLayer(tlCand, innerLayerIdx, innerLayerSubDet);
            }

        }
    }
}

// Create segments from two layers (inefficient way)
void SDL::Event::createSegmentsFromTwoLayers(int innerLayerIdx, SDL::Layer::SubDet innerLayerSubDet, int outerLayerIdx, SDL::Layer::SubDet outerLayerSubDet, SGAlgo algo)
{
    Layer& innerLayer = getLayer(innerLayerIdx, innerLayerSubDet);
    Layer& outerLayer = getLayer(outerLayerIdx, outerLayerSubDet);

    for (auto& innerMiniDoubletPtr : innerLayer.getMiniDoubletPtrs())
    {
        SDL::MiniDoublet& innerMiniDoublet = *innerMiniDoubletPtr;

        for (auto& outerMiniDoubletPtr : outerLayer.getMiniDoubletPtrs())
        {
            SDL::MiniDoublet& outerMiniDoublet = *outerMiniDoubletPtr;

            SDL::Segment sgCand(innerMiniDoubletPtr, outerMiniDoubletPtr);

            sgCand.runSegmentAlgo(algo, logLevel_);

            if (sgCand.passesSegmentAlgo(algo))
            {
                const SDL::Module& innerLowerModule = innerMiniDoubletPtr->lowerHitPtr()->getModule();
                if (innerLowerModule.subdet() == SDL::Module::Barrel)
                    addSegmentToEvent(sgCand, innerLowerModule.detId(), innerLowerModule.layer(), SDL::Layer::Barrel);
                else
                    addSegmentToEvent(sgCand, innerLowerModule.detId(), innerLowerModule.layer(), SDL::Layer::Endcap);
            }

        }
    }
}

void SDL::Event::createTrackCandidates(TCAlgo algo)
{
    // TODO Implement some structure for Track Candidates
    // for (auto& trackCandidate_compatible_layer_pair : SDL::Layer::getListOfTrackCandidateCompatibleLayerPairs())
    // {
    //     int innerLayerIdx = trackCandidate_compatible_layer_pair.first.first;
    //     SDL::Layer::SubDet innerLayerSubDet = trackCandidate_compatible_layer_pair.first.second;
    //     int outerLayerIdx = trackCandidate_compatible_layer_pair.second.first;
    //     SDL::Layer::SubDet outerLayerSubDet = trackCandidate_compatible_layer_pair.second.second;
    //     createTrackCandidatesFromTwoLayers(innerLayerIdx, innerLayerSubDet, outerLayerIdx, outerLayerSubDet, algo);
    // }

    createTrackCandidatesFromTwoLayers(1, SDL::Layer::Barrel, 3, SDL::Layer::Barrel, algo);

}

// Create trackCandidates from two layers (inefficient way)
void SDL::Event::createTrackCandidatesFromTwoLayers(int innerLayerIdx, SDL::Layer::SubDet innerLayerSubDet, int outerLayerIdx, SDL::Layer::SubDet outerLayerSubDet, TCAlgo algo)
{
    Layer& innerLayer = getLayer(innerLayerIdx, innerLayerSubDet);
    Layer& outerLayer = getLayer(outerLayerIdx, outerLayerSubDet);

    for (auto& innerTrackletPtr : innerLayer.getTrackletPtrs())
    {
        SDL::Tracklet& innerTracklet = *innerTrackletPtr;

        for (auto& outerTrackletPtr : outerLayer.getTrackletPtrs())
        {

            SDL::Tracklet& outerTracklet = *outerTrackletPtr;

            SDL::TrackCandidate tcCand(innerTrackletPtr, outerTrackletPtr);

            tcCand.runTrackCandidateAlgo(algo, logLevel_);

            // Count the # of track candidates considered
            incrementNumberOfTrackCandidateCandidates(innerLayer);

            if (tcCand.passesTrackCandidateAlgo(algo))
            {

                // Count the # of track candidates considered
                incrementNumberOfTrackCandidates(innerLayer);

                addTrackCandidateToLowerLayer(tcCand, innerLayerIdx, innerLayerSubDet);
            }

        }
    }
}

void SDL::Event::createTrackCandidatesFromTriplets(TCAlgo algo)
{
    // Loop over lower modules
    for (auto& lowerModulePtr : getLowerModulePtrs())
    {

        // if (not (lowerModulePtr->layer() == 1))
        //     continue;

        // Create mini doublets
        createTrackCandidatesFromInnerModulesFromTriplets(lowerModulePtr->detId(), algo);

    }
}

void SDL::Event::createTrackCandidatesFromInnerModulesFromTriplets(unsigned int detId, SDL::TCAlgo algo)
{

    // Get reference to the inner lower Module
    Module& innerLowerModule = *getModule(detId);

    // Triple nested loops
    // Loop over inner lower module for segments
    for (auto& innerTripletPtr : innerLowerModule.getTripletPtrs())
    {

        // Get reference to segment in inner lower module
        SDL::Triplet& innerTriplet = *innerTripletPtr;

        // Get the outer mini-doublet module detId
        const SDL::Module& innerTripletOutermostModule = innerTriplet.outerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->getModule();

        unsigned int innerTripletOutermostModuleDetId = innerTripletOutermostModule.detId();

        // Get connected outer lower module detids
        const std::vector<unsigned int>& connectedModuleDetIds = moduleConnectionMap.getConnectedModuleDetIds(innerTripletOutermostModuleDetId);

        // Loop over connected outer lower modules
        for (auto& outerLowerModuleDetId : connectedModuleDetIds)
        {

            if (not hasModule(outerLowerModuleDetId))
                continue;

            // Get reference to the outer lower module
            Module& outerLowerModule = *getModule(outerLowerModuleDetId);

            // Loop over outer lower module mini-doublets
            for (auto& outerTripletPtr : outerLowerModule.getTripletPtrs())
            {

                // Count the # of tlCands considered by layer
                incrementNumberOfTrackCandidateCandidates(innerLowerModule);

                // Segment between innerSgOuterMD - outerSgInnerMD
                SDL::Segment sgCand(innerTripletPtr->outerSegmentPtr()->outerMiniDoubletPtr(),outerTripletPtr->innerSegmentPtr()->innerMiniDoubletPtr());

                // Run the segment algo (supposedly is fast)
                sgCand.runSegmentAlgo(SDL::Default_SGAlgo, logLevel_);

                if (not (sgCand.passesSegmentAlgo(SDL::Default_SGAlgo)))
                {
                    continue;
                }

                // SDL::Tracklet tlCand(innerTripletPtr->innerSegmentPtr(), &sgCand);

                // // Run the segment algo (supposedly is fast)
                // tlCand.runTrackletAlgo(SDL::Default_TLAlgo, logLevel_);

                // if (not (tlCand.passesTrackletAlgo(SDL::Default_TLAlgo)))
                // {
                //     continue;
                // }

                SDL::Tracklet tlCandOuter(&sgCand, outerTripletPtr->outerSegmentPtr());

                // Run the segment algo (supposedly is fast)
                tlCandOuter.runTrackletAlgo(SDL::Default_TLAlgo, logLevel_);

                if (not (tlCandOuter.passesTrackletAlgo(SDL::Default_TLAlgo)))
                {
                    continue;
                }

                SDL::TrackCandidate tcCand(innerTripletPtr, outerTripletPtr);

                // if (tcCand.passesTrackCandidateAlgo(algo))
                // {

                // Count the # of track candidates considered
                incrementNumberOfTrackCandidates(innerLowerModule);

                addTrackCandidateToLowerLayer(tcCand, 1, SDL::Layer::Barrel);
                // if (innerLowerModule.subdet() == SDL::Module::Barrel)
                //     addTrackCandidateToLowerLayer(tcCand, innerLowerModule.layer(), SDL::Layer::Barrel);
                // else
                //     addTrackCandidateToLowerLayer(tcCand, innerLowerModule.layer(), SDL::Layer::Endcap);

                // }

            }

        }

    }


}

void SDL::Event::createTrackCandidatesFromTracklets(TCAlgo algo)
{
    // Loop over lower modules
    for (auto& lowerModulePtr : getLowerModulePtrs())
    {

        // if (not (lowerModulePtr->layer() == 1))
        //     continue;

        // Create mini doublets
        createTrackCandidatesFromInnerModulesFromTracklets(lowerModulePtr->detId(), algo);

    }
}

void SDL::Event::createTrackCandidatesFromInnerModulesFromTracklets(unsigned int detId, SDL::TCAlgo algo)
{

    // Get reference to the inner lower Module
    Module& innerLowerModule = *getModule(detId);

    // Triple nested loops
    // Loop over inner lower module for segments
    for (auto& innerTrackletPtr : innerLowerModule.getTrackletPtrs())
    {

        // Get reference to segment in inner lower module
        SDL::Tracklet& innerTracklet = *innerTrackletPtr;

        // Get the outer mini-doublet module detId
        const SDL::Module& innerTrackletSecondModule = innerTracklet.innerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->getModule();

        unsigned int innerTrackletSecondModuleDetId = innerTrackletSecondModule.detId();

        // Get connected outer lower module detids
        const std::vector<unsigned int>& connectedModuleDetIds = moduleConnectionMap.getConnectedModuleDetIds(innerTrackletSecondModuleDetId);

        // Loop over connected outer lower modules
        for (auto& outerLowerModuleDetId : connectedModuleDetIds)
        {

            if (not hasModule(outerLowerModuleDetId))
                continue;

            // Get reference to the outer lower module
            Module& outerLowerModule = *getModule(outerLowerModuleDetId);

            // Loop over outer lower module mini-doublets
            for (auto& outerTrackletPtr : outerLowerModule.getTrackletPtrs())
            {

                SDL::Tracklet& outerTracklet = *outerTrackletPtr;

                SDL::TrackCandidate tcCand(innerTrackletPtr, outerTrackletPtr);

                tcCand.runTrackCandidateAlgo(algo, logLevel_);

                // Count the # of track candidates considered
                incrementNumberOfTrackCandidateCandidates(innerLowerModule);

                if (tcCand.passesTrackCandidateAlgo(algo))
                {

                    // Count the # of track candidates considered
                    incrementNumberOfTrackCandidates(innerLowerModule);

                    if (innerLowerModule.subdet() == SDL::Module::Barrel)
                        addTrackCandidateToLowerLayer(tcCand, innerLowerModule.layer(), SDL::Layer::Barrel);
                    else
                        addTrackCandidateToLowerLayer(tcCand, innerLowerModule.layer(), SDL::Layer::Endcap);
                }

            }

        }

    }

}*/

// Multiplicity of mini-doublets
unsigned int SDL::Event::getNumberOfHits() { return hits_.size(); }

// Multiplicity of mini-doublet candidates considered in this event
unsigned int SDL::Event::getNumberOfHitsByLayerBarrel(unsigned int ilayer) { return n_hits_by_layer_barrel_[ilayer]; }

// Multiplicity of mini-doublet candidates considered in this event
unsigned int SDL::Event::getNumberOfHitsByLayerEndcap(unsigned int ilayer) { return n_hits_by_layer_endcap_[ilayer]; }

// Multiplicity of mini-doublet candidates considered in this event
unsigned int SDL::Event::getNumberOfHitsByLayerBarrelUpperModule(unsigned int ilayer) { return n_hits_by_layer_barrel_upper_[ilayer]; }

// Multiplicity of mini-doublet candidates considered in this event
unsigned int SDL::Event::getNumberOfHitsByLayerEndcapUpperModule(unsigned int ilayer) { return n_hits_by_layer_endcap_upper_[ilayer]; }

// Multiplicity of mini-doublets
unsigned int SDL::Event::getNumberOfMiniDoublets() 
{
    unsigned int nMiniDoublets = 0;
    for(int i=0; i<lowerModuleMemoryCounter;i++)
    {
        for(int j = 0; j<mdMemoryCounter[i];j++)
        {
            nMiniDoublets++;
        }
    }
    return nMiniDoublets;
}

// Multiplicity of segments
unsigned int SDL::Event::getNumberOfSegments() { return *segmentMemoryCounter; }

// Multiplicity of tracklets
//unsigned int SDL::Event::getNumberOfTracklets() { return tracklets_.size(); }

// Multiplicity of triplets
//unsigned int SDL::Event::getNumberOfTriplets() { return triplets_.size(); }

// Multiplicity of track candidates
//unsigned int SDL::Event::getNumberOfTrackCandidates() { return trackcandidates_.size(); }

/*
// Multiplicity of mini-doublet candidates considered in this event
unsigned int SDL::Event::getNumberOfMiniDoubletCandidates() { unsigned int n = 0; for (unsigned int i = 0; i < 6; ++i) {n += n_miniDoublet_candidates_by_layer_barrel_[i];} for (unsigned int i = 0; i < 5; ++i) {n += n_miniDoublet_candidates_by_layer_endcap_[i];} return n; }

// Multiplicity of segment candidates considered in this event
unsigned int SDL::Event::getNumberOfSegmentCandidates() { unsigned int n = 0; for (unsigned int i = 0; i < 6; ++i) {n += n_segment_candidates_by_layer_barrel_[i];} for (unsigned int i = 0; i < 5; ++i) {n += n_segment_candidates_by_layer_endcap_[i];} return n; }

// Multiplicity of tracklet candidates considered in this event
unsigned int SDL::Event::getNumberOfTrackletCandidates() { unsigned int n = 0; for (unsigned int i = 0; i < 6; ++i) {n += n_tracklet_candidates_by_layer_barrel_[i];} for (unsigned int i = 0; i < 5; ++i) {n += n_tracklet_candidates_by_layer_endcap_[i];} return n; }

// Multiplicity of triplet candidates considered in this event
unsigned int SDL::Event::getNumberOfTripletCandidates() { unsigned int n = 0; for (unsigned int i = 0; i < 6; ++i) {n += n_triplet_candidates_by_layer_barrel_[i];} for (unsigned int i = 0; i < 5; ++i) {n += n_triplet_candidates_by_layer_endcap_[i];} return n; }

// Multiplicity of track candidate candidates considered in this event
unsigned int SDL::Event::getNumberOfTrackCandidateCandidates() { unsigned int n = 0; for (unsigned int i = 0; i < 6; ++i) {n += n_trackcandidate_candidates_by_layer_barrel_[i];} for (unsigned int i = 0; i < 5; ++i) {n += n_trackcandidate_candidates_by_layer_endcap_[i];} return n; }

// Multiplicity of mini-doublet candidates considered in this event
unsigned int SDL::Event::getNumberOfMiniDoubletCandidatesByLayerBarrel(unsigned int ilayer) { return n_miniDoublet_candidates_by_layer_barrel_[ilayer]; }

// Multiplicity of segment candidates considered in this event
unsigned int SDL::Event::getNumberOfSegmentCandidatesByLayerBarrel(unsigned int ilayer) { return n_segment_candidates_by_layer_barrel_[ilayer]; }

// Multiplicity of tracklet candidates considered in this event
unsigned int SDL::Event::getNumberOfTrackletCandidatesByLayerBarrel(unsigned int ilayer) { return n_tracklet_candidates_by_layer_barrel_[ilayer]; }

// Multiplicity of triplet candidates considered in this event
unsigned int SDL::Event::getNumberOfTripletCandidatesByLayerBarrel(unsigned int ilayer) { return n_triplet_candidates_by_layer_barrel_[ilayer]; }

// Multiplicity of track candidate candidates considered in this event
unsigned int SDL::Event::getNumberOfTrackCandidateCandidatesByLayerBarrel(unsigned int ilayer) { return n_trackcandidate_candidates_by_layer_barrel_[ilayer]; }

// Multiplicity of mini-doublet candidates considered in this event
unsigned int SDL::Event::getNumberOfMiniDoubletCandidatesByLayerEndcap(unsigned int ilayer) { return n_miniDoublet_candidates_by_layer_endcap_[ilayer]; }

// Multiplicity of segment candidates considered in this event
unsigned int SDL::Event::getNumberOfSegmentCandidatesByLayerEndcap(unsigned int ilayer) { return n_segment_candidates_by_layer_endcap_[ilayer]; }

// Multiplicity of tracklet candidates considered in this event
unsigned int SDL::Event::getNumberOfTrackletCandidatesByLayerEndcap(unsigned int ilayer) { return n_tracklet_candidates_by_layer_endcap_[ilayer]; }

// Multiplicity of triplet candidates considered in this event
unsigned int SDL::Event::getNumberOfTripletCandidatesByLayerEndcap(unsigned int ilayer) { return n_triplet_candidates_by_layer_endcap_[ilayer]; }

// Multiplicity of track candidate candidates considered in this event
unsigned int SDL::Event::getNumberOfTrackCandidateCandidatesByLayerEndcap(unsigned int ilayer) { return n_trackcandidate_candidates_by_layer_endcap_[ilayer]; }
*/
// Multiplicity of mini-doublet formed in this event
unsigned int SDL::Event::getNumberOfMiniDoubletsByLayerBarrel(unsigned int ilayer) { return n_miniDoublet_by_layer_barrel_[ilayer]; }

// Multiplicity of segment formed in this event
unsigned int SDL::Event::getNumberOfSegmentsByLayerBarrel(unsigned int ilayer) { return n_segment_by_layer_barrel_[ilayer]; }
/*
// Multiplicity of tracklet formed in this event
unsigned int SDL::Event::getNumberOfTrackletsByLayerBarrel(unsigned int ilayer) { return n_tracklet_by_layer_barrel_[ilayer]; }

// Multiplicity of triplet formed in this event
unsigned int SDL::Event::getNumberOfTripletsByLayerBarrel(unsigned int ilayer) { return n_triplet_by_layer_barrel_[ilayer]; }

// Multiplicity of track candidate formed in this event
unsigned int SDL::Event::getNumberOfTrackCandidatesByLayerBarrel(unsigned int ilayer) { return n_trackcandidate_by_layer_barrel_[ilayer]; }*/

// Multiplicity of mini-doublet formed in this event
unsigned int SDL::Event::getNumberOfMiniDoubletsByLayerEndcap(unsigned int ilayer) { return n_miniDoublet_by_layer_endcap_[ilayer]; }

/*
// Multiplicity of segment formed in this event
unsigned int SDL::Event::getNumberOfSegmentsByLayerEndcap(unsigned int ilayer) { return n_segment_by_layer_endcap_[ilayer]; }

// Multiplicity of tracklet formed in this event
unsigned int SDL::Event::getNumberOfTrackletsByLayerEndcap(unsigned int ilayer) { return n_tracklet_by_layer_endcap_[ilayer]; }

// Multiplicity of triplet formed in this event
unsigned int SDL::Event::getNumberOfTripletsByLayerEndcap(unsigned int ilayer) { return n_triplet_by_layer_endcap_[ilayer]; }

// Multiplicity of track candidate formed in this event
unsigned int SDL::Event::getNumberOfTrackCandidatesByLayerEndcap(unsigned int ilayer) { return n_trackcandidate_by_layer_endcap_[ilayer]; }*/

// Multiplicity of hits in this event
void SDL::Event::incrementNumberOfHits(SDL::Module& module)
{
    int layer = module.layer();
    int isbarrel = (module.subdet() == SDL::Module::Barrel);

    // Only count hits in lower module
    if (not module.isLower())
    {
        if (isbarrel)
            n_hits_by_layer_barrel_upper_[layer-1]++;
        else
            n_hits_by_layer_endcap_upper_[layer-1]++;
    }
    else
    {
        if (isbarrel)
            n_hits_by_layer_barrel_[layer-1]++;
        else
            n_hits_by_layer_endcap_[layer-1]++;
    }
}

// Multiplicity of mini-doublet candidates considered in this event
void SDL::Event::incrementNumberOfMiniDoubletCandidates(SDL::Module& module,int number)
{
    int layer = module.layer();
    int isbarrel = (module.subdet() == SDL::Module::Barrel);
    if (isbarrel)
        n_miniDoublet_candidates_by_layer_barrel_[layer-1]+=number;
    else
        n_miniDoublet_candidates_by_layer_endcap_[layer-1]+=number;
}

/*
// Multiplicity of segment candidates considered in this event
void SDL::Event::incrementNumberOfSegmentCandidates(SDL::Module& module)
{
    int layer = module.layer();
    int isbarrel = (module.subdet() == SDL::Module::Barrel);
    if (isbarrel)
        n_segment_candidates_by_layer_barrel_[layer-1]++;
    else
        n_segment_candidates_by_layer_endcap_[layer-1]++;
}

// Multiplicity of triplet candidates considered in this event
void SDL::Event::incrementNumberOfTripletCandidates(SDL::Module& module)
{
    int layer = module.layer();
    int isbarrel = (module.subdet() == SDL::Module::Barrel);
    if (isbarrel)
        n_triplet_candidates_by_layer_barrel_[layer-1]++;
    else
        n_triplet_candidates_by_layer_endcap_[layer-1]++;
}

// Multiplicity of tracklet candidates considered in this event
void SDL::Event::incrementNumberOfTrackletCandidates(SDL::Layer& _layer)
{
    int layer = _layer.layerIdx();
    int isbarrel = (_layer.subdet() == SDL::Layer::Barrel);
    if (isbarrel)
        n_tracklet_candidates_by_layer_barrel_[layer-1]++;
    else
        n_tracklet_candidates_by_layer_endcap_[layer-1]++;
}

// Multiplicity of tracklet candidates considered in this event
void SDL::Event::incrementNumberOfTrackletCandidates(SDL::Module& _module)
{
    int layer = _module.layer();
    int isbarrel = (_module.subdet() == SDL::Module::Barrel);
    if (isbarrel)
        n_tracklet_candidates_by_layer_barrel_[layer-1]++;
    else
        n_tracklet_candidates_by_layer_endcap_[layer-1]++;
}

// Multiplicity of track candidate candidates considered in this event
void SDL::Event::incrementNumberOfTrackCandidateCandidates(SDL::Layer& _layer)
{
    int layer = _layer.layerIdx();
    int isbarrel = (_layer.subdet() == SDL::Layer::Barrel);
    if (isbarrel)
        n_trackcandidate_candidates_by_layer_barrel_[layer-1]++;
    else
        n_trackcandidate_candidates_by_layer_endcap_[layer-1]++;
}

// Multiplicity of track candidate candidates considered in this event
void SDL::Event::incrementNumberOfTrackCandidateCandidates(SDL::Module& _module)
{
    int layer = _module.layer();
    int isbarrel = (_module.subdet() == SDL::Module::Barrel);
    if (isbarrel)
        n_trackcandidate_candidates_by_layer_barrel_[layer-1]++;
    else
        n_trackcandidate_candidates_by_layer_endcap_[layer-1]++;
}*/

// Multiplicity of mini-doublet formed in this event
void SDL::Event::incrementNumberOfMiniDoublets(SDL::Module& module)
{
    int layer = module.layer();
    int isbarrel = (module.subdet() == SDL::Module::Barrel);
    if (isbarrel)
        n_miniDoublet_by_layer_barrel_[layer-1]++;
    else
        n_miniDoublet_by_layer_endcap_[layer-1]++;
}

// Multiplicity of segment formed in this event

void SDL::Event::incrementNumberOfSegments(SDL::Module& module)
{
    int layer = module.layer();
    int isbarrel = (module.subdet() == SDL::Module::Barrel);
    if (isbarrel)
        n_segment_by_layer_barrel_[layer-1]++;
    else
        n_segment_by_layer_endcap_[layer-1]++;
}


// Multiplicity of triplet formed in this event
/*void SDL::Event::incrementNumberOfTriplets(SDL::Module& module)
{
    int layer = module.layer();
    int isbarrel = (module.subdet() == SDL::Module::Barrel);
    if (isbarrel)
        n_triplet_by_layer_barrel_[layer-1]++;
    else
        n_triplet_by_layer_endcap_[layer-1]++;
}

// Multiplicity of tracklet formed in this event
void SDL::Event::incrementNumberOfTracklets(SDL::Layer& _layer)
{
    int layer = _layer.layerIdx();
    int isbarrel = (_layer.subdet() == SDL::Layer::Barrel);
    if (isbarrel)
        n_tracklet_by_layer_barrel_[layer-1]++;
    else
        n_tracklet_by_layer_endcap_[layer-1]++;
}

// Multiplicity of tracklet formed in this event
void SDL::Event::incrementNumberOfTracklets(SDL::Module& _module)
{
    int layer = _module.layer();
    int isbarrel = (_module.subdet() == SDL::Module::Barrel);
    if (isbarrel)
        n_tracklet_by_layer_barrel_[layer-1]++;
    else
        n_tracklet_by_layer_endcap_[layer-1]++;
}

// Multiplicity of track candidate formed in this event
void SDL::Event::incrementNumberOfTrackCandidates(SDL::Layer& _layer)
{
    int layer = _layer.layerIdx();
    int isbarrel = (_layer.subdet() == SDL::Layer::Barrel);
    if (isbarrel)
        n_trackcandidate_by_layer_barrel_[layer-1]++;
    else
        n_trackcandidate_by_layer_endcap_[layer-1]++;
}

// Multiplicity of track candidate formed in this event
void SDL::Event::incrementNumberOfTrackCandidates(SDL::Module& _module)
{
    int layer = _module.layer();
    int isbarrel = (_module.subdet() == SDL::Module::Barrel);
    if (isbarrel)
        n_trackcandidate_by_layer_barrel_[layer-1]++;
    else
        n_trackcandidate_by_layer_endcap_[layer-1]++;
}*/

namespace SDL
{
    std::ostream& operator<<(std::ostream& out, const Event& event)
    {

        out << "" << std::endl;
        out << "==============" << std::endl;
        out << "Printing Event" << std::endl;
        out << "==============" << std::endl;
        out << "" << std::endl;

        for (auto& modulePtr : event.modulePtrs_)
        {
            out << modulePtr;
        }

/*        for (auto& layerPtr : event.layerPtrs_)
        {
            out << layerPtr;
        }*/

        return out;
    }

    std::ostream& operator<<(std::ostream& out, const Event* event)
    {
        out << *event;
        return out;
    }

}



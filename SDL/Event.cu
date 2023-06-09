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
    if(rangesInGPU != nullptr){delete rangesInGPU; delete rangesBuffers;}
    if(mdsInGPU != nullptr){delete mdsInGPU; delete miniDoubletsBuffers;}
    if(segmentsInGPU != nullptr){delete segmentsInGPU; delete segmentsBuffers;}
    if(tripletsInGPU!= nullptr){delete tripletsInGPU; delete tripletsBuffers;}
    if(trackCandidatesInGPU!= nullptr){delete trackCandidatesInGPU; delete trackCandidatesBuffers;}
    if(hitsInGPU!= nullptr){delete hitsInGPU; delete hitsBuffers;}
    if(pixelTripletsInGPU!= nullptr){delete pixelTripletsInGPU; delete pixelTripletsBuffers;}
    if(pixelQuintupletsInGPU!= nullptr){delete pixelQuintupletsInGPU; delete pixelQuintupletsBuffers;}
    if(quintupletsInGPU!= nullptr){delete quintupletsInGPU; delete quintupletsBuffers;}

    if(hitsInCPU != nullptr)
    {
        delete hitsInCPU;
    }
    if(rangesInCPU != nullptr)
    {
        delete rangesInCPU;
    }
    if(mdsInCPU != nullptr)
    {
        delete mdsInCPU;
    }
    if(segmentsInCPU != nullptr)
    {
        delete segmentsInCPU;
    }
    if(tripletsInCPU != nullptr)
    {
        delete tripletsInCPU;
    }
    if(quintupletsInCPU != nullptr)
    {
        delete quintupletsInCPU;
    }
    if(pixelTripletsInCPU != nullptr)
    {
        delete pixelTripletsInCPU;
    }
    if(pixelQuintupletsInCPU != nullptr)
    {
        delete pixelQuintupletsInCPU;
    }
    if(trackCandidatesInCPU != nullptr)
    {
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
    if(hitsInGPU){delete hitsInGPU; delete hitsBuffers;
      hitsInGPU = nullptr;}
    if(mdsInGPU){delete mdsInGPU; delete miniDoubletsBuffers;
      mdsInGPU = nullptr;}
    if(rangesInGPU){delete rangesInGPU; delete rangesBuffers;
      rangesInGPU = nullptr;}
    if(segmentsInGPU){delete segmentsInGPU; delete segmentsBuffers;
      segmentsInGPU = nullptr;}
    if(tripletsInGPU){delete tripletsInGPU; delete tripletsBuffers;
      tripletsInGPU = nullptr;}
    if(quintupletsInGPU){delete quintupletsInGPU; delete quintupletsBuffers;
      quintupletsInGPU = nullptr;}
    if(trackCandidatesInGPU){delete trackCandidatesInGPU; delete trackCandidatesBuffers;
      trackCandidatesInGPU = nullptr;}
    if(pixelTripletsInGPU){delete pixelTripletsInGPU; delete pixelTripletsBuffers;
      pixelTripletsInGPU = nullptr;}
    if(pixelQuintupletsInGPU){delete pixelQuintupletsInGPU; delete pixelQuintupletsBuffers;
      pixelQuintupletsInGPU = nullptr;}

    if(hitsInCPU != nullptr)
    {
        delete hitsInCPU;
        hitsInCPU = nullptr;
    }
    if(rangesInCPU != nullptr)
    {
        delete rangesInCPU;
        rangesInCPU = nullptr;
    }
    if(mdsInCPU != nullptr)
    {
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
        delete tripletsInCPU;
        tripletsInCPU = nullptr;
    }
    if(quintupletsInCPU != nullptr)
    {
        delete quintupletsInCPU;
        quintupletsInCPU = nullptr;
    }
    if(pixelTripletsInCPU != nullptr)
    {
        delete pixelTripletsInCPU;
        pixelTripletsInCPU = nullptr;
    }
    if(pixelQuintupletsInCPU != nullptr)
    {
        delete pixelQuintupletsInCPU;
        pixelQuintupletsInCPU = nullptr;
    }
    if(trackCandidatesInCPU != nullptr)
    {
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
}

void SDL::cleanModules()
{
    freeModules(*modulesInGPU, *pixelMapping);
    cudaFreeHost(modulesInGPU);
    cudaFreeHost(pixelMapping);
}

void SDL::Event::addHitToEvent(std::vector<float> x, std::vector<float> y, std::vector<float> z, std::vector<unsigned int> detId, std::vector<unsigned int> idxInNtuple)
{
    // Use the actual number of hits instead of a max.
    const int nHits = x.size();

    // Needed for the memcpy to hitsInGPU below.
    auto nHits_buf = allocBufWrapper<unsigned int>(devHost, 1);
    *alpaka::getPtrNative(nHits_buf) = nHits;

    // Get current device for future use.
    cudaGetDevice(&dev);

    // Initialize space on device/host for next event.
    if (hitsInGPU == nullptr)
    {
        hitsInGPU = new SDL::hits();
        hitsBuffers = new SDL::hitsBuffer<Acc>(nModules, nHits, devAcc, queue);
        hitsInGPU->setData(*hitsBuffers);
    }

    if (rangesInGPU == nullptr)
    {
        rangesInGPU = new SDL::objectRanges();
        rangesBuffers = new SDL::objectRangesBuffer<Acc>(nModules, nLowerModules, devAcc, queue);
        rangesInGPU->setData(*rangesBuffers);
    }

    // Copy the host arrays to the GPU.
    alpaka::memcpy(queue, hitsBuffers->xs_buf, x, nHits);
    alpaka::memcpy(queue, hitsBuffers->ys_buf, y, nHits);
    alpaka::memcpy(queue, hitsBuffers->zs_buf, z, nHits);
    alpaka::memcpy(queue, hitsBuffers->detid_buf, detId, nHits);
    alpaka::memcpy(queue, hitsBuffers->idxs_buf, idxInNtuple, nHits);
    alpaka::memcpy(queue, hitsBuffers->nHits_buf, nHits_buf, 1);
    alpaka::wait(queue);

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
        *modulesInGPU,
        *hitsInGPU,
        nHits));

    alpaka::enqueue(queue, hit_loop_task);

    Vec const threadsPerBlock2(static_cast<Idx>(1), static_cast<Idx>(1), static_cast<Idx>(256));
    Vec const blocksPerGrid2(static_cast<Idx>(1), static_cast<Idx>(1), static_cast<Idx>(MAX_BLOCKS));
    WorkDiv const module_ranges_workdiv(blocksPerGrid2, threadsPerBlock2, elementsPerThread);

    moduleRangesKernel module_ranges_kernel;
    auto const module_ranges_task(alpaka::createTaskKernel<Acc>(
        module_ranges_workdiv,
        module_ranges_kernel,
        *modulesInGPU,
        *hitsInGPU,
        nLowerModules));

    // Waiting isn't needed after second kernel call. Saves ~100 us.
    // This is because addPixelSegmentToEvent (which is run next) doesn't rely on hitsBuffers->hitrange variables.
    // Also, modulesInGPU->partnerModuleIndices is not alterned in addPixelSegmentToEvent.
    alpaka::enqueue(queue, module_ranges_task);
}

void SDL::Event::addPixelSegmentToEvent(std::vector<unsigned int> hitIndices0,std::vector<unsigned int> hitIndices1,std::vector<unsigned int> hitIndices2,std::vector<unsigned int> hitIndices3, std::vector<float> dPhiChange, std::vector<float> ptIn, std::vector<float> ptErr, std::vector<float> px, std::vector<float> py, std::vector<float> pz, std::vector<float> eta, std::vector<float> etaErr, std::vector<float> phi, std::vector<int> charge, std::vector<unsigned int> seedIdx, std::vector<int> superbin, std::vector<int8_t> pixelType, std::vector<char> isQuad)
{
    const int size = ptIn.size();
    unsigned int mdSize = 2 * size;
    uint16_t pixelModuleIndex = (*detIdToIndex)[1];

    if(mdsInGPU == nullptr)
    {
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
        nTotalMDs += N_MAX_PIXEL_MD_PER_MODULES;

        mdsInGPU = new SDL::miniDoublets();
        miniDoubletsBuffers = new SDL::miniDoubletsBuffer<Acc>(nTotalMDs, nLowerModules, devAcc, queue);
        mdsInGPU->setData(*miniDoubletsBuffers);

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

        segmentsInGPU = new SDL::segments();
        segmentsBuffers = new SDL::segmentsBuffer<Acc>(nTotalSegments, nLowerModules, N_MAX_PIXEL_SEGMENTS_PER_MODULE, devAcc, queue);
        segmentsInGPU->setData(*segmentsBuffers);

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

    alpaka::memcpy(queue, segmentsBuffers->ptIn_buf, ptIn, size);
    alpaka::memcpy(queue, segmentsBuffers->ptErr_buf, ptErr, size);
    alpaka::memcpy(queue, segmentsBuffers->px_buf, px, size);
    alpaka::memcpy(queue, segmentsBuffers->py_buf, py, size);
    alpaka::memcpy(queue, segmentsBuffers->pz_buf, pz, size);
    alpaka::memcpy(queue, segmentsBuffers->etaErr_buf, etaErr, size);
    alpaka::memcpy(queue, segmentsBuffers->isQuad_buf, isQuad, size);
    alpaka::memcpy(queue, segmentsBuffers->eta_buf, eta, size);
    alpaka::memcpy(queue, segmentsBuffers->phi_buf, phi, size);
    alpaka::memcpy(queue, segmentsBuffers->charge_buf, charge, size);
    alpaka::memcpy(queue, segmentsBuffers->seedIdx_buf, seedIdx, size);
    alpaka::memcpy(queue, segmentsBuffers->superbin_buf, superbin, size);
    alpaka::memcpy(queue, segmentsBuffers->pixelType_buf, pixelType, size);

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
        mdsInGPU = new SDL::miniDoublets();
        miniDoubletsBuffers = new SDL::miniDoubletsBuffer<Acc>(nTotalMDs, nLowerModules, devAcc, queue);
        mdsInGPU->setData(*miniDoubletsBuffers);
    }

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
    Vec const blocksPerGridCreateMDInGPU(static_cast<Idx>(1), static_cast<Idx>(MAX_BLOCKS), static_cast<Idx>(1));
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
        segmentsInGPU = new SDL::segments();
        segmentsBuffers = new SDL::segmentsBuffer<Acc>(nTotalSegments, nLowerModules, N_MAX_PIXEL_SEGMENTS_PER_MODULE, devAcc, queue);
        segmentsInGPU->setData(*segmentsBuffers);
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

        tripletsInGPU = new SDL::triplets();
        tripletsBuffers = new SDL::tripletsBuffer<Acc>(maxTriplets, nLowerModules, devAcc, queue);
        tripletsInGPU->setData(*tripletsBuffers);

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
        trackCandidatesInGPU = new SDL::trackCandidates();
        trackCandidatesBuffers = new SDL::trackCandidatesBuffer<Acc>(N_MAX_TRACK_CANDIDATES + N_MAX_PIXEL_TRACK_CANDIDATES, devAcc, queue);
        trackCandidatesInGPU->setData(*trackCandidatesBuffers);
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
        pixelTripletsInGPU = new SDL::pixelTriplets();
        pixelTripletsBuffers = new SDL::pixelTripletsBuffer<Acc>(N_MAX_PIXEL_TRIPLETS, devAcc, queue);
        pixelTripletsInGPU->setData(*pixelTripletsBuffers);
    }

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
        quintupletsInGPU = new SDL::quintuplets();
        quintupletsBuffers = new SDL::quintupletsBuffer<Acc>(nTotalQuintuplets, nLowerModules, devAcc, queue);
        quintupletsInGPU->setData(*quintupletsBuffers);

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
        pixelQuintupletsInGPU = new SDL::pixelQuintuplets();
        pixelQuintupletsBuffers = new SDL::pixelQuintupletsBuffer<Acc>(N_MAX_PIXEL_QUINTUPLETS, devAcc, queue);
        pixelQuintupletsInGPU->setData(*pixelQuintupletsBuffers);
    }
    if(trackCandidatesInGPU == nullptr)
    {
        trackCandidatesInGPU = new SDL::trackCandidates();
        trackCandidatesBuffers = new SDL::trackCandidatesBuffer<Acc>(N_MAX_TRACK_CANDIDATES + N_MAX_PIXEL_TRACK_CANDIDATES, devAcc, queue);
        trackCandidatesInGPU->setData(*trackCandidatesBuffers);
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

SDL::hitsBuffer<alpaka::DevCpu>* SDL::Event::getHits() //std::shared_ptr should take care of garbage collection
{
    if(hitsInCPU == nullptr)
    {
        auto nHits_buf = allocBufWrapper<unsigned int>(devHost, 1);
        alpaka::memcpy(queue, nHits_buf, hitsBuffers->nHits_buf, 1);
        alpaka::wait(queue);

        unsigned int nHits = *alpaka::getPtrNative(nHits_buf);
        hitsInCPU = new SDL::hitsBuffer<alpaka::DevCpu>(nModules, nHits, devHost, queue);
        hitsInCPU->setData(*hitsInCPU);

        *alpaka::getPtrNative(hitsInCPU->nHits_buf) = nHits;
        alpaka::memcpy(queue, hitsInCPU->idxs_buf, hitsBuffers->idxs_buf, nHits);
        alpaka::memcpy(queue, hitsInCPU->detid_buf, hitsBuffers->detid_buf, nHits);
        alpaka::memcpy(queue, hitsInCPU->xs_buf, hitsBuffers->xs_buf, nHits);
        alpaka::memcpy(queue, hitsInCPU->ys_buf, hitsBuffers->ys_buf, nHits);
        alpaka::memcpy(queue, hitsInCPU->zs_buf, hitsBuffers->zs_buf, nHits);
        alpaka::memcpy(queue, hitsInCPU->moduleIndices_buf, hitsBuffers->moduleIndices_buf, nHits);
        alpaka::wait(queue);
    }
    return hitsInCPU;
}

SDL::hitsBuffer<alpaka::DevCpu>* SDL::Event::getHitsInCMSSW()
{
    if(hitsInCPU == nullptr)
    {
        auto nHits_buf = allocBufWrapper<unsigned int>(devHost, 1);
        alpaka::memcpy(queue, nHits_buf, hitsBuffers->nHits_buf, 1);
        alpaka::wait(queue);

        unsigned int nHits = *alpaka::getPtrNative(nHits_buf);
        hitsInCPU = new SDL::hitsBuffer<alpaka::DevCpu>(nModules, nHits, devHost, queue);
        hitsInCPU->setData(*hitsInCPU);

        *alpaka::getPtrNative(hitsInCPU->nHits_buf) = nHits;
        alpaka::memcpy(queue, hitsInCPU->idxs_buf, hitsBuffers->idxs_buf, nHits);
        alpaka::wait(queue);
    }
    return hitsInCPU;
}

SDL::objectRangesBuffer<alpaka::DevCpu>* SDL::Event::getRanges()
{
    if(rangesInCPU == nullptr)
    {
        rangesInCPU = new SDL::objectRangesBuffer<alpaka::DevCpu>(nModules, nLowerModules, devHost, queue);
        rangesInCPU->setData(*rangesInCPU);

        alpaka::memcpy(queue, rangesInCPU->hitRanges_buf, rangesBuffers->hitRanges_buf, 2 * nModules);
        alpaka::memcpy(queue, rangesInCPU->quintupletModuleIndices_buf, rangesBuffers->quintupletModuleIndices_buf, nLowerModules);
        alpaka::memcpy(queue, rangesInCPU->miniDoubletModuleIndices_buf, rangesBuffers->miniDoubletModuleIndices_buf, nLowerModules + 1);
        alpaka::memcpy(queue, rangesInCPU->segmentModuleIndices_buf, rangesBuffers->segmentModuleIndices_buf, nLowerModules + 1);
        alpaka::memcpy(queue, rangesInCPU->tripletModuleIndices_buf, rangesBuffers->tripletModuleIndices_buf, nLowerModules);
        alpaka::wait(queue);
    }
    return rangesInCPU;
}

SDL::miniDoubletsBuffer<alpaka::DevCpu>* SDL::Event::getMiniDoublets()
{
    if(mdsInCPU == nullptr)
    {
        // Get nMemoryLocations parameter to initialize host based mdsInCPU
        auto nMemLocal_buf = allocBufWrapper<unsigned int>(devHost, 1);
        alpaka::memcpy(queue, nMemLocal_buf, miniDoubletsBuffers->nMemoryLocations_buf, 1);
        alpaka::wait(queue);

        unsigned int nMemLocal = *alpaka::getPtrNative(nMemLocal_buf);
        mdsInCPU = new SDL::miniDoubletsBuffer<alpaka::DevCpu>(nMemLocal, nLowerModules, devHost, queue);
        mdsInCPU->setData(*mdsInCPU);

        *alpaka::getPtrNative(mdsInCPU->nMemoryLocations_buf) = nMemLocal;
        alpaka::memcpy(queue, mdsInCPU->anchorHitIndices_buf, miniDoubletsBuffers->anchorHitIndices_buf, nMemLocal);
        alpaka::memcpy(queue, mdsInCPU->outerHitIndices_buf, miniDoubletsBuffers->outerHitIndices_buf, nMemLocal);
        alpaka::memcpy(queue, mdsInCPU->dphichanges_buf, miniDoubletsBuffers->dphichanges_buf, nMemLocal);
        alpaka::memcpy(queue, mdsInCPU->nMDs_buf, miniDoubletsBuffers->nMDs_buf, (nLowerModules+1));
        alpaka::memcpy(queue, mdsInCPU->totOccupancyMDs_buf, miniDoubletsBuffers->totOccupancyMDs_buf, (nLowerModules+1));
        alpaka::wait(queue);
    }
    return mdsInCPU;
}

SDL::segmentsBuffer<alpaka::DevCpu>* SDL::Event::getSegments()
{
    if(segmentsInCPU == nullptr)
    {
        // Get nMemoryLocations parameter to initilize host based segmentsInCPU
        auto nMemLocal_buf = allocBufWrapper<unsigned int>(devHost, 1);
        alpaka::memcpy(queue, nMemLocal_buf, segmentsBuffers->nMemoryLocations_buf, 1);
        alpaka::wait(queue);

        unsigned int nMemLocal = *alpaka::getPtrNative(nMemLocal_buf);
        segmentsInCPU = new SDL::segmentsBuffer<alpaka::DevCpu>(nMemLocal, nLowerModules, N_MAX_PIXEL_SEGMENTS_PER_MODULE, devHost, queue);
        segmentsInCPU->setData(*segmentsInCPU);

        *alpaka::getPtrNative(segmentsInCPU->nMemoryLocations_buf) = nMemLocal;
        alpaka::memcpy(queue, segmentsInCPU->nSegments_buf, segmentsBuffers->nSegments_buf, (nLowerModules+1));
        alpaka::memcpy(queue, segmentsInCPU->mdIndices_buf, segmentsBuffers->mdIndices_buf, 2 * nMemLocal);
        alpaka::memcpy(queue, segmentsInCPU->innerMiniDoubletAnchorHitIndices_buf, segmentsBuffers->innerMiniDoubletAnchorHitIndices_buf, nMemLocal);
        alpaka::memcpy(queue, segmentsInCPU->outerMiniDoubletAnchorHitIndices_buf, segmentsBuffers->outerMiniDoubletAnchorHitIndices_buf, nMemLocal);
        alpaka::memcpy(queue, segmentsInCPU->totOccupancySegments_buf, segmentsBuffers->totOccupancySegments_buf, (nLowerModules+1));
        alpaka::memcpy(queue, segmentsInCPU->ptIn_buf, segmentsBuffers->ptIn_buf, N_MAX_PIXEL_SEGMENTS_PER_MODULE);
        alpaka::memcpy(queue, segmentsInCPU->eta_buf, segmentsBuffers->eta_buf, N_MAX_PIXEL_SEGMENTS_PER_MODULE);
        alpaka::memcpy(queue, segmentsInCPU->phi_buf, segmentsBuffers->phi_buf, N_MAX_PIXEL_SEGMENTS_PER_MODULE);
        alpaka::memcpy(queue, segmentsInCPU->seedIdx_buf, segmentsBuffers->seedIdx_buf, N_MAX_PIXEL_SEGMENTS_PER_MODULE);
        alpaka::memcpy(queue, segmentsInCPU->isDup_buf, segmentsBuffers->isDup_buf, N_MAX_PIXEL_SEGMENTS_PER_MODULE);
        alpaka::memcpy(queue, segmentsInCPU->isQuad_buf, segmentsBuffers->isQuad_buf, N_MAX_PIXEL_SEGMENTS_PER_MODULE);
        alpaka::memcpy(queue, segmentsInCPU->score_buf, segmentsBuffers->score_buf, N_MAX_PIXEL_SEGMENTS_PER_MODULE);
        alpaka::wait(queue);
    }
    return segmentsInCPU;
}

SDL::tripletsBuffer<alpaka::DevCpu>* SDL::Event::getTriplets()
{
    if(tripletsInCPU == nullptr)
    {
        // Get nMemoryLocations parameter to initilize host based tripletsInCPU
        auto nMemLocal_buf = allocBufWrapper<unsigned int>(devHost, 1);
        alpaka::memcpy(queue, nMemLocal_buf, tripletsBuffers->nMemoryLocations_buf, 1);
        alpaka::wait(queue);

        unsigned int nMemLocal = *alpaka::getPtrNative(nMemLocal_buf);
        tripletsInCPU = new SDL::tripletsBuffer<alpaka::DevCpu>(nMemLocal, nLowerModules, devHost, queue);
        tripletsInCPU->setData(*tripletsInCPU);

        *alpaka::getPtrNative(tripletsInCPU->nMemoryLocations_buf) = nMemLocal;
#ifdef CUT_VALUE_DEBUG
        alpaka::memcpy(queue, tripletsInCPU->zOut_buf, tripletsBuffers->zOut_buf, 4 * nMemLocal);
        alpaka::memcpy(queue, tripletsInCPU->zLo_buf, tripletsBuffers->zLo_buf, nMemLocal);
        alpaka::memcpy(queue, tripletsInCPU->zHi_buf, tripletsBuffers->zHi_buf, nMemLocal);
        alpaka::memcpy(queue, tripletsInCPU->zLoPointed_buf, tripletsBuffers->zLoPointed_buf, 4 * nMemLocal);
        alpaka::memcpy(queue, tripletsInCPU->zHiPointed_buf, tripletsBuffers->zHiPointed_buf, nMemLocal);
        alpaka::memcpy(queue, tripletsInCPU->sdlCut_buf, tripletsBuffers->sdlCut_buf, nMemLocal);
        alpaka::memcpy(queue, tripletsInCPU->betaInCut_buf, tripletsBuffers->betaInCut_buf, nMemLocal);
        alpaka::memcpy(queue, tripletsInCPU->betaOutCut_buf, tripletsBuffers->betaOutCut_buf, nMemLocal);
        alpaka::memcpy(queue, tripletsInCPU->deltaBetaCut_buf, tripletsBuffers->deltaBetaCut_buf, nMemLocal);
        alpaka::memcpy(queue, tripletsInCPU->rtLo_buf, tripletsBuffers->rtLo_buf, nMemLocal);
        alpaka::memcpy(queue, tripletsInCPU->rtHi_buf, tripletsBuffers->rtHi_buf, nMemLocal);
        alpaka::memcpy(queue, tripletsInCPU->kZ_buf, tripletsBuffers->kZ_buf, nMemLocal);
#endif
        alpaka::memcpy(queue, tripletsInCPU->hitIndices_buf, tripletsBuffers->hitIndices_buf, 6 * nMemLocal);
        alpaka::memcpy(queue, tripletsInCPU->logicalLayers_buf, tripletsBuffers->logicalLayers_buf, 3 * nMemLocal);
        alpaka::memcpy(queue, tripletsInCPU->segmentIndices_buf, tripletsBuffers->segmentIndices_buf, 2 * nMemLocal);
        alpaka::memcpy(queue, tripletsInCPU->betaIn_buf, tripletsBuffers->betaIn_buf, nMemLocal);
        alpaka::memcpy(queue, tripletsInCPU->betaOut_buf, tripletsBuffers->betaOut_buf, nMemLocal);
        alpaka::memcpy(queue, tripletsInCPU->pt_beta_buf, tripletsBuffers->pt_beta_buf, nMemLocal);
        alpaka::memcpy(queue, tripletsInCPU->nTriplets_buf, tripletsBuffers->nTriplets_buf, nLowerModules);
        alpaka::memcpy(queue, tripletsInCPU->totOccupancyTriplets_buf, tripletsBuffers->totOccupancyTriplets_buf, nLowerModules);
        alpaka::wait(queue);
    }
    return tripletsInCPU;
}

SDL::quintupletsBuffer<alpaka::DevCpu>* SDL::Event::getQuintuplets()
{
    if(quintupletsInCPU == nullptr)
    {
        // Get nMemoryLocations parameter to initilize host based quintupletsInCPU
        auto nMemLocal_buf = allocBufWrapper<unsigned int>(devHost, 1);
        alpaka::memcpy(queue, nMemLocal_buf, quintupletsBuffers->nMemoryLocations_buf, 1);
        alpaka::wait(queue);

        unsigned int nMemLocal = *alpaka::getPtrNative(nMemLocal_buf);
        quintupletsInCPU = new SDL::quintupletsBuffer<alpaka::DevCpu>(nMemLocal, nLowerModules, devHost, queue);
        quintupletsInCPU->setData(*quintupletsInCPU);

        *alpaka::getPtrNative(quintupletsInCPU->nMemoryLocations_buf) = nMemLocal;
        alpaka::memcpy(queue, quintupletsInCPU->nQuintuplets_buf, quintupletsBuffers->nQuintuplets_buf, nLowerModules);
        alpaka::memcpy(queue, quintupletsInCPU->totOccupancyQuintuplets_buf, quintupletsBuffers->totOccupancyQuintuplets_buf, nLowerModules);
        alpaka::memcpy(queue, quintupletsInCPU->tripletIndices_buf, quintupletsBuffers->tripletIndices_buf, 2 * nMemLocal);
        alpaka::memcpy(queue, quintupletsInCPU->lowerModuleIndices_buf, quintupletsBuffers->lowerModuleIndices_buf, 5 * nMemLocal);
        alpaka::memcpy(queue, quintupletsInCPU->innerRadius_buf, quintupletsBuffers->innerRadius_buf, nMemLocal);
        alpaka::memcpy(queue, quintupletsInCPU->bridgeRadius_buf, quintupletsBuffers->bridgeRadius_buf, nMemLocal);
        alpaka::memcpy(queue, quintupletsInCPU->outerRadius_buf, quintupletsBuffers->outerRadius_buf, nMemLocal);
        alpaka::memcpy(queue, quintupletsInCPU->isDup_buf, quintupletsBuffers->isDup_buf, nMemLocal);
        alpaka::memcpy(queue, quintupletsInCPU->score_rphisum_buf, quintupletsBuffers->score_rphisum_buf, nMemLocal);
        alpaka::memcpy(queue, quintupletsInCPU->eta_buf, quintupletsBuffers->eta_buf, nMemLocal);
        alpaka::memcpy(queue, quintupletsInCPU->phi_buf, quintupletsBuffers->phi_buf, nMemLocal);
        alpaka::memcpy(queue, quintupletsInCPU->chiSquared_buf, quintupletsBuffers->chiSquared_buf, nMemLocal);
        alpaka::memcpy(queue, quintupletsInCPU->rzChiSquared_buf, quintupletsBuffers->rzChiSquared_buf, nMemLocal);
        alpaka::memcpy(queue, quintupletsInCPU->nonAnchorChiSquared_buf, quintupletsBuffers->nonAnchorChiSquared_buf, nMemLocal);
        alpaka::wait(queue);
    }
    return quintupletsInCPU;
}

SDL::pixelTripletsBuffer<alpaka::DevCpu>* SDL::Event::getPixelTriplets()
{
    if(pixelTripletsInCPU == nullptr)
    {
        // Get nMemoryLocations parameter to initilize host based quintupletsInCPU
        auto nPixelTriplets_buf = allocBufWrapper<int>(devHost, 1);
        alpaka::memcpy(queue, nPixelTriplets_buf, pixelTripletsBuffers->nPixelTriplets_buf, 1);
        alpaka::wait(queue);

        int nPixelTriplets = *alpaka::getPtrNative(nPixelTriplets_buf);
        pixelTripletsInCPU = new SDL::pixelTripletsBuffer<alpaka::DevCpu>(nPixelTriplets, devHost, queue);
        pixelTripletsInCPU->setData(*pixelTripletsInCPU);

        *alpaka::getPtrNative(pixelTripletsInCPU->nPixelTriplets_buf) = nPixelTriplets;
        alpaka::memcpy(queue, pixelTripletsInCPU->totOccupancyPixelTriplets_buf, pixelTripletsBuffers->totOccupancyPixelTriplets_buf, 1);
        alpaka::memcpy(queue, pixelTripletsInCPU->rzChiSquared_buf, pixelTripletsBuffers->rzChiSquared_buf, nPixelTriplets);
        alpaka::memcpy(queue, pixelTripletsInCPU->rPhiChiSquared_buf, pixelTripletsBuffers->rPhiChiSquared_buf, nPixelTriplets);
        alpaka::memcpy(queue, pixelTripletsInCPU->rPhiChiSquaredInwards_buf, pixelTripletsBuffers->rPhiChiSquaredInwards_buf, nPixelTriplets);
        alpaka::memcpy(queue, pixelTripletsInCPU->tripletIndices_buf, pixelTripletsBuffers->tripletIndices_buf, nPixelTriplets);
        alpaka::memcpy(queue, pixelTripletsInCPU->pixelSegmentIndices_buf, pixelTripletsBuffers->pixelSegmentIndices_buf, nPixelTriplets);
        alpaka::memcpy(queue, pixelTripletsInCPU->pixelRadius_buf, pixelTripletsBuffers->pixelRadius_buf, nPixelTriplets);
        alpaka::memcpy(queue, pixelTripletsInCPU->tripletRadius_buf, pixelTripletsBuffers->tripletRadius_buf, nPixelTriplets);
        alpaka::memcpy(queue, pixelTripletsInCPU->isDup_buf, pixelTripletsBuffers->isDup_buf, nPixelTriplets);
        alpaka::memcpy(queue, pixelTripletsInCPU->eta_buf, pixelTripletsBuffers->eta_buf, nPixelTriplets);
        alpaka::memcpy(queue, pixelTripletsInCPU->phi_buf, pixelTripletsBuffers->phi_buf, nPixelTriplets);
        alpaka::memcpy(queue, pixelTripletsInCPU->score_buf, pixelTripletsBuffers->score_buf, nPixelTriplets);
        alpaka::wait(queue);
    }
    return pixelTripletsInCPU;
}

SDL::pixelQuintupletsBuffer<alpaka::DevCpu>* SDL::Event::getPixelQuintuplets()
{
    if(pixelQuintupletsInCPU == nullptr)
    {
        // Get nMemoryLocations parameter to initilize host based quintupletsInCPU
        auto nPixelQuintuplets_buf = allocBufWrapper<int>(devHost, 1);
        alpaka::memcpy(queue, nPixelQuintuplets_buf, pixelQuintupletsBuffers->nPixelQuintuplets_buf, 1);
        alpaka::wait(queue);

        int nPixelQuintuplets = *alpaka::getPtrNative(nPixelQuintuplets_buf);
        pixelQuintupletsInCPU = new SDL::pixelQuintupletsBuffer<alpaka::DevCpu>(nPixelQuintuplets, devHost, queue);
        pixelQuintupletsInCPU->setData(*pixelQuintupletsInCPU);

        *alpaka::getPtrNative(pixelQuintupletsInCPU->nPixelQuintuplets_buf) = nPixelQuintuplets;
        alpaka::memcpy(queue, pixelQuintupletsInCPU->totOccupancyPixelQuintuplets_buf, pixelQuintupletsBuffers->totOccupancyPixelQuintuplets_buf, 1);
        alpaka::memcpy(queue, pixelQuintupletsInCPU->rzChiSquared_buf, pixelQuintupletsBuffers->rzChiSquared_buf, nPixelQuintuplets);
        alpaka::memcpy(queue, pixelQuintupletsInCPU->rPhiChiSquared_buf, pixelQuintupletsBuffers->rPhiChiSquared_buf, nPixelQuintuplets);
        alpaka::memcpy(queue, pixelQuintupletsInCPU->rPhiChiSquaredInwards_buf, pixelQuintupletsBuffers->rPhiChiSquaredInwards_buf, nPixelQuintuplets);
        alpaka::memcpy(queue, pixelQuintupletsInCPU->pixelIndices_buf, pixelQuintupletsBuffers->pixelIndices_buf, nPixelQuintuplets);
        alpaka::memcpy(queue, pixelQuintupletsInCPU->T5Indices_buf, pixelQuintupletsBuffers->T5Indices_buf, nPixelQuintuplets);
        alpaka::memcpy(queue, pixelQuintupletsInCPU->isDup_buf, pixelQuintupletsBuffers->isDup_buf, nPixelQuintuplets);
        alpaka::memcpy(queue, pixelQuintupletsInCPU->score_buf, pixelQuintupletsBuffers->score_buf, nPixelQuintuplets);
        alpaka::wait(queue);
    }
    return pixelQuintupletsInCPU;
}

SDL::trackCandidatesBuffer<alpaka::DevCpu>* SDL::Event::getTrackCandidates()
{
    if(trackCandidatesInCPU == nullptr)
    {
        // Get nTrackLocal parameter to initialize host based trackCandidatesInCPU
        auto nTrackLocal_buf = allocBufWrapper<int>(devHost, 1);
        alpaka::memcpy(queue, nTrackLocal_buf, trackCandidatesBuffers->nTrackCandidates_buf, 1);
        alpaka::wait(queue);

        int nTrackLocal = *alpaka::getPtrNative(nTrackLocal_buf);
        trackCandidatesInCPU = new SDL::trackCandidatesBuffer<alpaka::DevCpu>(N_MAX_TRACK_CANDIDATES + N_MAX_PIXEL_TRACK_CANDIDATES, devHost, queue);
        trackCandidatesInCPU->setData(*trackCandidatesInCPU);

        *alpaka::getPtrNative(trackCandidatesInCPU->nTrackCandidates_buf) = nTrackLocal;
        alpaka::memcpy(queue, trackCandidatesInCPU->hitIndices_buf, trackCandidatesBuffers->hitIndices_buf, 14 * nTrackLocal);
        alpaka::memcpy(queue, trackCandidatesInCPU->pixelSeedIndex_buf, trackCandidatesBuffers->pixelSeedIndex_buf, nTrackLocal);
        alpaka::memcpy(queue, trackCandidatesInCPU->logicalLayers_buf, trackCandidatesBuffers->logicalLayers_buf, 7 * nTrackLocal);
        alpaka::memcpy(queue, trackCandidatesInCPU->directObjectIndices_buf, trackCandidatesBuffers->directObjectIndices_buf, nTrackLocal);
        alpaka::memcpy(queue, trackCandidatesInCPU->objectIndices_buf, trackCandidatesBuffers->objectIndices_buf, 2 * nTrackLocal);
        alpaka::memcpy(queue, trackCandidatesInCPU->trackCandidateType_buf, trackCandidatesBuffers->trackCandidateType_buf, nTrackLocal);
        alpaka::wait(queue);
    }
    return trackCandidatesInCPU;
}

SDL::trackCandidatesBuffer<alpaka::DevCpu>* SDL::Event::getTrackCandidatesInCMSSW()
{
    if(trackCandidatesInCPU == nullptr)
    {
        // Get nTrackLocal parameter to initialize host based trackCandidatesInCPU
        auto nTrackLocal_buf = allocBufWrapper<int>(devHost, 1);
        alpaka::memcpy(queue, nTrackLocal_buf, trackCandidatesBuffers->nTrackCandidates_buf, 1);
        alpaka::wait(queue);

        int nTrackLocal = *alpaka::getPtrNative(nTrackLocal_buf);
        trackCandidatesInCPU = new SDL::trackCandidatesBuffer<alpaka::DevCpu>(N_MAX_TRACK_CANDIDATES + N_MAX_PIXEL_TRACK_CANDIDATES, devHost, queue);
        trackCandidatesInCPU->setData(*trackCandidatesInCPU);

        *alpaka::getPtrNative(trackCandidatesInCPU->nTrackCandidates_buf) = nTrackLocal;
        alpaka::memcpy(queue, trackCandidatesInCPU->hitIndices_buf, trackCandidatesBuffers->hitIndices_buf, 14 * nTrackLocal);
        alpaka::memcpy(queue, trackCandidatesInCPU->pixelSeedIndex_buf, trackCandidatesBuffers->pixelSeedIndex_buf, nTrackLocal);
        alpaka::memcpy(queue, trackCandidatesInCPU->trackCandidateType_buf, trackCandidatesBuffers->trackCandidateType_buf, nTrackLocal);
        alpaka::wait(queue);
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

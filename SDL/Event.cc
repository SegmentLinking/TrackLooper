#include "Event.h"

SDL::CPU::Event::Event() : logLevel_(SDL::CPU::Log_Nothing), pixelLayer_(0, 0)
{
    createLayers();
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

}

SDL::CPU::Event::~Event()
{
}

bool SDL::CPU::Event::hasModule(unsigned int detId)
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

void SDL::CPU::Event::setLogLevel(SDL::CPU::LogLevel logLevel)
{
    logLevel_ = logLevel;
}

SDL::CPU::Module& SDL::CPU::Event::getModule(unsigned int detId)
{
    // using std::map::emplace
    std::pair<std::map<unsigned int, Module>::iterator, bool> emplace_result = modulesMapByDetId_.emplace(detId, detId);

    // Retreive the module
    auto& inserted_or_existing = (*(emplace_result.first)).second;

    // If new was inserted, then insert to modulePtrs_ pointer list
    if (emplace_result.second) // if true, new was inserted
    {

        // The pointer to be added
        Module* module_ptr = &((*(emplace_result.first)).second);

        // Add the module pointer to the list of modules
        modulePtrs_.push_back(module_ptr);

        // If the module is lower module then add to list of lower modules
        if (module_ptr->isLower())
            lowerModulePtrs_.push_back(module_ptr);
    }

    return inserted_or_existing;
}

const std::vector<SDL::CPU::Module*> SDL::CPU::Event::getModulePtrs() const
{
    return modulePtrs_;
}

const std::vector<SDL::CPU::Module*> SDL::CPU::Event::getLowerModulePtrs() const
{
    return lowerModulePtrs_;
}

void SDL::CPU::Event::createLayers()
{
    // Create barrel layers
    for (int ilayer = SDL::CPU::Layer::BarrelLayer0; ilayer < SDL::CPU::Layer::nBarrelLayer; ++ilayer)
    {
        barrelLayers_[ilayer] = SDL::CPU::Layer(ilayer, SDL::CPU::Layer::Barrel);
        layerPtrs_.push_back(&(barrelLayers_[ilayer]));
    }

    // Create endcap layers
    for (int ilayer = SDL::CPU::Layer::EndcapLayer0; ilayer < SDL::CPU::Layer::nEndcapLayer; ++ilayer)
    {
        endcapLayers_[ilayer] = SDL::CPU::Layer(ilayer, SDL::CPU::Layer::Endcap);
        layerPtrs_.push_back(&(endcapLayers_[ilayer]));
    }
}

SDL::CPU::Layer& SDL::CPU::Event::getLayer(int ilayer, SDL::CPU::Layer::SubDet subdet)
{
    if (subdet == SDL::CPU::Layer::Barrel)
        return barrelLayers_[ilayer];
    else // if (subdet == SDL::CPU::Layer::Endcap)
        return endcapLayers_[ilayer];
}

SDL::CPU::Layer& SDL::CPU::Event::getPixelLayer()
{
    return pixelLayer_;
}

const std::vector<SDL::CPU::Layer*> SDL::CPU::Event::getLayerPtrs() const
{
    return layerPtrs_;
}

void SDL::CPU::Event::addHitToModule(SDL::CPU::Hit hit, unsigned int detId)
{
    // Add to global list of hits, where it will hold the object's instance
    hits_.push_back(hit);

    // And get the module (if not exists, then create), and add the address to Module.hits_
    getModule(detId).addHit(&(hits_.back()));

    // Count number of hits in the event
    incrementNumberOfHits(getModule(detId));

    // If the hit is 2S in the endcap then the hit boundary needs to be set
    if (getModule(detId).subdet() == SDL::CPU::Module::Endcap and getModule(detId).moduleType() == SDL::CPU::Module::TwoS)
    {
        hits_2s_edges_.push_back(GeometryUtil::stripHighEdgeHit(hits_.back()));
        hits_.back().setHitHighEdgePtr(&(hits_2s_edges_.back()));
        hits_2s_edges_.push_back(GeometryUtil::stripLowEdgeHit(hits_.back()));
        hits_.back().setHitLowEdgePtr(&(hits_2s_edges_.back()));
    }
}

void SDL::CPU::Event::addMiniDoubletToEvent(SDL::CPU::MiniDoublet md, unsigned int detId, int layerIdx, SDL::CPU::Layer::SubDet subdet)
{
    // Add to global list of mini doublets, where it will hold the object's instance
    miniDoublets_.push_back(md);

    // And get the module (if not exists, then create), and add the address to Module.hits_
    getModule(detId).addMiniDoublet(&(miniDoublets_.back()));

    // And get the layer
    getLayer(layerIdx, subdet).addMiniDoublet(&(miniDoublets_.back()));
}

[[deprecated("SDL::CPU:: addMiniDoubletToLowerModule() is deprecated. Use addMiniDoubletToEvent")]]
void SDL::CPU::Event::addMiniDoubletToLowerModule(SDL::CPU::MiniDoublet md, unsigned int detId)
{
    // Add to global list of mini doublets, where it will hold the object's instance
    miniDoublets_.push_back(md);

    // And get the module (if not exists, then create), and add the address to Module.hits_
    getModule(detId).addMiniDoublet(&(miniDoublets_.back()));
}

void SDL::CPU::Event::addSegmentToEvent(SDL::CPU::Segment sg, unsigned int detId, int layerIdx, SDL::CPU::Layer::SubDet subdet)
{
    // Add to global list of segments, where it will hold the object's instance
    segments_.push_back(sg);

    // And get the module (if not exists, then create), and add the address to Module.hits_
    getModule(detId).addSegment(&(segments_.back()));

    // And get the layer andd the segment to it
    getLayer(layerIdx, subdet).addSegment(&(segments_.back()));

    // Link segments to mini-doublets
    segments_.back().addSelfPtrToMiniDoublets();

}

void SDL::CPU::Event::addTrackletToEvent(SDL::CPU::Tracklet tl, unsigned int detId, int layerIdx, SDL::CPU::Layer::SubDet subdet)
{
    // Add to global list of segments, where it will hold the object's instance
    tracklets_.push_back(tl);

    // And get the module (if not exists, then create), and add the address to Module.hits_
    getModule(detId).addTracklet(&(tracklets_.back()));

    // And get the layer andd the segment to it
    if (layerIdx == 0)
        getPixelLayer().addTracklet(&(tracklets_.back()));
    else
        getLayer(layerIdx, subdet).addTracklet(&(tracklets_.back()));

    // Link segments to mini-doublets
    tracklets_.back().addSelfPtrToSegments();

}

void SDL::CPU::Event::addTripletToEvent(SDL::CPU::Triplet tp, unsigned int detId, int layerIdx, SDL::CPU::Layer::SubDet subdet)
{
    // Add to global list of segments, where it will hold the object's instance
    triplets_.push_back(tp);

    // And get the module (if not exists, then create), and add the address to Module.hits_
    getModule(detId).addTriplet(&(triplets_.back()));

    // And get the layer andd the triplet to it
    getLayer(layerIdx, subdet).addTriplet(&(triplets_.back()));
}

[[deprecated("SDL::CPU:: addSegmentToLowerModule() is deprecated. Use addSegmentToEvent")]]
void SDL::CPU::Event::addSegmentToLowerModule(SDL::CPU::Segment sg, unsigned int detId)
{
    // Add to global list of segments, where it will hold the object's instance
    segments_.push_back(sg);

    // And get the module (if not exists, then create), and add the address to Module.hits_
    getModule(detId).addSegment(&(segments_.back()));
}

[[deprecated("SDL::CPU:: addSegmentToLowerLayer() is deprecated. Use addSegmentToEvent")]]
void SDL::CPU::Event::addSegmentToLowerLayer(SDL::CPU::Segment sg, int layerIdx, SDL::CPU::Layer::SubDet subdet)
{
    // Add to global list of segments, where it will hold the object's instance
    segments_.push_back(sg);

    // And get the layer
    getLayer(layerIdx, subdet).addSegment(&(segments_.back()));
}

void SDL::CPU::Event::addTrackletToLowerLayer(SDL::CPU::Tracklet tl, int layerIdx, SDL::CPU::Layer::SubDet subdet)
{
    // Add to global list of tracklets, where it will hold the object's instance
    tracklets_.push_back(tl);

    // And get the layer
    getLayer(layerIdx, subdet).addTracklet(&(tracklets_.back()));
}

void SDL::CPU::Event::addTrackCandidateToLowerLayer(SDL::CPU::TrackCandidate tc, int layerIdx, SDL::CPU::Layer::SubDet subdet)
{
    // Add to global list of trackcandidates, where it will hold the object's instance
    trackcandidates_.push_back(tc);

    // And get the layer
    if (layerIdx == 0)
        getPixelLayer().addTrackCandidate(&(trackcandidates_.back()));
    else
        getLayer(layerIdx, subdet).addTrackCandidate(&(trackcandidates_.back()));
}

void SDL::CPU::Event::addPixelSegmentsToEvent(std::vector<SDL::CPU::Hit> hits, float dPhiChange, float ptIn, float ptErr, float px, float py, float pz, float etaErr, int iSeed)
{
    // detId = 1 module means grand "pixel module" where everything related to pixel hits/md/segments will be stored to
    Module& pixelModule = getModule(1);

    // Assert that we provided a quadruplet pixel segment
    assert(hits.size() == 4);
    pixel_hits_.push_back(hits[0]);
    SDL::CPU::Hit* hit0_ptr = &pixel_hits_.back();
    pixel_hits_.push_back(hits[1]);
    SDL::CPU::Hit* hit1_ptr = &pixel_hits_.back();
    pixel_hits_.push_back(hits[2]);
    SDL::CPU::Hit* hit2_ptr = &pixel_hits_.back();
    pixel_hits_.push_back(hits[3]);
    SDL::CPU::Hit* hit3_ptr = &pixel_hits_.back();

    // Add hits to the pixel module
    pixelModule.addHit(hit0_ptr);
    pixelModule.addHit(hit1_ptr);
    pixelModule.addHit(hit2_ptr);
    pixelModule.addHit(hit3_ptr);

    // Create MiniDoublets
    SDL::CPU::MiniDoublet innerMiniDoublet(hit0_ptr, hit1_ptr);
    SDL::CPU::MiniDoublet outerMiniDoublet(hit2_ptr, hit3_ptr);
    // innerMiniDoublet.runMiniDoubletAllCombAlgo(); // Setting "all combination" pass in order to flag the pass bool flag
    // outerMiniDoublet.runMiniDoubletAllCombAlgo(); // Setting "all combination" pass in order to flag the pass bool flag

    // std::cout << "debugging pixel line segment hit loading" << std::endl;
    // std::cout << innerMiniDoublet.anchorHitPtr() << std::endl;
    // std::cout << innerMiniDoublet.upperHitPtr() << std::endl;
    // std::cout << innerMiniDoublet.lowerHitPtr() << std::endl;
    // std::cout << outerMiniDoublet.anchorHitPtr() << std::endl;
    // std::cout << outerMiniDoublet.upperHitPtr() << std::endl;
    // std::cout << outerMiniDoublet.lowerHitPtr() << std::endl;

    pixel_miniDoublets_.push_back(innerMiniDoublet);
    SDL::CPU::MiniDoublet* innerMiniDoubletPtr = &pixel_miniDoublets_.back();
    pixel_miniDoublets_.push_back(outerMiniDoublet);
    SDL::CPU::MiniDoublet* outerMiniDoubletPtr = &pixel_miniDoublets_.back();

    // Create Segments
    segments_.push_back(SDL::CPU::Segment(innerMiniDoubletPtr, outerMiniDoubletPtr));
    SDL::CPU::Segment* pixelSegmentPtr = &segments_.back();

    // Set the deltaPhiChange
    pixelSegmentPtr->setDeltaPhiChange(dPhiChange);
    pixelSegmentPtr->setRecoVars("ptIn", ptIn);
    pixelSegmentPtr->setRecoVars("ptErr", ptErr);
    pixelSegmentPtr->setRecoVars("px", px);
    pixelSegmentPtr->setRecoVars("py", py);
    pixelSegmentPtr->setRecoVars("pz", pz);
    pixelSegmentPtr->setRecoVars("etaErr", etaErr);
    pixelSegmentPtr->setRecoVars("iSeed", iSeed);

    getPixelLayer().addSegment(pixelSegmentPtr);

}

void SDL::CPU::Event::createMiniDoublets(MDAlgo algo)
{
    // Loop over lower modules
    for (auto& lowerModulePtr : getLowerModulePtrs())
    {
        lowerModulePtr->detId();
        // Create mini doublets
        createMiniDoubletsFromLowerModule(lowerModulePtr->detId(), algo);

    }
}

void SDL::CPU::Event::createMiniDoubletsFromLowerModule(unsigned int detId, SDL::CPU::MDAlgo algo)
{
    // Get reference to the lower Module
    Module& lowerModule = getModule(detId);

    // Get reference to the upper Module
    Module& upperModule = getModule(lowerModule.partnerDetId());

    // Double nested loops
    // Loop over lower module hits
    for (auto& lowerHitPtr : lowerModule.getHitPtrs())
    {

        // Get reference to lower Hit
        SDL::CPU::Hit& lowerHit = *lowerHitPtr;

        // Loop over upper module hits
        for (auto& upperHitPtr : upperModule.getHitPtrs())
        {

            // Get reference to upper Hit
            SDL::CPU::Hit& upperHit = *upperHitPtr;

            // Create a mini-doublet candidate
            SDL::CPU::MiniDoublet mdCand(lowerHitPtr, upperHitPtr);

            // Count the number of mdCand considered
            incrementNumberOfMiniDoubletCandidates(lowerModule);

            // Run mini-doublet algorithm on mdCand (mini-doublet candidate)
            mdCand.runMiniDoubletAlgo(algo, logLevel_);

            if (mdCand.passesMiniDoubletAlgo(algo))
            {

                // Count the number of md formed
                incrementNumberOfMiniDoublets(lowerModule);

                if (lowerModule.subdet() == SDL::CPU::Module::Barrel)
                    addMiniDoubletToEvent(mdCand, lowerModule.detId(), lowerModule.layer(), SDL::CPU::Layer::Barrel);
                else
                    addMiniDoubletToEvent(mdCand, lowerModule.detId(), lowerModule.layer(), SDL::CPU::Layer::Endcap);
            }

        }

    }

}

void SDL::CPU::Event::createPseudoMiniDoubletsFromAnchorModule(SDL::CPU::MDAlgo algo)
{

    // Loop over lower modules
    for (auto& lowerModulePtr : getLowerModulePtrs())
    {

        unsigned int detId = lowerModulePtr->detId();

        // Get reference to the lower Module
        Module& lowerModule = getModule(detId);

        // Assign anchor hit pointers based on their hit type
        bool loopLower = true;
        if (lowerModule.moduleType() == SDL::CPU::Module::PS)
        {
            if (lowerModule.moduleLayerType() == SDL::CPU::Module::Pixel)
            {
                loopLower = true;
            }
            else
            {
                loopLower = false;
            }
        }
        else
        {
            loopLower = true;
        }

        // Get reference to the upper Module
        Module& upperModule = getModule(lowerModule.partnerDetId());

        if (loopLower)
        {
            // Loop over lower module hits
            for (auto& lowerHitPtr : lowerModule.getHitPtrs())
            {
                // Get reference to lower Hit
                SDL::CPU::Hit& lowerHit = *lowerHitPtr;

                // Loop over upper module hits
                for (auto& upperHitPtr : upperModule.getHitPtrs())
                {

                    // Get reference to upper Hit
                    SDL::CPU::Hit& upperHit = *upperHitPtr;

                    // Create a mini-doublet candidate
                    SDL::CPU::MiniDoublet mdCand(lowerHitPtr, upperHitPtr);

                    // Count the number of mdCand considered
                    incrementNumberOfMiniDoubletCandidates(lowerModule);

                    // Run mini-doublet algorithm on mdCand (mini-doublet candidate)
                    mdCand.runMiniDoubletAlgo(SDL::CPU::AllComb_MDAlgo, logLevel_);

                    if (mdCand.passesMiniDoubletAlgo(SDL::CPU::AllComb_MDAlgo))
                    {

                        // Count the number of md formed
                        incrementNumberOfMiniDoublets(lowerModule);

                        if (lowerModule.subdet() == SDL::CPU::Module::Barrel)
                            addMiniDoubletToEvent(mdCand, lowerModule.detId(), lowerModule.layer(), SDL::CPU::Layer::Barrel);
                        else
                            addMiniDoubletToEvent(mdCand, lowerModule.detId(), lowerModule.layer(), SDL::CPU::Layer::Endcap);

                        // Break to exit on first pseudo mini-doublet
                        break;
                    }

                }

            }

        }
        else
        {
            // Loop over lower module hits
            for (auto& upperHitPtr : upperModule.getHitPtrs())
            {
                // Get reference to upper Hit
                SDL::CPU::Hit& upperHit = *upperHitPtr;

                // Loop over upper module hits
                for (auto& lowerHitPtr : lowerModule.getHitPtrs())
                {

                    // Get reference to lower Hit
                    SDL::CPU::Hit& lowerHit = *lowerHitPtr;

                    // Create a mini-doublet candidate
                    SDL::CPU::MiniDoublet mdCand(lowerHitPtr, upperHitPtr);

                    // Count the number of mdCand considered
                    incrementNumberOfMiniDoubletCandidates(lowerModule);

                    // Run mini-doublet algorithm on mdCand (mini-doublet candidate)
                    mdCand.runMiniDoubletAlgo(SDL::CPU::AllComb_MDAlgo, logLevel_);

                    if (mdCand.passesMiniDoubletAlgo(SDL::CPU::AllComb_MDAlgo))
                    {

                        // Count the number of md formed
                        incrementNumberOfMiniDoublets(lowerModule);

                        if (lowerModule.subdet() == SDL::CPU::Module::Barrel)
                            addMiniDoubletToEvent(mdCand, lowerModule.detId(), lowerModule.layer(), SDL::CPU::Layer::Barrel);
                        else
                            addMiniDoubletToEvent(mdCand, lowerModule.detId(), lowerModule.layer(), SDL::CPU::Layer::Endcap);

                        // Break to exit on first pseudo mini-doublet
                        break;
                    }

                }

            }

        }
    }

}

void SDL::CPU::Event::createSegments(SGAlgo algo)
{

    for (auto& segment_compatible_layer_pair : SDL::CPU::Layer::getListOfSegmentCompatibleLayerPairs())
    {
        int innerLayerIdx = segment_compatible_layer_pair.first.first;
        SDL::CPU::Layer::SubDet innerLayerSubDet = segment_compatible_layer_pair.first.second;
        int outerLayerIdx = segment_compatible_layer_pair.second.first;
        SDL::CPU::Layer::SubDet outerLayerSubDet = segment_compatible_layer_pair.second.second;
        createSegmentsFromTwoLayers(innerLayerIdx, innerLayerSubDet, outerLayerIdx, outerLayerSubDet, algo);
    }
}

void SDL::CPU::Event::createSegmentsWithModuleMap(SGAlgo algo)
{
    // Loop over lower modules
    for (auto& lowerModulePtr : getLowerModulePtrs())
    {

        // Create mini doublets
        createSegmentsFromInnerLowerModule(lowerModulePtr->detId(), algo);

    }
}

void SDL::CPU::Event::createSegmentsFromInnerLowerModule(unsigned int detId, SDL::CPU::SGAlgo algo)
{

    // x's and y's are mini doublets
    // -------x--------
    // --------x------- <--- outer lower module
    //
    // --------y-------
    // -------y-------- <--- inner lower module

    // Get reference to the inner lower Module
    Module& innerLowerModule = getModule(detId);

    // Triple nested loops
    // Loop over inner lower module mini-doublets
    for (auto& innerMiniDoubletPtr : innerLowerModule.getMiniDoubletPtrs())
    {

        // Get reference to mini-doublet in inner lower module
        SDL::CPU::MiniDoublet& innerMiniDoublet = *innerMiniDoubletPtr;

        // Get connected outer lower module detids
        const std::vector<unsigned int>& connectedModuleDetIds = moduleConnectionMap.getConnectedModuleDetIds(detId);

        // Loop over connected outer lower modules
        for (auto& outerLowerModuleDetId : connectedModuleDetIds)
        {

            if (not hasModule(outerLowerModuleDetId))
                continue;

            // Get reference to the outer lower module
            Module& outerLowerModule = getModule(outerLowerModuleDetId);

            // Loop over outer lower module mini-doublets
            for (auto& outerMiniDoubletPtr : outerLowerModule.getMiniDoubletPtrs())
            {

                // Get reference to mini-doublet in outer lower module
                SDL::CPU::MiniDoublet& outerMiniDoublet = *outerMiniDoubletPtr;

                // Create a segment candidate
                SDL::CPU::Segment sgCand(innerMiniDoubletPtr, outerMiniDoubletPtr);

                // Run segment algorithm on sgCand (segment candidate)
                sgCand.runSegmentAlgo(algo, logLevel_);

                // Count the # of sgCands considered by layer
                incrementNumberOfSegmentCandidates(innerLowerModule);

                if (sgCand.passesSegmentAlgo(algo))
                {

                    // Count the # of sg formed by layer
                    incrementNumberOfSegments(innerLowerModule);

                    if (innerLowerModule.subdet() == SDL::CPU::Module::Barrel)
                        addSegmentToEvent(sgCand, innerLowerModule.detId(), innerLowerModule.layer(), SDL::CPU::Layer::Barrel);
                    else
                        addSegmentToEvent(sgCand, innerLowerModule.detId(), innerLowerModule.layer(), SDL::CPU::Layer::Endcap);
                }

            }
        }
    }
}

void SDL::CPU::Event::createTriplets(TPAlgo algo)
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

void SDL::CPU::Event::createTripletsFromInnerLowerModule(unsigned int detId, SDL::CPU::TPAlgo algo)
{

    // Get reference to the inner lower Module
    Module& innerLowerModule = getModule(detId);

    // Triple nested loops
    // Loop over inner lower module for segments
    for (auto& innerSegmentPtr : innerLowerModule.getSegmentPtrs())
    {

        // Get reference to segment in inner lower module
        SDL::CPU::Segment& innerSegment = *innerSegmentPtr;

        // Get connected outer lower module detids
        const std::vector<unsigned int>& connectedModuleDetIds = moduleConnectionMap.getConnectedModuleDetIds(detId);

        // Loop over connected outer lower modules
        for (auto& outerLowerModuleDetId : connectedModuleDetIds)
        {

            if (not hasModule(outerLowerModuleDetId))
                continue;

            // Get reference to the outer lower module
            Module& outerLowerModule = getModule(outerLowerModuleDetId);

            // Loop over outer lower module mini-doublets
            for (auto& outerSegmentPtr : outerLowerModule.getSegmentPtrs())
            {

                // Get reference to mini-doublet in outer lower module
                SDL::CPU::Segment& outerSegment = *outerSegmentPtr;

                // Create a segment candidate
                SDL::CPU::Triplet tpCand(innerSegmentPtr, outerSegmentPtr);

                // Run segment algorithm on tpCand (segment candidate)
                tpCand.runTripletAlgo(algo, logLevel_);

                // Count the # of tpCands considered by layer
                incrementNumberOfTripletCandidates(innerLowerModule);

                if (tpCand.passesTripletAlgo(algo))
                {

                    // Count the # of sg formed by layer
                    incrementNumberOfTriplets(innerLowerModule);

                    if (innerLowerModule.subdet() == SDL::CPU::Module::Barrel)
                        addTripletToEvent(tpCand, innerLowerModule.detId(), innerLowerModule.layer(), SDL::CPU::Layer::Barrel);
                    else
                        addTripletToEvent(tpCand, innerLowerModule.detId(), innerLowerModule.layer(), SDL::CPU::Layer::Endcap);
                }

            }
        }
    }
}

void SDL::CPU::Event::createTracklets(TLAlgo algo)
{
    for (auto& tracklet_compatible_layer_pair : SDL::CPU::Layer::getListOfTrackletCompatibleLayerPairs())
    {
        int innerLayerIdx = tracklet_compatible_layer_pair.first.first;
        SDL::CPU::Layer::SubDet innerLayerSubDet = tracklet_compatible_layer_pair.first.second;
        int outerLayerIdx = tracklet_compatible_layer_pair.second.first;
        SDL::CPU::Layer::SubDet outerLayerSubDet = tracklet_compatible_layer_pair.second.second;
        createTrackletsFromTwoLayers(innerLayerIdx, innerLayerSubDet, outerLayerIdx, outerLayerSubDet, algo);
    }
}

void SDL::CPU::Event::createTrackletsWithModuleMap(TLAlgo algo)
{
    if (logLevel_ == SDL::CPU::Log_Debug)
        SDL::CPU::cout << "SDL::CPU::Event::createTrackletsWithModuleMap()" << std::endl;

    // Loop over lower modules
    int nModuleProcessed = 0;
    int nTotalLowerModule = getLowerModulePtrs().size();

    if (logLevel_ == SDL::CPU::Log_Debug)
        SDL::CPU::cout <<  " nTotalLowerModule: " << nTotalLowerModule <<  std::endl;

    for (auto& lowerModulePtr : getLowerModulePtrs())
    {

        if (logLevel_ == SDL::CPU::Log_Debug)
            if (nModuleProcessed % 1000 == 0)
                SDL::CPU::cout <<  "    nModuleProcessed: " << nModuleProcessed <<  std::endl;

        if (logLevel_ == SDL::CPU::Log_Debug)
        {
            std::cout <<  " lowerModulePtr->subdet(): " << lowerModulePtr->subdet() <<  std::endl;
            std::cout <<  " lowerModulePtr->layer(): " << lowerModulePtr->layer() <<  std::endl;
            std::cout <<  " lowerModulePtr->getSegmentPtrs().size(): " << lowerModulePtr->getSegmentPtrs().size() <<  std::endl;
        }

        // if (lowerModulePtr->layer() != 1)
        //     continue;

        // Create mini doublets
        createTrackletsFromInnerLowerModule(lowerModulePtr->detId(), algo);

        nModuleProcessed++;

    }
}

// Create tracklets from inner modules
void SDL::CPU::Event::createTrackletsFromInnerLowerModule(unsigned int detId, SDL::CPU::TLAlgo algo)
{

    // Get reference to the inner lower Module
    Module& innerLowerModule = getModule(detId);

    // Triple nested loops
    // Loop over inner lower module for segments
    for (auto& innerSegmentPtr : innerLowerModule.getSegmentPtrs())
    {

        // Get reference to segment in inner lower module
        SDL::CPU::Segment& innerSegment = *innerSegmentPtr;

        // Get the outer mini-doublet module detId
        const SDL::CPU::Module& innerSegmentOuterModule = innerSegment.outerMiniDoubletPtr()->lowerHitPtr()->getModule();

        unsigned int innerSegmentOuterModuleDetId = innerSegmentOuterModule.detId();

        // Get connected outer lower module detids
        const std::vector<unsigned int>& connectedModuleDetIds = moduleConnectionMap.getConnectedModuleDetIds(innerSegmentOuterModuleDetId);

        // Loop over connected outer lower modules
        for (auto& outerLowerModuleDetId : connectedModuleDetIds)
        {

            if (not hasModule(outerLowerModuleDetId))
                continue;

            // Get reference to the outer lower module
            Module& outerLowerModule = getModule(outerLowerModuleDetId);

            // Loop over outer lower module mini-doublets
            for (auto& outerSegmentPtr : outerLowerModule.getSegmentPtrs())
            {

                // Count the # of tlCands considered by layer
                incrementNumberOfTrackletCandidates(innerLowerModule);

                // Get reference to mini-doublet in outer lower module
                SDL::CPU::Segment& outerSegment = *outerSegmentPtr;

                // Create a tracklet candidate
                SDL::CPU::Tracklet tlCand(innerSegmentPtr, outerSegmentPtr);

                // Run segment algorithm on tlCand (tracklet candidate)
                tlCand.runTrackletAlgo(algo, logLevel_);

                if (tlCand.passesTrackletAlgo(algo))
                {

                    // Count the # of sg formed by layer
                    incrementNumberOfTracklets(innerLowerModule);

                    if (innerLowerModule.subdet() == SDL::CPU::Module::Barrel)
                        addTrackletToEvent(tlCand, innerLowerModule.detId(), innerLowerModule.layer(), SDL::CPU::Layer::Barrel);
                    else
                        addTrackletToEvent(tlCand, innerLowerModule.detId(), innerLowerModule.layer(), SDL::CPU::Layer::Endcap);
                }

            }

        }

    }

}

// Create tracklets
void SDL::CPU::Event::createTrackletsWithAGapWithModuleMap(TLAlgo algo)
{
    if (logLevel_ == SDL::CPU::Log_Debug)
        SDL::CPU::cout << "SDL::CPU::Event::createTrackletsWithAGapWithModuleMap()" << std::endl;

    // Loop over lower modules
    int nModuleProcessed = 0;
    int nTotalLowerModule = getLowerModulePtrs().size();

    if (logLevel_ == SDL::CPU::Log_Debug)
        SDL::CPU::cout <<  " nTotalLowerModule: " << nTotalLowerModule <<  std::endl;

    // Loop over lower modules
    for (auto& lowerModulePtr : getLowerModulePtrs())
    {

        if (logLevel_ == SDL::CPU::Log_Debug)
            if (nModuleProcessed % 1000 == 0)
                SDL::CPU::cout <<  "    nModuleProcessed: " << nModuleProcessed <<  std::endl;

        if (logLevel_ == SDL::CPU::Log_Debug)
        {
            std::cout <<  " lowerModulePtr->subdet(): " << lowerModulePtr->subdet() <<  std::endl;
            std::cout <<  " lowerModulePtr->layer(): " << lowerModulePtr->layer() <<  std::endl;
            std::cout <<  " lowerModulePtr->getSegmentPtrs().size(): " << lowerModulePtr->getSegmentPtrs().size() <<  std::endl;
        }

        // if (lowerModulePtr->layer() != 1)
        //     continue;

        // Create mini doublets
        createTrackletsWithAGapFromInnerLowerModule(lowerModulePtr->detId(), algo);

        nModuleProcessed++;

    }
}

// Create tracklets from inner modules
void SDL::CPU::Event::createTrackletsWithAGapFromInnerLowerModule(unsigned int detId, SDL::CPU::TLAlgo algo)
{

    // Get reference to the inner lower Module
    Module& innerLowerModule = getModule(detId);

    // Triple nested loops
    // Loop over inner lower module for segments
    for (auto& innerSegmentPtr : innerLowerModule.getSegmentPtrs())
    {

        // Get reference to segment in inner lower module
        SDL::CPU::Segment& innerSegment = *innerSegmentPtr;

        // Get the outer mini-doublet module detId
        const SDL::CPU::Module& innerSegmentOuterModule = innerSegment.outerMiniDoubletPtr()->lowerHitPtr()->getModule();

        unsigned int innerSegmentOuterModuleDetId = innerSegmentOuterModule.detId();

        // Get connected middle module detids
        const std::vector<unsigned int>& connectedMiddleModuleDetIds = moduleConnectionMap.getConnectedModuleDetIds(innerSegmentOuterModuleDetId);

        for (auto& middleLowerModuleDetId : connectedMiddleModuleDetIds)
        {

            // Get connected outer lower module detids
            const std::vector<unsigned int>& connectedModuleDetIds = moduleConnectionMap.getConnectedModuleDetIds(middleLowerModuleDetId);

            // Loop over connected outer lower modules
            for (auto& outerLowerModuleDetId : connectedModuleDetIds)
            {

                if (not hasModule(outerLowerModuleDetId))
                    continue;

                // Get reference to the outer lower module
                Module& outerLowerModule = getModule(outerLowerModuleDetId);

                // Loop over outer lower module mini-doublets
                for (auto& outerSegmentPtr : outerLowerModule.getSegmentPtrs())
                {

                    // Count the # of tlCands considered by layer
                    incrementNumberOfTrackletCandidates(innerLowerModule);

                    // Get reference to mini-doublet in outer lower module
                    SDL::CPU::Segment& outerSegment = *outerSegmentPtr;

                    // Create a tracklet candidate
                    SDL::CPU::Tracklet tlCand(innerSegmentPtr, outerSegmentPtr);

                    // Run segment algorithm on tlCand (tracklet candidate)
                    tlCand.runTrackletAlgo(algo, logLevel_);

                    if (tlCand.passesTrackletAlgo(algo))
                    {

                        // Count the # of sg formed by layer
                        incrementNumberOfTracklets(innerLowerModule);

                        if (innerLowerModule.subdet() == SDL::CPU::Module::Barrel)
                            addTrackletToEvent(tlCand, innerLowerModule.detId(), innerLowerModule.layer(), SDL::CPU::Layer::Barrel);
                        else
                            addTrackletToEvent(tlCand, innerLowerModule.detId(), innerLowerModule.layer(), SDL::CPU::Layer::Endcap);
                    }

                }

            }

        }

    }

}

// Create tracklets
void SDL::CPU::Event::createTrackletsWithTwoGapsWithModuleMap(TLAlgo algo)
{
    if (logLevel_ == SDL::CPU::Log_Debug)
        SDL::CPU::cout << "SDL::CPU::Event::createTrackletsWithTwoGapsWithModuleMap()" << std::endl;

    // Loop over lower modules
    int nModuleProcessed = 0;
    int nTotalLowerModule = getLowerModulePtrs().size();

    if (logLevel_ == SDL::CPU::Log_Debug)
        SDL::CPU::cout <<  " nTotalLowerModule: " << nTotalLowerModule <<  std::endl;

    // Loop over lower modules
    for (auto& lowerModulePtr : getLowerModulePtrs())
    {

        if (logLevel_ == SDL::CPU::Log_Debug)
            if (nModuleProcessed % 1000 == 0)
                SDL::CPU::cout <<  "    nModuleProcessed: " << nModuleProcessed <<  std::endl;

        if (logLevel_ == SDL::CPU::Log_Debug)
        {
            std::cout <<  " lowerModulePtr->subdet(): " << lowerModulePtr->subdet() <<  std::endl;
            std::cout <<  " lowerModulePtr->layer(): " << lowerModulePtr->layer() <<  std::endl;
            std::cout <<  " lowerModulePtr->getSegmentPtrs().size(): " << lowerModulePtr->getSegmentPtrs().size() <<  std::endl;
        }

        // if (lowerModulePtr->layer() != 1)
        //     continue;

        // Create mini doublets
        createTrackletsWithTwoGapsFromInnerLowerModule(lowerModulePtr->detId(), algo);

        nModuleProcessed++;

    }
}

// Create tracklets from inner modules
void SDL::CPU::Event::createTrackletsWithTwoGapsFromInnerLowerModule(unsigned int detId, SDL::CPU::TLAlgo algo)
{

    // Get reference to the inner lower Module
    Module& innerLowerModule = getModule(detId);

    // Triple nested loops
    // Loop over inner lower module for segments
    for (auto& innerSegmentPtr : innerLowerModule.getSegmentPtrs())
    {

        // Get reference to segment in inner lower module
        SDL::CPU::Segment& innerSegment = *innerSegmentPtr;

        // Get the outer mini-doublet module detId
        const SDL::CPU::Module& innerSegmentOuterModule = innerSegment.outerMiniDoubletPtr()->lowerHitPtr()->getModule();

        unsigned int innerSegmentOuterModuleDetId = innerSegmentOuterModule.detId();

        // Get connected middle module detids
        const std::vector<unsigned int>& connectedMiddleModuleDetIds = moduleConnectionMap.getConnectedModuleDetIds(innerSegmentOuterModuleDetId);

        for (auto& middleLowerModuleDetId : connectedMiddleModuleDetIds)
        {

            // Get connected second middle module detids
            const std::vector<unsigned int>& connectedMiddle2ModuleDetIds = moduleConnectionMap.getConnectedModuleDetIds(middleLowerModuleDetId);

            for (auto& middle2LowerModuleDetId : connectedMiddle2ModuleDetIds)
            {

                // Get connected outer lower module detids
                const std::vector<unsigned int>& connectedModuleDetIds = moduleConnectionMap.getConnectedModuleDetIds(middle2LowerModuleDetId);

                // Loop over connected outer lower modules
                for (auto& outerLowerModuleDetId : connectedModuleDetIds)
                {

                    if (not hasModule(outerLowerModuleDetId))
                        continue;

                    // Get reference to the outer lower module
                    Module& outerLowerModule = getModule(outerLowerModuleDetId);

                    // Loop over outer lower module mini-doublets
                    for (auto& outerSegmentPtr : outerLowerModule.getSegmentPtrs())
                    {

                        // Count the # of tlCands considered by layer
                        incrementNumberOfTrackletCandidates(innerLowerModule);

                        // Get reference to mini-doublet in outer lower module
                        SDL::CPU::Segment& outerSegment = *outerSegmentPtr;

                        // Create a tracklet candidate
                        SDL::CPU::Tracklet tlCand(innerSegmentPtr, outerSegmentPtr);

                        // Run segment algorithm on tlCand (tracklet candidate)
                        tlCand.runTrackletAlgo(algo, logLevel_);

                        if (tlCand.passesTrackletAlgo(algo))
                        {

                            // Count the # of sg formed by layer
                            incrementNumberOfTracklets(innerLowerModule);

                            if (innerLowerModule.subdet() == SDL::CPU::Module::Barrel)
                                addTrackletToEvent(tlCand, innerLowerModule.detId(), innerLowerModule.layer(), SDL::CPU::Layer::Barrel);
                            else
                                addTrackletToEvent(tlCand, innerLowerModule.detId(), innerLowerModule.layer(), SDL::CPU::Layer::Endcap);
                        }

                    }

                }

            }

        }

    }

}

// Create tracklets via navigation
void SDL::CPU::Event::createTrackletsViaNavigation(SDL::CPU::TLAlgo algo)
{
    // Loop over lower modules
    for (auto& lowerModulePtr : getLowerModulePtrs())
    {

        // Get reference to the inner lower Module
        Module& innerLowerModule = getModule(lowerModulePtr->detId());

        // Triple nested loops
        // Loop over inner lower module for segments
        for (auto& innerSegmentPtr : innerLowerModule.getSegmentPtrs())
        {

            // Get reference to segment in inner lower module
            SDL::CPU::Segment& innerSegment = *innerSegmentPtr;

            // Get the connecting segment ptrs
            for (auto& connectingSegmentPtr : innerSegmentPtr->outerMiniDoubletPtr()->getListOfOutwardSegmentPtrs())
            {

                for (auto& outerSegmentPtr : connectingSegmentPtr->outerMiniDoubletPtr()->getListOfOutwardSegmentPtrs())
                {

                    // Count the # of tlCands considered by layer
                    incrementNumberOfTrackletCandidates(innerLowerModule);

                    // Get reference to mini-doublet in outer lower module
                    SDL::CPU::Segment& outerSegment = *outerSegmentPtr;

                    // Create a tracklet candidate
                    SDL::CPU::Tracklet tlCand(innerSegmentPtr, outerSegmentPtr);

                    // Run segment algorithm on tlCand (tracklet candidate)
                    tlCand.runTrackletAlgo(algo, logLevel_);

                    if (tlCand.passesTrackletAlgo(algo))
                    {

                        // Count the # of sg formed by layer
                        incrementNumberOfTracklets(innerLowerModule);

                        if (innerLowerModule.subdet() == SDL::CPU::Module::Barrel)
                            addTrackletToEvent(tlCand, innerLowerModule.detId(), innerLowerModule.layer(), SDL::CPU::Layer::Barrel);
                        else
                            addTrackletToEvent(tlCand, innerLowerModule.detId(), innerLowerModule.layer(), SDL::CPU::Layer::Endcap);
                    }

                }

            }

        }
    }

}


// Create tracklets with pixel line segments
void SDL::CPU::Event::createTrackletsWithPixelLineSegments_v2(TLAlgo algo)
{
    if (logLevel_ == SDL::CPU::Log_Debug)
        SDL::CPU::cout << "SDL::CPU::Event::createTrackletsWithPixelLineSegments_v2()" << std::endl;

    // Loop over lower modules
    int nModuleProcessed = 0;
    int nTotalLowerModule = getLowerModulePtrs().size();

    if (logLevel_ == SDL::CPU::Log_Debug)
        SDL::CPU::cout <<  " nTotalLowerModule: " << nTotalLowerModule <<  std::endl;

    int nCombinations = 0;


    // Loop over inner lower module for segments
    for (auto& innerSegmentPtr : getPixelLayer().getSegmentPtrs())
    {

        // Get reference to segment in inner lower module
        SDL::CPU::Segment& innerSegment = *innerSegmentPtr;

        // Get reference to the inner lower Module
        Module& pixelModule = getModule(1);
        Module& innerLowerModule = pixelModule;

        // obtain pt eta phi
        float pt = innerSegment.innerMiniDoubletPtr()->lowerHitPtr()->x(); // The x coordinate is set to Pt at the pLS loading stage
        float eta = innerSegment.innerMiniDoubletPtr()->lowerHitPtr()->y(); // The y coordinate is set to Eta at the pLS loading stage
        float phi = innerSegment.innerMiniDoubletPtr()->lowerHitPtr()->z(); // The z coordinate is set to Phi at the pLS loading stage
        // float dxy = innerSegment.innerMiniDoubletPtr()->upperHitPtr()->y(); // The y coordinate is set to dxy of PCA at the pLS loading stage
        float dz = innerSegment.outerMiniDoubletPtr()->lowerHitPtr()->z(); // The x coordinate is set to dz of PCA at the pLS loading stage

        // Determine superbin index

        // Split by two pt bins below and above 2 GeV
        int ptbin = -1;
        if (pt >= 0.9 and pt < 2.0)
            ptbin = 0;
        if (pt >= 2.0)
            ptbin = 1;

        if (ptbin < 0)
            continue;

        // --------------------------------------------------------------------------------------
        // Comments on the pLS:
        // We slice pixel line segments (pLS) into different phase-space "superbins".
        // The slicing is done in pt, eta, phi, and dz.
        //
        // The maps like "SDL::moduleConnectionMap_pLStoLayer1Subdet5" contains a map of following.
        // "superbin_index" -> {detId1, detId2, detId3, ... }
        //
        // So for a given pLS, we need to compute which "superbin_index" it belongs to and then aggregate the list of "detId"'s in given layer in a give subdet
        // Then, we would aggregate the list of modules to loop over in the outer tracker to link pLS's to.
        //
        // The super bin indices are assigned at the time the map was created.
        // Therefore, the convention of the definition index is very important and must be carefully communicated and defined.
        //
        // For example, if a pLS falls into (ipt, ieta, iphi, idz) then the superbin index is
        // isuperbin = (nz*nphi*neta) * ipt + (nz*nphi) * ieta + (nz) * iphi + idz
        //
        // In this case, we bin:
        // pt : (0.9 - 2.0), (2.0 and above)
        // eta : -2.6 to 2.6 with 20 bins
        // phi : -pi to pi with 72 bins
        // dz : -30 to 30 cm with 24 bins
        //
        // --------------------------------------------------------------------------------------
        float neta = 25.; // # of eta bins
        float nphi = 72.; // # of phi bins
        float nz = 25.; // # of z bins

        int etabin = (eta + 2.6) / ((2*2.6) / neta);
        int phibin = (phi + 3.14159265358979323846) / ((2.*3.14159265358979323846) / nphi);
        int dzbin = (dz + 30) / (2*30 / nz);
        int isuperbin = (nz * nphi * neta) * ptbin + (nz * nphi) * etabin + (nz) * phibin + dzbin;
        int charge = (innerSegment.getDeltaPhiChange() > 0) - (innerSegment.getDeltaPhiChange() < 0);

        // std::cout <<  " pt: " << pt <<  " eta: " << eta <<  " phi: " << phi <<  " dz: " << dz <<  std::endl;
        // std::cout <<  " ptbin: " << ptbin <<  " etabin: " << etabin <<  " phibin: " << phibin <<  " dzbin: " << dzbin <<  std::endl;

        std::vector<unsigned int> connectedModuleDetIds_pLStoLayer1Subdet5 = SDL::moduleConnectionMap_pLStoLayer1Subdet5.getConnectedModuleDetIds(isuperbin);
        std::vector<unsigned int> connectedModuleDetIds_pLStoLayer2Subdet5 = SDL::moduleConnectionMap_pLStoLayer2Subdet5.getConnectedModuleDetIds(isuperbin);
        std::vector<unsigned int> connectedModuleDetIds_pLStoLayer3Subdet5 = SDL::moduleConnectionMap_pLStoLayer3Subdet5.getConnectedModuleDetIds(isuperbin);
        std::vector<unsigned int> connectedModuleDetIds_pLStoLayer1Subdet4 = SDL::moduleConnectionMap_pLStoLayer1Subdet4.getConnectedModuleDetIds(isuperbin);
        std::vector<unsigned int> connectedModuleDetIds_pLStoLayer2Subdet4 = SDL::moduleConnectionMap_pLStoLayer2Subdet4.getConnectedModuleDetIds(isuperbin);
        std::vector<unsigned int> connectedModuleDetIds_pLStoLayer3Subdet4 = SDL::moduleConnectionMap_pLStoLayer3Subdet4.getConnectedModuleDetIds(isuperbin);
        std::vector<unsigned int> connectedModuleDetIds_pLStoLayer4Subdet4 = SDL::moduleConnectionMap_pLStoLayer4Subdet4.getConnectedModuleDetIds(isuperbin);
        std::vector<unsigned int> connectedModuleDetIds_pos_pLStoLayer1Subdet5 = SDL::moduleConnectionMap_pLStoLayer1Subdet5_pos.getConnectedModuleDetIds(isuperbin);
        std::vector<unsigned int> connectedModuleDetIds_pos_pLStoLayer2Subdet5 = SDL::moduleConnectionMap_pLStoLayer2Subdet5_pos.getConnectedModuleDetIds(isuperbin);
        std::vector<unsigned int> connectedModuleDetIds_pos_pLStoLayer3Subdet5 = SDL::moduleConnectionMap_pLStoLayer3Subdet5_pos.getConnectedModuleDetIds(isuperbin);
        std::vector<unsigned int> connectedModuleDetIds_pos_pLStoLayer1Subdet4 = SDL::moduleConnectionMap_pLStoLayer1Subdet4_pos.getConnectedModuleDetIds(isuperbin);
        std::vector<unsigned int> connectedModuleDetIds_pos_pLStoLayer2Subdet4 = SDL::moduleConnectionMap_pLStoLayer2Subdet4_pos.getConnectedModuleDetIds(isuperbin);
        std::vector<unsigned int> connectedModuleDetIds_pos_pLStoLayer3Subdet4 = SDL::moduleConnectionMap_pLStoLayer3Subdet4_pos.getConnectedModuleDetIds(isuperbin);
        std::vector<unsigned int> connectedModuleDetIds_pos_pLStoLayer4Subdet4 = SDL::moduleConnectionMap_pLStoLayer4Subdet4_pos.getConnectedModuleDetIds(isuperbin);
        std::vector<unsigned int> connectedModuleDetIds_neg_pLStoLayer1Subdet5 = SDL::moduleConnectionMap_pLStoLayer1Subdet5_neg.getConnectedModuleDetIds(isuperbin);
        std::vector<unsigned int> connectedModuleDetIds_neg_pLStoLayer2Subdet5 = SDL::moduleConnectionMap_pLStoLayer2Subdet5_neg.getConnectedModuleDetIds(isuperbin);
        std::vector<unsigned int> connectedModuleDetIds_neg_pLStoLayer3Subdet5 = SDL::moduleConnectionMap_pLStoLayer3Subdet5_neg.getConnectedModuleDetIds(isuperbin);
        std::vector<unsigned int> connectedModuleDetIds_neg_pLStoLayer1Subdet4 = SDL::moduleConnectionMap_pLStoLayer1Subdet4_neg.getConnectedModuleDetIds(isuperbin);
        std::vector<unsigned int> connectedModuleDetIds_neg_pLStoLayer2Subdet4 = SDL::moduleConnectionMap_pLStoLayer2Subdet4_neg.getConnectedModuleDetIds(isuperbin);
        std::vector<unsigned int> connectedModuleDetIds_neg_pLStoLayer3Subdet4 = SDL::moduleConnectionMap_pLStoLayer3Subdet4_neg.getConnectedModuleDetIds(isuperbin);
        std::vector<unsigned int> connectedModuleDetIds_neg_pLStoLayer4Subdet4 = SDL::moduleConnectionMap_pLStoLayer4Subdet4_neg.getConnectedModuleDetIds(isuperbin);

        std::vector<unsigned int> connectedModuleDetIds;
        if (ptbin == 1)
        {
            connectedModuleDetIds.insert(connectedModuleDetIds.end(), connectedModuleDetIds_pLStoLayer1Subdet5.begin(), connectedModuleDetIds_pLStoLayer1Subdet5.end());
            connectedModuleDetIds.insert(connectedModuleDetIds.end(), connectedModuleDetIds_pLStoLayer2Subdet5.begin(), connectedModuleDetIds_pLStoLayer2Subdet5.end());
            connectedModuleDetIds.insert(connectedModuleDetIds.end(), connectedModuleDetIds_pLStoLayer3Subdet5.begin(), connectedModuleDetIds_pLStoLayer3Subdet5.end());
            connectedModuleDetIds.insert(connectedModuleDetIds.end(), connectedModuleDetIds_pLStoLayer1Subdet4.begin(), connectedModuleDetIds_pLStoLayer1Subdet4.end());
            connectedModuleDetIds.insert(connectedModuleDetIds.end(), connectedModuleDetIds_pLStoLayer2Subdet4.begin(), connectedModuleDetIds_pLStoLayer2Subdet4.end());
            connectedModuleDetIds.insert(connectedModuleDetIds.end(), connectedModuleDetIds_pLStoLayer3Subdet4.begin(), connectedModuleDetIds_pLStoLayer3Subdet4.end());
            connectedModuleDetIds.insert(connectedModuleDetIds.end(), connectedModuleDetIds_pLStoLayer4Subdet4.begin(), connectedModuleDetIds_pLStoLayer4Subdet4.end());
        }
        else
        {
            if (charge > 0)
            {
                connectedModuleDetIds.insert(connectedModuleDetIds.end(), connectedModuleDetIds_pos_pLStoLayer1Subdet5.begin(), connectedModuleDetIds_pos_pLStoLayer1Subdet5.end());
                connectedModuleDetIds.insert(connectedModuleDetIds.end(), connectedModuleDetIds_pos_pLStoLayer2Subdet5.begin(), connectedModuleDetIds_pos_pLStoLayer2Subdet5.end());
                connectedModuleDetIds.insert(connectedModuleDetIds.end(), connectedModuleDetIds_pos_pLStoLayer3Subdet5.begin(), connectedModuleDetIds_pos_pLStoLayer3Subdet5.end());
                connectedModuleDetIds.insert(connectedModuleDetIds.end(), connectedModuleDetIds_pos_pLStoLayer1Subdet4.begin(), connectedModuleDetIds_pos_pLStoLayer1Subdet4.end());
                connectedModuleDetIds.insert(connectedModuleDetIds.end(), connectedModuleDetIds_pos_pLStoLayer2Subdet4.begin(), connectedModuleDetIds_pos_pLStoLayer2Subdet4.end());
                connectedModuleDetIds.insert(connectedModuleDetIds.end(), connectedModuleDetIds_pos_pLStoLayer3Subdet4.begin(), connectedModuleDetIds_pos_pLStoLayer3Subdet4.end());
                connectedModuleDetIds.insert(connectedModuleDetIds.end(), connectedModuleDetIds_pos_pLStoLayer4Subdet4.begin(), connectedModuleDetIds_pos_pLStoLayer4Subdet4.end());
            }
            else
            {
                connectedModuleDetIds.insert(connectedModuleDetIds.end(), connectedModuleDetIds_neg_pLStoLayer1Subdet5.begin(), connectedModuleDetIds_neg_pLStoLayer1Subdet5.end());
                connectedModuleDetIds.insert(connectedModuleDetIds.end(), connectedModuleDetIds_neg_pLStoLayer2Subdet5.begin(), connectedModuleDetIds_neg_pLStoLayer2Subdet5.end());
                connectedModuleDetIds.insert(connectedModuleDetIds.end(), connectedModuleDetIds_neg_pLStoLayer3Subdet5.begin(), connectedModuleDetIds_neg_pLStoLayer3Subdet5.end());
                connectedModuleDetIds.insert(connectedModuleDetIds.end(), connectedModuleDetIds_neg_pLStoLayer1Subdet4.begin(), connectedModuleDetIds_neg_pLStoLayer1Subdet4.end());
                connectedModuleDetIds.insert(connectedModuleDetIds.end(), connectedModuleDetIds_neg_pLStoLayer2Subdet4.begin(), connectedModuleDetIds_neg_pLStoLayer2Subdet4.end());
                connectedModuleDetIds.insert(connectedModuleDetIds.end(), connectedModuleDetIds_neg_pLStoLayer3Subdet4.begin(), connectedModuleDetIds_neg_pLStoLayer3Subdet4.end());
                connectedModuleDetIds.insert(connectedModuleDetIds.end(), connectedModuleDetIds_neg_pLStoLayer4Subdet4.begin(), connectedModuleDetIds_neg_pLStoLayer4Subdet4.end());
            }
        }

        // std::cout <<  " isuperbin: " << isuperbin <<  std::endl;
        // for (auto& detid : connectedModuleDetIds)
        // {
        //     std::cout <<  " detid: " << detid <<  std::endl;
        // }

        // Loop over connected outer lower modules
        for (auto& outerLowerModuleDetId : connectedModuleDetIds)
        {

            if (not hasModule(outerLowerModuleDetId))
                continue;

            // Get reference to the outer lower module
            Module& outerLowerModule = getModule(outerLowerModuleDetId);

            // Loop over outer lower module mini-doublets
            for (auto& outerSegmentPtr : outerLowerModule.getSegmentPtrs())
            {

                if (outerSegmentPtr->outerMiniDoubletPtr()->anchorHitPtr()->getModule().moduleType() != SDL::CPU::Module::PS)
                    continue;

                // Get reference to mini-doublet in outer lower module
                SDL::CPU::Segment& outerSegment = *outerSegmentPtr;

                // Create a tracklet candidate
                SDL::CPU::Tracklet tlCand(innerSegmentPtr, outerSegmentPtr);

                // Run segment algorithm on tlCand (tracklet candidate)
                tlCand.runTrackletAlgo(algo, logLevel_);

                nCombinations++;

                if (tlCand.passesTrackletAlgo(algo))
                {

                    // Count the # of sg formed by layer
                    incrementNumberOfTracklets(innerLowerModule);

                    addTrackletToEvent(tlCand, 1/*pixel module=1*/, 0/*pixel is layer=0*/, SDL::CPU::Layer::Barrel);
                }

            }

            nModuleProcessed++;

        }


    }

    if (logLevel_ == SDL::CPU::Log_Debug)
        std::cout << "SDL::CPU::Event::createTrackletsWithPixelLineSegments_v2(): nCombinations = " << nCombinations << std::endl;

}

// Create tracklets with pixel line segments
void SDL::CPU::Event::createTrackletsWithPixelLineSegments(TLAlgo algo)
{
    if (logLevel_ == SDL::CPU::Log_Debug)
        SDL::CPU::cout << "SDL::CPU::Event::createTrackletsWithPixelLineSegments()" << std::endl;

    // Loop over lower modules
    int nModuleProcessed = 0;
    int nTotalLowerModule = getLowerModulePtrs().size();

    if (logLevel_ == SDL::CPU::Log_Debug)
        SDL::CPU::cout <<  " nTotalLowerModule: " << nTotalLowerModule <<  std::endl;

    int nCombinations = 0;


    // Loop over inner lower module for segments
    for (auto& innerSegmentPtr : getPixelLayer().getSegmentPtrs())
    {

        // Get reference to segment in inner lower module
        SDL::CPU::Segment& innerSegment = *innerSegmentPtr;

        // Get reference to the inner lower Module
        Module& pixelModule = getModule(1);
        Module& innerLowerModule = pixelModule;

        for (auto& lowerModulePtr : getLowerModulePtrs())
        {

            if (lowerModulePtr->getSegmentPtrs().size() == 0)
                continue;

            if (lowerModulePtr->moduleType() != SDL::CPU::Module::PS)
                continue;

            // if (logLevel_ == SDL::CPU::Log_Debug)
            //     if (nModuleProcessed % 1000 == 0)
            //         SDL::CPU::cout <<  "    nModuleProcessed: " << nModuleProcessed <<  std::endl;

            // if (logLevel_ == SDL::CPU::Log_Debug)
            // {
            //     std::cout <<  " lowerModulePtr->subdet(): " << lowerModulePtr->subdet() <<  std::endl;
            //     std::cout <<  " lowerModulePtr->layer(): " << lowerModulePtr->layer() <<  std::endl;
            //     std::cout <<  " lowerModulePtr->getSegmentPtrs().size(): " << lowerModulePtr->getSegmentPtrs().size() <<  std::endl;
            // }

            // Get reference to the outer lower module
            Module& outerLowerModule = *lowerModulePtr;

            // Loop over outer lower module mini-doublets
            for (auto& outerSegmentPtr : outerLowerModule.getSegmentPtrs())
            {

                if (outerSegmentPtr->outerMiniDoubletPtr()->anchorHitPtr()->getModule().moduleType() != SDL::CPU::Module::PS)
                    continue;

                // // Count the # of tlCands considered by layer
                // incrementNumberOfTrackletCandidates(innerLowerModule);

                // Get reference to mini-doublet in outer lower module
                SDL::CPU::Segment& outerSegment = *outerSegmentPtr;

                // Create a tracklet candidate
                SDL::CPU::Tracklet tlCand(innerSegmentPtr, outerSegmentPtr);

                // Run segment algorithm on tlCand (tracklet candidate)
                tlCand.runTrackletAlgo(algo, logLevel_);

                if (logLevel_ == SDL::CPU::Log_Debug)
                {
                    // int passbit = tlCand.getPassBitsDefaultAlgo();
                    // std::bitset<8> x(passbit);
                    // std::cout <<  " passbit: " << x <<  std::endl;
                }

                nCombinations++;

                if (tlCand.passesTrackletAlgo(algo))
                {

                    // Count the # of sg formed by layer
                    incrementNumberOfTracklets(innerLowerModule);

                    addTrackletToEvent(tlCand, 1/*pixel module=1*/, 0/*pixel is layer=0*/, SDL::CPU::Layer::Barrel);
                }

            }

            nModuleProcessed++;

        }

    }

    if (logLevel_ == SDL::CPU::Log_Debug)
        std::cout << "SDL::CPU::Event::createTrackletsWithPixelAndBarrel(): nCombinations = " << nCombinations << std::endl;


    // for (auto& lowerModulePtr : getLowerModulePtrs())
    // {

    //     if (lowerModulePtr->getSegmentPtrs().size() == 0)
    //         continue;

    //     if (logLevel_ == SDL::CPU::Log_Debug)
    //         if (nModuleProcessed % 1000 == 0)
    //             SDL::CPU::cout <<  "    nModuleProcessed: " << nModuleProcessed <<  std::endl;

    //     if (logLevel_ == SDL::CPU::Log_Debug)
    //     {
    //         std::cout <<  " lowerModulePtr->subdet(): " << lowerModulePtr->subdet() <<  std::endl;
    //         std::cout <<  " lowerModulePtr->layer(): " << lowerModulePtr->layer() <<  std::endl;
    //         std::cout <<  " lowerModulePtr->getSegmentPtrs().size(): " << lowerModulePtr->getSegmentPtrs().size() <<  std::endl;
    //     }

    //     // Get reference to the inner lower Module
    //     Module& pixelModule = getModule(1);
    //     Module& innerLowerModule = pixelModule;

    //     // Triple nested loops
    //     // Loop over inner lower module for segments
    //     for (auto& innerSegmentPtr : getPixelLayer().getSegmentPtrs())
    //     {

    //         // Get reference to segment in inner lower module
    //         SDL::CPU::Segment& innerSegment = *innerSegmentPtr;

    //         // Get reference to the outer lower module
    //         Module& outerLowerModule = *lowerModulePtr;

    //         // Loop over outer lower module mini-doublets
    //         for (auto& outerSegmentPtr : outerLowerModule.getSegmentPtrs())
    //         {

    //             // // Count the # of tlCands considered by layer
    //             // incrementNumberOfTrackletCandidates(innerLowerModule);

    //             // Get reference to mini-doublet in outer lower module
    //             SDL::CPU::Segment& outerSegment = *outerSegmentPtr;

    //             // Create a tracklet candidate
    //             SDL::CPU::Tracklet tlCand(innerSegmentPtr, outerSegmentPtr);

    //             // Run segment algorithm on tlCand (tracklet candidate)
    //             tlCand.runTrackletAlgo(algo, logLevel_);

    //             nCombinations++;

    //             if (tlCand.passesTrackletAlgo(algo))
    //             {

    //                 // Count the # of sg formed by layer
    //                 incrementNumberOfTracklets(innerLowerModule);

    //                 addTrackletToEvent(tlCand, 1/*pixel module=1*/, 0/*pixel is layer=0*/, SDL::CPU::Layer::Barrel);
    //             }

    //         }

    //     }

    //     nModuleProcessed++;

    // }

    // if (logLevel_ == SDL::CPU::Log_Debug)
    //     std::cout << "SDL::CPU::Event::createTrackletsWithPixelAndBarrel(): nCombinations = " << nCombinations << std::endl;
}


// Create tracklets from two layers (inefficient way)
void SDL::CPU::Event::createTrackletsFromTwoLayers(int innerLayerIdx, SDL::CPU::Layer::SubDet innerLayerSubDet, int outerLayerIdx, SDL::CPU::Layer::SubDet outerLayerSubDet, TLAlgo algo)
{
    Layer& innerLayer = getLayer(innerLayerIdx, innerLayerSubDet);
    Layer& outerLayer = getLayer(outerLayerIdx, outerLayerSubDet);

    for (auto& innerSegmentPtr : innerLayer.getSegmentPtrs())
    {
        SDL::CPU::Segment& innerSegment = *innerSegmentPtr;
        for (auto& outerSegmentPtr : outerLayer.getSegmentPtrs())
        {
            // SDL::CPU::Segment& outerSegment = *outerSegmentPtr;

            // if (SDL::CPU::Tracklet::isSegmentPairATracklet(innerSegment, outerSegment, algo, logLevel_))
            //     addTrackletToLowerLayer(SDL::CPU::Tracklet(innerSegmentPtr, outerSegmentPtr), innerLayerIdx, innerLayerSubDet);

            SDL::CPU::Segment& outerSegment = *outerSegmentPtr;

            SDL::CPU::Tracklet tlCand(innerSegmentPtr, outerSegmentPtr);

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
void SDL::CPU::Event::createSegmentsFromTwoLayers(int innerLayerIdx, SDL::CPU::Layer::SubDet innerLayerSubDet, int outerLayerIdx, SDL::CPU::Layer::SubDet outerLayerSubDet, SGAlgo algo)
{
    Layer& innerLayer = getLayer(innerLayerIdx, innerLayerSubDet);
    Layer& outerLayer = getLayer(outerLayerIdx, outerLayerSubDet);

    for (auto& innerMiniDoubletPtr : innerLayer.getMiniDoubletPtrs())
    {
        SDL::CPU::MiniDoublet& innerMiniDoublet = *innerMiniDoubletPtr;

        for (auto& outerMiniDoubletPtr : outerLayer.getMiniDoubletPtrs())
        {
            SDL::CPU::MiniDoublet& outerMiniDoublet = *outerMiniDoubletPtr;

            SDL::CPU::Segment sgCand(innerMiniDoubletPtr, outerMiniDoubletPtr);

            sgCand.runSegmentAlgo(algo, logLevel_);

            if (sgCand.passesSegmentAlgo(algo))
            {
                const SDL::CPU::Module& innerLowerModule = innerMiniDoubletPtr->lowerHitPtr()->getModule();
                if (innerLowerModule.subdet() == SDL::CPU::Module::Barrel)
                    addSegmentToEvent(sgCand, innerLowerModule.detId(), innerLowerModule.layer(), SDL::CPU::Layer::Barrel);
                else
                    addSegmentToEvent(sgCand, innerLowerModule.detId(), innerLowerModule.layer(), SDL::CPU::Layer::Endcap);
            }

        }
    }
}

// Create trackCandidates from two layers (inefficient way)
void SDL::CPU::Event::createTrackCandidatesFromTwoLayers(int innerLayerIdx, SDL::CPU::Layer::SubDet innerLayerSubDet, int outerLayerIdx, SDL::CPU::Layer::SubDet outerLayerSubDet, TCAlgo algo)
{
    Layer& innerLayer = getLayer(innerLayerIdx, innerLayerSubDet);
    Layer& outerLayer = getLayer(outerLayerIdx, outerLayerSubDet);

    for (auto& innerTrackletPtr : innerLayer.getTrackletPtrs())
    {
        SDL::CPU::Tracklet& innerTracklet = *innerTrackletPtr;

        for (auto& outerTrackletPtr : outerLayer.getTrackletPtrs())
        {

            SDL::CPU::Tracklet& outerTracklet = *outerTrackletPtr;

            SDL::CPU::TrackCandidate tcCand(innerTrackletPtr, outerTrackletPtr);

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

void SDL::CPU::Event::createTrackCandidatesFromTriplets(TCAlgo algo)
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

void SDL::CPU::Event::createTrackCandidatesFromInnerModulesFromTriplets(unsigned int detId, SDL::CPU::TCAlgo algo)
{

    // Get reference to the inner lower Module
    Module& innerLowerModule = getModule(detId);

    // Triple nested loops
    // Loop over inner lower module for segments
    for (auto& innerTripletPtr : innerLowerModule.getTripletPtrs())
    {

        // Get reference to segment in inner lower module
        SDL::CPU::Triplet& innerTriplet = *innerTripletPtr;

        // Get the outer mini-doublet module detId
        const SDL::CPU::Module& innerTripletOutermostModule = innerTriplet.outerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->getModule();

        unsigned int innerTripletOutermostModuleDetId = innerTripletOutermostModule.detId();

        // Get connected outer lower module detids
        const std::vector<unsigned int>& connectedModuleDetIds = moduleConnectionMap.getConnectedModuleDetIds(innerTripletOutermostModuleDetId);

        // Loop over connected outer lower modules
        for (auto& outerLowerModuleDetId : connectedModuleDetIds)
        {

            if (not hasModule(outerLowerModuleDetId))
                continue;

            // Get reference to the outer lower module
            Module& outerLowerModule = getModule(outerLowerModuleDetId);

            // Loop over outer lower module mini-doublets
            for (auto& outerTripletPtr : outerLowerModule.getTripletPtrs())
            {

                // Count the # of tlCands considered by layer
                incrementNumberOfTrackCandidateCandidates(innerLowerModule);

                // Segment between innerSgOuterMD - outerSgInnerMD
                SDL::CPU::Segment sgCand(innerTripletPtr->outerSegmentPtr()->outerMiniDoubletPtr(),outerTripletPtr->innerSegmentPtr()->innerMiniDoubletPtr());

                // Run the segment algo (supposedly is fast)
                sgCand.runSegmentAlgo(SDL::CPU::Default_SGAlgo, logLevel_);

                if (not (sgCand.passesSegmentAlgo(SDL::CPU::Default_SGAlgo)))
                {
                    continue;
                }

                // SDL::CPU::Tracklet tlCand(innerTripletPtr->innerSegmentPtr(), &sgCand);

                // // Run the segment algo (supposedly is fast)
                // tlCand.runTrackletAlgo(SDL::CPU::Default_TLAlgo, logLevel_);

                // if (not (tlCand.passesTrackletAlgo(SDL::CPU::Default_TLAlgo)))
                // {
                //     continue;
                // }

                SDL::CPU::Tracklet tlCandOuter(&sgCand, outerTripletPtr->outerSegmentPtr());

                // Run the segment algo (supposedly is fast)
                tlCandOuter.runTrackletAlgo(SDL::CPU::Default_TLAlgo, logLevel_);

                if (not (tlCandOuter.passesTrackletAlgo(SDL::CPU::Default_TLAlgo)))
                {
                    continue;
                }

                SDL::CPU::TrackCandidate tcCand(innerTripletPtr, outerTripletPtr);

                // if (tcCand.passesTrackCandidateAlgo(algo))
                // {

                // Count the # of track candidates considered
                incrementNumberOfTrackCandidates(innerLowerModule);

                addTrackCandidateToLowerLayer(tcCand, 1, SDL::CPU::Layer::Barrel);
                // if (innerLowerModule.subdet() == SDL::CPU::Module::Barrel)
                //     addTrackCandidateToLowerLayer(tcCand, innerLowerModule.layer(), SDL::CPU::Layer::Barrel);
                // else
                //     addTrackCandidateToLowerLayer(tcCand, innerLowerModule.layer(), SDL::CPU::Layer::Endcap);

                // }

            }

        }

    }


}

void SDL::CPU::Event::createTrackCandidatesFromTracklets(TCAlgo algo)
{
    // Loop over lower modules
    for (auto& lowerModulePtr : getLowerModulePtrs())
    {

        // if (not (lowerModulePtr->layer() == 1))
        //     continue;

        // Create mini doublets
        createTrackCandidatesFromInnerModulesFromTracklets(lowerModulePtr->detId(), algo);
        // createTrackCandidatesFromInnerModulesFromTrackletsToTriplets(lowerModulePtr->detId(), algo);

    }
}

void SDL::CPU::Event::createTrackCandidatesFromInnerModulesFromTracklets(unsigned int detId, SDL::CPU::TCAlgo algo)
{

    // Get reference to the inner lower Module
    Module& innerLowerModule = getModule(detId);

    // Triple nested loops
    // Loop over inner lower module for segments
    for (auto& innerTrackletPtr : innerLowerModule.getTrackletPtrs())
    {

        // Get reference to segment in inner lower module
        SDL::CPU::Tracklet& innerTracklet = *innerTrackletPtr;

        // Get the outer mini-doublet module detId
        const SDL::CPU::Module& innerTrackletSecondModule = innerTracklet.innerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->getModule();

        unsigned int innerTrackletSecondModuleDetId = innerTrackletSecondModule.detId();

        // Get connected outer lower module detids
        const std::vector<unsigned int>& connectedModuleDetIds = moduleConnectionMap.getConnectedModuleDetIds(innerTrackletSecondModuleDetId);

        // Loop over connected outer lower modules
        for (auto& outerLowerModuleDetId : connectedModuleDetIds)
        {

            if (not hasModule(outerLowerModuleDetId))
                continue;

            // Get reference to the outer lower module
            Module& outerLowerModule = getModule(outerLowerModuleDetId);

            // Loop over outer lower module mini-doublets
            for (auto& outerTrackletPtr : outerLowerModule.getTrackletPtrs())
            {

                SDL::CPU::Tracklet& outerTracklet = *outerTrackletPtr;

                SDL::CPU::TrackCandidate tcCand(innerTrackletPtr, outerTrackletPtr);

                tcCand.runTrackCandidateAlgo(algo, logLevel_);

                // Count the # of track candidates considered
                incrementNumberOfTrackCandidateCandidates(innerLowerModule);

                if (tcCand.passesTrackCandidateAlgo(algo))
                {

                    // Count the # of track candidates considered
                    incrementNumberOfTrackCandidates(innerLowerModule);

                    if (innerLowerModule.subdet() == SDL::CPU::Module::Barrel)
                        addTrackCandidateToLowerLayer(tcCand, innerLowerModule.layer(), SDL::CPU::Layer::Barrel);
                    else
                        addTrackCandidateToLowerLayer(tcCand, innerLowerModule.layer(), SDL::CPU::Layer::Endcap);
                }

            }

        }

    }

}

void SDL::CPU::Event::createTrackCandidatesTest_v1(SDL::CPU::TCAlgo algo)
{

    // August 31, 2020 Considering only the following kinds of track candidates in barrel
    // - 1 2 3 4 + 3 4 5 6  : 0 missing
    // - 1 2 3 4 + 3 4 5    : 6 missing
    // - 1 2 4 5 + 4 5 6    : 3 missing
    // - 2 3 4 5 + 4 5 6    : 1 missing
    // - 1 2 3 + 2 3 5 6    : 4 missing
    // 2 and 5 missing not done at this point

    const SDL::CPU::Layer& barrelLayer1 = getLayer(1, SDL::CPU::Layer::Barrel);
    const SDL::CPU::Layer& barrelLayer2 = getLayer(2, SDL::CPU::Layer::Barrel);
    const SDL::CPU::Layer& barrelLayer3 = getLayer(3, SDL::CPU::Layer::Barrel);
    const SDL::CPU::Layer& barrelLayer4 = getLayer(4, SDL::CPU::Layer::Barrel);
    const SDL::CPU::Layer& barrelLayer5 = getLayer(5, SDL::CPU::Layer::Barrel);
    const SDL::CPU::Layer& barrelLayer6 = getLayer(6, SDL::CPU::Layer::Barrel);

    for (auto& innerTrackletPtr : barrelLayer1.getTrackletPtrs())
    {

        // Check if it is a barrel only tracklet
        bool isBBBB =
        ((innerTrackletPtr->innerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->getModule()).subdet() == SDL::CPU::Module::Barrel) and
        ((innerTrackletPtr->innerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->getModule()).subdet() == SDL::CPU::Module::Barrel) and
        ((innerTrackletPtr->outerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->getModule()).subdet() == SDL::CPU::Module::Barrel) and
        ((innerTrackletPtr->outerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->getModule()).subdet() == SDL::CPU::Module::Barrel);

        bool is1234 = 
        ((innerTrackletPtr->innerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->getModule()).layer() == 1) and
        ((innerTrackletPtr->innerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->getModule()).layer() == 2) and
        ((innerTrackletPtr->outerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->getModule()).layer() == 3) and
        ((innerTrackletPtr->outerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->getModule()).layer() == 4);

        bool is1245 = 
        ((innerTrackletPtr->innerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->getModule()).layer() == 1) and
        ((innerTrackletPtr->innerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->getModule()).layer() == 2) and
        ((innerTrackletPtr->outerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->getModule()).layer() == 4) and
        ((innerTrackletPtr->outerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->getModule()).layer() == 5);

        if (isBBBB and is1234)
        {

            // Get reference to the inner lower Module
            Module& innerLowerModule = getModule((innerTrackletPtr->innerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->getModule()).detId());

            unsigned int innerTrackletModule2 = (innerTrackletPtr->innerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->getModule()).detId();
            unsigned int innerTrackletModule3 = (innerTrackletPtr->outerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->getModule()).detId();

            // Get reference to the outer lower module
            Module& outerLowerModule = getModule(innerTrackletModule3);

            // Loop over outer lower module Tracklets
            for (auto& outerTrackletPtr : outerLowerModule.getTrackletPtrs())
            {

                // Check if it is a barrel only tracklet
                bool isOuterBBBB =
                    ((outerTrackletPtr->innerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->getModule()).subdet() == SDL::CPU::Module::Barrel) and
                    ((outerTrackletPtr->innerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->getModule()).subdet() == SDL::CPU::Module::Barrel) and
                    ((outerTrackletPtr->outerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->getModule()).subdet() == SDL::CPU::Module::Barrel) and
                    ((outerTrackletPtr->outerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->getModule()).subdet() == SDL::CPU::Module::Barrel);

                // Check if it is a barrel only tracklet
                bool isOuter3456 =
                    ((outerTrackletPtr->innerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->getModule()).layer() == 3) and
                    ((outerTrackletPtr->innerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->getModule()).layer() == 4) and
                    ((outerTrackletPtr->outerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->getModule()).layer() == 5) and
                    ((outerTrackletPtr->outerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->getModule()).layer() == 6);

                if (not (isOuterBBBB and isOuter3456))
                {
                    continue;
                }

                SDL::CPU::Tracklet& outerTracklet = *outerTrackletPtr;

                //============================================================
                //
                //
                // Type 1: 1 2 3 4 + 3 4 5 6 : no layer missing
                //
                //
                //============================================================
                SDL::CPU::TrackCandidate tcCand(innerTrackletPtr, outerTrackletPtr);

                tcCand.runTrackCandidateAlgo(algo, logLevel_);

                // Count the # of track candidates considered
                incrementNumberOfTrackCandidateCandidates(innerLowerModule);

                if (tcCand.passesTrackCandidateAlgo(algo))
                {

                    // Count the # of track candidates considered
                    incrementNumberOfTrackCandidates(innerLowerModule);

                    if (innerLowerModule.subdet() == SDL::CPU::Module::Barrel)
                        addTrackCandidateToLowerLayer(tcCand, innerLowerModule.layer(), SDL::CPU::Layer::Barrel);
                    else
                        addTrackCandidateToLowerLayer(tcCand, innerLowerModule.layer(), SDL::CPU::Layer::Endcap);
                }

            }

            // Loop over outer lower triplets
            for (auto& outerTripletPtr : outerLowerModule.getTripletPtrs())
            {

                // Check if it is a barrel only triplet
                bool isOuterBBB =
                    ((outerTripletPtr->innerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->getModule()).subdet() == SDL::CPU::Module::Barrel) and
                    ((outerTripletPtr->innerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->getModule()).subdet() == SDL::CPU::Module::Barrel) and
                    ((outerTripletPtr->outerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->getModule()).subdet() == SDL::CPU::Module::Barrel);

                // Check if it is a barrel only triplet
                bool isOuter345 =
                    ((outerTripletPtr->innerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->getModule()).layer() == 3) and
                    ((outerTripletPtr->innerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->getModule()).layer() == 4) and
                    ((outerTripletPtr->outerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->getModule()).layer() == 5);

                if (not (isOuterBBB and isOuter345))
                {
                    continue;
                }

                SDL::CPU::Triplet& outerTriplet = *outerTripletPtr;

                //============================================================
                //
                //
                // Type 1: 1 2 3 4 + 3 4 5   : layer 6 is missing
                //
                //
                //============================================================
                SDL::CPU::TrackCandidate tcCand(innerTrackletPtr, outerTripletPtr);

                tcCand.runTrackCandidateInnerTrackletToOuterTriplet(logLevel_);

                // Count the # of track candidates considered
                incrementNumberOfTrackCandidateCandidates(innerLowerModule);

                if (tcCand.passesTrackCandidateAlgo(algo))
                {

                    // Count the # of track candidates considered
                    incrementNumberOfTrackCandidates(innerLowerModule);

                    if (innerLowerModule.subdet() == SDL::CPU::Module::Barrel)
                        addTrackCandidateToLowerLayer(tcCand, innerLowerModule.layer(), SDL::CPU::Layer::Barrel);
                    else
                        addTrackCandidateToLowerLayer(tcCand, innerLowerModule.layer(), SDL::CPU::Layer::Endcap);
                }

            }


        }
        else if (isBBBB and is1245)
        {

            // Get reference to the inner lower Module
            Module& innerLowerModule = getModule((innerTrackletPtr->innerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->getModule()).detId());

            unsigned int innerTrackletModule2 = (innerTrackletPtr->innerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->getModule()).detId();
            unsigned int innerTrackletModule3 = (innerTrackletPtr->outerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->getModule()).detId();

            // Get reference to the outer lower module
            Module& outerLowerModule = getModule(innerTrackletModule3);

            // Loop over outer lower triplets
            for (auto& outerTripletPtr : outerLowerModule.getTripletPtrs())
            {

                // Check if it is a barrel only triplet
                bool isOuterBBB =
                    ((outerTripletPtr->innerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->getModule()).subdet() == SDL::CPU::Module::Barrel) and
                    ((outerTripletPtr->innerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->getModule()).subdet() == SDL::CPU::Module::Barrel) and
                    ((outerTripletPtr->outerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->getModule()).subdet() == SDL::CPU::Module::Barrel);

                // Check if it is a barrel only triplet
                bool isOuter456 =
                    ((outerTripletPtr->innerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->getModule()).layer() == 4) and
                    ((outerTripletPtr->innerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->getModule()).layer() == 5) and
                    ((outerTripletPtr->outerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->getModule()).layer() == 6);

                if (not (isOuterBBB and isOuter456))
                {
                    continue;
                }

                SDL::CPU::Triplet& outerTriplet = *outerTripletPtr;

                //============================================================
                //
                //
                // Type 1: 1 2 4 5 + 4 5 6   : layer 3 is missing
                //
                //
                //============================================================
                SDL::CPU::TrackCandidate tcCand(innerTrackletPtr, outerTripletPtr);

                tcCand.runTrackCandidateInnerTrackletToOuterTriplet(logLevel_);

                // Count the # of track candidates considered
                incrementNumberOfTrackCandidateCandidates(innerLowerModule);

                if (tcCand.passesTrackCandidateAlgo(algo))
                {

                    // Count the # of track candidates considered
                    incrementNumberOfTrackCandidates(innerLowerModule);

                    if (innerLowerModule.subdet() == SDL::CPU::Module::Barrel)
                        addTrackCandidateToLowerLayer(tcCand, innerLowerModule.layer(), SDL::CPU::Layer::Barrel);
                    else
                        addTrackCandidateToLowerLayer(tcCand, innerLowerModule.layer(), SDL::CPU::Layer::Endcap);
                }

            }

        }

    }

    for (auto& innerTrackletPtr : barrelLayer2.getTrackletPtrs())
    {

        // Check if it is a barrel only tracklet
        bool isBBBB =
        ((innerTrackletPtr->innerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->getModule()).subdet() == SDL::CPU::Module::Barrel) and
        ((innerTrackletPtr->innerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->getModule()).subdet() == SDL::CPU::Module::Barrel) and
        ((innerTrackletPtr->outerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->getModule()).subdet() == SDL::CPU::Module::Barrel) and
        ((innerTrackletPtr->outerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->getModule()).subdet() == SDL::CPU::Module::Barrel);

        bool is2345 = 
        ((innerTrackletPtr->innerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->getModule()).layer() == 2) and
        ((innerTrackletPtr->innerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->getModule()).layer() == 3) and
        ((innerTrackletPtr->outerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->getModule()).layer() == 4) and
        ((innerTrackletPtr->outerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->getModule()).layer() == 5);

        if (isBBBB and is2345)
        {

            // Get reference to the inner lower Module
            Module& innerLowerModule = getModule((innerTrackletPtr->innerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->getModule()).detId());

            unsigned int innerTrackletModule2 = (innerTrackletPtr->innerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->getModule()).detId();
            unsigned int innerTrackletModule3 = (innerTrackletPtr->outerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->getModule()).detId();

            // Get reference to the outer lower module
            Module& outerLowerModule = getModule(innerTrackletModule3);

            // Loop over outer lower triplets
            for (auto& outerTripletPtr : outerLowerModule.getTripletPtrs())
            {

                // Check if it is a barrel only triplet
                bool isOuterBBB =
                    ((outerTripletPtr->innerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->getModule()).subdet() == SDL::CPU::Module::Barrel) and
                    ((outerTripletPtr->innerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->getModule()).subdet() == SDL::CPU::Module::Barrel) and
                    ((outerTripletPtr->outerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->getModule()).subdet() == SDL::CPU::Module::Barrel);

                // Check if it is a barrel only triplet
                bool isOuter456 =
                    ((outerTripletPtr->innerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->getModule()).layer() == 4) and
                    ((outerTripletPtr->innerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->getModule()).layer() == 5) and
                    ((outerTripletPtr->outerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->getModule()).layer() == 6);

                if (not (isOuterBBB and isOuter456))
                {
                    continue;
                }

                SDL::CPU::Triplet& outerTriplet = *outerTripletPtr;

                //============================================================
                //
                //
                // Type 1: 2 3 4 5 + 4 5 6   : layer 1 is missing
                //
                //
                //============================================================
                SDL::CPU::TrackCandidate tcCand(innerTrackletPtr, outerTripletPtr);

                tcCand.runTrackCandidateInnerTrackletToOuterTriplet(logLevel_);

                // Count the # of track candidates considered
                incrementNumberOfTrackCandidateCandidates(innerLowerModule);

                if (tcCand.passesTrackCandidateAlgo(algo))
                {

                    // Count the # of track candidates considered
                    incrementNumberOfTrackCandidates(innerLowerModule);

                    if (innerLowerModule.subdet() == SDL::CPU::Module::Barrel)
                        addTrackCandidateToLowerLayer(tcCand, innerLowerModule.layer(), SDL::CPU::Layer::Barrel);
                    else
                        addTrackCandidateToLowerLayer(tcCand, innerLowerModule.layer(), SDL::CPU::Layer::Endcap);
                }

            }

        }

    }

    for (auto& innerTripletPtr : barrelLayer1.getTripletPtrs())
    {

        // Check if it is a barrel only tracklet
        bool isBBB =
        ((innerTripletPtr->innerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->getModule()).subdet() == SDL::CPU::Module::Barrel) and
        ((innerTripletPtr->innerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->getModule()).subdet() == SDL::CPU::Module::Barrel) and
        ((innerTripletPtr->outerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->getModule()).subdet() == SDL::CPU::Module::Barrel);

        bool is123 = 
        ((innerTripletPtr->innerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->getModule()).layer() == 1) and
        ((innerTripletPtr->innerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->getModule()).layer() == 2) and
        ((innerTripletPtr->outerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->getModule()).layer() == 3);

        if (isBBB and is123)
        {

            // Get reference to the inner lower Module
            Module& innerLowerModule = getModule((innerTripletPtr->innerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->getModule()).detId());

            // unsigned int innerTripletModule2 = (innerTripletPtr->innerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->getModule()).detId();
            unsigned int innerTripletModule2 = (innerTripletPtr->outerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->getModule()).detId();

            // Get reference to the outer lower module
            Module& outerLowerModule = getModule(innerTripletModule2);

            // Loop over outer lower triplets
            for (auto& outerTrackletPtr : outerLowerModule.getTrackletPtrs())
            {

                // Check if it is a barrel only triplet
                bool isOuterBBBB =
                    ((outerTrackletPtr->innerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->getModule()).subdet() == SDL::CPU::Module::Barrel) and
                    ((outerTrackletPtr->innerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->getModule()).subdet() == SDL::CPU::Module::Barrel) and
                    ((outerTrackletPtr->outerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->getModule()).subdet() == SDL::CPU::Module::Barrel) and
                    ((outerTrackletPtr->outerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->getModule()).subdet() == SDL::CPU::Module::Barrel);

                // Check if it is a barrel only triplet
                bool isOuter2356 =
                    ((outerTrackletPtr->innerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->getModule()).layer() == 2) and
                    ((outerTrackletPtr->innerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->getModule()).layer() == 3) and
                    ((outerTrackletPtr->outerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->getModule()).layer() == 5) and
                    ((outerTrackletPtr->outerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->getModule()).layer() == 6);

                if (not (isOuterBBBB and isOuter2356))
                {
                    continue;
                }

                SDL::CPU::Tracklet& outerTracklet = *outerTrackletPtr;

                //============================================================
                //
                //
                // Type 1: 2 3 4 5 + 4 5 6   : layer 1 is missing
                //
                //
                //============================================================
                SDL::CPU::TrackCandidate tcCand(innerTripletPtr, outerTrackletPtr);

                tcCand.runTrackCandidateInnerTripletToOuterTracklet(logLevel_);

                // Count the # of track candidates considered
                incrementNumberOfTrackCandidateCandidates(innerLowerModule);

                if (tcCand.passesTrackCandidateAlgo(algo))
                {

                    // Count the # of track candidates considered
                    incrementNumberOfTrackCandidates(innerLowerModule);

                    if (innerLowerModule.subdet() == SDL::CPU::Module::Barrel)
                        addTrackCandidateToLowerLayer(tcCand, innerLowerModule.layer(), SDL::CPU::Layer::Barrel);
                    else
                        addTrackCandidateToLowerLayer(tcCand, innerLowerModule.layer(), SDL::CPU::Layer::Endcap);
                }

            }

        }

    }

    for (auto& innerTripletPtr : barrelLayer2.getTripletPtrs())
    {

        // Check if it is a barrel only tracklet
        bool isBBB =
        ((innerTripletPtr->innerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->getModule()).subdet() == SDL::CPU::Module::Barrel) and
        ((innerTripletPtr->innerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->getModule()).subdet() == SDL::CPU::Module::Barrel) and
        ((innerTripletPtr->outerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->getModule()).subdet() == SDL::CPU::Module::Barrel);

        bool is234 = 
        ((innerTripletPtr->innerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->getModule()).layer() == 2) and
        ((innerTripletPtr->innerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->getModule()).layer() == 3) and
        ((innerTripletPtr->outerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->getModule()).layer() == 4);

        if (isBBB and is234)
        {

            // Get reference to the inner lower Module
            Module& innerLowerModule = getModule((innerTripletPtr->innerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->getModule()).detId());

            // unsigned int innerTripletModule2 = (innerTripletPtr->innerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->getModule()).detId();
            unsigned int innerTripletModule2 = (innerTripletPtr->outerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->getModule()).detId();

            // Get reference to the outer lower module
            Module& outerLowerModule = getModule(innerTripletModule2);

            // Loop over outer lower triplets
            for (auto& outerTrackletPtr : outerLowerModule.getTrackletPtrs())
            {

                // Check if it is a barrel only triplet
                bool isOuterBBBB =
                    ((outerTrackletPtr->innerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->getModule()).subdet() == SDL::CPU::Module::Barrel) and
                    ((outerTrackletPtr->innerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->getModule()).subdet() == SDL::CPU::Module::Barrel) and
                    ((outerTrackletPtr->outerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->getModule()).subdet() == SDL::CPU::Module::Barrel) and
                    ((outerTrackletPtr->outerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->getModule()).subdet() == SDL::CPU::Module::Barrel);

                // Check if it is a barrel only triplet
                bool isOuter3456 =
                    ((outerTrackletPtr->innerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->getModule()).layer() == 3) and
                    ((outerTrackletPtr->innerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->getModule()).layer() == 4) and
                    ((outerTrackletPtr->outerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->getModule()).layer() == 5) and
                    ((outerTrackletPtr->outerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->getModule()).layer() == 6);

                if (not (isOuterBBBB and isOuter3456))
                {
                    continue;
                }

                SDL::CPU::Tracklet& outerTracklet = *outerTrackletPtr;

                //============================================================
                //
                //
                // Type 1: 2 3 4 5 + 4 5 6   : layer 1 is missing
                //
                //
                //============================================================
                SDL::CPU::TrackCandidate tcCand(innerTripletPtr, outerTrackletPtr);

                tcCand.runTrackCandidateInnerTripletToOuterTracklet(logLevel_);

                // Count the # of track candidates considered
                incrementNumberOfTrackCandidateCandidates(innerLowerModule);

                if (tcCand.passesTrackCandidateAlgo(algo))
                {

                    // Count the # of track candidates considered
                    incrementNumberOfTrackCandidates(innerLowerModule);

                    if (innerLowerModule.subdet() == SDL::CPU::Module::Barrel)
                        addTrackCandidateToLowerLayer(tcCand, innerLowerModule.layer(), SDL::CPU::Layer::Barrel);
                    else
                        addTrackCandidateToLowerLayer(tcCand, innerLowerModule.layer(), SDL::CPU::Layer::Endcap);
                }

            }

        }

    }

}

void SDL::CPU::Event::createTrackCandidatesTest_v2(SDL::CPU::TCAlgo algo)
{

    // September 10, 2020 Consider ALL cases
    // Loop over all tracklets, a-b-c-d go to c then get tracklets or triplets and ask whether segment is shared
    // Ditto for Triplet -> Tracklet
    for (auto& layerPtr : getLayerPtrs())
    {

        for (auto& innerTrackletPtr : layerPtr->getTrackletPtrs())
        {
            SDL::CPU::Module& innerLowerModule = getModule(innerTrackletPtr->innerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->getModule().detId());
            const SDL::CPU::Module& commonModule = innerTrackletPtr->outerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->getModule();

            for (auto& outerTrackletPtr : commonModule.getTrackletPtrs())
            {

                SDL::CPU::TrackCandidate tcCand(innerTrackletPtr, outerTrackletPtr);

                tcCand.runTrackCandidateAlgo(algo, logLevel_);

                // Count the # of track candidates considered
                incrementNumberOfTrackCandidateCandidates(innerLowerModule);

                if (tcCand.passesTrackCandidateAlgo(algo))
                {

                    // Count the # of track candidates considered
                    incrementNumberOfTrackCandidates(innerLowerModule);

                    if (innerLowerModule.subdet() == SDL::CPU::Module::Barrel)
                        addTrackCandidateToLowerLayer(tcCand, innerLowerModule.layer(), SDL::CPU::Layer::Barrel);
                    else
                        addTrackCandidateToLowerLayer(tcCand, innerLowerModule.layer(), SDL::CPU::Layer::Endcap);
                }

            }

            for (auto& outerTripletPtr : commonModule.getTripletPtrs())
            {

                SDL::CPU::TrackCandidate tcCand(innerTrackletPtr, outerTripletPtr);

                tcCand.runTrackCandidateInnerTrackletToOuterTriplet(logLevel_);

                // Count the # of track candidates considered
                incrementNumberOfTrackCandidateCandidates(innerLowerModule);

                if (tcCand.passesTrackCandidateAlgo(algo))
                {

                    // Count the # of track candidates considered
                    incrementNumberOfTrackCandidates(innerLowerModule);

                    if (innerLowerModule.subdet() == SDL::CPU::Module::Barrel)
                        addTrackCandidateToLowerLayer(tcCand, innerLowerModule.layer(), SDL::CPU::Layer::Barrel);
                    else
                        addTrackCandidateToLowerLayer(tcCand, innerLowerModule.layer(), SDL::CPU::Layer::Endcap);
                }

            }

        }

        for (auto& innerTripletPtr : layerPtr->getTripletPtrs())
        {
            SDL::CPU::Module& innerLowerModule = getModule(innerTripletPtr->innerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->getModule().detId());
            const SDL::CPU::Module& commonModule = innerTripletPtr->outerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->getModule();

            for (auto& outerTrackletPtr : commonModule.getTrackletPtrs())
            {

                SDL::CPU::TrackCandidate tcCand(innerTripletPtr, outerTrackletPtr);

                tcCand.runTrackCandidateInnerTripletToOuterTracklet(logLevel_);

                // Count the # of track candidates considered
                incrementNumberOfTrackCandidateCandidates(innerLowerModule);

                if (tcCand.passesTrackCandidateAlgo(algo))
                {

                    // Count the # of track candidates considered
                    incrementNumberOfTrackCandidates(innerLowerModule);

                    if (innerLowerModule.subdet() == SDL::CPU::Module::Barrel)
                        addTrackCandidateToLowerLayer(tcCand, innerLowerModule.layer(), SDL::CPU::Layer::Barrel);
                    else
                        addTrackCandidateToLowerLayer(tcCand, innerLowerModule.layer(), SDL::CPU::Layer::Endcap);
                }

            }

        }

    }

}

void SDL::CPU::Event::createTrackCandidates(SDL::CPU::TCAlgo algo)
{

    // September 10, 2020 Consider ALL cases
    // Loop over all tracklets, a-b-c-d go to c then get tracklets or triplets and ask whether segment is shared
    // Ditto for Triplet -> Tracklet
    for (auto& layerPtr : getLayerPtrs())
    {

        for (auto& innerTrackletPtr : layerPtr->getTrackletPtrs())
        {
            SDL::CPU::Module& innerLowerModule = getModule(innerTrackletPtr->innerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->getModule().detId());
            const SDL::CPU::Module& commonModule = innerTrackletPtr->outerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->getModule();

            for (auto& outerTrackletPtr : commonModule.getTrackletPtrs())
            {

                SDL::CPU::TrackCandidate tcCand(innerTrackletPtr, outerTrackletPtr);

                tcCand.runTrackCandidateAlgo(algo, logLevel_);

                // Count the # of track candidates considered
                incrementNumberOfTrackCandidateCandidates(innerLowerModule);

                if (tcCand.passesTrackCandidateAlgo(algo))
                {

                    // Count the # of track candidates considered
                    incrementNumberOfTrackCandidates(innerLowerModule);

                    if (innerLowerModule.subdet() == SDL::CPU::Module::Barrel)
                        addTrackCandidateToLowerLayer(tcCand, innerLowerModule.layer(), SDL::CPU::Layer::Barrel);
                    else
                        addTrackCandidateToLowerLayer(tcCand, innerLowerModule.layer(), SDL::CPU::Layer::Endcap);
                }

            }

            for (auto& outerTripletPtr : commonModule.getTripletPtrs())
            {

                SDL::CPU::TrackCandidate tcCand(innerTrackletPtr, outerTripletPtr);

                tcCand.runTrackCandidateInnerTrackletToOuterTriplet(logLevel_);

                // Count the # of track candidates considered
                incrementNumberOfTrackCandidateCandidates(innerLowerModule);

                if (tcCand.passesTrackCandidateAlgo(algo))
                {

                    // Count the # of track candidates considered
                    incrementNumberOfTrackCandidates(innerLowerModule);

                    if (innerLowerModule.subdet() == SDL::CPU::Module::Barrel)
                        addTrackCandidateToLowerLayer(tcCand, innerLowerModule.layer(), SDL::CPU::Layer::Barrel);
                    else
                        addTrackCandidateToLowerLayer(tcCand, innerLowerModule.layer(), SDL::CPU::Layer::Endcap);
                }

            }

        }

        for (auto& innerTripletPtr : layerPtr->getTripletPtrs())
        {
            SDL::CPU::Module& innerLowerModule = getModule(innerTripletPtr->innerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->getModule().detId());
            const SDL::CPU::Module& commonModule = innerTripletPtr->outerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->getModule();

            for (auto& outerTrackletPtr : commonModule.getTrackletPtrs())
            {

                SDL::CPU::TrackCandidate tcCand(innerTripletPtr, outerTrackletPtr);

                tcCand.runTrackCandidateInnerTripletToOuterTracklet(logLevel_);

                // Count the # of track candidates considered
                incrementNumberOfTrackCandidateCandidates(innerLowerModule);

                if (tcCand.passesTrackCandidateAlgo(algo))
                {

                    // Count the # of track candidates considered
                    incrementNumberOfTrackCandidates(innerLowerModule);

                    if (innerLowerModule.subdet() == SDL::CPU::Module::Barrel)
                        addTrackCandidateToLowerLayer(tcCand, innerLowerModule.layer(), SDL::CPU::Layer::Barrel);
                    else
                        addTrackCandidateToLowerLayer(tcCand, innerLowerModule.layer(), SDL::CPU::Layer::Endcap);
                }

            }

        }

    }

    for (auto& innerTrackletPtr : getPixelLayer().getTrackletPtrs())
    {
        SDL::CPU::Module& innerLowerModule = getModule(innerTrackletPtr->innerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->getModule().detId());
        const SDL::CPU::Module& commonModule = innerTrackletPtr->outerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->getModule();

        for (auto& outerTrackletPtr : commonModule.getTrackletPtrs())
        {

            SDL::CPU::TrackCandidate tcCand(innerTrackletPtr, outerTrackletPtr);

            tcCand.runTrackCandidateAlgo(algo, logLevel_);

            if (tcCand.passesTrackCandidateAlgo(algo))
            {
                addTrackCandidateToLowerLayer(tcCand, 0, SDL::CPU::Layer::Barrel);
            }

        }

        for (auto& outerTripletPtr : commonModule.getTripletPtrs())
        {

            SDL::CPU::TrackCandidate tcCand(innerTrackletPtr, outerTripletPtr);

            tcCand.runTrackCandidateInnerTrackletToOuterTriplet(logLevel_);

            if (tcCand.passesTrackCandidateAlgo(algo))
            {
                addTrackCandidateToLowerLayer(tcCand, 0, SDL::CPU::Layer::Barrel);
            }

        }

    }

}


void SDL::CPU::Event::createTrackCandidatesFromInnerModulesFromTrackletsToTriplets(unsigned int detId, SDL::CPU::TCAlgo algo)
{

    // Get reference to the inner lower Module
    Module& innerLowerModule = getModule(detId);

    // Triple nested loops
    // Loop over inner lower module for segments
    for (auto& innerTrackletPtr : innerLowerModule.getTrackletPtrs())
    {

        // Get reference to segment in inner lower module
        SDL::CPU::Tracklet& innerTracklet = *innerTrackletPtr;

        // Get the outer mini-doublet module detId
        const SDL::CPU::Module& innerTrackletSecondModule = innerTracklet.innerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->getModule();

        unsigned int innerTrackletSecondModuleDetId = innerTrackletSecondModule.detId();

        // Get connected outer lower module detids
        const std::vector<unsigned int>& connectedModuleDetIds = moduleConnectionMap.getConnectedModuleDetIds(innerTrackletSecondModuleDetId);

        // Loop over connected outer lower modules
        for (auto& outerLowerModuleDetId : connectedModuleDetIds)
        {

            if (not hasModule(outerLowerModuleDetId))
                continue;

            // Get reference to the outer lower module
            Module& outerLowerModule = getModule(outerLowerModuleDetId);

            // Loop over outer lower module mini-doublets
            for (auto& outerTripletPtr : outerLowerModule.getTripletPtrs())
            {

                SDL::CPU::Triplet& outerTriplet = *outerTripletPtr;

                SDL::CPU::TrackCandidate tcCand(innerTrackletPtr, outerTripletPtr);

                tcCand.runTrackCandidateInnerTrackletToOuterTriplet(logLevel_);

                // Count the # of track candidates considered
                incrementNumberOfTrackCandidateCandidates(innerLowerModule);

                if (tcCand.passesTrackCandidateAlgo(algo))
                {

                    // Count the # of track candidates considered
                    incrementNumberOfTrackCandidates(innerLowerModule);

                    if (innerLowerModule.subdet() == SDL::CPU::Module::Barrel)
                        addTrackCandidateToLowerLayer(tcCand, innerLowerModule.layer(), SDL::CPU::Layer::Barrel);
                    else
                        addTrackCandidateToLowerLayer(tcCand, innerLowerModule.layer(), SDL::CPU::Layer::Endcap);
                }

            }

        }

    }

}

// Multiplicity of mini-doublets
unsigned int SDL::CPU::Event::getNumberOfHits() { return hits_.size(); }

// Multiplicity of mini-doublet candidates considered in this event
unsigned int SDL::CPU::Event::getNumberOfHitsByLayerBarrel(unsigned int ilayer) { return n_hits_by_layer_barrel_[ilayer]; }

// Multiplicity of mini-doublet candidates considered in this event
unsigned int SDL::CPU::Event::getNumberOfHitsByLayerEndcap(unsigned int ilayer) { return n_hits_by_layer_endcap_[ilayer]; }

// Multiplicity of mini-doublet candidates considered in this event
unsigned int SDL::CPU::Event::getNumberOfHitsByLayerBarrelUpperModule(unsigned int ilayer) { return n_hits_by_layer_barrel_upper_[ilayer]; }

// Multiplicity of mini-doublet candidates considered in this event
unsigned int SDL::CPU::Event::getNumberOfHitsByLayerEndcapUpperModule(unsigned int ilayer) { return n_hits_by_layer_endcap_upper_[ilayer]; }

// Multiplicity of mini-doublets
unsigned int SDL::CPU::Event::getNumberOfMiniDoublets() { return miniDoublets_.size(); }

// Multiplicity of segments
unsigned int SDL::CPU::Event::getNumberOfSegments() { return segments_.size(); }

// Multiplicity of tracklets
unsigned int SDL::CPU::Event::getNumberOfTracklets() { return tracklets_.size(); }

// Multiplicity of triplets
unsigned int SDL::CPU::Event::getNumberOfTriplets() { return triplets_.size(); }

// Multiplicity of track candidates
unsigned int SDL::CPU::Event::getNumberOfTrackCandidates() { return trackcandidates_.size(); }

// Multiplicity of mini-doublet candidates considered in this event
unsigned int SDL::CPU::Event::getNumberOfMiniDoubletCandidates() { unsigned int n = 0; for (unsigned int i = 0; i < 6; ++i) {n += n_miniDoublet_candidates_by_layer_barrel_[i];} for (unsigned int i = 0; i < 5; ++i) {n += n_miniDoublet_candidates_by_layer_endcap_[i];} return n; }

// Multiplicity of segment candidates considered in this event
unsigned int SDL::CPU::Event::getNumberOfSegmentCandidates() { unsigned int n = 0; for (unsigned int i = 0; i < 6; ++i) {n += n_segment_candidates_by_layer_barrel_[i];} for (unsigned int i = 0; i < 5; ++i) {n += n_segment_candidates_by_layer_endcap_[i];} return n; }

// Multiplicity of tracklet candidates considered in this event
unsigned int SDL::CPU::Event::getNumberOfTrackletCandidates() { unsigned int n = 0; for (unsigned int i = 0; i < 6; ++i) {n += n_tracklet_candidates_by_layer_barrel_[i];} for (unsigned int i = 0; i < 5; ++i) {n += n_tracklet_candidates_by_layer_endcap_[i];} return n; }

// Multiplicity of triplet candidates considered in this event
unsigned int SDL::CPU::Event::getNumberOfTripletCandidates() { unsigned int n = 0; for (unsigned int i = 0; i < 6; ++i) {n += n_triplet_candidates_by_layer_barrel_[i];} for (unsigned int i = 0; i < 5; ++i) {n += n_triplet_candidates_by_layer_endcap_[i];} return n; }

// Multiplicity of track candidate candidates considered in this event
unsigned int SDL::CPU::Event::getNumberOfTrackCandidateCandidates() { unsigned int n = 0; for (unsigned int i = 0; i < 6; ++i) {n += n_trackcandidate_candidates_by_layer_barrel_[i];} for (unsigned int i = 0; i < 5; ++i) {n += n_trackcandidate_candidates_by_layer_endcap_[i];} return n; }

// Multiplicity of mini-doublet candidates considered in this event
unsigned int SDL::CPU::Event::getNumberOfMiniDoubletCandidatesByLayerBarrel(unsigned int ilayer) { return n_miniDoublet_candidates_by_layer_barrel_[ilayer]; }

// Multiplicity of segment candidates considered in this event
unsigned int SDL::CPU::Event::getNumberOfSegmentCandidatesByLayerBarrel(unsigned int ilayer) { return n_segment_candidates_by_layer_barrel_[ilayer]; }

// Multiplicity of tracklet candidates considered in this event
unsigned int SDL::CPU::Event::getNumberOfTrackletCandidatesByLayerBarrel(unsigned int ilayer) { return n_tracklet_candidates_by_layer_barrel_[ilayer]; }

// Multiplicity of triplet candidates considered in this event
unsigned int SDL::CPU::Event::getNumberOfTripletCandidatesByLayerBarrel(unsigned int ilayer) { return n_triplet_candidates_by_layer_barrel_[ilayer]; }

// Multiplicity of track candidate candidates considered in this event
unsigned int SDL::CPU::Event::getNumberOfTrackCandidateCandidatesByLayerBarrel(unsigned int ilayer) { return n_trackcandidate_candidates_by_layer_barrel_[ilayer]; }

// Multiplicity of mini-doublet candidates considered in this event
unsigned int SDL::CPU::Event::getNumberOfMiniDoubletCandidatesByLayerEndcap(unsigned int ilayer) { return n_miniDoublet_candidates_by_layer_endcap_[ilayer]; }

// Multiplicity of segment candidates considered in this event
unsigned int SDL::CPU::Event::getNumberOfSegmentCandidatesByLayerEndcap(unsigned int ilayer) { return n_segment_candidates_by_layer_endcap_[ilayer]; }

// Multiplicity of tracklet candidates considered in this event
unsigned int SDL::CPU::Event::getNumberOfTrackletCandidatesByLayerEndcap(unsigned int ilayer) { return n_tracklet_candidates_by_layer_endcap_[ilayer]; }

// Multiplicity of triplet candidates considered in this event
unsigned int SDL::CPU::Event::getNumberOfTripletCandidatesByLayerEndcap(unsigned int ilayer) { return n_triplet_candidates_by_layer_endcap_[ilayer]; }

// Multiplicity of track candidate candidates considered in this event
unsigned int SDL::CPU::Event::getNumberOfTrackCandidateCandidatesByLayerEndcap(unsigned int ilayer) { return n_trackcandidate_candidates_by_layer_endcap_[ilayer]; }

// Multiplicity of mini-doublet formed in this event
unsigned int SDL::CPU::Event::getNumberOfMiniDoubletsByLayerBarrel(unsigned int ilayer) { return n_miniDoublet_by_layer_barrel_[ilayer]; }

// Multiplicity of segment formed in this event
unsigned int SDL::CPU::Event::getNumberOfSegmentsByLayerBarrel(unsigned int ilayer) { return n_segment_by_layer_barrel_[ilayer]; }

// Multiplicity of tracklet formed in this event
unsigned int SDL::CPU::Event::getNumberOfTrackletsByLayerBarrel(unsigned int ilayer) { return n_tracklet_by_layer_barrel_[ilayer]; }

// Multiplicity of triplet formed in this event
unsigned int SDL::CPU::Event::getNumberOfTripletsByLayerBarrel(unsigned int ilayer) { return n_triplet_by_layer_barrel_[ilayer]; }

// Multiplicity of track candidate formed in this event
unsigned int SDL::CPU::Event::getNumberOfTrackCandidatesByLayerBarrel(unsigned int ilayer) { return n_trackcandidate_by_layer_barrel_[ilayer]; }

// Multiplicity of mini-doublet formed in this event
unsigned int SDL::CPU::Event::getNumberOfMiniDoubletsByLayerEndcap(unsigned int ilayer) { return n_miniDoublet_by_layer_endcap_[ilayer]; }

// Multiplicity of segment formed in this event
unsigned int SDL::CPU::Event::getNumberOfSegmentsByLayerEndcap(unsigned int ilayer) { return n_segment_by_layer_endcap_[ilayer]; }

// Multiplicity of tracklet formed in this event
unsigned int SDL::CPU::Event::getNumberOfTrackletsByLayerEndcap(unsigned int ilayer) { return n_tracklet_by_layer_endcap_[ilayer]; }

// Multiplicity of triplet formed in this event
unsigned int SDL::CPU::Event::getNumberOfTripletsByLayerEndcap(unsigned int ilayer) { return n_triplet_by_layer_endcap_[ilayer]; }

// Multiplicity of track candidate formed in this event
unsigned int SDL::CPU::Event::getNumberOfTrackCandidatesByLayerEndcap(unsigned int ilayer) { return n_trackcandidate_by_layer_endcap_[ilayer]; }

// Multiplicity of hits in this event
void SDL::CPU::Event::incrementNumberOfHits(SDL::CPU::Module& module)
{
    int layer = module.layer();
    int isbarrel = (module.subdet() == SDL::CPU::Module::Barrel);

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
void SDL::CPU::Event::incrementNumberOfMiniDoubletCandidates(SDL::CPU::Module& module)
{
    int layer = module.layer();
    int isbarrel = (module.subdet() == SDL::CPU::Module::Barrel);
    if (isbarrel)
        n_miniDoublet_candidates_by_layer_barrel_[layer-1]++;
    else
        n_miniDoublet_candidates_by_layer_endcap_[layer-1]++;
}

// Multiplicity of segment candidates considered in this event
void SDL::CPU::Event::incrementNumberOfSegmentCandidates(SDL::CPU::Module& module)
{
    int layer = module.layer();
    int isbarrel = (module.subdet() == SDL::CPU::Module::Barrel);
    if (isbarrel)
        n_segment_candidates_by_layer_barrel_[layer-1]++;
    else
        n_segment_candidates_by_layer_endcap_[layer-1]++;
}

// Multiplicity of triplet candidates considered in this event
void SDL::CPU::Event::incrementNumberOfTripletCandidates(SDL::CPU::Module& module)
{
    int layer = module.layer();
    int isbarrel = (module.subdet() == SDL::CPU::Module::Barrel);
    if (isbarrel)
        n_triplet_candidates_by_layer_barrel_[layer-1]++;
    else
        n_triplet_candidates_by_layer_endcap_[layer-1]++;
}

// Multiplicity of tracklet candidates considered in this event
void SDL::CPU::Event::incrementNumberOfTrackletCandidates(SDL::CPU::Layer& _layer)
{
    int layer = _layer.layerIdx();
    int isbarrel = (_layer.subdet() == SDL::CPU::Layer::Barrel);
    if (isbarrel)
        n_tracklet_candidates_by_layer_barrel_[layer-1]++;
    else
        n_tracklet_candidates_by_layer_endcap_[layer-1]++;
}

// Multiplicity of tracklet candidates considered in this event
void SDL::CPU::Event::incrementNumberOfTrackletCandidates(SDL::CPU::Module& _module)
{
    int layer = _module.layer();
    int isbarrel = (_module.subdet() == SDL::CPU::Module::Barrel);
    if (isbarrel)
        n_tracklet_candidates_by_layer_barrel_[layer-1]++;
    else
        n_tracklet_candidates_by_layer_endcap_[layer-1]++;
}

// Multiplicity of track candidate candidates considered in this event
void SDL::CPU::Event::incrementNumberOfTrackCandidateCandidates(SDL::CPU::Layer& _layer)
{
    int layer = _layer.layerIdx();
    int isbarrel = (_layer.subdet() == SDL::CPU::Layer::Barrel);
    if (isbarrel)
        n_trackcandidate_candidates_by_layer_barrel_[layer-1]++;
    else
        n_trackcandidate_candidates_by_layer_endcap_[layer-1]++;
}

// Multiplicity of track candidate candidates considered in this event
void SDL::CPU::Event::incrementNumberOfTrackCandidateCandidates(SDL::CPU::Module& _module)
{
    int layer = _module.layer();
    int isbarrel = (_module.subdet() == SDL::CPU::Module::Barrel);
    if (isbarrel)
        n_trackcandidate_candidates_by_layer_barrel_[layer-1]++;
    else
        n_trackcandidate_candidates_by_layer_endcap_[layer-1]++;
}

// Multiplicity of mini-doublet formed in this event
void SDL::CPU::Event::incrementNumberOfMiniDoublets(SDL::CPU::Module& module)
{
    int layer = module.layer();
    int isbarrel = (module.subdet() == SDL::CPU::Module::Barrel);
    if (isbarrel)
        n_miniDoublet_by_layer_barrel_[layer-1]++;
    else
        n_miniDoublet_by_layer_endcap_[layer-1]++;
}

// Multiplicity of segment formed in this event
void SDL::CPU::Event::incrementNumberOfSegments(SDL::CPU::Module& module)
{
    int layer = module.layer();
    int isbarrel = (module.subdet() == SDL::CPU::Module::Barrel);
    if (isbarrel)
        n_segment_by_layer_barrel_[layer-1]++;
    else
        n_segment_by_layer_endcap_[layer-1]++;
}

// Multiplicity of triplet formed in this event
void SDL::CPU::Event::incrementNumberOfTriplets(SDL::CPU::Module& module)
{
    int layer = module.layer();
    int isbarrel = (module.subdet() == SDL::CPU::Module::Barrel);
    if (isbarrel)
        n_triplet_by_layer_barrel_[layer-1]++;
    else
        n_triplet_by_layer_endcap_[layer-1]++;
}

// Multiplicity of tracklet formed in this event
void SDL::CPU::Event::incrementNumberOfTracklets(SDL::CPU::Layer& _layer)
{
    int layer = _layer.layerIdx();
    int isbarrel = (_layer.subdet() == SDL::CPU::Layer::Barrel);
    if (isbarrel)
        n_tracklet_by_layer_barrel_[layer-1]++;
    else
        n_tracklet_by_layer_endcap_[layer-1]++;
}

// Multiplicity of tracklet formed in this event
void SDL::CPU::Event::incrementNumberOfTracklets(SDL::CPU::Module& _module)
{
    int layer = _module.layer();
    int isbarrel = (_module.subdet() == SDL::CPU::Module::Barrel);
    if (isbarrel)
        n_tracklet_by_layer_barrel_[layer-1]++;
    else
        n_tracklet_by_layer_endcap_[layer-1]++;
}

// Multiplicity of track candidate formed in this event
void SDL::CPU::Event::incrementNumberOfTrackCandidates(SDL::CPU::Layer& _layer)
{
    int layer = _layer.layerIdx();
    int isbarrel = (_layer.subdet() == SDL::CPU::Layer::Barrel);
    if (isbarrel)
        n_trackcandidate_by_layer_barrel_[layer-1]++;
    else
        n_trackcandidate_by_layer_endcap_[layer-1]++;
}

// Multiplicity of track candidate formed in this event
void SDL::CPU::Event::incrementNumberOfTrackCandidates(SDL::CPU::Module& _module)
{
    int layer = _module.layer();
    int isbarrel = (_module.subdet() == SDL::CPU::Module::Barrel);
    if (isbarrel)
        n_trackcandidate_by_layer_barrel_[layer-1]++;
    else
        n_trackcandidate_by_layer_endcap_[layer-1]++;
}

namespace SDL
{
    namespace CPU
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

            for (auto& layerPtr : event.layerPtrs_)
            {
                out << layerPtr;
            }

            return out;
        }

        std::ostream& operator<<(std::ostream& out, const Event* event)
        {
            out << *event;
            return out;
        }
    }

}

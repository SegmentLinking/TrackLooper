#include "EventForAnalysisInterface.h"

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


void SDL::EventForAnalysisInterface::addModulesToAnalysisInterface(struct modules& modulesInGPU, struct miniDoublets* mdsInGPU, struct segments* segmentsInGPU, struct tracklets* trackletsInGPU, struct triplets* tripletsInGPU, struct trackCandidates* trackCandidatesInGPU)
{
    unsigned int lowerModuleIndex = 0;

    for(unsigned int idx = 0; idx < *modulesInGPU.nModules; idx++)
    {
      moduleMapByIndex_[idx] = std::make_shared<SDL::Module>(modulesInGPU.detIds[idx],modulesInGPU.layers[idx], modulesInGPU.rings[idx], modulesInGPU.rods[idx], modulesInGPU.modules[idx], modulesInGPU.isInverted[idx], modulesInGPU.isLower[idx], modulesInGPU.subdets[idx], modulesInGPU.moduleType[idx], modulesInGPU.moduleLayerType[idx],modulesInGPU.sides[idx]);

        modulePointers.push_back(moduleMapByIndex_[idx]);
        if(modulesInGPU.isLower[idx])
        {
            lowerModulePointers.push_back(moduleMapByIndex_[idx]);
            if(trackletsInGPU != nullptr)
            {
                moduleMapByIndex_[idx]->setNumberOfTracklets(trackletsInGPU->nTracklets[lowerModuleIndex]);
            }
            if(tripletsInGPU != nullptr)
            {
                moduleMapByIndex_[idx]->setNumberOfTriplets(tripletsInGPU->nTriplets[lowerModuleIndex]);
            }
            if(trackCandidatesInGPU != nullptr)
            {
                moduleMapByIndex_[idx]->setNumberOfTrackCandidates(trackCandidatesInGPU->nTrackCandidates[lowerModuleIndex]);
                moduleMapByIndex_[idx]->setNumberOfTrackCandidatesT4T4(trackCandidatesInGPU->nTrackCandidatesT4T4[lowerModuleIndex]);
                moduleMapByIndex_[idx]->setNumberOfTrackCandidatesT4T3(trackCandidatesInGPU->nTrackCandidatesT4T3[lowerModuleIndex]);
                moduleMapByIndex_[idx]->setNumberOfTrackCandidatesT3T4(trackCandidatesInGPU->nTrackCandidatesT3T4[lowerModuleIndex]);
            }

            lowerModuleIndex++;
        }
        detIdToIndex_[modulesInGPU.detIds[idx]] = idx;
        
        if(tripletsInGPU != nullptr)
        {
            moduleMapByIndex_[idx]->setNumberOfMiniDoublets(mdsInGPU->nMDs[idx]);
            moduleMapByIndex_[idx]->setNumberOfSegments(segmentsInGPU->nSegments[idx]); 
        }
    }
}

SDL::EventForAnalysisInterface::~EventForAnalysisInterface()
{

}

void SDL::EventForAnalysisInterface::addHitsToAnalysisInterface(struct hits&  hitsInGPU)
{
    for(unsigned int idx = 0; idx < *(hitsInGPU.nHits); idx++)
    {
        std::shared_ptr<Module> lowerModule = moduleMapByIndex_[hitsInGPU.moduleIndices[idx]];

        hits_[idx] = std::make_shared<SDL::Hit>(hitsInGPU.xs[idx],hitsInGPU.ys[idx],hitsInGPU.zs[idx],hitsInGPU.rts[idx],hitsInGPU.phis[idx], hitsInGPU.idxs[idx], lowerModule);
        hitPointers.push_back(hits_[idx]);
        lowerModule->addHit(hits_[idx]);
    }
}

void SDL::EventForAnalysisInterface::addMDsToAnalysisInterface(struct miniDoublets& mdsInGPU)
{
    for(unsigned int idx = 0; idx < modulePointers.size(); idx++) 
    {
        for(unsigned int jdx = 0; jdx < mdsInGPU.nMDs[idx]; jdx++)
        {

            unsigned int mdIndex = idx * N_MAX_MD_PER_MODULES + jdx;
            std::shared_ptr<Hit> lowerHitPtr = hits_[mdsInGPU.hitIndices[2 * mdIndex]];
            std::shared_ptr<Hit> upperHitPtr = hits_[mdsInGPU.hitIndices[2 * mdIndex + 1]];

            miniDoublets_[mdIndex] = std::make_shared<SDL::MiniDoublet>(mdsInGPU.dzs[mdIndex], mdsInGPU.drts[mdIndex], mdsInGPU.dphis[mdIndex], mdsInGPU.dphichanges[mdIndex], mdsInGPU.noShiftedDphis[mdIndex], mdsInGPU.noShiftedDphiChanges[mdIndex], mdsInGPU.dzCuts[mdIndex], mdsInGPU.drtCuts[mdIndex], mdsInGPU.miniCuts[mdIndex], lowerHitPtr, upperHitPtr);

            mdPointers.push_back(miniDoublets_[mdIndex]);
            Module& lowerModule = lowerHitPtr->getModule();
            lowerModule.addMiniDoublet(miniDoublets_[mdIndex]);
            if(lowerModule.subdet() == SDL::Module::Barrel)
            {
                getLayer(lowerModule.layer(),SDL::Layer::Barrel).addMiniDoublet(miniDoublets_[mdIndex]);
            }
            else if(lowerModule.subdet() == SDL::Module::Endcap)
            {
                getLayer(lowerModule.layer(),SDL::Layer::Endcap).addMiniDoublet(miniDoublets_[mdIndex]);
            }
            else
            {
                getPixelLayer().addMiniDoublet(miniDoublets_[mdIndex]);
            }

        }
    }
}

void SDL::EventForAnalysisInterface::addSegmentsToAnalysisInterface(struct segments& segmentsInGPU)
{
    for(unsigned int idx = 0; idx < modulePointers.size(); idx++)
    {
        for(unsigned int jdx = 0; jdx < segmentsInGPU.nSegments[idx]; jdx++)
        {
            unsigned int segmentIndex = idx * N_MAX_SEGMENTS_PER_MODULE + jdx;
            std::shared_ptr<SDL::MiniDoublet> lowerMD =miniDoublets_[segmentsInGPU.mdIndices[segmentIndex * 2]];
            std::shared_ptr<SDL::MiniDoublet> upperMD = miniDoublets_[segmentsInGPU.mdIndices[segmentIndex * 2 + 1]];

            segments_[segmentIndex] = std::make_shared<SDL::Segment>(segmentsInGPU.zIns[segmentIndex], segmentsInGPU.zOuts[segmentIndex], segmentsInGPU.rtIns[segmentIndex], segmentsInGPU.rtOuts[segmentIndex], segmentsInGPU.dPhis[segmentIndex], segmentsInGPU.dPhiMins[segmentIndex], segmentsInGPU.dPhiMaxs[segmentIndex], segmentsInGPU.dPhiChanges[segmentIndex], segmentsInGPU.dPhiChangeMins[segmentIndex], segmentsInGPU.dAlphaInnerMDSegments[segmentIndex], segmentsInGPU.dAlphaOuterMDSegments[segmentIndex], segmentsInGPU.dAlphaInnerMDOuterMDs[segmentIndex], segmentsInGPU.dPhiChangeMaxs[segmentIndex], segmentsInGPU.zLo[segmentIndex], segmentsInGPU.zHi[segmentIndex], segmentsInGPU.rtLo[segmentIndex], segmentsInGPU.rtHi[segmentIndex], segmentsInGPU.sdCut[segmentIndex], segmentsInGPU.dAlphaInnerMDSegmentThreshold[segmentIndex], segmentsInGPU.dAlphaOuterMDSegmentThreshold[segmentIndex], segmentsInGPU.dAlphaInnerMDOuterMDThreshold[segmentIndex], lowerMD, upperMD);
            
            segmentPointers.push_back(segments_[segmentIndex]);
            Module& innerLowerModule = (lowerMD->lowerHitPtr())->getModule();
            innerLowerModule.addSegment(segments_[segmentIndex]);
            if(innerLowerModule.subdet() == SDL::Module::Barrel)
            {
                getLayer(innerLowerModule.layer(),SDL::Layer::Barrel).addSegment(segments_[segmentIndex]);
            }
            else if(innerLowerModule.subdet() == SDL::Module::Endcap)
            {
                getLayer(innerLowerModule.layer(),SDL::Layer::Endcap).addSegment(segments_[segmentIndex]);
            }
            else
            {
                getPixelLayer().addSegment(segments_[segmentIndex]);
                segments_[segmentIndex]->setPixelVariables(segmentsInGPU.ptIn[jdx],segmentsInGPU.ptErr[jdx], segmentsInGPU.px[jdx], segmentsInGPU.py[jdx], segmentsInGPU.pz[jdx], segmentsInGPU.etaErr[jdx]);
            }
        }
    }
}

void SDL::EventForAnalysisInterface::addTrackletsToAnalysisInterface(struct tracklets& trackletsInGPU)
{
    for(unsigned int idx = 0; idx <= lowerModulePointers.size(); idx++) //cheating for pixel module
    {
        for(unsigned int jdx = 0; jdx < trackletsInGPU.nTracklets[idx]; jdx++)
        {
            unsigned int trackletIndex = idx * N_MAX_TRACKLETS_PER_MODULE + jdx;
            std::shared_ptr<Segment> innerSegment = segments_[trackletsInGPU.segmentIndices[2 * trackletIndex]];
            std::shared_ptr<Segment> outerSegment = segments_[trackletsInGPU.segmentIndices[2 * trackletIndex + 1]];
            tracklets_[trackletIndex] = std::make_shared<SDL::Tracklet>(trackletsInGPU.zOut[trackletIndex], trackletsInGPU.rtOut[trackletIndex], trackletsInGPU.deltaPhiPos[trackletIndex], trackletsInGPU.deltaPhi[trackletIndex], trackletsInGPU.betaIn[trackletIndex], trackletsInGPU.betaOut[trackletIndex], trackletsInGPU.zLo[trackletIndex], trackletsInGPU.zHi[trackletIndex], trackletsInGPU.zLoPointed[trackletIndex], trackletsInGPU.zHiPointed[trackletIndex], trackletsInGPU.sdlCut[trackletIndex], trackletsInGPU.betaInCut[trackletIndex], trackletsInGPU.betaOutCut[trackletIndex], trackletsInGPU.deltaBetaCut[trackletIndex], trackletsInGPU.rtLo[trackletIndex], trackletsInGPU.rtHi[trackletIndex], trackletsInGPU.kZ[trackletIndex], innerSegment, outerSegment);

            trackletPointers.push_back(tracklets_[trackletIndex]);
            Module& innerInnerLowerModule = ((innerSegment->innerMiniDoubletPtr())->lowerHitPtr())->getModule();
            innerInnerLowerModule.addTracklet(tracklets_[trackletIndex]);

            if(innerInnerLowerModule.subdet() == SDL::Module::Barrel)
            {
                getLayer(innerInnerLowerModule.layer(),SDL::Layer::Barrel).addTracklet(tracklets_[trackletIndex]);
            }
            else if(innerInnerLowerModule.subdet() == SDL::Module::Endcap)
            {
                getLayer(innerInnerLowerModule.layer(),SDL::Layer::Endcap).addTracklet(tracklets_[trackletIndex]);
            }
            else
            {
                getPixelLayer().addTracklet(tracklets_[trackletIndex]);
            }
        }
    }
}

void SDL::EventForAnalysisInterface::addTripletsToAnalysisInterface(struct triplets& tripletsInGPU)
{
    for(unsigned int idx = 0; idx < lowerModulePointers.size(); idx++)
    {
        for(unsigned int jdx = 0; jdx < tripletsInGPU.nTriplets[idx]; jdx++)
        {
            unsigned int tripletIndex = idx * N_MAX_TRIPLETS_PER_MODULE + jdx;
            std::shared_ptr<Segment> innerSegment = segments_[tripletsInGPU.segmentIndices[2 * tripletIndex]];
            std::shared_ptr<Segment> outerSegment = segments_[tripletsInGPU.segmentIndices[2 * tripletIndex + 1]];

            triplets_[tripletIndex] = std::make_shared<Triplet>(tripletsInGPU.zOut[tripletIndex], tripletsInGPU.rtOut[tripletIndex], tripletsInGPU.deltaPhiPos[tripletIndex], tripletsInGPU.deltaPhi[tripletIndex], tripletsInGPU.betaIn[tripletIndex], tripletsInGPU.betaOut[tripletIndex], tripletsInGPU.zLo[tripletIndex], tripletsInGPU.zHi[tripletIndex], tripletsInGPU.zLoPointed[tripletIndex], tripletsInGPU.zHiPointed[tripletIndex], tripletsInGPU.sdlCut[tripletIndex], tripletsInGPU.betaInCut[tripletIndex], tripletsInGPU.betaOutCut[tripletIndex], tripletsInGPU.deltaBetaCut[tripletIndex], tripletsInGPU.rtLo[tripletIndex], tripletsInGPU.rtHi[tripletIndex], tripletsInGPU.kZ[tripletIndex], innerSegment, outerSegment);

            tripletPointers.push_back(triplets_[tripletIndex]);
            Module& innerInnerLowerModule = ((innerSegment->innerMiniDoubletPtr())->lowerHitPtr())->getModule();
            innerInnerLowerModule.addTriplet(triplets_[tripletIndex]);

            if(innerInnerLowerModule.subdet() == SDL::Module::Barrel)
            {
                getLayer(innerInnerLowerModule.layer(),SDL::Layer::Barrel).addTriplet(triplets_[tripletIndex]);
            }
            else if(innerInnerLowerModule.subdet() == SDL::Module::Endcap)
            {
                getLayer(innerInnerLowerModule.layer(),SDL::Layer::Endcap).addTriplet(triplets_[tripletIndex]);
            }
            //no pixel triplets
        }
    }

}


void SDL::EventForAnalysisInterface::addTrackCandidatesToAnalysisInterface(struct trackCandidates& trackCandidatesInGPU)
{
    for(unsigned int idx = 0; idx <= lowerModulePointers.size(); idx++) //cheating to include pixel track candidate lower module
    {
        for(unsigned int jdx = 0; jdx < trackCandidatesInGPU.nTrackCandidates[idx]; jdx++)
        {
            unsigned int trackCandidateIndex = idx * N_MAX_TRACK_CANDIDATES_PER_MODULE + jdx;
            short trackCandidateType = trackCandidatesInGPU.trackCandidateType[trackCandidateIndex];
            std::shared_ptr<TrackletBase> innerTrackletPtr = nullptr;
            std::shared_ptr<TrackletBase> outerTrackletPtr = nullptr;
            if(trackCandidateType == 0) //T4T4
            {
                innerTrackletPtr = std::dynamic_pointer_cast<TrackletBase>(tracklets_[trackCandidatesInGPU.objectIndices[2 * trackCandidateIndex]]);
                outerTrackletPtr = std::dynamic_pointer_cast<TrackletBase>(tracklets_[trackCandidatesInGPU.objectIndices[2 * trackCandidateIndex + 1]]);
            }
            else if(trackCandidateType == 1) //T4T3
            {
                innerTrackletPtr = std::dynamic_pointer_cast<TrackletBase>(tracklets_[trackCandidatesInGPU.objectIndices[2 * trackCandidateIndex]]);
                outerTrackletPtr = std::dynamic_pointer_cast<TrackletBase>(triplets_[trackCandidatesInGPU.objectIndices[2 * trackCandidateIndex + 1]]);

            }
            else if(trackCandidateType == 2) //T3T4
            {
                innerTrackletPtr = std::dynamic_pointer_cast<TrackletBase>(triplets_[trackCandidatesInGPU.objectIndices[2 * trackCandidateIndex]]);
                outerTrackletPtr = std::dynamic_pointer_cast<TrackletBase>(tracklets_[trackCandidatesInGPU.objectIndices[2 * trackCandidateIndex + 1]]);
            }
            else
            {
                std::cout<<"Issue in TrackCandidatesInGPU struct!!! TrackCandidateType "<<trackCandidateType<<" not one of the approved types!"<<std::endl;
            }
           trackCandidates_[trackCandidateIndex] = std::make_shared<SDL::TrackCandidate>(innerTrackletPtr, outerTrackletPtr, trackCandidateType);
           Module& innerInnerInnerLowerModule = (innerTrackletPtr->innerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr())->getModule();


           innerInnerInnerLowerModule.addTrackCandidate(trackCandidates_[trackCandidateIndex]);
            if(innerInnerInnerLowerModule.subdet() == SDL::Module::Barrel)
            {
                getLayer(innerInnerInnerLowerModule.layer(),SDL::Layer::Barrel).addTrackCandidate(trackCandidates_[trackCandidateIndex]);
            }
            else if(innerInnerInnerLowerModule.subdet() == SDL::Module::Endcap)
            {
                getLayer(innerInnerInnerLowerModule.layer(),SDL::Layer::Endcap).addTrackCandidate(trackCandidates_[trackCandidateIndex]);
            }
            else
            {
                getPixelLayer().addTrackCandidate(trackCandidates_[trackCandidateIndex]);
            }

            //printTrackCandidateLayers(trackCandidates_[trackCandidateIndex]);    

        } 
    }
}


void SDL::EventForAnalysisInterface::printTrackCandidateLayers(std::shared_ptr<TrackCandidate> tc)
{
    auto innerTrackletPtr = tc->innerTrackletBasePtr();
    auto outerTrackletPtr = tc->outerTrackletBasePtr();
    //start printing

    std::cout<<"[";
    Module& innerInnerInnerLowerModule = (innerTrackletPtr->innerSegmentPtr())->innerMiniDoubletPtr()->lowerHitPtr()->getModule();
    if(innerInnerInnerLowerModule.subdet() == SDL::Module::Endcap)
    {
        std::cout<<6 + innerInnerInnerLowerModule.layer()<<",";
    }
    else
    {
        std::cout<<innerInnerInnerLowerModule.layer()<<",";
    }
                
    Module& innerInnerOuterLowerModule = (innerTrackletPtr->innerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->getModule());

    if(innerInnerOuterLowerModule.subdet() == SDL::Module::Endcap)
    {
        std::cout<<6 + innerInnerOuterLowerModule.layer()<<",";
    }
    else
    {
        std::cout<<innerInnerOuterLowerModule.layer()<<",";
    }
                  
    Module& innerOuterInnerLowerModule = (innerTrackletPtr->outerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->getModule());

    if(innerOuterInnerLowerModule.subdet() == SDL::Module::Endcap)
    {
        std::cout<<6 + innerOuterInnerLowerModule.layer()<<",";
    }
    else
    {
        std::cout<<innerOuterInnerLowerModule.layer()<<",";
    }
                
    Module& innerOuterOuterLowerModule = (innerTrackletPtr->outerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->getModule());
    if(innerOuterOuterLowerModule.subdet() == SDL::Module::Endcap)
    {
        std::cout<<6 + innerOuterOuterLowerModule.layer()<<",";
    }
    else
    {
        std::cout<<innerOuterOuterLowerModule.layer()<<",";
    }
    
    Module& outerOuterInnerLowerModule = (outerTrackletPtr->outerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->getModule());
    if(outerOuterInnerLowerModule.subdet() == SDL::Module::Endcap)
    {
        std::cout<<6 + outerOuterInnerLowerModule.layer()<<",";
    }
    else
    {
        std::cout<<outerOuterInnerLowerModule.layer()<<",";
    }
        
    Module& outerOuterOuterLowerModule = (outerTrackletPtr->outerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->getModule());

    if(outerOuterOuterLowerModule.subdet() == SDL::Module::Endcap)
    {
        std::cout<<6 + outerOuterOuterLowerModule.layer()<<"]"<<std::endl;
    }
    else
    {
        std::cout<<outerOuterOuterLowerModule.layer()<<"]"<<std::endl;
    } 
}


SDL::EventForAnalysisInterface::EventForAnalysisInterface(struct modules* modulesInGPU, struct hits* hitsInGPU, struct miniDoublets* mdsInGPU, struct segments* segmentsInGPU, struct tracklets* trackletsInGPU, struct triplets* tripletsInGPU, struct trackCandidates* trackCandidatesInGPU)
{
    createLayers();
    addModulesToAnalysisInterface(*modulesInGPU,mdsInGPU,segmentsInGPU,trackletsInGPU, tripletsInGPU, trackCandidatesInGPU);
    if(hitsInGPU != nullptr)
    {
        addHitsToAnalysisInterface(*hitsInGPU);
    }
    if(mdsInGPU != nullptr)
    {
        addMDsToAnalysisInterface(*mdsInGPU);
    }
    if(segmentsInGPU != nullptr)
    {
        addSegmentsToAnalysisInterface(*segmentsInGPU);
    }
    if(trackletsInGPU != nullptr)
    {
        addTrackletsToAnalysisInterface(*trackletsInGPU);
    }
    if(tripletsInGPU != nullptr)
    {
        addTripletsToAnalysisInterface(*tripletsInGPU);
    }
    if(trackCandidatesInGPU != nullptr)
    {
//        addTrackCandidatesToAnalysisInterface(*trackCandidatesInGPU);
    }
}

const std::vector<std::shared_ptr<SDL::Module>> SDL::EventForAnalysisInterface::getModulePtrs() const
{
    return modulePointers;
}

const std::vector<std::shared_ptr<SDL::Module>> SDL::EventForAnalysisInterface::getLowerModulePtrs() const
{
    return lowerModulePointers;
}

SDL::Layer& SDL::EventForAnalysisInterface::getPixelLayer()
{
    return pixelLayer_;
}

void SDL::EventForAnalysisInterface::createLayers()
{
    // Create barrel layers
    for (int ilayer = SDL::Layer::BarrelLayer0; ilayer < SDL::Layer::nBarrelLayer; ++ilayer)
    {
        barrelLayers_[ilayer] = std::make_shared<SDL::Layer>(ilayer, SDL::Layer::Barrel);
        layerPtrs_.push_back(barrelLayers_[ilayer]);
    }

    // Create endcap layers
    for (int ilayer = SDL::Layer::EndcapLayer0; ilayer < SDL::Layer::nEndcapLayer; ++ilayer)
    {
        endcapLayers_[ilayer] = std::make_shared<SDL::Layer>(ilayer, SDL::Layer::Endcap);
        layerPtrs_.push_back(endcapLayers_[ilayer]);
    }
}

SDL::Layer& SDL::EventForAnalysisInterface::getLayer(int ilayer, SDL::Layer::SubDet subdet)
{
    if (subdet == SDL::Layer::Barrel)
        return *barrelLayers_[ilayer];
    else // if (subdet == SDL::Layer::Endcap)
        return *endcapLayers_[ilayer];
}

const std::vector<std::shared_ptr<SDL::Layer>> SDL::EventForAnalysisInterface::getLayerPtrs() const
{
    return layerPtrs_;
}


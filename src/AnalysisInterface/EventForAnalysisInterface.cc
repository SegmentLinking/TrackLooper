#include "EventForAnalysisInterface.h"

const unsigned int N_MAX_HITS_PER_MODULE = 100;
const unsigned int N_MAX_MDS_PER_MODULE = 100;
const unsigned int N_MAX_SEGMENTS_PER_MODULE = 600; //WHY!
const unsigned int MAX_CONNECTED_MODULES = 40;
const unsigned int N_MAX_TRACKLETS_PER_MODULE = 5000;//temporary
const unsigned int N_MAX_TRIPLETS_PER_MODULE = 1000; 

void SDL::EventForAnalysisInterface::addModulesToAnalysisInterface(struct modules& modulesInGPU, struct miniDoublets* mdsInGPU, struct segments* segmentsInGPU, struct tracklets* trackletsInGPU, struct triplets* tripletsInGPU)
{
    unsigned int lowerModuleIndex = 0;

    for(unsigned int idx = 0; idx < *modulesInGPU.nModules; idx++)
    {
        moduleMapByIndex_[idx] = new SDL::Module(modulesInGPU.detIds[idx],modulesInGPU.layers[idx], modulesInGPU.rings[idx], modulesInGPU.rods[idx], modulesInGPU.modules[idx], modulesInGPU.isInverted[idx], modulesInGPU.isLower[idx], modulesInGPU.subdets[idx], modulesInGPU.moduleType[idx], modulesInGPU.moduleLayerType[idx],modulesInGPU.sides[idx]);

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

void SDL::EventForAnalysisInterface::addHitsToAnalysisInterface(struct hits&  hitsInGPU)
{
    for(unsigned int idx = 0; idx < *(hitsInGPU.nHits); idx++)
    {
        Module* lowerModule = moduleMapByIndex_[hitsInGPU.moduleIndices[idx]];

        hits_[idx] = new SDL::Hit(hitsInGPU.xs[idx],hitsInGPU.ys[idx],hitsInGPU.zs[idx],hitsInGPU.rts[idx],hitsInGPU.phis[idx], idx, lowerModule);
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

            unsigned int mdIndex = idx * N_MAX_MDS_PER_MODULE + jdx;
            Hit* lowerHitPtr = hits_[mdsInGPU.hitIndices[2 * mdIndex]];
            Hit* upperHitPtr = hits_[mdsInGPU.hitIndices[2 * mdIndex + 1]];

            miniDoublets_[mdIndex] = new SDL::MiniDoublet(mdsInGPU.dzs[mdIndex], mdsInGPU.dphis[mdIndex], mdsInGPU.dphichanges[mdIndex], mdsInGPU.noShiftedDphis[mdIndex], mdsInGPU.noShiftedDphiChanges[mdIndex], lowerHitPtr, upperHitPtr);

            mdPointers.push_back(miniDoublets_[mdIndex]);
            Module& lowerModule = lowerHitPtr->getModule();
            lowerModule.addMiniDoublet(miniDoublets_[mdIndex]);
            if(lowerModule.subdet() == SDL::Module::Barrel)
            {
                getLayer(lowerModule.layer(),SDL::Layer::Barrel).addMiniDoublet(miniDoublets_[mdIndex]);
            }
            else
            {
                getLayer(lowerModule.layer(),SDL::Layer::Endcap).addMiniDoublet(miniDoublets_[mdIndex]);
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
            MiniDoublet* lowerMD =miniDoublets_[segmentsInGPU.mdIndices[segmentIndex * 2]];
            MiniDoublet* upperMD = miniDoublets_[segmentsInGPU.mdIndices[segmentIndex * 2 + 1]];

            segments_[segmentIndex] = new SDL::Segment(segmentsInGPU.zIns[segmentIndex], segmentsInGPU.zOuts[segmentIndex], segmentsInGPU.rtIns[segmentIndex], segmentsInGPU.rtOuts[segmentIndex], segmentsInGPU.dPhis[segmentIndex], segmentsInGPU.dPhiMins[segmentIndex], segmentsInGPU.dPhiMaxs[segmentIndex], segmentsInGPU.dPhiChanges[segmentIndex], segmentsInGPU.dPhiChangeMins[segmentIndex], segmentsInGPU.dAlphaInnerMDSegments[segmentIndex], segmentsInGPU.dAlphaOuterMDSegments[segmentIndex], segmentsInGPU.dAlphaInnerMDOuterMDs[segmentIndex], segmentsInGPU.dPhiChangeMaxs[segmentIndex], lowerMD, upperMD);
            

            segmentPointers.push_back(segments_[segmentIndex]);
            Module& innerLowerModule = (lowerMD->lowerHitPtr())->getModule();
            innerLowerModule.addSegment(segments_[segmentIndex]);
            if(innerLowerModule.subdet() == SDL::Module::Barrel)
            {
                getLayer(innerLowerModule.layer(),SDL::Layer::Barrel).addSegment(segments_[segmentIndex]);
            }
            else
            {
                getLayer(innerLowerModule.layer(),SDL::Layer::Endcap).addSegment(segments_[segmentIndex]);
            }
        }
    }
}

//void SDL::EventForAnalysisInterface::addTrackletsToAnalysisInterface(struct tracklets& trackletsInGPU)
//{
//    for(unsigned int idx = 0; idx < lowerModulePointers.size(); idx++)
//    {
//        for(unsigned int jdx = 0; jdx < trackletsInGPU.nTracklets[idx]; jdx++)
//        {
//            unsigned int trackletIndex = idx * N_MAX_TRACKLETS_PER_MODULE + jdx;
//            Segment* innerSegment = segments_[trackletsInGPU.segmentIndices[2 * trackletIndex]];
//            Segment* outerSegment = segments_[trackletsInGPU.segmentIndices[2 * trackletIndex + 1]];
//            tracklets_[trackletIndex] = new SDL::Tracklet(trackletsInGPU.zOut[trackletIndex], trackletsInGPU.rtOut[trackletIndex], trackletsInGPU.deltaPhiPos[trackletIndex], trackletsInGPU.deltaPhi[trackletIndex], trackletsInGPU.betaIn[trackletIndex], trackletsInGPU.betaOut[trackletIndex], trackletsInGPU.betaInCut[trackletIndex], trackletsInGPU.betaOutCut[trackletIndex], trackletsInGPU.dBetaCut[trackletIndex], innerSegment, outerSegment);
//
//            trackletPointers.push_back(tracklets_[trackletIndex]);
//            Module& innerInnerLowerModule = ((innerSegment->innerMiniDoubletPtr())->lowerHitPtr())->getModule();
//            innerInnerLowerModule.addTracklet(tracklets_[trackletIndex]);
//
//            if(innerInnerLowerModule.subdet() == SDL::Module::Barrel)
//            {
//                getLayer(innerInnerLowerModule.layer(),SDL::Layer::Barrel).addTracklet(tracklets_[trackletIndex]);
//            }
//            else
//            {
//                getLayer(innerInnerLowerModule.layer(),SDL::Layer::Endcap).addTracklet(tracklets_[trackletIndex]);
//            }
//        }
//    }
//}
//
//void SDL::EventForAnalysisInterface::addTripletsToAnalysisInterface(struct triplets& tripletsInGPU)
//{
//    for(unsigned int idx = 0; idx < lowerModulePointers.size(); idx++)
//    {
//        for(unsigned int jdx = 0; jdx < tripletsInGPU.nTriplets[idx]; jdx++)
//        {
//            unsigned int tripletIndex = idx * N_MAX_TRIPLETS_PER_MODULE + jdx;
//            Segment* innerSegment = segments_[tripletsInGPU.segmentIndices[2 * tripletIndex]];
//            Segment* outerSegment = segments_[tripletsInGPU.segmentIndices[2 * tripletIndex + 1]];
//            triplets_[tripletIndex] = new SDL::Triplet(tripletsInGPU.zOut[tripletIndex], tripletsInGPU.rtOut[tripletIndex], tripletsInGPU.deltaPhiPos[tripletIndex], tripletsInGPU.deltaPhi[tripletIndex], tripletsInGPU.betaIn[tripletIndex], tripletsInGPU.betaOut[tripletIndex], tripletsInGPU.betaInCut[tripletIndex], tripletsInGPU.betaOutCut[tripletIndex], tripletsInGPU.dBetaCut[tripletIndex], innerSegment, outerSegment);
//
//            tripletPointers.push_back(triplets_[tripletIndex]);
//            Module& innerInnerLowerModule = ((innerSegment->innerMiniDoubletPtr())->lowerHitPtr())->getModule();
//            innerInnerLowerModule.addTriplet(triplets_[tripletIndex]);
//
//            if(innerInnerLowerModule.subdet() == SDL::Module::Barrel)
//            {
//                getLayer(innerInnerLowerModule.layer(),SDL::Layer::Barrel).addTriplet(triplets_[tripletIndex]);
//            }
//            else
//            {
//                getLayer(innerInnerLowerModule.layer(),SDL::Layer::Endcap).addTriplet(triplets_[tripletIndex]);
//            }
//        }
//    }
//
//}

SDL::EventForAnalysisInterface::EventForAnalysisInterface(struct modules* modulesInGPU, struct hits* hitsInGPU, struct miniDoublets* mdsInGPU, struct segments* segmentsInGPU, struct tracklets* trackletsInGPU, struct triplets* tripletsInGPU)
{
//    createLayers();
//    addModulesToAnalysisInterface(*modulesInGPU,mdsInGPU,segmentsInGPU,trackletsInGPU, tripletsInGPU);
//    if(hitsInGPU != nullptr)
//    {
//        addHitsToAnalysisInterface(*hitsInGPU);
//    }
//    if(mdsInGPU != nullptr)
//    {
//        addMDsToAnalysisInterface(*mdsInGPU);
//    }
//    if(segmentsInGPU != nullptr)
//    {
//        addSegmentsToAnalysisInterface(*segmentsInGPU);
//    }
    //if(trackletsInGPU != nullptr)
    //{
    //    addTrackletsToAnalysisInterface(*trackletsInGPU);
    //}
    //if(tripletsInGPU != nullptr)
    //{
    //    addTripletsToAnalysisInterface(*tripletsInGPU);
    //}
}

const std::vector<SDL::Module*> SDL::EventForAnalysisInterface::getModulePtrs() const
{
    return modulePointers;
}

const std::vector<SDL::Module*> SDL::EventForAnalysisInterface::getLowerModulePtrs() const
{
    return lowerModulePointers;
}

void SDL::EventForAnalysisInterface::createLayers()
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

SDL::Layer& SDL::EventForAnalysisInterface::getLayer(int ilayer, SDL::Layer::SubDet subdet)
{
    if (subdet == SDL::Layer::Barrel)
        return barrelLayers_[ilayer];
    else // if (subdet == SDL::Layer::Endcap)
        return endcapLayers_[ilayer];
}

const std::vector<SDL::Layer*> SDL::EventForAnalysisInterface::getLayerPtrs() const
{
    return layerPtrs_;
}


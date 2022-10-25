#include "AccessHelper.h"

// ===============
// ----* pLS *----
// ===============

//____________________________________________________________________________________________
std::vector<unsigned int> getPixelHitsFrompLS(SDL::Event* event, unsigned int pLS)
{
    SDL::segments& segments_ = *(event->getSegments());
    SDL::miniDoublets& miniDoublets_ = *(event->getMiniDoublets());
    SDL::objectRanges& rangesInGPU = (*event->getRanges());
    SDL::modules& modulesInGPU = (*event->getModules());
    const unsigned int pLS_offset = rangesInGPU.segmentModuleIndices[*(modulesInGPU.nLowerModules)];
    unsigned int MD_1 = segments_.mdIndices[2 * (pLS + pLS_offset)];
    unsigned int MD_2 = segments_.mdIndices[2 * (pLS + pLS_offset) + 1];
    unsigned int hit_1 = miniDoublets_.anchorHitIndices[MD_1];
    unsigned int hit_2 = miniDoublets_.outerHitIndices [MD_1];
    unsigned int hit_3 = miniDoublets_.anchorHitIndices[MD_2];
    unsigned int hit_4 = miniDoublets_.outerHitIndices [MD_2];
    if (hit_3 == hit_4)
        return {hit_1, hit_2, hit_3};
    else
        return {hit_1, hit_2, hit_3, hit_4};
}

//____________________________________________________________________________________________
std::vector<unsigned int> getPixelHitIdxsFrompLS(SDL::Event* event, unsigned int pLS)
{
    SDL::hits& hitsInGPU = *(event->getHits());
    std::vector<unsigned int> hits = getPixelHitsFrompLS(event, pLS);
    std::vector<unsigned int> hitidxs;
    for (auto& hit : hits)
        hitidxs.push_back(hitsInGPU.idxs[hit]);
    return hitidxs;
}

//____________________________________________________________________________________________
std::vector<unsigned int> getPixelHitTypesFrompLS(SDL::Event* event, unsigned int pLS)
{
    SDL::hits& hitsInGPU = *(event->getHits());
    std::vector<unsigned int> hits = getPixelHitsFrompLS(event, pLS);
    std::vector<unsigned int> hittypes;
    for (auto& hit : hits)
        hittypes.push_back(0);
    return hittypes;
}

// ==============
// ----* MD *----
// ==============

//____________________________________________________________________________________________
std::vector<unsigned int> getHitsFromMD(SDL::Event* event, unsigned int MD)
{
    SDL::miniDoublets& miniDoublets_ = *(event->getMiniDoublets());
    unsigned int hit_1 = miniDoublets_.anchorHitIndices[MD];
    unsigned int hit_2 = miniDoublets_.outerHitIndices [MD];
    return {hit_1, hit_2};
}

// ==============
// ----* LS *----
// ==============

//____________________________________________________________________________________________
std::vector<unsigned int> getMDsFromLS(SDL::Event* event, unsigned int LS)
{
    SDL::segments& segments_ = *(event->getSegments());
    unsigned int MD_1 = segments_.mdIndices[2 * LS];
    unsigned int MD_2 = segments_.mdIndices[2 * LS + 1];
    return {MD_1, MD_2};
}

//____________________________________________________________________________________________
std::vector<unsigned int> getHitsFromLS(SDL::Event* event, unsigned int LS)
{
    std::vector<unsigned int> MDs = getMDsFromLS(event, LS);
    std::vector<unsigned int> hits_0 = getHitsFromMD(event, MDs[0]);
    std::vector<unsigned int> hits_1 = getHitsFromMD(event, MDs[1]);
    return {hits_0[0], hits_0[1], hits_1[0], hits_1[1]};
}

// ==============
// ----* T3 *----
// ==============

//____________________________________________________________________________________________
std::vector<unsigned int> getLSsFromT3(SDL::Event* event, unsigned int T3)
{
    SDL::triplets& triplets_ = *(event->getTriplets());
    unsigned int LS_1 = triplets_.segmentIndices[2 * T3];
    unsigned int LS_2 = triplets_.segmentIndices[2 * T3 + 1];
    return {LS_1, LS_2};
}

//____________________________________________________________________________________________
std::vector<unsigned int> getMDsFromT3(SDL::Event* event, unsigned int T3)
{
    std::vector<unsigned int> LSs = getLSsFromT3(event, T3);
    std::vector<unsigned int> MDs_0 = getMDsFromLS(event, LSs[0]);
    std::vector<unsigned int> MDs_1 = getMDsFromLS(event, LSs[1]);
    return {MDs_0[0], MDs_0[1], MDs_1[1]};
}

//____________________________________________________________________________________________
std::vector<unsigned int> getHitsFromT3(SDL::Event* event, unsigned int T3)
{
    std::vector<unsigned int> MDs = getMDsFromT3(event, T3);
    std::vector<unsigned int> hits_0 = getHitsFromMD(event, MDs[0]);
    std::vector<unsigned int> hits_1 = getHitsFromMD(event, MDs[1]);
    std::vector<unsigned int> hits_2 = getHitsFromMD(event, MDs[2]);
    return {hits_0[0], hits_0[1], hits_1[0], hits_1[1], hits_2[0], hits_2[1]};
}

// ==============
// ----* T5 *----
// ==============

//____________________________________________________________________________________________
std::vector<unsigned int> getT3sFromT5(SDL::Event* event, unsigned int T5)
{
    SDL::quintuplets& quintuplets_ = *(event->getQuintuplets());
    unsigned int T3_1 = quintuplets_.tripletIndices[2 * T5];
    unsigned int T3_2 = quintuplets_.tripletIndices[2 * T5 + 1];
    return {T3_1, T3_2};
}

//____________________________________________________________________________________________
std::vector<unsigned int> getLSsFromT5(SDL::Event* event, unsigned int T5)
{
    std::vector<unsigned int> T3s = getT3sFromT5(event, T5);
    std::vector<unsigned int> LSs_0 = getLSsFromT3(event, T3s[0]);
    std::vector<unsigned int> LSs_1 = getLSsFromT3(event, T3s[1]);
    return {LSs_0[0], LSs_0[1], LSs_1[0], LSs_1[1]};
}

//____________________________________________________________________________________________
std::vector<unsigned int> getMDsFromT5(SDL::Event* event, unsigned int T5)
{
    SDL::segments& segments_ = *(event->getSegments());
    std::vector<unsigned int> LSs = getLSsFromT5(event, T5);
    std::vector<unsigned int> MDs_0 = getMDsFromLS(event, LSs[0]);
    std::vector<unsigned int> MDs_1 = getMDsFromLS(event, LSs[1]);
    std::vector<unsigned int> MDs_2 = getMDsFromLS(event, LSs[2]);
    std::vector<unsigned int> MDs_3 = getMDsFromLS(event, LSs[3]);
    return {MDs_0[0], MDs_0[1], MDs_1[1], MDs_2[1], MDs_3[1]};
}

//____________________________________________________________________________________________
std::vector<unsigned int> getHitsFromT5(SDL::Event* event, unsigned int T5)
{
    std::vector<unsigned int> MDs = getMDsFromT5(event, T5);
    std::vector<unsigned int> hits_0 = getHitsFromMD(event, MDs[0]);
    std::vector<unsigned int> hits_1 = getHitsFromMD(event, MDs[1]);
    std::vector<unsigned int> hits_2 = getHitsFromMD(event, MDs[2]);
    std::vector<unsigned int> hits_3 = getHitsFromMD(event, MDs[3]);
    std::vector<unsigned int> hits_4 = getHitsFromMD(event, MDs[4]);
    return {hits_0[0], hits_0[1], hits_1[0], hits_1[1], hits_2[0], hits_2[1], hits_3[0], hits_3[1], hits_4[0], hits_4[1]};
}

//____________________________________________________________________________________________
std::vector<unsigned int> getHitIdxsFromT5(SDL::Event* event, unsigned int T5)
{
    SDL::hits& hitsInGPU = *(event->getHits());
    std::vector<unsigned int> hits = getHitsFromT5(event, T5);
    std::vector<unsigned int> hitidxs;
    for (auto& hit : hits)
        hitidxs.push_back(hitsInGPU.idxs[hit]);
    return hitidxs;
}

//____________________________________________________________________________________________
std::vector<unsigned int> getHitTypesFromT5(SDL::Event* event, unsigned int T5)
{
    return {4, 4, 4, 4, 4, 4, 4, 4, 4, 4};;
}

// ===============
// ----* pT3 *----
// ===============

//____________________________________________________________________________________________
unsigned int getPixelLSFrompT3(SDL::Event* event, unsigned int pT3)
{
    SDL::pixelTriplets& pixelTriplets_ = *(event->getPixelTriplets());
    SDL::objectRanges& rangesInGPU = (*event->getRanges());
    SDL::modules& modulesInGPU = (*event->getModules());
    const unsigned int pLS_offset = rangesInGPU.segmentModuleIndices[*(modulesInGPU.nLowerModules)];
    return pixelTriplets_.pixelSegmentIndices[pT3] - pLS_offset;
}

//____________________________________________________________________________________________
unsigned int getT3FrompT3(SDL::Event* event, unsigned int pT3)
{
    SDL::pixelTriplets& pixelTriplets_ = *(event->getPixelTriplets());
    return pixelTriplets_.tripletIndices[pT3];
}

//____________________________________________________________________________________________
std::vector<unsigned int> getLSsFrompT3(SDL::Event* event, unsigned int pT3)
{
    unsigned int T3 = getT3FrompT3(event, pT3);
    return getLSsFromT3(event, T3);
}

//____________________________________________________________________________________________
std::vector<unsigned int> getMDsFrompT3(SDL::Event* event, unsigned int pT3)
{
    unsigned int T3 = getT3FrompT3(event, pT3);
    return getMDsFromT3(event, T3);
}

//____________________________________________________________________________________________
std::vector<unsigned int> getOuterTrackerHitsFrompT3(SDL::Event* event, unsigned int pT3)
{
    unsigned int T3 = getT3FrompT3(event, pT3);
    return getHitsFromT3(event, T3);
}

//____________________________________________________________________________________________
std::vector<unsigned int> getPixelHitsFrompT3(SDL::Event* event, unsigned int pT3)
{
    unsigned int pLS = getPixelLSFrompT3(event, pT3);
    return getPixelHitsFrompLS(event, pLS);
}

//____________________________________________________________________________________________
std::vector<unsigned int> getHitsFrompT3(SDL::Event* event, unsigned int pT3)
{
    unsigned int pLS = getPixelLSFrompT3(event, pT3);
    unsigned int T3 = getT3FrompT3(event, pT3);
    std::vector<unsigned int> pixelHits = getPixelHitsFrompLS(event, pLS);
    std::vector<unsigned int> outerTrackerHits = getHitsFromT3(event, T3);
    pixelHits.insert(pixelHits.end(), outerTrackerHits.begin(), outerTrackerHits.end());
    return pixelHits;
}

//____________________________________________________________________________________________
std::vector<unsigned int> getHitIdxsFrompT3(SDL::Event* event, unsigned int pT3)
{
    SDL::hits& hitsInGPU = *(event->getHits());
    std::vector<unsigned int> hits = getHitsFrompT3(event, pT3);
    std::vector<unsigned int> hitidxs;
    for (auto& hit : hits)
        hitidxs.push_back(hitsInGPU.idxs[hit]);
    return hitidxs;
}

//____________________________________________________________________________________________
std::vector<unsigned int> getHitTypesFrompT3(SDL::Event* event, unsigned int pT3)
{
    unsigned int pLS = getPixelLSFrompT3(event, pT3);
    std::vector<unsigned int> pixelHits = getPixelHitsFrompLS(event, pLS);
    // pixel Hits list will be either 3 or 4 and depending on it return accordingly
    if (pixelHits.size() == 3)
        return {0, 0, 0, 4, 4, 4, 4, 4, 4};
    else
        return {0, 0, 0, 0, 4, 4, 4, 4, 4, 4};
}

// ===============
// ----* pT5 *----
// ===============

//____________________________________________________________________________________________
unsigned int getPixelLSFrompT5(SDL::Event* event, unsigned int pT5)
{
    SDL::pixelQuintuplets& pixelQuintuplets_ = *(event->getPixelQuintuplets());
    SDL::objectRanges& rangesInGPU = (*event->getRanges());
    SDL::modules& modulesInGPU = (*event->getModules());
    const unsigned int pLS_offset = rangesInGPU.segmentModuleIndices[*(modulesInGPU.nLowerModules)];
    return pixelQuintuplets_.pixelIndices[pT5] - pLS_offset;
}

//____________________________________________________________________________________________
unsigned int getT5FrompT5(SDL::Event* event, unsigned int pT5)
{
    SDL::pixelQuintuplets& pixelQuintuplets_ = *(event->getPixelQuintuplets());
    return pixelQuintuplets_.T5Indices[pT5];
}

//____________________________________________________________________________________________
std::vector<unsigned int> getT3sFrompT5(SDL::Event* event, unsigned int pT5)
{
    unsigned int T5 = getT5FrompT5(event, pT5);
    return getT3sFromT5(event, T5);
}

//____________________________________________________________________________________________
std::vector<unsigned int> getLSsFrompT5(SDL::Event* event, unsigned int pT5)
{
    unsigned int T5 = getT5FrompT5(event, pT5);
    return getLSsFromT5(event, T5);
}

//____________________________________________________________________________________________
std::vector<unsigned int> getMDsFrompT5(SDL::Event* event, unsigned int pT5)
{
    unsigned int T5 = getT5FrompT5(event, pT5);
    return getMDsFromT5(event, T5);
}

//____________________________________________________________________________________________
std::vector<unsigned int> getOuterTrackerHitsFrompT5(SDL::Event* event, unsigned int pT5)
{
    unsigned int T5 = getT5FrompT5(event, pT5);
    return getHitsFromT5(event, T5);
}

//____________________________________________________________________________________________
std::vector<unsigned int> getPixelHitsFrompT5(SDL::Event* event, unsigned int pT5)
{
    unsigned int pLS = getPixelLSFrompT5(event, pT5);
    return getPixelHitsFrompLS(event, pLS);
}


//____________________________________________________________________________________________
std::vector<unsigned int> getHitsFrompT5(SDL::Event* event, unsigned int pT5)
{
    unsigned int pLS = getPixelLSFrompT5(event, pT5);
    unsigned int T5 = getT5FrompT5(event, pT5);
    std::vector<unsigned int> pixelHits = getPixelHitsFrompLS(event, pLS);
    std::vector<unsigned int> outerTrackerHits = getHitsFromT5(event, T5);
    pixelHits.insert(pixelHits.end(), outerTrackerHits.begin(), outerTrackerHits.end());
    return pixelHits;
}

//____________________________________________________________________________________________
std::vector<unsigned int> getHitIdxsFrompT5(SDL::Event* event, unsigned int pT5)
{
    SDL::hits& hitsInGPU = *(event->getHits());
    std::vector<unsigned int> hits = getHitsFrompT5(event, pT5);
    std::vector<unsigned int> hitidxs;
    for (auto& hit : hits)
        hitidxs.push_back(hitsInGPU.idxs[hit]);
    return hitidxs;
}

//____________________________________________________________________________________________
std::vector<unsigned int> getHitTypesFrompT5(SDL::Event* event, unsigned int pT5)
{
    unsigned int pLS = getPixelLSFrompT5(event, pT5);
    std::vector<unsigned int> pixelHits = getPixelHitsFrompLS(event, pLS);
    // pixel Hits list will be either 3 or 4 and depending on it return accordingly
    if (pixelHits.size() == 3)
        return {0, 0, 0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4};
    else
        return {0, 0, 0, 0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4};
}


#include "write_sdl_ntuple.h"

//________________________________________________________________________________________________________________________________
void createOutputBranches()
{
    // Setup output TTree
    ana.tx->createBranch<vector<float>>("sim_pt");
    ana.tx->createBranch<vector<float>>("sim_eta");
    ana.tx->createBranch<vector<float>>("sim_phi");
    ana.tx->createBranch<vector<float>>("sim_pca_dxy");
    ana.tx->createBranch<vector<float>>("sim_pca_dz");
    ana.tx->createBranch<vector<int>>("sim_q");
    ana.tx->createBranch<vector<int>>("sim_event");
    ana.tx->createBranch<vector<int>>("sim_pdgId");
    ana.tx->createBranch<vector<int>>("sim_bunchCrossing");
    ana.tx->createBranch<vector<int>>("sim_parentVtxIdx");

    // Sim vertex
    ana.tx->createBranch<vector<float>>("simvtx_x");
    ana.tx->createBranch<vector<float>>("simvtx_y");
    ana.tx->createBranch<vector<float>>("simvtx_z");

    ana.tx->createBranch<vector<vector<int>>>("sim_tcIdx");

    // Matched to track candidate
    ana.tx->createBranch<vector<int>>("sim_TC_matched");
    ana.tx->createBranch<vector<vector<int>>>("sim_TC_types");

    // Track candidates
    ana.tx->createBranch<vector<float>>("tc_pt");
    ana.tx->createBranch<vector<float>>("tc_eta");
    ana.tx->createBranch<vector<float>>("tc_phi");
    ana.tx->createBranch<vector<int>>("tc_isFake");
    ana.tx->createBranch<vector<int>>("tc_isDuplicate");

    createOccupancyBranches();
   
    if (ana.do_lower_level)
    {
        createLowerLevelOutputBranches();
    }
}

//________________________________________________________________________________________________________________________________

void createOccupancyBranches()
{
    //one entry per lower module
    ana.tx->createBranch<vector<int>>("module_layers");
    ana.tx->createBranch<vector<int>>("module_subdets");
    ana.tx->createBranch<vector<int>>("module_rings");
    ana.tx->createBranch<vector<int>>("md_occupancies");
    ana.tx->createBranch<vector<int>>("sg_occupancies");
    ana.tx->createBranch<vector<int>>("t4_occupancies");
    ana.tx->createBranch<vector<int>>("t3_occupancies");
    ana.tx->createBranch<vector<int>>("tc_occupancies");
    ana.tx->createBranch<vector<int>>("t5_occupancies");
    ana.tx->createBranch<int>("pT3_occupancies");
}

//________________________________________________________________________________________________________________________________
void createLowerLevelOutputBranches()
{
    // Matched to quadruplet
    ana.tx->createBranch<vector<int>>("sim_T4_matched");
    ana.tx->createBranch<vector<vector<int>>>("sim_T4_types");

    // Matched to pixel quadruplet
    ana.tx->createBranch<vector<int>>("sim_pT4_matched");
    ana.tx->createBranch<vector<vector<int>>>("sim_pT4_types");


    // T4s
    ana.tx->createBranch<vector<float>>("t4_pt");
    ana.tx->createBranch<vector<float>>("t4_eta");
    ana.tx->createBranch<vector<float>>("t4_phi");
    ana.tx->createBranch<vector<int>>("t4_isFake");
    ana.tx->createBranch<vector<int>>("t4_isDuplicate");

    // pT4s
    ana.tx->createBranch<vector<float>>("pT4_pt");
    ana.tx->createBranch<vector<float>>("pT4_eta");
    ana.tx->createBranch<vector<float>>("pT4_phi");
    ana.tx->createBranch<vector<int>>("pT4_isFake");
    ana.tx->createBranch<vector<int>>("pT4_isDuplicate");

    // Matched to triplet
    ana.tx->createBranch<vector<int>>("sim_T3_matched");
    ana.tx->createBranch<vector<vector<int>>>("sim_T3_types");

    // T3
    ana.tx->createBranch<vector<float>>("t3_pt");
    ana.tx->createBranch<vector<float>>("t3_eta");
    ana.tx->createBranch<vector<float>>("t3_phi");
    ana.tx->createBranch<vector<int>>("t3_isFake");
    ana.tx->createBranch<vector<int>>("t3_isDuplicate");

#ifdef DO_QUINTUPLET
    //T5 - new kid
    ana.tx->createBranch<vector<int>>("sim_T5_matched");
    ana.tx->createBranch<vector<vector<int>>>("sim_T5_types");
    ana.tx->createBranch<vector<int>>("t5_isFake");
    ana.tx->createBranch<vector<int>>("t5_isDuplicate");
    ana.tx->createBranch<vector<float>>("t5_pt");
    ana.tx->createBranch<vector<float>>("t5_eta");
    ana.tx->createBranch<vector<float>>("t5_phi");
#endif
    //pLS
    ana.tx->createBranch<vector<int>>("sim_pLS_matched");
    ana.tx->createBranch<vector<vector<int>>>("sim_pLS_types");
    ana.tx->createBranch<vector<int>>("pLS_isFake");
    ana.tx->createBranch<vector<int>>("pLS_isDuplicate");
    ana.tx->createBranch<vector<float>>("pLS_pt");
    ana.tx->createBranch<vector<float>>("pLS_eta");
    ana.tx->createBranch<vector<float>>("pLS_phi");


    ana.tx->createBranch<vector<int>>("sim_pT3_matched");
    ana.tx->createBranch<vector<vector<int>>>("sim_pT3_types");
//    ana.tx->createBranch<vector<float>>("pT3_pt");
//    ana.tx->createBranch<vector<float>>("pT3_eta");
//    ana.tx->createBranch<vector<float>>("pT3_phi");
    ana.tx->createBranch<vector<int>>("pT3_isFake");
    ana.tx->createBranch<vector<int>>("pT3_isDuplicate");


#ifdef CUT_VALUE_DEBUG
    createQuadrupletCutValueBranches();
    createTripletCutValueBranches();
    createSegmentCutValueBranches();
    createMiniDoubletCutValueBranches();
    createPixelQuadrupletCutValueBranches();
#ifdef DO_QUINTUPLET
    createQuintupletCutValueBranches();
#endif
#endif
}

#ifdef DO_QUINTUPLET
void createQuintupletCutValueBranches()
{
    ana.tx->createBranch<vector<int>>("t5_layer_binary");
    ana.tx->createBranch<vector<vector<float>>>("t5_matched_pt");
    ana.tx->createBranch<vector<float>>("t5_innerRadius");
    ana.tx->createBranch<vector<float>>("t5_innerRadiusMin");
    ana.tx->createBranch<vector<float>>("t5_innerRadiusMax");
    ana.tx->createBranch<vector<float>>("t5_outerRadius");
    ana.tx->createBranch<vector<float>>("t5_outerRadiusMin");
    ana.tx->createBranch<vector<float>>("t5_outerRadiusMax");
    ana.tx->createBranch<vector<float>>("t5_bridgeRadius");
    ana.tx->createBranch<vector<float>>("t5_bridgeRadiusMin");
    ana.tx->createBranch<vector<float>>("t5_bridgeRadiusMax");
    ana.tx->createBranch<vector<float>>("t5_innerRadiusMin2S");
    ana.tx->createBranch<vector<float>>("t5_innerRadiusMax2S");
    ana.tx->createBranch<vector<float>>("t5_bridgeRadiusMin2S");
    ana.tx->createBranch<vector<float>>("t5_bridgeRadiusMax2S");
    ana.tx->createBranch<vector<float>>("t5_outerRadiusMin2S");
    ana.tx->createBranch<vector<float>>("t5_outerRadiusMax2S");

}
#endif
void createQuadrupletCutValueBranches()
{
    ana.tx->createBranch<vector<float>>("t4_zOut");
    ana.tx->createBranch<vector<float>>("t4_rtOut");
    ana.tx->createBranch<vector<float>>("t4_deltaPhiPos");
    ana.tx->createBranch<vector<float>>("t4_deltaPhi");
    ana.tx->createBranch<vector<float>>("t4_betaIn");
    ana.tx->createBranch<vector<float>>("t4_betaOut");
    ana.tx->createBranch<vector<float>>("t4_deltaBeta");

    ana.tx->createBranch<vector<float>>("t4_zLo");
    ana.tx->createBranch<vector<float>>("t4_zHi");
    ana.tx->createBranch<vector<float>>("t4_rtLo");
    ana.tx->createBranch<vector<float>>("t4_rtHi");
    ana.tx->createBranch<vector<float>>("t4_kZ");
    ana.tx->createBranch<vector<float>>("t4_zLoPointed");
    ana.tx->createBranch<vector<float>>("t4_zHiPointed");
    ana.tx->createBranch<vector<float>>("t4_sdlCut");
    ana.tx->createBranch<vector<float>>("t4_betaInCut");
    ana.tx->createBranch<vector<float>>("t4_betaOutCut");
    ana.tx->createBranch<vector<float>>("t4_deltaBetaCut");
    ana.tx->createBranch<vector<int>>("t4_layer_binary");
    ana.tx->createBranch<vector<int>>("t4_moduleType_binary");

}

void createPixelQuadrupletCutValueBranches()
{
    ana.tx->createBranch<vector<float>>("pT4_zOut");
    ana.tx->createBranch<vector<float>>("pT4_rtOut");
    ana.tx->createBranch<vector<float>>("pT4_deltaPhiPos");
    ana.tx->createBranch<vector<float>>("pT4_deltaPhi");
    ana.tx->createBranch<vector<float>>("pT4_betaIn");
    ana.tx->createBranch<vector<float>>("pT4_betaOut");
    ana.tx->createBranch<vector<float>>("pT4_deltaBeta");

    ana.tx->createBranch<vector<float>>("pT4_zLo");
    ana.tx->createBranch<vector<float>>("pT4_zHi");
    ana.tx->createBranch<vector<float>>("pT4_rtLo");
    ana.tx->createBranch<vector<float>>("pT4_rtHi");
    ana.tx->createBranch<vector<float>>("pT4_kZ");
    ana.tx->createBranch<vector<float>>("pT4_zLoPointed");
    ana.tx->createBranch<vector<float>>("pT4_zHiPointed");
    ana.tx->createBranch<vector<float>>("pT4_sdlCut");
    ana.tx->createBranch<vector<float>>("pT4_betaInCut");
    ana.tx->createBranch<vector<float>>("pT4_betaOutCut");
    ana.tx->createBranch<vector<float>>("pT4_deltaBetaCut");
    ana.tx->createBranch<vector<int>>("pT4_layer_binary");
}

void createTripletCutValueBranches()
{
    ana.tx->createBranch<vector<float>>("t3_zOut");
    ana.tx->createBranch<vector<float>>("t3_rtOut");
    ana.tx->createBranch<vector<float>>("t3_deltaPhiPos");
    ana.tx->createBranch<vector<float>>("t3_deltaPhi");
    ana.tx->createBranch<vector<float>>("t3_betaIn");
    ana.tx->createBranch<vector<float>>("t3_betaOut");
    ana.tx->createBranch<vector<float>>("t3_deltaBeta");

    ana.tx->createBranch<vector<float>>("t3_zLo");
    ana.tx->createBranch<vector<float>>("t3_zHi");
    ana.tx->createBranch<vector<float>>("t3_rtLo");
    ana.tx->createBranch<vector<float>>("t3_rtHi");
    ana.tx->createBranch<vector<float>>("t3_kZ");
    ana.tx->createBranch<vector<float>>("t3_zLoPointed");
    ana.tx->createBranch<vector<float>>("t3_zHiPointed");
    ana.tx->createBranch<vector<float>>("t3_sdlCut");
    ana.tx->createBranch<vector<float>>("t3_betaInCut");
    ana.tx->createBranch<vector<float>>("t3_betaOutCut");
    ana.tx->createBranch<vector<float>>("t3_deltaBetaCut");
    ana.tx->createBranch<vector<int>>("t3_layer_binary");
    ana.tx->createBranch<vector<int>>("t3_moduleType_binary");

}

void createSegmentCutValueBranches()
{

}

void createMiniDoubletCutValueBranches()
{
    
}

//________________________________________________________________________________________________________________________________
void fillOutputBranches(SDL::Event& event)
{
    fillSimTrackOutputBranches();
    fillTrackCandidateOutputBranches(event);
    fillOccupancyBranches(event);

    if (ana.do_lower_level)
    {
        fillLowerLevelOutputBranches(event);
    }

    ana.tx->fill();
    ana.tx->clear();

}


//________________________________________________________________________________________________________________________________
void fillOccupancyBranches(SDL::Event& event)
{
    SDL::trackCandidates& trackCandidatesInGPU = (*event.getTrackCandidates());
    SDL::tracklets& trackletsInGPU = (*event.getTracklets());
    SDL::triplets& tripletsInGPU = (*event.getTriplets());
    SDL::segments& segmentsInGPU = (*event.getSegments());
    SDL::miniDoublets& mdsInGPU = (*event.getMiniDoublets());
    SDL::hits& hitsInGPU = (*event.getHits());
    SDL::modules& modulesInGPU = (*event.getModules());
#ifdef DO_QUINTUPLET
    SDL::quintuplets&  quintupletsInGPU = (*event.getQuintuplets());
#endif
    SDL::pixelTriplets& pixelTripletsInGPU = (*event.getPixelTriplets());
    //get the occupancies from these dudes
    std::vector<int> moduleLayer;
    std::vector<int> moduleSubdet;
    std::vector<int> moduleRing;
    std::vector<int> trackCandidateOccupancy;
    std::vector<int> trackletOccupancy;
    std::vector<int> tripletOccupancy;
    std::vector<int> segmentOccupancy;
    std::vector<int> mdOccupancy;
    std::vector<int> quintupletOccupancy;

    for(unsigned int idx = 0; idx <= *(modulesInGPU.nLowerModules); idx++)
    {
        //layer = 0, subdet = 0 => pixel module
        //module, md and segment - need some gymnastics
        unsigned int lowerIdx = modulesInGPU.lowerModuleIndices[idx];
        moduleLayer.push_back(modulesInGPU.layers[lowerIdx]);
        moduleSubdet.push_back(modulesInGPU.subdets[lowerIdx]);
        moduleRing.push_back(modulesInGPU.rings[lowerIdx]);
        segmentOccupancy.push_back(segmentsInGPU.nSegments[lowerIdx]);
        mdOccupancy.push_back(mdsInGPU.nMDs[lowerIdx]);

        trackCandidateOccupancy.push_back(trackCandidatesInGPU.nTrackCandidates[idx]);
        trackletOccupancy.push_back(trackletsInGPU.nTracklets[idx]);
        tripletOccupancy.push_back(tripletsInGPU.nTriplets[idx]);
#ifdef DO_QUINTUPLET
        quintupletOccupancy.push_back(quintupletsInGPU.nQuintuplets[idx]);
#endif
    }
    ana.tx->setBranch<vector<int>>("module_layers",moduleLayer);
    ana.tx->setBranch<vector<int>>("module_subdets",moduleSubdet);
    ana.tx->setBranch<vector<int>>("module_rings",moduleRing);
    ana.tx->setBranch<vector<int>>("md_occupancies",mdOccupancy);
    ana.tx->setBranch<vector<int>>("sg_occupancies",segmentOccupancy);
    ana.tx->setBranch<vector<int>>("t4_occupancies",trackletOccupancy);
    ana.tx->setBranch<vector<int>>("t3_occupancies",tripletOccupancy);
    ana.tx->setBranch<vector<int>>("tc_occupancies",trackCandidateOccupancy);
    ana.tx->setBranch<int>("pT3_occupancies", pixelTripletsInGPU.nPixelTriplets);
#ifdef DO_QUINTUPLET
    ana.tx->setBranch<vector<int>>("t5_occupancies", quintupletOccupancy);
#endif
}

//________________________________________________________________________________________________________________________________
void fillSimTrackOutputBranches()
{

    // Sim tracks
    ana.tx->setBranch<vector<float>>("sim_pt", trk.sim_pt());
    ana.tx->setBranch<vector<float>>("sim_eta", trk.sim_eta());
    ana.tx->setBranch<vector<float>>("sim_phi", trk.sim_phi());
    ana.tx->setBranch<vector<float>>("sim_pca_dxy", trk.sim_pca_dxy());
    ana.tx->setBranch<vector<float>>("sim_pca_dz", trk.sim_pca_dz());
    ana.tx->setBranch<vector<int>>("sim_q", trk.sim_q());
    ana.tx->setBranch<vector<int>>("sim_event", trk.sim_event());
    ana.tx->setBranch<vector<int>>("sim_pdgId", trk.sim_pdgId());
    ana.tx->setBranch<vector<int>>("sim_bunchCrossing", trk.sim_bunchCrossing());
    ana.tx->setBranch<vector<int>>("sim_parentVtxIdx", trk.sim_parentVtxIdx());

    // simvtx
    ana.tx->setBranch<vector<float>>("simvtx_x", trk.simvtx_x());
    ana.tx->setBranch<vector<float>>("simvtx_y", trk.simvtx_y());
    ana.tx->setBranch<vector<float>>("simvtx_z", trk.simvtx_z());
}

//________________________________________________________________________________________________________________________________
void fillTrackCandidateOutputBranches(SDL::Event& event)
{

    SDL::trackCandidates& trackCandidatesInGPU = (*event.getTrackCandidates());
    SDL::tracklets& trackletsInGPU = (*event.getTracklets());
    SDL::triplets& tripletsInGPU = (*event.getTriplets());
    SDL::segments& segmentsInGPU = (*event.getSegments());
    SDL::miniDoublets& miniDoubletsInGPU = (*event.getMiniDoublets());
    SDL::hits& hitsInGPU = (*event.getHits());
    SDL::modules& modulesInGPU = (*event.getModules());
#ifdef DO_QUINTUPLET
    SDL::quintuplets& quintupletsInGPU = (*event.getQuintuplets());
#endif

    // Did it match to track candidate?
    std::vector<int> sim_TC_matched(trk.sim_pt().size());
    std::vector<vector<int>> sim_TC_types(trk.sim_pt().size());
    std::vector<int> tc_isFake;
    std::vector<vector<int>> tc_matched_simIdx;
    std::vector<float> tc_pt;
    std::vector<float> tc_eta;
    std::vector<float> tc_phi;
    for (unsigned int idx = 0; idx <= *(modulesInGPU.nLowerModules); idx++) // "<=" because cheating to include pixel track candidate lower module
    {

        if (modulesInGPU.trackCandidateModuleIndices[idx] == -1)
        {
            continue;
        }

        unsigned int nTrackCandidates = trackCandidatesInGPU.nTrackCandidates[idx];

        if (idx == *(modulesInGPU.nLowerModules) and nTrackCandidates > 5000000)
        {
            nTrackCandidates = 5000000;
        }

        if (idx < *(modulesInGPU.nLowerModules) and nTrackCandidates > 50000)
        {
            nTrackCandidates = 50000;
        }

        for (unsigned int jdx = 0; jdx < nTrackCandidates; jdx++)
        {
//#ifdef DO_QUINTUPLET
//            unsigned int quintupletIndex = modulesInGPU.quintupletModuleIndices[idx] + jdx;
//            unsigned int innerTrackletIndex = quintupletsInGPU.tripletIndices[2 * quintupletIndex];
//            unsigned int outerTrackletIndex = quintupletsInGPU.tripletIndices[2 * quintupletIndex + 1];
//            unsigned int innerTrackletInnerSegmentIndex = tripletsInGPU.segmentIndices[2 * innerTrackletIndex];
//            unsigned int innerTrackletOuterSegmentIndex = tripletsInGPU.segmentIndices[2 * innerTrackletIndex + 1];
//            unsigned int outerTrackletInnerSegmentIndex = tripletsInGPU.segmentIndices[2 * outerTrackletIndex];
//            unsigned int outerTrackletOuterSegmentIndex = tripletsInGPU.segmentIndices[2 * outerTrackletIndex + 1];
//// come back to the beta
//            float betaIn_in = 0;
//            float betaOut_in = 0;
//            float betaIn_out = 0;
//            float betaOut_out = 0;
//                betaIn_in = tripletsInGPU.betaIn[innerTrackletIndex];
//                betaOut_in = tripletsInGPU.betaOut[innerTrackletIndex];
//                betaIn_out = tripletsInGPU.betaIn[outerTrackletIndex];
//                betaOut_out = tripletsInGPU.betaOut[outerTrackletIndex];
//
//#else
            unsigned int trackCandidateIndex = modulesInGPU.trackCandidateModuleIndices[idx] + jdx; // this line causes the issue
            short trackCandidateType = trackCandidatesInGPU.trackCandidateType[trackCandidateIndex];
            unsigned int innerTrackletIdx = trackCandidatesInGPU.objectIndices[2 * trackCandidateIndex];
            unsigned int outerTrackletIdx = trackCandidatesInGPU.objectIndices[2 * trackCandidateIndex + 1];

            unsigned int innerTrackletInnerSegmentIndex = -1;
            unsigned int innerTrackletOuterSegmentIndex = -1;
            unsigned int outerTrackletOuterSegmentIndex = -1;

            float betaIn_in = 0;
            float betaOut_in = 0;
            float betaIn_out = 0;
            float betaOut_out = 0;

            if (trackCandidateType == 0) // T4T4
            {
                innerTrackletInnerSegmentIndex = trackletsInGPU.segmentIndices[2 * innerTrackletIdx];
                innerTrackletOuterSegmentIndex = trackletsInGPU.segmentIndices[2 * innerTrackletIdx + 1];
                outerTrackletOuterSegmentIndex = trackletsInGPU.segmentIndices[2 * outerTrackletIdx + 1];
                betaIn_in = trackletsInGPU.betaIn[innerTrackletIdx];
                betaOut_in = trackletsInGPU.betaOut[innerTrackletIdx];
                betaIn_out = trackletsInGPU.betaIn[outerTrackletIdx];
                betaOut_out = trackletsInGPU.betaOut[outerTrackletIdx];
            }
            else if (trackCandidateType == 1) // T4T3
            {

                innerTrackletInnerSegmentIndex = trackletsInGPU.segmentIndices[2 * innerTrackletIdx];
                innerTrackletOuterSegmentIndex = trackletsInGPU.segmentIndices[2 * innerTrackletIdx + 1];
                outerTrackletOuterSegmentIndex = tripletsInGPU.segmentIndices[2 * outerTrackletIdx + 1];
                betaIn_in = trackletsInGPU.betaIn[innerTrackletIdx];
                betaOut_in = trackletsInGPU.betaOut[innerTrackletIdx];
                betaIn_out = tripletsInGPU.betaIn[outerTrackletIdx];
                betaOut_out = tripletsInGPU.betaOut[outerTrackletIdx];
            }
            else if (trackCandidateType == 2) // T3T4
            {
                innerTrackletInnerSegmentIndex = tripletsInGPU.segmentIndices[2 * innerTrackletIdx];
                innerTrackletOuterSegmentIndex = tripletsInGPU.segmentIndices[2 * innerTrackletIdx + 1];
                outerTrackletOuterSegmentIndex = trackletsInGPU.segmentIndices[2 * outerTrackletIdx + 1];
                betaIn_in = tripletsInGPU.betaIn[innerTrackletIdx];
                betaOut_in = tripletsInGPU.betaOut[innerTrackletIdx];
                betaIn_out = trackletsInGPU.betaIn[outerTrackletIdx];
                betaOut_out = trackletsInGPU.betaOut[outerTrackletIdx];
            }
            if (trackCandidateType == 3) // pT2
            {
                innerTrackletInnerSegmentIndex = trackletsInGPU.segmentIndices[2 * innerTrackletIdx];
                innerTrackletOuterSegmentIndex = trackletsInGPU.segmentIndices[2 * innerTrackletIdx + 1];
                outerTrackletOuterSegmentIndex = trackletsInGPU.segmentIndices[2 * outerTrackletIdx + 1];
                betaIn_in = trackletsInGPU.betaIn[innerTrackletIdx];
                betaOut_in = trackletsInGPU.betaOut[innerTrackletIdx];
                betaIn_out = trackletsInGPU.betaIn[outerTrackletIdx];
                betaOut_out = trackletsInGPU.betaOut[outerTrackletIdx];
            }
#ifdef DO_QUINTUPLET
            if (trackCandidateType == 4) // T5
            {
            unsigned int innerTrackletIndex = quintupletsInGPU.tripletIndices[2 * innerTrackletIdx];
            unsigned int outerTrackletIndex = quintupletsInGPU.tripletIndices[2 * innerTrackletIdx + 1];
             innerTrackletInnerSegmentIndex = tripletsInGPU.segmentIndices[2 * innerTrackletIndex];
             innerTrackletOuterSegmentIndex = tripletsInGPU.segmentIndices[2 * innerTrackletIndex + 1];
             outerTrackletOuterSegmentIndex = tripletsInGPU.segmentIndices[2 * outerTrackletIndex + 1];
                betaIn_in = tripletsInGPU.betaIn[innerTrackletIndex];
                betaOut_in = tripletsInGPU.betaOut[innerTrackletIndex];
                betaIn_out = tripletsInGPU.betaIn[outerTrackletIndex];
                betaOut_out = tripletsInGPU.betaOut[outerTrackletIndex];
            }
#endif
            unsigned int innerTrackletInnerSegmentInnerMiniDoubletIndex = segmentsInGPU.mdIndices[2 * innerTrackletInnerSegmentIndex];
            unsigned int innerTrackletInnerSegmentOuterMiniDoubletIndex = segmentsInGPU.mdIndices[2 * innerTrackletInnerSegmentIndex + 1];
            unsigned int innerTrackletOuterSegmentInnerMiniDoubletIndex = segmentsInGPU.mdIndices[2 * innerTrackletOuterSegmentIndex];
            unsigned int innerTrackletOuterSegmentOuterMiniDoubletIndex = segmentsInGPU.mdIndices[2 * innerTrackletOuterSegmentIndex + 1];
            unsigned int outerTrackletOuterSegmentInnerMiniDoubletIndex = segmentsInGPU.mdIndices[2 * outerTrackletOuterSegmentIndex];
            unsigned int outerTrackletOuterSegmentOuterMiniDoubletIndex = segmentsInGPU.mdIndices[2 * outerTrackletOuterSegmentIndex + 1];
            unsigned int innerTrackletInnerSegmentInnerMiniDoubletLowerHitIndex = miniDoubletsInGPU.hitIndices[2 * innerTrackletInnerSegmentInnerMiniDoubletIndex];
            unsigned int innerTrackletInnerSegmentInnerMiniDoubletUpperHitIndex = miniDoubletsInGPU.hitIndices[2 * innerTrackletInnerSegmentInnerMiniDoubletIndex + 1];
            unsigned int innerTrackletInnerSegmentOuterMiniDoubletLowerHitIndex = miniDoubletsInGPU.hitIndices[2 * innerTrackletInnerSegmentOuterMiniDoubletIndex];
            unsigned int innerTrackletInnerSegmentOuterMiniDoubletUpperHitIndex = miniDoubletsInGPU.hitIndices[2 * innerTrackletInnerSegmentOuterMiniDoubletIndex + 1];
            unsigned int innerTrackletOuterSegmentInnerMiniDoubletLowerHitIndex = miniDoubletsInGPU.hitIndices[2 * innerTrackletOuterSegmentInnerMiniDoubletIndex];
            unsigned int innerTrackletOuterSegmentInnerMiniDoubletUpperHitIndex = miniDoubletsInGPU.hitIndices[2 * innerTrackletOuterSegmentInnerMiniDoubletIndex + 1];
            unsigned int innerTrackletOuterSegmentOuterMiniDoubletLowerHitIndex = miniDoubletsInGPU.hitIndices[2 * innerTrackletOuterSegmentOuterMiniDoubletIndex];
            unsigned int innerTrackletOuterSegmentOuterMiniDoubletUpperHitIndex = miniDoubletsInGPU.hitIndices[2 * innerTrackletOuterSegmentOuterMiniDoubletIndex + 1];
            unsigned int outerTrackletOuterSegmentInnerMiniDoubletLowerHitIndex = miniDoubletsInGPU.hitIndices[2 * outerTrackletOuterSegmentInnerMiniDoubletIndex];
            unsigned int outerTrackletOuterSegmentInnerMiniDoubletUpperHitIndex = miniDoubletsInGPU.hitIndices[2 * outerTrackletOuterSegmentInnerMiniDoubletIndex + 1];
            unsigned int outerTrackletOuterSegmentOuterMiniDoubletLowerHitIndex = miniDoubletsInGPU.hitIndices[2 * outerTrackletOuterSegmentOuterMiniDoubletIndex];
            unsigned int outerTrackletOuterSegmentOuterMiniDoubletUpperHitIndex = miniDoubletsInGPU.hitIndices[2 * outerTrackletOuterSegmentOuterMiniDoubletIndex + 1];

            std::vector<int> hit_idx = {
                (int) hitsInGPU.idxs[innerTrackletInnerSegmentInnerMiniDoubletLowerHitIndex],
                (int) hitsInGPU.idxs[innerTrackletInnerSegmentInnerMiniDoubletUpperHitIndex],
                (int) hitsInGPU.idxs[innerTrackletInnerSegmentOuterMiniDoubletLowerHitIndex],
                (int) hitsInGPU.idxs[innerTrackletInnerSegmentOuterMiniDoubletUpperHitIndex],
                (int) hitsInGPU.idxs[innerTrackletOuterSegmentInnerMiniDoubletLowerHitIndex],
                (int) hitsInGPU.idxs[innerTrackletOuterSegmentInnerMiniDoubletUpperHitIndex],
                (int) hitsInGPU.idxs[innerTrackletOuterSegmentOuterMiniDoubletLowerHitIndex],
                (int) hitsInGPU.idxs[innerTrackletOuterSegmentOuterMiniDoubletUpperHitIndex],
                (int) hitsInGPU.idxs[outerTrackletOuterSegmentInnerMiniDoubletLowerHitIndex],
                (int) hitsInGPU.idxs[outerTrackletOuterSegmentInnerMiniDoubletUpperHitIndex],
                (int) hitsInGPU.idxs[outerTrackletOuterSegmentOuterMiniDoubletLowerHitIndex],
                (int) hitsInGPU.idxs[outerTrackletOuterSegmentOuterMiniDoubletUpperHitIndex],
            };
            unsigned int iiia_idx = -1;
            unsigned int iooa_idx = -1;
            unsigned int oiia_idx = -1;
            unsigned int oooa_idx = -1;

            if (idx == *(modulesInGPU.nLowerModules))
            {
                iiia_idx = segmentsInGPU.outerMiniDoubletAnchorHitIndices[innerTrackletInnerSegmentIndex]; // for pLS the innerSegment outerMiniDoublet
                iooa_idx = segmentsInGPU.outerMiniDoubletAnchorHitIndices[innerTrackletOuterSegmentIndex];
                oiia_idx = segmentsInGPU.innerMiniDoubletAnchorHitIndices[innerTrackletOuterSegmentIndex];
                oooa_idx = segmentsInGPU.outerMiniDoubletAnchorHitIndices[outerTrackletOuterSegmentIndex];
            }
            else
            {
                iiia_idx = segmentsInGPU.innerMiniDoubletAnchorHitIndices[innerTrackletInnerSegmentIndex];
                iooa_idx = segmentsInGPU.outerMiniDoubletAnchorHitIndices[innerTrackletOuterSegmentIndex];
                oiia_idx = segmentsInGPU.innerMiniDoubletAnchorHitIndices[innerTrackletOuterSegmentIndex];
                oooa_idx = segmentsInGPU.outerMiniDoubletAnchorHitIndices[outerTrackletOuterSegmentIndex];
            }

            const float dr_in = sqrt(pow(hitsInGPU.xs[iiia_idx] - hitsInGPU.xs[iooa_idx], 2) + pow(hitsInGPU.ys[iiia_idx] - hitsInGPU.ys[iooa_idx], 2));
            const float dr_out = sqrt(pow(hitsInGPU.xs[oiia_idx] - hitsInGPU.xs[oooa_idx], 2) + pow(hitsInGPU.ys[oiia_idx] - hitsInGPU.ys[oooa_idx], 2));
            const float kRinv1GeVf = (2.99792458e-3 * 3.8);
            const float k2Rinv1GeVf = kRinv1GeVf / 2.;
            const float ptAv_in = (idx == *(modulesInGPU.nLowerModules)) ? segmentsInGPU.ptIn[innerTrackletInnerSegmentIndex-((*(modulesInGPU.nModules))-1)*600] : dr_in * k2Rinv1GeVf / sin((betaIn_in + betaOut_in) / 2.);
            const float ptAv_out = dr_out * k2Rinv1GeVf / sin((betaIn_out + betaOut_out) / 2.);
            const float ptAv = (ptAv_in + ptAv_out) / 2.;

            std::vector<int> hit_types;
            if (idx == *(modulesInGPU.nLowerModules)) // Then this means this track candidate is a pLS-based
            {
                hit_types.push_back(0);
                hit_types.push_back(0);
                hit_types.push_back(0);
                hit_types.push_back(0);
            }
            else
            {
                hit_types.push_back(4);
                hit_types.push_back(4);
                hit_types.push_back(4);
                hit_types.push_back(4);
            }

            hit_types.push_back(4);
            hit_types.push_back(4);
            hit_types.push_back(4);
            hit_types.push_back(4);
            hit_types.push_back(4);
            hit_types.push_back(4);
            hit_types.push_back(4);
            hit_types.push_back(4);

            std::vector<int> module_idxs = {
                (int) hitsInGPU.moduleIndices[innerTrackletInnerSegmentInnerMiniDoubletLowerHitIndex],
                (int) hitsInGPU.moduleIndices[innerTrackletInnerSegmentInnerMiniDoubletUpperHitIndex],
                (int) hitsInGPU.moduleIndices[innerTrackletInnerSegmentOuterMiniDoubletLowerHitIndex],
                (int) hitsInGPU.moduleIndices[innerTrackletInnerSegmentOuterMiniDoubletUpperHitIndex],
                (int) hitsInGPU.moduleIndices[innerTrackletOuterSegmentInnerMiniDoubletLowerHitIndex],
                (int) hitsInGPU.moduleIndices[innerTrackletOuterSegmentInnerMiniDoubletUpperHitIndex],
                (int) hitsInGPU.moduleIndices[innerTrackletOuterSegmentOuterMiniDoubletLowerHitIndex],
                (int) hitsInGPU.moduleIndices[innerTrackletOuterSegmentOuterMiniDoubletUpperHitIndex],
                (int) hitsInGPU.moduleIndices[outerTrackletOuterSegmentInnerMiniDoubletLowerHitIndex],
                (int) hitsInGPU.moduleIndices[outerTrackletOuterSegmentInnerMiniDoubletUpperHitIndex],
                (int) hitsInGPU.moduleIndices[outerTrackletOuterSegmentOuterMiniDoubletLowerHitIndex],
                (int) hitsInGPU.moduleIndices[outerTrackletOuterSegmentOuterMiniDoubletUpperHitIndex],
            };
            bool isPixel0 = (idx == *(modulesInGPU.nLowerModules));
            // bool isPixel1 = (idx == *(modulesInGPU.nLowerModules));
            bool isPixel2 = (idx == *(modulesInGPU.nLowerModules));
            // bool isPixel3 = (idx == *(modulesInGPU.nLowerModules));
            bool isPixel4 = false;
            // bool isPixel5 = false;
            bool isPixel6 = false;
            // bool isPixel7 = false;
            bool isPixel8 = false;
            // bool isPixel9 = false;
            bool isPixel10 = false;
            // bool isPixel11 = false;

            int layer0 = modulesInGPU.layers[module_idxs[0]];
            // int layer1 = modulesInGPU.layers[module_idxs[1]];
            int layer2 = modulesInGPU.layers[module_idxs[2]];
            // int layer3 = modulesInGPU.layers[module_idxs[3]];
            int layer4 = modulesInGPU.layers[module_idxs[4]];
            // int layer5 = modulesInGPU.layers[module_idxs[5]];
            int layer6 = modulesInGPU.layers[module_idxs[6]];
            // int layer7 = modulesInGPU.layers[module_idxs[7]];
            int layer8 = modulesInGPU.layers[module_idxs[8]];
            // int layer9 = modulesInGPU.layers[module_idxs[9]];
            int layer10 = modulesInGPU.layers[module_idxs[10]];
            // int layer11 = modulesInGPU.layers[module_idxs[11]];

            int subdet0 = modulesInGPU.subdets[module_idxs[0]];
            // int subdet1 = modulesInGPU.subdets[module_idxs[1]];
            int subdet2 = modulesInGPU.subdets[module_idxs[2]];
            // int subdet3 = modulesInGPU.subdets[module_idxs[3]];
            int subdet4 = modulesInGPU.subdets[module_idxs[4]];
            // int subdet5 = modulesInGPU.subdets[module_idxs[5]];
            int subdet6 = modulesInGPU.subdets[module_idxs[6]];
            // int subdet7 = modulesInGPU.subdets[module_idxs[7]];
            int subdet8 = modulesInGPU.subdets[module_idxs[8]];
            // int subdet9 = modulesInGPU.subdets[module_idxs[9]];
            int subdet10 = modulesInGPU.subdets[module_idxs[10]];
            // int subdet11 = modulesInGPU.subdets[module_idxs[11]];

            int logicallayer0 = isPixel0 ? 0 : layer0 + 6 * (subdet0 == 4);
            // int logicallayer1 = isPixel1 ? 0 : layer1 + 6 * (subdet1 == 4);
            int logicallayer2 = isPixel2 ? 0 : layer2 + 6 * (subdet2 == 4);
            // int logicallayer3 = isPixel3 ? 0 : layer3 + 6 * (subdet3 == 4);
            int logicallayer4 = isPixel4 ? 0 : layer4 + 6 * (subdet4 == 4);
            // int logicallayer5 = isPixel5 ? 0 : layer5 + 6 * (subdet5 == 4);
            int logicallayer6 = isPixel6 ? 0 : layer6 + 6 * (subdet6 == 4);
            // int logicallayer7 = isPixel7 ? 0 : layer7 + 6 * (subdet7 == 4);
            int logicallayer8 = isPixel8 ? 0 : layer8 + 6 * (subdet8 == 4);
            // int logicallayer9 = isPixel9 ? 0 : layer9 + 6 * (subdet9 == 4);
            int logicallayer10 = isPixel10 ? 0 : layer10 + 6 * (subdet10 == 4);
            // int logicallayer11 = isPixel11 ? 0 : layer11 + 6 * (subdet11 == 4);

            int layer_binary = 0;
            layer_binary |= (1 << logicallayer0);
            layer_binary |= (1 << logicallayer2);
            layer_binary |= (1 << logicallayer4);
            layer_binary |= (1 << logicallayer6);
            layer_binary |= (1 << logicallayer8);
            layer_binary |= (1 << logicallayer10);

            // sim track matched index
            std::vector<int> matched_sim_trk_idxs = matchedSimTrkIdxs(hit_idx, hit_types);

            for (auto &isimtrk : matched_sim_trk_idxs)
            {
                sim_TC_matched[isimtrk]++;
            }

            for (auto &isimtrk : matched_sim_trk_idxs)
            {
                sim_TC_types[isimtrk].push_back(layer_binary);
            }

            // Compute pt, eta, phi of TC
            const float pt = ptAv;
            float eta = -999;
            float phi = -999;
            if (hit_types[0] == 4)
            {
                SDL::CPU::Hit hitA(trk.ph2_x()[hit_idx[0]], trk.ph2_y()[hit_idx[0]], trk.ph2_z()[hit_idx[0]]);
                SDL::CPU::Hit hitB(trk.ph2_x()[hit_idx[11]], trk.ph2_y()[hit_idx[11]], trk.ph2_z()[hit_idx[11]]);
                eta = hitB.eta();
                phi = hitA.phi();
            }
            else
            {
                SDL::CPU::Hit hitA(trk.pix_x()[hit_idx[0]], trk.pix_y()[hit_idx[0]], trk.pix_z()[hit_idx[0]]);
                SDL::CPU::Hit hitB(trk.ph2_x()[hit_idx[11]], trk.ph2_y()[hit_idx[11]], trk.ph2_z()[hit_idx[11]]);
                eta = hitB.eta();
                phi = hitA.phi();
            }

            tc_isFake.push_back(matched_sim_trk_idxs.size() == 0);
            tc_pt.push_back(pt);
            tc_eta.push_back(eta);
            tc_phi.push_back(phi);
            tc_matched_simIdx.push_back(matched_sim_trk_idxs);

      }
    }

    ana.tx->setBranch<vector<int>>("sim_TC_matched", sim_TC_matched);
    ana.tx->setBranch<vector<vector<int>>>("sim_TC_types", sim_TC_types);

    vector<int> tc_isDuplicate(tc_matched_simIdx.size());

    for (unsigned int i = 0; i < tc_matched_simIdx.size(); ++i)
    {
        bool isDuplicate = false;
        for (unsigned int isim = 0; isim < tc_matched_simIdx[i].size(); ++isim)
        {
            if (sim_TC_matched[tc_matched_simIdx[i][isim]] > 1)
            {
                isDuplicate = true;
            }
        }
        tc_isDuplicate[i] = isDuplicate;
    }

    ana.tx->setBranch<vector<float>>("tc_pt", tc_pt);
    ana.tx->setBranch<vector<float>>("tc_eta", tc_eta);
    ana.tx->setBranch<vector<float>>("tc_phi", tc_phi);
    ana.tx->setBranch<vector<int>>("tc_isFake", tc_isFake);
    ana.tx->setBranch<vector<int>>("tc_isDuplicate", tc_isDuplicate);
}

//________________________________________________________________________________________________________________________________
void fillLowerLevelOutputBranches(SDL::Event& event)
{
    fillQuadrupletOutputBranches(event);
    fillPixelQuadrupletOutputBranches(event);
    fillTripletOutputBranches(event);
    fillPixelLineSegmentOutputBranches(event);
#ifdef DO_QUINTUPLET
    fillQuintupletOutputBranches(event);
#endif
    fillPixelTripletOutputBranches(event);
}

#ifdef DO_QUINTUPLET
//________________________________________________________________________________________________________________________________
void fillQuintupletOutputBranches(SDL::Event& event)
{
    SDL::quintuplets& quintupletsInGPU = (*event.getQuintuplets());
    SDL::triplets& tripletsInGPU = (*event.getTriplets());
    SDL::segments& segmentsInGPU = (*event.getSegments());
    SDL::miniDoublets& miniDoubletsInGPU = (*event.getMiniDoublets());
    SDL::hits& hitsInGPU = (*event.getHits());
    SDL::modules& modulesInGPU = (*event.getModules());

    // Did it match to track candidate?
    std::vector<int> sim_T5_matched(trk.sim_pt().size());
    std::vector<vector<int>> sim_T5_types(trk.sim_pt().size());
    std::vector<int> t5_isFake;
    std::vector<vector<int>> t5_matched_simIdx;
    std::vector<float> t5_pt;
    std::vector<float> t5_eta;
    std::vector<float> t5_phi;

#ifdef CUT_VALUE_DEBUG
    std::vector<float> t5_innerRadius;
    std::vector<float> t5_innerRadiusMin;
    std::vector<float> t5_innerRadiusMax;
    std::vector<float> t5_innerRadiusMin2S;
    std::vector<float> t5_innerRadiusMax2S;
    std::vector<float> t5_outerRadius;
    std::vector<float> t5_outerRadiusMin;
    std::vector<float> t5_outerRadiusMax;
    std::vector<float> t5_outerRadiusMin2S;
    std::vector<float> t5_outerRadiusMax2S;
    std::vector<float> t5_bridgeRadius;
    std::vector<float> t5_bridgeRadiusMin;
    std::vector<float> t5_bridgeRadiusMax;
    std::vector<float> t5_bridgeRadiusMin2S;
    std::vector<float> t5_bridgeRadiusMax2S;
    std::vector<std::vector<float>> t5_simpt;
    std::vector<int> layer_binaries;
#endif

    const int MAX_NQUINTUPLET_PER_MODULE = 5000;
    const float kRinv1GeVf = (2.99792458e-3 * 3.8);
    const float k2Rinv1GeVf = kRinv1GeVf / 2.;
    
    for(unsigned int idx = 0; idx < *(modulesInGPU.nLowerModules); idx++)
    {
        if(modulesInGPU.quintupletModuleIndices[idx] == -1)
        {
            continue;
        }

        unsigned int nQuintuplets = quintupletsInGPU.nQuintuplets[idx];
        
        if(nQuintuplets > MAX_NQUINTUPLET_PER_MODULE)
        {
            nQuintuplets = MAX_NQUINTUPLET_PER_MODULE;
        }

        for(unsigned int jdx = 0; jdx < nQuintuplets; jdx++)
        {
            unsigned int quintupletIndex = modulesInGPU.quintupletModuleIndices[idx] + jdx;
            unsigned int innerTripletIndex = quintupletsInGPU.tripletIndices[2 * quintupletIndex];
            unsigned int outerTripletIndex = quintupletsInGPU.tripletIndices[2 * quintupletIndex + 1];



#ifdef CUT_VALUE_DEBUG
            t5_innerRadius.push_back(quintupletsInGPU.innerRadius[quintupletIndex]);
            t5_innerRadiusMin.push_back(quintupletsInGPU.innerRadiusMin[quintupletIndex]);
            t5_innerRadiusMax.push_back(quintupletsInGPU.innerRadiusMax[quintupletIndex]);
            t5_innerRadiusMin2S.push_back(quintupletsInGPU.innerRadiusMin2S[quintupletIndex]);
            t5_innerRadiusMax2S.push_back(quintupletsInGPU.innerRadiusMax2S[quintupletIndex]);

            t5_outerRadius.push_back(quintupletsInGPU.outerRadius[quintupletIndex]);
            t5_outerRadiusMin.push_back(quintupletsInGPU.outerRadiusMin[quintupletIndex]);
            t5_outerRadiusMax.push_back(quintupletsInGPU.outerRadiusMax[quintupletIndex]);
            t5_outerRadiusMin2S.push_back(quintupletsInGPU.outerRadiusMin2S[quintupletIndex]);
            t5_outerRadiusMax2S.push_back(quintupletsInGPU.outerRadiusMax2S[quintupletIndex]);

            t5_bridgeRadius.push_back(quintupletsInGPU.bridgeRadius[quintupletIndex]);
            t5_bridgeRadiusMin.push_back(quintupletsInGPU.bridgeRadiusMin[quintupletIndex]);
            t5_bridgeRadiusMax.push_back(quintupletsInGPU.bridgeRadiusMax[quintupletIndex]);
            t5_bridgeRadiusMin2S.push_back(quintupletsInGPU.bridgeRadiusMin2S[quintupletIndex]);
            t5_bridgeRadiusMax2S.push_back(quintupletsInGPU.bridgeRadiusMax2S[quintupletIndex]);
#endif

            unsigned int innerTripletInnerSegmentIndex = tripletsInGPU.segmentIndices[2 * innerTripletIndex];
            unsigned int innerTripletOuterSegmentIndex = tripletsInGPU.segmentIndices[2 * innerTripletIndex + 1];
            unsigned int outerTripletInnerSegmentIndex = tripletsInGPU.segmentIndices[2 * outerTripletIndex];
            unsigned int outerTripletOuterSegmentIndex = tripletsInGPU.segmentIndices[2 * outerTripletIndex + 1];

            unsigned int innerTripletInnerSegmentInnerMiniDoubletIndex = segmentsInGPU.mdIndices[2 * innerTripletInnerSegmentIndex];

             //same as innerTripletOuterSegmentInnerMiniDoubletIndex
            unsigned int innerTripletInnerSegmentOuterMiniDoubletIndex = segmentsInGPU.mdIndices[2 * innerTripletInnerSegmentIndex + 1];
            
            //same as outerTripletInnerSegmentInnerMiniDoubletIndex
            unsigned int innerTripletOuterSegmentOuterMiniDoubletIndex = segmentsInGPU.mdIndices[2 * innerTripletOuterSegmentIndex + 1];
            //printf("T5 %u %u %u %u %u %u\n",quintupletIndex,innerTripletIndex,outerTripletIndex,innerTripletInnerSegmentIndex,innerTripletOuterSegmentIndex,outerTripletOuterSegmentIndex);

            //same as outerTripletOuterSegmentInnerMiniDoubletIndex
            unsigned int outerTripletInnerSegmentOuterMiniDoubletIndex = segmentsInGPU.mdIndices[2 * outerTripletInnerSegmentIndex + 1];

            unsigned int outerTripletOuterSegmentOuterMiniDoubletIndex = segmentsInGPU.mdIndices[2 * outerTripletOuterSegmentIndex + 1];

            unsigned int innerTripletInnerSegmentInnerMiniDoubletLowerHitIndex = miniDoubletsInGPU.hitIndices[2 * innerTripletInnerSegmentInnerMiniDoubletIndex];
            unsigned int innerTripletInnerSegmentInnerMiniDoubletUpperHitIndex = miniDoubletsInGPU.hitIndices[2 * innerTripletInnerSegmentInnerMiniDoubletIndex + 1];

            unsigned int innerTripletInnerSegmentOuterMiniDoubletLowerHitIndex = miniDoubletsInGPU.hitIndices[2 * innerTripletInnerSegmentOuterMiniDoubletIndex];
            unsigned int innerTripletInnerSegmentOuterMiniDoubletUpperHitIndex = miniDoubletsInGPU.hitIndices[2 * innerTripletInnerSegmentOuterMiniDoubletIndex + 1];

            unsigned int innerTripletOuterSegmentOuterMiniDoubletLowerHitIndex = miniDoubletsInGPU.hitIndices[2 * innerTripletOuterSegmentOuterMiniDoubletIndex];
            unsigned int innerTripletOuterSegmentOuterMiniDoubletUpperHitIndex = miniDoubletsInGPU.hitIndices[2 * innerTripletInnerSegmentOuterMiniDoubletIndex + 1];

            unsigned int outerTripletInnerSegmentOuterMiniDoubletLowerHitIndex = miniDoubletsInGPU.hitIndices[2 * outerTripletInnerSegmentOuterMiniDoubletIndex];
            unsigned int outerTripletInnerSegmentOuterMiniDoubletUpperHitIndex = miniDoubletsInGPU.hitIndices[2 * outerTripletInnerSegmentOuterMiniDoubletIndex + 1];

            unsigned int outerTripletOuterSegmentOuterMiniDoubletLowerHitIndex = miniDoubletsInGPU.hitIndices[2 * outerTripletOuterSegmentOuterMiniDoubletIndex];
            unsigned int outerTripletOuterSegmentOuterMiniDoubletUpperHitIndex = miniDoubletsInGPU.hitIndices[2 * outerTripletOuterSegmentOuterMiniDoubletIndex + 1];

            std::vector<int> hit_idxs = {
                (int) hitsInGPU.idxs[innerTripletInnerSegmentInnerMiniDoubletLowerHitIndex],
                (int) hitsInGPU.idxs[innerTripletInnerSegmentInnerMiniDoubletUpperHitIndex],
                (int) hitsInGPU.idxs[innerTripletInnerSegmentOuterMiniDoubletLowerHitIndex],
                (int) hitsInGPU.idxs[innerTripletInnerSegmentOuterMiniDoubletUpperHitIndex],
                (int) hitsInGPU.idxs[innerTripletOuterSegmentOuterMiniDoubletLowerHitIndex],
                (int) hitsInGPU.idxs[innerTripletOuterSegmentOuterMiniDoubletUpperHitIndex],
                (int) hitsInGPU.idxs[outerTripletInnerSegmentOuterMiniDoubletLowerHitIndex],
                (int) hitsInGPU.idxs[outerTripletInnerSegmentOuterMiniDoubletUpperHitIndex],
                (int) hitsInGPU.idxs[outerTripletOuterSegmentOuterMiniDoubletLowerHitIndex],
                (int) hitsInGPU.idxs[outerTripletOuterSegmentOuterMiniDoubletUpperHitIndex]
            };

            std::vector<int> hit_types(hit_idxs.size(), 4);
            std::vector<int> module_idxs = {
                (int) hitsInGPU.moduleIndices[innerTripletInnerSegmentInnerMiniDoubletLowerHitIndex],
                (int) hitsInGPU.moduleIndices[innerTripletInnerSegmentInnerMiniDoubletUpperHitIndex],
                (int) hitsInGPU.moduleIndices[innerTripletInnerSegmentOuterMiniDoubletLowerHitIndex],
                (int) hitsInGPU.moduleIndices[innerTripletInnerSegmentOuterMiniDoubletUpperHitIndex],
                (int) hitsInGPU.moduleIndices[innerTripletOuterSegmentOuterMiniDoubletLowerHitIndex],
                (int) hitsInGPU.moduleIndices[innerTripletOuterSegmentOuterMiniDoubletUpperHitIndex],
                (int) hitsInGPU.moduleIndices[outerTripletInnerSegmentOuterMiniDoubletLowerHitIndex],
                (int) hitsInGPU.moduleIndices[outerTripletInnerSegmentOuterMiniDoubletUpperHitIndex],
                (int) hitsInGPU.moduleIndices[outerTripletOuterSegmentOuterMiniDoubletLowerHitIndex],
                (int) hitsInGPU.moduleIndices[outerTripletOuterSegmentOuterMiniDoubletUpperHitIndex]
            };

            int layer0 = modulesInGPU.layers[module_idxs[0]];
            int layer2 = modulesInGPU.layers[module_idxs[2]];
            int layer4 = modulesInGPU.layers[module_idxs[4]];
            int layer6 = modulesInGPU.layers[module_idxs[6]];
            int layer8 = modulesInGPU.layers[module_idxs[8]];

            int subdet0 = modulesInGPU.subdets[module_idxs[0]];
            int subdet2 = modulesInGPU.subdets[module_idxs[2]];
            int subdet4 = modulesInGPU.subdets[module_idxs[4]];
            int subdet6 = modulesInGPU.subdets[module_idxs[6]];
            int subdet8 = modulesInGPU.subdets[module_idxs[8]];

            int logicallayer0 = layer0 + 6 * (subdet0 == 4);
            int logicallayer2 = layer2 + 6 * (subdet2 == 4);
            int logicallayer4 = layer4 + 6 * (subdet4 == 4);
            int logicallayer6 = layer6 + 6 * (subdet6 == 4);
            int logicallayer8 = layer8 + 6 * (subdet8 == 4);

            int layer_binary = 0;
            layer_binary |= (1 << logicallayer0);
            layer_binary |= (1 << logicallayer2);
            layer_binary |= (1 << logicallayer4);
            layer_binary |= (1 << logicallayer6);
            layer_binary |= (1 << logicallayer8);

            // radius values now not covered under CUT_VALUE_DEBUG
            // using innerRadius and outerRadius to match up with CPU implementation

            float pt = k2Rinv1GeVf * (quintupletsInGPU.innerRadius[quintupletIndex] + quintupletsInGPU.outerRadius[quintupletIndex]);

            //copyting stuff from before for eta and phi
            SDL::CPU::Hit hitA(trk.ph2_x()[hit_idxs[0]], trk.ph2_y()[hit_idxs[0]], trk.ph2_z()[hit_idxs[0]]);
            SDL::CPU::Hit hitB(trk.ph2_x()[hit_idxs[9]], trk.ph2_y()[hit_idxs[9]], trk.ph2_z()[hit_idxs[9]]);

            float eta = hitB.eta();
            float phi = hitA.phi();

            t5_pt.push_back(pt);
            t5_eta.push_back(eta);
            t5_phi.push_back(phi);

#ifdef CUT_VALUE_DEBUG
            layer_binaries.push_back(layer_binary);
#endif
            std::vector<int> matched_sim_trk_idxs = matchedSimTrkIdxs(hit_idxs, hit_types);
            for (auto &isimtrk : matched_sim_trk_idxs)
            {
                sim_T5_matched[isimtrk]++;
            }
#ifdef CUT_VALUE_DEBUG
            std::vector<float> sim_pt_per_t5;
            if(matched_sim_trk_idxs.size() == 0)
            {
                sim_pt_per_t5.push_back(-999);
            }
            else
            {
		sim_pt_per_t5.push_back(trk.sim_pt()[matched_sim_trk_idxs[0]]);
            }
            t5_simpt.push_back(sim_pt_per_t5);
#endif

            for (auto &isimtrk : matched_sim_trk_idxs)
            {
                sim_T5_types[isimtrk].push_back(layer_binary);
            }
            t5_isFake.push_back(matched_sim_trk_idxs.size() == 0);
            t5_matched_simIdx.push_back(matched_sim_trk_idxs);
        }
    }

    std::vector<int> t5_isDuplicate(t5_matched_simIdx.size());

    for (unsigned int i = 0; i < t5_matched_simIdx.size(); ++i)
    {
        bool isDuplicate = false;
        for (unsigned int isim = 0; isim < t5_matched_simIdx[i].size(); ++isim)
        {
            if (sim_T5_matched[t5_matched_simIdx[i][isim]] > 1)
            {
                isDuplicate = true;
            }
        }
        t5_isDuplicate[i] = isDuplicate;
    }
    ana.tx->setBranch<vector<int>>("sim_T5_matched", sim_T5_matched);
    ana.tx->setBranch<vector<vector<int>>>("sim_T5_types", sim_T5_types);
    ana.tx->setBranch<vector<int>>("t5_isFake", t5_isFake);
    ana.tx->setBranch<vector<int>>("t5_isDuplicate", t5_isDuplicate);
    ana.tx->setBranch<vector<float>>("t5_pt", t5_pt);
    ana.tx->setBranch<vector<float>>("t5_eta", t5_eta);
    ana.tx->setBranch<vector<float>>("t5_phi", t5_phi);
#ifdef CUT_VALUE_DEBUG
    ana.tx->setBranch<vector<vector<float>>>("t5_matched_pt",t5_simpt);

    ana.tx->setBranch<vector<float>>("t5_outerRadius",t5_outerRadius);
    ana.tx->setBranch<vector<float>>("t5_outerRadiusMin",t5_outerRadiusMin);
    ana.tx->setBranch<vector<float>>("t5_outerRadiusMax",t5_outerRadiusMax);
    ana.tx->setBranch<vector<float>>("t5_outerRadiusMin2S",t5_outerRadiusMin2S);
    ana.tx->setBranch<vector<float>>("t5_outerRadiusMax2S",t5_outerRadiusMax2S);

    ana.tx->setBranch<vector<float>>("t5_innerRadius",t5_innerRadius);
    ana.tx->setBranch<vector<float>>("t5_innerRadiusMin",t5_innerRadiusMin);
    ana.tx->setBranch<vector<float>>("t5_innerRadiusMax",t5_innerRadiusMax);
    ana.tx->setBranch<vector<float>>("t5_innerRadiusMin2S",t5_innerRadiusMin2S);
    ana.tx->setBranch<vector<float>>("t5_innerRadiusMax2S",t5_innerRadiusMax2S);
    ana.tx->setBranch<vector<float>>("t5_bridgeRadius",t5_bridgeRadius);
    ana.tx->setBranch<vector<float>>("t5_bridgeRadiusMin",t5_bridgeRadiusMin);
    ana.tx->setBranch<vector<float>>("t5_bridgeRadiusMax",t5_bridgeRadiusMax);
    ana.tx->setBranch<vector<float>>("t5_bridgeRadiusMin2S",t5_bridgeRadiusMin2S);
    ana.tx->setBranch<vector<float>>("t5_bridgeRadiusMax2S",t5_bridgeRadiusMax2S);
    ana.tx->setBranch<vector<int>>("t5_layer_binary",layer_binaries); 
#endif

}
#endif

void fillPixelTripletOutputBranches(SDL::Event& event)
{
    SDL::pixelTriplets& pixelTripletsInGPU = (*event.getPixelTriplets());
    SDL::segments& segmentsInGPU = (*event.getSegments());
    SDL::miniDoublets& mdsInGPU = (*event.getMiniDoublets());
    SDL::hits& hitsInGPU = (*event.getHits());
    SDL::modules& modulesInGPU = (*event.getModules());

    std::vector<int> sim_pT3_matched(trk.sim_pt().size(), 0);
    std::vctor<vector<int>> sim_pT3_types(trk.sim_pt().size());
    std::vector<int> pT3_isFake;
    std::vector<vector<int>> pT3_matched_simIdx;
    //std::vector<float> pT3_pt;
    //std::vector<float> pT3_eta;
    //std::vector<float> pT3_phi;
    const unsigned int N_MAX_PIXEL_TRIPLETS = 3000000;

    unsigned int nPixelTriplets = std::min(pixelTripletsInGPU.nPixelTriplets, N_MAX_PIXEL_TRIPLETS);

    for(unsigned int jdx = 0; jdx < nPixelTriplets; jdx++)
    {
        unsigned int pixelSegmentIndex = pixelTripletsInGPU.pixelSegmentIndices[jdx];
        unsigned int tripletIndex = pixelTripletsInGPU.tripletIndices[jdx];
        
        unsigned int pixelInnerMDIndex = segmentsInGPU.mdIndices[2 * pixelSegmentIndex];
        unsigned int pixelOuterMDIndex = segmentsInGPU.mdIndices[2 * pixelSegmentIndex + 1];
        unsigned int tripletInnerSegmentIndex = tripletsInGPU.segmentIndices[2 * tripletIndex];
        unsigned int tripletOuterSegmentIndex = tripletsInGPU.segmentIndices[2 * tripletIndex + 1];

        unsigned int tripletInnerMDIndex = segmentsInGPU.mdIndices[2 * tripletInnerSegmentIndex];
        unsigned int tripletMiddleMDIndex = segmentsInGPU.mdIndices[2 * tripletInnerSegmentIndex + 1];
        unsigned int tripletOuterMDIndex = segmentsInGPU.mdIndices[2 * tripletOuterSegmentIndex + 1];

        unsigned int pixelInnerMDLowerHitIndex = mdsInGPU.hitIndices[2 * pixelInnerMDIndex];
        unsigned int pixelInnerMDUpperHitIndex = mdsInGPU.hitIndices[2 * pixelInnerMDIndex + 1];
        unsigned int pixelOuterMDLowerHitIndex = mdsInGPU.hitIndices[2 * pixelOuterMDIndex];
        unsigned int pixelOuterMDUpperHitIndex = mdsInGPU.hitIndices[2 * pixelOuterMDIndex + 1];

        unsigned int tripletInnerMDLowerHitIndex = mdsInGPU.hitIndices[2 * tripletInnerMDIndex];
        unsigned int tripletInnerMDUpperHitIndex = mdsInGPU.hitIndices[2 * tripletInnerMDIndex + 1];
        unsigned int tripletMiddleMDLowerHitIndex = mdsInGPU.hitIndices[2 * tripletMiddleMDIndex];
        unsigned int tripletMiddleMDUpperHitIndex = mdsInGPU.hitIndices[2 * tripletMiddleMDIndex + 1];
        unsigned int tripletOuterMDLowerHitIndex = mdsInGPU.hitIndices[2 * tripletOuterMDIndex];
        unsigned int tripletOuterMDUpperHitIndex = mdsInGPU.hitIndices[2 * tripletOuterMDIndex + 1];

        std::vector<int> hit_idxs = {
            (int) hitsInGPU.idxs[pixelInnerMDLowerHitIndex];
            (int) hitsInGPU.idxs[pixelInnerMDUpperHitIndex];
            (int) hitsInGPU.idxs[pixelOuterMDLowerHitIndex];
            (int) hitsInGPU.idxs[pixelOuterMDUpperHitIndex];
            (int) hitsInGPU.idxs[tripletInnerMDLowerHitIndex];
            (int) hitsInGPU.idxs[tripletInnerMDUpperHitIndex];
            (int) hitsInGPU.idxs[tripletMiddleMDLowerHitIndex];
            (int) hitsInGPU.idxs[tripletMiddleMDUpperHitIndex];
            (int) hitsInGPU.idxs[tripletOuterMDLowerHitIndex];
            (int) hitsInGPU.idxs[tripletOuterMDUpperHitIndex];
        };
        std::vector<int> hit_types;
        hit_types.push_back(0);
        hit_types.push_back(0);
        hit_types.push_back(0);
        hit_types.push_back(0);
        hit_types.push_back(4);
        hit_types.push_back(4);
        hit_types.push_back(4);
        hit_types.push_back(4);
        hit_types.push_back(4);
        hit_types.push_back(4);

        std::vector<int> module_idxs = {
            (int) hitsInGPU.moduleIndices[pixelInnerMDLowerHitIndex];
            (int) hitsInGPU.moduleIndices[pixelInnerMDUpperHitIndex];
            (int) hitsInGPU.moduleIndices[pixelOuterMDLowerHitIndex];
            (int) hitsInGPU.moduleIndices[pixelOuterMDUpperHitIndex];
            (int) hitsInGPU.moduleIndices[tripletInnerMDLowerHitIndex];
            (int) hitsInGPU.moduleIndices[tripletInnerMDUpperHitIndex];
            (int) hitsInGPU.moduleIndices[tripletMiddleMDLowerHitIndex];
            (int) hitsInGPU.moduleIndices[tripletMiddleMDUpperHitIndex];
            (int) hitsInGPU.moduleIndices[tripletOuterMDLowerHitIndex];
            (int) hitsInGPU.moduleIndices[tripletOuterMDUpperHitIndex];
        };
        int layer0 = modulesInGPU.layers[module_idxs[0]];
        int layer2 = modulesInGPU.layers[module_idxs[2]];
        int layer4 = modulesInGPU.layers[module_idxs[4]];
        int layer6 = modulesInGPU.layers[module_idxs[6]];
        int layer8 = modulesInGPU.layers[module_idxs[8]];

        int subdet0 = modulesInGPU.subdets[module_idxs[0]];
        int subdet2 = modulesInGPU.subdets[module_idxs[2]];
        int subdet4 = modulesInGPU.subdets[module_idxs[4]];
        int subdet6 = modulesInGPU.subdets[module_idxs[6]];
        int subdet8 = modulesInGPU.subdets[module_idxs[8]];

        int logicallayer0 = 0;
        int logicallayer2 = 0;
        int logicallayer4 = layer4 + 6 * (subdet4 == 4);
        int logicallayer6 = layer6 + 6 * (subdet6 == 4);
        int logicallayer8 = layer8 + 6 * (subdet8 == 4);

        int layer_binary = 0;
        layer_binary |= (1 << logicallayer0);
        layer_binary |= (1 << logicallayer2);
        layer_binary |= (1 << logicallayer4);
        layer_binary |= (1 << logicallayer6);
        layer_binary |= (1 << logicallayer8);

        //bare bones implementation only
        std::vector<int> matched_sim_trk_idxs = matchedSimTrkIdxs(hit_idxs, hit_types);
        for (auto &isimtrk : matched_sim_trk_idxs)
        {
            sim_pT3_matched[isimtrk]++;
        }

        for (auto &isimtrk : matched_sim_trk_idxs)
        {
            sim_pT3_types[isimtrk].push_back(layer_binary);
        }
        pT3_isFake.push_back(matched_sim_trk_idxs.size() == 0);
        pT3_matched_simIdx.push_back(matched_sim_trk_idxs);
    }

    vector<int> pT3_isDuplicate(pT4_matched_simIdx.size());

    for (unsigned int i = 0; i < pT3_matched_simIdx.size(); ++i)
    {
        bool isDuplicate = false;
        for (unsigned int isim = 0; isim < pT3_matched_simIdx[i].size(); ++isim)
        {
            if (sim_pT3_matched[pT3_matched_simIdx[i][isim]] > 1)
            {
                isDuplicate = true;
            }
        }
        pT3_isDuplicate[i] = isDuplicate;
    }

    ana.tx->setBranch<vector<int>>("sim_pT3_matched", sim_pT3_matched);
    ana.tx->setBranch<vector<vector<int>>>("sim_pT3_types", sim_pT3_types);
    ana.tx->setBranch<vector<int>>("pT3_isFake", pT3_isFake);
    ana.tx->setBranch<vector<int>>("pT3_isDuplicate", pT3_isDuplicate);
//    ana.tx->setBranch<vector<float>>("pT4_pt", pT4_pt);
//    ana.tx->setBranch<vector<float>>("pT4_eta", pT4_eta);
//    ana.tx->setBranch<vector<float>>("pT4_phi", pT4_phi);

}

//________________________________________________________________________________________________________________________________
void fillPixelLineSegmentOutputBranches(SDL::Event& event)
{
    SDL::segments& segmentsInGPU = (*event.getSegments());
    SDL::miniDoublets& miniDoubletsInGPU = (*event.getMiniDoublets());
    SDL::hits& hitsInGPU = (*event.getHits());
    SDL::modules& modulesInGPU = (*event.getModules());

    std::vector<int> sim_pLS_matched(trk.sim_pt().size(), 0);
    std::vector<vector<int>> sim_pLS_types(trk.sim_pt().size());
    std::vector<int> pLS_isFake;
    std::vector<vector<int>> pLS_matched_simIdx;
    std::vector<float> pLS_pt;
    std::vector<float> pLS_eta;
    std::vector<float> pLS_phi;

    const unsigned int N_MAX_PIXEL_SEGMENTS_PER_MODULE = 50000; 
    const unsigned int N_MAX_SEGMENTS_PER_MODULE = 600;
    unsigned int pixelModuleIndex = *(modulesInGPU.nModules) - 1;
    unsigned int nPixelSegments = std::min(segmentsInGPU.nSegments[pixelModuleIndex], N_MAX_PIXEL_SEGMENTS_PER_MODULE);
    for(unsigned int jdx = 0; jdx < nPixelSegments; jdx++)
    {
        unsigned int pixelSegmentIndex = pixelModuleIndex * N_MAX_SEGMENTS_PER_MODULE + jdx;
        unsigned int innerMiniDoubletIndex = segmentsInGPU.mdIndices[2 * pixelSegmentIndex];
        unsigned int outerMiniDoubletIndex = segmentsInGPU.mdIndices[2 * pixelSegmentIndex + 1];
        unsigned int innerMiniDoubletLowerHitIndex = miniDoubletsInGPU.hitIndices[2 * innerMiniDoubletIndex];
        unsigned int innerMiniDoubletUpperHitIndex = miniDoubletsInGPU.hitIndices[2 * innerMiniDoubletIndex + 1];
        unsigned int outerMiniDoubletLowerHitIndex = miniDoubletsInGPU.hitIndices[2 * outerMiniDoubletIndex];
        unsigned int outerMiniDoubletUpperHitIndex = miniDoubletsInGPU.hitIndices[2 * outerMiniDoubletIndex + 1];

        std::vector<int> hit_idxs = {
            (int) hitsInGPU.idxs[innerMiniDoubletLowerHitIndex], 
            (int) hitsInGPU.idxs[innerMiniDoubletUpperHitIndex], 
            (int) hitsInGPU.idxs[outerMiniDoubletLowerHitIndex], 
            (int) hitsInGPU.idxs[outerMiniDoubletUpperHitIndex]
        };

        std::vector<int> hit_types = {0,0,0,0};
        std::vector<int> module_idxs = {
            pixelModuleIndex, pixelModuleIndex, pixelModuleIndex, pixelModuleIndex
        };
        int layer0 = modulesInGPU.layers[module_idxs[0]];
        int layer2 = modulesInGPU.layers[module_idxs[2]];

        int subdet0 = modulesInGPU.subdets[module_idxs[0]];
        int subdet2 = modulesInGPU.subdets[module_idxs[2]];

        int logicallayer0 = 0;
        int logicallayer2 = 0;

        int layer_binary = 0;
        layer_binary |= (1 << logicallayer0);
        layer_binary |= (1 << logicallayer2);

        // sim track matched index
        std::vector<int> matched_sim_trk_idxs = matchedSimTrkIdxs(hit_idxs, hit_types);

        for (auto &isimtrk : matched_sim_trk_idxs)
        {
            sim_pLS_matched[isimtrk]++;
        }

        for (auto &isimtrk : matched_sim_trk_idxs)
        {
            sim_pLS_types[isimtrk].push_back(layer_binary);
        }


        pLS_isFake.push_back(matched_sim_trk_idxs.size() == 0);
        pLS_matched_simIdx.push_back(matched_sim_trk_idxs);

        pLS_pt.push_back(segmentsInGPU.ptIn[jdx]);
        pLS_eta.push_back(segmentsInGPU.eta[jdx]);
        pLS_phi.push_back(segmentsInGPU.phi[jdx]);

    }
    vector<int> pLS_isDuplicate(pLS_matched_simIdx.size());

    for (unsigned int i = 0; i < pLS_matched_simIdx.size(); ++i)
    {
        bool isDuplicate = false;
        for (unsigned int isim = 0; isim < pLS_matched_simIdx[i].size(); ++isim)
        {
            if (sim_pLS_matched[pLS_matched_simIdx[i][isim]] > 1)
            {
                isDuplicate = true;
            }
        }
        pLS_isDuplicate[i] = isDuplicate;
    }

    ana.tx->setBranch<vector<int>>("sim_pLS_matched",sim_pLS_matched);
    ana.tx->setBranch<vector<vector<int>>>("sim_pLS_types",sim_pLS_types);
    ana.tx->setBranch<vector<float>>("pLS_pt",pLS_pt);
    ana.tx->setBranch<vector<float>>("pLS_eta",pLS_eta);
    ana.tx->setBranch<vector<float>>("pLS_phi",pLS_phi);
    ana.tx->setBranch<vector<int>>("pLS_isFake",pLS_isFake);
    ana.tx->setBranch<vector<int>>("pLS_isDuplicate",pLS_isDuplicate);
}

//________________________________________________________________________________________________________________________________
void fillPixelQuadrupletOutputBranches(SDL::Event& event)
{
 

    SDL::tracklets& trackletsInGPU = (*event.getTracklets());
    SDL::segments& segmentsInGPU = (*event.getSegments());
    SDL::miniDoublets& miniDoubletsInGPU = (*event.getMiniDoublets());
    SDL::hits& hitsInGPU = (*event.getHits());
    SDL::modules& modulesInGPU = (*event.getModules());

    // Did it match to track candidate?
    std::vector<int> sim_pT4_matched(trk.sim_pt().size());
    std::vector<vector<int>> sim_pT4_types(trk.sim_pt().size());
    std::vector<int> pT4_isFake;
    std::vector<vector<int>> pT4_matched_simIdx;
    std::vector<float> pT4_pt;
    std::vector<float> pT4_eta;
    std::vector<float> pT4_phi;
#ifdef CUT_VALUE_DEBUG
    std::vector<float> pT4_ZOut;
    std::vector<float> pT4_RtOut;
    std::vector<float> pT4_deltaPhiPos;
    std::vector<float> pT4_deltaPhi;
    std::vector<float> pT4_betaIn;
    std::vector<float> pT4_betaOut;
    std::vector<float> pT4_deltaBeta;
    std::vector<float> pT4_ZLo;
    std::vector<float> pT4_ZHi;
    std::vector<float> pT4_RtLo;
    std::vector<float> pT4_RtHi;
    std::vector<float> pT4_kZ;
    std::vector<float> pT4_ZLoPointed;
    std::vector<float> pT4_ZHiPointed;
    std::vector<float> pT4_sdlCut;
    std::vector<float> pT4_betaInCut;
    std::vector<float> pT4_betaOutCut;
    std::vector<float> pT4_deltaBetaCut;
    std::vector<int> layer_binaries;
    std::vector<int> moduleType_binaries;
#endif

    const unsigned int N_MAX_PIXEL_TRACKLETS_PER_MODULE = 3000000;
    const int N_MAX_TRACKLETS_PER_MODULE = 8000;

    unsigned int pixelModuleIndex = *(modulesInGPU.nLowerModules);
    unsigned int nPixelTracklets = std::min(trackletsInGPU.nTracklets[pixelModuleIndex],N_MAX_PIXEL_TRACKLETS_PER_MODULE);

    for(unsigned int jdx = 0; jdx < nPixelTracklets; jdx++)
    {
        unsigned int trackletIndex = pixelModuleIndex * N_MAX_TRACKLETS_PER_MODULE + jdx;
        unsigned int innerSegmentIndex = trackletsInGPU.segmentIndices[2 * trackletIndex];
        unsigned int outerSegmentIndex = trackletsInGPU.segmentIndices[2 * trackletIndex + 1];
        float betaIn = trackletsInGPU.betaIn[trackletIndex];
        float betaOut = trackletsInGPU.betaOut[trackletIndex];


        unsigned int innerSegmentInnerMiniDoubletIndex = segmentsInGPU.mdIndices[2 * innerSegmentIndex];
        unsigned int innerSegmentOuterMiniDoubletIndex = segmentsInGPU.mdIndices[2 * innerSegmentIndex + 1];
        unsigned int outerSegmentInnerMiniDoubletIndex = segmentsInGPU.mdIndices[2 * outerSegmentIndex];
        unsigned int outerSegmentOuterMiniDoubletIndex = segmentsInGPU.mdIndices[2 * outerSegmentIndex + 1];

        unsigned int innerSegmentInnerMiniDoubletLowerHitIndex = miniDoubletsInGPU.hitIndices[2 * innerSegmentInnerMiniDoubletIndex];
        unsigned int innerSegmentInnerMiniDoubletUpperHitIndex = miniDoubletsInGPU.hitIndices[2 * innerSegmentInnerMiniDoubletIndex + 1];
        unsigned int innerSegmentOuterMiniDoubletLowerHitIndex = miniDoubletsInGPU.hitIndices[2 * innerSegmentOuterMiniDoubletIndex];
        unsigned int innerSegmentOuterMiniDoubletUpperHitIndex = miniDoubletsInGPU.hitIndices[2 * innerSegmentOuterMiniDoubletIndex + 1];
        unsigned int outerSegmentInnerMiniDoubletLowerHitIndex = miniDoubletsInGPU.hitIndices[2 * outerSegmentInnerMiniDoubletIndex];
        unsigned int outerSegmentInnerMiniDoubletUpperHitIndex = miniDoubletsInGPU.hitIndices[2 * outerSegmentInnerMiniDoubletIndex + 1];
        unsigned int outerSegmentOuterMiniDoubletLowerHitIndex = miniDoubletsInGPU.hitIndices[2 * outerSegmentOuterMiniDoubletIndex];
        unsigned int outerSegmentOuterMiniDoubletUpperHitIndex = miniDoubletsInGPU.hitIndices[2 * outerSegmentOuterMiniDoubletIndex + 1];

        std::vector<int> hit_idxs = {
                (int) hitsInGPU.idxs[innerSegmentInnerMiniDoubletLowerHitIndex],
                (int) hitsInGPU.idxs[innerSegmentInnerMiniDoubletUpperHitIndex],
                (int) hitsInGPU.idxs[innerSegmentOuterMiniDoubletLowerHitIndex],
                (int) hitsInGPU.idxs[innerSegmentOuterMiniDoubletUpperHitIndex],
                (int) hitsInGPU.idxs[outerSegmentInnerMiniDoubletLowerHitIndex],
                (int) hitsInGPU.idxs[outerSegmentInnerMiniDoubletUpperHitIndex],
                (int) hitsInGPU.idxs[outerSegmentOuterMiniDoubletLowerHitIndex],
                (int) hitsInGPU.idxs[outerSegmentOuterMiniDoubletUpperHitIndex],
        };

        unsigned int iia_idx = segmentsInGPU.innerMiniDoubletAnchorHitIndices[innerSegmentIndex];
        unsigned int ooa_idx = segmentsInGPU.outerMiniDoubletAnchorHitIndices[outerSegmentIndex];

        const float dr = sqrt(pow(hitsInGPU.xs[iia_idx] - hitsInGPU.xs[ooa_idx], 2) + pow(hitsInGPU.ys[iia_idx] - hitsInGPU.ys[ooa_idx], 2));
        const float kRinv1GeVf = (2.99792458e-3 * 3.8);
        const float k2Rinv1GeVf = kRinv1GeVf / 2.;
        const float ptAv = dr * k2Rinv1GeVf / sin((betaIn + betaOut) / 2.);

        std::vector<int> hit_types;
        hit_types.push_back(0);
        hit_types.push_back(0);
        hit_types.push_back(0);
        hit_types.push_back(0);
        hit_types.push_back(4);
        hit_types.push_back(4);
        hit_types.push_back(4);
        hit_types.push_back(4);

        std::vector<int> module_idxs = {
                (int) hitsInGPU.moduleIndices[innerSegmentInnerMiniDoubletLowerHitIndex],
                (int) hitsInGPU.moduleIndices[innerSegmentInnerMiniDoubletUpperHitIndex],
                (int) hitsInGPU.moduleIndices[innerSegmentOuterMiniDoubletLowerHitIndex],
                (int) hitsInGPU.moduleIndices[innerSegmentOuterMiniDoubletUpperHitIndex],
                (int) hitsInGPU.moduleIndices[outerSegmentInnerMiniDoubletLowerHitIndex],
                (int) hitsInGPU.moduleIndices[outerSegmentInnerMiniDoubletUpperHitIndex],
                (int) hitsInGPU.moduleIndices[outerSegmentOuterMiniDoubletLowerHitIndex],
                (int) hitsInGPU.moduleIndices[outerSegmentOuterMiniDoubletUpperHitIndex],
        };

        int layer0 = modulesInGPU.layers[module_idxs[0]];
        int layer2 = modulesInGPU.layers[module_idxs[2]];
        int layer4 = modulesInGPU.layers[module_idxs[4]];
        int layer6 = modulesInGPU.layers[module_idxs[6]];

        int subdet0 = modulesInGPU.subdets[module_idxs[0]];
        int subdet2 = modulesInGPU.subdets[module_idxs[2]];
        int subdet4 = modulesInGPU.subdets[module_idxs[4]];
        int subdet6 = modulesInGPU.subdets[module_idxs[6]];

        int logicallayer0 = 0;
        int logicallayer2 = 0;
        int logicallayer4 = layer4 + 6 * (subdet4 == 4);
        int logicallayer6 = layer6 + 6 * (subdet6 == 4);

        int layer_binary = 0;
        layer_binary |= (1 << logicallayer0);
        layer_binary |= (1 << logicallayer2);
        layer_binary |= (1 << logicallayer4);
        layer_binary |= (1 << logicallayer6);
#ifdef CUT_VALUE_DEBUG
        layer_binaries.push_back(layer_binary);
#endif

        // sim track matched index
        std::vector<int> matched_sim_trk_idxs = matchedSimTrkIdxs(hit_idxs, hit_types);

        for (auto &isimtrk : matched_sim_trk_idxs)
        {
            sim_pT4_matched[isimtrk]++;
        }

        for (auto &isimtrk : matched_sim_trk_idxs)
        {
            sim_pT4_types[isimtrk].push_back(layer_binary);
        }

        const float pt = ptAv;
        float eta = -999;
        float phi = -999;
        SDL::CPU::Hit hitA(trk.pix_x()[hit_idxs[0]], trk.pix_y()[hit_idxs[0]], trk.pix_z()[hit_idxs[0]]);
        SDL::CPU::Hit hitB(trk.ph2_x()[hit_idxs[7]], trk.ph2_y()[hit_idxs[7]], trk.ph2_z()[hit_idxs[7]]);
        eta = hitB.eta();
        phi = hitA.phi();
        pT4_isFake.push_back(matched_sim_trk_idxs.size() == 0);
        pT4_pt.push_back(pt);
        pT4_eta.push_back(eta);
        pT4_phi.push_back(phi);
        pT4_matched_simIdx.push_back(matched_sim_trk_idxs);

#ifdef CUT_VALUE_DEBUG
        //debug stuff
        float zOut = trackletsInGPU.zOut[trackletIndex];
        float rtOut = trackletsInGPU.rtOut[trackletIndex];
        float deltaPhiPos = trackletsInGPU.deltaPhiPos[trackletIndex];
        float deltaPhi = trackletsInGPU.deltaPhi[trackletIndex];
        //betaIn and betaOut already defined!
        float deltaBeta = betaIn - betaOut;
        float zLo = trackletsInGPU.zLo[trackletIndex];
        float zHi = trackletsInGPU.zHi[trackletIndex];
        float rtLo = trackletsInGPU.rtLo[trackletIndex];
        float rtHi = trackletsInGPU.rtHi[trackletIndex];
        float kZ = trackletsInGPU.kZ[trackletIndex];
        float zLoPointed = trackletsInGPU.zLoPointed[trackletIndex];
        float zHiPointed = trackletsInGPU.zHiPointed[trackletIndex];
        float sdlCut = trackletsInGPU.sdlCut[trackletIndex];
        float betaInCut = trackletsInGPU.betaInCut[trackletIndex];
        float betaOutCut = trackletsInGPU.betaOutCut[trackletIndex];
        float deltaBetaCut = trackletsInGPU.deltaBetaCut[trackletIndex];

        pT4_ZOut.push_back(zOut);
        pT4_RtOut.push_back(rtOut);
        pT4_deltaPhiPos.push_back(deltaPhiPos);
        pT4_deltaPhi.push_back(deltaPhi);
        pT4_betaIn.push_back(betaIn);
        pT4_betaOut.push_back(betaOut);
        pT4_deltaBeta.push_back(deltaBeta);
        pT4_ZLo.push_back(zLo);
        pT4_ZHi.push_back(zHi);
        pT4_RtLo.push_back(rtLo);
        pT4_RtHi.push_back(rtHi);
        pT4_kZ.push_back(kZ);
        pT4_ZLoPointed.push_back(zLoPointed);
        pT4_ZHiPointed.push_back(zHiPointed);
        pT4_sdlCut.push_back(sdlCut);
        pT4_betaInCut.push_back(betaInCut);
        pT4_betaOutCut.push_back(betaOutCut);
        pT4_deltaBetaCut.push_back(deltaBetaCut);
#endif
    }

    vector<int> pT4_isDuplicate(pT4_matched_simIdx.size());

    for (unsigned int i = 0; i < pT4_matched_simIdx.size(); ++i)
    {
        bool isDuplicate = false;
        for (unsigned int isim = 0; isim < pT4_matched_simIdx[i].size(); ++isim)
        {
            if (sim_pT4_matched[pT4_matched_simIdx[i][isim]] > 1)
            {
                isDuplicate = true;
            }
        }
        pT4_isDuplicate[i] = isDuplicate;
    }

    ana.tx->setBranch<vector<int>>("sim_pT4_matched", sim_pT4_matched);
    ana.tx->setBranch<vector<vector<int>>>("sim_pT4_types", sim_pT4_types);
    ana.tx->setBranch<vector<int>>("pT4_isFake", pT4_isFake);
    ana.tx->setBranch<vector<int>>("pT4_isDuplicate", pT4_isDuplicate);
    ana.tx->setBranch<vector<float>>("pT4_pt", pT4_pt);
    ana.tx->setBranch<vector<float>>("pT4_eta", pT4_eta);
    ana.tx->setBranch<vector<float>>("pT4_phi", pT4_phi);
#ifdef CUT_VALUE_DEBUG
    ana.tx->setBranch<vector<float>>("pT4_zOut",pT4_ZOut);
    ana.tx->setBranch<vector<float>>("pT4_rtOut",pT4_RtOut);
    ana.tx->setBranch<vector<float>>("pT4_deltaPhiPos",pT4_deltaPhiPos);
    ana.tx->setBranch<vector<float>>("pT4_deltaPhi",pT4_deltaPhi);
    ana.tx->setBranch<vector<float>>("pT4_betaIn",pT4_betaIn);
    ana.tx->setBranch<vector<float>>("pT4_betaOut",pT4_betaOut);
    ana.tx->setBranch<vector<float>>("pT4_deltaBeta",pT4_deltaBeta);
    ana.tx->setBranch<vector<float>>("pT4_zLo",pT4_ZLo);
    ana.tx->setBranch<vector<float>>("pT4_zHi",pT4_ZHi);
    ana.tx->setBranch<vector<float>>("pT4_rtLo",pT4_RtLo);
    ana.tx->setBranch<vector<float>>("pT4_rtHi",pT4_RtHi);
    ana.tx->setBranch<vector<float>>("pT4_kZ",pT4_kZ);
    ana.tx->setBranch<vector<float>>("pT4_zLoPointed",pT4_ZLoPointed);
    ana.tx->setBranch<vector<float>>("pT4_zHiPointed",pT4_ZHiPointed);
    ana.tx->setBranch<vector<float>>("pT4_sdlCut",pT4_sdlCut);
    ana.tx->setBranch<vector<float>>("pT4_betaInCut",pT4_betaInCut);
    ana.tx->setBranch<vector<float>>("pT4_betaOutCut",pT4_betaOutCut);
    ana.tx->setBranch<vector<float>>("pT4_deltaBetaCut",pT4_deltaBetaCut);
    ana.tx->setBranch<vector<int>>("pT4_layer_binary",layer_binaries);
#endif
}


//________________________________________________________________________________________________________________________________
void fillQuadrupletOutputBranches(SDL::Event& event)
{

    SDL::tracklets& trackletsInGPU = (*event.getTracklets());
    SDL::segments& segmentsInGPU = (*event.getSegments());
    SDL::miniDoublets& miniDoubletsInGPU = (*event.getMiniDoublets());
    SDL::hits& hitsInGPU = (*event.getHits());
    SDL::modules& modulesInGPU = (*event.getModules());

    // Did it match to track candidate?
    std::vector<int> sim_T4_matched(trk.sim_pt().size());
    std::vector<vector<int>> sim_T4_types(trk.sim_pt().size());
    std::vector<int> t4_isFake;
    std::vector<vector<int>> t4_matched_simIdx;
    std::vector<float> t4_pt;
    std::vector<float> t4_eta;
    std::vector<float> t4_phi;
#ifdef CUT_VALUE_DEBUG
    std::vector<float> t4_ZOut;
    std::vector<float> t4_RtOut;
    std::vector<float> t4_deltaPhiPos;
    std::vector<float> t4_deltaPhi;
    std::vector<float> t4_betaIn;
    std::vector<float> t4_betaOut;
    std::vector<float> t4_deltaBeta;
    std::vector<float> t4_ZLo;
    std::vector<float> t4_ZHi;
    std::vector<float> t4_RtLo;
    std::vector<float> t4_RtHi;
    std::vector<float> t4_kZ;
    std::vector<float> t4_ZLoPointed;
    std::vector<float> t4_ZHiPointed;
    std::vector<float> t4_sdlCut;
    std::vector<float> t4_betaInCut;
    std::vector<float> t4_betaOutCut;
    std::vector<float> t4_deltaBetaCut;
    std::vector<int> layer_binaries;
    std::vector<int> moduleType_binaries;
#endif

    const int MAX_NTRACKLET_PER_MODULE = 8000;
    for (unsigned int idx = 0; idx < *(modulesInGPU.nLowerModules); idx++) // "<=" because cheating to include pixel track candidate lower module
    {

        unsigned int nTracklets = trackletsInGPU.nTracklets[idx];

        if (idx < *(modulesInGPU.nLowerModules) and nTracklets > MAX_NTRACKLET_PER_MODULE)
        {
            nTracklets = MAX_NTRACKLET_PER_MODULE;
        }

        for (unsigned int jdx = 0; jdx < nTracklets; jdx++)
        {
            unsigned int trackletIndex = MAX_NTRACKLET_PER_MODULE * idx + jdx; // this line causes the issue
            unsigned int innerSegmentIndex = -1;
            unsigned int outerSegmentIndex = -1;

            innerSegmentIndex = trackletsInGPU.segmentIndices[2 * trackletIndex];
            outerSegmentIndex = trackletsInGPU.segmentIndices[2 * trackletIndex + 1];
            float betaIn = trackletsInGPU.betaIn[trackletIndex];
            float betaOut = trackletsInGPU.betaOut[trackletIndex];


            unsigned int innerSegmentInnerMiniDoubletIndex = segmentsInGPU.mdIndices[2 * innerSegmentIndex];
            unsigned int innerSegmentOuterMiniDoubletIndex = segmentsInGPU.mdIndices[2 * innerSegmentIndex + 1];
            unsigned int outerSegmentInnerMiniDoubletIndex = segmentsInGPU.mdIndices[2 * outerSegmentIndex];
            unsigned int outerSegmentOuterMiniDoubletIndex = segmentsInGPU.mdIndices[2 * outerSegmentIndex + 1];

            unsigned int innerSegmentInnerMiniDoubletLowerHitIndex = miniDoubletsInGPU.hitIndices[2 * innerSegmentInnerMiniDoubletIndex];
            unsigned int innerSegmentInnerMiniDoubletUpperHitIndex = miniDoubletsInGPU.hitIndices[2 * innerSegmentInnerMiniDoubletIndex + 1];
            unsigned int innerSegmentOuterMiniDoubletLowerHitIndex = miniDoubletsInGPU.hitIndices[2 * innerSegmentOuterMiniDoubletIndex];
            unsigned int innerSegmentOuterMiniDoubletUpperHitIndex = miniDoubletsInGPU.hitIndices[2 * innerSegmentOuterMiniDoubletIndex + 1];
            unsigned int outerSegmentInnerMiniDoubletLowerHitIndex = miniDoubletsInGPU.hitIndices[2 * outerSegmentInnerMiniDoubletIndex];
            unsigned int outerSegmentInnerMiniDoubletUpperHitIndex = miniDoubletsInGPU.hitIndices[2 * outerSegmentInnerMiniDoubletIndex + 1];
            unsigned int outerSegmentOuterMiniDoubletLowerHitIndex = miniDoubletsInGPU.hitIndices[2 * outerSegmentOuterMiniDoubletIndex];
            unsigned int outerSegmentOuterMiniDoubletUpperHitIndex = miniDoubletsInGPU.hitIndices[2 * outerSegmentOuterMiniDoubletIndex + 1];

            std::vector<int> hit_idxs = {
                (int) hitsInGPU.idxs[innerSegmentInnerMiniDoubletLowerHitIndex],
                (int) hitsInGPU.idxs[innerSegmentInnerMiniDoubletUpperHitIndex],
                (int) hitsInGPU.idxs[innerSegmentOuterMiniDoubletLowerHitIndex],
                (int) hitsInGPU.idxs[innerSegmentOuterMiniDoubletUpperHitIndex],
                (int) hitsInGPU.idxs[outerSegmentInnerMiniDoubletLowerHitIndex],
                (int) hitsInGPU.idxs[outerSegmentInnerMiniDoubletUpperHitIndex],
                (int) hitsInGPU.idxs[outerSegmentOuterMiniDoubletLowerHitIndex],
                (int) hitsInGPU.idxs[outerSegmentOuterMiniDoubletUpperHitIndex],
            };

            unsigned int iia_idx = segmentsInGPU.innerMiniDoubletAnchorHitIndices[innerSegmentIndex];
            unsigned int ooa_idx = segmentsInGPU.outerMiniDoubletAnchorHitIndices[outerSegmentIndex];

            const float dr = sqrt(pow(hitsInGPU.xs[iia_idx] - hitsInGPU.xs[ooa_idx], 2) + pow(hitsInGPU.ys[iia_idx] - hitsInGPU.ys[ooa_idx], 2));
            const float kRinv1GeVf = (2.99792458e-3 * 3.8);
            const float k2Rinv1GeVf = kRinv1GeVf / 2.;
            const float ptAv = dr * k2Rinv1GeVf / sin((betaIn + betaOut) / 2.);

            std::vector<int> hit_types;
            hit_types.push_back(4);
            hit_types.push_back(4);
            hit_types.push_back(4);
            hit_types.push_back(4);
            hit_types.push_back(4);
            hit_types.push_back(4);
            hit_types.push_back(4);
            hit_types.push_back(4);

            std::vector<int> module_idxs = {
                (int) hitsInGPU.moduleIndices[innerSegmentInnerMiniDoubletLowerHitIndex],
                (int) hitsInGPU.moduleIndices[innerSegmentInnerMiniDoubletUpperHitIndex],
                (int) hitsInGPU.moduleIndices[innerSegmentOuterMiniDoubletLowerHitIndex],
                (int) hitsInGPU.moduleIndices[innerSegmentOuterMiniDoubletUpperHitIndex],
                (int) hitsInGPU.moduleIndices[outerSegmentInnerMiniDoubletLowerHitIndex],
                (int) hitsInGPU.moduleIndices[outerSegmentInnerMiniDoubletUpperHitIndex],
                (int) hitsInGPU.moduleIndices[outerSegmentOuterMiniDoubletLowerHitIndex],
                (int) hitsInGPU.moduleIndices[outerSegmentOuterMiniDoubletUpperHitIndex],
            };

            int layer0 = modulesInGPU.layers[module_idxs[0]];
            int layer2 = modulesInGPU.layers[module_idxs[2]];
            int layer4 = modulesInGPU.layers[module_idxs[4]];
            int layer6 = modulesInGPU.layers[module_idxs[6]];

            int subdet0 = modulesInGPU.subdets[module_idxs[0]];
            int subdet2 = modulesInGPU.subdets[module_idxs[2]];
            int subdet4 = modulesInGPU.subdets[module_idxs[4]];
            int subdet6 = modulesInGPU.subdets[module_idxs[6]];

            int logicallayer0 = layer0 + 6 * (subdet0 == 4);
            int logicallayer2 = layer2 + 6 * (subdet2 == 4);
            int logicallayer4 = layer4 + 6 * (subdet4 == 4);
            int logicallayer6 = layer6 + 6 * (subdet6 == 4);

            int layer_binary = 0;
            layer_binary |= (1 << logicallayer0);
            layer_binary |= (1 << logicallayer2);
            layer_binary |= (1 << logicallayer4);
            layer_binary |= (1 << logicallayer6);
#ifdef CUT_VALUE_DEBUG
            int moduleType_binary = 0;

            int moduleType0 = modulesInGPU.moduleType[module_idxs[0]];
            int moduleType2 = modulesInGPU.moduleType[module_idxs[2]];
            int moduleType4 = modulesInGPU.moduleType[module_idxs[4]];
            int moduleType6 = modulesInGPU.moduleType[module_idxs[6]];
            
            moduleType_binary |= (moduleType0 << 0);
            moduleType_binary |= (moduleType2 << 2);
            moduleType_binary |= (moduleType4 << 4);
            moduleType_binary |= (moduleType6 << 6);
           
            layer_binaries.push_back(layer_binary);
            moduleType_binaries.push_back(moduleType_binary);
#endif

            // sim track matched index
            std::vector<int> matched_sim_trk_idxs = matchedSimTrkIdxs(hit_idxs, hit_types);

            for (auto &isimtrk : matched_sim_trk_idxs)
            {
                sim_T4_matched[isimtrk]++;
            }

            for (auto &isimtrk : matched_sim_trk_idxs)
            {
                sim_T4_types[isimtrk].push_back(layer_binary);
            }

            // Compute pt, eta, phi of T4
            const float pt = ptAv;
            float eta = -999;
            float phi = -999;
            SDL::CPU::Hit hitA(trk.ph2_x()[hit_idxs[0]], trk.ph2_y()[hit_idxs[0]], trk.ph2_z()[hit_idxs[0]]);
            SDL::CPU::Hit hitB(trk.ph2_x()[hit_idxs[7]], trk.ph2_y()[hit_idxs[7]], trk.ph2_z()[hit_idxs[7]]);
            eta = hitB.eta();
            phi = hitA.phi();
            // std::cout <<  " hit_idx[0]: " << hit_idxs[0] <<  " hit_idx[1]: " << hit_idxs[1] <<  " hit_idx[2]: " << hit_idxs[2] <<  " hit_idx[3]: " << hit_idxs[3] <<  " hit_idx[4]: " << hit_idxs[4] <<  " hit_idx[5]: " << hit_idxs[5] <<  " hit_idx[6]: " << hit_idxs[6] <<  " hit_idx[7]: " << hit_idxs[7] <<  " betaIn: " << betaIn <<  " betaOut: " << betaOut <<  " dr: " << dr <<  std::endl;

            t4_isFake.push_back(matched_sim_trk_idxs.size() == 0);
            t4_pt.push_back(pt);
            t4_eta.push_back(eta);
            t4_phi.push_back(phi);
            t4_matched_simIdx.push_back(matched_sim_trk_idxs);

            //debug stuff
#ifdef CUT_VALUE_DEBUG
            float zOut = trackletsInGPU.zOut[trackletIndex];
            float rtOut = trackletsInGPU.rtOut[trackletIndex];
            float deltaPhiPos = trackletsInGPU.deltaPhiPos[trackletIndex];
            float deltaPhi = trackletsInGPU.deltaPhi[trackletIndex];
            //betaIn and betaOut already defined!
            float deltaBeta = betaIn - betaOut;
            float zLo = trackletsInGPU.zLo[trackletIndex];
            float zHi = trackletsInGPU.zHi[trackletIndex];
            float rtLo = trackletsInGPU.rtLo[trackletIndex];
            float rtHi = trackletsInGPU.rtHi[trackletIndex];
            float kZ = trackletsInGPU.kZ[trackletIndex];
            float zLoPointed = trackletsInGPU.zLoPointed[trackletIndex];
            float zHiPointed = trackletsInGPU.zHiPointed[trackletIndex];
            float sdlCut = trackletsInGPU.sdlCut[trackletIndex];
            float betaInCut = trackletsInGPU.betaInCut[trackletIndex];
            float betaOutCut = trackletsInGPU.betaOutCut[trackletIndex];
            float deltaBetaCut = trackletsInGPU.deltaBetaCut[trackletIndex];

            t4_ZOut.push_back(zOut);
            t4_RtOut.push_back(rtOut);
            t4_deltaPhiPos.push_back(deltaPhiPos);
            t4_deltaPhi.push_back(deltaPhi);
            t4_betaIn.push_back(betaIn);
            t4_betaOut.push_back(betaOut);
            t4_deltaBeta.push_back(deltaBeta);
            t4_ZLo.push_back(zLo);
            t4_ZHi.push_back(zHi);
            t4_RtLo.push_back(rtLo);
            t4_RtHi.push_back(rtHi);
            t4_kZ.push_back(kZ);
            t4_ZLoPointed.push_back(zLoPointed);
            t4_ZHiPointed.push_back(zHiPointed);
            t4_sdlCut.push_back(sdlCut);
            t4_betaInCut.push_back(betaInCut);
            t4_betaOutCut.push_back(betaOutCut);
            t4_deltaBetaCut.push_back(deltaBetaCut);
#endif
        }

    }

    ana.tx->setBranch<vector<int>>("sim_T4_matched", sim_T4_matched);
    ana.tx->setBranch<vector<vector<int>>>("sim_T4_types", sim_T4_types);

    vector<int> t4_isDuplicate(t4_matched_simIdx.size());

    for (unsigned int i = 0; i < t4_matched_simIdx.size(); ++i)
    {
        bool isDuplicate = false;
        for (unsigned int isim = 0; isim < t4_matched_simIdx[i].size(); ++isim)
        {
            if (sim_T4_matched[t4_matched_simIdx[i][isim]] > 1)
            {
                isDuplicate = true;
            }
        }
        t4_isDuplicate[i] = isDuplicate;
    }

    ana.tx->setBranch<vector<float>>("t4_pt", t4_pt);
    ana.tx->setBranch<vector<float>>("t4_eta", t4_eta);
    ana.tx->setBranch<vector<float>>("t4_phi", t4_phi);
    ana.tx->setBranch<vector<int>>("t4_isFake", t4_isFake);
    ana.tx->setBranch<vector<int>>("t4_isDuplicate", t4_isDuplicate);
#ifdef CUT_VALUE_DEBUG
    ana.tx->setBranch<vector<float>>("t4_zOut",t4_ZOut);
    ana.tx->setBranch<vector<float>>("t4_rtOut",t4_RtOut);
    ana.tx->setBranch<vector<float>>("t4_deltaPhiPos",t4_deltaPhiPos);
    ana.tx->setBranch<vector<float>>("t4_deltaPhi",t4_deltaPhi);
    ana.tx->setBranch<vector<float>>("t4_betaIn",t4_betaIn);
    ana.tx->setBranch<vector<float>>("t4_betaOut",t4_betaOut);
    ana.tx->setBranch<vector<float>>("t4_deltaBeta",t4_deltaBeta);
    ana.tx->setBranch<vector<float>>("t4_zLo",t4_ZLo);
    ana.tx->setBranch<vector<float>>("t4_zHi",t4_ZHi);
    ana.tx->setBranch<vector<float>>("t4_rtLo",t4_RtLo);
    ana.tx->setBranch<vector<float>>("t4_rtHi",t4_RtHi);
    ana.tx->setBranch<vector<float>>("t4_kZ",t4_kZ);
    ana.tx->setBranch<vector<float>>("t4_zLoPointed",t4_ZLoPointed);
    ana.tx->setBranch<vector<float>>("t4_zHiPointed",t4_ZHiPointed);
    ana.tx->setBranch<vector<float>>("t4_sdlCut",t4_sdlCut);
    ana.tx->setBranch<vector<float>>("t4_betaInCut",t4_betaInCut);
    ana.tx->setBranch<vector<float>>("t4_betaOutCut",t4_betaOutCut);
    ana.tx->setBranch<vector<float>>("t4_deltaBetaCut",t4_deltaBetaCut);
    ana.tx->setBranch<vector<int>>("t4_layer_binary",layer_binaries);
    ana.tx->setBranch<vector<int>>("t4_moduleType_binary",moduleType_binaries);
#endif
}

//________________________________________________________________________________________________________________________________
void fillTripletOutputBranches(SDL::Event& event)
{

    SDL::triplets& tripletsInGPU = (*event.getTriplets());
    SDL::segments& segmentsInGPU = (*event.getSegments());
    SDL::miniDoublets& miniDoubletsInGPU = (*event.getMiniDoublets());
    SDL::hits& hitsInGPU = (*event.getHits());
    SDL::modules& modulesInGPU = (*event.getModules());

    // Did it match to track candidate?
    std::vector<int> sim_T3_matched(trk.sim_pt().size());
    std::vector<vector<int>> sim_T3_types(trk.sim_pt().size());
    std::vector<int> t3_isFake;
    std::vector<vector<int>> t3_matched_simIdx;
    std::vector<float> t3_pt;
    std::vector<float> t3_eta;
    std::vector<float> t3_phi;
#ifdef CUT_VALUE_DEBUG
    std::vector<float> t3_ZOut;
    std::vector<float> t3_RtOut;
    std::vector<float> t3_deltaPhiPos;
    std::vector<float> t3_deltaPhi;
    std::vector<float> t3_betaIn;
    std::vector<float> t3_betaOut;
    std::vector<float> t3_deltaBeta;
    std::vector<float> t3_ZLo;
    std::vector<float> t3_ZHi;
    std::vector<float> t3_RtLo;
    std::vector<float> t3_RtHi;
    std::vector<float> t3_kZ;
    std::vector<float> t3_ZLoPointed;
    std::vector<float> t3_ZHiPointed;
    std::vector<float> t3_sdlCut;
    std::vector<float> t3_betaInCut;
    std::vector<float> t3_betaOutCut;
    std::vector<float> t3_deltaBetaCut;
    std::vector<int> layer_binaries;
    std::vector<int> moduleType_binaries;
#endif

    const int MAX_NTRIPLET_PER_MODULE = 5000;
    for (unsigned int idx = 0; idx < *(modulesInGPU.nLowerModules); idx++) // "<=" because cheating to include pixel track candidate lower module
    {

        unsigned int nTriplets = tripletsInGPU.nTriplets[idx];

        if (idx < *(modulesInGPU.nLowerModules) and nTriplets > MAX_NTRIPLET_PER_MODULE)
        {
            nTriplets = MAX_NTRIPLET_PER_MODULE;
        }

        for (unsigned int jdx = 0; jdx < nTriplets; jdx++)
        {
            unsigned int tripletIndex = MAX_NTRIPLET_PER_MODULE * idx + jdx; // this line causes the issue
            unsigned int innerSegmentIndex = -1;
            unsigned int outerSegmentIndex = -1;

            innerSegmentIndex = tripletsInGPU.segmentIndices[2 * tripletIndex];
            outerSegmentIndex = tripletsInGPU.segmentIndices[2 * tripletIndex + 1];
            float betaIn = tripletsInGPU.betaIn[tripletIndex];
            float betaOut = tripletsInGPU.betaOut[tripletIndex];

            unsigned int innerSegmentInnerMiniDoubletIndex = segmentsInGPU.mdIndices[2 * innerSegmentIndex];
            unsigned int innerSegmentOuterMiniDoubletIndex = segmentsInGPU.mdIndices[2 * innerSegmentIndex + 1];
            unsigned int outerSegmentInnerMiniDoubletIndex = segmentsInGPU.mdIndices[2 * outerSegmentIndex];
            unsigned int outerSegmentOuterMiniDoubletIndex = segmentsInGPU.mdIndices[2 * outerSegmentIndex + 1];

            unsigned int innerSegmentInnerMiniDoubletLowerHitIndex = miniDoubletsInGPU.hitIndices[2 * innerSegmentInnerMiniDoubletIndex];
            unsigned int innerSegmentInnerMiniDoubletUpperHitIndex = miniDoubletsInGPU.hitIndices[2 * innerSegmentInnerMiniDoubletIndex + 1];
            unsigned int innerSegmentOuterMiniDoubletLowerHitIndex = miniDoubletsInGPU.hitIndices[2 * innerSegmentOuterMiniDoubletIndex];
            unsigned int innerSegmentOuterMiniDoubletUpperHitIndex = miniDoubletsInGPU.hitIndices[2 * innerSegmentOuterMiniDoubletIndex + 1];
            unsigned int outerSegmentInnerMiniDoubletLowerHitIndex = miniDoubletsInGPU.hitIndices[2 * outerSegmentInnerMiniDoubletIndex];
            unsigned int outerSegmentInnerMiniDoubletUpperHitIndex = miniDoubletsInGPU.hitIndices[2 * outerSegmentInnerMiniDoubletIndex + 1];
            unsigned int outerSegmentOuterMiniDoubletLowerHitIndex = miniDoubletsInGPU.hitIndices[2 * outerSegmentOuterMiniDoubletIndex];
            unsigned int outerSegmentOuterMiniDoubletUpperHitIndex = miniDoubletsInGPU.hitIndices[2 * outerSegmentOuterMiniDoubletIndex + 1];

            std::vector<int> hit_idxs = {
                (int) hitsInGPU.idxs[innerSegmentInnerMiniDoubletLowerHitIndex],
                (int) hitsInGPU.idxs[innerSegmentInnerMiniDoubletUpperHitIndex],
                (int) hitsInGPU.idxs[innerSegmentOuterMiniDoubletLowerHitIndex],
                (int) hitsInGPU.idxs[innerSegmentOuterMiniDoubletUpperHitIndex],
                (int) hitsInGPU.idxs[outerSegmentInnerMiniDoubletLowerHitIndex],
                (int) hitsInGPU.idxs[outerSegmentInnerMiniDoubletUpperHitIndex],
                (int) hitsInGPU.idxs[outerSegmentOuterMiniDoubletLowerHitIndex],
                (int) hitsInGPU.idxs[outerSegmentOuterMiniDoubletUpperHitIndex],
            };

            unsigned int iia_idx = segmentsInGPU.innerMiniDoubletAnchorHitIndices[innerSegmentIndex];
            unsigned int ooa_idx = segmentsInGPU.outerMiniDoubletAnchorHitIndices[outerSegmentIndex];

            const float dr = sqrt(pow(hitsInGPU.xs[iia_idx] - hitsInGPU.xs[ooa_idx], 2) + pow(hitsInGPU.ys[iia_idx] - hitsInGPU.ys[ooa_idx], 2));
            const float kRinv1GeVf = (2.99792458e-3 * 3.8);
            const float k2Rinv1GeVf = kRinv1GeVf / 2.;
            const float ptAv = dr * k2Rinv1GeVf / sin((betaIn + betaOut) / 2.);

            std::vector<int> hit_types;
            hit_types.push_back(4);
            hit_types.push_back(4);
            hit_types.push_back(4);
            hit_types.push_back(4);
            hit_types.push_back(4);
            hit_types.push_back(4);
            hit_types.push_back(4);
            hit_types.push_back(4);

            std::vector<int> module_idxs = {
                (int) hitsInGPU.moduleIndices[innerSegmentInnerMiniDoubletLowerHitIndex],
                (int) hitsInGPU.moduleIndices[innerSegmentInnerMiniDoubletUpperHitIndex],
                (int) hitsInGPU.moduleIndices[innerSegmentOuterMiniDoubletLowerHitIndex],
                (int) hitsInGPU.moduleIndices[innerSegmentOuterMiniDoubletUpperHitIndex],
                (int) hitsInGPU.moduleIndices[outerSegmentInnerMiniDoubletLowerHitIndex],
                (int) hitsInGPU.moduleIndices[outerSegmentInnerMiniDoubletUpperHitIndex],
                (int) hitsInGPU.moduleIndices[outerSegmentOuterMiniDoubletLowerHitIndex],
                (int) hitsInGPU.moduleIndices[outerSegmentOuterMiniDoubletUpperHitIndex],
            };

            int layer0 = modulesInGPU.layers[module_idxs[0]];
            int layer2 = modulesInGPU.layers[module_idxs[2]];
            int layer4 = modulesInGPU.layers[module_idxs[4]];
            int layer6 = modulesInGPU.layers[module_idxs[6]];

            int subdet0 = modulesInGPU.subdets[module_idxs[0]];
            int subdet2 = modulesInGPU.subdets[module_idxs[2]];
            int subdet4 = modulesInGPU.subdets[module_idxs[4]];
            int subdet6 = modulesInGPU.subdets[module_idxs[6]];

            int logicallayer0 = layer0 + 6 * (subdet0 == 4);
            int logicallayer2 = layer2 + 6 * (subdet2 == 4);
            int logicallayer4 = layer4 + 6 * (subdet4 == 4);
            int logicallayer6 = layer6 + 6 * (subdet6 == 4);

            int layer_binary = 0;
            layer_binary |= (1 << logicallayer0);
            layer_binary |= (1 << logicallayer2);
            layer_binary |= (1 << logicallayer4);
            layer_binary |= (1 << logicallayer6);

#ifdef CUT_VALUE_DEBUG
            int moduleType_binary = 0;
            int moduleType0 = modulesInGPU.moduleType[module_idxs[0]];
            int moduleType2 = modulesInGPU.moduleType[module_idxs[2]];
            int moduleType4 = modulesInGPU.moduleType[module_idxs[4]];
            int moduleType6 = modulesInGPU.moduleType[module_idxs[6]];
            
            moduleType_binary |= (moduleType0 << 0);
            moduleType_binary |= (moduleType2 << 2);
            moduleType_binary |= (moduleType4 << 4);
            moduleType_binary |= (moduleType6 << 6);
           
            layer_binaries.push_back(layer_binary);
            moduleType_binaries.push_back(moduleType_binary);

#endif


            // sim track matched index
            std::vector<int> matched_sim_trk_idxs = matchedSimTrkIdxs(hit_idxs, hit_types);

            for (auto &isimtrk : matched_sim_trk_idxs)
            {
                sim_T3_matched[isimtrk]++;
            }

            for (auto &isimtrk : matched_sim_trk_idxs)
            {
                sim_T3_types[isimtrk].push_back(layer_binary);
            }

            // Compute pt, eta, phi of T3
            const float pt = ptAv;
            float eta = -999;
            float phi = -999;
            SDL::CPU::Hit hitA(trk.ph2_x()[hit_idxs[0]], trk.ph2_y()[hit_idxs[0]], trk.ph2_z()[hit_idxs[0]]);
            SDL::CPU::Hit hitB(trk.ph2_x()[hit_idxs[7]], trk.ph2_y()[hit_idxs[7]], trk.ph2_z()[hit_idxs[7]]);
            eta = hitB.eta();
            phi = hitA.phi();

            t3_isFake.push_back(matched_sim_trk_idxs.size() == 0);
            t3_pt.push_back(pt);
            t3_eta.push_back(eta);
            t3_phi.push_back(phi);
            t3_matched_simIdx.push_back(matched_sim_trk_idxs);

#ifdef CUT_VALUE_DEBUG
            float zOut = tripletsInGPU.zOut[tripletIndex];
            float rtOut = tripletsInGPU.rtOut[tripletIndex];
            float deltaPhiPos = tripletsInGPU.deltaPhiPos[tripletIndex];
            float deltaPhi = tripletsInGPU.deltaPhi[tripletIndex];
            //betaIn and betaOut already defined!
            float deltaBeta = betaIn - betaOut;
            float zLo = tripletsInGPU.zLo[tripletIndex];
            float zHi = tripletsInGPU.zHi[tripletIndex];
            float rtLo = tripletsInGPU.rtLo[tripletIndex];
            float rtHi = tripletsInGPU.rtHi[tripletIndex];
            float kZ = tripletsInGPU.kZ[tripletIndex];
            float zLoPointed = tripletsInGPU.zLoPointed[tripletIndex];
            float zHiPointed = tripletsInGPU.zHiPointed[tripletIndex];
            float sdlCut = tripletsInGPU.sdlCut[tripletIndex];
            float betaInCut = tripletsInGPU.betaInCut[tripletIndex];
            float betaOutCut = tripletsInGPU.betaOutCut[tripletIndex];
            float deltaBetaCut = tripletsInGPU.deltaBetaCut[tripletIndex];

            t3_ZOut.push_back(zOut);
            t3_RtOut.push_back(rtOut);
            t3_deltaPhiPos.push_back(deltaPhiPos);
            t3_deltaPhi.push_back(deltaPhi);
            t3_betaIn.push_back(betaIn);
            t3_betaOut.push_back(betaOut);
            t3_deltaBeta.push_back(deltaBeta);
            t3_ZLo.push_back(zLo);
            t3_ZHi.push_back(zHi);
            t3_RtLo.push_back(rtLo);
            t3_RtHi.push_back(rtHi);
            t3_kZ.push_back(kZ);
            t3_ZLoPointed.push_back(zLoPointed);
            t3_ZHiPointed.push_back(zHiPointed);
            t3_sdlCut.push_back(sdlCut);
            t3_betaInCut.push_back(betaInCut);
            t3_betaOutCut.push_back(betaOutCut);
            t3_deltaBetaCut.push_back(deltaBetaCut);
#endif


        }

    }

    ana.tx->setBranch<vector<int>>("sim_T3_matched", sim_T3_matched);
    ana.tx->setBranch<vector<vector<int>>>("sim_T3_types", sim_T3_types);

    vector<int> t3_isDuplicate(t3_matched_simIdx.size());

    for (unsigned int i = 0; i < t3_matched_simIdx.size(); ++i)
    {
        bool isDuplicate = false;
        for (unsigned int isim = 0; isim < t3_matched_simIdx[i].size(); ++isim)
        {
            if (sim_T3_matched[t3_matched_simIdx[i][isim]] > 1)
            {
                isDuplicate = true;
            }
        }
        t3_isDuplicate[i] = isDuplicate;
    }

    ana.tx->setBranch<vector<float>>("t3_pt", t3_pt);
    ana.tx->setBranch<vector<float>>("t3_eta", t3_eta);
    ana.tx->setBranch<vector<float>>("t3_phi", t3_phi);
    ana.tx->setBranch<vector<int>>("t3_isFake", t3_isFake);
    ana.tx->setBranch<vector<int>>("t3_isDuplicate", t3_isDuplicate);

#ifdef CUT_VALUE_DEBUG
    ana.tx->setBranch<vector<float>>("t3_zOut",t3_ZOut);
    ana.tx->setBranch<vector<float>>("t3_rtOut",t3_RtOut);
    ana.tx->setBranch<vector<float>>("t3_deltaPhiPos",t3_deltaPhiPos);
    ana.tx->setBranch<vector<float>>("t3_deltaPhi",t3_deltaPhi);
    ana.tx->setBranch<vector<float>>("t3_betaIn",t3_betaIn);
    ana.tx->setBranch<vector<float>>("t3_betaOut",t3_betaOut);
    ana.tx->setBranch<vector<float>>("t3_deltaBeta",t3_deltaBeta);
    ana.tx->setBranch<vector<float>>("t3_zLo",t3_ZLo);
    ana.tx->setBranch<vector<float>>("t3_zHi",t3_ZHi);
    ana.tx->setBranch<vector<float>>("t3_rtLo",t3_RtLo);
    ana.tx->setBranch<vector<float>>("t3_rtHi",t3_RtHi);
    ana.tx->setBranch<vector<float>>("t3_kZ",t3_kZ);
    ana.tx->setBranch<vector<float>>("t3_zLoPointed",t3_ZLoPointed);
    ana.tx->setBranch<vector<float>>("t3_zHiPointed",t3_ZHiPointed);
    ana.tx->setBranch<vector<float>>("t3_sdlCut",t3_sdlCut);
    ana.tx->setBranch<vector<float>>("t3_betaInCut",t3_betaInCut);
    ana.tx->setBranch<vector<float>>("t3_betaOutCut",t3_betaOutCut);
    ana.tx->setBranch<vector<float>>("t3_deltaBetaCut",t3_deltaBetaCut);
    ana.tx->setBranch<vector<int>>("t3_layer_binary",layer_binaries);
    ana.tx->setBranch<vector<int>>("t3_moduleType_binary",moduleType_binaries);
#endif

}


//________________________________________________________________________________________________________________________________
void fillOutputBranches_for_CPU(SDL::CPU::Event& event)
{
    fillSimTrackOutputBranches();
    fillTrackCandidateOutputBranches_for_CPU(event);
    if (ana.do_lower_level)
    {
        fillLowerLevelOutputBranches_for_CPU(event);
    }

    ana.tx->fill();
    ana.tx->clear();

}

//________________________________________________________________________________________________________________________________
void fillTrackCandidateOutputBranches_for_CPU(SDL::CPU::Event& event)
{
    // Did it match to track candidate?
    std::vector<int> sim_TC_matched(trk.sim_pt().size());
    std::vector<vector<int>> sim_TC_types(trk.sim_pt().size());

    // get layer ptrs
    std::vector<SDL::CPU::Layer*> layerPtrs = event.getLayerPtrs();
    layerPtrs.push_back(&(event.getPixelLayer()));

    std::vector<int> tc_isFake;
    std::vector<vector<int>> tc_matched_simIdx;
    std::vector<float> tc_pt;
    std::vector<float> tc_eta;
    std::vector<float> tc_phi;

    // Loop over layers and access track candidates
    for (auto& layerPtr : layerPtrs)
    {

        // Track Candidate ptrs
        const std::vector<SDL::CPU::TrackCandidate*>& trackCandidatePtrs = layerPtr->getTrackCandidatePtrs();


        // Loop over trackCandidate ptrs
        for (auto& trackCandidatePtr : trackCandidatePtrs)
        {

            // hit idx
            std::vector<int> hit_idx;
            hit_idx.push_back(trackCandidatePtr->innerTrackletBasePtr()->innerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->idx());
            hit_idx.push_back(trackCandidatePtr->innerTrackletBasePtr()->innerSegmentPtr()->innerMiniDoubletPtr()->upperHitPtr()->idx());
            hit_idx.push_back(trackCandidatePtr->innerTrackletBasePtr()->innerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->idx());
            hit_idx.push_back(trackCandidatePtr->innerTrackletBasePtr()->innerSegmentPtr()->outerMiniDoubletPtr()->upperHitPtr()->idx());
            hit_idx.push_back(trackCandidatePtr->innerTrackletBasePtr()->outerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->idx());
            hit_idx.push_back(trackCandidatePtr->innerTrackletBasePtr()->outerSegmentPtr()->innerMiniDoubletPtr()->upperHitPtr()->idx());
            hit_idx.push_back(trackCandidatePtr->innerTrackletBasePtr()->outerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->idx());
            hit_idx.push_back(trackCandidatePtr->innerTrackletBasePtr()->outerSegmentPtr()->outerMiniDoubletPtr()->upperHitPtr()->idx());
            hit_idx.push_back(trackCandidatePtr->outerTrackletBasePtr()->innerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->idx());
            hit_idx.push_back(trackCandidatePtr->outerTrackletBasePtr()->innerSegmentPtr()->innerMiniDoubletPtr()->upperHitPtr()->idx());
            hit_idx.push_back(trackCandidatePtr->outerTrackletBasePtr()->innerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->idx());
            hit_idx.push_back(trackCandidatePtr->outerTrackletBasePtr()->innerSegmentPtr()->outerMiniDoubletPtr()->upperHitPtr()->idx());
            hit_idx.push_back(trackCandidatePtr->outerTrackletBasePtr()->outerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->idx());
            hit_idx.push_back(trackCandidatePtr->outerTrackletBasePtr()->outerSegmentPtr()->innerMiniDoubletPtr()->upperHitPtr()->idx());
            hit_idx.push_back(trackCandidatePtr->outerTrackletBasePtr()->outerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->idx());
            hit_idx.push_back(trackCandidatePtr->outerTrackletBasePtr()->outerSegmentPtr()->outerMiniDoubletPtr()->upperHitPtr()->idx());

            std::vector<int> hit_types;
            if (trackCandidatePtr->innerTrackletBasePtr()->innerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->getModule().detId() == 1)
            {
                hit_types.push_back(0);
                hit_types.push_back(0);
                hit_types.push_back(0);
                hit_types.push_back(0);
            }
            else
            {
                hit_types.push_back(4);
                hit_types.push_back(4);
                hit_types.push_back(4);
                hit_types.push_back(4);
            }

            hit_types.push_back(4);
            hit_types.push_back(4);
            hit_types.push_back(4);
            hit_types.push_back(4);
            hit_types.push_back(4);
            hit_types.push_back(4);
            hit_types.push_back(4);
            hit_types.push_back(4);
            hit_types.push_back(4);
            hit_types.push_back(4);
            hit_types.push_back(4);
            hit_types.push_back(4);

            // 0 -- 0
            //      0 -- 0
            //
            //           0 -- 0
            //                0 -- 0
            // 01  23   
            //     45   67
            //          89   1011
            //               1213  1415

            // 0 -- 0
            //      0 -- 0
            //
            //           0 -- 0
            //                0 -- 0
            // 0    2   
            //      4    6
            //           8   10  
            //               1213  1415

            const SDL::CPU::Module& module0  = trackCandidatePtr->innerTrackletBasePtr()->innerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->getModule();
            const SDL::CPU::Module& module2  = trackCandidatePtr->innerTrackletBasePtr()->innerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->getModule();
            const SDL::CPU::Module& module4  = trackCandidatePtr->innerTrackletBasePtr()->outerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->getModule();
            const SDL::CPU::Module& module6  = trackCandidatePtr->innerTrackletBasePtr()->outerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->getModule();
            const SDL::CPU::Module& module8  = trackCandidatePtr->outerTrackletBasePtr()->innerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->getModule();
            const SDL::CPU::Module& module10 = trackCandidatePtr->outerTrackletBasePtr()->innerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->getModule();
            const SDL::CPU::Module& module12 = trackCandidatePtr->outerTrackletBasePtr()->outerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->getModule();
            const SDL::CPU::Module& module14 = trackCandidatePtr->outerTrackletBasePtr()->outerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->getModule();

            bool isPixel0  = module0.isPixelLayerModule();
            bool isPixel2  = module2.isPixelLayerModule();
            bool isPixel4  = module4.isPixelLayerModule();
            bool isPixel6  = module6.isPixelLayerModule();
            bool isPixel8  = module8.isPixelLayerModule();
            bool isPixel10 = module10.isPixelLayerModule();
            bool isPixel12 = module12.isPixelLayerModule();
            bool isPixel14 = module14.isPixelLayerModule();

            int layer0  = module0.layer();
            int layer2  = module2.layer();
            int layer4  = module4.layer();
            int layer6  = module6.layer();
            int layer8  = module8.layer();
            int layer10 = module10.layer();
            int layer12 = module12.layer();
            int layer14 = module14.layer();

            int subdet0  = module0.subdet();
            int subdet2  = module2.subdet();
            int subdet4  = module4.subdet();
            int subdet6  = module6.subdet();
            int subdet8  = module8.subdet();
            int subdet10 = module10.subdet();
            int subdet12 = module12.subdet();
            int subdet14 = module14.subdet();

            int logicallayer0  = isPixel0  ? 0 : layer0  + 6 * (subdet0  == 4);
            int logicallayer2  = isPixel2  ? 0 : layer2  + 6 * (subdet2  == 4);
            int logicallayer4  = isPixel4  ? 0 : layer4  + 6 * (subdet4  == 4);
            int logicallayer6  = isPixel6  ? 0 : layer6  + 6 * (subdet6  == 4);
            int logicallayer8  = isPixel8  ? 0 : layer8  + 6 * (subdet8  == 4);
            int logicallayer10 = isPixel10 ? 0 : layer10 + 6 * (subdet10 == 4);
            int logicallayer12 = isPixel12 ? 0 : layer12 + 6 * (subdet12 == 4);
            int logicallayer14 = isPixel14 ? 0 : layer14 + 6 * (subdet14 == 4);

            int layer_binary = 0;
            layer_binary |= (1 << logicallayer0);
            layer_binary |= (1 << logicallayer2);
            layer_binary |= (1 << logicallayer4);
            layer_binary |= (1 << logicallayer6);
            layer_binary |= (1 << logicallayer8);
            layer_binary |= (1 << logicallayer10);
            layer_binary |= (1 << logicallayer12);
            layer_binary |= (1 << logicallayer14);

            // sim track matched index
            std::vector<int> matched_sim_trk_idxs = matchedSimTrkIdxs(hit_idx, hit_types);

            for (auto& isimtrk : matched_sim_trk_idxs)
            {
                sim_TC_matched[isimtrk]++;
            }

            for (auto& isimtrk : matched_sim_trk_idxs)
            {
                sim_TC_types[isimtrk].push_back(layer_binary);
            }

            // Compute pt, eta, phi of TC
            float eta = -999;
            float phi = -999;

            bool isInnerTrackletTriplet = (hit_idx[2] == hit_idx[4] and hit_idx[3] == hit_idx[5] and hit_types[2] == hit_types[4] and hit_types[3] == hit_types[5]);
            bool isOuterTrackletTriplet = (hit_idx[10] == hit_idx[12] and hit_idx[11] == hit_idx[13] and hit_types[10] == hit_types[12] and hit_types[11] == hit_types[13]);

            // // if (isInnerTrackletTriplet and isOuterTrackletTriplet)
            // if (true)
            // {
            //     std::cout << "here1" << std::endl;
            //     std::cout <<  " isInnerTrackletTriplet: " << isInnerTrackletTriplet <<  " isOuterTrackletTriplet: " << isOuterTrackletTriplet <<  std::endl;
            //     std::cout <<  " logicallayer0: " << logicallayer0 <<  " logicallayer2: " << logicallayer2 <<  " logicallayer4: " << logicallayer4 <<  " logicallayer6: " << logicallayer6 <<  " logicallayer8: " << logicallayer8 <<  " logicallayer10: " << logicallayer10 <<  std::endl;
            //     for (unsigned int ihit = 0; ihit < hit_idx.size(); ++ihit)
            //     {
            //         std::cout <<  " ihit: " << ihit <<  " hit_idx[ihit]: " << hit_idx[ihit] <<  std::endl;
            //     }
            //     for (unsigned int ihit = 0; ihit < hit_types.size(); ++ihit)
            //     {
            //         std::cout <<  " ihit: " << ihit <<  " hit_types[ihit]: " << hit_types[ihit] <<  std::endl;
            //     }
            //     std::cout << "recovar size: " << ((SDL::CPU::Triplet*) trackCandidatePtr->innerTrackletBasePtr())->tlCand.getRecoVars().size() << std::endl;
            //     for (auto& [k, v]: ((SDL::CPU::Triplet*) trackCandidatePtr->innerTrackletBasePtr())->tlCand.getRecoVars())
            //     {
            //         std::cout <<  " k: " << k <<  std::endl;
            //     }
            //     std::cout << "recovar size: " << ((SDL::CPU::Triplet*) trackCandidatePtr->outerTrackletBasePtr())->tlCand.getRecoVars().size() << std::endl;
            //     for (auto& [k, v]: ((SDL::CPU::Triplet*) trackCandidatePtr->outerTrackletBasePtr())->tlCand.getRecoVars())
            //     {
            //         std::cout <<  " k: " << k <<  std::endl;
            //     }
            // }
            float pt_in  = isInnerTrackletTriplet ? ((SDL::CPU::Triplet*) trackCandidatePtr->innerTrackletBasePtr())->tlCand.getRecoVar("pt_beta") : trackCandidatePtr->innerTrackletBasePtr()->getRecoVar("pt_beta");
            // std::cout <<  " pt_in: " << pt_in <<  std::endl;
            float pt_out = isOuterTrackletTriplet ? ((SDL::CPU::Triplet*) trackCandidatePtr->outerTrackletBasePtr())->tlCand.getRecoVar("pt_beta") : trackCandidatePtr->outerTrackletBasePtr()->getRecoVar("pt_beta");
            // std::cout <<  " pt_out: " << pt_out <<  std::endl;
            float pt = (pt_in + pt_out) / 2.;
            // std::cout << "here2" << std::endl;

            // float ptBetaIn_in = isInnerTrackletTriplet ? ((SDL::CPU::Triplet*) trackCandidatePtr->innerTrackletBasePtr())->tlCand.getRecoVar("pt_betaIn") : trackCandidatePtr->innerTrackletBasePtr()->getRecoVar("pt_betaIn");
            // float ptBetaOut_in = isInnerTrackletTriplet ? ((SDL::CPU::Triplet*) trackCandidatePtr->innerTrackletBasePtr())->tlCand.getRecoVar("pt_betaOut") : trackCandidatePtr->innerTrackletBasePtr()->getRecoVar("pt_betaOut");

            // if ((trackCandidatePtr->innerTrackletBasePtr()->innerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->getModule().detId() == 1))
            //     std::cout << " " << (trackCandidatePtr->innerTrackletBasePtr()->innerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->getModule().detId() == 1) <<  " " << hit_idx[0] <<  " " << hit_idx[1] <<  " " << hit_idx[2] <<  " " << hit_idx[3] <<  " " << hit_idx[4] <<  " " << hit_idx[5] <<  " " << hit_idx[6] <<  " " << hit_idx[7] <<  " " << hit_idx[8] <<  " " << hit_idx[9] <<  " " << hit_idx[10] <<  " " << hit_idx[11] <<  " pt_in: " << pt_in <<  " pt_out: " << pt_out <<  " ptBetaIn_in: " << ptBetaIn_in <<  " ptBetaOut_in: " << ptBetaOut_in << std::endl;

            if (hit_types[0] == 4)
            {
                SDL::CPU::Hit hitA(trk.ph2_x()[hit_idx[0]], trk.ph2_y()[hit_idx[0]], trk.ph2_z()[hit_idx[0]]);
                SDL::CPU::Hit hitB(trk.ph2_x()[hit_idx[15]], trk.ph2_y()[hit_idx[15]], trk.ph2_z()[hit_idx[15]]);
                eta = hitB.eta();
                phi = hitA.phi();
            }
            else
            {
                SDL::CPU::Hit hitA(trk.pix_x()[hit_idx[0]], trk.pix_y()[hit_idx[0]], trk.pix_z()[hit_idx[0]]);
                SDL::CPU::Hit hitB(trk.ph2_x()[hit_idx[15]], trk.ph2_y()[hit_idx[15]], trk.ph2_z()[hit_idx[15]]);
                eta = hitB.eta();
                phi = hitA.phi();
            }

            tc_isFake.push_back(matched_sim_trk_idxs.size() == 0);
            tc_pt.push_back(pt);
            tc_eta.push_back(eta);
            tc_phi.push_back(phi);
            tc_matched_simIdx.push_back(matched_sim_trk_idxs);

        }

    }

    ana.tx->setBranch<vector<int>>("sim_TC_matched", sim_TC_matched);
    ana.tx->setBranch<vector<vector<int>>>("sim_TC_types", sim_TC_types);

    vector<int> tc_isDuplicate(tc_matched_simIdx.size());

    for (unsigned int i = 0; i < tc_matched_simIdx.size(); ++i)
    {
        bool isDuplicate = false;
        for (unsigned int isim = 0; isim < tc_matched_simIdx[i].size(); ++isim)
        {
            if (sim_TC_matched[tc_matched_simIdx[i][isim]] > 1)
            {
                isDuplicate = true;
            }
        }
        tc_isDuplicate[i] = isDuplicate;
    }

    ana.tx->setBranch<vector<float>>("tc_pt", tc_pt);
    ana.tx->setBranch<vector<float>>("tc_eta", tc_eta);
    ana.tx->setBranch<vector<float>>("tc_phi", tc_phi);
    ana.tx->setBranch<vector<int>>("tc_isFake", tc_isFake);
    ana.tx->setBranch<vector<int>>("tc_isDuplicate", tc_isDuplicate);

}

//________________________________________________________________________________________________________________________________
void fillLowerLevelOutputBranches_for_CPU(SDL::CPU::Event& event)
{
    fillQuadrupletOutputBranches_for_CPU(event);
    fillTripletOutputBranches_for_CPU(event);
    fillPixelQuadrupletOutputBranches_for_CPU(event);
#ifdef DO_QUINTUPLET
    fillQuintupletOutputBranches_for_CPU(event);
#endif
}

//________________________________________________________________________________________________________________________________
void fillQuadrupletOutputBranches_for_CPU(SDL::CPU::Event& event)
{
    // Did it match to track candidate?
    std::vector<int> sim_T4_matched(trk.sim_pt().size());
    std::vector<vector<int>> sim_T4_types(trk.sim_pt().size());

    // get layer ptrs
    std::vector<SDL::CPU::Layer*> layerPtrs = event.getLayerPtrs();
    // layerPtrs.push_back(&(event.getPixelLayer()));

    std::vector<int> t4_isFake;
    std::vector<vector<int>> t4_matched_simIdx;
    std::vector<float> t4_pt;
    std::vector<float> t4_eta;
    std::vector<float> t4_phi;

    // Loop over layers and access track candidates
    for (auto& layerPtr : layerPtrs)
    {

        // Quadruplets ptrs
        const std::vector<SDL::CPU::Tracklet*>& trackletPtrs = layerPtr->getTrackletPtrs();


        // Loop over trackCandidate ptrs
        for (auto& trackletPtr : trackletPtrs)
        {

            // hit idx
            std::vector<int> hit_idx;
            hit_idx.push_back(trackletPtr->innerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->idx());
            hit_idx.push_back(trackletPtr->innerSegmentPtr()->innerMiniDoubletPtr()->upperHitPtr()->idx());
            hit_idx.push_back(trackletPtr->innerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->idx());
            hit_idx.push_back(trackletPtr->innerSegmentPtr()->outerMiniDoubletPtr()->upperHitPtr()->idx());
            hit_idx.push_back(trackletPtr->outerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->idx());
            hit_idx.push_back(trackletPtr->outerSegmentPtr()->innerMiniDoubletPtr()->upperHitPtr()->idx());
            hit_idx.push_back(trackletPtr->outerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->idx());
            hit_idx.push_back(trackletPtr->outerSegmentPtr()->outerMiniDoubletPtr()->upperHitPtr()->idx());

            std::vector<int> hit_types;
            hit_types.push_back(4);
            hit_types.push_back(4);
            hit_types.push_back(4);
            hit_types.push_back(4);
            hit_types.push_back(4);
            hit_types.push_back(4);
            hit_types.push_back(4);
            hit_types.push_back(4);

            const SDL::CPU::Module& module0 = trackletPtr->innerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->getModule();
            const SDL::CPU::Module& module2 = trackletPtr->innerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->getModule();
            const SDL::CPU::Module& module4 = trackletPtr->outerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->getModule();
            const SDL::CPU::Module& module6 = trackletPtr->outerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->getModule();

            bool isPixel0 = false;
            bool isPixel2 = false;
            bool isPixel4 = false;
            bool isPixel6 = false;

            int layer0 = module0.layer();
            int layer2 = module2.layer();
            int layer4 = module4.layer();
            int layer6 = module6.layer();

            int subdet0 = module0.subdet();
            int subdet2 = module2.subdet();
            int subdet4 = module4.subdet();
            int subdet6 = module6.subdet();

            int logicallayer0 = isPixel0 ? 0 : layer0  + 6 * (subdet0 == 4);
            int logicallayer2 = isPixel2 ? 0 : layer2  + 6 * (subdet2 == 4);
            int logicallayer4 = isPixel4 ? 0 : layer4  + 6 * (subdet4 == 4);
            int logicallayer6 = isPixel6 ? 0 : layer6  + 6 * (subdet6 == 4);

            int layer_binary = 0;
            layer_binary |= (1 << logicallayer0);
            layer_binary |= (1 << logicallayer2);
            layer_binary |= (1 << logicallayer4);
            layer_binary |= (1 << logicallayer6);

            // sim track matched index
            std::vector<int> matched_sim_trk_idxs = matchedSimTrkIdxs(hit_idx, hit_types);

            for (auto& isimtrk : matched_sim_trk_idxs)
            {
                sim_T4_matched[isimtrk]++;
            }

            for (auto& isimtrk : matched_sim_trk_idxs)
            {
                sim_T4_types[isimtrk].push_back(layer_binary);
            }

            // Compute pt, eta, phi of T4
            const float pt = trackletPtr->getRecoVar("pt_beta");
            float eta = -999;
            float phi = -999;
            SDL::CPU::Hit hitA(trk.ph2_x()[hit_idx[0]], trk.ph2_y()[hit_idx[0]], trk.ph2_z()[hit_idx[0]]);
            SDL::CPU::Hit hitB(trk.ph2_x()[hit_idx[7]], trk.ph2_y()[hit_idx[7]], trk.ph2_z()[hit_idx[7]]);
            eta = hitB.eta();
            phi = hitA.phi();

            t4_isFake.push_back(matched_sim_trk_idxs.size() == 0);
            t4_pt.push_back(pt);
            t4_eta.push_back(eta);
            t4_phi.push_back(phi);
            t4_matched_simIdx.push_back(matched_sim_trk_idxs);

        }

    }

    ana.tx->setBranch<vector<int>>("sim_T4_matched", sim_T4_matched);
    ana.tx->setBranch<vector<vector<int>>>("sim_T4_types", sim_T4_types);

    vector<int> t4_isDuplicate(t4_matched_simIdx.size());

    for (unsigned int i = 0; i < t4_matched_simIdx.size(); ++i)
    {
        bool isDuplicate = false;
        for (unsigned int isim = 0; isim < t4_matched_simIdx[i].size(); ++isim)
        {
            if (sim_T4_matched[t4_matched_simIdx[i][isim]] > 1)
            {
                isDuplicate = true;
            }
        }
        t4_isDuplicate[i] = isDuplicate;
    }

    ana.tx->setBranch<vector<float>>("t4_pt", t4_pt);
    ana.tx->setBranch<vector<float>>("t4_eta", t4_eta);
    ana.tx->setBranch<vector<float>>("t4_phi", t4_phi);
    ana.tx->setBranch<vector<int>>("t4_isFake", t4_isFake);
    ana.tx->setBranch<vector<int>>("t4_isDuplicate", t4_isDuplicate);

}

//________________________________________________________________________________________________________________________________
void fillTripletOutputBranches_for_CPU(SDL::CPU::Event& event)
{
    // Did it match to track candidate?
    std::vector<int> sim_T3_matched(trk.sim_pt().size());
    std::vector<vector<int>> sim_T3_types(trk.sim_pt().size());

    // get layer ptrs
    std::vector<SDL::CPU::Layer*> layerPtrs = event.getLayerPtrs();
    // layerPtrs.push_back(&(event.getPixelLayer()));

    std::vector<int> t3_isFake;
    std::vector<vector<int>> t3_matched_simIdx;
    std::vector<float> t3_pt;
    std::vector<float> t3_eta;
    std::vector<float> t3_phi;

    // Loop over layers and access track candidates
    for (auto& layerPtr : layerPtrs)
    {

        // Triplets ptrs
        const std::vector<SDL::CPU::Triplet*>& tripletPtrs = layerPtr->getTripletPtrs();


        // Loop over trackCandidate ptrs
        for (auto& tripletPtr : tripletPtrs)
        {

            // hit idx
            std::vector<int> hit_idx;
            hit_idx.push_back(tripletPtr->innerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->idx());
            hit_idx.push_back(tripletPtr->innerSegmentPtr()->innerMiniDoubletPtr()->upperHitPtr()->idx());
            hit_idx.push_back(tripletPtr->innerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->idx());
            hit_idx.push_back(tripletPtr->innerSegmentPtr()->outerMiniDoubletPtr()->upperHitPtr()->idx());
            hit_idx.push_back(tripletPtr->outerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->idx());
            hit_idx.push_back(tripletPtr->outerSegmentPtr()->innerMiniDoubletPtr()->upperHitPtr()->idx());
            hit_idx.push_back(tripletPtr->outerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->idx());
            hit_idx.push_back(tripletPtr->outerSegmentPtr()->outerMiniDoubletPtr()->upperHitPtr()->idx());

            std::vector<int> hit_types;
            hit_types.push_back(4);
            hit_types.push_back(4);
            hit_types.push_back(4);
            hit_types.push_back(4);
            hit_types.push_back(4);
            hit_types.push_back(4);
            hit_types.push_back(4);
            hit_types.push_back(4);

            const SDL::CPU::Module& module0 = tripletPtr->innerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->getModule();
            const SDL::CPU::Module& module2 = tripletPtr->innerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->getModule();
            const SDL::CPU::Module& module4 = tripletPtr->outerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->getModule();
            const SDL::CPU::Module& module6 = tripletPtr->outerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->getModule();

            bool isPixel0 = false;
            bool isPixel2 = false;
            bool isPixel4 = false;
            bool isPixel6 = false;

            int layer0 = module0.layer();
            int layer2 = module2.layer();
            int layer4 = module4.layer();
            int layer6 = module6.layer();

            int subdet0 = module0.subdet();
            int subdet2 = module2.subdet();
            int subdet3 = module4.subdet();
            int subdet6 = module6.subdet();

            int logicallayer0 = isPixel0 ? 0 : layer0  + 6 * (subdet0 == 4);
            int logicallayer2 = isPixel2 ? 0 : layer2  + 6 * (subdet2 == 4);
            int logicallayer4 = isPixel4 ? 0 : layer4  + 6 * (subdet3 == 4);
            int logicallayer6 = isPixel6 ? 0 : layer6  + 6 * (subdet6 == 4);

            int layer_binary = 0;
            layer_binary |= (1 << logicallayer0);
            layer_binary |= (1 << logicallayer2);
            layer_binary |= (1 << logicallayer4);
            layer_binary |= (1 << logicallayer6);

            // sim track matched index
            std::vector<int> matched_sim_trk_idxs = matchedSimTrkIdxs(hit_idx, hit_types);

            for (auto& isimtrk : matched_sim_trk_idxs)
            {
                sim_T3_matched[isimtrk]++;
            }

            for (auto& isimtrk : matched_sim_trk_idxs)
            {
                sim_T3_types[isimtrk].push_back(layer_binary);
            }

            // Compute pt, eta, phi of T3
            const float pt = tripletPtr->tlCand.getRecoVar("pt_beta");
            float eta = -999;
            float phi = -999;
            SDL::CPU::Hit hitA(trk.ph2_x()[hit_idx[0]], trk.ph2_y()[hit_idx[0]], trk.ph2_z()[hit_idx[0]]);
            SDL::CPU::Hit hitB(trk.ph2_x()[hit_idx[7]], trk.ph2_y()[hit_idx[7]], trk.ph2_z()[hit_idx[7]]);
            eta = hitB.eta();
            phi = hitA.phi();

            t3_isFake.push_back(matched_sim_trk_idxs.size() == 0);
            t3_pt.push_back(pt);
            t3_eta.push_back(eta);
            t3_phi.push_back(phi);
            t3_matched_simIdx.push_back(matched_sim_trk_idxs);

        }

    }

    ana.tx->setBranch<vector<int>>("sim_T3_matched", sim_T3_matched);
    ana.tx->setBranch<vector<vector<int>>>("sim_T3_types", sim_T3_types);

    vector<int> t3_isDuplicate(t3_matched_simIdx.size());

    for (unsigned int i = 0; i < t3_matched_simIdx.size(); ++i)
    {
        bool isDuplicate = false;
        for (unsigned int isim = 0; isim < t3_matched_simIdx[i].size(); ++isim)
        {
            if (sim_T3_matched[t3_matched_simIdx[i][isim]] > 1)
            {
                isDuplicate = true;
            }
        }
        t3_isDuplicate[i] = isDuplicate;
    }

    ana.tx->setBranch<vector<float>>("t3_pt", t3_pt);
    ana.tx->setBranch<vector<float>>("t3_eta", t3_eta);
    ana.tx->setBranch<vector<float>>("t3_phi", t3_phi);
    ana.tx->setBranch<vector<int>>("t3_isFake", t3_isFake);
    ana.tx->setBranch<vector<int>>("t3_isDuplicate", t3_isDuplicate);

}

//________________________________________________________________________________________________________________________________
void fillPixelQuadrupletOutputBranches_for_CPU(SDL::CPU::Event& event)
{
    // Did it match to track candidate?
    std::vector<int> sim_pT4_matched(trk.sim_pt().size());
    std::vector<vector<int>> sim_pT4_types(trk.sim_pt().size());

    // get layer ptrs
    std::vector<SDL::CPU::Layer*> layerPtrs;
    layerPtrs.push_back(&(event.getPixelLayer()));

    std::vector<int> pT4_isFake;
    std::vector<vector<int>> pT4_matched_simIdx;
    std::vector<float> pT4_pt;
    std::vector<float> pT4_eta;
    std::vector<float> pT4_phi;

    // Loop over layers and access track candidates
    for (auto& layerPtr : layerPtrs)
    {

        // Quadruplets ptrs
        const std::vector<SDL::CPU::Tracklet*>& trackletPtrs = layerPtr->getTrackletPtrs();


        // Loop over trackCandidate ptrs
        for (auto& trackletPtr : trackletPtrs)
        {

            // hit idx
            std::vector<int> hit_idx;
            hit_idx.push_back(trackletPtr->innerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->idx());
            hit_idx.push_back(trackletPtr->innerSegmentPtr()->innerMiniDoubletPtr()->upperHitPtr()->idx());
            hit_idx.push_back(trackletPtr->innerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->idx());
            hit_idx.push_back(trackletPtr->innerSegmentPtr()->outerMiniDoubletPtr()->upperHitPtr()->idx());
            hit_idx.push_back(trackletPtr->outerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->idx());
            hit_idx.push_back(trackletPtr->outerSegmentPtr()->innerMiniDoubletPtr()->upperHitPtr()->idx());
            hit_idx.push_back(trackletPtr->outerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->idx());
            hit_idx.push_back(trackletPtr->outerSegmentPtr()->outerMiniDoubletPtr()->upperHitPtr()->idx());

            std::vector<int> hit_types;
            hit_types.push_back(0);
            hit_types.push_back(0);
            hit_types.push_back(0);
            hit_types.push_back(0);
            hit_types.push_back(4);
            hit_types.push_back(4);
            hit_types.push_back(4);
            hit_types.push_back(4);

            const SDL::CPU::Module& module0 = trackletPtr->innerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->getModule();
            const SDL::CPU::Module& module2 = trackletPtr->innerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->getModule();
            const SDL::CPU::Module& module4 = trackletPtr->outerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->getModule();
            const SDL::CPU::Module& module6 = trackletPtr->outerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->getModule();

            const bool isPixel0 = true;
            const bool isPixel2 = true;
            const bool isPixel4 = false;
            const bool isPixel6 = false;

            const int layer0 = module0.layer();
            const int layer2 = module2.layer();
            const int layer4 = module4.layer();
            const int layer6 = module6.layer();

            const int subdet0 = module0.subdet();
            const int subdet2 = module2.subdet();
            const int subdet4 = module4.subdet();
            const int subdet6 = module6.subdet();

            const int logicallayer0 = isPixel0 ? 0 : layer0  + 6 * (subdet0 == 4);
            const int logicallayer2 = isPixel2 ? 0 : layer2  + 6 * (subdet2 == 4);
            const int logicallayer4 = isPixel4 ? 0 : layer4  + 6 * (subdet4 == 4);
            const int logicallayer6 = isPixel6 ? 0 : layer6  + 6 * (subdet6 == 4);

            int layer_binary = 0;
            layer_binary |= (1 << logicallayer0);
            layer_binary |= (1 << logicallayer2);
            layer_binary |= (1 << logicallayer4);
            layer_binary |= (1 << logicallayer6);

            // sim track matched index
            std::vector<int> matched_sim_trk_idxs = matchedSimTrkIdxs(hit_idx, hit_types);

            for (auto& isimtrk : matched_sim_trk_idxs)
            {
                sim_pT4_matched[isimtrk]++;
            }

            for (auto& isimtrk : matched_sim_trk_idxs)
            {
                sim_pT4_types[isimtrk].push_back(layer_binary);
            }

            // Compute pt, eta, phi of pT4
            const float pt = trackletPtr->getRecoVar("pt_beta");
            float eta = -999;
            float phi = -999;
            SDL::CPU::Hit hitA(trk.pix_x()[hit_idx[0]], trk.pix_y()[hit_idx[0]], trk.pix_z()[hit_idx[0]]);
            SDL::CPU::Hit hitB(trk.ph2_x()[hit_idx[7]], trk.ph2_y()[hit_idx[7]], trk.ph2_z()[hit_idx[7]]);
            eta = hitB.eta();
            phi = hitA.phi();

            pT4_isFake.push_back(matched_sim_trk_idxs.size() == 0);
            pT4_pt.push_back(pt);
            pT4_eta.push_back(eta);
            pT4_phi.push_back(phi);
            pT4_matched_simIdx.push_back(matched_sim_trk_idxs);

        }

    }

    ana.tx->setBranch<vector<int>>("sim_pT4_matched", sim_pT4_matched);
    ana.tx->setBranch<vector<vector<int>>>("sim_pT4_types", sim_pT4_types);

    vector<int> pT4_isDuplicate(pT4_matched_simIdx.size());

    for (unsigned int i = 0; i < pT4_matched_simIdx.size(); ++i)
    {
        bool isDuplicate = false;
        for (unsigned int isim = 0; isim < pT4_matched_simIdx[i].size(); ++isim)
        {
            if (sim_pT4_matched[pT4_matched_simIdx[i][isim]] > 1)
            {
                isDuplicate = true;
            }
        }
        pT4_isDuplicate[i] = isDuplicate;
    }

    ana.tx->setBranch<vector<float>>("pT4_pt", pT4_pt);
    ana.tx->setBranch<vector<float>>("pT4_eta", pT4_eta);
    ana.tx->setBranch<vector<float>>("pT4_phi", pT4_phi);
    ana.tx->setBranch<vector<int>>("pT4_isFake", pT4_isFake);
    ana.tx->setBranch<vector<int>>("pT4_isDuplicate", pT4_isDuplicate);

}

//________________________________________________________________________________________________________________________________
void fillQuintupletOutputBranches_for_CPU(SDL::CPU::Event& event)
{
    // Did it match to track candidate?
    std::vector<int> sim_T5_matched(trk.sim_pt().size());
    std::vector<vector<int>> sim_T5_types(trk.sim_pt().size());

    // get layer ptrs
    std::vector<SDL::CPU::Layer*> layerPtrs = event.getLayerPtrs();
    layerPtrs.push_back(&(event.getPixelLayer()));

    std::vector<int> t5_isFake;
    std::vector<vector<int>> t5_matched_simIdx;
    std::vector<float> t5_pt;
    std::vector<float> t5_eta;
    std::vector<float> t5_phi;

    // Loop over layers and access track candidates
    for (auto& layerPtr : layerPtrs)
    {

        // Track Candidate ptrs
        const std::vector<SDL::CPU::TrackCandidate*>& trackCandidatePtrs = layerPtr->getTrackCandidatePtrs();


        // Loop over trackCandidate ptrs
        for (auto& trackCandidatePtr : trackCandidatePtrs)
        {

            // hit idx
            std::vector<int> hit_idx;
            hit_idx.push_back(trackCandidatePtr->innerTrackletBasePtr()->innerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->idx());
            hit_idx.push_back(trackCandidatePtr->innerTrackletBasePtr()->innerSegmentPtr()->innerMiniDoubletPtr()->upperHitPtr()->idx());
            hit_idx.push_back(trackCandidatePtr->innerTrackletBasePtr()->innerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->idx());
            hit_idx.push_back(trackCandidatePtr->innerTrackletBasePtr()->innerSegmentPtr()->outerMiniDoubletPtr()->upperHitPtr()->idx());
            hit_idx.push_back(trackCandidatePtr->innerTrackletBasePtr()->outerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->idx());
            hit_idx.push_back(trackCandidatePtr->innerTrackletBasePtr()->outerSegmentPtr()->innerMiniDoubletPtr()->upperHitPtr()->idx());
            hit_idx.push_back(trackCandidatePtr->innerTrackletBasePtr()->outerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->idx());
            hit_idx.push_back(trackCandidatePtr->innerTrackletBasePtr()->outerSegmentPtr()->outerMiniDoubletPtr()->upperHitPtr()->idx());
            hit_idx.push_back(trackCandidatePtr->outerTrackletBasePtr()->innerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->idx());
            hit_idx.push_back(trackCandidatePtr->outerTrackletBasePtr()->innerSegmentPtr()->innerMiniDoubletPtr()->upperHitPtr()->idx());
            hit_idx.push_back(trackCandidatePtr->outerTrackletBasePtr()->innerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->idx());
            hit_idx.push_back(trackCandidatePtr->outerTrackletBasePtr()->innerSegmentPtr()->outerMiniDoubletPtr()->upperHitPtr()->idx());
            hit_idx.push_back(trackCandidatePtr->outerTrackletBasePtr()->outerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->idx());
            hit_idx.push_back(trackCandidatePtr->outerTrackletBasePtr()->outerSegmentPtr()->innerMiniDoubletPtr()->upperHitPtr()->idx());
            hit_idx.push_back(trackCandidatePtr->outerTrackletBasePtr()->outerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->idx());
            hit_idx.push_back(trackCandidatePtr->outerTrackletBasePtr()->outerSegmentPtr()->outerMiniDoubletPtr()->upperHitPtr()->idx());

            std::vector<int> hit_types;
            if (trackCandidatePtr->innerTrackletBasePtr()->innerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->getModule().detId() == 1)
            {
                hit_types.push_back(0);
                hit_types.push_back(0);
                hit_types.push_back(0);
                hit_types.push_back(0);
            }
            else
            {
                hit_types.push_back(4);
                hit_types.push_back(4);
                hit_types.push_back(4);
                hit_types.push_back(4);
            }

            hit_types.push_back(4);
            hit_types.push_back(4);
            hit_types.push_back(4);
            hit_types.push_back(4);
            hit_types.push_back(4);
            hit_types.push_back(4);
            hit_types.push_back(4);
            hit_types.push_back(4);
            hit_types.push_back(4);
            hit_types.push_back(4);
            hit_types.push_back(4);
            hit_types.push_back(4);

            // 0 -- 0
            //      0 -- 0
            //
            //           0 -- 0
            //                0 -- 0
            // 01  23   
            //     45   67
            //          89   1011
            //               1213  1415

            // 0 -- 0
            //      0 -- 0
            //
            //           0 -- 0
            //                0 -- 0
            // 0    2   
            //      4    6
            //           8   10  
            //               1213  1415

            bool isInnerTrackletTriplet = (hit_idx[2] == hit_idx[4] and hit_idx[3] == hit_idx[5] and hit_types[2] == hit_types[4] and hit_types[3] == hit_types[5]);
            bool isOuterTrackletTriplet = (hit_idx[10] == hit_idx[12] and hit_idx[11] == hit_idx[13] and hit_types[10] == hit_types[12] and hit_types[11] == hit_types[13]);
            bool isMiddleTrackletTriplet = (hit_idx[6] == hit_idx[8] and hit_idx[7] == hit_idx[9] and hit_types[6] == hit_types[8] and hit_types[7] == hit_types[9]);
            bool isT5 = isInnerTrackletTriplet and isOuterTrackletTriplet and isMiddleTrackletTriplet;

            if (not isT5)
                continue;

            const SDL::CPU::Module& module0  = trackCandidatePtr->innerTrackletBasePtr()->innerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->getModule();
            const SDL::CPU::Module& module2  = trackCandidatePtr->innerTrackletBasePtr()->innerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->getModule();
            const SDL::CPU::Module& module4  = trackCandidatePtr->innerTrackletBasePtr()->outerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->getModule();
            const SDL::CPU::Module& module6  = trackCandidatePtr->innerTrackletBasePtr()->outerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->getModule();
            const SDL::CPU::Module& module8  = trackCandidatePtr->outerTrackletBasePtr()->innerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->getModule();
            const SDL::CPU::Module& module10 = trackCandidatePtr->outerTrackletBasePtr()->innerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->getModule();
            const SDL::CPU::Module& module12 = trackCandidatePtr->outerTrackletBasePtr()->outerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->getModule();
            const SDL::CPU::Module& module14 = trackCandidatePtr->outerTrackletBasePtr()->outerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->getModule();

            bool isPixel0  = module0.isPixelLayerModule();
            bool isPixel2  = module2.isPixelLayerModule();
            bool isPixel4  = module4.isPixelLayerModule();
            bool isPixel6  = module6.isPixelLayerModule();
            bool isPixel8  = module8.isPixelLayerModule();
            bool isPixel10 = module10.isPixelLayerModule();
            bool isPixel12 = module12.isPixelLayerModule();
            bool isPixel14 = module14.isPixelLayerModule();

            int layer0  = module0.layer();
            int layer2  = module2.layer();
            int layer4  = module4.layer();
            int layer6  = module6.layer();
            int layer8  = module8.layer();
            int layer10 = module10.layer();
            int layer12 = module12.layer();
            int layer14 = module14.layer();

            int subdet0  = module0.subdet();
            int subdet2  = module2.subdet();
            int subdet4  = module4.subdet();
            int subdet6  = module6.subdet();
            int subdet8  = module8.subdet();
            int subdet10 = module10.subdet();
            int subdet12 = module12.subdet();
            int subdet14 = module14.subdet();

            int logicallayer0  = isPixel0  ? 0 : layer0  + 6 * (subdet0  == 4);
            int logicallayer2  = isPixel2  ? 0 : layer2  + 6 * (subdet2  == 4);
            int logicallayer4  = isPixel4  ? 0 : layer4  + 6 * (subdet4  == 4);
            int logicallayer6  = isPixel6  ? 0 : layer6  + 6 * (subdet6  == 4);
            int logicallayer8  = isPixel8  ? 0 : layer8  + 6 * (subdet8  == 4);
            int logicallayer10 = isPixel10 ? 0 : layer10 + 6 * (subdet10 == 4);
            int logicallayer12 = isPixel12 ? 0 : layer12 + 6 * (subdet12 == 4);
            int logicallayer14 = isPixel14 ? 0 : layer14 + 6 * (subdet14 == 4);

            int layer_binary = 0;
            layer_binary |= (1 << logicallayer0);
            layer_binary |= (1 << logicallayer2);
            layer_binary |= (1 << logicallayer4);
            layer_binary |= (1 << logicallayer6);
            layer_binary |= (1 << logicallayer8);
            layer_binary |= (1 << logicallayer10);
            layer_binary |= (1 << logicallayer12);
            layer_binary |= (1 << logicallayer14);

            // sim track matched index
            std::vector<int> matched_sim_trk_idxs = matchedSimTrkIdxs(hit_idx, hit_types);

            for (auto& isimtrk : matched_sim_trk_idxs)
            {
                sim_T5_matched[isimtrk]++;
            }

            for (auto& isimtrk : matched_sim_trk_idxs)
            {
                sim_T5_types[isimtrk].push_back(layer_binary);
            }

            // Compute pt, eta, phi of T5
            float eta = -999;
            float phi = -999;

            // // if (isInnerTrackletTriplet and isOuterTrackletTriplet)
            // if (true)
            // {
            //     std::cout << "here1" << std::endl;
            //     std::cout <<  " isInnerTrackletTriplet: " << isInnerTrackletTriplet <<  " isOuterTrackletTriplet: " << isOuterTrackletTriplet <<  std::endl;
            //     std::cout <<  " logicallayer0: " << logicallayer0 <<  " logicallayer2: " << logicallayer2 <<  " logicallayer4: " << logicallayer4 <<  " logicallayer6: " << logicallayer6 <<  " logicallayer8: " << logicallayer8 <<  " logicallayer10: " << logicallayer10 <<  std::endl;
            //     for (unsigned int ihit = 0; ihit < hit_idx.size(); ++ihit)
            //     {
            //         std::cout <<  " ihit: " << ihit <<  " hit_idx[ihit]: " << hit_idx[ihit] <<  std::endl;
            //     }
            //     for (unsigned int ihit = 0; ihit < hit_types.size(); ++ihit)
            //     {
            //         std::cout <<  " ihit: " << ihit <<  " hit_types[ihit]: " << hit_types[ihit] <<  std::endl;
            //     }
            //     std::cout << "recovar size: " << ((SDL::CPU::Triplet*) trackCandidatePtr->innerTrackletBasePtr())->tlCand.getRecoVars().size() << std::endl;
            //     for (auto& [k, v]: ((SDL::CPU::Triplet*) trackCandidatePtr->innerTrackletBasePtr())->tlCand.getRecoVars())
            //     {
            //         std::cout <<  " k: " << k <<  std::endl;
            //     }
            //     std::cout << "recovar size: " << ((SDL::CPU::Triplet*) trackCandidatePtr->outerTrackletBasePtr())->tlCand.getRecoVars().size() << std::endl;
            //     for (auto& [k, v]: ((SDL::CPU::Triplet*) trackCandidatePtr->outerTrackletBasePtr())->tlCand.getRecoVars())
            //     {
            //         std::cout <<  " k: " << k <<  std::endl;
            //     }
            // }
            float pt_in  = isInnerTrackletTriplet ? ((SDL::CPU::Triplet*) trackCandidatePtr->innerTrackletBasePtr())->tlCand.getRecoVar("pt_beta") : trackCandidatePtr->innerTrackletBasePtr()->getRecoVar("pt_beta");
            // std::cout <<  " pt_in: " << pt_in <<  std::endl;
            float pt_out = isOuterTrackletTriplet ? ((SDL::CPU::Triplet*) trackCandidatePtr->outerTrackletBasePtr())->tlCand.getRecoVar("pt_beta") : trackCandidatePtr->outerTrackletBasePtr()->getRecoVar("pt_beta");
            // std::cout <<  " pt_out: " << pt_out <<  std::endl;
            float pt = (pt_in + pt_out) / 2.;
            // std::cout << "here2" << std::endl;

            // float ptBetaIn_in = isInnerTrackletTriplet ? ((SDL::CPU::Triplet*) trackCandidatePtr->innerTrackletBasePtr())->tlCand.getRecoVar("pt_betaIn") : trackCandidatePtr->innerTrackletBasePtr()->getRecoVar("pt_betaIn");
            // float ptBetaOut_in = isInnerTrackletTriplet ? ((SDL::CPU::Triplet*) trackCandidatePtr->innerTrackletBasePtr())->tlCand.getRecoVar("pt_betaOut") : trackCandidatePtr->innerTrackletBasePtr()->getRecoVar("pt_betaOut");

            // if ((trackCandidatePtr->innerTrackletBasePtr()->innerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->getModule().detId() == 1))
            //     std::cout << " " << (trackCandidatePtr->innerTrackletBasePtr()->innerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->getModule().detId() == 1) <<  " " << hit_idx[0] <<  " " << hit_idx[1] <<  " " << hit_idx[2] <<  " " << hit_idx[3] <<  " " << hit_idx[4] <<  " " << hit_idx[5] <<  " " << hit_idx[6] <<  " " << hit_idx[7] <<  " " << hit_idx[8] <<  " " << hit_idx[9] <<  " " << hit_idx[10] <<  " " << hit_idx[11] <<  " pt_in: " << pt_in <<  " pt_out: " << pt_out <<  " ptBetaIn_in: " << ptBetaIn_in <<  " ptBetaOut_in: " << ptBetaOut_in << std::endl;

            if (hit_types[0] == 4)
            {
                SDL::CPU::Hit hitA(trk.ph2_x()[hit_idx[0]], trk.ph2_y()[hit_idx[0]], trk.ph2_z()[hit_idx[0]]);
                SDL::CPU::Hit hitB(trk.ph2_x()[hit_idx[15]], trk.ph2_y()[hit_idx[15]], trk.ph2_z()[hit_idx[15]]);
                eta = hitB.eta();
                phi = hitA.phi();
            }
            else
            {
                SDL::CPU::Hit hitA(trk.pix_x()[hit_idx[0]], trk.pix_y()[hit_idx[0]], trk.pix_z()[hit_idx[0]]);
                SDL::CPU::Hit hitB(trk.ph2_x()[hit_idx[15]], trk.ph2_y()[hit_idx[15]], trk.ph2_z()[hit_idx[15]]);
                eta = hitB.eta();
                phi = hitA.phi();
            }

            t5_isFake.push_back(matched_sim_trk_idxs.size() == 0);
            t5_pt.push_back(pt);
            t5_eta.push_back(eta);
            t5_phi.push_back(phi);
            t5_matched_simIdx.push_back(matched_sim_trk_idxs);

        }

    }

    ana.tx->setBranch<vector<int>>("sim_T5_matched", sim_T5_matched);
    ana.tx->setBranch<vector<vector<int>>>("sim_T5_types", sim_T5_types);

    vector<int> t5_isDuplicate(t5_matched_simIdx.size());

    for (unsigned int i = 0; i < t5_matched_simIdx.size(); ++i)
    {
        bool isDuplicate = false;
        for (unsigned int isim = 0; isim < t5_matched_simIdx[i].size(); ++isim)
        {
            if (sim_T5_matched[t5_matched_simIdx[i][isim]] > 1)
            {
                isDuplicate = true;
            }
        }
        t5_isDuplicate[i] = isDuplicate;
    }

    ana.tx->setBranch<vector<float>>("t5_pt", t5_pt);
    ana.tx->setBranch<vector<float>>("t5_eta", t5_eta);
    ana.tx->setBranch<vector<float>>("t5_phi", t5_phi);
    ana.tx->setBranch<vector<int>>("t5_isFake", t5_isFake);
    ana.tx->setBranch<vector<int>>("t5_isDuplicate", t5_isDuplicate);

}

//________________________________________________________________________________________________________________________________
void printTimingInformation(std::vector<std::vector<float>>& timing_information)
{

    if (ana.verbose == 0)
        return;

    std::cout << showpoint;
    std::cout << fixed;
    std::cout << setprecision(2);
    std::cout << right;
    std::cout << "Timing summary" << std::endl;
    std::cout << "Evt     Hits         MD       LS      T4      T4x       pT4        T3       TC       T5       pT3        Total" << std::endl;
    std::vector<float> timing_sum_information(timing_information[0].size());
    for (auto&& [ievt, timing] : iter::enumerate(timing_information))
    {
        float timing_total = 0.f;
        timing_total += timing[0]*1000; // Hits
        timing_total += timing[1]*1000; // MD
        timing_total += timing[2]*1000; // LS
        timing_total += timing[3]*1000; // T4
        timing_total += timing[4]*1000; // T4x
        timing_total += timing[5]*1000; // pT4
        timing_total += timing[6]*1000; // T3
        timing_total += timing[7]*1000; // TC
        timing_total += timing[8]*1000; //T5
        timing_total += timing[9]*1000; //pT3
        std::cout << setw(6) << ievt;
        std::cout << "   "<<setw(6) << timing[0]*1000; // Hits
        std::cout << "   "<<setw(6) << timing[1]*1000; // MD
        std::cout << "   "<<setw(6) << timing[2]*1000; // LS
        std::cout << "   "<<setw(6) << timing[3]*1000; // T4
        std::cout << "   "<<setw(6) << timing[4]*1000; // T4x
        std::cout << "   "<<setw(6) << timing[5]*1000; // pT4
        std::cout << "   "<<setw(6) << timing[6]*1000; // T3
        std::cout << "   "<<setw(6) << timing[7]*1000; // TC
        std::cout << "   "<<setw(6) << timing[8]*1000; //T5
        std::cout << "   "<<setw(6) << timing[9]*1000; //pT3
        std::cout << "   "<<setw(7) << timing_total; // Total time
        std::cout << std::endl;
        timing_sum_information[0] += timing[0]*1000; // Hits
        timing_sum_information[1] += timing[1]*1000; // MD
        timing_sum_information[2] += timing[2]*1000; // LS
        timing_sum_information[3] += timing[3]*1000; // T4
        timing_sum_information[4] += timing[4]*1000; // T4x
        timing_sum_information[5] += timing[5]*1000; // pT4
        timing_sum_information[6] += timing[6]*1000; // T3
        timing_sum_information[7] += timing[7]*1000; // TC
        timing_sum_information[8] += timing[8]*1000; // T5
        timing_sum_information[9] += timing[9]*1000; //pT3
    }
    timing_sum_information[0] /= timing_information.size(); // Hits
    timing_sum_information[1] /= timing_information.size(); // MD
    timing_sum_information[2] /= timing_information.size(); // LS
    timing_sum_information[3] /= timing_information.size(); // T4
    timing_sum_information[4] /= timing_information.size(); // T4x
    timing_sum_information[5] /= timing_information.size(); // pT4
    timing_sum_information[6] /= timing_information.size(); // T3
    timing_sum_information[7] /= timing_information.size(); // TC
    timing_sum_information[8] /= timing_information.size(); //T5
    timing_sum_information[9] /= timing_information.size(); //pT3
    float timing_total_avg = 0.f;
    timing_total_avg += timing_sum_information[0]; // Hits
    timing_total_avg += timing_sum_information[1]; // MD
    timing_total_avg += timing_sum_information[2]; // LS
    timing_total_avg += timing_sum_information[3]; // T4
    timing_total_avg += timing_sum_information[4]; // T4x
    timing_total_avg += timing_sum_information[5]; // pT4
    timing_total_avg += timing_sum_information[6]; // T3
    timing_total_avg += timing_sum_information[7]; // TC
    timing_total_avg += timing_sum_information[8]; //T5
    timing_total_avg += timing_sum_information[9]; //pT3
    std::cout << setprecision(0);
    std::cout << setw(6) << "avg";
    std::cout << "   "<<setw(6) << timing_sum_information[0]; // Hits
    std::cout << "   "<<setw(6) << timing_sum_information[1]; // MD
    std::cout << "   "<<setw(6) << timing_sum_information[2]; // LS
    std::cout << "   "<<setw(6) << timing_sum_information[3]; // T4
    std::cout << "   "<<setw(6) << timing_sum_information[4]; // T4x
    std::cout << "   "<<setw(6) << timing_sum_information[5]; // pT4
    std::cout << "   "<<setw(6) << timing_sum_information[6]; // T3
    std::cout << "   "<<setw(6) << timing_sum_information[7]; // TC
    std::cout << "   "<<setw(6) << timing_sum_information[8]; //T5
    std::cout << "   "<<setw(6) << timing_sum_information[9]; //pT3
    std::cout << "   "<<setw(7) << timing_total_avg; // Average total time
    std::cout << "   "<<ana.compilation_target;
    std::cout << std::endl;

    std::cout << left;

}

//________________________________________________________________________________________________________________________________
void printQuadrupletMultiplicities(SDL::Event& event)
{
    SDL::tracklets& trackletsInGPU = (*event.getTracklets());
    SDL::modules& modulesInGPU = (*event.getModules());

    int nTracklets = 0;
    //for (unsigned int idx = 0; idx <= *(SDL::modulesInGPU->nLowerModules); idx++) // "<=" because cheating to include pixel track candidate lower module
    for (unsigned int idx = 0; idx <= *(modulesInGPU.nLowerModules); idx++) // "<=" because cheating to include pixel track candidate lower module
    {
        nTracklets += trackletsInGPU.nTracklets[idx];
    }
    std::cout <<  " nTracklets: " << nTracklets <<  std::endl;
}

//________________________________________________________________________________________________________________________________
void printHitMultiplicities(SDL::Event& event)
{
    SDL::hits& hitsInGPU = (*event.getHits());
    SDL::modules& modulesInGPU = (*event.getModules());

    int nHits = 0;
    for (unsigned int idx = 0; idx <= *(modulesInGPU.nLowerModules); idx++) // "<=" because cheating to include pixel track candidate lower module
    {
        nHits += modulesInGPU.hitRanges[4 * idx + 1] - modulesInGPU.hitRanges[4 * idx] + 1;       
        nHits += modulesInGPU.hitRanges[4 * idx + 3] - modulesInGPU.hitRanges[4 * idx + 2] + 1;
    }
    std::cout <<  " nHits: " << nHits <<  std::endl;
}

//________________________________________________________________________________________________________________________________
void printMiniDoubletMultiplicities(SDL::Event& event)
{
    SDL::miniDoublets& miniDoubletsInGPU = (*event.getMiniDoublets());
    SDL::modules& modulesInGPU = (*event.getModules());

    int nMiniDoublets = 0;
    for (unsigned int idx = 0; idx <= *(modulesInGPU.nModules); idx++) // "<=" because cheating to include pixel track candidate lower module
    {
        if(modulesInGPU.isLower[idx])
        {
            nMiniDoublets += miniDoubletsInGPU.nMDs[idx];
        }
    }
    std::cout <<  " nMiniDoublets: " << nMiniDoublets <<  std::endl;
}

//________________________________________________________________________________________________________________________________
void printAllObjects(SDL::Event& event)
{
    printMDs(event);
    printLSs(event);
    printpLSs(event);
    printT3s(event);
    printT4s(event);
    printpT4s(event);
    printTCs(event);
}

//________________________________________________________________________________________________________________________________
void printAllObjects_for_CPU(SDL::CPU::Event& event)
{
    printMDs_for_CPU(event);
    printLSs_for_CPU(event);
    printpLSs_for_CPU(event);
    printT3s_for_CPU(event);
    printT4s_for_CPU(event);
    printpT4s_for_CPU(event);
    printTCs_for_CPU(event);
}

//________________________________________________________________________________________________________________________________
void printpT4s(SDL::Event& event)
{
    SDL::tracklets& trackletsInGPU = (*event.getTracklets());
    SDL::triplets& tripletsInGPU = (*event.getTriplets());
    SDL::segments& segmentsInGPU = (*event.getSegments());
    SDL::miniDoublets& miniDoubletsInGPU = (*event.getMiniDoublets());
    SDL::hits& hitsInGPU = (*event.getHits());
    SDL::modules& modulesInGPU = (*event.getModules());
    for (unsigned int itl = 0; itl < trackletsInGPU.nTracklets[*(modulesInGPU.nLowerModules)]; ++itl)
    {

        unsigned int trackletIndex = (*(modulesInGPU.nLowerModules)) * 8000/*_N_MAX_TRACK_CANDIDATES_PER_MODULE*/ + itl;
        unsigned int InnerSegmentIndex = trackletsInGPU.segmentIndices[2 * trackletIndex];
        unsigned int OuterSegmentIndex = trackletsInGPU.segmentIndices[2 * trackletIndex + 1];

        unsigned int InnerSegmentInnerMiniDoubletIndex = segmentsInGPU.mdIndices[2 * InnerSegmentIndex];
        unsigned int InnerSegmentOuterMiniDoubletIndex = segmentsInGPU.mdIndices[2 * InnerSegmentIndex + 1];
        unsigned int OuterSegmentInnerMiniDoubletIndex = segmentsInGPU.mdIndices[2 * OuterSegmentIndex];
        unsigned int OuterSegmentOuterMiniDoubletIndex = segmentsInGPU.mdIndices[2 * OuterSegmentIndex + 1];

        unsigned int InnerSegmentInnerMiniDoubletLowerHitIndex = miniDoubletsInGPU.hitIndices[2 * InnerSegmentInnerMiniDoubletIndex];
        unsigned int InnerSegmentInnerMiniDoubletUpperHitIndex = miniDoubletsInGPU.hitIndices[2 * InnerSegmentInnerMiniDoubletIndex + 1];
        unsigned int InnerSegmentOuterMiniDoubletLowerHitIndex = miniDoubletsInGPU.hitIndices[2 * InnerSegmentOuterMiniDoubletIndex];
        unsigned int InnerSegmentOuterMiniDoubletUpperHitIndex = miniDoubletsInGPU.hitIndices[2 * InnerSegmentOuterMiniDoubletIndex + 1];
        unsigned int OuterSegmentInnerMiniDoubletLowerHitIndex = miniDoubletsInGPU.hitIndices[2 * OuterSegmentInnerMiniDoubletIndex];
        unsigned int OuterSegmentInnerMiniDoubletUpperHitIndex = miniDoubletsInGPU.hitIndices[2 * OuterSegmentInnerMiniDoubletIndex + 1];
        unsigned int OuterSegmentOuterMiniDoubletLowerHitIndex = miniDoubletsInGPU.hitIndices[2 * OuterSegmentOuterMiniDoubletIndex];
        unsigned int OuterSegmentOuterMiniDoubletUpperHitIndex = miniDoubletsInGPU.hitIndices[2 * OuterSegmentOuterMiniDoubletIndex + 1];

        unsigned int hit0 = hitsInGPU.idxs[InnerSegmentInnerMiniDoubletLowerHitIndex];
        unsigned int hit1 = hitsInGPU.idxs[InnerSegmentInnerMiniDoubletUpperHitIndex];
        unsigned int hit2 = hitsInGPU.idxs[InnerSegmentOuterMiniDoubletLowerHitIndex];
        unsigned int hit3 = hitsInGPU.idxs[InnerSegmentOuterMiniDoubletUpperHitIndex];
        unsigned int hit4 = hitsInGPU.idxs[OuterSegmentInnerMiniDoubletLowerHitIndex];
        unsigned int hit5 = hitsInGPU.idxs[OuterSegmentInnerMiniDoubletUpperHitIndex];
        unsigned int hit6 = hitsInGPU.idxs[OuterSegmentOuterMiniDoubletLowerHitIndex];
        unsigned int hit7 = hitsInGPU.idxs[OuterSegmentOuterMiniDoubletUpperHitIndex];

        std::cout <<  "VALIDATION 'pT4': " << "pT4" <<  " hit0: " << hit0 <<  " hit1: " << hit1 <<  " hit2: " << hit2 <<  " hit3: " << hit3 <<  " hit4: " << hit4 <<  " hit5: " << hit5 <<  " hit6: " << hit6 <<  " hit7: " << hit7 <<  std::endl;

    }
}

//________________________________________________________________________________________________________________________________
void printpT4s_for_CPU(SDL::CPU::Event& event)
{
    // pixelLayer
    const SDL::CPU::Layer& pixelLayer = event.getPixelLayer();

    // Quadruplet ptrs
    const std::vector<SDL::CPU::Tracklet*>& trackletPtrs = pixelLayer.getTrackletPtrs();

    // Loop over tracklet ptrs
    for (auto& trackletPtr : trackletPtrs)
    {

        // hit idx
        unsigned int hit0 = trackletPtr->innerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->idx();
        unsigned int hit1 = trackletPtr->innerSegmentPtr()->innerMiniDoubletPtr()->upperHitPtr()->idx();
        unsigned int hit2 = trackletPtr->innerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->idx();
        unsigned int hit3 = trackletPtr->innerSegmentPtr()->outerMiniDoubletPtr()->upperHitPtr()->idx();
        unsigned int hit4 = trackletPtr->outerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->idx();
        unsigned int hit5 = trackletPtr->outerSegmentPtr()->innerMiniDoubletPtr()->upperHitPtr()->idx();
        unsigned int hit6 = trackletPtr->outerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->idx();
        unsigned int hit7 = trackletPtr->outerSegmentPtr()->outerMiniDoubletPtr()->upperHitPtr()->idx();

        std::cout <<  "VALIDATION 'pT4': " << "pT4" <<  " hit0: " << hit0 <<  " hit1: " << hit1 <<  " hit2: " << hit2 <<  " hit3: " << hit3 <<  " hit4: " << hit4 <<  " hit5: " << hit5 <<  " hit6: " << hit6 <<  " hit7: " << hit7 <<  std::endl;

    }
}

//________________________________________________________________________________________________________________________________
void printMDs(SDL::Event& event)
{
    SDL::miniDoublets& miniDoubletsInGPU = (*event.getMiniDoublets());
    SDL::hits& hitsInGPU = (*event.getHits());
    SDL::modules& modulesInGPU = (*event.getModules());
    for (unsigned int idx = 0; idx <= *(modulesInGPU.nLowerModules); ++idx)
    {
        for (unsigned int jdx = 0; jdx < miniDoubletsInGPU.nMDs[2*idx]; jdx++)
        {
            unsigned int mdIdx = (2*idx) * 100 + jdx;
            unsigned int LowerHitIndex = miniDoubletsInGPU.hitIndices[2 * mdIdx];
            unsigned int UpperHitIndex = miniDoubletsInGPU.hitIndices[2 * mdIdx + 1];
            unsigned int hit0 = hitsInGPU.idxs[LowerHitIndex];
            unsigned int hit1 = hitsInGPU.idxs[UpperHitIndex];
            std::cout <<  "VALIDATION 'MD': " << "MD" <<  " hit0: " << hit0 <<  " hit1: " << hit1 <<  std::endl;
        }
        for (unsigned int jdx = 0; jdx < miniDoubletsInGPU.nMDs[2*idx+1]; jdx++)
        {
            unsigned int mdIdx = (2*idx+1) * 100 + jdx;
            unsigned int LowerHitIndex = miniDoubletsInGPU.hitIndices[2 * mdIdx];
            unsigned int UpperHitIndex = miniDoubletsInGPU.hitIndices[2 * mdIdx + 1];
            unsigned int hit0 = hitsInGPU.idxs[LowerHitIndex];
            unsigned int hit1 = hitsInGPU.idxs[UpperHitIndex];
            std::cout <<  "VALIDATION 'MD': " << "MD" <<  " hit0: " << hit0 <<  " hit1: " << hit1 <<  std::endl;
        }
    }
}

//________________________________________________________________________________________________________________________________
void printMDs_for_CPU(SDL::CPU::Event& event)
{
    // get layer ptrs
    const std::vector<SDL::CPU::Layer*> layerPtrs = event.getLayerPtrs();

    // Loop over layers and access minidoublets
    for (auto& layerPtr : layerPtrs)
    {

        // MiniDoublet ptrs
        const std::vector<SDL::CPU::MiniDoublet*>& minidoubletPtrs = layerPtr->getMiniDoubletPtrs();

        // Loop over minidoublet ptrs
        for (auto& minidoubletPtr : minidoubletPtrs)
        {

            // hit idx
            unsigned int hit0 = minidoubletPtr->lowerHitPtr()->idx();
            unsigned int hit1 = minidoubletPtr->upperHitPtr()->idx();

            std::cout <<  "VALIDATION 'MD': " << "MD" <<  " hit0: " << hit0 <<  " hit1: " << hit1 <<  std::endl;

        }

    }

}

//________________________________________________________________________________________________________________________________
void printLSs(SDL::Event& event)
{
    SDL::segments& segmentsInGPU = (*event.getSegments());
    SDL::miniDoublets& miniDoubletsInGPU = (*event.getMiniDoublets());
    SDL::hits& hitsInGPU = (*event.getHits());
    SDL::modules& modulesInGPU = (*event.getModules());
    int nSegments = 0;
    for (unsigned int i = 0; i <  *(modulesInGPU.nLowerModules); ++i)
    {
        unsigned int idx = modulesInGPU.lowerModuleIndices[i];
        nSegments += segmentsInGPU.nSegments[idx];
        for (unsigned int jdx = 0; jdx < segmentsInGPU.nSegments[idx]; jdx++)
        {
            unsigned int sgIdx = idx * 600 + jdx;
            unsigned int InnerMiniDoubletIndex = segmentsInGPU.mdIndices[2 * sgIdx];
            unsigned int OuterMiniDoubletIndex = segmentsInGPU.mdIndices[2 * sgIdx + 1];
            unsigned int InnerMiniDoubletLowerHitIndex = miniDoubletsInGPU.hitIndices[2 * InnerMiniDoubletIndex];
            unsigned int InnerMiniDoubletUpperHitIndex = miniDoubletsInGPU.hitIndices[2 * InnerMiniDoubletIndex + 1];
            unsigned int OuterMiniDoubletLowerHitIndex = miniDoubletsInGPU.hitIndices[2 * OuterMiniDoubletIndex];
            unsigned int OuterMiniDoubletUpperHitIndex = miniDoubletsInGPU.hitIndices[2 * OuterMiniDoubletIndex + 1];
            unsigned int hit0 = hitsInGPU.idxs[InnerMiniDoubletLowerHitIndex];
            unsigned int hit1 = hitsInGPU.idxs[InnerMiniDoubletUpperHitIndex];
            unsigned int hit2 = hitsInGPU.idxs[OuterMiniDoubletLowerHitIndex];
            unsigned int hit3 = hitsInGPU.idxs[OuterMiniDoubletUpperHitIndex];
            std::cout <<  "VALIDATION 'LS': " << "LS" <<  " hit0: " << hit0 <<  " hit1: " << hit1 <<  " hit2: " << hit2 <<  " hit3: " << hit3 <<  std::endl;
        }
    }
    std::cout <<  "VALIDATION nSegments: " << nSegments <<  std::endl;
}

//________________________________________________________________________________________________________________________________
void printLSs_for_CPU(SDL::CPU::Event& event)
{
    // get layer ptrs
    const std::vector<SDL::CPU::Layer*> layerPtrs = event.getLayerPtrs();

    // Loop over layers and access segments
    for (auto& layerPtr : layerPtrs)
    {

        // MiniDoublet ptrs
        const std::vector<SDL::CPU::Segment*>& segmentPtrs = layerPtr->getSegmentPtrs();

        // Loop over segment ptrs
        for (auto& segmentPtr : segmentPtrs)
        {

            // hit idx
            unsigned int hit0 = segmentPtr->innerMiniDoubletPtr()->lowerHitPtr()->idx();
            unsigned int hit1 = segmentPtr->innerMiniDoubletPtr()->upperHitPtr()->idx();
            unsigned int hit2 = segmentPtr->outerMiniDoubletPtr()->lowerHitPtr()->idx();
            unsigned int hit3 = segmentPtr->outerMiniDoubletPtr()->upperHitPtr()->idx();

            std::cout <<  "VALIDATION 'LS': " << "LS" <<  " hit0: " << hit0 <<  " hit1: " << hit1 <<  " hit2: " << hit2 <<  " hit3: " << hit3 <<  std::endl;

        }

    }

}

//________________________________________________________________________________________________________________________________
void printpLSs(SDL::Event& event)
{
    SDL::segments& segmentsInGPU = (*event.getSegments());
    SDL::miniDoublets& miniDoubletsInGPU = (*event.getMiniDoublets());
    SDL::hits& hitsInGPU = (*event.getHits());
    SDL::modules& modulesInGPU = (*event.getModules());
    unsigned int i = *(modulesInGPU.nLowerModules);
    unsigned int idx = modulesInGPU.lowerModuleIndices[i];
    int npLS = segmentsInGPU.nSegments[idx];
    for (unsigned int jdx = 0; jdx < segmentsInGPU.nSegments[idx]; jdx++)
    {
        unsigned int sgIdx = idx * 600 + jdx;
        unsigned int InnerMiniDoubletIndex = segmentsInGPU.mdIndices[2 * sgIdx];
        unsigned int OuterMiniDoubletIndex = segmentsInGPU.mdIndices[2 * sgIdx + 1];
        unsigned int InnerMiniDoubletLowerHitIndex = miniDoubletsInGPU.hitIndices[2 * InnerMiniDoubletIndex];
        unsigned int InnerMiniDoubletUpperHitIndex = miniDoubletsInGPU.hitIndices[2 * InnerMiniDoubletIndex + 1];
        unsigned int OuterMiniDoubletLowerHitIndex = miniDoubletsInGPU.hitIndices[2 * OuterMiniDoubletIndex];
        unsigned int OuterMiniDoubletUpperHitIndex = miniDoubletsInGPU.hitIndices[2 * OuterMiniDoubletIndex + 1];
        unsigned int hit0 = hitsInGPU.idxs[InnerMiniDoubletLowerHitIndex];
        unsigned int hit1 = hitsInGPU.idxs[InnerMiniDoubletUpperHitIndex];
        unsigned int hit2 = hitsInGPU.idxs[OuterMiniDoubletLowerHitIndex];
        unsigned int hit3 = hitsInGPU.idxs[OuterMiniDoubletUpperHitIndex];
        std::cout <<  "VALIDATION 'pLS': " << "pLS" <<  " hit0: " << hit0 <<  " hit1: " << hit1 <<  " hit2: " << hit2 <<  " hit3: " << hit3 <<  std::endl;
    }
    std::cout <<  "VALIDATION npLS: " << npLS <<  std::endl;
}

//________________________________________________________________________________________________________________________________
void printpLSs_for_CPU(SDL::CPU::Event& event)
{
    // get layer ptr
    SDL::CPU::Layer* layerPtr = &(event.getPixelLayer());

    // MiniDoublet ptrs
    const std::vector<SDL::CPU::Segment*>& segmentPtrs = layerPtr->getSegmentPtrs();

    // Number of pLS
    int npLS = segmentPtrs.size();

    // Loop over segment ptrs
    for (auto& segmentPtr : segmentPtrs)
    {

        // hit idx
        unsigned int hit0 = segmentPtr->innerMiniDoubletPtr()->lowerHitPtr()->idx();
        unsigned int hit1 = segmentPtr->innerMiniDoubletPtr()->upperHitPtr()->idx();
        unsigned int hit2 = segmentPtr->outerMiniDoubletPtr()->lowerHitPtr()->idx();
        unsigned int hit3 = segmentPtr->outerMiniDoubletPtr()->upperHitPtr()->idx();

        std::cout <<  "VALIDATION 'pLS': " << "pLS" <<  " hit0: " << hit0 <<  " hit1: " << hit1 <<  " hit2: " << hit2 <<  " hit3: " << hit3 <<  std::endl;

    }
    std::cout <<  "VALIDATION npLS: " << npLS <<  std::endl;

}

//________________________________________________________________________________________________________________________________
void printT3s(SDL::Event& event)
{
    SDL::triplets& tripletsInGPU = (*event.getTriplets());
    SDL::segments& segmentsInGPU = (*event.getSegments());
    SDL::miniDoublets& miniDoubletsInGPU = (*event.getMiniDoublets());
    SDL::hits& hitsInGPU = (*event.getHits());
    SDL::modules& modulesInGPU = (*event.getModules());
    int nTriplets = 0;
    for (unsigned int i = 0; i <  *(modulesInGPU.nLowerModules); ++i)
    {
        // unsigned int idx = SDL::modulesInGPU->lowerModuleIndices[i];
        nTriplets += tripletsInGPU.nTriplets[i];
        unsigned int idx = i;
        for (unsigned int jdx = 0; jdx < tripletsInGPU.nTriplets[idx]; jdx++)
        {
            unsigned int tpIdx = idx * 5000 + jdx;
            unsigned int InnerSegmentIndex = tripletsInGPU.segmentIndices[2 * tpIdx];
            unsigned int OuterSegmentIndex = tripletsInGPU.segmentIndices[2 * tpIdx + 1];
            unsigned int InnerSegmentInnerMiniDoubletIndex = segmentsInGPU.mdIndices[2 * InnerSegmentIndex];
            unsigned int InnerSegmentOuterMiniDoubletIndex = segmentsInGPU.mdIndices[2 * InnerSegmentIndex + 1];
            unsigned int OuterSegmentOuterMiniDoubletIndex = segmentsInGPU.mdIndices[2 * OuterSegmentIndex + 1];

            unsigned int hit_idx0 = miniDoubletsInGPU.hitIndices[2 * InnerSegmentInnerMiniDoubletIndex];
            unsigned int hit_idx1 = miniDoubletsInGPU.hitIndices[2 * InnerSegmentInnerMiniDoubletIndex + 1];
            unsigned int hit_idx2 = miniDoubletsInGPU.hitIndices[2 * InnerSegmentOuterMiniDoubletIndex];
            unsigned int hit_idx3 = miniDoubletsInGPU.hitIndices[2 * InnerSegmentOuterMiniDoubletIndex + 1];
            unsigned int hit_idx4 = miniDoubletsInGPU.hitIndices[2 * OuterSegmentOuterMiniDoubletIndex];
            unsigned int hit_idx5 = miniDoubletsInGPU.hitIndices[2 * OuterSegmentOuterMiniDoubletIndex + 1];

            unsigned int hit0 = hitsInGPU.idxs[hit_idx0];
            unsigned int hit1 = hitsInGPU.idxs[hit_idx1];
            unsigned int hit2 = hitsInGPU.idxs[hit_idx2];
            unsigned int hit3 = hitsInGPU.idxs[hit_idx3];
            unsigned int hit4 = hitsInGPU.idxs[hit_idx4];
            unsigned int hit5 = hitsInGPU.idxs[hit_idx5];
            std::cout <<  "VALIDATION 'T3': " << "T3" <<  " hit0: " << hit0 <<  " hit1: " << hit1 <<  " hit2: " << hit2 <<  " hit3: " << hit3 <<  " hit4: " << hit4 <<  " hit5: " << hit5 <<  std::endl;
        }
    }
    std::cout <<  "VALIDATION nTriplets: " << nTriplets <<  std::endl;
}

//________________________________________________________________________________________________________________________________
void printT3s_for_CPU(SDL::CPU::Event& event)
{
    // get layer ptrs
    const std::vector<SDL::CPU::Layer*> layerPtrs = event.getLayerPtrs();

    // Loop over layers and access triplets
    for (auto& layerPtr : layerPtrs)
    {

        // MiniDoublet ptrs
        const std::vector<SDL::CPU::Triplet*>& tripletPtrs = layerPtr->getTripletPtrs();

        // Loop over triplet ptrs
        for (auto& tripletPtr : tripletPtrs)
        {

            // hit idx
            unsigned int hit0 = tripletPtr->innerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->idx();
            unsigned int hit1 = tripletPtr->innerSegmentPtr()->innerMiniDoubletPtr()->upperHitPtr()->idx();
            unsigned int hit2 = tripletPtr->innerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->idx();
            unsigned int hit3 = tripletPtr->innerSegmentPtr()->outerMiniDoubletPtr()->upperHitPtr()->idx();
            unsigned int hit4 = tripletPtr->outerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->idx();
            unsigned int hit5 = tripletPtr->outerSegmentPtr()->outerMiniDoubletPtr()->upperHitPtr()->idx();

            std::cout <<  "VALIDATION 'T3': " << "T3" <<  " hit0: " << hit0 <<  " hit1: " << hit1 <<  " hit2: " << hit2 <<  " hit3: " << hit3 <<  " hit4: " << hit4 <<  " hit5: " << hit5 <<  std::endl;

        }

    }

}

//________________________________________________________________________________________________________________________________
void printT4s(SDL::Event& event)
{
    SDL::tracklets& trackletsInGPU = (*event.getTracklets());
    SDL::segments& segmentsInGPU = (*event.getSegments());
    SDL::miniDoublets& miniDoubletsInGPU = (*event.getMiniDoublets());
    SDL::hits& hitsInGPU = (*event.getHits());
    SDL::modules& modulesInGPU = (*event.getModules());
    int nTracklets = 0;
    for (unsigned int i = 0; i <  *(modulesInGPU.nLowerModules); ++i)
    {
        // unsigned int idx = SDL::modulesInGPU->lowerModuleIndices[i];
        // nTracklets += trackletsInGPU.nTracklets[idx];
        nTracklets += trackletsInGPU.nTracklets[i];
        unsigned int idx = i;
        for (unsigned int jdx = 0; jdx < trackletsInGPU.nTracklets[idx]; jdx++)
        {
            unsigned int tlIdx = idx * 8000 + jdx;
            unsigned int InnerSegmentIndex = trackletsInGPU.segmentIndices[2 * tlIdx];
            unsigned int OuterSegmentIndex = trackletsInGPU.segmentIndices[2 * tlIdx + 1];
            unsigned int InnerSegmentInnerMiniDoubletIndex = segmentsInGPU.mdIndices[2 * InnerSegmentIndex];
            unsigned int InnerSegmentOuterMiniDoubletIndex = segmentsInGPU.mdIndices[2 * InnerSegmentIndex + 1];
            unsigned int OuterSegmentInnerMiniDoubletIndex = segmentsInGPU.mdIndices[2 * OuterSegmentIndex];
            unsigned int OuterSegmentOuterMiniDoubletIndex = segmentsInGPU.mdIndices[2 * OuterSegmentIndex + 1];

            unsigned int hit_idx0 = miniDoubletsInGPU.hitIndices[2 * InnerSegmentInnerMiniDoubletIndex];
            unsigned int hit_idx1 = miniDoubletsInGPU.hitIndices[2 * InnerSegmentInnerMiniDoubletIndex + 1];
            unsigned int hit_idx2 = miniDoubletsInGPU.hitIndices[2 * InnerSegmentOuterMiniDoubletIndex];
            unsigned int hit_idx3 = miniDoubletsInGPU.hitIndices[2 * InnerSegmentOuterMiniDoubletIndex + 1];
            unsigned int hit_idx4 = miniDoubletsInGPU.hitIndices[2 * OuterSegmentInnerMiniDoubletIndex];
            unsigned int hit_idx5 = miniDoubletsInGPU.hitIndices[2 * OuterSegmentInnerMiniDoubletIndex + 1];
            unsigned int hit_idx6 = miniDoubletsInGPU.hitIndices[2 * OuterSegmentOuterMiniDoubletIndex];
            unsigned int hit_idx7 = miniDoubletsInGPU.hitIndices[2 * OuterSegmentOuterMiniDoubletIndex + 1];

            unsigned int hit0 = hitsInGPU.idxs[hit_idx0];
            unsigned int hit1 = hitsInGPU.idxs[hit_idx1];
            unsigned int hit2 = hitsInGPU.idxs[hit_idx2];
            unsigned int hit3 = hitsInGPU.idxs[hit_idx3];
            unsigned int hit4 = hitsInGPU.idxs[hit_idx4];
            unsigned int hit5 = hitsInGPU.idxs[hit_idx5];
            unsigned int hit6 = hitsInGPU.idxs[hit_idx6];
            unsigned int hit7 = hitsInGPU.idxs[hit_idx7];
            std::cout <<  "VALIDATION 'T4': " << "T4" <<  " hit0: " << hit0 <<  " hit1: " << hit1 <<  " hit2: " << hit2 <<  " hit3: " << hit3 <<  " hit4: " << hit4 <<  " hit5: " << hit5 <<  " hit6: " << hit6 <<  " hit7: " << hit7 <<  std::endl;
        }
    }
    std::cout <<  "VALIDATION nTracklets: " << nTracklets <<  std::endl;
}

//________________________________________________________________________________________________________________________________
void printT4s_for_CPU(SDL::CPU::Event& event)
{
    // get layer ptrs
    const std::vector<SDL::CPU::Layer*> layerPtrs = event.getLayerPtrs();

    // Loop over layers and access tracklets
    for (auto& layerPtr : layerPtrs)
    {

        // MiniDoublet ptrs
        const std::vector<SDL::CPU::Tracklet*>& trackletPtrs = layerPtr->getTrackletPtrs();

        // Loop over tracklet ptrs
        for (auto& trackletPtr : trackletPtrs)
        {

            // hit idx
            unsigned int hit0 = trackletPtr->innerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->idx();
            unsigned int hit1 = trackletPtr->innerSegmentPtr()->innerMiniDoubletPtr()->upperHitPtr()->idx();
            unsigned int hit2 = trackletPtr->innerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->idx();
            unsigned int hit3 = trackletPtr->innerSegmentPtr()->outerMiniDoubletPtr()->upperHitPtr()->idx();
            unsigned int hit4 = trackletPtr->outerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->idx();
            unsigned int hit5 = trackletPtr->outerSegmentPtr()->innerMiniDoubletPtr()->upperHitPtr()->idx();
            unsigned int hit6 = trackletPtr->outerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->idx();
            unsigned int hit7 = trackletPtr->outerSegmentPtr()->outerMiniDoubletPtr()->upperHitPtr()->idx();

            std::cout <<  "VALIDATION 'T4': " << "T4" <<  " hit0: " << hit0 <<  " hit1: " << hit1 <<  " hit2: " << hit2 <<  " hit3: " << hit3 <<  " hit4: " << hit4 <<  " hit5: " << hit5 <<  " hit6: " << hit6 <<  " hit7: " << hit7 <<  std::endl;

        }

    }

}

//________________________________________________________________________________________________________________________________
void printTCs(SDL::Event& event)
{
    SDL::trackCandidates& trackCandidatesInGPU = (*event.getTrackCandidates());
    SDL::tracklets& trackletsInGPU = (*event.getTracklets());
    SDL::triplets& tripletsInGPU = (*event.getTriplets());
    SDL::segments& segmentsInGPU = (*event.getSegments());
    SDL::miniDoublets& miniDoubletsInGPU = (*event.getMiniDoublets());
    SDL::hits& hitsInGPU = (*event.getHits());
    SDL::modules& modulesInGPU = (*event.getModules());
    int nTrackCandidates = 0;
    for (unsigned int idx = 0; idx <= *(modulesInGPU.nLowerModules); ++idx)
    {
            if(modulesInGPU.trackCandidateModuleIndices[idx] == -1)
                continue;
        for (unsigned int jdx = 0; jdx < trackCandidatesInGPU.nTrackCandidates[idx]; jdx++)
        {
            //unsigned int trackCandidateIndex = idx * 50000/*_N_MAX_TRACK_CANDIDATES_PER_MODULE*/ + jdx;
            unsigned int trackCandidateIndex = modulesInGPU.trackCandidateModuleIndices[idx] + jdx;

            short trackCandidateType = trackCandidatesInGPU.trackCandidateType[trackCandidateIndex];

            unsigned int innerTrackletIdx = trackCandidatesInGPU.objectIndices[2 * trackCandidateIndex];
            unsigned int outerTrackletIdx = trackCandidatesInGPU.objectIndices[2 * trackCandidateIndex + 1];

            unsigned int innerTrackletInnerSegmentIndex = -1;
            unsigned int innerTrackletOuterSegmentIndex = -1;
            unsigned int outerTrackletOuterSegmentIndex = -1;

            if (trackCandidateType == 0) // T4T4
            {
                innerTrackletInnerSegmentIndex = trackletsInGPU.segmentIndices[2 * innerTrackletIdx];
                innerTrackletOuterSegmentIndex = trackletsInGPU.segmentIndices[2 * innerTrackletIdx + 1];
                outerTrackletOuterSegmentIndex = trackletsInGPU.segmentIndices[2 * outerTrackletIdx + 1];
            }
            else if (trackCandidateType == 1) // T4T3
            {
                innerTrackletInnerSegmentIndex = trackletsInGPU.segmentIndices[2 * innerTrackletIdx];
                innerTrackletOuterSegmentIndex = trackletsInGPU.segmentIndices[2 * innerTrackletIdx + 1];
                outerTrackletOuterSegmentIndex = tripletsInGPU.segmentIndices[2 * outerTrackletIdx + 1];
            }
            else if (trackCandidateType == 2) // T3T4
            {
                innerTrackletInnerSegmentIndex = tripletsInGPU.segmentIndices[2 * innerTrackletIdx];
                innerTrackletOuterSegmentIndex = tripletsInGPU.segmentIndices[2 * innerTrackletIdx + 1];
                outerTrackletOuterSegmentIndex = trackletsInGPU.segmentIndices[2 * outerTrackletIdx + 1];
            }

            unsigned int innerTrackletInnerSegmentInnerMiniDoubletIndex = segmentsInGPU.mdIndices[2 * innerTrackletInnerSegmentIndex];
            unsigned int innerTrackletInnerSegmentOuterMiniDoubletIndex = segmentsInGPU.mdIndices[2 * innerTrackletInnerSegmentIndex + 1];
            unsigned int innerTrackletOuterSegmentInnerMiniDoubletIndex = segmentsInGPU.mdIndices[2 * innerTrackletOuterSegmentIndex];
            unsigned int innerTrackletOuterSegmentOuterMiniDoubletIndex = segmentsInGPU.mdIndices[2 * innerTrackletOuterSegmentIndex + 1];
            unsigned int outerTrackletOuterSegmentInnerMiniDoubletIndex = segmentsInGPU.mdIndices[2 * outerTrackletOuterSegmentIndex];
            unsigned int outerTrackletOuterSegmentOuterMiniDoubletIndex = segmentsInGPU.mdIndices[2 * outerTrackletOuterSegmentIndex + 1];

            unsigned int innerTrackletInnerSegmentInnerMiniDoubletLowerHitIndex = miniDoubletsInGPU.hitIndices[2 * innerTrackletInnerSegmentInnerMiniDoubletIndex];
            unsigned int innerTrackletInnerSegmentInnerMiniDoubletUpperHitIndex = miniDoubletsInGPU.hitIndices[2 * innerTrackletInnerSegmentInnerMiniDoubletIndex + 1];
            unsigned int innerTrackletInnerSegmentOuterMiniDoubletLowerHitIndex = miniDoubletsInGPU.hitIndices[2 * innerTrackletInnerSegmentOuterMiniDoubletIndex];
            unsigned int innerTrackletInnerSegmentOuterMiniDoubletUpperHitIndex = miniDoubletsInGPU.hitIndices[2 * innerTrackletInnerSegmentOuterMiniDoubletIndex + 1];
            unsigned int innerTrackletOuterSegmentInnerMiniDoubletLowerHitIndex = miniDoubletsInGPU.hitIndices[2 * innerTrackletOuterSegmentInnerMiniDoubletIndex];
            unsigned int innerTrackletOuterSegmentInnerMiniDoubletUpperHitIndex = miniDoubletsInGPU.hitIndices[2 * innerTrackletOuterSegmentInnerMiniDoubletIndex + 1];
            unsigned int innerTrackletOuterSegmentOuterMiniDoubletLowerHitIndex = miniDoubletsInGPU.hitIndices[2 * innerTrackletOuterSegmentOuterMiniDoubletIndex];
            unsigned int innerTrackletOuterSegmentOuterMiniDoubletUpperHitIndex = miniDoubletsInGPU.hitIndices[2 * innerTrackletOuterSegmentOuterMiniDoubletIndex + 1];
            unsigned int outerTrackletOuterSegmentInnerMiniDoubletLowerHitIndex = miniDoubletsInGPU.hitIndices[2 * outerTrackletOuterSegmentInnerMiniDoubletIndex];
            unsigned int outerTrackletOuterSegmentInnerMiniDoubletUpperHitIndex = miniDoubletsInGPU.hitIndices[2 * outerTrackletOuterSegmentInnerMiniDoubletIndex + 1];
            unsigned int outerTrackletOuterSegmentOuterMiniDoubletLowerHitIndex = miniDoubletsInGPU.hitIndices[2 * outerTrackletOuterSegmentOuterMiniDoubletIndex];
            unsigned int outerTrackletOuterSegmentOuterMiniDoubletUpperHitIndex = miniDoubletsInGPU.hitIndices[2 * outerTrackletOuterSegmentOuterMiniDoubletIndex + 1];

            unsigned int hit0 = hitsInGPU.idxs[innerTrackletInnerSegmentInnerMiniDoubletLowerHitIndex];
            unsigned int hit1 = hitsInGPU.idxs[innerTrackletInnerSegmentInnerMiniDoubletUpperHitIndex];
            unsigned int hit2 = hitsInGPU.idxs[innerTrackletInnerSegmentOuterMiniDoubletLowerHitIndex];
            unsigned int hit3 = hitsInGPU.idxs[innerTrackletInnerSegmentOuterMiniDoubletUpperHitIndex];
            unsigned int hit4 = hitsInGPU.idxs[innerTrackletOuterSegmentInnerMiniDoubletLowerHitIndex];
            unsigned int hit5 = hitsInGPU.idxs[innerTrackletOuterSegmentInnerMiniDoubletUpperHitIndex];
            unsigned int hit6 = hitsInGPU.idxs[innerTrackletOuterSegmentOuterMiniDoubletLowerHitIndex];
            unsigned int hit7 = hitsInGPU.idxs[innerTrackletOuterSegmentOuterMiniDoubletUpperHitIndex];
            unsigned int hit8 = hitsInGPU.idxs[outerTrackletOuterSegmentInnerMiniDoubletLowerHitIndex];
            unsigned int hit9 = hitsInGPU.idxs[outerTrackletOuterSegmentInnerMiniDoubletUpperHitIndex];
            unsigned int hit10 = hitsInGPU.idxs[outerTrackletOuterSegmentOuterMiniDoubletLowerHitIndex];
            unsigned int hit11 = hitsInGPU.idxs[outerTrackletOuterSegmentOuterMiniDoubletUpperHitIndex];

            std::cout <<  "VALIDATION 'TC': " << "TC" <<  " hit0: " << hit0 <<  " hit1: " << hit1 <<  " hit2: " << hit2 <<  " hit3: " << hit3 <<  " hit4: " << hit4 <<  " hit5: " << hit5 <<  " hit6: " << hit6 <<  " hit7: " << hit7 <<  " hit8: " << hit8 <<  " hit9: " << hit9 <<  " hit10: " << hit10 <<  " hit11: " << hit11 <<  std::endl;
        }
    }
}

//________________________________________________________________________________________________________________________________
void printTCs_for_CPU(SDL::CPU::Event& event)
{
    // get layer ptrs
    std::vector<SDL::CPU::Layer*> layerPtrs = event.getLayerPtrs();
    layerPtrs.push_back(&(event.getPixelLayer()));

    // Loop over layers and access track candidates
    for (auto& layerPtr : layerPtrs)
    {

        // Track Candidate ptrs
        const std::vector<SDL::CPU::TrackCandidate*>& trackCandidatePtrs = layerPtr->getTrackCandidatePtrs();


        // Loop over trackCandidate ptrs
        for (auto& trackCandidatePtr : trackCandidatePtrs)
        {
            // hit idx
            unsigned int hit0 = trackCandidatePtr->innerTrackletBasePtr()->innerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->idx();
            unsigned int hit1 = trackCandidatePtr->innerTrackletBasePtr()->innerSegmentPtr()->innerMiniDoubletPtr()->upperHitPtr()->idx();
            unsigned int hit2 = trackCandidatePtr->innerTrackletBasePtr()->innerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->idx();
            unsigned int hit3 = trackCandidatePtr->innerTrackletBasePtr()->innerSegmentPtr()->outerMiniDoubletPtr()->upperHitPtr()->idx();
            unsigned int hit4 = trackCandidatePtr->innerTrackletBasePtr()->outerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->idx();
            unsigned int hit5 = trackCandidatePtr->innerTrackletBasePtr()->outerSegmentPtr()->innerMiniDoubletPtr()->upperHitPtr()->idx();
            unsigned int hit6 = trackCandidatePtr->innerTrackletBasePtr()->outerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->idx();
            unsigned int hit7 = trackCandidatePtr->innerTrackletBasePtr()->outerSegmentPtr()->outerMiniDoubletPtr()->upperHitPtr()->idx();
            unsigned int hit8 = trackCandidatePtr->outerTrackletBasePtr()->outerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->idx();
            unsigned int hit9 = trackCandidatePtr->outerTrackletBasePtr()->outerSegmentPtr()->innerMiniDoubletPtr()->upperHitPtr()->idx();
            unsigned int hit10 = trackCandidatePtr->outerTrackletBasePtr()->outerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->idx();
            unsigned int hit11 = trackCandidatePtr->outerTrackletBasePtr()->outerSegmentPtr()->outerMiniDoubletPtr()->upperHitPtr()->idx();

            std::cout <<  "VALIDATION 'TC': " << "TC" <<  " hit0: " << hit0 <<  " hit1: " << hit1 <<  " hit2: " << hit2 <<  " hit3: " << hit3 <<  " hit4: " << hit4 <<  " hit5: " << hit5 <<  " hit6: " << hit6 <<  " hit7: " << hit7 <<  " hit8: " << hit8 <<  " hit9: " << hit9 <<  " hit10: " << hit10 <<  " hit11: " << hit11 <<  std::endl;
        }

    }

}


//________________________________________________________________________________________________________________________________
void debugPrintOutlierMultiplicities(SDL::Event& event)
{
    SDL::trackCandidates& trackCandidatesInGPU = (*event.getTrackCandidates());
    SDL::tracklets& trackletsInGPU = (*event.getTracklets());
    SDL::triplets& tripletsInGPU = (*event.getTriplets());
    SDL::segments& segmentsInGPU = (*event.getSegments());
    SDL::miniDoublets& miniDoubletsInGPU = (*event.getMiniDoublets());
    SDL::hits& hitsInGPU = (*event.getHits());
    SDL::modules& modulesInGPU = (*event.getModules());
    int nTrackCandidates = 0;
    for (unsigned int idx = 0; idx <= *(modulesInGPU.nLowerModules); ++idx)
    {
        if (trackCandidatesInGPU.nTrackCandidates[idx] > 50000)
        {
            std::cout <<  " SDL::modulesInGPU->detIds[SDL::modulesInGPU->lowerModuleIndices[idx]]: " << modulesInGPU.detIds[modulesInGPU.lowerModuleIndices[idx]] <<  std::endl;
            std::cout <<  " idx: " << idx <<  " trackCandidatesInGPU.nTrackCandidates[idx]: " << trackCandidatesInGPU.nTrackCandidates[idx] <<  std::endl;
            std::cout <<  " idx: " << idx <<  " trackletsInGPU.nTracklets[idx]: " << trackletsInGPU.nTracklets[idx] <<  std::endl;
            std::cout <<  " idx: " << idx <<  " tripletsInGPU.nTriplets[idx]: " << tripletsInGPU.nTriplets[idx] <<  std::endl;
            unsigned int i = modulesInGPU.lowerModuleIndices[idx];
            std::cout <<  " idx: " << idx <<  " i: " << i <<  " segmentsInGPU.nSegments[i]: " << segmentsInGPU.nSegments[i] <<  std::endl;
            int nMD = miniDoubletsInGPU.nMDs[2*idx]+miniDoubletsInGPU.nMDs[2*idx+1] ;
            std::cout <<  " idx: " << idx <<  " nMD: " << nMD <<  std::endl;
            int nHits = 0;
            nHits += modulesInGPU.hitRanges[4*idx+1] - modulesInGPU.hitRanges[4*idx] + 1;       
            nHits += modulesInGPU.hitRanges[4*idx+3] - modulesInGPU.hitRanges[4*idx+2] + 1;
            std::cout <<  " idx: " << idx <<  " nHits: " << nHits <<  std::endl;
        }
    }
}

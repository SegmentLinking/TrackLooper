#include "StudySegmentSelectionOnTruth.h"

StudySegmentSelectionOnTruth::StudySegmentSelectionOnTruth(const char* studyName, StudySegmentSelectionOnTruth::StudySegmentSelectionOnTruthMode mode_)
{

    studyname = studyName;
    mode = mode_;
    switch (mode)
    {
        case kStudySelAll: modename = "all"; break;
        case kStudySelBB12: modename = "bb12"; break;
        case kStudySelBB23: modename = "bb23"; break;
        case kStudySelBB34: modename = "bb34"; break;
        case kStudySelBB45: modename = "bb45"; break;
        case kStudySelBB56: modename = "bb56"; break;
        case kStudySelSpecific: modename = "specific"; break;
        default: modename = "UNDEFINED"; break;
    }

}

void StudySegmentSelectionOnTruth::bookStudy()
{
    // Book Histograms
    ana.histograms.addVecHistogram(TString::Format("sg_truth_%s_cutflow", modename), 7, 0, 7, [&]() { return sg_cutflow; } );

    ana.histograms.addVecHistogram(TString::Format("sg_truth_%s_zLo_cut", modename), 180, -15, 15, [&]() { return sg_zLo_cut; } );
    ana.histograms.addVecHistogram(TString::Format("sg_truth_%s_zHi_cut", modename), 180, -15, 15, [&]() { return sg_zHi_cut; } );
}

void StudySegmentSelectionOnTruth::doStudy(SDL::Event& event, std::vector<std::tuple<unsigned int, SDL::Event*>> simtrkevents)
{

    // Each do study is performed per event

    // The study assumes that the createSegments has been run with "AllComb" algorithm
    // The DefaultAlgo will be run individually here

    // First clear all the output variables that will be used to fill the histograms for this event
    sg_cutflow.clear();
    sg_zLo_cut.clear();
    sg_zHi_cut.clear();

    //***********************
    // Studying selections
    //***********************

    // Loop over track events
    for (auto& simtrkevent : simtrkevents)
    {

        // Unpack the tuple (sim_track_index, SDL::Event containing reco hits only matched to the given sim track)
        unsigned int& isimtrk = std::get<0>(simtrkevent);
        SDL::Event& trackevent = *(std::get<1>(simtrkevent));

        // Loop over tracklets in event
        for (auto& layerPtr : trackevent.getLayerPtrs())
        {

            // Parse the layer index later to be used for indexing
            int layer_idx = layerPtr->layerIdx() - 1;

            // This means no tracklets in this layer
            if (layerPtr->getSegmentPtrs().size() == 0)
            {
                continue;
            }

            // Assuming I have at least one tracklets from this track
            std::vector<SDL::Segment*> sgs_of_interest;
            for (auto& sg : layerPtr->getSegmentPtrs())
            {
                const SDL::Module& innerLowerModule = sg->innerMiniDoubletPtr()->lowerHitPtr()->getModule();
                const SDL::Module& outerLowerModule = sg->outerMiniDoubletPtr()->lowerHitPtr()->getModule();
                bool isInnerMiniDoubletBarrel = innerLowerModule.subdet() == SDL::Module::Barrel;
                bool isOuterMiniDoubletBarrel = outerLowerModule.subdet() == SDL::Module::Barrel;
                bool isInnerMiniDoubletBarrelFlat = innerLowerModule.subdet() == SDL::Module::Barrel and innerLowerModule.side() == SDL::Module::Center;
                bool isOuterMiniDoubletBarrelFlat = outerLowerModule.subdet() == SDL::Module::Barrel and outerLowerModule.side() == SDL::Module::Center;

                // Depending on the mode, only include a subset of interested tracklets
                switch (mode)
                {
                    case kStudySelAll: /* do nothing */ break;
                    case kStudySelBB12: if (not (isInnerMiniDoubletBarrel and isOuterMiniDoubletBarrel and innerLowerModule.layer() == 1 and outerLowerModule.layer() == 2)) continue; break;
                    case kStudySelBB23: if (not (isInnerMiniDoubletBarrel and isOuterMiniDoubletBarrel and innerLowerModule.layer() == 2 and outerLowerModule.layer() == 3)) continue; break;
                    case kStudySelBB34: if (not (isInnerMiniDoubletBarrel and isOuterMiniDoubletBarrel and innerLowerModule.layer() == 3 and outerLowerModule.layer() == 4)) continue; break;
                    case kStudySelBB45: if (not (isInnerMiniDoubletBarrel and isOuterMiniDoubletBarrel and innerLowerModule.layer() == 4 and outerLowerModule.layer() == 5)) continue; break;
                    case kStudySelBB56: if (not (isInnerMiniDoubletBarrel and isOuterMiniDoubletBarrel and innerLowerModule.layer() == 5 and outerLowerModule.layer() == 6)) continue; break;
                    case kStudySelSpecific: break;
                    default: /* skip everything should not be here anyways...*/ continue; break;
                }

                // If this tracklet passes the condition that it is of interest then, add to the list of tracklets of interest
                sgs_of_interest.push_back(sg);

            }

            // If no sgs of interest are found then skip
            if (sgs_of_interest.size() == 0)
                continue;

            // The sgs_of_interest holds only the sg "candidate" that we think are of interest for the given study mode
            for (auto& sg : sgs_of_interest)
            {

                sg->runSegmentAlgo(SDL::Default_SGAlgo);

                // Cutflow
                //------------------------
                sg_cutflow.push_back(0);

                const int& passbit = sg->getPassBitsDefaultAlgo();

                for (unsigned int i = 0; i < SDL::Segment::SegmentSelection::nCut; ++i)
                {
                    if (passbit & (1 << i))
                    {
                        sg_cutflow.push_back(i + 1);
                    }
                    else
                    {
                        break;
                    }

                }

                if (true)
                {
                    sg_zLo_cut.push_back(sg->getZOut() - sg->getZLo());
                    sg_zHi_cut.push_back(sg->getZLo() - sg->getZOut());
                }

            }

        }

    }

}

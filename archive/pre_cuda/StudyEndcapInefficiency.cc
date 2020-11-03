#include "StudyEndcapInefficiency.h"

StudyEndcapInefficiency::StudyEndcapInefficiency(const char* studyName, StudyEndcapInefficiency::StudyEndcapInefficiencyMode mode_)
{

    studyname = studyName;
    mode = mode_;
    switch (mode)
    {
        case kStudyEndcapIneffAll: modename = "all"; break;
        case kStudyEndcapIneffPS: modename = "PS"; break;
        case kStudyEndcapIneff2S: modename = "2S"; break;
        case kStudyEndcapIneffPSLowP: modename = "PSLowP"; break;
        case kStudyEndcapIneffPSLowS: modename = "PSLowS"; break;
        default: modename = "UNDEFINED"; break;
    }
}

void StudyEndcapInefficiency::bookStudy()
{

    // Book histograms to study the mini-doublet candidate either passing or failing
    ana.histograms.addVecHistogram(TString::Format("mdcand_%s_dzs_passed"             , modename) , 50 , 0  , 2    , [&]() { return dzs_passed              ; } );
    ana.histograms.addVecHistogram(TString::Format("mdcand_%s_drts_passed"            , modename) , 50 ,-2  , 2    , [&]() { return drts_passed             ; } );
    ana.histograms.addVecHistogram(TString::Format("mdcand_%s_fabsdPhis_passed"       , modename) , 50 , 0  , 0.1  , [&]() { return fabsdPhis_passed        ; } );
    ana.histograms.addVecHistogram(TString::Format("mdcand_%s_dzfracs_passed"         , modename) , 50 , 0  , 0.005 , [&]() { return dzfracs_passed          ; } );
    ana.histograms.addVecHistogram(TString::Format("mdcand_%s_zs_passed"              , modename) , 50 , 0  , 400  , [&]() { return zs_passed               ; } );
    ana.histograms.addVecHistogram(TString::Format("mdcand_%s_fabsdPhiMods_passed"    , modename) , 60 , -1 , 5    , [&]() { return fabsdPhiMods_passed     ; } );
    ana.histograms.addVecHistogram(TString::Format("mdcand_%s_fabsdPhiModDiffs_passed", modename) , 60 , -1 , 5    , [&]() { return fabsdPhiModDiffs_passed ; } );

    ana.histograms.addVecHistogram(TString::Format("mdcand_%s_dzs_failed"             , modename) , 50 , 0  , 2    , [&]() { return dzs_failed              ; } );
    ana.histograms.addVecHistogram(TString::Format("mdcand_%s_drts_failed"            , modename) , 50 ,-2  , 2    , [&]() { return drts_failed             ; } );
    ana.histograms.addVecHistogram(TString::Format("mdcand_%s_fabsdPhis_failed"       , modename) , 50 , 0  , 0.1  , [&]() { return fabsdPhis_failed        ; } );
    ana.histograms.addVecHistogram(TString::Format("mdcand_%s_dzfracs_failed"         , modename) , 50 , 0  , 0.005, [&]() { return dzfracs_failed          ; } );
    ana.histograms.addVecHistogram(TString::Format("mdcand_%s_zs_failed"              , modename) , 50 , 0  , 400  , [&]() { return zs_failed               ; } );
    ana.histograms.addVecHistogram(TString::Format("mdcand_%s_fabsdPhiMods_failed"    , modename) , 60 , -1 , 5    , [&]() { return fabsdPhiMods_failed     ; } );
    ana.histograms.addVecHistogram(TString::Format("mdcand_%s_fabsdPhiModDiffs_failed", modename) , 60 , -1 , 5    , [&]() { return fabsdPhiModDiffs_failed ; } );

}

void StudyEndcapInefficiency::doStudy(SDL::Event& event, std::vector<std::tuple<unsigned int, SDL::Event*>> simtrkevents)
{

    dzs_passed.clear();
    drts_passed.clear();
    fabsdPhis_passed.clear();
    zs_passed.clear();
    dzfracs_passed.clear();
    fabsdPhiMods_passed.clear();
    fabsdPhiModDiffs_passed.clear();

    dzs_failed.clear();
    drts_failed.clear();
    fabsdPhis_failed.clear();
    zs_failed.clear();
    dzfracs_failed.clear();
    fabsdPhiMods_failed.clear();
    fabsdPhiModDiffs_failed.clear();

    //*******************
    // Inefficiency study
    //*******************

    // Loop over track events
    for (auto& simtrkevent : simtrkevents)
    {

        // Unpack the tuple (sim_track_index, SDL::Event containing reco hits only matched to the given sim track)
        // unsigned int& isimtrk = std::get<0>(simtrkevent);
        SDL::Event& trackevent = *(std::get<1>(simtrkevent));

        // Loop over the lower modules that contains hits for this track
        for (auto& lowerModulePtr_Track : trackevent.getLowerModulePtrs())
        {

            // Depending on the mode, only run a subset of interested modules
            switch (mode)
            {
                case kStudyEndcapIneffAll: if (not (lowerModulePtr_Track->subdet() == SDL::Module::Endcap)) { continue; } break;
                case kStudyEndcapIneffPS: if (not (lowerModulePtr_Track->subdet() == SDL::Module::Endcap and lowerModulePtr_Track->moduleType() == SDL::Module::PS)) { continue; } break;
                case kStudyEndcapIneff2S: if (not (lowerModulePtr_Track->subdet() == SDL::Module::Endcap and lowerModulePtr_Track->moduleType() == SDL::Module::TwoS)) { continue; } break;
                case kStudyEndcapIneffPSLowP: if (not (lowerModulePtr_Track->subdet() == SDL::Module::Endcap and lowerModulePtr_Track->moduleType() == SDL::Module::PS and lowerModulePtr_Track->moduleLayerType() == SDL::Module::Pixel)) { continue; } break;
                case kStudyEndcapIneffPSLowS: if (not (lowerModulePtr_Track->subdet() == SDL::Module::Endcap and lowerModulePtr_Track->moduleType() == SDL::Module::PS and lowerModulePtr_Track->moduleLayerType() == SDL::Module::Strip)) { continue; } break;
                default: /* skip everything should not be here anyways...*/ continue; break;
            }

            // Loop over the md "candidate" from the module that a sim-track passed through and left at least one hit in each module
            for (auto& md_Track : lowerModulePtr_Track->getMiniDoubletPtrs())
            {

                SDL::Module& lowerModule = *lowerModulePtr_Track;
                SDL::Hit& lowerHit = *(md_Track->lowerHitPtr());
                SDL::Hit& upperHit = *(md_Track->upperHitPtr());

                // These are the individual component for mini-doublet calculation
                // Copied from SDL::MiniDoublet code
                float z = fabs(lowerHit.z());
                float dz = std::abs(lowerHit.z() - upperHit.z());
                float drt = lowerHit.rt() - upperHit.rt();
                float fabsdPhi = (lowerModule.moduleType() == SDL::Module::PS) ?
                    SDL::MiniDoublet::fabsdPhiPixelShift(lowerHit, upperHit, lowerModule) : std::abs(lowerHit.deltaPhi(upperHit));
                float dzfrac = dz / fabs(lowerHit.z());
                float fabsdPhiMod = fabsdPhi / dzfrac * (1.f + dzfrac);
                float miniCut = SDL::MiniDoublet::dPhiThreshold(lowerHit, lowerModule);
                float fabsdPhiModDiff = fabsdPhiMod - miniCut;

                // Actually use the static function to perform the calculation
                if (SDL::MiniDoublet::isHitPairAMiniDoublet(lowerHit, upperHit, lowerModule, SDL::Default_MDAlgo, SDL::Log_Nothing))
                {
                    // Passed
                    dzs_passed.push_back(dz);
                    drts_passed.push_back(drt);
                    fabsdPhis_passed.push_back(fabsdPhi);
                    zs_passed.push_back(z);
                    dzfracs_passed.push_back(dzfrac);
                    fabsdPhiMods_passed.push_back(fabsdPhiMod);
                    fabsdPhiModDiffs_passed.push_back(fabsdPhiModDiff);

                }
                else
                {
                    // Failed
                    dzs_failed.push_back(dz);
                    drts_failed.push_back(drt);
                    fabsdPhis_failed.push_back(fabsdPhi);
                    zs_failed.push_back(z);
                    dzfracs_failed.push_back(dzfrac);
                    fabsdPhiMods_failed.push_back(fabsdPhiMod);
                    fabsdPhiModDiffs_failed.push_back(fabsdPhiModDiff);

                }

            }

        }

    }

}

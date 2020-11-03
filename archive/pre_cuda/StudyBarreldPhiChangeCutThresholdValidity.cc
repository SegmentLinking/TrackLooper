#include "StudyBarreldPhiChangeCutThresholdValidity.h"

void StudyBarreldPhiChangeCutThresholdValidity::bookStudy()
{
    // Book Histograms
    ana.histograms.add2DVecHistogram("rt", 50, 20, 120, "phim", 50, 0., 1., [&]() { return rt_v_phim__rt; }, [&]() { return rt_v_phim__phim; } );
    for (int ii = 0; ii < NLAYERS; ++ii)
    {
        ana.histograms.add2DVecHistogram("rt", 50, 20, 120, TString::Format("phim_by_layer%d", ii), 50, 0., 1., [&, ii]() { return rt_v_phim__rt_by_layer[ii]; }, [&, ii]() { return rt_v_phim__phim_by_layer[ii]; } );
    }

    ana.histograms.add2DVecHistogram("rt", 50, 20, 120, "dphi", 50, 0., 1., [&]() { return rt_v_dphi__rt; }, [&]() { return rt_v_dphi__dphi; } );
    for (int ii = 0; ii < NLAYERS; ++ii)
    {
        ana.histograms.add2DVecHistogram("rt", 50, 20, 120, TString::Format("dphi_by_layer%d", ii), 50, 0., 1., [&, ii]() { return rt_v_dphi__rt_by_layer[ii]; }, [&, ii]() { return rt_v_dphi__dphi_by_layer[ii]; } );
    }

}

void StudyBarreldPhiChangeCutThresholdValidity::doStudy(SDL::Event& recoevent, std::vector<std::tuple<unsigned int, SDL::Event*>> simtrkevents)
{

    // Each do study is performed per event

    // First clear all the output variables that will be used to fill the histograms for this event
    rt_v_phim__rt.clear();
    rt_v_phim__phim.clear();
    for (int ii = 0; ii < NLAYERS; ++ii)
    {
        rt_v_phim__rt_by_layer[ii].clear();
        rt_v_phim__phim_by_layer[ii].clear();
    }

    rt_v_dphi__rt.clear();
    rt_v_dphi__dphi.clear();
    for (int ii = 0; ii < NLAYERS; ++ii)
    {
        rt_v_dphi__rt_by_layer[ii].clear();
        rt_v_dphi__dphi_by_layer[ii].clear();
    }

    for (auto& simtrkevent : simtrkevents)
    {

        unsigned int& isimtrk = std::get<0>(simtrkevent);
        SDL::Event& trackevent = *(std::get<1>(simtrkevent));

        // Plotting dPhiChange's and the thresholds etc. (study done for 04/29/2019 http://uaf-10.t2.ucsd.edu/~phchang/talks/PhilipChang20190429_SDL_Update.pdf)
        for (auto& lowerModulePtr_Track : trackevent.getLowerModulePtrs())
        {

            float pt = trk.sim_pt()[isimtrk];

            // Commented in or out for the plots
            // if (pt > 1.)
            //     continue;

            if (pt < 1.)
                continue;

            // Fou
            if (lowerModulePtr_Track->subdet() == SDL::Module::Barrel and lowerModulePtr_Track->side() == SDL::Module::Center)
            {

                for (auto& lowerHitPtr : lowerModulePtr_Track->getHitPtrs())
                {
                    rt_v_phim__rt.push_back(lowerHitPtr->rt());
                    rt_v_phim__phim.push_back(SDL::MiniDoublet::dPhiThreshold(*lowerHitPtr, *lowerModulePtr_Track));
                    int layer_idx = lowerModulePtr_Track->layer() - 1;
                    rt_v_phim__rt_by_layer[layer_idx].push_back(lowerHitPtr->rt());
                    rt_v_phim__phim_by_layer[layer_idx].push_back(SDL::MiniDoublet::dPhiThreshold(*lowerHitPtr, *lowerModulePtr_Track));
                }

                for (auto& md_Track : lowerModulePtr_Track->getMiniDoubletPtrs())
                {
                    SDL::Hit* lowerHitPtr = md_Track->lowerHitPtr();
                    float rt = lowerHitPtr->rt();
                    float dphi = lowerHitPtr->deltaPhiChange(*(md_Track->upperHitPtr()));
                    rt_v_dphi__rt.push_back(rt);
                    rt_v_dphi__dphi.push_back(dphi);
                    int layer_idx = lowerModulePtr_Track->layer() - 1;
                    rt_v_dphi__rt_by_layer[layer_idx].push_back(rt);
                    rt_v_dphi__dphi_by_layer[layer_idx].push_back(dphi);
                }

            }

        }

    }

}

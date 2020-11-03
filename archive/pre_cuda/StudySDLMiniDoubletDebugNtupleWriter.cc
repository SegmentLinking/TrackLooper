#include "StudySDLMiniDoubletDebugNtupleWriter.h"

StudySDLMiniDoubletDebugNtupleWriter::StudySDLMiniDoubletDebugNtupleWriter()
{
}

void StudySDLMiniDoubletDebugNtupleWriter::bookStudy()
{
    ana.tx->createBranch<float>("pt");
    ana.tx->createBranch<float>("eta");
    ana.tx->createBranch<float>("dxy");
    ana.tx->createBranch<float>("dz");
    ana.tx->createBranch<float>("pdgid");
    ana.tx->createBranch<float>("itrk");
    ana.tx->createBranch<int  >("is_trk_bbbbbb");
    ana.tx->createBranch<int  >("is_trk_bbbbbe");
    ana.tx->createBranch<int  >("is_trk_bbbbee");
    ana.tx->createBranch<int  >("is_trk_bbbeee");
    ana.tx->createBranch<int  >("is_trk_bbeeee");
    ana.tx->createBranch<int  >("is_trk_beeeee");
    ana.tx->createBranch<vector<float>>("simhit_x");
    ana.tx->createBranch<vector<float>>("simhit_y");
    ana.tx->createBranch<vector<float>>("simhit_z");
    ana.tx->createBranch<vector<float>>("simhit_px");
    ana.tx->createBranch<vector<float>>("simhit_py");
    ana.tx->createBranch<vector<float>>("simhit_pz");
    ana.tx->createBranch<vector<float>>("ph2_x");
    ana.tx->createBranch<vector<float>>("ph2_y");
    ana.tx->createBranch<vector<float>>("ph2_z");

    ana.tx->createBranch<vector<float>>("simhit_x_pdgid_matched");
    ana.tx->createBranch<vector<float>>("simhit_y_pdgid_matched");
    ana.tx->createBranch<vector<float>>("simhit_z_pdgid_matched");
    ana.tx->createBranch<vector<float>>("simhit_px_pdgid_matched");
    ana.tx->createBranch<vector<float>>("simhit_py_pdgid_matched");
    ana.tx->createBranch<vector<float>>("simhit_pz_pdgid_matched");
    ana.tx->createBranch<vector<float>>("ph2_x_pdgid_matched");
    ana.tx->createBranch<vector<float>>("ph2_y_pdgid_matched");
    ana.tx->createBranch<vector<float>>("ph2_z_pdgid_matched");


    for (unsigned int ilayer = 1; ilayer <= 6; ++ilayer)
    {
        ana.tx->createBranch<int  >(TString::Format("md%d_ncand", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("md%d_dz", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("md%d_dphi", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("md%d_dphichange", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("md%d_minicut", ilayer));
        ana.tx->createBranch<vector<int  >>(TString::Format("md%d_pass", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("md%d_lower_hit_x", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("md%d_lower_hit_y", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("md%d_lower_hit_z", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("md%d_lower_hit_rt", ilayer));
        ana.tx->createBranch<vector<int  >>(TString::Format("md%d_lower_hit_side", ilayer));
        ana.tx->createBranch<vector<int  >>(TString::Format("md%d_lower_hit_rod", ilayer));
        ana.tx->createBranch<vector<int  >>(TString::Format("md%d_lower_hit_module", ilayer));
        ana.tx->createBranch<vector<int  >>(TString::Format("md%d_lower_hit_subdet", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("md%d_upper_hit_x", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("md%d_upper_hit_y", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("md%d_upper_hit_z", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("md%d_upper_hit_rt", ilayer));
        ana.tx->createBranch<vector<int  >>(TString::Format("md%d_upper_hit_side", ilayer));
        ana.tx->createBranch<vector<int  >>(TString::Format("md%d_upper_hit_rod", ilayer));
        ana.tx->createBranch<vector<int  >>(TString::Format("md%d_upper_hit_module", ilayer));
        ana.tx->createBranch<vector<int  >>(TString::Format("md%d_upper_hit_subdet", ilayer));
    }

    for (unsigned int ilayer = 1; ilayer <= 5; ++ilayer)
    {
        ana.tx->createBranch<int  >(TString::Format("mdendcap%d_ncand", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("mdendcap%d_dz", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("mdendcap%d_dphi", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("mdendcap%d_dphichange", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("mdendcap%d_minicut", ilayer));
        ana.tx->createBranch<vector<int  >>(TString::Format("mdendcap%d_pass", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("mdendcap%d_lower_hit_x", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("mdendcap%d_lower_hit_y", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("mdendcap%d_lower_hit_z", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("mdendcap%d_lower_hit_rt", ilayer));
        ana.tx->createBranch<vector<int  >>(TString::Format("mdendcap%d_lower_hit_side", ilayer));
        ana.tx->createBranch<vector<int  >>(TString::Format("mdendcap%d_lower_hit_rod", ilayer));
        ana.tx->createBranch<vector<int  >>(TString::Format("mdendcap%d_lower_hit_module", ilayer));
        ana.tx->createBranch<vector<int  >>(TString::Format("mdendcap%d_lower_hit_subdet", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("mdendcap%d_upper_hit_x", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("mdendcap%d_upper_hit_y", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("mdendcap%d_upper_hit_z", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("mdendcap%d_upper_hit_rt", ilayer));
        ana.tx->createBranch<vector<int  >>(TString::Format("mdendcap%d_upper_hit_side", ilayer));
        ana.tx->createBranch<vector<int  >>(TString::Format("mdendcap%d_upper_hit_rod", ilayer));
        ana.tx->createBranch<vector<int  >>(TString::Format("mdendcap%d_upper_hit_module", ilayer));
        ana.tx->createBranch<vector<int  >>(TString::Format("mdendcap%d_upper_hit_subdet", ilayer));
    }

}

void StudySDLMiniDoubletDebugNtupleWriter::doStudy(SDL::Event& event, std::vector<std::tuple<unsigned int, SDL::Event*>> simtrkevents)
{

    ana.tx->clear();

    // Loop over track events
    for (auto& simtrkevent : simtrkevents)
    {
        // Unpack the tuple (sim_track_index, SDL::Event containing reco hits only matched to the given sim track)
        unsigned int& isimtrk = std::get<0>(simtrkevent);
        SDL::Event& trackevent = *(std::get<1>(simtrkevent));

        ana.tx->setBranch<float>("pt", trk.sim_pt()[isimtrk]);
        ana.tx->setBranch<float>("eta", trk.sim_eta()[isimtrk]);
        ana.tx->setBranch<float>("dxy", trk.sim_pca_dxy()[isimtrk]);
        ana.tx->setBranch<float>("dz", trk.sim_pca_dz()[isimtrk]);
        ana.tx->setBranch<float>("pdgid", trk.sim_pdgId()[isimtrk]);
        ana.tx->setBranch<float>("itrk", isimtrk);
        ana.tx->setBranch<int>("is_trk_bbbbbb", hasAll12HitsWithNBarrelUsingModuleMap(isimtrk, 6));
        ana.tx->setBranch<int>("is_trk_bbbbbe", hasAll12HitsWithNBarrelUsingModuleMap(isimtrk, 5));
        ana.tx->setBranch<int>("is_trk_bbbbee", hasAll12HitsWithNBarrelUsingModuleMap(isimtrk, 4));
        ana.tx->setBranch<int>("is_trk_bbbeee", hasAll12HitsWithNBarrelUsingModuleMap(isimtrk, 3));
        ana.tx->setBranch<int>("is_trk_bbeeee", hasAll12HitsWithNBarrelUsingModuleMap(isimtrk, 2));
        ana.tx->setBranch<int>("is_trk_beeeee", hasAll12HitsWithNBarrelUsingModuleMap(isimtrk, 1));


        // Writing out the sim hits
        for (unsigned int ith_hit = 0; ith_hit < trk.sim_simHitIdx()[isimtrk].size(); ++ith_hit)
        {

            // Retrieve the sim hit idx
            unsigned int simhitidx = trk.sim_simHitIdx()[isimtrk][ith_hit];

            // Select only the hits in the outer tracker
            if (not (trk.simhit_subdet()[simhitidx] == 4 or trk.simhit_subdet()[simhitidx] == 5))
                continue;

            // Select only the muon hits
            // if (not (trk.simhit_particle()[simhitidx] == trk.sim_pdgId()[isimtrk]))
            //     continue;

            // Exclude what I think is muon curling hit
            if (isMuonCurlingHit(isimtrk, ith_hit))
                break;

            ana.tx->pushbackToBranch<float>("simhit_x", trk.simhit_x()[simhitidx]);
            ana.tx->pushbackToBranch<float>("simhit_y", trk.simhit_y()[simhitidx]);
            ana.tx->pushbackToBranch<float>("simhit_z", trk.simhit_z()[simhitidx]);
            ana.tx->pushbackToBranch<float>("simhit_px", trk.simhit_px()[simhitidx]);
            ana.tx->pushbackToBranch<float>("simhit_py", trk.simhit_py()[simhitidx]);
            ana.tx->pushbackToBranch<float>("simhit_pz", trk.simhit_pz()[simhitidx]);
            for (unsigned int ireco_hit = 0; ireco_hit < trk.simhit_hitIdx()[simhitidx].size(); ++ireco_hit)
            {
                ana.tx->pushbackToBranch<float>("ph2_x", trk.ph2_x()[trk.simhit_hitIdx()[simhitidx][ireco_hit]]);
                ana.tx->pushbackToBranch<float>("ph2_y", trk.ph2_y()[trk.simhit_hitIdx()[simhitidx][ireco_hit]]);
                ana.tx->pushbackToBranch<float>("ph2_z", trk.ph2_z()[trk.simhit_hitIdx()[simhitidx][ireco_hit]]);
            }

            // Select only the muon hits
            if (trk.simhit_particle()[simhitidx] == trk.sim_pdgId()[isimtrk])
            {
                ana.tx->pushbackToBranch<float>("simhit_x_pdgid_matched", trk.simhit_x()[simhitidx]);
                ana.tx->pushbackToBranch<float>("simhit_y_pdgid_matched", trk.simhit_y()[simhitidx]);
                ana.tx->pushbackToBranch<float>("simhit_z_pdgid_matched", trk.simhit_z()[simhitidx]);
                ana.tx->pushbackToBranch<float>("simhit_px_pdgid_matched", trk.simhit_px()[simhitidx]);
                ana.tx->pushbackToBranch<float>("simhit_py_pdgid_matched", trk.simhit_py()[simhitidx]);
                ana.tx->pushbackToBranch<float>("simhit_pz_pdgid_matched", trk.simhit_pz()[simhitidx]);
                for (unsigned int ireco_hit = 0; ireco_hit < trk.simhit_hitIdx()[simhitidx].size(); ++ireco_hit)
                {
                    ana.tx->pushbackToBranch<float>("ph2_x_pdgid_matched", trk.ph2_x()[trk.simhit_hitIdx()[simhitidx][ireco_hit]]);
                    ana.tx->pushbackToBranch<float>("ph2_y_pdgid_matched", trk.ph2_y()[trk.simhit_hitIdx()[simhitidx][ireco_hit]]);
                    ana.tx->pushbackToBranch<float>("ph2_z_pdgid_matched", trk.ph2_z()[trk.simhit_hitIdx()[simhitidx][ireco_hit]]);
                }
            }
        }

        for (unsigned int ilayer = 1; ilayer <= 6; ilayer++)
        {

            ana.tx->setBranch<int>(TString::Format("md%d_ncand", ilayer), trackevent.getLayer(ilayer, SDL::Layer::Barrel).getMiniDoubletPtrs().size() );

            for (auto& mdPtr : trackevent.getLayer(ilayer, SDL::Layer::Barrel).getMiniDoubletPtrs())
            {

                mdPtr->runMiniDoubletAlgo(SDL::Default_MDAlgo);
                ana.tx->pushbackToBranch<float>(TString::Format("md%d_dz"               , ilayer), mdPtr->getDz()                                   );
                ana.tx->pushbackToBranch<float>(TString::Format("md%d_dphi"             , ilayer), mdPtr->getDeltaPhi()                             );
                ana.tx->pushbackToBranch<float>(TString::Format("md%d_dphichange"       , ilayer), mdPtr->getDeltaPhiChange()                       );
                ana.tx->pushbackToBranch<float>(TString::Format("md%d_minicut"          , ilayer), mdPtr->getRecoVar("miniCut")                     );
                ana.tx->pushbackToBranch<int  >(TString::Format("md%d_pass"             , ilayer), mdPtr->passesMiniDoubletAlgo(SDL::Default_MDAlgo));
                ana.tx->pushbackToBranch<float>(TString::Format("md%d_lower_hit_x"      , ilayer), mdPtr->lowerHitPtr()->x()                        );
                ana.tx->pushbackToBranch<float>(TString::Format("md%d_lower_hit_y"      , ilayer), mdPtr->lowerHitPtr()->y()                        );
                ana.tx->pushbackToBranch<float>(TString::Format("md%d_lower_hit_z"      , ilayer), mdPtr->lowerHitPtr()->z()                        );
                ana.tx->pushbackToBranch<float>(TString::Format("md%d_lower_hit_rt"     , ilayer), mdPtr->lowerHitPtr()->rt()                       );
                ana.tx->pushbackToBranch<int  >(TString::Format("md%d_lower_hit_side"   , ilayer), mdPtr->lowerHitPtr()->getModule().side()         );
                ana.tx->pushbackToBranch<int  >(TString::Format("md%d_lower_hit_rod"    , ilayer), mdPtr->lowerHitPtr()->getModule().rod()          );
                ana.tx->pushbackToBranch<int  >(TString::Format("md%d_lower_hit_module" , ilayer), mdPtr->lowerHitPtr()->getModule().module()       );
                ana.tx->pushbackToBranch<int  >(TString::Format("md%d_lower_hit_subdet" , ilayer), mdPtr->lowerHitPtr()->getModule().subdet()       );
                ana.tx->pushbackToBranch<float>(TString::Format("md%d_upper_hit_x"      , ilayer), mdPtr->upperHitPtr()->x()                        );
                ana.tx->pushbackToBranch<float>(TString::Format("md%d_upper_hit_y"      , ilayer), mdPtr->upperHitPtr()->y()                        );
                ana.tx->pushbackToBranch<float>(TString::Format("md%d_upper_hit_z"      , ilayer), mdPtr->upperHitPtr()->z()                        );
                ana.tx->pushbackToBranch<float>(TString::Format("md%d_upper_hit_rt"     , ilayer), mdPtr->upperHitPtr()->rt()                       );
                ana.tx->pushbackToBranch<int  >(TString::Format("md%d_upper_hit_side"   , ilayer), mdPtr->upperHitPtr()->getModule().side()         );
                ana.tx->pushbackToBranch<int  >(TString::Format("md%d_upper_hit_rod"    , ilayer), mdPtr->upperHitPtr()->getModule().rod()          );
                ana.tx->pushbackToBranch<int  >(TString::Format("md%d_upper_hit_module" , ilayer), mdPtr->upperHitPtr()->getModule().module()       );
                ana.tx->pushbackToBranch<int  >(TString::Format("md%d_upper_hit_subdet" , ilayer), mdPtr->upperHitPtr()->getModule().subdet()       );

            }

        }

        for (unsigned int ilayer = 1; ilayer <= 5; ilayer++)
        {

            ana.tx->setBranch<int>(TString::Format("mdendcap%d_ncand", ilayer), trackevent.getLayer(ilayer, SDL::Layer::Endcap).getMiniDoubletPtrs().size() );

            for (auto& mdPtr : trackevent.getLayer(ilayer, SDL::Layer::Endcap).getMiniDoubletPtrs())
            {

                mdPtr->runMiniDoubletAlgo(SDL::Default_MDAlgo);
                ana.tx->pushbackToBranch<float>(TString::Format("mdendcap%d_dz"               , ilayer), mdPtr->getDz()                                   );
                ana.tx->pushbackToBranch<float>(TString::Format("mdendcap%d_dphi"             , ilayer), mdPtr->getDeltaPhi()                             );
                ana.tx->pushbackToBranch<float>(TString::Format("mdendcap%d_dphichange"       , ilayer), mdPtr->getDeltaPhiChange()                       );
                ana.tx->pushbackToBranch<float>(TString::Format("mdendcap%d_minicut"          , ilayer), mdPtr->getRecoVar("miniCut")                     );
                ana.tx->pushbackToBranch<int  >(TString::Format("mdendcap%d_pass"             , ilayer), mdPtr->passesMiniDoubletAlgo(SDL::Default_MDAlgo));
                ana.tx->pushbackToBranch<float>(TString::Format("mdendcap%d_lower_hit_x"      , ilayer), mdPtr->lowerHitPtr()->x()                        );
                ana.tx->pushbackToBranch<float>(TString::Format("mdendcap%d_lower_hit_y"      , ilayer), mdPtr->lowerHitPtr()->y()                        );
                ana.tx->pushbackToBranch<float>(TString::Format("mdendcap%d_lower_hit_z"      , ilayer), mdPtr->lowerHitPtr()->z()                        );
                ana.tx->pushbackToBranch<float>(TString::Format("mdendcap%d_lower_hit_rt"     , ilayer), mdPtr->lowerHitPtr()->rt()                       );
                ana.tx->pushbackToBranch<int  >(TString::Format("mdendcap%d_lower_hit_side"   , ilayer), mdPtr->lowerHitPtr()->getModule().side()         );
                ana.tx->pushbackToBranch<int  >(TString::Format("mdendcap%d_lower_hit_rod"    , ilayer), mdPtr->lowerHitPtr()->getModule().rod()          );
                ana.tx->pushbackToBranch<int  >(TString::Format("mdendcap%d_lower_hit_module" , ilayer), mdPtr->lowerHitPtr()->getModule().module()       );
                ana.tx->pushbackToBranch<int  >(TString::Format("mdendcap%d_lower_hit_subdet" , ilayer), mdPtr->lowerHitPtr()->getModule().subdet()       );
                ana.tx->pushbackToBranch<float>(TString::Format("mdendcap%d_upper_hit_x"      , ilayer), mdPtr->upperHitPtr()->x()                        );
                ana.tx->pushbackToBranch<float>(TString::Format("mdendcap%d_upper_hit_y"      , ilayer), mdPtr->upperHitPtr()->y()                        );
                ana.tx->pushbackToBranch<float>(TString::Format("mdendcap%d_upper_hit_z"      , ilayer), mdPtr->upperHitPtr()->z()                        );
                ana.tx->pushbackToBranch<float>(TString::Format("mdendcap%d_upper_hit_rt"     , ilayer), mdPtr->upperHitPtr()->rt()                       );
                ana.tx->pushbackToBranch<int  >(TString::Format("mdendcap%d_upper_hit_side"   , ilayer), mdPtr->upperHitPtr()->getModule().side()         );
                ana.tx->pushbackToBranch<int  >(TString::Format("mdendcap%d_upper_hit_rod"    , ilayer), mdPtr->upperHitPtr()->getModule().rod()          );
                ana.tx->pushbackToBranch<int  >(TString::Format("mdendcap%d_upper_hit_module" , ilayer), mdPtr->upperHitPtr()->getModule().module()       );
                ana.tx->pushbackToBranch<int  >(TString::Format("mdendcap%d_upper_hit_subdet" , ilayer), mdPtr->upperHitPtr()->getModule().subdet()       );

            }

        }

        ana.tx->fill();
        ana.tx->clear();

    }

}


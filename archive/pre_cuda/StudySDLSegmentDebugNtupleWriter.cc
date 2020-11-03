#include "StudySDLSegmentDebugNtupleWriter.h"

StudySDLSegmentDebugNtupleWriter::StudySDLSegmentDebugNtupleWriter()
{
}

void StudySDLSegmentDebugNtupleWriter::bookStudy()
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

    //segment specific stuff
    for(unsigned int ilayer = 1; ilayer <=6; ilayer++)
    {
        ana.tx->createBranch<vector<int  >>(TString::Format("sg%d_ncand", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("sg%d_dphi", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("sg%d_dphichange", ilayer));

        ana.tx->createBranch<vector<float>>(TString::Format("sg%d_dAlpha_innerMD_segment",ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("sg%d_dAlpha_outerMD_segment",ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("sg%d_dAlpha_innerMD_outerMD",ilayer));

        ana.tx->createBranch<vector<float>>(TString::Format("sg%d_sdCut",ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("sg%d_dAlpha_innerMD_segment_threshold",ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("sg%d_dAlpha_outerMD_segment_threshold",ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("sg%d_dAlpha_innerMD_outerMD_threshold",ilayer));

        ana.tx->createBranch<vector<float>>(TString::Format("sg%d_zLo",ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("sg%d_zHi",ilayer));

        ana.tx->createBranch<vector<int  >>(TString::Format("sg%d_pass", ilayer));

        ana.tx->createBranch<vector<float>>(TString::Format("sg%d_inner_md_lower_hit_x", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("sg%d_inner_md_lower_hit_y", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("sg%d_inner_md_lower_hit_z", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("sg%d_outer_md_lower_hit_x", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("sg%d_outer_md_lower_hit_y", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("sg%d_outer_md_lower_hit_z", ilayer));

        ana.tx->createBranch<vector<float>>(TString::Format("sg%d_inner_md_upper_hit_x", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("sg%d_inner_md_upper_hit_y", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("sg%d_inner_md_upper_hit_z", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("sg%d_outer_md_upper_hit_x", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("sg%d_outer_md_upper_hit_y", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("sg%d_outer_md_upper_hit_z", ilayer));


        ana.tx->createBranch<vector<int  >>(TString::Format("sg%d_inner_md_lower_hit_side", ilayer));
        ana.tx->createBranch<vector<int  >>(TString::Format("sg%d_outer_md_lower_hit_side", ilayer));

        ana.tx->createBranch<vector<int  >>(TString::Format("sg%d_inner_md_lower_hit_module", ilayer));
        ana.tx->createBranch<vector<int  >>(TString::Format("sg%d_inner_md_upper_hit_module", ilayer));
        ana.tx->createBranch<vector<int  >>(TString::Format("sg%d_outer_md_lower_hit_module", ilayer));
        ana.tx->createBranch<vector<int  >>(TString::Format("sg%d_outer_md_upper_hit_module", ilayer));

        ana.tx->createBranch<vector<int  >>(TString::Format("sg%d_inner_md_subdet", ilayer));
        ana.tx->createBranch<vector<int  >>(TString::Format("sg%d_outer_md_subdet", ilayer));

    }

    for(unsigned int ilayer = 1; ilayer <=5; ilayer++)
    {
        ana.tx->createBranch<vector<int  >>(TString::Format("sgendcap%d_ncand", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("sgendcap%d_dphi", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("sgendcap%d_dphichange", ilayer));

        ana.tx->createBranch<vector<float>>(TString::Format("sgendcap%d_dAlpha_innerMD_segment",ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("sgendcap%d_dAlpha_outerMD_segment",ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("sgendcap%d_dAlpha_innerMD_outerMD",ilayer));

        ana.tx->createBranch<vector<float>>(TString::Format("sgendcap%d_sdCut",ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("sgendcap%d_dAlpha_innerMD_segment_threshold",ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("sgendcap%d_dAlpha_outerMD_segment_threshold",ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("sgendcap%d_dAlpha_innerMD_outerMD_threshold",ilayer));

        ana.tx->createBranch<vector<float>>(TString::Format("sgendcap%d_RtLo",ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("sgendcap%d_RtHi",ilayer));

        ana.tx->createBranch<vector<int  >>(TString::Format("sgendcap%d_pass", ilayer));

        ana.tx->createBranch<vector<float>>(TString::Format("sgendcap%d_RtIn",ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("sgendcap%d_RtOut",ilayer));

        ana.tx->createBranch<vector<float>>(TString::Format("sgendcap%d_inner_md_lower_hit_x", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("sgendcap%d_inner_md_lower_hit_y",ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("sgendcap%d_inner_md_lower_hit_z",ilayer));

        ana.tx->createBranch<vector<float>>(TString::Format("sgendcap%d_outer_md_lower_hit_x", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("sgendcap%d_outer_md_lower_hit_y", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("sgendcap%d_outer_md_lower_hit_z", ilayer));

        ana.tx->createBranch<vector<float>>(TString::Format("sgendcap%d_inner_md_upper_hit_x", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("sgendcap%d_inner_md_upper_hit_y",ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("sgendcap%d_inner_md_upper_hit_z",ilayer));

        ana.tx->createBranch<vector<float>>(TString::Format("sgendcap%d_outer_md_upper_hit_x", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("sgendcap%d_outer_md_upper_hit_y", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("sgendcap%d_outer_md_upper_hit_z", ilayer));


        ana.tx->createBranch<vector<int>>(TString::Format("sgendcap%d_inner_md_lower_hit_side", ilayer));
        ana.tx->createBranch<vector<int  >>(TString::Format("sgendcap%d_outer_md_lower_hit_side", ilayer));

        ana.tx->createBranch<vector<int  >>(TString::Format("sgendcap%d_inner_md_lower_hit_module", ilayer));
        ana.tx->createBranch<vector<int  >>(TString::Format("sgendcap%d_inner_md_upper_hit_module", ilayer));
        ana.tx->createBranch<vector<int  >>(TString::Format("sgendcap%d_outer_md_lower_hit_module", ilayer));
        ana.tx->createBranch<vector<int  >>(TString::Format("sgendcap%d_outer_md_upper_hit_module", ilayer));

        ana.tx->createBranch<vector<int  >>(TString::Format("sgendcap%d_inner_md_subdet", ilayer));
        ana.tx->createBranch<vector<int  >>(TString::Format("sgendcap%d_outer_md_subdet", ilayer));

    }


}

void StudySDLSegmentDebugNtupleWriter::doStudy(SDL::Event& event, std::vector<std::tuple<unsigned int, SDL::Event*>> simtrkevents)
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

        }

        for(unsigned int ilayer = 1; ilayer <= 6; ilayer++)
        {
            ana.tx->pushbackToBranch<int>(TString::Format("sg%d_ncand",ilayer),trackevent.getLayer(ilayer,SDL::Layer::Barrel).getSegmentPtrs().size());

            for(auto &sgPtr:trackevent.getLayer(ilayer,SDL::Layer::Barrel).getSegmentPtrs())
            {
                sgPtr->runSegmentAlgo(SDL::Default_SGAlgo);

                ana.tx->pushbackToBranch<float>(TString::Format("sg%d_dphi", ilayer),sgPtr->getRecoVar("deltaPhi"));
                ana.tx->pushbackToBranch<float>(TString::Format("sg%d_dphichange", ilayer),sgPtr->getDeltaPhiChange());
    
                ana.tx->pushbackToBranch<float>(TString::Format("sg%d_dAlpha_innerMD_segment",ilayer),sgPtr->getdAlphaInnerMDSegment());
                ana.tx->pushbackToBranch<float>(TString::Format("sg%d_dAlpha_outerMD_segment",ilayer),sgPtr->getdAlphaOuterMDSegment());
                ana.tx->pushbackToBranch<float>(TString::Format("sg%d_dAlpha_innerMD_outerMD",ilayer),sgPtr->getdAlphaInnerMDOuterMD());

                ana.tx->pushbackToBranch<float>(TString::Format("sg%d_sdCut",ilayer),sgPtr->getRecoVar("sdCut"));
                ana.tx->pushbackToBranch<float>(TString::Format("sg%d_dAlpha_innerMD_segment_threshold",ilayer),sgPtr->getRecoVar("dAlpha_innerMD_segment"));
                ana.tx->pushbackToBranch<float>(TString::Format("sg%d_dAlpha_outerMD_segment_threshold",ilayer),sgPtr->getRecoVar("dAlpha_outerMD_segment"));
                ana.tx->pushbackToBranch<float>(TString::Format("sg%d_dAlpha_innerMD_outerMD_threshold",ilayer),sgPtr->getRecoVar("dAlpha_innerMD_outerMD"));

                ana.tx->pushbackToBranch<float>(TString::Format("sg%d_zLo",ilayer),sgPtr->getZLo());
                ana.tx->pushbackToBranch<float>(TString::Format("sg%d_zHi",ilayer),sgPtr->getZHi());

                ana.tx->pushbackToBranch<int>(TString::Format("sg%d_pass", ilayer),sgPtr->passesSegmentAlgo(SDL::Default_SGAlgo));

                ana.tx->pushbackToBranch<float>(TString::Format("sg%d_inner_md_lower_hit_x", ilayer),sgPtr->innerMiniDoubletPtr()->lowerHitPtr()->x());
                ana.tx->pushbackToBranch<float>(TString::Format("sg%d_inner_md_lower_hit_y", ilayer),sgPtr->innerMiniDoubletPtr()->lowerHitPtr()->y());
                ana.tx->pushbackToBranch<float>(TString::Format("sg%d_inner_md_lower_hit_z", ilayer),sgPtr->innerMiniDoubletPtr()->lowerHitPtr()->z());

                ana.tx->pushbackToBranch<float>(TString::Format("sg%d_inner_md_upper_hit_x", ilayer),sgPtr->innerMiniDoubletPtr()->upperHitPtr()->x());
                ana.tx->pushbackToBranch<float>(TString::Format("sg%d_inner_md_upper_hit_y", ilayer),sgPtr->innerMiniDoubletPtr()->upperHitPtr()->y());
                ana.tx->pushbackToBranch<float>(TString::Format("sg%d_inner_md_upper_hit_z", ilayer),sgPtr->innerMiniDoubletPtr()->upperHitPtr()->z());

                ana.tx->pushbackToBranch<float>(TString::Format("sg%d_outer_md_lower_hit_x", ilayer),sgPtr->outerMiniDoubletPtr()->upperHitPtr()->x());
                ana.tx->pushbackToBranch<float>(TString::Format("sg%d_outer_md_lower_hit_y", ilayer),sgPtr->outerMiniDoubletPtr()->lowerHitPtr()->y());
                ana.tx->pushbackToBranch<float>(TString::Format("sg%d_outer_md_lower_hit_z", ilayer),sgPtr->outerMiniDoubletPtr()->lowerHitPtr()->z());

                ana.tx->pushbackToBranch<float>(TString::Format("sg%d_outer_md_upper_hit_x", ilayer),sgPtr->outerMiniDoubletPtr()->upperHitPtr()->x());
                ana.tx->pushbackToBranch<float>(TString::Format("sg%d_outer_md_upper_hit_y", ilayer),sgPtr->outerMiniDoubletPtr()->upperHitPtr()->y());
                ana.tx->pushbackToBranch<float>(TString::Format("sg%d_outer_md_upper_hit_z", ilayer),sgPtr->outerMiniDoubletPtr()->upperHitPtr()->z());

                ana.tx->pushbackToBranch<int>(TString::Format("sg%d_inner_md_lower_hit_side", ilayer),(sgPtr->innerMiniDoubletPtr()->lowerHitPtr()->getModule()).side());
                ana.tx->pushbackToBranch<int>(TString::Format("sg%d_outer_md_lower_hit_side", ilayer),(sgPtr->innerMiniDoubletPtr()->upperHitPtr()->getModule()).side());
                ana.tx->pushbackToBranch<int>(TString::Format("sg%d_inner_md_lower_hit_module", ilayer),(sgPtr->outerMiniDoubletPtr()->lowerHitPtr()->getModule()).side());
                ana.tx->pushbackToBranch<int>(TString::Format("sg%d_inner_md_upper_hit_module", ilayer),(sgPtr->outerMiniDoubletPtr()->upperHitPtr()->getModule()).side());

                ana.tx->pushbackToBranch<int>(TString::Format("sg%d_outer_md_lower_hit_module", ilayer),(sgPtr->innerMiniDoubletPtr()->lowerHitPtr()->getModule()).module());
                ana.tx->pushbackToBranch<int>(TString::Format("sg%d_outer_md_upper_hit_module", ilayer),(sgPtr->innerMiniDoubletPtr()->lowerHitPtr()->getModule()).module());

                ana.tx->pushbackToBranch<int>(TString::Format("sg%d_inner_md_subdet", ilayer),(sgPtr->innerMiniDoubletPtr()->lowerHitPtr()->getModule()).subdet());
                ana.tx->pushbackToBranch<int>(TString::Format("sg%d_outer_md_subdet", ilayer),(sgPtr->outerMiniDoubletPtr()->lowerHitPtr()->getModule()).subdet());

            }
        }

        for(unsigned int ilayer = 1; ilayer <= 5; ilayer++)
        {
            ana.tx->pushbackToBranch<int>(TString::Format("sgendcap%d_ncand",ilayer),trackevent.getLayer(ilayer,SDL::Layer::Endcap).getSegmentPtrs().size());

            for(auto &sgPtr:trackevent.getLayer(ilayer,SDL::Layer::Endcap).getSegmentPtrs())
            {
                sgPtr->runSegmentAlgo(SDL::Default_SGAlgo);


                ana.tx->pushbackToBranch<float>(TString::Format("sgendcap%d_dphi", ilayer),sgPtr->getRecoVar("deltaPhi"));
                ana.tx->pushbackToBranch<float>(TString::Format("sgendcap%d_dphichange", ilayer),sgPtr->getDeltaPhiChange());
    
                ana.tx->pushbackToBranch<float>(TString::Format("sgendcap%d_dAlpha_innerMD_segment",ilayer),sgPtr->getdAlphaInnerMDSegment());
                ana.tx->pushbackToBranch<float>(TString::Format("sgendcap%d_dAlpha_outerMD_segment",ilayer),sgPtr->getdAlphaOuterMDSegment());
                ana.tx->pushbackToBranch<float>(TString::Format("sgendcap%d_dAlpha_innerMD_outerMD",ilayer),sgPtr->getdAlphaInnerMDOuterMD());

                ana.tx->pushbackToBranch<float>(TString::Format("sgendcap%d_sdCut",ilayer),sgPtr->getRecoVar("sdCut"));
                ana.tx->pushbackToBranch<float>(TString::Format("sgendcap%d_dAlpha_innerMD_segment_threshold",ilayer),sgPtr->getRecoVar("dAlpha_innerMD_segment"));
                ana.tx->pushbackToBranch<float>(TString::Format("sgendcap%d_dAlpha_outerMD_segment_threshold",ilayer),sgPtr->getRecoVar("dAlpha_outerMD_segment"));
                ana.tx->pushbackToBranch<float>(TString::Format("sgendcap%d_dAlpha_innerMD_outerMD_threshold",ilayer),sgPtr->getRecoVar("dAlpha_innerMD_outerMD"));


                ana.tx->pushbackToBranch<int>(TString::Format("sgendcap%d_pass", ilayer),sgPtr->passesSegmentAlgo(SDL::Default_SGAlgo));

                ana.tx->pushbackToBranch<float>(TString::Format("sgendcap%d_inner_md_lower_hit_x", ilayer),sgPtr->innerMiniDoubletPtr()->lowerHitPtr()->x());
                ana.tx->pushbackToBranch<float>(TString::Format("sgendcap%d_inner_md_lower_hit_y", ilayer),sgPtr->innerMiniDoubletPtr()->lowerHitPtr()->y());
                ana.tx->pushbackToBranch<float>(TString::Format("sgendcap%d_inner_md_lower_hit_z", ilayer),sgPtr->innerMiniDoubletPtr()->lowerHitPtr()->z());

                ana.tx->pushbackToBranch<float>(TString::Format("sgendcap%d_inner_md_upper_hit_x", ilayer),sgPtr->innerMiniDoubletPtr()->upperHitPtr()->x());
                ana.tx->pushbackToBranch<float>(TString::Format("sgendcap%d_inner_md_upper_hit_y", ilayer),sgPtr->innerMiniDoubletPtr()->upperHitPtr()->y());
                ana.tx->pushbackToBranch<float>(TString::Format("sgendcap%d_inner_md_upper_hit_z", ilayer),sgPtr->innerMiniDoubletPtr()->upperHitPtr()->z());

                ana.tx->pushbackToBranch<float>(TString::Format("sgendcap%d_outer_md_lower_hit_x", ilayer),sgPtr->outerMiniDoubletPtr()->upperHitPtr()->x());
                ana.tx->pushbackToBranch<float>(TString::Format("sgendcap%d_outer_md_lower_hit_y", ilayer),sgPtr->outerMiniDoubletPtr()->lowerHitPtr()->y());
                ana.tx->pushbackToBranch<float>(TString::Format("sgendcap%d_outer_md_lower_hit_z", ilayer),sgPtr->outerMiniDoubletPtr()->lowerHitPtr()->z());

                ana.tx->pushbackToBranch<float>(TString::Format("sgendcap%d_outer_md_upper_hit_x", ilayer),sgPtr->outerMiniDoubletPtr()->upperHitPtr()->x());
                ana.tx->pushbackToBranch<float>(TString::Format("sgendcap%d_outer_md_upper_hit_y", ilayer),sgPtr->outerMiniDoubletPtr()->upperHitPtr()->y());
                ana.tx->pushbackToBranch<float>(TString::Format("sgendcap%d_outer_md_upper_hit_z", ilayer),sgPtr->outerMiniDoubletPtr()->upperHitPtr()->z());

                ana.tx->pushbackToBranch<int>(TString::Format("sgendcap%d_inner_md_lower_hit_side", ilayer),(sgPtr->innerMiniDoubletPtr()->lowerHitPtr()->getModule()).side());
                ana.tx->pushbackToBranch<int>(TString::Format("sgendcap%d_outer_md_lower_hit_side", ilayer),(sgPtr->innerMiniDoubletPtr()->upperHitPtr()->getModule()).side());
                ana.tx->pushbackToBranch<int>(TString::Format("sgendcap%d_inner_md_lower_hit_module", ilayer),(sgPtr->outerMiniDoubletPtr()->lowerHitPtr()->getModule()).side());
                ana.tx->pushbackToBranch<int>(TString::Format("sgendcap%d_inner_md_upper_hit_module", ilayer),(sgPtr->outerMiniDoubletPtr()->upperHitPtr()->getModule()).side());

                ana.tx->pushbackToBranch<int>(TString::Format("sgendcap%d_outer_md_lower_hit_module", ilayer),(sgPtr->innerMiniDoubletPtr()->lowerHitPtr()->getModule()).module());
                ana.tx->pushbackToBranch<int>(TString::Format("sgendcap%d_outer_md_upper_hit_module", ilayer),(sgPtr->innerMiniDoubletPtr()->lowerHitPtr()->getModule()).module());

                ana.tx->pushbackToBranch<int>(TString::Format("sgendcap%d_inner_md_subdet", ilayer),(sgPtr->innerMiniDoubletPtr()->lowerHitPtr()->getModule()).subdet());
                ana.tx->pushbackToBranch<int>(TString::Format("sgendcap%d_outer_md_subdet", ilayer),(sgPtr->outerMiniDoubletPtr()->lowerHitPtr()->getModule()).subdet());

                ana.tx->pushbackToBranch<float>(TString::Format("sgendcap%d_RtIn",ilayer),(sgPtr->getRtIn()));
                ana.tx->pushbackToBranch<float>(TString::Format("sgendcap%d_RtOut",ilayer),(sgPtr->getRtOut()));

		ana.tx->pushbackToBranch<float>(TString::Format("sgendcap%d_RtLo",ilayer),(sgPtr->getRtLo()));
                ana.tx->pushbackToBranch<float>(TString::Format("sgendcap%d_RtHi",ilayer),(sgPtr->getRtHi()));


            }
        }

        ana.tx->fill();
        ana.tx->clear();

    }

}


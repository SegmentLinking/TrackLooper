# include "StudySimtrackMatchedMDCuts.h"


StudySimtrackMatchedMDCuts::StudySimtrackMatchedMDCuts(const char * studyName)
{
    studyName = studyName;
}


void StudySimtrackMatchedMDCuts::bookStudy()
{
        //SIM TRACK MATCHED STUFF
    ana.histograms.addVecHistogram(TString::Format("sim_matched_MD_dz"),400,-20,20,[&](){return dzValues;});
    ana.histograms.addVecHistogram(TString::Format("sim_matched_MD_dPhi"),200,-6.28,6.28,[&](){return dPhiValues;});
    ana.histograms.addVecHistogram(TString::Format("sim_matched_MD_dPhiChange"),200,-6.28,6.28,[&](){return dPhiChangeValues;});


    ana.histograms.addVecHistogram(TString::Format("sim_matched_MD_barrel_dz"),400,-20,20,[&](){return barreldzValues;});
    ana.histograms.addVecHistogram(TString::Format("sim_matched_MD_barrel_dPhi"),200,-6.28,6.28,[&](){return barreldPhiValues;});
    ana.histograms.addVecHistogram(TString::Format("sim_matched_MD_barrel_dPhiChange"),200,-6.28,6.28,[&](){return barreldPhiChangeValues;});


    ana.histograms.addVecHistogram(TString::Format("sim_matched_MD_endcap_dz"),400,-20,20,[&](){return endcapdzValues;});
    ana.histograms.addVecHistogram(TString::Format("sim_matched_MD_endcap_drt"),400,-20,20,[&](){return endcapdrtValues;});
    ana.histograms.addVecHistogram(TString::Format("sim_matched_MD_endcap_dPhi"),200,-6.28,6.28,[&](){return endcapdPhiValues;});
    ana.histograms.addVecHistogram(TString::Format("sim_matched_MD_endcap_dPhiChange"),200,-6.28,6.28,[&](){return endcapdPhiChangeValues;});

    //one per layer
    for(size_t i = 0; i < 6; i++)
    {
        ana.histograms.addVecHistogram(TString::Format("sim_matched_MD_dz_layer_%ld",i+1),400,-20,20,[&,i](){return layerdzValues[i];});
        ana.histograms.addVecHistogram(TString::Format("sim_matched_MD_dPhi_layer_%ld",i+1),200,-6.28,6.28,[&,i](){return layerdPhiValues[i];});
        ana.histograms.addVecHistogram(TString::Format("sim_matched_MD_dPhiChange_layer_%ld",i+1),200,-6.28,6.28,[&,i](){return layerdPhiChangeValues[i];});


        ana.histograms.addVecHistogram(TString::Format("sim_matched_MD_barrel_dz_layer_%ld",i+1),400,-20,20,[&,i](){return layerBarreldzValues[i];});
        ana.histograms.addVecHistogram(TString::Format("sim_matched_MD_barrel_dPhi_layer_%ld",i+1),200,-6.28,6.28,[&,i](){return layerBarreldPhiValues[i];});
        ana.histograms.addVecHistogram(TString::Format("sim_matched_MD_barrel_dPhiChange_layer_%ld",i+1),200,-6.28,6.28,[&,i](){return layerBarreldPhiChangeValues[i];});

        //Add barrel center modules (non tilted)
        ana.histograms.addVecHistogram(TString::Format("sim_matched_MD_barrel_center_dPhi_layer_%ld",i+1),200,-6.28,6.28,[&,i](){return layerBarrelCenterdPhiValues[i];});
        ana.histograms.addVecHistogram(TString::Format("sim_matched_MD_barrel_center_dPhiChange_layer_%ld",i+1),200,-6.28,6.28,[&,i](){return layerBarrelCenterdPhiChangeValues[i];});


        if(i < 3) //Barrel tilted modules - normal and endcap logic
        {
            ana.histograms.addVecHistogram(TString::Format("sim_matched_MD_dPhi_barrel_normal_tilted_layer_%ld",i+1),200,-6.28,6.28,[&,i](){return layerBarrelNormalTilteddPhiValues[i];});
            ana.histograms.addVecHistogram(TString::Format("sim_matched_MD_dPhi_barrel_endcapLogic_tilted_layer_%ld",i+1),200,-6.28,6.28,[&,i](){return layerBarrelEndcapTilteddPhiValues[i];});   

            ana.histograms.addVecHistogram(TString::Format("sim_matched_MD_dPhiChange_barrel_normal_tilted_layer_%ld",i+1),200,-6.28,6.28,[&,i](){return layerBarrelNormalTilteddPhiChangeValues[i];});
            ana.histograms.addVecHistogram(TString::Format("sim_matched_MD_dPhiChange_barrel_endcapLogic_tilted_layer_%ld",i+1),200,-6.28,6.28,[&,i](){return layerBarrelEndcapTilteddPhiChangeValues[i];});

        }

        if(i < 5)
        {
            ana.histograms.addVecHistogram(TString::Format("sim_matched_MD_endcap_dz_layer_%ld",i+1),400,-20,20,[&,i](){return layerEndcapdzValues[i];});
            ana.histograms.addVecHistogram(TString::Format("sim_matched_MD_endcap_drt_layer_%ld",i+1),400,-20,20,[&,i](){return layerEndcapdrtValues[i];});
            ana.histograms.addVecHistogram(TString::Format("sim_matched_MD_endcap_dPhi_layer_%ld",i+1),200,-6.28,6.28,[&,i](){return layerEndcapdPhiValues[i];});
            ana.histograms.addVecHistogram(TString::Format("sim_matched_MD_endcap_dPhiChange_layer_%ld",i+1),200,-6.28,6.28,[&,i](){return layerEndcapdPhiChangeValues[i];});
        }
    }

    for(size_t i = 0; i < 15; i++)
    {
        ana.histograms.addVecHistogram(TString::Format("sim_matched_MD_endcap_drt_ring_%ld",i+1),400,-20,20,[&,i](){return ringEndcapdrtValues[i];});
    }





}


void StudySimtrackMatchedMDCuts::resetVariables()
{
    dzValues.clear();
    dPhiValues.clear();
    dPhiChangeValues.clear();

    barreldzValues.clear();
    barreldPhiValues.clear();
    barreldPhiChangeValues.clear();

    endcapdzValues.clear();
    endcapdrtValues.clear();
    endcapdPhiValues.clear();
    endcapdPhiChangeValues.clear();

    layerdzValues.clear();
    layerdPhiValues.clear();
    layerdPhiChangeValues.clear();

    layerBarreldzValues.clear();
    layerBarreldPhiValues.clear();
    layerBarreldPhiChangeValues.clear();

    layerEndcapdzValues.clear();
    layerEndcapdrtValues.clear();
    layerEndcapdPhiValues.clear();
    layerEndcapdPhiChangeValues.clear();

    layerBarrelCenterdPhiValues.clear();
    layerBarrelNormalTilteddPhiValues.clear();
    layerBarrelEndcapTilteddPhiValues.clear();

    layerBarrelCenterdPhiChangeValues.clear();
    layerBarrelNormalTilteddPhiChangeValues.clear();
    layerBarrelEndcapTilteddPhiChangeValues.clear();
    
    ringEndcapdrtValues.clear();

    for(size_t i = 1; i <= 6; i++)
    {
        layerdzValues.push_back(std::vector<float>());
        layerdPhiValues.push_back(std::vector<float>());
        layerdPhiChangeValues.push_back(std::vector<float>());

        layerBarreldzValues.push_back(std::vector<float>());
        layerBarreldPhiValues.push_back(std::vector<float>());
        layerBarreldPhiChangeValues.push_back(std::vector<float>());
       
        layerBarrelCenterdPhiValues.push_back(std::vector<float>());
        layerBarrelCenterdPhiChangeValues.push_back(std::vector<float>());
        if(i < 6)
        {
            layerEndcapdzValues.push_back(std::vector<float>());
            layerEndcapdrtValues.push_back(std::vector<float>());
            layerEndcapdPhiValues.push_back(std::vector<float>());
            layerEndcapdPhiChangeValues.push_back(std::vector<float>());            
        }
        

        if(i <= 3)
        {
            layerBarrelNormalTilteddPhiValues.push_back(std::vector<float>());
            layerBarrelEndcapTilteddPhiValues.push_back(std::vector<float>());

            layerBarrelNormalTilteddPhiChangeValues.push_back(std::vector<float>());
            layerBarrelEndcapTilteddPhiChangeValues.push_back(std::vector<float>());
        }
    }

    for(size_t i = 0;i < 15; i++)
    {
        ringEndcapdrtValues.push_back(std::vector<float>());
    }

}

void StudySimtrackMatchedMDCuts::doStudy(SDL::Event &event,std::vector<std::tuple<unsigned int, SDL::Event*>> simtrkevents)
{
    resetVariables();

    float dzCut = 10.0;
   
    //Every sim track is stored in an Event* container, and an event consists of a list of such containers
    for(auto &matchedTrack:simtrkevents)
    {
        std::vector<SDL::Module*> moduleList = std::get<1>(matchedTrack)->getLowerModulePtrs();
        for(auto &module:moduleList)
        {
            std::vector<SDL::MiniDoublet*> miniDoublets = module->getMiniDoubletPtrs();
            for(auto &md:miniDoublets)
            {
                //Step 1 : Reproducing Philip's plots
//                md->runMiniDoubletAlgo(SDL::Default_MDAlgo);

                dzValues.push_back(md->getDz());
                dPhiValues.push_back(md->getDeltaPhi());
                dPhiChangeValues.push_back(md->getDeltaPhiChange());
                layerdzValues.at(module->layer()-1).push_back(md->getDz());
                layerdPhiValues.at(module->layer()-1).push_back(md->getDeltaPhi());
                layerdPhiChangeValues.at(module->layer()-1).push_back(md->getDeltaPhiChange());


                if(module->subdet() == SDL::Module::Barrel)
                {
                    barreldzValues.push_back(md->getDz());
                    barreldPhiValues.push_back(md->getDeltaPhi());
                    barreldPhiChangeValues.push_back(md->getDeltaPhiChange());
                    layerBarreldzValues.at(module->layer()-1).push_back(md->getDz());
                    layerBarreldPhiValues.at(module->layer()-1).push_back(md->getDeltaPhi());
                    layerBarreldPhiChangeValues.at(module->layer()-1).push_back(md->getDeltaPhiChange());

                    layerBarreldPhiChangeValues.at(module->layer()-1).push_back(md->getDeltaPhiChange());


                    
                    if(module->side() == SDL::Module::Center)
                    {
                        layerBarrelCenterdPhiValues[module->layer()-1].push_back(md->getDeltaPhi());
                        layerBarrelCenterdPhiChangeValues[module->layer()-1].push_back(md->getDeltaPhiChange());

                    }
                    else
                    {
                        if(SDL::MiniDoublet::isNormalTiltedModules(*module))
                        {
                            layerBarrelNormalTilteddPhiValues[module->layer()-1].push_back(md->getDeltaPhi());
                            layerBarrelNormalTilteddPhiChangeValues[module->layer()-1].push_back(md->getDeltaPhiChange());

                        }
                        else
                        {
                            layerBarrelEndcapTilteddPhiValues[module->layer()-1].push_back(md->getDeltaPhi());
                            layerBarrelEndcapTilteddPhiChangeValues[module->layer()-1].push_back(md->getDeltaPhiChange());
                        }
                    }
                }

                else if(module->subdet() == SDL::Module::Endcap)
                {

                    endcapdzValues.push_back(md->getDz());
                    endcapdrtValues.push_back(md->getDrt());
                    endcapdPhiValues.push_back(md->getDeltaPhi());
                    endcapdPhiChangeValues.push_back(md->getDeltaPhiChange());
                    layerEndcapdzValues.at(module->layer()-1).push_back(md->getDz());
                    layerEndcapdrtValues.at(module->layer()-1).push_back(md->getDrt());
                    ringEndcapdrtValues.at(module->ring()-1).push_back(md->getDrt());
                    layerEndcapdPhiValues.at(module->layer()-1).push_back(md->getDeltaPhi());
                    layerEndcapdPhiChangeValues.at(module->layer()-1).push_back(md->getDeltaPhiChange());
                }
            }

        }
    }
}

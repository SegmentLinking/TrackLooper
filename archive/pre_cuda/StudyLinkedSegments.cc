# include "StudyLinkedSegments.h"

StudyLinkedSegments::StudyLinkedSegments(const char* studyName)
{
    studyname = studyName;
}

void StudyLinkedSegments::bookStudy()
{
    ana.histograms.addVecHistogram(TString::Format("Linked_Segment_occupancy_in_barrel"),5000,0,5000,[&](){return BarrelLinkedSegmentOccupancy;});
    ana.histograms.addHistogram(TString::Format("average_Linked_Segment_occupancy_in_barrel"),50000,0,5000,[&](){return averageBarrelLinkedSegmentOccupancy;});
    ana.histograms.addVecHistogram(TString::Format("Linked_Segment_occupancy_in_endcap"),5000,0,5000,[&](){return EndcapLinkedSegmentOccupancy;});
    ana.histograms.addHistogram(TString::Format("average_Linked_Segment_occupancy_in_endcap"),50000,0,5000,[&](){return averageEndcapLinkedSegmentOccupancy;});

    for(int i = 0; i<6;i++)
    {
        ana.histograms.addVecHistogram(TString::Format("Linked_Segment_occupancy_in_barrel_for_layer_%d",i+1),5000,0,5000,[&,i](){return LayerBarrelLinkedSegmentOccupancy[i];});
        ana.histograms.addHistogram(TString::Format("average_Linked_Segment_occupancy_in_barrel_for_layer_%d",i+1),50000,0,5000,[&,i](){return averageLayerBarrelLinkedSegmentOccupancy[i];});

        ana.histograms.addVecHistogram(TString::Format("Linked_Segment_occupancy_in_endcap_for_layer_%d",i+1),5000,0,5000,[&,i](){return LayerEndcapLinkedSegmentOccupancy[i];});
        ana.histograms.addHistogram(TString::Format("average_Linked_Segment_occupancy_in_endcap_for_layer_%d",i+1),50000,0,5000,[&,i](){return averageLayerEndcapLinkedSegmentOccupancy[i];});

        for(int j = 0; j<15; j++)
        {
            ana.histograms.addVecHistogram(TString::Format("Linked_Segment_occupancy_layer_%d_ring_%d",i+1,j+1),5000,0,5000,[&,i,j](){return EndcapLayerRingLinkedSegmentOccupancy[i][j];});
        }

    }

    for(int i = 0; i<15; i++)
    {
        ana.histograms.addVecHistogram(TString::Format("Linked_Segment_occupancy_in_endcap_for_ring_%d",i+1),5000,0,5000,[&,i](){return EndcapRingLinkedSegmentOccupancy[i];});
        ana.histograms.addHistogram(TString::Format("average_Linked_Segment_occupancy_in_endcap_for_ring_%d",i+1),50000,0,5000,[&,i](){return averageEndcapRingLinkedSegmentOccupancy[i];});
    }
}

void StudyLinkedSegments::doStudy(SDL::Event &event, std::vector<std::tuple<unsigned int, SDL::Event*>> simtrkevents)
{
    averageBarrelLinkedSegmentOccupancy = 0;
    averageEndcapLinkedSegmentOccupancy = 0;
    int nBarrelModules = 0, nEndcapModules = 0;

    BarrelLinkedSegmentOccupancy.clear();
    EndcapLinkedSegmentOccupancy.clear();

    averageLayerBarrelLinkedSegmentOccupancy.clear();
    averageLayerEndcapLinkedSegmentOccupancy.clear();
    averageEndcapRingLinkedSegmentOccupancy.clear();

    LayerBarrelLinkedSegmentOccupancy.clear();
    LayerEndcapLinkedSegmentOccupancy.clear();
    EndcapLayerRingLinkedSegmentOccupancy.clear();
    EndcapRingLinkedSegmentOccupancy.clear();

    for(int i = 0; i<15;i++)
    {
        averageEndcapRingLinkedSegmentOccupancy.push_back(0);
        EndcapRingLinkedSegmentOccupancy.push_back(std::vector<float>());
    }
    
    for(int i =0; i<6;i++)
    {
        LayerEndcapLinkedSegmentOccupancy.push_back(std::vector<float>());
        LayerBarrelLinkedSegmentOccupancy.push_back(std::vector<float>());
        averageLayerEndcapLinkedSegmentOccupancy.push_back(0);
        averageLayerBarrelLinkedSegmentOccupancy.push_back(0);

        EndcapLayerRingLinkedSegmentOccupancy.push_back(std::vector<std::vector<float>>());
        for(int j = 0; j<15; j++)
        {
            EndcapLayerRingLinkedSegmentOccupancy[i].push_back(std::vector<float>());
        }
    }
    
    std::vector<int> nBarrelLayerModules(6,0);
    std::vector<int> nEndcapLayerModules(5,0);
    std::vector<int> nEndcapRingModules(15,0);


    std::vector<SDL::Module*> moduleList = event.getModulePtrs();

    for(auto &module:moduleList)
    {
        std::vector<unsigned int> lowerConnectedModuleDetIds = SDL::moduleConnectionMap.getConnectedModuleDetIds(module->detId());
        std::vector<unsigned int> connectedModuleDetIds;

        for(auto &lowerConnectedModuleId:lowerConnectedModuleDetIds)
        {
            SDL::Module &lowerConnectedModule = event.getModule(lowerConnectedModuleId);
            std::vector<unsigned int> upperConnectedModuleDetIds = SDL::moduleConnectionMap.getConnectedModuleDetIds(lowerConnectedModule.detId());
            for(auto &connectedModuleId:upperConnectedModuleDetIds)
            {
                connectedModuleDetIds.push_back(connectedModuleId);
            }
        }

        int connectedModules = 0, connectedModuleOccupancy = 0;
        for(auto &connectedModuleId:connectedModuleDetIds)
        {
            SDL::Module &connectedModule = event.getModule(connectedModuleId);
            connectedModuleOccupancy += connectedModule.getSegmentPtrs().size(); 
        } 

        //Fill the histogram vectors
        if(module->subdet() == SDL::Module::Barrel)
        {
            if((module->getSegmentPtrs()).size() != 0) //What talking about linked segments and tracklets when you don't have segments yourself
            {
                BarrelLinkedSegmentOccupancy.push_back(connectedModuleOccupancy);
                averageBarrelLinkedSegmentOccupancy++;
                LayerBarrelLinkedSegmentOccupancy.at(module->layer()-1).push_back(connectedModuleOccupancy);
                averageLayerBarrelLinkedSegmentOccupancy.at(module->layer()-1) += connectedModuleOccupancy;
                nBarrelLayerModules.at(module->layer()-1)++;
                nBarrelModules ++;
            }
        }
        else if(module->subdet() == SDL::Module::Endcap)
        {
            if((module->getSegmentPtrs()).size() == 0)
            {
                EndcapLinkedSegmentOccupancy.push_back(connectedModuleOccupancy);
                averageEndcapLinkedSegmentOccupancy++;
                LayerEndcapLinkedSegmentOccupancy.at(module->layer()-1).push_back(connectedModuleOccupancy);
                averageLayerEndcapLinkedSegmentOccupancy.at(module->layer()-1) += connectedModuleOccupancy;

                EndcapRingLinkedSegmentOccupancy.at(module->ring()-1).push_back(connectedModuleOccupancy);
                averageEndcapRingLinkedSegmentOccupancy.at(module->ring()-1) += connectedModuleOccupancy;
                nEndcapRingModules.at(module->ring()-1)++;

                EndcapLayerRingLinkedSegmentOccupancy[module->layer()-1][module->ring()-1].push_back(connectedModuleOccupancy);
                nEndcapModules ++;
            }
        }
    }

    averageBarrelLinkedSegmentOccupancy = (nBarrelModules != 0) ? averageBarrelLinkedSegmentOccupancy / nBarrelModules : 0;
    averageEndcapLinkedSegmentOccupancy = (nEndcapModules != 0) ? averageEndcapLinkedSegmentOccupancy / nEndcapModules : 0;

    for(int i = 0; i<6; i++)
    {
        averageLayerBarrelLinkedSegmentOccupancy[i] = (nBarrelLayerModules[i] != 0) ? averageLayerBarrelLinkedSegmentOccupancy[i]/nBarrelLayerModules[i] : 0;
        averageLayerEndcapLinkedSegmentOccupancy[i] = (nEndcapLayerModules[i] != 0) ? averageLayerEndcapLinkedSegmentOccupancy[i]/nEndcapLayerModules[i] : 0;
    }

    for(int i = 0; i<15; i++)
    {
        averageEndcapRingLinkedSegmentOccupancy[i] = (nEndcapRingModules[i] != 0) ? averageEndcapRingLinkedSegmentOccupancy[i]/nEndcapRingModules[i] : 0;
    }
}

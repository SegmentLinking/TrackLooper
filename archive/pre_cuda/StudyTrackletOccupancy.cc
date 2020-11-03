# include "StudyTrackletOccupancy.h"

StudyTrackletOccupancy::StudyTrackletOccupancy(const char* studyName)
{
    studyname = studyName;
}

void StudyTrackletOccupancy::bookStudy()
{
    ana.histograms.addVecHistogram(TString::Format("Tracklet_occupancy_in_barrel"),5000,0,5000,[&](){return occupancyInBarrel;});

    ana.histograms.addHistogram(TString::Format("average_Tracklet_occupancy_in_barrel"),50000,0,5000,[&](){return averageOccupancyInBarrel;});

    ana.histograms.addVecHistogram(TString::Format("Tracklet_occupancy_in_endcap"),5000,0,5000,[&](){return occupancyInEndcap;});

    ana.histograms.addHistogram(TString::Format("average_Tracklet_occupancy_in_endcap"),50000,0,5000,[&](){return averageOccupancyInEndcap;});

    for(int i = 0; i<6; i++)
    {
        ana.histograms.addHistogram(TString::Format("average_Tracklet_occupancy_in_layer_%d",i+1),50000,0,5000,[&,i](){return averageLayerOccupancy[i];});

        ana.histograms.addVecHistogram(TString::Format("Tracklet_occupancy_in_layer_%d",i+1),5000,0,5000,[&,i](){return LayerOccupancy.at(i);});

        ana.histograms.addHistogram(TString::Format("average_Tracklet_occupancy_in_barrel_for_layer_%d",i+1),50000,0,5000,[&,i](){return averageBarrelLayerOccupancy[i];});

        ana.histograms.addVecHistogram(TString::Format("Tracklet_occupancy_in_barrel_for_layer_%d",i+1),5000,0,5000,[&,i](){return BarrelLayerOccupancy.at(i);});

        ana.histograms.addHistogram(TString::Format("average_Tracklet_occupancy_in_endcap_for_layer_%d",i+1),50000,0,5000,[&,i](){return averageEndcapLayerOccupancy[i];});

        ana.histograms.addVecHistogram(TString::Format("Tracklet_occupancy_in_endcap_for_layer_%d",i+1),5000,0,5000,[&,i](){return EndcapLayerOccupancy.at(i);});

        for(int j = 0; j<15; j++)
        {
            ana.histograms.addVecHistogram(TString::Format("Tracklet_occupancy_layer_%d_ring_%d",i+1,j+1),5000,0,5000,[&,i,j](){return EndcapLayerRingTrackletOccupancy[i][j];});
        }

    }

    for(int i=0;i<15;i++)
    {
      ana.histograms.addHistogram(TString::Format("average_Tracklet_occupancy_in_endcap_for_ring_%d",i+1),50000,0,5000,[&,i](){return averageEndcapRingOccupancy[i];});

      ana.histograms.addVecHistogram(TString::Format("Tracklet_occupancy_in_endcap_for_ring_%d",i+1),5000,0,5000,[&,i](){return EndcapRingOccupancy.at(i);});
    }

}

void StudyTrackletOccupancy::doStudy(SDL::Event &event, std::vector<std::tuple<unsigned int, SDL::Event*>> simtrkevents)
{
    std::vector<SDL::Module*> moduleList = event.getLowerModulePtrs();

    averageOccupancyInBarrel = 0;
    averageOccupancyInEndcap = 0;
    averageLayerOccupancy.clear();
    averageLayerOccupancy = {0,0,0,0,0,0};
    std::vector<int> nLayerModules = {0,0,0,0,0,0};
    averageBarrelLayerOccupancy.clear();
    averageBarrelLayerOccupancy = {0,0,0,0,0,0};
    std::vector<int> nBarrelLayerModules = {0,0,0,0,0,0};
    averageEndcapLayerOccupancy.clear();
    averageEndcapLayerOccupancy = {0,0,0,0,0,0};
    std::vector<int> nEndcapLayerModules = {0,0,0,0,0,0};
    std::vector<int> nEndcapRingModules(15,0);
    int nBarrelModules = 0, nEndcapModules = 0;
    averageEndcapRingOccupancy.clear();

    for(int i=0;i<15;i++)
    {
        averageEndcapRingOccupancy.push_back(0);
    }

    occupancyInBarrel.clear();
    occupancyInEndcap.clear();

    //Layer occupancy
    LayerOccupancy.clear();
    BarrelLayerOccupancy.clear();
    EndcapLayerOccupancy.clear();
    EndcapRingOccupancy.clear();
    EndcapLayerRingTrackletOccupancy.clear();
    //setting up sub-vectors for the barrel and endcap layer occupancy
    for(int i = 1; i<=6;i++)
    {
        LayerOccupancy.push_back(std::vector<float>());
        BarrelLayerOccupancy.push_back(std::vector<float>());
        EndcapLayerOccupancy.push_back(std::vector<float>());

        EndcapLayerRingTrackletOccupancy.push_back(std::vector<std::vector<float>>());
        for(int j = 0; j<15; j++)
        {

            EndcapLayerRingTrackletOccupancy[i-1].push_back(std::vector<float>());
        
        }

    }
    for(int i=0;i<15;i++)
    {
        EndcapRingOccupancy.push_back(std::vector<float>());
    }

    for(auto &module:moduleList)
    {
        averageLayerOccupancy.at(module->layer()-1) += (module->getTrackletPtrs()).size();
        if((module->getTrackletPtrs()).size() != 0)
        {
            nLayerModules.at(module->layer()-1)++;
        }
        LayerOccupancy.at(module->layer()-1).push_back((module->getTrackletPtrs()).size());
        if(module->subdet() == SDL::Module::Barrel) //barrel module
        {
            averageOccupancyInBarrel += (module->getTrackletPtrs()).size();
            if((module->getTrackletPtrs()).size() != 0)
            {
                nBarrelModules ++;
            }
            occupancyInBarrel.push_back((module->getTrackletPtrs()).size());
            averageBarrelLayerOccupancy.at(module->layer()-1) += (module->getTrackletPtrs()).size();

            nBarrelLayerModules.at(module->layer()-1) ++;
            if((module->getTrackletPtrs()).size() != 0)
            {
            BarrelLayerOccupancy.at(module->layer()-1).push_back((module->getTrackletPtrs().size()));
            }
        }
        else if(module->subdet() == SDL::Module::Endcap) //endcap module
        {
            averageOccupancyInEndcap += (module->getTrackletPtrs()).size();
            if((module->getTrackletPtrs()).size() != 0)
            {
                nEndcapModules ++;
            }
            occupancyInEndcap.push_back((module->getTrackletPtrs()).size());

            averageEndcapLayerOccupancy.at(module->layer()-1) += (module->getTrackletPtrs()).size();

            if((module->getTrackletPtrs()).size() != 0)
            {
                nEndcapLayerModules.at(module->layer()-1) ++;
            }

            EndcapLayerOccupancy.at(module->layer()-1).push_back((module->getTrackletPtrs().size()));


            averageEndcapRingOccupancy.at(module->ring()-1) += (module->getTrackletPtrs()).size();

            if((module->getTrackletPtrs()).size() != 0)
            {
                nEndcapRingModules.at(module->ring()-1) ++;
            }

            EndcapRingOccupancy.at(module->ring()-1).push_back((module->getTrackletPtrs().size()));

            EndcapLayerRingTrackletOccupancy[module->layer()-1][module->ring()-1].push_back(module->getTrackletPtrs().size());

        }
    }


    averageOccupancyInBarrel = (nBarrelModules != 0) ? averageOccupancyInBarrel / nBarrelModules : 0;
    averageOccupancyInEndcap = (nEndcapModules != 0) ? averageOccupancyInEndcap / nEndcapModules : 0;

    for(int i=0;i<6;i++)
    {

        averageBarrelLayerOccupancy[i] = (nBarrelLayerModules[i] != 0) ? averageBarrelLayerOccupancy[i]/nBarrelLayerModules[i] : 0;

        averageEndcapLayerOccupancy[i] = (nEndcapLayerModules[i] != 0) ? averageEndcapLayerOccupancy[i]/nEndcapLayerModules[i] : 0;

        averageLayerOccupancy[i] = (nLayerModules[i] != 0) ? averageLayerOccupancy[i]/nLayerModules[i] : 0;
    }

    for(int i=0;i<15;i++)
    {
        averageEndcapRingOccupancy.at(i) = (nEndcapRingModules.at(i) !=0) ? averageEndcapRingOccupancy.at(i)/nEndcapRingModules.at(i) : 0;
    }

}

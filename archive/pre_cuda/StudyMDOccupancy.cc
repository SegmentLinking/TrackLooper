# include "StudyMDOccupancy.h"

StudyMDOccupancy::StudyMDOccupancy(const char *studyName)
{
    studyname = studyName;
}

void StudyMDOccupancy::bookStudy()
{
    ana.histograms.addVecHistogram(TString::Format("MD_occupancy_in_barrel"),100,0,100,[&](){return occupancyInBarrel;});

    ana.histograms.addHistogram(TString::Format("average_MD_occupancy_in_barrel"),1000,0,100,[&](){return averageOccupancyInBarrel;});

    ana.histograms.addVecHistogram(TString::Format("MD_occupancy_in_endcap"),100,0,100,[&](){return occupancyInEndcap;});

    ana.histograms.addHistogram(TString::Format("average_MD_occupancy_in_endcap"),1000,0,100,[&](){return averageOccupancyInEndcap;});

    for(int i = 0; i<6; i++)
    {
      ana.histograms.addHistogram(TString::Format("average_MD_occupancy_in_layer_%d",i+1),1000,0,100,[&,i](){return averageLayerOccupancy[i];});

      ana.histograms.addVecHistogram(TString::Format("MD_occupancy_in_layer_%d",i+1),100,0,100,[&,i](){return LayerOccupancy.at(i);});

      ana.histograms.addHistogram(TString::Format("average_MD_occupancy_in_barrel_for_layer_%d",i+1),1000,0,100,[&,i](){return averageBarrelLayerOccupancy[i];});

      ana.histograms.addVecHistogram(TString::Format("MD_occupancy_in_barrel_for_layer_%d",i+1),100,0,100,[&,i](){return BarrelLayerOccupancy.at(i);});

      ana.histograms.addHistogram(TString::Format("average_MD_occupancy_in_endcap_for_layer_%d",i+1),1000,0,100,[&,i](){return averageEndcapLayerOccupancy[i];});

      ana.histograms.addVecHistogram(TString::Format("MD_occupancy_in_endcap_for_layer_%d",i+1),100,0,100,[&,i](){return EndcapLayerOccupancy.at(i);});

    }

    for(int i=0;i<15;i++)
    {
      ana.histograms.addHistogram(TString::Format("average_MD_occupancy_in_endcap_for_ring_%d",i+1),1000,0,100,[&,i](){return averageEndcapRingOccupancy[i];});

      ana.histograms.addVecHistogram(TString::Format("MD_occupancy_in_endcap_for_ring_%d",i+1),100,0,100,[&,i](){return EndcapRingOccupancy.at(i);});
    }
}

void StudyMDOccupancy::doStudy(SDL::Event &event,std::vector<std::tuple<unsigned int, SDL::Event*>> simtrkevents)
{
    std::vector<SDL::Module*> moduleList = event.getLowerModulePtrs();

    averageOccupancyInBarrel = 0;
    averageOccupancyInEndcap = 0;
    averageLayerOccupancy.clear();
    averageLayerOccupancy  = {0,0,0,0,0,0};
    std::vector<int> nLayerModules = {0,0,0,0,0,0};
    averageBarrelLayerOccupancy.clear();
    averageBarrelLayerOccupancy = {0,0,0,0,0,0};
    std::vector<int>nBarrelLayerModules = {0,0,0,0,0,0};
    averageEndcapLayerOccupancy.clear();
    averageEndcapLayerOccupancy = {0,0,0,0,0,0};
    std::vector<int> nEndcapLayerModules = {0,0,0,0,0,0};
    std::vector<int> nEndcapRingModules(15,0);
    int nBarrelModules = 0, nEndcapModules = 0;

    averageEndcapRingOccupancy.clear();
    for(int i=0;i<15;i++)
      averageEndcapRingOccupancy.push_back(0);

    occupancyInBarrel.clear();
    occupancyInEndcap.clear();

    //Layer occupancy
    LayerOccupancy.clear();
    BarrelLayerOccupancy.clear();
    EndcapLayerOccupancy.clear();
    EndcapRingOccupancy.clear();
    //setting up sub-vectors for the barrel and endcap layer occupancy
    for(int i = 1; i<=6;i++)
    {
        LayerOccupancy.push_back(std::vector<float>());
        BarrelLayerOccupancy.push_back(std::vector<float>());
        EndcapLayerOccupancy.push_back(std::vector<float>());
    }
    for(int i=0;i<15;i++)
      EndcapRingOccupancy.push_back(std::vector<float>());


    //Loop over modules, and get mini-doublet occupancy

    for(auto &module:moduleList)
    {
      averageLayerOccupancy.at(module->layer()-1) += (module->getMiniDoubletPtrs()).size();
      if((module->getMiniDoubletPtrs()).size() != 0)
      {
        nLayerModules.at(module->layer()-1)++;
      }
      LayerOccupancy.at(module->layer()-1).push_back((module->getMiniDoubletPtrs()).size());
      if(module->subdet() == SDL::Module::Barrel) //barrel module
      {
        averageOccupancyInBarrel += (module->getMiniDoubletPtrs()).size();
        if((module->getMiniDoubletPtrs()).size() != 0)
        {
          nBarrelModules ++;
        }
        occupancyInBarrel.push_back((module->getMiniDoubletPtrs()).size());

        averageBarrelLayerOccupancy.at(module->layer()-1) += (module->getMiniDoubletPtrs()).size();

        nBarrelLayerModules.at(module->layer()-1) ++;
        if((module->getMiniDoubletPtrs()).size() != 0)
        {
          BarrelLayerOccupancy.at(module->layer()-1).push_back((module->getMiniDoubletPtrs().size()));
        }
      }
      else if(module->subdet() == SDL::Module::Endcap) //endcap module
      {
        averageOccupancyInEndcap += (module->getMiniDoubletPtrs()).size();
        if((module->getMiniDoubletPtrs()).size() != 0)
        {
          nEndcapModules ++;
        }
        occupancyInEndcap.push_back((module->getMiniDoubletPtrs()).size());

        averageEndcapLayerOccupancy.at(module->layer()-1) += (module->getMiniDoubletPtrs()).size();

        if((module->getMiniDoubletPtrs()).size() != 0)
        {
          nEndcapLayerModules.at(module->layer()-1) ++;
        }

        EndcapLayerOccupancy.at(module->layer()-1).push_back((module->getMiniDoubletPtrs().size()));


        averageEndcapRingOccupancy.at(module->ring()-1) += (module->getMiniDoubletPtrs()).size();

        if((module->getMiniDoubletPtrs()).size() != 0)
        {
          nEndcapRingModules.at(module->ring()-1) ++;
        }

        EndcapRingOccupancy.at(module->ring()-1).push_back((module->getMiniDoubletPtrs().size()));
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

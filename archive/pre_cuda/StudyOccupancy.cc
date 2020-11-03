# include "StudyOccupancy.h"

StudyOccupancy::StudyOccupancy(const char *studyName)
{
  studyname = studyName;
}

void StudyOccupancy::bookStudy()
{

    //adding vector histograms for dumping occupancy numbers

    ana.histograms.addVecHistogram(TString::Format("occupancy_in_barrel"),100,0,100,[&](){return occupancyInBarrel;});

    ana.histograms.addHistogram(TString::Format("average_occupancy_in_barrel"),1000,0,100,[&](){return averageOccupancyInBarrel;});

    ana.histograms.addVecHistogram(TString::Format("occupancy_in_endcap"),100,0,100,[&](){return occupancyInEndcap;});

    ana.histograms.addHistogram(TString::Format("average_occupancy_in_endcap"),100,0,100,[&](){return averageOccupancyInEndcap;});

    for(int i = 0; i<6; i++)
    {
      ana.histograms.addHistogram(TString::Format("average_occupancy_in_layer_%d",i+1),1000,0,100,[&,i](){return averageLayerOccupancy[i];});

      ana.histograms.addVecHistogram(TString::Format("occupancy_in_layer_%d",i+1),100,0,100,[&,i](){return LayerOccupancy.at(i);});

      ana.histograms.addHistogram(TString::Format("average_occupancy_in_barrel_for_layer_%d",i+1),1000,0,100,[&,i](){return averageBarrelLayerOccupancy[i];});

      ana.histograms.addVecHistogram(TString::Format("occupancy_in_barrel_for_layer_%d",i+1),100,0,100,[&,i](){return BarrelLayerOccupancy.at(i);});

      ana.histograms.addHistogram(TString::Format("average_occupancy_in_endcap_for_layer_%d",i+1),1000,0,100,[&,i](){return averageEndcapLayerOccupancy[i];});

      ana.histograms.addVecHistogram(TString::Format("occupancy_in_endcap_for_layer_%d",i+1),100,0,100,[&,i](){return EndcapLayerOccupancy.at(i);});

    }

    for(int i=0;i<15;i++)
    {
      ana.histograms.addHistogram(TString::Format("average_occupancy_in_endcap_for_ring_%d",i+1),1000,0,100,[&,i](){return averageEndcapRingOccupancy[i];});

      ana.histograms.addVecHistogram(TString::Format("occupancy_in_endcap_for_ring_%d",i+1),100,0,100,[&,i](){return EndcapRingOccupancy.at(i);});
    }

}

void StudyOccupancy::doStudy(SDL::Event& event, std::vector<std::tuple<unsigned int, SDL::Event*>> simtrkevents)
{
    //Get a list of modules in the event
    std::vector<SDL::Module*> moduleList = event.getModulePtrs();
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
    {
      EndcapRingOccupancy.push_back(std::vector<float>());
    }

    //To get the occupancy, iterate through the modules and get the length
    //of the vector of pointers containing the hits on that module
    for(auto &module:moduleList)
    {
      averageLayerOccupancy.at(module->layer()-1) += (module->getHitPtrs()).size();
      nLayerModules.at(module->layer()-1)++;
      LayerOccupancy.at(module->layer()-1).push_back((module->getHitPtrs()).size());
      if(module->subdet() == SDL::Module::Barrel) //barrel module
      {
        averageOccupancyInBarrel += (module->getHitPtrs()).size();
        nBarrelModules ++;

        occupancyInBarrel.push_back((module->getHitPtrs()).size());

        averageBarrelLayerOccupancy.at(module->layer()-1) += (module->getHitPtrs()).size();
        nBarrelLayerModules.at(module->layer()-1) ++;

        BarrelLayerOccupancy.at(module->layer()-1).push_back((module->getHitPtrs().size()));

      }
      else if(module->subdet() == SDL::Module::Endcap) //endcap module
      {
        averageOccupancyInEndcap += (module->getHitPtrs()).size();
        nEndcapModules ++;

        occupancyInEndcap.push_back((module->getHitPtrs()).size());

        averageEndcapLayerOccupancy.at(module->layer()-1) += (module->getHitPtrs()).size();
        nEndcapLayerModules.at(module->layer()-1) ++;

        EndcapLayerOccupancy.at(module->layer()-1).push_back((module->getHitPtrs().size()));


        averageEndcapRingOccupancy.at(module->ring()-1) += (module->getHitPtrs()).size();
        nEndcapRingModules.at(module->ring()-1) ++;

        EndcapRingOccupancy.at(module->ring()-1).push_back((module->getHitPtrs().size()));


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

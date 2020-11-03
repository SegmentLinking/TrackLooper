# ifndef StudyLinkedModule_h
#define StudyLinkedModule_h

# include "Study.h"
# include "SDL/Event.h"
# include <vector>
# include <tuple>

# include "TString.h"
# include "trktree.h"
# include "constants.h"

# include "AnalysisConfig.h"

class StudyLinkedModule : public Study
{
  public:
    const char *studyname;
    std::vector<float> BarrelLinkedModuleOccupancy;
    std::vector<float> EndcapLinkedModuleOccupancy;
    std::vector<float> nBarrelLinkedModules;
    std::vector<float> nEndcapLinkedModules;

    float averageBarrelLinkedModuleOccupancy;
    float averageEndcapLinkedModuleOccupancy;
    int averagenBarrelLinkedModules;
    int averagenEndcapLinkedModules;

    std::vector<std::vector<float>> LayerBarrelLinkedModuleOccupancy;
    std::vector<std::vector<float>> nLayerBarrelLinkedModules;

    std::vector<float> averageLayerBarrelLinkedModuleOccupancy;
    std::vector<int> averagenLayerBarrelLinkedModules;

    std::vector<std::vector<float>> LayerEndcapLinkedModuleOccupancy;
    std::vector<std::vector<float>> nLayerEndcapLinkedModules;

    std::vector<float> averageLayerEndcapLinkedModuleOccupancy;
    std::vector<int> averagenLayerEndcapLinkedModules;

    std::vector<std::vector<float>> RingEndcapLinkedModuleOccupancy;
    std::vector<std::vector<float>> nRingEndcapLinkedModules;

    std::vector<float> averageEndcapRingLinkedModuleOccupancy;
    std::vector<int> averagenRingEndcapLinkedModules;

    std::vector<std::vector<float>> EndcapInnerRingLinkedModuleOccupancy;
    std::vector<std::vector<float>> nEndcapInnerRingLinkedModules;

    std::vector<float> averageEndcapInnerRingLinkedModuleOccupancy;
    std::vector<int> nInnerRingEndcapLinkedModules;

    std::vector<std::vector<float>> EndcapOuterRingLinkedModuleOccupancy;
    std::vector<std::vector<float>> nEndcapOuterRingLinkedModuleOccupancy;

    std::vector<float> averageEndcapOuterRingLinkedModuleOccupancy;
    std::vector<int> nOuterRingEndcapLinkedModules;

    std::vector<std::vector<std::vector<float>>> EndcapLayerRingLinkedModuleOccupancy;
    std::vector<std::vector<std::vector<float>>> nEndcapLayerRingLinkedModules;

    StudyLinkedModule(const char* studyName);
    virtual void bookStudy();
    virtual void doStudy(SDL::Event &event,std::vector<std::tuple<unsigned int, SDL::Event*>> simtrkevents);
};
#endif

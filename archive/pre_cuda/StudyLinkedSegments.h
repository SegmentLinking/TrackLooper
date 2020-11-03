#ifndef StudyLinkedSegments_h
#define StudyLinkedSegments_h
# include "Study.h"
# include "SDL/Event.h"
# include <vector>
# include <tuple>

# include "TString.h"
# include "trktree.h"
# include "constants.h"

# include "AnalysisConfig.h"

class StudyLinkedSegments : public Study
{
    public:
        const char* studyname;
        std::vector<float> BarrelLinkedSegmentOccupancy;
        float averageBarrelLinkedSegmentOccupancy;
        std::vector<float> EndcapLinkedSegmentOccupancy;
        float averageEndcapLinkedSegmentOccupancy;

        std::vector<std::vector<float>> LayerBarrelLinkedSegmentOccupancy;
        std::vector<std::vector<float>> LayerEndcapLinkedSegmentOccupancy;
        std::vector<float> averageLayerBarrelLinkedSegmentOccupancy;
        std::vector<float> averageLayerEndcapLinkedSegmentOccupancy;

        std::vector<std::vector<float>> EndcapRingLinkedSegmentOccupancy;
        std::vector<float> averageEndcapRingLinkedSegmentOccupancy;

        std::vector<std::vector<std::vector<float>>> EndcapLayerRingLinkedSegmentOccupancy;

        StudyLinkedSegments(const char* studyName);
        virtual void bookStudy();
        virtual void doStudy(SDL::Event &event, std::vector<std::tuple<unsigned int, SDL::Event*>> simtrkevents);
};
#endif

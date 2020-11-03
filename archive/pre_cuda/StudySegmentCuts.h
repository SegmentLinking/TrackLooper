#ifndef StudySegmentCuts_h
#define StudySegmentCuts_h

# include "Study.h"
# include "SDL/Event.h"
# include "SDL/Segment.h"
# include <vector>
# include <tuple>

# include "TString.h"
# include "trktree.h"
# include "constants.h"

# include "AnalysisConfig.h"

class StudySegmentCuts : public Study
{
    public:
        const char * studyName;

        std::vector<float> dzDiffLowValues;
        std::vector<float> dzDiffHighValues;
        std::vector<float> dPhiValues;
        std::vector<float> dPhiChangeValues;
        std::vector<float> dAlphaInnerMDSegmentValues;
        std::vector<float> dAlphaOuterMDSegmentValues;
        std::vector<float> dAlphaInnerMDOuterMDValues;
        //one vector per layer
        std::vector<std::vector<float>> layerdzDiffLowValues;
        std::vector<std::vector<float>> layerdzDiffHighValues;
        std::vector<std::vector<float>> layerdPhiValues;
        std::vector<std::vector<float>> layerdPhiChangeValues;
        std::vector<std::vector<float>> layerdAlphaInnerMDSegmentValues;
        std::vector<std::vector<float>> layerdAlphaOuterMDSegmentValues;
        std::vector<std::vector<float>> layerdAlphaInnerMDOuterMDValues;

        std::vector<float> barreldzDiffLowValues;
        std::vector<float> barreldzDiffHighValues;
        std::vector<float> barreldPhiValues;
        std::vector<float> barreldPhiChangeValues;
        std::vector<float> barreldAlphaInnerMDSegmentValues;
        std::vector<float> barreldAlphaOuterMDSegmentValues;
        std::vector<float> barreldAlphaInnerMDOuterMDValues;
        //one vector per layer
        std::vector<std::vector<float>> layerBarreldzDiffLowValues;
        std::vector<std::vector<float>> layerBarreldzDiffHighValues;

        std::vector<std::vector<float>> layerBarrelToBarreldzDiffLowValues;
        std::vector<std::vector<float>> layerBarrelToBarreldzDiffHighValues;
        std::vector<std::vector<float>> layerBarrelToEndcapdzDiffLowValues;
        std::vector<std::vector<float>> layerBarrelToEndcapdzDiffHighValues;
        std::vector<std::vector<float>> layerBarrelToEndcapdrtDiffLowValues;
        std::vector<std::vector<float>> layerBarrelToEndcapdrtDiffHighValues;

        std::vector<std::vector<float>> layerBarrelToBarreldAlphaInnerMDSegmentValues;
        std::vector<std::vector<float>> layerBarrelToBarreldAlphaOuterMDSegmentValues;
        std::vector<std::vector<float>> layerBarrelToBarreldAlphaInnerMDOuterMDValues;

        std::vector<std::vector<float>> layerBarrelToEndcapdAlphaInnerMDSegmentValues;
        std::vector<std::vector<float>> layerBarrelToEndcapdAlphaOuterMDSegmentValues;
        std::vector<std::vector<float>> layerBarrelToEndcapdAlphaInnerMDOuterMDValues;

        std::vector<std::vector<float>> layerBarrelzLowValues;
        std::vector<std::vector<float>> layerBarrelzHighValues;
        std::vector<std::vector<float>> layerBarreldPhiValues;
        std::vector<std::vector<float>> layerBarreldPhiChangeValues;
        std::vector<std::vector<float>> layerBarreldAlphaInnerMDSegmentValues;
        std::vector<std::vector<float>> layerBarreldAlphaOuterMDSegmentValues;
        std::vector<std::vector<float>> layerBarreldAlphaInnerMDOuterMDValues;

        //Additional splitting in the barrel
        std::vector<std::vector<float>> layerBarrelCenterdPhiValues;
        std::vector<std::vector<float>> layerBarrelNormalTilteddPhiValues;
        std::vector<std::vector<float>> layerBarrelEndcapTilteddPhiValues;

        std::vector<std::vector<float>> layerBarrelCenterdPhiChangeValues;
        std::vector<std::vector<float>> layerBarrelNormalTilteddPhiChangeValues;
        std::vector<std::vector<float>> layerBarrelEndcapTilteddPhiChangeValues;

        std::vector<std::vector<float>> layerBarrelCenterdAlphaInnerMDSegmentValues;
        std::vector<std::vector<float>> layerBarrelNormalTilteddAlphaInnerMDSegmentValues;
        std::vector<std::vector<float>> layerBarrelEndcapTilteddAlphaInnerMDSegmentValues;

        std::vector<std::vector<float>> layerBarrelCenterdAlphaOuterMDSegmentValues;
        std::vector<std::vector<float>> layerBarrelNormalTilteddAlphaOuterMDSegmentValues;
        std::vector<std::vector<float>> layerBarrelEndcapTilteddAlphaOuterMDSegmentValues;

        std::vector<std::vector<float>> layerBarrelCenterdAlphaInnerMDOuterMDValues;
        std::vector<std::vector<float>> layerBarrelNormalTilteddAlphaInnerMDOuterMDValues;
        std::vector<std::vector<float>> layerBarrelEndcapTilteddAlphaInnerMDOuterMDValues;


        std::vector<float> endcapdrtDiffLowValues;
        std::vector<float> endcapdrtDiffHighValues;
        std::vector<float> endcapdPhiValues;
        std::vector<float> endcapdPhiChangeValues;
        std::vector<float> endcapdAlphaInnerMDSegmentValues;
        std::vector<float> endcapdAlphaOuterMDSegmentValues;
        std::vector<float> endcapdAlphaInnerMDOuterMDValues;
        //one vector per layer
        std::vector<std::vector<float>> layerEndcapdrtDiffLowValues;
        std::vector<std::vector<float>> layerEndcapdrtDiffHighValues;
        std::vector<std::vector<float>> layerEndcapdPhiValues;
        std::vector<std::vector<float>> layerEndcapdPhiChangeValues;
        std::vector<std::vector<float>> layerEndcapdAlphaInnerMDSegmentValues;
        std::vector<std::vector<float>> layerEndcapdAlphaOuterMDSegmentValues;
        std::vector<std::vector<float>> layerEndcapdAlphaInnerMDOuterMDValues;
        //ring
        std::vector<std::vector<float>> ringEndcapdrtDiffLowValues;
        std::vector<std::vector<float>> ringEndcapdrtDiffHighValues;
        std::vector<std::vector<float>> ringEndcapdPhiValues;
        std::vector<std::vector<float>> ringEndcapdPhiChangeValues;
        std::vector<std::vector<float>> ringEndcapdAlphaInnerMDSegmentValues;
        std::vector<std::vector<float>> ringEndcapdAlphaOuterMDSegmentValues;
        std::vector<std::vector<float>> ringEndcapdAlphaInnerMDOuterMDValues;

        std::vector<std::vector<float>> ringEndcapToEndcapdrtDiffLowValues;
        std::vector<std::vector<float>> ringEndcapToEndcapdrtDiffHighValues;
        std::vector<std::vector<float>> ringEndcapToBarreldrtDiffLowValues;
        std::vector<std::vector<float>> ringEndcapToBarreldrtDiffHighValues;

        StudySegmentCuts(const char * studyName);
        virtual void bookStudy();
        virtual void resetVariables();
        virtual void doStudy(SDL::Event &event, std::vector<std::tuple<unsigned int, SDL::Event*>> simtrkevents);

};

#endif

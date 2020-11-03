# include "StudySegmentCuts.h"

StudySegmentCuts::StudySegmentCuts(const char * studyName)
{
    studyName = studyName;
}

void StudySegmentCuts::bookStudy()
{
    ana.histograms.addVecHistogram(TString::Format("Segment_dzDiffLow"),1000,0,100,[&](){return dzDiffLowValues;});
    ana.histograms.addVecHistogram(TString::Format("Segment_dzDiffHigh"),1000,0,100,[&](){return dzDiffHighValues;});
    ana.histograms.addVecHistogram(TString::Format("Segment_dPhi"),200,-6.28,6.28,[&](){return dPhiValues;});
    ana.histograms.addVecHistogram(TString::Format("Segment_dPhiChange"),200,-6.28,6.28,[&](){return dPhiChangeValues;});
    ana.histograms.addVecHistogram(TString::Format("Segment_dAlphaInnerMDSegment"),200,-6.28,6.28,[&](){return dAlphaInnerMDSegmentValues;});
    ana.histograms.addVecHistogram(TString::Format("Segment_dAlphaOuterMDSegment"),200,-6.28,6.28,[&](){return dAlphaOuterMDSegmentValues;});
    ana.histograms.addVecHistogram(TString::Format("Segment_dAlphaInnerMDOuterMD"),200,-6.28,6.28,[&](){return dAlphaInnerMDOuterMDValues;});


    ana.histograms.addVecHistogram(TString::Format("Segment_barrel_dzDiffLow"),1000,0,100,[&](){return barreldzDiffLowValues;});
    ana.histograms.addVecHistogram(TString::Format("Segment_barrel_dzDiffHigh"),1000,0,100,[&](){return barreldzDiffHighValues;});
    ana.histograms.addVecHistogram(TString::Format("Segment_barrel_dPhi"),200,-6.28,6.28,[&](){return barreldPhiValues;});
    ana.histograms.addVecHistogram(TString::Format("Segment_barrel_dPhiChange"),200,-6.28,6.28,[&](){return barreldPhiChangeValues;});
    ana.histograms.addVecHistogram(TString::Format("Segment_barrel_dAlphaInnerMDSegment"),200,-6.28,6.28,[&](){return barreldAlphaInnerMDSegmentValues;});
    ana.histograms.addVecHistogram(TString::Format("Segment_barrel_dAlphaOuterMDSegment"),200,-6.28,6.28,[&](){return barreldAlphaOuterMDSegmentValues;});
    ana.histograms.addVecHistogram(TString::Format("Segment_barrel_dAlphaInnerMDOuterMD"),200,-6.28,6.28,[&](){return barreldAlphaInnerMDOuterMDValues;});



    ana.histograms.addVecHistogram(TString::Format("Segment_endcap_drtDiffLow"),1000,0,100,[&](){return endcapdrtDiffLowValues;});
    ana.histograms.addVecHistogram(TString::Format("Segment_endcap_drtDiffHigh"),1000,0,100,[&](){return endcapdrtDiffHighValues;});
    ana.histograms.addVecHistogram(TString::Format("Segment_endcap_dPhi"),200,-6.28,6.28,[&](){return endcapdPhiValues;});
    ana.histograms.addVecHistogram(TString::Format("Segment_endcap_dPhiChange"),200,-6.28,6.28,[&](){return endcapdPhiChangeValues;});
    ana.histograms.addVecHistogram(TString::Format("Segment_endcap_dAlphaInnerMDSegment"),200,-6.28,6.28,[&](){return endcapdAlphaInnerMDSegmentValues;});
    ana.histograms.addVecHistogram(TString::Format("Segment_endcap_dAlphaOuterMDSegment"),200,-6.28,6.28,[&](){return endcapdAlphaOuterMDSegmentValues;});
    ana.histograms.addVecHistogram(TString::Format("Segment_endcap_dAlphaInnerMDOuterMD"),200,-6.28,6.28,[&](){return endcapdAlphaInnerMDOuterMDValues;});


    //one per layer
    for(size_t i = 0; i < 6; i++)
    {
        ana.histograms.addVecHistogram(TString::Format("Segment_dzDiffLow_layer_%ld",i+1),1000,0,100,[&,i](){return layerdzDiffLowValues[i];});
        ana.histograms.addVecHistogram(TString::Format("Segment_dzDiffHigh_layer_%ld",i+1),1000,0,100,[&,i](){return layerdzDiffHighValues[i];});
        ana.histograms.addVecHistogram(TString::Format("Segment_dPhi_layer_%ld",i+1),200,-6.28,6.28,[&,i](){return layerdPhiValues[i];});
        ana.histograms.addVecHistogram(TString::Format("Segment_dPhiChange_layer_%ld",i+1),200,-6.28,6.28,[&,i](){return layerdPhiChangeValues[i];});
        ana.histograms.addVecHistogram(TString::Format("Segment_dAlphaInnerMDSegment_layer%ld",i+1),200,-6.28,6.28,[&,i](){return layerdAlphaInnerMDSegmentValues[i];});
        ana.histograms.addVecHistogram(TString::Format("Segment_dAlphaOuterMDSegment_layer%ld",i+1),200,-6.28,6.28,[&,i](){return layerdAlphaOuterMDSegmentValues[i];});
        ana.histograms.addVecHistogram(TString::Format("Segment_dAlphaInnerMDOuterMD_layer%ld",i+1),200,-6.28,6.28,[&,i](){return layerdAlphaInnerMDOuterMDValues[i];});


//       ana.histograms.addVecHistogram(TString::Format("Segment_barrel_zLow_layer_%ld",i+1),1000,0,100,[&,i](){return layerBarrelzLowValues[i];});
//       ana.histograms.addVecHistogram(TString::Format("Segment_barrel_zHigh_layer_%ld",i+1),1000,0,100,[&,i](){return layerBarrelzHighValues[i];});
       
        ana.histograms.add2DVecHistogram(TString::Format("Segment_barrel_zLow"),1000,-100,100,TString::Format("z_High_layer_%ld",i+1),1000,-100,100,[&,i](){return layerBarrelzLowValues[i];},[&,i](){return layerBarrelzHighValues[i];});

        ana.histograms.addVecHistogram(TString::Format("Segment_barrel_dzDiffLow_layer_%ld",i+1),1000,0,100,[&,i](){return layerBarreldzDiffLowValues[i];});
        ana.histograms.addVecHistogram(TString::Format("Segment_barrel_dzDiffHigh_layer_%ld",i+1),1000,0,100,[&,i](){return layerBarreldzDiffHighValues[i];});

        ana.histograms.addVecHistogram(TString::Format("Segment_barrel_to_barrel_dzDiffLow_layer_%ld",i+1),1000,0,100,[&,i](){return layerBarrelToBarreldzDiffLowValues[i];});
        ana.histograms.addVecHistogram(TString::Format("Segment_barrel_to_barrel_dzDiffHigh_layer_%ld",i+1),1000,0,100,[&,i](){return layerBarrelToBarreldzDiffHighValues[i];});

        ana.histograms.addVecHistogram(TString::Format("Segment_barrel_to_endcap_dzDiffLow_layer_%ld",i+1),1000,0,100,[&,i](){return layerBarrelToEndcapdzDiffLowValues[i];});
        ana.histograms.addVecHistogram(TString::Format("Segment_barrel_to_endcap_dzDiffHigh_layer_%ld",i+1),1000,0,100,[&,i](){return layerBarrelToEndcapdzDiffHighValues[i];});

        ana.histograms.addVecHistogram(TString::Format("Segment_barrel_to_endcap_drtDiffLow_layer_%ld",i+1),1000,0,100,[&,i](){return layerBarrelToEndcapdrtDiffLowValues[i];});
        ana.histograms.addVecHistogram(TString::Format("Segment_barrel_to_endcap_drtDiffHigh_layer_%ld",i+1),1000,0,100,[&,i](){return layerBarrelToEndcapdrtDiffHighValues[i];});

        
        ana.histograms.addVecHistogram(TString::Format("Segment_barrel_dPhi_layer_%ld",i+1),200,-6.28,6.28,[&,i](){return layerBarreldPhiValues[i];});
        ana.histograms.addVecHistogram(TString::Format("Segment_barrel_dPhiChange_layer_%ld",i+1),200,-6.28,6.28,[&,i](){return layerBarreldPhiChangeValues[i];});
        ana.histograms.addVecHistogram(TString::Format("Segment_barrel_dAlphaInnerMDSegment_layer%ld",i+1),200,-6.28,6.28,[&,i](){return layerBarreldAlphaInnerMDSegmentValues[i];});
        ana.histograms.addVecHistogram(TString::Format("Segment_barrel_dAlphaOuterMDSegment_layer%ld",i+1),200,-6.28,6.28,[&,i](){return layerBarreldAlphaOuterMDSegmentValues[i];});
        ana.histograms.addVecHistogram(TString::Format("Segment_barrel_dAlphaInnerMDOuterMD_layer%ld",i+1),200,-6.28,6.28,[&,i](){return layerBarreldAlphaInnerMDOuterMDValues[i];});

        ana.histograms.addVecHistogram(TString::Format("Segment_barrel_to_barrel_dAlphaInnerMDSegment_layer%ld",i+1),200,-6.28,6.28,[&,i](){return layerBarrelToBarreldAlphaInnerMDSegmentValues[i];});
        ana.histograms.addVecHistogram(TString::Format("Segment_barrel_to_barrel_dAlphaOuterMDSegment_layer%ld",i+1),200,-6.28,6.28,[&,i](){return layerBarrelToBarreldAlphaOuterMDSegmentValues[i];});
        ana.histograms.addVecHistogram(TString::Format("Segment_barrel_to_barrel_dAlphaInnerMDOuterMD_layer%ld",i+1),200,-6.28,6.28,[&,i](){return layerBarrelToBarreldAlphaInnerMDOuterMDValues[i];});

        ana.histograms.addVecHistogram(TString::Format("Segment_barrel_to_endcap_dAlphaInnerMDSegment_layer%ld",i+1),200,-6.28,6.28,[&,i](){return layerBarrelToEndcapdAlphaInnerMDSegmentValues[i];});
        ana.histograms.addVecHistogram(TString::Format("Segment_barrel_to_endcap_dAlphaOuterMDSegment_layer%ld",i+1),200,-6.28,6.28,[&,i](){return layerBarrelToEndcapdAlphaOuterMDSegmentValues[i];});
        ana.histograms.addVecHistogram(TString::Format("Segment_barrel_to_endcap_dAlphaInnerMDOuterMD_layer%ld",i+1),200,-6.28,6.28,[&,i](){return layerBarrelToEndcapdAlphaInnerMDOuterMDValues[i];});

        if(i < 5)
        {
            ana.histograms.addVecHistogram(TString::Format("Segment_endcap_drtDiffLow_layer_%ld",i+1),1000,0,100,[&,i](){return layerEndcapdrtDiffLowValues[i];});
            ana.histograms.addVecHistogram(TString::Format("Segment_endcap_drtDiffHigh_layer_%ld",i+1),1000,0,100,[&,i](){return layerEndcapdrtDiffHighValues[i];});
            ana.histograms.addVecHistogram(TString::Format("Segment_endcap_dPhi_layer_%ld",i+1),200,-6.28,6.28,[&,i](){return layerEndcapdPhiValues[i];});
            ana.histograms.addVecHistogram(TString::Format("Segment_endcap_dPhiChange_layer_%ld",i+1),200,-6.28,6.28,[&,i](){return layerEndcapdPhiChangeValues[i];});
            ana.histograms.addVecHistogram(TString::Format("Segment_endcap_dAlphaInnerMDSegment_layer%ld",i+1),200,-6.28,6.28,[&,i](){return layerEndcapdAlphaInnerMDSegmentValues[i];});
            ana.histograms.addVecHistogram(TString::Format("Segment_endcap_dAlphaOuterMDSegment_layer%ld",i+1),200,-6.28,6.28,[&,i](){return layerEndcapdAlphaOuterMDSegmentValues[i];});
            ana.histograms.addVecHistogram(TString::Format("Segment_endcap_dAlphaInnerMDOuterMD_layer%ld",i+1),200,-6.28,6.28,[&,i](){return layerEndcapdAlphaInnerMDOuterMDValues[i];});
        }

        ana.histograms.addVecHistogram(TString::Format("Segment_barrel_center_dPhi_layer_%ld",i+1),200,-6.28,6.28,[&,i](){return layerBarrelCenterdPhiValues[i];});
        ana.histograms.addVecHistogram(TString::Format("Segment_barrel_center_dPhiChange_layer_%ld",i+1),200,-6.28,6.28,[&,i](){return layerBarrelCenterdPhiChangeValues[i];});
        ana.histograms.addVecHistogram(TString::Format("_Segment_barrel_center_dAlphaInnerMDSegment_layer_%ld",i+1),200,-6.28,6.28,[&,i](){return layerBarrelCenterdAlphaInnerMDSegmentValues[i];});
        ana.histograms.addVecHistogram(TString::Format("Segment_barrel_center_dAlphaOuterMDSegment_layer_%ld",i+1),200,-6.28,6.28,[&,i](){return layerBarrelCenterdAlphaOuterMDSegmentValues[i];});
        ana.histograms.addVecHistogram(TString::Format("Segment_barrel_center_dAlphaInnerMDOuterMD_layer_%ld",i+1),200,-6.28,6.28,[&,i](){return layerBarrelCenterdAlphaInnerMDOuterMDValues[i];});

        if(i < 3)
        {
            ana.histograms.addVecHistogram(TString::Format("Segment_barrel_normal_tilted_dPhi_layer_%ld",i+1),200,-6.28,6.28,[&,i](){return layerBarrelNormalTilteddPhiValues[i];});
            ana.histograms.addVecHistogram(TString::Format("Segment_barrel_normal_tilted_dPhiChange_layer_%ld",i+1),200,-6.28,6.28,[&,i](){return layerBarrelNormalTilteddPhiChangeValues[i];});
            ana.histograms.addVecHistogram(TString::Format("Segment_barrel_normal_tilted_dAlphaInnerMDSegment_layer_%ld",i+1),200,-6.28,6.28,[&,i](){return layerBarrelNormalTilteddAlphaInnerMDSegmentValues[i];});
            ana.histograms.addVecHistogram(TString::Format("Segment_barrel_normal_tilted_dAlphaOuterMDSegment_layer_%ld",i+1),200,-6.28,6.28,[&,i](){return layerBarrelNormalTilteddAlphaOuterMDSegmentValues[i];});
            ana.histograms.addVecHistogram(TString::Format("Segment_barrel_normal_tilted_dAlphaInnerMDOuterMD_layer_%ld",i+1),200,-6.28,6.28,[&,i](){return layerBarrelNormalTilteddAlphaInnerMDOuterMDValues[i];});

            ana.histograms.addVecHistogram(TString::Format("Segment_barrel_endcap_tilted_dPhi_layer_%ld",i+1),200,-6.28,6.28,[&,i](){return layerBarrelEndcapTilteddPhiValues[i];});
            ana.histograms.addVecHistogram(TString::Format("Segment_barrel_endcap_tilted_dPhiChange_layer_%ld",i+1),200,-6.28,6.28,[&,i](){return layerBarrelEndcapTilteddPhiChangeValues[i];});
        ana.histograms.addVecHistogram(TString::Format("Segment_barrel_endcap_tilted_dAlphaInnerMDSegment_layer_%ld",i+1),200,-6.28,6.28,[&,i](){return layerBarrelEndcapTilteddAlphaInnerMDSegmentValues[i];});
            ana.histograms.addVecHistogram(TString::Format("Segment_barrel_endcap_tilted_dAlphaOuterMDSegment_layer_%ld",i+1),200,-6.28,6.28,[&,i](){return layerBarrelEndcapTilteddAlphaOuterMDSegmentValues[i];});
            ana.histograms.addVecHistogram(TString::Format("Segment_barrel_endcap_tilted_dAlphaInnerMDOuterMD_layer_%ld",i+1),200,-6.28,6.28,[&,i](){return layerBarrelEndcapTilteddAlphaInnerMDOuterMDValues[i];});
        }
    }


    for(size_t i = 0; i < 15; i++)
    {
        ana.histograms.addVecHistogram(TString::Format("Segment_endcap_drtDiffLow_ring_%ld",i+1),1000,0,100,[&,i](){return ringEndcapdrtDiffLowValues[i];});
        ana.histograms.addVecHistogram(TString::Format("Segment_endcap_drtDiffHigh_ring_%ld",i+1),1000,0,100,[&,i](){return ringEndcapdrtDiffHighValues[i];});
        ana.histograms.addVecHistogram(TString::Format("Segment_endcap_dPhi_ring%ld",i+1),200,-6.28,6.28,[&,i](){return ringEndcapdPhiValues[i];});
        ana.histograms.addVecHistogram(TString::Format("Segment_endcap_dPhiChange_ring%ld",i+1),200,-6.28,6.28,[&,i](){return ringEndcapdPhiChangeValues[i];});
        ana.histograms.addVecHistogram(TString::Format("Segment_endcap_dAlphaInnerMDSegment_ring%ld",i+1),200,-6.28,6.28,[&,i](){return ringEndcapdAlphaInnerMDSegmentValues[i];});
        ana.histograms.addVecHistogram(TString::Format("Segment_endcap_dAlphaOuterMDSegment_ring%ld",i+1),200,-6.28,6.28,[&,i](){return ringEndcapdAlphaOuterMDSegmentValues[i];});
        ana.histograms.addVecHistogram(TString::Format("Segment_endcap_dAlphaInnerMDOuterMD_ring%ld",i+1),200,-6.28,6.28,[&,i](){return ringEndcapdAlphaInnerMDOuterMDValues[i];});

        ana.histograms.addVecHistogram(TString::Format("Segment_endcap_to_endcap_drtDiffLow_ring_%ld",i+1),1000,0,100,[&,i](){return ringEndcapToEndcapdrtDiffLowValues[i];});
        ana.histograms.addVecHistogram(TString::Format("Segment_endcap_to_endcap_drtDiffHigh_ring_%ld",i+1),1000,0,100,[&,i](){return ringEndcapToEndcapdrtDiffHighValues[i];});

        ana.histograms.addVecHistogram(TString::Format("Segment_endcap_to_barrel_drtDiffLow_ring_%ld",i+1),1000,0,100,[&,i](){return ringEndcapToBarreldrtDiffLowValues[i];});
        ana.histograms.addVecHistogram(TString::Format("Segment_endcap_to_barrel_drtDiffHigh_ring_%ld",i+1),1000,0,100,[&,i](){return ringEndcapToBarreldrtDiffHighValues[i];});

    }
}

void StudySegmentCuts::resetVariables()
{
    dzDiffLowValues.clear();
    dzDiffHighValues.clear();
    dPhiValues.clear();
    dPhiChangeValues.clear();
    dAlphaInnerMDSegmentValues.clear();
    dAlphaOuterMDSegmentValues.clear();
    dAlphaInnerMDOuterMDValues.clear();

    barreldzDiffLowValues.clear();
    barreldzDiffHighValues.clear();
    barreldPhiValues.clear();
    barreldPhiChangeValues.clear();
    barreldAlphaInnerMDSegmentValues.clear();
    barreldAlphaOuterMDSegmentValues.clear();
    barreldAlphaInnerMDOuterMDValues.clear();

    endcapdrtDiffLowValues.clear();
    endcapdrtDiffHighValues.clear();
    endcapdPhiValues.clear();
    endcapdPhiChangeValues.clear();
    endcapdAlphaInnerMDSegmentValues.clear();
    endcapdAlphaOuterMDSegmentValues.clear();
    endcapdAlphaInnerMDOuterMDValues.clear();

    layerdzDiffLowValues.clear();
    layerdzDiffHighValues.clear();
    layerdPhiValues.clear();
    layerdPhiChangeValues.clear();
    layerdAlphaInnerMDSegmentValues.clear();
    layerdAlphaOuterMDSegmentValues.clear();
    layerdAlphaInnerMDOuterMDValues.clear();

    layerBarrelzLowValues.clear();
    layerBarrelzHighValues.clear();
    layerBarreldzDiffLowValues.clear();
    layerBarreldzDiffHighValues.clear();
   
    layerBarrelToBarreldzDiffLowValues.clear();
    layerBarrelToBarreldzDiffHighValues.clear();
    layerBarrelToEndcapdzDiffLowValues.clear();
    layerBarrelToEndcapdzDiffHighValues.clear();
    
    layerBarrelToEndcapdrtDiffLowValues.clear();
    layerBarrelToEndcapdrtDiffHighValues.clear();

    ringEndcapToEndcapdrtDiffLowValues.clear();
    ringEndcapToEndcapdrtDiffHighValues.clear();
    ringEndcapToBarreldrtDiffLowValues.clear();
    ringEndcapToBarreldrtDiffHighValues.clear();

    layerBarreldPhiValues.clear();
    layerBarreldPhiChangeValues.clear();
    layerBarreldAlphaInnerMDSegmentValues.clear();
    layerBarreldAlphaOuterMDSegmentValues.clear();
    layerBarreldAlphaInnerMDOuterMDValues.clear();

    layerBarrelToBarreldAlphaInnerMDSegmentValues.clear();
    layerBarrelToBarreldAlphaOuterMDSegmentValues.clear();
    layerBarrelToBarreldAlphaInnerMDOuterMDValues.clear();

    layerBarrelToEndcapdAlphaInnerMDSegmentValues.clear();
    layerBarrelToEndcapdAlphaOuterMDSegmentValues.clear();
    layerBarrelToEndcapdAlphaInnerMDOuterMDValues.clear();

    layerBarrelCenterdPhiValues.clear();
    layerBarrelNormalTilteddPhiValues.clear();
    layerBarrelEndcapTilteddPhiValues.clear();

    layerBarrelCenterdPhiChangeValues.clear();
    layerBarrelNormalTilteddPhiChangeValues.clear();
    layerBarrelEndcapTilteddPhiChangeValues.clear();

    layerBarrelCenterdAlphaInnerMDSegmentValues.clear();
    layerBarrelNormalTilteddAlphaInnerMDSegmentValues.clear();
    layerBarrelEndcapTilteddAlphaInnerMDSegmentValues.clear();

    layerBarrelCenterdAlphaOuterMDSegmentValues.clear();
    layerBarrelNormalTilteddAlphaOuterMDSegmentValues.clear();
    layerBarrelEndcapTilteddAlphaOuterMDSegmentValues.clear();

    layerBarrelCenterdAlphaInnerMDOuterMDValues.clear();
    layerBarrelNormalTilteddAlphaInnerMDOuterMDValues.clear();
    layerBarrelEndcapTilteddAlphaInnerMDOuterMDValues.clear();


    layerEndcapdrtDiffLowValues.clear();
    layerEndcapdrtDiffHighValues.clear();
    layerEndcapdPhiValues.clear();
    layerEndcapdPhiChangeValues.clear();
    layerEndcapdAlphaInnerMDSegmentValues.clear();
    layerEndcapdAlphaOuterMDSegmentValues.clear();
    layerEndcapdAlphaInnerMDOuterMDValues.clear();

    ringEndcapdrtDiffLowValues.clear();
    ringEndcapdrtDiffHighValues.clear();
    ringEndcapdPhiValues.clear();
    ringEndcapdPhiChangeValues.clear();
    ringEndcapdAlphaInnerMDSegmentValues.clear();
    ringEndcapdAlphaOuterMDSegmentValues.clear();
    ringEndcapdAlphaInnerMDOuterMDValues.clear();


    for(size_t i = 1; i <= 6; i++)
    {
        layerdzDiffLowValues.push_back(std::vector<float>());
        layerdzDiffHighValues.push_back(std::vector<float>());
        layerdPhiValues.push_back(std::vector<float>());
        layerdPhiChangeValues.push_back(std::vector<float>());
        layerdAlphaInnerMDSegmentValues.push_back(std::vector<float>());
        layerdAlphaOuterMDSegmentValues.push_back(std::vector<float>());
        layerdAlphaInnerMDOuterMDValues.push_back(std::vector<float>());

        layerBarrelzLowValues.push_back(std::vector<float>());
        layerBarrelzHighValues.push_back(std::vector<float>());

        layerBarreldzDiffLowValues.push_back(std::vector<float>());
        layerBarreldzDiffHighValues.push_back(std::vector<float>());

        layerBarrelToBarreldzDiffLowValues.push_back(std::vector<float>());
        layerBarrelToBarreldzDiffHighValues.push_back(std::vector<float>());
        layerBarrelToEndcapdzDiffLowValues.push_back(std::vector<float>());
        layerBarrelToEndcapdzDiffHighValues.push_back(std::vector<float>());

        layerBarrelToEndcapdrtDiffLowValues.push_back(std::vector<float>());
        layerBarrelToEndcapdrtDiffHighValues.push_back(std::vector<float>());

        layerBarrelToBarreldAlphaInnerMDSegmentValues.push_back(std::vector<float>());
        layerBarrelToBarreldAlphaOuterMDSegmentValues.push_back(std::vector<float>());
        layerBarrelToBarreldAlphaInnerMDOuterMDValues.push_back(std::vector<float>());

        layerBarrelToEndcapdAlphaInnerMDSegmentValues.push_back(std::vector<float>());
        layerBarrelToEndcapdAlphaOuterMDSegmentValues.push_back(std::vector<float>());
        layerBarrelToEndcapdAlphaInnerMDOuterMDValues.push_back(std::vector<float>());

 
        layerBarreldPhiValues.push_back(std::vector<float>());
        layerBarreldPhiChangeValues.push_back(std::vector<float>());
        layerBarreldAlphaInnerMDSegmentValues.push_back(std::vector<float>());
        layerBarreldAlphaOuterMDSegmentValues.push_back(std::vector<float>());
        layerBarreldAlphaInnerMDOuterMDValues.push_back(std::vector<float>());

        if(i < 6)
        {
            layerEndcapdrtDiffLowValues.push_back(std::vector<float>());
            layerEndcapdrtDiffHighValues.push_back(std::vector<float>());
            layerEndcapdPhiValues.push_back(std::vector<float>());
            layerEndcapdPhiChangeValues.push_back(std::vector<float>());
            layerEndcapdAlphaInnerMDSegmentValues.push_back(std::vector<float>());
            layerEndcapdAlphaOuterMDSegmentValues.push_back(std::vector<float>());
            layerEndcapdAlphaInnerMDOuterMDValues.push_back(std::vector<float>());
        }

        layerBarrelCenterdPhiValues.push_back(std::vector<float>());
        layerBarrelCenterdPhiChangeValues.push_back(std::vector<float>());
        layerBarrelCenterdAlphaInnerMDSegmentValues.push_back(std::vector<float>());
        layerBarrelCenterdAlphaOuterMDSegmentValues.push_back(std::vector<float>());
        layerBarrelCenterdAlphaInnerMDOuterMDValues.push_back(std::vector<float>());

        if(i <= 3)
        {
            layerBarrelNormalTilteddPhiValues.push_back(std::vector<float>());
            layerBarrelNormalTilteddPhiChangeValues.push_back(std::vector<float>());
            layerBarrelNormalTilteddAlphaInnerMDSegmentValues.push_back(std::vector<float>());
            layerBarrelNormalTilteddAlphaOuterMDSegmentValues.push_back(std::vector<float>());
            layerBarrelNormalTilteddAlphaInnerMDOuterMDValues.push_back(std::vector<float>());

            layerBarrelEndcapTilteddPhiValues.push_back(std::vector<float>());
            layerBarrelEndcapTilteddPhiChangeValues.push_back(std::vector<float>());
            layerBarrelEndcapTilteddAlphaInnerMDSegmentValues.push_back(std::vector<float>());
            layerBarrelEndcapTilteddAlphaOuterMDSegmentValues.push_back(std::vector<float>());
            layerBarrelEndcapTilteddAlphaInnerMDOuterMDValues.push_back(std::vector<float>());
        }

      
    }

    for(size_t i = 0; i < 15; i++)
    {
        ringEndcapdrtDiffLowValues.push_back(std::vector<float>());
        ringEndcapdrtDiffHighValues.push_back(std::vector<float>());
        ringEndcapdPhiValues.push_back(std::vector<float>());
        ringEndcapdPhiChangeValues.push_back(std::vector<float>());
        ringEndcapdAlphaInnerMDSegmentValues.push_back(std::vector<float>());
        ringEndcapdAlphaOuterMDSegmentValues.push_back(std::vector<float>());
        ringEndcapdAlphaInnerMDOuterMDValues.push_back(std::vector<float>());

        ringEndcapToEndcapdrtDiffLowValues.push_back(std::vector<float>());
        ringEndcapToEndcapdrtDiffHighValues.push_back(std::vector<float>());
        ringEndcapToBarreldrtDiffHighValues.push_back(std::vector<float>());
        ringEndcapToBarreldrtDiffLowValues.push_back(std::vector<float>());
    }

}


void StudySegmentCuts::doStudy(SDL::Event &event, std::vector<std::tuple<unsigned int, SDL::Event*>> simtrkevents)
{
    resetVariables();

    std::vector<SDL::Module*> moduleList = event.getLowerModulePtrs();
    for(auto &module:moduleList)
    {
        std::vector<SDL::Segment*> segments = module->getSegmentPtrs();
        for(auto &sg:segments)
        {
            dPhiValues.push_back((sg->getRecoVars()).at("deltaPhi"));
            dPhiChangeValues.push_back(sg->getDeltaPhiChange());

            layerdPhiValues.at(module->layer()-1).push_back((sg->getRecoVars()).at("deltaPhi"));
            dzDiffLowValues.push_back(sg->getZOut() - sg->getZLo());
            dzDiffHighValues.push_back(sg->getZHi() - sg->getZOut());

            layerdzDiffLowValues.at(module->layer()-1).push_back(sg->getZOut() - sg->getZLo());
            layerdzDiffHighValues.at(module->layer()-1).push_back(sg->getZHi() - sg->getZOut());
            layerdPhiChangeValues.at(module->layer()-1).push_back(sg->getDeltaPhiChange());

            float dPhiChangeInnerMD = (sg->innerMiniDoubletPtr())->getDeltaPhiChange();
            float dPhiChangeOuterMD = (sg->outerMiniDoubletPtr())->getDeltaPhiChange();
            float dPhiChangeSegment = sg->getDeltaPhiChange();

            float dAlphaInnerMDSegment = dPhiChangeInnerMD - dPhiChangeSegment;
            float dAlphaOuterMDSegment = dPhiChangeOuterMD - dPhiChangeSegment;
            float dAlphaInnerMDOuterMD = dPhiChangeInnerMD - dPhiChangeOuterMD;

            dAlphaInnerMDSegmentValues.push_back(dAlphaInnerMDSegment);
            dAlphaOuterMDSegmentValues.push_back(dAlphaOuterMDSegment);
            dAlphaInnerMDOuterMDValues.push_back(dAlphaInnerMDOuterMD);
            layerdAlphaInnerMDSegmentValues.at(module->layer()-1).push_back(dAlphaInnerMDSegment);
            layerdAlphaOuterMDSegmentValues.at(module->layer()-1).push_back(dAlphaOuterMDSegment);
            layerdAlphaInnerMDOuterMDValues.at(module->layer()-1).push_back(dAlphaInnerMDOuterMD);

            if(module->subdet() == SDL::Module::Barrel)
            {
                barreldPhiValues.push_back((sg->getRecoVars()).at("deltaPhi"));
                barreldzDiffLowValues.push_back(sg->getZOut() - sg->getZLo());
                barreldzDiffHighValues.push_back(sg->getZHi() - sg->getZOut());
                barreldPhiChangeValues.push_back(sg->getDeltaPhiChange());
                barreldAlphaInnerMDSegmentValues.push_back(dAlphaInnerMDSegment);
                barreldAlphaOuterMDSegmentValues.push_back(dAlphaOuterMDSegment);
                barreldAlphaInnerMDOuterMDValues.push_back(dAlphaInnerMDOuterMD);

		layerBarrelzLowValues.at(module->layer()-1).push_back(sg->getZLo() - sg->getZIn());
                layerBarrelzHighValues.at(module->layer()-1).push_back(sg->getZHi() - sg->getZIn());

                layerBarreldPhiValues.at(module->layer()-1).push_back((sg->getRecoVars()).at("deltaPhi"));
                layerBarreldzDiffLowValues.at(module->layer()-1).push_back(sg->getZOut() - sg->getZLo());
                layerBarreldzDiffHighValues.at(module->layer()-1).push_back(sg->getZHi() - sg->getZOut());
                layerBarreldPhiChangeValues.at(module->layer()-1).push_back(sg->getDeltaPhiChange());
                layerBarreldAlphaInnerMDSegmentValues.at(module->layer()-1).push_back(dAlphaInnerMDSegment);
                layerBarreldAlphaOuterMDSegmentValues.at(module->layer()-1).push_back(dAlphaOuterMDSegment);
                layerBarreldAlphaInnerMDOuterMDValues.at(module->layer()-1).push_back(dAlphaInnerMDOuterMD);

                if((((sg->outerMiniDoubletPtr())->lowerHitPtr())->getModule()).subdet() == SDL::Module::Barrel)
                {
                    layerBarrelToBarreldzDiffLowValues.at(module->layer()-1).push_back(sg->getZOut() - sg->getZLo());
                       
                    layerBarrelToBarreldzDiffHighValues.at(module->layer()-1).push_back(sg->getZHi() - sg->getZOut());

                    layerBarrelToBarreldAlphaInnerMDSegmentValues.at(module->layer()-1).push_back(dAlphaInnerMDSegment);
                    layerBarrelToBarreldAlphaOuterMDSegmentValues.at(module->layer()-1).push_back(dAlphaOuterMDSegment);
                    layerBarrelToBarreldAlphaInnerMDOuterMDValues.at(module->layer()-1).push_back(dAlphaInnerMDOuterMD);
  
                }

                else if((((sg->outerMiniDoubletPtr())->lowerHitPtr())->getModule()).subdet() == SDL::Module::Endcap)
                {
                    layerBarrelToEndcapdzDiffLowValues.at(module->layer()-1).push_back(sg->getZOut() - sg->getZLo()); 
                    layerBarrelToEndcapdzDiffHighValues.at(module->layer()-1).push_back(sg->getZHi() - sg->getZOut());

                    layerBarrelToEndcapdrtDiffLowValues.at(module->layer()-1).push_back(sg->getRtOut() - sg->getRtLo());
                    layerBarrelToEndcapdrtDiffHighValues.at(module->layer()-1).push_back(sg->getRtHi() - sg->getRtOut());

                    layerBarrelToEndcapdAlphaInnerMDSegmentValues.at(module->layer()-1).push_back(dAlphaInnerMDSegment);
                    layerBarrelToEndcapdAlphaOuterMDSegmentValues.at(module->layer()-1).push_back(dAlphaOuterMDSegment);
                    layerBarrelToEndcapdAlphaInnerMDOuterMDValues.at(module->layer()-1).push_back(dAlphaInnerMDOuterMD);

                }

                if(module->side() == SDL::Module::Center)
                {
                    layerBarrelCenterdPhiValues.at(module->layer()-1).push_back(sg->getRecoVars().at("deltaPhi"));
                    layerBarrelCenterdPhiChangeValues.at(module->layer()-1).push_back(sg->getDeltaPhiChange());
                    layerBarrelCenterdAlphaInnerMDSegmentValues.at(module->layer()-1).push_back(dAlphaInnerMDSegment);
                    layerBarrelCenterdAlphaOuterMDSegmentValues.at(module->layer()-1).push_back(dAlphaOuterMDSegment);
                    layerBarrelCenterdAlphaInnerMDOuterMDValues.at(module->layer()-1).push_back(dAlphaInnerMDOuterMD);
                }
                else
                {
                    //redirect this to innerminidoubletptr somehow
                    if(SDL::MiniDoublet::isNormalTiltedModules(*module))
                    {
                        layerBarrelNormalTilteddPhiValues.at(module->layer()-1).push_back(sg->getRecoVars().at("deltaPhi"));
                        layerBarrelNormalTilteddPhiChangeValues.at(module->layer()-1).push_back(sg->getDeltaPhiChange());
                        layerBarrelNormalTilteddAlphaInnerMDSegmentValues.at(module->layer()-1).push_back(dAlphaInnerMDSegment);
                        layerBarrelNormalTilteddAlphaOuterMDSegmentValues.at(module->layer()-1).push_back(dAlphaOuterMDSegment);
                        layerBarrelNormalTilteddAlphaInnerMDOuterMDValues.at(module->layer()-1).push_back(dAlphaInnerMDOuterMD);
                    }
                    else
                    {
                        layerBarrelEndcapTilteddPhiValues.at(module->layer()-1).push_back(sg->getRecoVars().at("deltaPhi"));
                        layerBarrelEndcapTilteddPhiChangeValues.at(module->layer()-1).push_back(sg->getDeltaPhiChange());
                        layerBarrelEndcapTilteddAlphaInnerMDSegmentValues.at(module->layer()-1).push_back(dAlphaInnerMDSegment);
                        layerBarrelEndcapTilteddAlphaOuterMDSegmentValues.at(module->layer()-1).push_back(dAlphaOuterMDSegment);
                        layerBarrelEndcapTilteddAlphaInnerMDOuterMDValues.at(module->layer()-1).push_back(dAlphaInnerMDOuterMD);
                    }
                }

            }
            else if(module->subdet() == SDL::Module::Endcap)
            {
                endcapdPhiValues.push_back((sg->getRecoVars()).at("deltaPhi"));
                endcapdrtDiffLowValues.push_back(sg->getRtOut() - sg->getRtLo());
                endcapdrtDiffHighValues.push_back(sg->getRtHi() - sg->getRtOut());
                endcapdPhiChangeValues.push_back(sg->getDeltaPhiChange());
                endcapdAlphaInnerMDSegmentValues.push_back(dAlphaInnerMDSegment);
                endcapdAlphaOuterMDSegmentValues.push_back(dAlphaOuterMDSegment);
                endcapdAlphaInnerMDOuterMDValues.push_back(dAlphaInnerMDOuterMD);

                layerEndcapdPhiValues.at(module->layer()-1).push_back((sg->getRecoVars()).at("deltaPhi"));
                layerEndcapdrtDiffLowValues.at(module->layer()-1).push_back(sg->getRtOut() - sg->getRtLo());
                layerEndcapdrtDiffHighValues.at(module->layer()-1).push_back(sg->getRtHi() - sg->getRtOut());
                layerEndcapdPhiChangeValues.at(module->layer()-1).push_back(sg->getDeltaPhiChange());
                layerEndcapdAlphaInnerMDSegmentValues.at(module->layer()-1).push_back(dAlphaInnerMDSegment);
                layerEndcapdAlphaOuterMDSegmentValues.at(module->layer()-1).push_back(dAlphaOuterMDSegment);
                layerEndcapdAlphaInnerMDOuterMDValues.at(module->layer()-1).push_back(dAlphaInnerMDOuterMD);

                ringEndcapdPhiValues.at(module->ring()-1).push_back((sg->getRecoVars()).at("deltaPhi"));
                ringEndcapdrtDiffLowValues.at(module->ring()-1).push_back(sg->getRtOut() - sg->getRtLo());
                ringEndcapdrtDiffHighValues.at(module->ring()-1).push_back(sg->getRtHi() - sg->getRtOut());
                ringEndcapdPhiChangeValues.at(module->ring()-1).push_back(sg->getDeltaPhiChange());
                ringEndcapdAlphaInnerMDSegmentValues.at(module->ring()-1).push_back(dAlphaInnerMDSegment);
                ringEndcapdAlphaOuterMDSegmentValues.at(module->ring()-1).push_back(dAlphaOuterMDSegment);
                ringEndcapdAlphaInnerMDOuterMDValues.at(module->ring()-1).push_back(dAlphaInnerMDOuterMD);

                if((((sg->outerMiniDoubletPtr())->lowerHitPtr())->getModule()).subdet() == SDL::Module::Barrel)
                {
                    ringEndcapToBarreldrtDiffLowValues.at(module->ring()-1).push_back(sg->getRtOut() - sg->getRtLo());
                       
                    ringEndcapToBarreldrtDiffHighValues.at(module->ring()-1).push_back(sg->getRtHi() - sg->getRtOut());
                }

                else if((((sg->outerMiniDoubletPtr())->lowerHitPtr())->getModule()).subdet() == SDL::Module::Endcap)
                {
                    ringEndcapToEndcapdrtDiffLowValues.at(module->ring()-1).push_back(sg->getRtOut() - sg->getRtLo()); 
                    ringEndcapToEndcapdrtDiffHighValues.at(module->ring()-1).push_back(sg->getRtHi() - sg->getRtOut());
                }
            }
        }
    }
}

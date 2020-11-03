#!/usr/bin/env python

#@description:
# test the functionality of the TQHistogramObservable class

from QFramework import TQXSecParser, TQSampleFolder
#from ROOT import 
import sys, os

from QFramework import TQPythonTest

class TQXSecParserTest(TQPythonTest):
  
  def setUp(self):
    super(TQXSecParserTest, self).setUp()
    #create a small XS file. The XS of the second entry is deliberately damaged (whitespace before the 'E-4') to test the more robust number conversion (see jira HWWATLAS-160)
    test = "SampleID , xsection , kfactor , filtereff , uncertainty , mh , generator , process , simulation, path\n\
    #some comment\n\
    341079, 0.11020e-2, 1.0, 4.9150E-01, --, 125,  Powheg+Pythia8+EvtGen, PowhegPythia8EvtGen_CT10_AZNLOCTEQ6L1_ggH125_WWlvlv_EF_15_5,  OFLCOND-RUN12-SDR-30, /sig/$(channel)/ggf\n\
    \n\
    341080, 0.8579 E-4, 1.0, 5.1025E-01, --, 125,  Powheg+Pythia8+EvtGen, PowhegPythia8EvtGen_CT10_AZNLOCTEQ6L1_VBFH125_WWlvlv_EF_15_5, OFLCOND-RUN12-SDR-30, /sig/$(channel)/vbf\n"
    with open(os.path.join(self.tempdir, "testXS.csv"), "w") as f:
      f.write(test)
    
  def tearDown(self):
    # delete temporary directory
    super(TQXSecParserTest, self).tearDown()
  
  def runTest(self):
    self.test_number_parsing()

  def test_number_parsing_wo_space(self):
    samples = TQSampleFolder("samples")
    parser = TQXSecParser(samples);
    parser.readCSVfile(os.path.join(self.tempdir, "testXS.csv"))
    parser.readMappingFromColumn("*path*")
    unit = TQXSecParser.unitName(TQXSecParser.unit("pb"))
    parser.setTagString("xSectionUnit",unit)
    parser.addPathVariant("channel","em")
    parser.addAllSamples(False)

    ggf = samples.getSampleFolder("sig/em/ggf/341079")
    self.assertTrue(ggf, msg="Failed to get ggf sample folder")

    xsGGF = ggf.getTagDoubleDefault(".xsp.xSection",-999.)
    self.assertAlmostEqual(xsGGF, 0.11020e-2, delta=1e-10)

  def test_number_parsing_w_space(self):
    samples = TQSampleFolder("samples")
    parser = TQXSecParser(samples);
    parser.readCSVfile(os.path.join(self.tempdir, "testXS.csv"))
    parser.readMappingFromColumn("*path*")
    unit = TQXSecParser.unitName(TQXSecParser.unit("pb"))
    parser.setTagString("xSectionUnit",unit)
    parser.addPathVariant("channel","em")
    parser.addAllSamples(False)

    vbf = samples.getSampleFolder("sig/em/vbf/341080")
    self.assertTrue(vbf, msg="Failed to get vbf sample folder")

    xsVBF = vbf.getTagDoubleDefault(".xsp.xSection",-999.)
    self.assertAlmostEqual(xsVBF, 0.8579E-4, delta=1e-10)

  def suite():
    tests = ['test_number_parsing_w_space_ggf', 'test_number_parsing_w_space_vbf']
    return unittest.TestSuite(map(TQXSecParserTest, tests))
  
  

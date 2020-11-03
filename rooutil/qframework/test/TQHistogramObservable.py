#!/usr/bin/env python

#@description:
# test the functionality of the TQHistogramObservable class

from QFramework import TQFolder, TQSampleFolder, TQSample, TQCut, TQAnalysisSampleVisitor, TQHistoMakerAnalysisJob
from ROOT import TFile, TTree, TH1F
import sys, os
from array import array
import random

from QFramework import TQPythonTest

class TQHistogramObservableTest(TQPythonTest):
  
  def setUp(self):
    super(TQHistogramObservableTest, self).setUp()
    
    f = TFile( os.path.join(self.tempdir, 'test.root'), 'recreate' )
    t = TTree( 'testTree', 'testTree' )
     
    d = array( 'f', [ 0. ] )
    t.Branch( 'testVal', d, 'testVal/F' )
     
    for i in range(1000):
       
       d[0] = random.gauss(0.,1.) 
       t.Fill()
     
    f.Write()
    f.Close()
    
    tf = TFile.Open(os.path.join(self.tempdir, "testHistogram.root"),"RECREATE")
    histMap = TH1F('mapping','',2,-20.,20.)
    histMap.SetBinContent(1,0.5)
    histMap.SetBinContent(2,2)
    #tf.cd()
    histMap.Write()
    tf.Close()
    
  def tearDown(self):
    # delete temporary directory
    super(TQHistogramObservableTest, self).tearDown()
  
  def runTest(self):
    self.test_simple_mapping()
  def test_simple_mapping(self):
    histogram = os.path.join(self.tempdir, "testHistogram.root:mapping")
    
    samples = TQSampleFolder("samples")
    testSample = samples.getSampleFolder("test+").getSample("testSample+")
    
    testSample.setTreeLocation(os.path.join(self.tempdir, "test.root:testTree"))
    testSample.setTagBool("usemcweights",True)
    #print "Creating test cuts"
    baseCutFolder = TQFolder("cuts")
    cutText = '+baseCut{<.cutExpression="1.",.weightExpression="1.",title="dummy base cut">\n+weightedCut{<.cutExpression="1.",.weightExpression="TH1Map:'+histogram+'([testVal])">}\n}'
    #print "Creating cuts from expression:"
    #print cutText
    baseCutFolder.importFromText(cutText)
    baseCut = TQCut.importFromFolder(baseCutFolder.getFolder("?"))
    #print "creating sample visitor"
    visitor = TQAnalysisSampleVisitor()
    visitor.setBaseCut(baseCut)
    #visitor.setVerbose(True)
    
    #print "creating histomaker analysis job"
    histoJob = TQHistoMakerAnalysisJob()
    if not histoJob.bookHistogram('TH1F("histo","",20,-2.,2.) << (testVal:"original distribution")'):
      print("Failed to book histogram")
        
    baseCut.addAnalysisJob(histoJob,"*")
    samples.visitMe(visitor)
    #samples.writeToFile("testSampleFolder.root")
    original = samples.getHistogram("test","baseCut/histo")
    scaled = samples.getHistogram("test","weightedCut/histo")
    ok = True
    for b in range(0,original.GetNbinsX()):
      exponent = -1 if original.GetBinCenter(b)<0 else 1
      #check if removing the scaling yield the same bin value
      ok = ok and (abs(original.GetBinContent(b) - scaled.GetBinContent(b)/pow(2,exponent)) < 1e-10)
      self.assertTrue(ok) #don't check for exact match, we might have some numerical discrepancies
    #print "Finished simple mapping test of TQHistogramObservable: "+("[ ok ]" if ok else "[fail]")
    
def suite():
  tests = ['test_simple_mapping']
  return unittest.TestSuite(map(TQHistogramObservableTest, tests))
  
  

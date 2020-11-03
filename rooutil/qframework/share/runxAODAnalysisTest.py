#!/usr/bin/env python2

def main():
  samples = TQSampleFolder.newSampleFolder("samples")
  sample = TQSample("test")
  testfile = "/afs/cern.ch/user/c/cburgard/public/testxAOD.root"
  sample.setTagString(".xsp.filepath",testfile)
  sample.setTagString(".xsp.treename","CollectionTree")
  samples.addObject(sample)
  
  TQTreeObservable.allowErrorMessages(True)
  init = TQSampleInitializer()
  init.setTagString("makeCounter","initial");
  sample.visitMe(init)

  tok = sample.getTreeToken()
  if tok:
    INFO("successfully opened xAOD file and retrieved TTree!")
  else:
    BREAK("unable to obtain tree token from '{:s}'!".format(testfile))
  sample.returnTreeToken(tok)

  basecut = TQCut("base")
  basecut.setWeightExpression("1.")
  basecut.setCutExpression("1.")

  histograms = TQHistoMakerAnalysisJob()
  histograms.bookHistogram("TH1F('leadmupt','p_{t}^{#mu,lead}',100,0,100)  << ( EventMM[0].part(0).pt()/1000. : 'p_{t}^{#mu,lead}')")
  basecut.addAnalysisJob(histograms)

  eventlists = TQEventlistAnalysisJob()
  eventlists.addColumn("EventInfo.runNumber()","run")
  eventlists.addColumn("EventInfo.eventNumber()","event")
  basecut.addAnalysisJob(eventlists)

  vis = TQAnalysisSampleVisitor()
  vis.setVerbose(True)
  vis.setBaseCut(basecut)
  samples.visitMe(vis)

  INFO("writing output file")

  samples.writeToFile("samples.root")
  
  #clean up function that needs to be called to prevent a segfault due to interference between ASG libraries and the python garbage collector:
  ROOT.xAOD.ClearTransientTrees()
  
if __name__ == "__main__":
  from QFramework import *
  from ROOT import *
  TQLibrary.getQLibrary().setApplicationName("libQFramwork_xAOD_test")
  main()

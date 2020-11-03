#!/usr/bin/env python


# the alias is the 'appliation name' that will be dumped to the info tags of any
# sample folder created
alias = "testAdd"

def main():
  base = TQSampleFolder("base")
  sf = TQSampleFolder("bkg")
  base.addFolder(sf)
  var1 = TQSampleFolder("em")
  var2 = TQSampleFolder("me")
  sf.addFolder(var1)
  sf.addFolder(var2)
  sf1 = TQSample("sample")
  sf2 = TQSample("sample")
  var1.addFolder(sf1)
  var2.addFolder(sf2)
  cf1 = TQFolder(".cutflow")
  cf2 = TQFolder(".cutflow")
  sf1.addFolder(cf1)
  sf2.addFolder(cf2)
  c1 = TQCounter("counter",11.,1.,111)
  c2 = TQCounter("counter",22.,2.,222)
  cf1.addObject(c1)
  cf2.addObject(c2)
  
  base.printContents("rdt")
  reader = TQSampleDataReader(base)
  reader.setVerbose(4)
  print("----------------------------------------------------")
  d1 = reader.getCounter("bkg/em/sample","counter")
  print("----------------------------------------------------")
  d2 = reader.getCounter("bkg/me/sample","counter")
  print("----------------------------------------------------")
  d3 = reader.getCounter("bkg/?/sample","counter")
  print("----------------------------------------------------")
  d4 = reader.getCounter("*/sample","counter")
  print("----------------------------------------------------")
  d5 = reader.getCounter("bkg/*","counter")
  print("----------------------------------------------------")
  d6 = reader.getCounter("bkg/?","counter")
  print("----------------------------------------------------")
  d7 = reader.getCounter("?/sample","counter") #intended to be non-matching
  print("----------------------------------------------------")
  
  if d1:
    print("d1 counter: value: {}, uncert: {}, raw: {}".format(d1.getCounter(),d1.getError(),d1.getRawCounter()))
  if d2:
    print("d2 counter: value: {}, uncert: {}, raw: {}".format(d2.getCounter(),d2.getError(),d2.getRawCounter()))
  if d3:
    print("d3 counter: value: {}, uncert: {}, raw: {}".format(d3.getCounter(),d3.getError(),d3.getRawCounter()))
  if d4:
    print("d4 counter: value: {}, uncert: {}, raw: {}".format(d4.getCounter(),d4.getError(),d4.getRawCounter()))
  if d5:
    print("d5 counter: value: {}, uncert: {}, raw: {}".format(d5.getCounter(),d5.getError(),d5.getRawCounter()))
  if d6:
    print("d6 counter: value: {}, uncert: {}, raw: {}".format(d6.getCounter(),d6.getError(),d6.getRawCounter()))
  if d7:
    #this should not be executed since d7 should be a null pointer
    print("d7 counter: value: {}, uncert: {}, raw: {}".format(d7.getCounter(),d7.getError(),d7.getRawCounter()))
  
  
  folderDesc = "+a/e/c/b/c/d;"
  otherSF = TQFolder("otherfolder")
  otherSF.importFromText(folderDesc)
  otherSF.printContents("rdt")
  print("Getting folders with pattern 'a/*/d'")
  listD = otherSF.getListOfFolders("a/*/d", TQFolder.Class(), True)
  listD.Print()
  for entry in listD:
    entry.__class__ = TQFolder
    entry.printContents()
    print("Path: {}".format(entry.getPath().Data()))
  print("Getting folders with pattern 'a/*/c'")
  listC = otherSF.getListOfFolders("a/*/c", TQFolder.Class(), True)
  listC.Print()
  for entry in listC:
    entry.__class__ = TQFolder
    entry.printContents()
    print("Path: {}".format(entry.getPath().Data()))
  
  

if __name__ == "__main__":
  # parse the CLI arguments
  from QFramework import *
  main()



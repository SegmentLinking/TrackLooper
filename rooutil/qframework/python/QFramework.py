def loadlibxml(libxmlpath):
  import ROOT
  # load libxml2, which is required for MVA stuff to work
  try:
    ROOT.gSystem.Load(libxmlpath)
  except Exception as ex:
    template = "unable to load libxml2 - an exception of type '{0}' occured: {1!s}"
    message = template.format(type(ex).__name__, ",".join(ex.args))
    raise ImportError("\033[1;31mFATAL\033[0m: "+message)

# import basics
from os import getenv
from distutils.version import StrictVersion
from platform import python_version

import ROOT
# check if the python version is compatible
if StrictVersion(python_version()) < StrictVersion('2.7.0'):
  raise ImportError("\033[1;31mFATAL\033[0m: unsupported python version, please use at least version 2.7.0")

# we will need this library later
ROOT.gSystem.Load("libTreePlayer")

# retrieve the root core dir environment variable
RootCoreDir = getenv ("ROOTCOREDIR")
CAF_BIN = getenv ("CAF_BIN")
if RootCoreDir:
    ROOT.gROOT.ProcessLine(".x $ROOTCOREDIR/scripts/load_packages.C")
    loadlibxml(ROOT.TQLibrary.getlibXMLpath().Data())
else:
    # we will need these libraries later
    loadlibxml(getenv ("LIBXMLPATH"))
    ROOT.gSystem.Load("libRooFit")
    if ROOT.gSystem.Load("libQFramework.so"): raise ImportError("unable to load QFramework")
    if ROOT.gSystem.DynamicPathName("libRooFitUtils.so",True):
      ROOT.gSystem.Load("libRooFitUtils.so")
    #ROOT.gSystem.Load("libSFramework.so")

# define some print commands that are available as preprocessor macros in the library
def BREAK(arg):
  ROOT.TQLibrary.msgStream.sendMessage(ROOT.TQMessageStream.BREAK,arg)
    
def ERROR(arg):
  ROOT.TQLibrary.msgStream.sendMessage(ROOT.TQMessageStream.ERROR,arg)

def CRITICAL(arg):
  ROOT.TQLibrary.msgStream.sendMessage(ROOT.TQMessageStream.CRITICAL,arg)
  
def INFO(arg):
  ROOT.TQLibrary.msgStream.sendMessage(ROOT.TQMessageStream.INFO,arg)

def WARN(arg):
  ROOT.TQLibrary.msgStream.sendMessage(ROOT.TQMessageStream.WARNING,arg)

def START(align,msg):
  ROOT.TQLibrary.msgStream.startProcessInfo(ROOT.TQMessageStream.INFO,min(ROOT.TQLibrary.getConsoleWidth(),120),align,msg)

def END(result):
  ROOT.TQLibrary.msgStream.endProcessInfo(result)

def parseException(ex):
  import re
  m = re.match("(.*) =>\n[ ]*(.*)[ ]\(C\+\+ exception of type ([^ ]*)\)",ex.args[0])
  if m:
    return m.group(3),m.group(1),m.group(2)
  else:
    raise RuntimeError("error parsing expression: "+str(ex.args[0]))
  
# provide sensible pretty-printing functionality for the basic classes
def TQTaggable__repr__(self):
  return "{:s}(\"{:s}\")".format(self.Class().GetName(),self.exportTagsAsString().Data())
ROOT.TQTaggable.__repr__ = TQTaggable__repr__

def TQTaggable__str__(self):
  return self.exportTagsAsString().Data()
ROOT.TQTaggable.__str__ = TQTaggable__str__

def TQFolder__repr__(self):
  return "{:s}(\"{:s}\") @ {:s}:{:s} {:s}".format(self.Class().GetName(),self.GetName(),self.getRoot().GetName(),self.getPath().Data(),self.exportTagsAsString().Data())
ROOT.TQFolder.__repr__ = TQFolder__repr__

def TQFolder__str__(self):
  return "{:s}:{:s} {:s}".format(ROOT.TQStringUtils.makeBoldBlue(self.getRoot().GetName()).Data(),ROOT.TQStringUtils.makeBoldWhite(self.getPath()).Data(),self.exportTagsAsString().Data())
ROOT.TQFolder.__str__ = TQFolder__str__

import unittest
class TQPythonTest(unittest.TestCase):
  """
  Base test for all QFramework test cases. This class provides a
  temporary directory which can be used to create external files. The
  directory is deleted when all test cases have been executed.
  """

  def setUp(self):
    """
    Create temporary directory.
    """
    import tempfile
    self.tempdir = tempfile.mkdtemp(prefix="qftest_")

  def tearDown(self):
    """
    Delete temporary directory.
    """   
    import os
    if os.path.exists(self.tempdir):
      import shutil
      shutil.rmtree(self.tempdir)

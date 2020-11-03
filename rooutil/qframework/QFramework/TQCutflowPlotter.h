//this file looks like plain C, but it's actually -*- c++ -*-
#ifndef __TQCutflowPlotter__
#define __TQCutflowPlotter__

#include "QFramework/TQCutflowPrinter.h"
#include "TColor.h"
#include "TROOT.h"
#include "QFramework/TQPlotter.h"

class TQCutflowPlotter : public TQPresenter {
protected:


  void writePlain(std::ostream& out, TQTaggable& tags);
  void writeTikZHead(std::ostream& out, TQTaggable& tags);
  void writeTikZBody(std::ostream& out, TQTaggable& tags);
  void writeTikZFoot(std::ostream& out, TQTaggable& tags);

public:

  static TString getColorDefStringLaTeX(const TString& name, int color);
  static TString getColorDefStringLaTeX(const TString& name, TColor* color);

  TQCutflowPlotter(TQSampleFolder* sf);
  TQCutflowPlotter(TQSampleDataReader* reader);

  void setup();

  bool writeToFile(const TString& filename, const TString& tags);
  bool writeToFile(const TString& filename, TQTaggable& tags);

  ClassDefOverride(TQCutflowPlotter,0) // presenter class to plot cutflows as bar charts

};

#endif

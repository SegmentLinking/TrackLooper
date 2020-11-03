//this file looks like plain C, but it's actually -*- c++ -*-
#ifndef __TQEventlistPrinter__
#define __TQEventlistPrinter__

#include "QFramework/TQPresenter.h"
#include "QFramework/TQTable.h"


class TQEventlistPrinter : public TQPresenter {

public:

  TQEventlistPrinter();
  TQEventlistPrinter(TQSampleFolder * samples);
  TQEventlistPrinter(TQSampleDataReader * reader);

  int writeEventlists(const TString& jobname, const TString& outputPath, const TString& tags);
  int writeEventlists(const TString& jobname, const TString& outputPath, TQTaggable tags);

  ClassDef(TQEventlistPrinter,0) // presenter class to print event lists

};

#endif

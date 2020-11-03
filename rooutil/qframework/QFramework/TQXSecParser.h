//this file looks like plain C, but it's actually -*- c++ -*-
#ifndef __TQXSecParser__
#define __TQXSecParser__

#include "QFramework/TQTable.h"
#include "QFramework/TQSample.h"
#include "TObjArray.h"

#ifdef __CINT__
typedef double Unit;
#endif

class TQXSecParser : public TQTable {
protected:
  TQSampleFolder* fSampleFolder; //!
  TObjArray* fPathVariants;

public:
  TQXSecParser(const TString& filename);
  TQXSecParser();
  TQXSecParser(TQTable& tab);
  TQXSecParser(TQTable* tab);
  TQXSecParser(TQXSecParser& parser);
  TQXSecParser(TQXSecParser* parser);
  TQXSecParser(TQSampleFolder* sf, const TString& xsecfilename);
  TQXSecParser(TQSampleFolder* sf);
  ~TQXSecParser();

  typedef double Unit;
  #ifndef __CINT__
  class Units { // nested
  public:
    static constexpr TQXSecParser::Unit millibarn = 1e9;
    static constexpr TQXSecParser::Unit microbarn = 1e6;
    static constexpr TQXSecParser::Unit nanobarn = 1e3;
    static constexpr TQXSecParser::Unit picobarn = 1;
    static constexpr TQXSecParser::Unit femtobarn = 1e-3;
    static constexpr TQXSecParser::Unit attobarn = 1e-6;
    static constexpr TQXSecParser::Unit UNKNOWN = std::numeric_limits<double>::quiet_NaN();
  };
  #endif
  static Unit unit(const TString& in);
  static TString unitName(Unit in);
  static double convertUnit(double in, const TString& inUnit, const TString& outUnit);
  static double convertUnit(double in, Unit inUnit, Unit outUnit);
  static void selectFirstColumn(std::vector<TString>* lines);

  int writeMappingToColumn (const TString& colname = "path");
  int readMappingFromColumn(const TString& colname = "path");
  int readFilePatternFromColumn(const TString& colname = "matchingName");

  int disableSamplesWithColumnStringMatch(const TString& colname, const TString& pattern, bool verbose = false);
  int enableSamplesWithColumnStringMatch(const TString& colname, const TString& pattern, bool verbose = false);
  void cloneSettingsFrom(TQXSecParser* parser);

  int enableSamplesWithPriorityLessThan (const TString& colname = "priority", int val = 2, bool verbose=false);
  int disableSamplesWithPriorityLessThan (const TString& colname = "priority", int val = 2, bool verbose=false);
  int enableSamplesWithPriorityGreaterThan (const TString& colname = "priority", int val = 1, bool verbose=false);
  int disableSamplesWithPriorityGreaterThan (const TString& colname = "priority", int val = 1, bool verbose=false);
  int enableAllSamples();
  int disableAllSamples();

  int applyWhitelist(const TString& filename);
  int applyWhitelist(std::vector<TString>* lines);
  int applyBlacklist(const TString& filename);
  int applyBlacklist(std::vector<TString>* lines);

  void addPathVariant(const TString& replacements);
  void addPathVariant(const TString& key, const TString& value);
  void printPathVariants();
  void clearPathVariants();
  bool isGood();

  int readMapping(const TString& filename, bool print=false);
  int readMapping(TQTable& tmp);
  bool hasCompleteMapping(bool requireEnabled = true);


  int addAllSamples(bool requireEnabled = true, bool requirePath = true);
  int addAllEnabledSamples();
  int addAllMappedSamples();
  int addAllListedSamples();
  int addAllSamplesFromPath(const TString& filesystempath, const TString& folderpath = "data", const TString& namefilter = "*.root", const TString& pathfilter = "*", const TString& tagstring = "generator='ATLAS', processinfo='p-p-collisions'");
  int addAllSamplesFromFolder(TQFolder* f, const TString& folderpath = "data", const TString& namefilter = "*.root", const TString& pathfilter = "./*", const TString& tagstring = "generator='ATLAS', processinfo='p-p-collisions'");
  int addAllSamplesFromList(const TString& inputfile, const TString& folderpath = "data", const TString& tagstring = "generator='ATLAS', processinfo='p-p-collisions'");

  TQSampleFolder* getSampleFolder();
  void setSampleFolder(TQSampleFolder* sf);

  ClassDefOverride(TQXSecParser,2) // parser for cross section files
};

#endif

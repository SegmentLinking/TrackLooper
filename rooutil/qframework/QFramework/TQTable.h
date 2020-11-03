//this file looks like plain C, but it's actually -*- c++ -*-
#ifndef __TQTABLE_H__
#define __TQTABLE_H__

#include <fstream>
#include <string>
#include <stdlib.h>
#include <iostream>
#include <map>

#include "QFramework/TQStringUtils.h"
#include "QFramework/TQNamedTaggable.h"
#include "TObjString.h"

class TQTable : public TQNamedTaggable {
protected:
  unsigned int nfields;
  unsigned int ncols;
  unsigned int nrows;
  TQTaggable** data; //[nfields]

  std::vector<int> hlines;
  std::vector<int> vlines;
  std::vector<TString> colAlign;
  bool autoExpand;
  bool manualAllocation; //!

  TQTaggable* getEntryInternal(unsigned int i, unsigned int j) const;
  TQTaggable* getEntryInternal(unsigned int i, unsigned int j, bool ensure);
  bool setContent(TQTaggable* entry,const TString& content, TString prior = "");
  TString formatEntryContents(TQTaggable* entry, const TString& format = "plain");
  void setup();
  TString makeExpSuffix(int exponent, const TString& format, bool useSIsuffix);

public: 
 
  int findColumn(TString colname, bool caseSensitive);
  int findColumn(TString colname, int row = 0, bool caseSensitive = true);
  int findRow(TString content, int column = 0, bool caseSensitive = true);
 
  TQTable();
  TQTable(const TString& name);
  TQTable(const TQTable* other);
  TQTable(const TQTable& other);
  TQTable(TList* l); 
  virtual ~TQTable();
 
  int appendLines(const TQTable& other, int startAt = 0, bool ignoreHlines = false);
  int appendLines(const TQTable* other, int startAt = 0, bool ignoreHlines = false);
  void merge(TQTable* other);
  

  TString getDetails();

  int setAllEntriesWhere(const TString& searchCol, const TString& searchVal, const TString& setCol, const TString& setVal, const TString& searchFormat = "ascii", const TString& setFormat = "ascii");

  std::map<TString,TString> getMap(const TString& key, const TString& value, const TString& keyformat = "ascii", const TString& valformat = "ascii");
  std::map<TString,TString> getMap(unsigned int keyidx, unsigned int validx, const TString& keyformat = "ascii", const TString& valformat = "ascii", bool skipfirstline = false);

  int getNcols();
  int getNrows();
  TString getColAlign(unsigned int col);
  TString getColAlignHTML(unsigned int col);

  void setAutoExpand(bool val);
  bool getAutoExpand();

  bool readCSVfile(const TString& fname, const TString& sep = ",", const TString& leftquote="\"", const TString& rightquote = "\"");
  bool readHTMLfile(const TString& fname);
  bool readLaTeXfile(const TString& fname);
  bool readTSVfile(const TString& fname, const TString& seps = " \t", int ignoreHeadLines=0, int nsepmin = 1);
  void readCSV(std::istream* input, const TString& sep = ",", const TString& leftquote="\"", const TString& rightquote = "\"");
  void readHTML(std::istream* input);
  void readLaTeX(std::istream* input);
  void readTSV(std::istream* input, const TString& seps = " \t", int nsepmin = 1);

  int readColumn(TQTable* other, const TString& colname, const TString& matchcolname);
  int readColumn(TQTable* other, const TString& colname, int thismatchcol, int othermatchcol);
  int readColumn(TQTable* other, int col, int thismatchcol,int othermatchcol);
  
  void dump();

  bool print (std::ostream* out, TQTaggable tags);
  bool printCSV (std::ostream* out, TQTaggable tags);
  bool printHTML (std::ostream* out, TQTaggable tags);
  bool printLaTeX(std::ostream* out, TQTaggable tags);
  bool printPlain(std::ostream* out, TQTaggable tags);
  bool print (std::ostream* out, TQTaggable* tags = NULL);
  bool printCSV (std::ostream* out, TQTaggable* tags = NULL);
  bool printHTML (std::ostream* out, TQTaggable* tags = NULL);
  bool printLaTeX(std::ostream* out, TQTaggable* tags = NULL);
  bool printPlain(std::ostream* out, TQTaggable* tags = NULL);
  bool print (std::ostream* out, const TString& tags);
  bool printCSV (std::ostream* out, const TString& tags);
  bool printHTML (std::ostream* out, const TString& tags);
  bool printLaTeX(std::ostream* out, const TString& tags);
  bool printPlain(std::ostream* out, const TString& tags);
  bool print (std::ostream* out, const char* tags);
  bool printCSV (std::ostream* out, const char* tags);
  bool printHTML (std::ostream* out, const char* tags);
  bool printLaTeX(std::ostream* out, const char* tags);
  bool printPlain(std::ostream* out, const char* tags);
  bool print (TQTaggable& tags);
  bool printCSV (TQTaggable& tags);
  bool printHTML (TQTaggable& tags);
  bool printLaTeX(TQTaggable& tags);
  bool printPlain(TQTaggable& tags);
  bool print (TQTaggable* tags = NULL);
  bool printCSV (TQTaggable* tags = NULL);
  bool printHTML (TQTaggable* tags = NULL);
  bool printLaTeX(TQTaggable* tags = NULL);
  bool printPlain(TQTaggable* tags = NULL);
  bool print (const TString& tags);
  bool printCSV (const TString& tags);
  bool printHTML (const TString& tags);
  bool printLaTeX(const TString& tags);
  bool printPlain(const TString& tags);
  bool print (const char* tags);
  bool printCSV (const char* tags);
  bool printHTML (const char* tags);
  bool printLaTeX(const char* tags);
  bool printPlain(const char* tags);

  bool write (const TString& fname, TQTaggable& tags);
  bool writeCSV (const TString& fname, TQTaggable& tags);
  bool writeHTML (const TString& fname, TQTaggable& tags);
  bool writeLaTeX(const TString& fname, TQTaggable& tags);
  bool writePlain(const TString& fname, TQTaggable& tags);
  bool write (const TString& fname, TQTaggable* tags = NULL);
  bool writeCSV (const TString& fname, TQTaggable* tags = NULL);
  bool writeHTML (const TString& fname, TQTaggable* tags = NULL);
  bool writeLaTeX(const TString& fname, TQTaggable* tags = NULL);
  bool writePlain(const TString& fname, TQTaggable* tags = NULL);
  bool write (const TString& fname, const TString& tags);
  bool writeCSV (const TString& fname, const TString& tags);
  bool writeHTML (const TString& fname, const TString& tags);
  bool writeLaTeX(const TString& fname, const TString& tags);
  bool writePlain(const TString& fname, const TString& tags);
  bool write (const TString& fname, const char* tags);
  bool writeCSV (const TString& fname, const char* tags);
  bool writeHTML (const TString& fname, const char* tags);
  bool writeLaTeX(const TString& fname, const char* tags);
  bool writePlain(const TString& fname, const char* tags);

  bool print(std::ostream* output, const TString& format, TQTaggable tags);
  bool write(const TString& fname, const TString& format, TQTaggable& tags);
  void clearColAlign();
 
  int getEntryInteger(unsigned int i, unsigned int j, bool sanitizeString = true);
  double getEntryDouble(unsigned int i, unsigned int j, bool sanitizeString = true);
  TString getEntry(unsigned int i, unsigned int j, TString format = "plain");
  TString getEntryPlain(unsigned int i, unsigned int j, bool allowUnicode = true);
  TString getEntryASCII(unsigned int i, unsigned int j);
  TString getEntryVerbatim(unsigned int i, unsigned int j);
  TString getEntryUnicode(unsigned int i, unsigned int j);
  TString getEntryLaTeX(unsigned int i, unsigned int j);
  TString getEntryHTML(unsigned int i, unsigned int j);
  bool setEntry(unsigned int i, unsigned int j, const TString& content, const TString& format = "");
  bool setEntry(unsigned int i, unsigned int j, const char* content, const TString& format = "");
  bool setEntryValue(unsigned int i, unsigned int j, int content);
  bool setEntryValue(unsigned int i, unsigned int j, double content);
  bool setEntryValueAndUncertainty(unsigned int i, unsigned int j, double value, double uncertainty);
  double getEntryValue(unsigned int i, unsigned int j, double defaultval);
  bool setProperty(unsigned int i, unsigned int j, const TString& key, const TString& value);
  bool setProperty(unsigned int i, unsigned int j, const TString& key, const char* value);
  bool setProperty(unsigned int i, unsigned int j, const TString& key, double value);
  bool setProperty(unsigned int i, unsigned int j, const TString& key, int value);
  void setColProperty(unsigned int j, const TString& key, const TString& value);
  void setColProperty(unsigned int j, const TString& key, const char* value);
  void setColProperty(unsigned int j, const TString& key, double value);
  void setColProperty(unsigned int j, const TString& key, int value);
  void setRowProperty(unsigned int i, const TString& key, const TString& value);
  void setRowProperty(unsigned int i, const TString& key, const char* value);
  void setRowProperty(unsigned int i, const TString& key, double value);
  void setRowProperty(unsigned int i, const TString& key, int value);
  void removeEntry(unsigned int i, unsigned int j);
  void clearRow(unsigned int row);
  void clearCol(unsigned int col);
 
  bool hasEntry(unsigned int i, unsigned int j);

  bool setVline(int col = 1, int type = 1);
  bool setHline(int row = 1, int type = 1);
  bool clearVlines();
  bool clearHlines();
  void setColAlign(unsigned int i, TString align);

  bool expand(unsigned int i, unsigned int j);
  bool shrink();
  int cleanup();
  int clear();

  TString getRowAsCSV(int row, const TString& sep = ",");

  const TList& makeTList(const TString& sep = ",");
  TList* makeTListPtr(const TString& sep = ",");
  void setFromTList(TList& l);
  void setFromTList(TList* l);
  void setListContents(TList* l, const TString& sep);
  void addToListContents(TList* l, const TString& sep);
  
  int markDifferences(TQTable* other, const TString& color, int colID = -1, int rowID = -1, const TString& format="plain");

  operator const TList& (){
    return this->makeTList(",");
  }
 
  operator TList* (){
    return this->makeTListPtr(",");
  }

  TQTable* operator=(TList* l){
    this->clear();
    this->appendLines(l);
    return this;
  }
 
  TQTable& operator=(TList& l){
    this->clear();
    this->appendLines(&l);
    return *this;
  }
 
  TList* operator=(TQTable* t){
    return t->makeTListPtr();
  }
 
 
#ifdef __TQXSecParser__
  friend class TQXSecParser;
#endif

  ClassDefOverride(TQTable,2) // representation of a table

};




#endif //__TABLE_H__

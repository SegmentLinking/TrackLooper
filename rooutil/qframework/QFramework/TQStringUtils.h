//this file looks like plain C, but it's actually -*- c++ -*-
#ifndef __TQStringUtils__
#define __TQStringUtils__

#include <iostream>
#include <sstream>
#include <math.h>


#include "TString.h"
#include "TList.h"
#include "TColor.h"
#include "TRegexp.h"
#include "Varargs.h"


namespace TQStringUtils {
 
  extern const TString lowerLetters;
  extern const TString upperLetters;
  extern const TString emptyString;
  extern const TString numerals;
  extern const TString letters;
  extern const TString alphanum;
  extern const TString alphanumvar;
  extern const TString alphanumvarext;
  extern const TString defaultIDchars;
  extern const TString blanks;
  extern const TString blanksAll;
  extern const TString controlReturnCharacters;
  extern const TRegexp latexcmd;
  extern const TRegexp latexmath;
  extern const TRegexp roottex;
  extern const TRegexp html;

  inline const TString& getLowerLetters() {
    return lowerLetters;
  }

  inline const TString& getCapitalLetters() {
    return upperLetters;
  }

  inline const TString& getLetters() {
    return letters;
  }
 
  inline const TString& getNumerals() {
    return numerals;
  }
 
  inline const TString& getDefaultIDCharacters() {
    return defaultIDchars;
  }
 
  inline const TString& getBlanks() {
    return blanks;
  }

  inline const TString& getAllBlanks() {
    return blanksAll;
  }

  inline TString makeBoldWhite(const TString& text) {
    return TString("\033[1m") + text + "\033[0m";
  }

  inline TString makeBoldPink(const TString& text) {
    return TString("\033[1;35m") + text + "\033[0m";
  }

  inline TString makeBoldBlue(const TString& text) {
    return TString("\033[1;34m") + text + "\033[0m";
  }

  inline TString makeBoldYellow(const TString& text) {
    return TString("\033[1;33m") + text + "\033[0m";
  }

  inline TString makeBoldGreen(const TString& text) {
    return TString("\033[1;32m") + text + "\033[0m";
  }

  inline TString makeBoldRed(const TString& text) {
    return TString("\033[1;31m") + text + "\033[0m";
  }

  inline TString makePink(const TString& text) {
    return TString("\033[35m") + text + "\033[0m";
  }

  inline TString makeTurquoise(const TString& text) {
    return TString("\033[36m") + text + "\033[0m";
  }

  inline TString makeBlue(const TString& text) {
    return TString("\033[34m") + text + "\033[0m";
  }

  inline TString makeYellow(const TString& text) {
    return TString("\033[33m") + text + "\033[0m";
  }

  inline TString makeGreen(const TString& text) {
    return TString("\033[32m") + text + "\033[0m";
  }

  inline TString makeRed(const TString& text) {
    return TString("\033[31m") + text + "\033[0m";
  }
 
  TString format(const char *va_(fmt), ...);
  char* vaFormat(const char *fmt, va_list ap); 

  bool readCharacter(const TString& text, int& index, int& count);
  TString getSIsuffix(int exponent, const TString& format = "ascii");
  TString getStatusBar(int pos, int max, const TString& def = "[> ]");

  TString getUniqueName(TDirectory * dir, TString proposal = "instance");

  TString getDetails(TObject * obj);
 
  bool isValidIdentifier(const TString& identifier, const TString& characters = getDefaultIDCharacters(), int minLength = -1, int maxLength = -1);
  TString makeValidIdentifier(const TString& identifier, const TString& characters = getDefaultIDCharacters(), const TString& replacement = "");

  bool getBoolFromString(TString boolString, bool &isBool);
  bool getBoolFromString(TString boolString);
  TString getStringFromBool(bool boolValue);
  int getColorFromString(TString colorString);
  int interpret(const TString& str);

  TString getColorDefStringLaTeX(const TString& name, int color);
  TString getColorDefStringLaTeX(const TString& name, TColor* color);

  bool isDouble (TString str);
  bool isInteger(TString str);
  bool isNumber (const TString& str);
  bool isBool (const TString& str);
 
  int getEditDistance(const TString& str1, const TString& str2);
  
  TString getLongestCommonSubstring(const std::vector<TString>& fullStrings_, const TString& seed);
  TString removeDuplicateSubStrings(const TString& longString, int minLength=16);
  TString removeDuplicateSubStrings(const TString& longString, TString& removedSequence, int minLength=16);
  
  int testNumber(double number, TString test);
 
  bool isEmpty(const TString& str, bool allowBlanks = false);

  bool matchesFilter(const TString& text, TString filter, const TString& orSep = "", bool ignoreBlanks = true);
  bool matches(const TString& text, const TString& pattern);
  bool hasWildcards(TString text);

  int compare(const TString& a, const TString& b);

  int compareHeads(const TString& str1, const TString& str2);
  int compareTails(const TString& str1, const TString& str2);
  TString getMaxCommonHead(TList * strings);

  TString trim(const TString& text, const TString& blanks = getBlanks());

  bool append(TString &text, const TString &appendix, const TString& sep = ", ");

  TString getFirstToken(TString text, const TString& sep = ",", bool trim = true, const TString& blocks = "", const TString& quotes = "");
  TList * tokenize(TString text, const TString& sep = ",", bool trim = true, const TString& blocks = "", const TString& quotes = "");
  std::vector<TString> tokenizeVector(TString text, const TString& sep = ",", bool trim = true, const TString& blocks = "", const TString& quotes = "");

  TString concat(TCollection * items, const TString& sep = ", ", const TString& quote="");

  template<class T, class U> T concat(const std::vector<T>& items, const U& sep) {
    // concatenate a vector of strings to a single string using the given separator
    bool first = true;
    std::stringstream ss;
    for(auto item:items){
      if (!first) {
        ss << sep;
      } else {
        first = false;
      }
      ss << item;
    }
    return ss.str().c_str();
  }
  template<class T> T concat(const std::vector<T>& items) {
    // concatenate a vector of strings to a single string using the given separator
    return TQStringUtils::concat(items,", ");
  }

#ifndef __CINT__
  template<class T> 
  TString concatNames(const std::vector<T*>& items, const TString& sep = ",") {
    // concatenate the names of a vector of TObject-derived objects to a single string using the given separator
    bool first = true;
    TString text;
    if(items.size() < 1) return "<empty>"; 
    for(size_t i=0; i<items.size(); i++){
      if (!first) {
        text.Append(sep);
      } else {
        first = false;
      }
      if(!items[i])
        text.Append("NULL");
      else
        text.Append(items[i]->GetName());
    }
 
    return text;
  }
#endif
 
  TString concatenate(int n, ...);
  
  bool isEscaped(const TString& text, int pos, const TString& escChar="\\");
  TString removeEscapes(const TString& text, const TString& escapes = "\\");
  TString insertEscapes(const TString& text, const TString& escapes = "\\\"");

  TString repeat(const TString& text, int n, const TString& sep = "");
  TString repeatSpaces(int n);
  TString repeatTabs(int n);

  int getWidth(const TString& text);
  int getCharIndex(const TString& text, int index);

  TString fixedWidth(const TString& text, int width, const TString& options = "l");
  TString fixedWidth(const char* text, int width, const TString& options = "l");
  TString fixedWidth(const TString& text, int width, bool rightJustified);
  TString fixedWidth(const TString& text, int width, const char* options);
  TString fixedWidth(const char* text, int width, const char* options);
  TString fixedWidth(const char* text, int width, bool rightJustified);

  TString maxLength(const TString& text, int maxLength, const TString& appendix = "...");
 
  TString getThousandsSeparators(int value, const TString& sep = "'");
  TString getThousandsSeparators(Long64_t value, const TString& sep = "'");
  TString getThousandsSeparators(TString value, const TString& sep = "'");
 
  TString getIndentationLines(int indent);

  TString formatSignificantDigits(double val, int nDigits);
  char hasStringSwitch(const TString& input);

  bool hasUnquotedStrings(const TString& text, const TString& quotes = "\"'");
  bool hasTFormulaParameters(const TString& text);

  bool printDiffOfLists(TList * l1, TList * l2, bool ignoreMatches = false);

  template <class T> 
  void printVector(const std::vector<T>& vec){
    int len = vec.size();
    TString idxf = TString::Format("%%%dd",(int)ceil(log(len)));
    for(size_t i=0; i<vec.size(); i++){
      std::cout << TString::Format(idxf.Data(),(int)i) << " " << vec[i] << std::endl;
    }
  }

  TString quote (const TString& str, char q = '"');
  TString unquote (TString str, const TString& quotes="\"'");
  void    unquoteInPlace(TString& text, const TString& quotes="\"'");
  TString unblock (TString str, const TString& blocks="(){}[]");


  TString cutUnit (TString &label);
  TString getUnit (TString label);
  TString getWithoutUnit (TString label);

  int removeLeading (TString &text, TString characters, int nMax = -1);
  int removeTrailing (TString &text, TString characters, int nMax = -1);

  bool removeLeadingText (TString &text, TString prefix);
  bool removeTrailingText (TString &text, TString appendix);

  int removeLeadingBlanks (TString &text, int nMax = -1);
  int removeTrailingBlanks(TString &text, int nMax = -1);
  
  int removeAll(TString &text, const TString& chars, TString::ECaseCompare comp = TString::ECaseCompare::kExact, int max=-1);
  
  int countLeading (const TString& text, const TString& characters, int nMax = -1);
  int countTrailing (const TString& text, const TString& characters, int nMax = -1);

  int countText(const TString& haystack, const TString& needle);

  int ensureTrailingText(TString &text, const TString& appendix);
  int ensureLeadingText(TString &text, const TString& prefix);

  int readBlanks (TString &in);
  int readBlanksAndNewlines(TString &in);
  int readBlanksAndNewlines(TString &in, int &nNewlines);
  int readToken (TString &in, TString &out, const TString& characters, int nMax = -1);
  int readBlock (TString &in, TString &out, const TString& blocks="()[]{}", const TString& quotes = "", bool keepEnclosed = false, int ignoreUnexpectedClosingQuotes = -1);
  int readUpTo (TString &in, TString &out, const TString& upTo, const TString& blocks = "", const TString& quotes = "", int ignoreUnexpectedClosingQuotes = -1);
  int readUpToText(TString &in, TString &out, const TString& upTo, const TString& blocks = "", const TString& quotes = "", int ignoreUnexpectedClosingQuotes = -1);
  int readTokenAndBlock(TString & in, TString & out, const TString& characters, const TString& blocks="()[]{}");
  TString readPrefix(TString &in, const TString& delim, const TString& defaultPrefix = "");
  TString expand(TString in, const TString& characters, const TString& blocks, bool embrace = false);
  
  TString replace(TString str, const TString& needle, const TString& newNeedle);
  TString replace(TString str, const char* needle, const TString& newNeedle);
  TString replace(TString str, const TString& needle, const char* newNeedle);
  TString replace(TString str, const char* needle, const char* newNeedle);

  bool equal(const TString& first, const TString& second);

  bool writeTextToFile(TList * text, const TString& filename = "");
  std::vector<TString>* readFileLines(const TString& filename, size_t len = 256, bool allowComments = true);
  size_t readFileLines(std::vector<TString>* lines, const TString& filename, size_t len = 256, bool allowComments = true);
  TString readFile(const TString& filename, const TString& blacklist = "", const TString& replace = "");
  TString readSVGtoDataURI(const TString& filename);
 
  TList* readDefinitionFile(const TString& filename);
 
  TString readTextFromFile(std::istream* input, const char* commentLine = "//", const char* commentBlockOpen = "/*", const char* commentBlockClose = "*/");
 
  TString compactify(const TString& text);
  TString minimize(TString text);

  inline TString chartostr(char c){ TString s(""); s+=c; return s;}
  size_t findParenthesisMatch (const TString& str, size_t nextpos, const TString& paropen, const TString& parclose);
  size_t findParenthesisMatch (const TString& str, size_t nextpos, char paropen, char parclose);
  size_t rfindParenthesisMatch(const TString& str, size_t nextpos, const TString& paropen, const TString& parclose);
  size_t rfindParenthesisMatch(const TString& str, size_t nextpos, char paropen, char parclose);
  size_t findFree (const TString& haystack, const TString& needle, const TString& paropen, const TString& parclose, size_t startpos = 0);
  size_t findFree (const TString& haystack, const TString& needle, const TString& parentheses, size_t startpos = 0);
  size_t rfindFree (const TString& haystack, const TString& needle, const TString& paropen, const TString& parclose, size_t startpos = -1);
  size_t rfindFree (const TString& haystack, const TString& needle, const TString& parentheses, size_t startpos = 0);
  size_t findFreeOf (const TString& haystack, const TString& needles, const TString& paropen, const TString& parclose, size_t startpos = 0);
  size_t rfindFreeOf (const TString& haystack, const TString& needles, const TString& paropen, const TString& parclose, size_t startpos = -1);

  int find(const TString& item, const std::vector<TString>& vec);

  int find(const TString& haystack, const TString& needle, int pos = 0);
  int rfind(const TString& haystack, const TString& needle, int pos);
  int findFirstOf(const TString& haystack, const TString& needle, int pos = 0);
  int rfindFirstOf(const TString& haystack, const TString& needle, int pos = 0);
  int findLastOf(const TString& haystack, const TString& needle, int pos);
  int findFirstNotOf(const TString& haystack, const TString& needle, int pos = 0);
  int rfindFirstNotOf(const TString& haystack, const TString& needle, int pos = 0);
  int findLastNotOf(const TString& haystack, const TString& needle, int pos);
 
  inline int rfind (const TString& haystack, const TString& needle){return TQStringUtils::rfind (haystack,needle,haystack.Length());}
  inline int findLastOf (const TString& haystack, const TString& needle){return TQStringUtils::findLastOf (haystack,needle,haystack.Length());}
  inline int findLastNotOf(const TString& haystack, const TString& needle){return TQStringUtils::findLastNotOf(haystack,needle,haystack.Length());}
 
  std::vector<TString> split(const TString& str, const TString& del = " ");
  std::vector<TString> split(const TString& str, const TString& del, const TString& paropen, const TString& parclose);

  TString findFormat(const TString& content);

  TString convertPlain2LaTeX(TString text);
  TString convertPlain2HTML(TString text); 
  TString convertLaTeX2Plain(TString text,bool unicod=true);
  TString convertLaTeX2HTML(TString text); 
  TString convertHTML2Plain(TString text,bool unicode=true);
  TString convertHTML2LaTeX(TString text);
  TString convertROOTTeX2Plain(TString text, bool unicode=true);
  TString convertROOTTeX2LaTeX(TString text);
  TString convertROOTTeX2HTML(TString text);
  TString convertLaTeX2ROOTTeX(TString text);
  TString makeUnicodeSuperscript(TString s);
  TString makeUnicodeSubscript(TString s);
  TString makeUppercase(TString s);
  TString makeLowercase(TString s);
  TString makeASCII(const TString& content);
  bool isASCII(TString content);
  
  TString concat(TString first, const TString& second);

  TString formatValueError(double val, double err, const TString& format = "%g +/- %g");
  TString formatValueErrorPDG(double val, double err, int expval, const TString& format);
  
  int reduceToCommonPrefix(TString& prefix, const TString& other);
  TString replaceEnclosed(TString haystack,TString needle,TString newNeedle, const TString& symbols);
  
  TString padNumber(int num, int length);

}

#endif

#ifdef _UNICODE
typedef wchar_t TCHAR;
#else
typedef char TCHAR;
#endif

#include "QFramework/TQStringUtils.h"
#include "QFramework/TQUtils.h"
#include "QFramework/TQHistogramUtils.h"
#include "QFramework/TQCounter.h"
#include "QFramework/TQIterator.h"
#include "QFramework/TQPCA.h"
#include "QFramework/TQSample.h"
#include "QFramework/TQLink.h"
#include "QFramework/TQTable.h"

// #define _DEBUG_
#include "QFramework/TQLibrary.h"

#include "TObjString.h"
#include "TMath.h"
#include "TColor.h"
#include "TPrincipal.h"
#include "THStack.h"
#include "Varargs.h"
#include "TLegend.h"
#include "TInterpreter.h"
#include "TH1.h"
#include "TParameter.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>

////////////////////////////////////////////////////////////////////////////////////////////////
//
// TQStringUtils:
//
// The TQStringUtils namespace provides a set of static utility methods related to the inspection
// and manipulation of strings.
//
////////////////////////////////////////////////////////////////////////////////////////////////

const TString TQStringUtils::lowerLetters = "abcdefghijklmnopqrstuvwxyz";
const TString TQStringUtils::upperLetters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
const TString TQStringUtils::emptyString = "";
const TString TQStringUtils::numerals = "0123456789";
const TString TQStringUtils::letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
const TString TQStringUtils::alphanum = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
const TString TQStringUtils::alphanumvar = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789$()";
const TString TQStringUtils::alphanumvarext = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789$()_";
const TString TQStringUtils::defaultIDchars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.";
const TString TQStringUtils::controlReturnCharacters = "mAC";
const TString TQStringUtils::blanks = " \t";
const TString TQStringUtils::blanksAll = " \t\n\r";


const TRegexp TQStringUtils::latexcmd = "\\\\[a-zA-Z]+[{ }$]?";
const TRegexp TQStringUtils::latexmath = "\\$.*\\$";
const TRegexp TQStringUtils::roottex = "#[a-zA-Z]+";
const TRegexp TQStringUtils::html = "<[a-zA-Z]+>.*</[a-zA-Z]+>";


//__________________________________________________________________________________|___________

TString TQStringUtils::getUniqueName(TDirectory * dir, TString proposal) {
  // Returns a string that does not exist as object name in TDirectory instance
  // <dir>. The string is constructed from input string <proposal> and if an object
  // with the same name is present in <dir> an increasing integer is appended
  // ("_iN") until the name is unique.
 
  // append increasing integer to <proposal> until name is unique
  TString name = proposal;
  int i = 2;
  while (dir && dir->FindObject(name.Data())) {
    name = TString::Format("%s_i%d", proposal.Data(), i++);
  }

  return name;
}


//__________________________________________________________________________________|___________

TString TQStringUtils::getDetails(TObject * obj) {
  // return details on some TObject as a TString
  if (!obj) {
    return TString("<invalid pointer>");
  } else if (obj->InheritsFrom(TH1::Class())) {
    return TQHistogramUtils::getDetailsAsString((TH1*)obj,2);
  } else if (obj->InheritsFrom(TQCounter::Class())) {
    return ((TQCounter*)obj)->getAsString();
  } else if (obj->InheritsFrom(TPrincipal::Class())) {
    return TQHistogramUtils::getDetailsAsString((TPrincipal*)obj);
  } else if (obj->InheritsFrom(TQPCA::Class())) {
    return ((TQPCA*)obj)->getDetailsAsString();
  } else if (obj->InheritsFrom(TQSample::Class())) {
    TString generator = "<unknown generator>";
    TString process = "<unknown process>";
    TString simulation = "";
    TQSample * s = (TQSample*)obj;
    TString retval;
    //@tag:[.xps.generator,.xsp.simulation,.xsp.process] The contents of these sample tags are included in the return value of TQStringUtils::getDetails(...).
    if(s->getTagString(".xsp.generator", generator)){
      retval.Append(generator);
    }
    if(s->getTagString(".xsp.simulation", simulation)){
      if(!retval.IsNull()){
        retval.Append(" (");
        retval.Append(simulation);
        retval.Append(")");
      }
    }
    if(s->getTagString(".xsp.process", process)){
      if(!retval.IsNull())
        retval.Append(": ");
      retval.Append(process);
    }
    double norm = s->getNormalisation();
    if(norm != 1){
      if(!retval.IsNull())
        retval.Append(" - ");
      retval.Append("norm.=");
      if(norm > 0) retval.Append(TString::Format("%.6f",norm));
      else if(norm < 0) retval.Append(TQStringUtils::makeBoldRed(TString::Format("%.3f",norm)));
      else retval.Append(TQStringUtils::makeBoldYellow("0"));
    } 
    if(s->hasSubSamples()){
      if(!retval.IsNull()) retval.Append(" - ");
      retval.Append("multisample");
    }
    return retval;
  } else if (obj->InheritsFrom(TQValue::Class())) {
    return ((TQValue*)obj)->getString();
  } else if (obj->InheritsFrom(TParameter<double>::Class())) {
    return TString::Format("%g", ((TParameter<double>*)obj)->GetVal());
  } else if (obj->InheritsFrom(TCollection::Class())) {
    int nEntries = ((TCollection*)obj)->GetEntries();
    if (nEntries != 1) return TString::Format("%d entries", nEntries);
    else return "1 entry";
  } else if (obj->InheritsFrom(TQLink::Class())) {
    return TString("--> ") + ((TQLink*)obj)->getDestAsString();
  } else if (obj->InheritsFrom(TQTable::Class())) {
    return ((TQTable*)obj)->getDetails();
  } else if (obj->InheritsFrom(TPrincipal::Class())) {
    return TQHistogramUtils::getDetailsAsString((TPrincipal*)obj);
  } else if (obj->InheritsFrom(TGraph::Class())) {
    return TQHistogramUtils::getDetailsAsString((TGraph*)obj);
  } else if (obj->InheritsFrom(TLegend::Class())){
    TLegend* l = (TLegend*)obj;
    return TString::Format("%d entries (%d columns, %d rows)",(l->GetListOfPrimitives() ? l->GetListOfPrimitives()->GetEntries() : 0),l->GetNColumns(),l->GetNRows());
  } else if (obj->InheritsFrom(THStack::Class())){
    THStack* s = (THStack*)(obj);
    return s->GetStack() ? TString::Format("%d histograms",(s->GetStack()->GetEntries())) : "(empty)";
  } else if (obj->InheritsFrom(TQFolder::Class())){
    TQFolder* f = dynamic_cast<TQFolder*>(obj);
    int nObj = f->getNObjects();
    int nTag = f->getNTags();
    TString retval;
    if(nObj == 0) retval += "no objects";
    else if(nObj == 1) retval += "1 object";
    else retval += TString::Format("%d objects",nObj);
    retval += ", ";
    if(nTag == 0) retval += "no tags";
    else if(nTag == 1) retval += "1 tag";
    else retval += TString::Format("%d tags",nTag);
    return retval;
  } else {
    return TString::Format("<no details for class %s available>", obj->IsA()->GetName());
  }
}


//__________________________________________________________________________________|___________

TString TQStringUtils::getStatusBar(int pos, int max, const TString& def) {
  // produce a "status bar" from some pos/max integer fraction
  // def is expected to have length 4 (default is "[> ]")
  // def[0] is a starting character (default: "[")
  // def[1] is a "done" character (default: ">")
  // def[2] is a "pending" character (default: " ")
  // def[3] is a terminating character (default: "]")

  if (def.Length() != 4) {
    return "";
  }

  TString bar;

  bar.Append(def[0]);

  for (int i = 1; i <= max; i++) {
    if (i <= pos) {
      bar.Append(def[1]);
    } else {
      bar.Append(def[2]);
    }
  }

  bar.Append(def[3]);

  return bar;
}


//__________________________________________________________________________________|___________

bool TQStringUtils::isValidIdentifier(const TString& identifier,
                                      const TString& characters, int minLength, int maxLength) {
  // check if the given string is a valid identifier, that is,
  // - contains only the allowed characters
  // - having a minimum length of minLength
  // - having a maximum length of maxLength

  if (minLength >= 0 && identifier.Length() < minLength) {
    // identifier too short
    return false;
  }

  if (maxLength >= 0 && identifier.Length() > maxLength) {
    // identifier too long
    return false;
  }

  // search for invalid characters
  for (int i = 0; i < identifier.Length(); i++) {
    if (characters.Index(identifier[i]) == kNPOS) {
      // character in <identifier> not listed in <characters>
      return false;
    }
  }

  return true;
}

//__________________________________________________________________________________|___________

TString TQStringUtils::getSIsuffix(int exponent, const TString& format){
  // retrieve the SI suffix corresponding to the given exponent
  // special formats include latex, html and unicode
  if(exponent > 0){
    if(exponent < 3) return "";
    if(exponent < 6) return "k";
    if(exponent < 9) return "M";
    if(exponent < 12) return "T";
    if(exponent < 15) return "P";
    if(exponent < 18) return "Z";
  } else {
    if(exponent > -3) return "";
    if(exponent > -6) return "m";
    if(exponent > -9){
      if(format == "latex"){
        return "\\mu";
      } else if(format=="html"){
        return "&mu;";
      } else if(format=="unicode"){
        return "µ";
      } else {
        return "mu";
      }
    }
    if(exponent > -12) return "n";
    if(exponent > -15) return "p";
    if(exponent > -18) return "f";
    if(exponent > -21) return "a";
    if(exponent > -24) return "z";
    if(exponent > -27) return "y";
  }
  return "?";
}

//__________________________________________________________________________________|___________

bool TQStringUtils::isEmpty(const TString& str, bool allowBlanks) {
  // Returns true if the input string <str> is empty and false otherwise. If
  // <allowBlanks> is true the input string is considered empty even if it still
  // contains blanks (listed in TQStringUtils::getBlanks()).
 
  if (allowBlanks) {
    // ignoring blanks means number of blanks in string must be equal to total length
    return TQStringUtils::countLeading(str, TQStringUtils::getBlanks()) == str.Length();
  } else {
    return str.IsNull();
  }
}

//__________________________________________________________________________________|___________

bool TQStringUtils::equal(const TString& first, const TString& second){
  // returns true if the input strings are equal
  return (first.CompareTo(second) == 0);
}

//__________________________________________________________________________________|___________

TString TQStringUtils::makeValidIdentifier(const TString& identifier,
                                           const TString& characters, const TString& replacement) {
  // converts the given string into a valid identifier
  // all characters not contained in the set of allowed characters
  // are replaced by the given string

  /* the string to return */
  TString result;

  /* replace invalid characters by 'replacement' */
  int pos = -1;
  while (++pos < identifier.Length()) {
    if (characters.Index(identifier[pos]) != kNPOS)
      result.Append(identifier[pos]);
    else
      result.Append(replacement);
  }

  /* return result */
  return result;
}

//__________________________________________________________________________________|___________

bool TQStringUtils::getBoolFromString(TString boolString, bool &isBool) {
  // converts any type of textual boolean expression into a bool
  // values like yes, no, ok, fail, true, false, etc. will be accepted
  // result will be returned, success is indicated by value written to second argument
  boolString = trim(boolString);

  bool isTrue =
    (boolString.CompareTo("yes", TString::kIgnoreCase) == 0) ||
    (boolString.CompareTo("ok", TString::kIgnoreCase) == 0) ||
    (boolString.CompareTo("true", TString::kIgnoreCase) == 0) ||
    (TQStringUtils::isNumber(boolString) && boolString.Atof() != 0.);

  bool isFalse =
    (boolString.CompareTo("no", TString::kIgnoreCase) == 0) ||
    (boolString.CompareTo("fail", TString::kIgnoreCase) == 0) ||
    (boolString.CompareTo("false", TString::kIgnoreCase) == 0) ||
    (TQStringUtils::isNumber(boolString) && boolString.Atof() == 0.);

  if (isFalse && !isTrue) {
    /* bool in string is "false" */
    isBool = true;
    return false;
  } else if (!isFalse && isTrue) {
    /* bool in string is "true" */
    isBool = true;
    return true;
  } else {
    /* no bool in string found or
     * both "true" and "false" */
    isBool = false;
    return false;
  }
}

//__________________________________________________________________________________|___________

int TQStringUtils::interpret(const TString& str){
  // pass some string through the root interpreter
  bool errMsg = gInterpreter->IsErrorMessagesEnabled();
  gInterpreter->SetErrorMessages(false);
  int val = gROOT->ProcessLine(str);
  gInterpreter->SetErrorMessages(errMsg);
  return val;
}

//__________________________________________________________________________________|___________

int TQStringUtils::getColorFromString(TString colorString) {
  // converts ROOT color identifiers (kRed, kGreen, ...) 
  // and rgb hex codes (#xxxxxx) 
  // to the corresponding integer color identifier

  if ((colorString.Index('#') == 0) && (colorString.Length() == 7)) {
    // test for color string of the form "#xxxxxx"
    return TColor::GetColor(colorString);
  }
 
  colorString.Append(";");
  return TQStringUtils::interpret(colorString);
}


//__________________________________________________________________________________|___________

bool TQStringUtils::getBoolFromString(TString boolString) {
  // converts any type of textual boolean expression into a bool
  // values like yes, no, ok, fail, true, false, etc. will be accepted
  bool isBool = false;
  return getBoolFromString(boolString, isBool);
}


//__________________________________________________________________________________|___________

TString TQStringUtils::getStringFromBool(bool boolValue) {
  // converts a boolean value to its corresponding string (true or false)
  if (boolValue)
    return "true";
  else
    return "false";
}


//__________________________________________________________________________________|___________

bool TQStringUtils::isDouble(TString str) {
  // Returns true if the input string <str> represents a double value or false
  // otherwise. Please note: for pure integer values this method returns false, e.g.
  // isDouble("4") will evaluate to false while isDouble("4.") and isDouble("4E0")
  // will evaluate to true.

  // remove leading and trailing blanks
  str = trim(str);

  // get the number of special characters
  TString dummy;
  int nSign = readToken(str, dummy, "+-");
  if(TQStringUtils::equal(str,"inf")) return true;
  int nNumPre = readToken(str, dummy, getNumerals());
  int nDots = readToken(str, dummy, ".");
  int nNumPost = readToken(str, dummy, getNumerals());
  int nExp = removeLeading(str, "eE");
  int nExpSign = readToken(str, dummy, "+-");
  int nNumExp = readToken(str, dummy, getNumerals());

  return str.IsNull() && nSign <= 1 && (nNumPre > 0 || nNumPost > 0) &&
    ((nDots == 1 && (nExp + nExpSign + nNumExp) == 0) ||
     (nExp == 1 && nExpSign <= 1 && nNumExp > 0 && nDots <= 1));
}


//__________________________________________________________________________________|___________

bool TQStringUtils::isInteger(TString str) {
  // Returns true if the input string <str> represents an integer value or false
  // otherwise. Please note: for integer values given in "double notation" false is
  // returned, e.g. isInteger("4.") and isInteger("4E0") will evaluate to false.
 
  // remove leading and trailing blanks
  str = trim(str);

  // get the number of special characters
  TString dummy;
  TString strExp;
  int nSign = readToken(str, dummy, "+-");
  int nNum = readToken(str, dummy, getNumerals());

  return str.IsNull() && nSign <= 1 && nNum > 0;
}


//__________________________________________________________________________________|___________

bool TQStringUtils::isNumber(const TString& str) {
  // Return true if input string <str> represents either a double or an integer
  // value. This method is equivalent to "isInteger(str) || isDouble(str)".
 
  return isInteger(str) || isDouble(str);
}


//__________________________________________________________________________________|___________

bool TQStringUtils::isBool(const TString& str) {
  // returns true if the given string is a valid boolean expression
  bool isBool = false;
  getBoolFromString(str, isBool);
  return isBool;
}


//__________________________________________________________________________________|___________

int TQStringUtils::getEditDistance(const TString& str1, const TString& str2) {
  // returns the number of edits required to convert str1 to str2
  int m = str1.Length();
  int n = str2.Length();
 
  // d[i, j] := d[i + (m + 1) * j]
  unsigned int * d = new unsigned int[(m + 1) * (n + 1)];
 
  for (int i = 0; i <= m; i++) {
    d[i] = i;
  }
  for (int j = 0; j <= n; j++) {
    d[(m + 1) * j] = j;
  }

  for (int j = 1; j <= n; j++) {
    for (int i = 1; i <= m; i++) {
      if (str1[i - 1] == str2[j - 1]) { 
        // no operation required
        d[i + (m + 1) * j] = d[(i - 1) + (m + 1) * (j - 1)]; 
      } else {
        // a deletion
        int del = d[(i - 1) + (m + 1) * j] + 1;
        // an insertion
        int ins = d[i + (m + 1) * (j - 1)] + 1;
        // a substitution
        int sub = d[(i - 1) + (m + 1) * (j - 1)] + 1;
 
        d[i + (m + 1) * j] = TMath::Min(del, TMath::Min(ins, sub));
      }
    }
  }
 
  // the final edit distance
  int dist = d[m + (m + 1) * n];

  // clean up
  delete [] d;

  return dist;
}

//__________________________________________________________________________________|___________

TString TQStringUtils::getLongestCommonSubstring(const std::vector<TString>& fullStrings_, const TString& seed){
  // returns the longest string containing the seed which is a substring of all strings provided.
  // Please note that if seed appears multiple times in a string, only the first occurance is considered!
  
  std::vector<TString> fullStrings = fullStrings_;
  bool ok = true;
  size_t seedLength = seed.Length();
  TString str1,str2;
  size_t pos1,pos2;
  while (ok && fullStrings.size()>1) {
    str1 = fullStrings.back(); fullStrings.pop_back();
    str2 = fullStrings.back(); fullStrings.pop_back();
    
    pos1 = TQStringUtils::find(str1,seed,0);
    if (pos1==kNPOS) { //we need to find at least one occurance of the seed
      ERRORfunc("Cannot determine longest common substring with seed '%s': failed to find seed in string '%s'",seed.Data(),str1.Data());
      return TString("");
    }
    pos2 = TQStringUtils::find(str2,seed,0);
    if (pos2==kNPOS) { //we need to find at least one occurance of the seed
      ERRORfunc("Cannot determine longest common substring with seed '%s': failed to find seed in string '%s'",seed.Data(),str2.Data());
      return TString("");
    }
    size_t lenPre = TQStringUtils::compareTails( str1(0,pos1) , str2(0,pos2) );
    size_t lenPost = TQStringUtils::compareHeads( str1(pos1+seedLength,str1.Length()), str2(pos2+seedLength,str2.Length()) );
    TString common = str1(pos1-lenPre,lenPre+seedLength+lenPost);
    fullStrings.push_back(common); //re-queue the common string
    
  }
  
  if ( ok && fullStrings.size()>0 ) return fullStrings[0];
  return TString("");
}

TString TQStringUtils::removeDuplicateSubStrings(const TString& longString, int minLength) {
  // convenience wrapper in case the removed sequence is not of interest
  TString dummy;
  return TQStringUtils::removeDuplicateSubStrings(longString,dummy, minLength);
}

TString TQStringUtils::removeDuplicateSubStrings(const TString& longString_, TString& removedSequence, int minLength) {
  // checks for substrings of 'longString' which are at least 'minLength' 
  // characters long and (disjointly) occur multiple times in 'longString', 
  // removing all but the first occurence. The identified substring is
  // additionally stored in the 'removedSequence' parameter.
  // In case of ambiguities, i.e., a shorter but more frequently occuring substring
  // versus a longer but less frequently occuring substring, the more frequent
  // one is treated (given it is at least 'minLength' characters long)  
  
  //reset "removedSequence" (we should always indicate what we removed, so if nothing to be purged is found below we ensure to report an empty string)
  removedSequence = "";
  int maxOccurances = 0;
  TString longString = longString_;
  int pos = 0;
  int firstPos = 0;
  //first identify the substring to be purged
  
  while (pos<longString.Length()-2*minLength) { //no need to continue checking if there is no chance we'll encounter at least two occurances
    int thisLength = minLength;
    int theseOccurances = TQStringUtils::countText(longString(pos+minLength,longString.Length()-pos-minLength),longString(pos,minLength));
    if (theseOccurances > maxOccurances) {
      maxOccurances = theseOccurances;
      firstPos = pos; //store for later use (so we don't need to find it again)
      //note on the loop condition: technically, we should never have more matches than before
      while( theseOccurances >= maxOccurances ) { //abort if we expanded the substring length such that we have fewer matches
        ++thisLength;
        //check how many matches we get with the inreased substring length
        theseOccurances = TQStringUtils::countText(longString(pos+thisLength,longString.Length()-pos-thisLength),longString(pos,thisLength));
      }
      //since we increase the substring length even if the number of matches decreases for that length, we need to adjust for this offset (-> pre-decrement when obtaining the substring!). 
      //make a copy of the found substring
      removedSequence = TString(longString(pos,--thisLength)); //pre-decrement, see above!
    }
    //move on to the next position
    ++pos;
  }
  //now we have the most frequent substring which fulfills the requirements
  //so let's remove all but the first occurance:
  int toRemove = TQStringUtils::find(longString,removedSequence,firstPos+removedSequence.Length());
  
  if (toRemove == kNPOS) ERRORfunc("Logic error in string parsing detected!"); 
  while (toRemove != kNPOS) {
    longString.Remove(toRemove,removedSequence.Length());
    toRemove = TQStringUtils::find(longString,removedSequence,firstPos+removedSequence.Length());
    --maxOccurances; //a little consistency check
  }
  if (maxOccurances != 0) {
    ERRORfunc("Logic error in string parsing detected! %d occurances have not been removed",maxOccurances);
  }    
  return longString;
  
}

//__________________________________________________________________________________|___________

int TQStringUtils::testNumber(double number, TString test) {
  // Tests number <number> with simple condition <test> and returns 1 if the
  // condition is fulfilled and 0 otherwise. The syntax of <test> is expected
  // to be "<operator> <number>". Supported operators are "==", "!=", ">=", ">",
  // "<=", and "<". -1 is returned in case the condition could not be parsed.
  // Examples:
  //
  // - testNumber(5, "== 5.") returns 1
  // - testNumber(4, "> 5.") returns 0
  // - testNumber(5, ">> 1") returns -1
 
  // read operator
  TString op;
  TQStringUtils::removeLeadingBlanks(test);
  if (!TQStringUtils::readToken(test, op, "!=<>")) {
    // missing operator
    return -1;
  }
 
  // read number
  TString strNum;
  TQStringUtils::removeLeadingBlanks(test);
  if (!TQStringUtils::readToken(test, strNum, TQStringUtils::getNumerals() + "+-.")) {
    // missing number
    return -1;
  }
  if (!TQStringUtils::isNumber(strNum)) {
    // not a valid number
    return -1;
  }
  double myNum = strNum.Atof();

  // don't expect anything more in input string
  TQStringUtils::removeLeadingBlanks(test);
  if (!test.IsNull()) {
    return -1;
  }
 
  if (op.CompareTo("==") == 0) {
    return (number == myNum) ? 1 : 0;
  } else if (op.CompareTo("!=") == 0) {
    return (number != myNum) ? 1 : 0;
  } else if (op.CompareTo(">=") == 0) {
    return (number >= myNum) ? 1 : 0;
  } else if (op.CompareTo("<=") == 0) {
    return (number <= myNum) ? 1 : 0;
  } else if (op.CompareTo(">") == 0) {
    return (number > myNum) ? 1 : 0;
  } else if (op.CompareTo("<") == 0) {
    return (number < myNum) ? 1 : 0;
  } else {
    // unknown operator
    return -1;
  }
}


//__________________________________________________________________________________|___________

bool TQStringUtils::matchesFilter(const TString& text, TString filter,
                                  const TString& orSep, bool ignoreBlanks) {

  // match with logical OR?
  if (!orSep.IsNull()) {
    /* loop over individual filter */
    while (!filter.IsNull()) {
      TString thisFilter;
      readUpTo(filter, thisFilter, orSep);
      if (ignoreBlanks)
        thisFilter = trim(thisFilter);
      if (matchesFilter(text, thisFilter))
        return true;
      removeLeading(filter, orSep, 1);
    }
    return false;
  } else {
    return ((TQStringUtils::removeLeading(filter, "!", 1) > 0) != matches(text, filter));
  }
}


//__________________________________________________________________________________|___________

bool TQStringUtils::matches(const TString& text, const TString& pattern) {
  // Performs a string match between the input string <text> and the string pattern
  // <pattern> and returns true in case of a match and false otherwise. The string
  // pattern may use wildcards "?" (matching exactly one character) and "*"
  // (matching any string sequence).
  //
  // Examples:
  //
  // - matches("hello", "h?llo") returns true
  // - matches("hallo", "h?llo") returns true
  // - matches("hello", "h*") returns true
  // - matches("hello", "h*a") returns false
 
  if (text.Length() > 0 && pattern.Length() > 0) {
    // direct and "?" match ?
    if (text[0] == pattern[0] || pattern[0] == '?') {
      // TODO: avoid recursion
      return matches(text(1, text.Length()), pattern(1, pattern.Length()));
    }
    // "*" match ?
    if (pattern[0] == '*') {
      // eating leading "*" in pattern ...
      return matches(text, pattern(1, pattern.Length()))
        // ... or matching leading character in text ?
        || matches(text(1, text.Length()), pattern);
    }
    // no match
    return false;
  } else {
    // empty text and/or pattern
    return (text.Length() == 0 && (pattern.CompareTo('*') == 0 || pattern.Length() == 0));
  }
}


//__________________________________________________________________________________|___________
/*
  bool TQStringUtils::matchesExperimental(TString text, TString pattern, TList * wildcardMatches) {

  if (text.Length() > 0 && pattern.Length() > 0) {
  int i = 0;
  while (pattern.Length() > i && (text[i] == pattern[i] || pattern[i] == '?')) {
  if (wildcardMatches && pattern[i] == '?') {
  TString character = text[i];
  wildcardMatches->AddLast(new TObjString(character.Data()));
  }
  i++;
  }
  if (i > 0) {
  return matchesExperimental(text(i, text.Length()),
  pattern(i, pattern.Length()), wildcardMatches);
  } else if (pattern[0] == '*') {
  TList * subWildcardMatches = NULL;
  if (wildcardMatches) {
  subWildcardMatches = new TList();
  subWildcardMatches->SetOwner(true);
  }
  bool eatWildcard = matchesExperimental(text, pattern(1, pattern.Length()), subWildcardMatches);
  bool keepWildcard = false;
  if (!eatWildcard) {
  if (subWildcardMatches) {
  subWildcardMatches->Delete();
  }
  keepWildcard = matchesExperimental(text(1, text.Length()), pattern, subWildcardMatches);
  }
  if (eatWildcard || keepWildcard) {
  if (wildcardMatches) {
  wildcardMatches->AddAll(subWildcardMatches);
  }
  return true;
  } else {
  return false;
  }
  } else {
  return false;
  }
  } else {
  return (text.Length() == 0 && (pattern.CompareTo('*') == 0 || pattern.Length() == 0));
  }
  }
*/

//__________________________________________________________________________________|___________

bool TQStringUtils::hasWildcards(TString text) {
  // Returns true if the input string <text> contains wildcards "?" or "*" and
  // false otherwise.
 
  return text.Contains("*") || text.Contains("?");
}


//__________________________________________________________________________________|___________

int TQStringUtils::compare(const TString& a, const TString& b) {
  // Compares the two input strings <a> and <b> and returns 0 in case both strings
  // are identical. In case string <a> (<b>) is longer or "greater" than string <b>
  // (<a>) 1 (-1) is returned. A string is considered greater than another string
  // if the character code at the left most position with non-matching characters
  // is larger.
  //
  // Examples:
  // - compare("a", "a") returns 0
  // - compare("a", "b") returns -1
  // - compare("ab", "a") returns 1
  // - compare("a", "A") returns 1
 
  // the current character index
  int i = 0;

  // iterate over characters until a mismatch or the end of one string is found
  while (i < a.Length() && i < b.Length()) {
    if (a[i] > b[i]) {
      // string <a> is "greater than" string <b>
      return 1;
    } else if (a[i] < b[i]) {
      // string <b> is "greater than" string <a>
      return -1;
    }
    // move to next character
    i++;
  }

  // one string shorter than the other: test remaining string
  if (a.Length() > b.Length()) {
    // string <a> longer than string <b>
    return 1;
  } else if (a.Length() < b.Length()) {
    // string <b> longer than string <a>
    return -1;
  } else {
    // strings are equal
    return 0;
  }
}


//__________________________________________________________________________________|___________

int TQStringUtils::compareHeads(const TString& str1, const TString& str2) {
  // Compares the heads of the two input strings <str1> and <str2> and returns the
  // length of the longest common string sequence both strings begin with.
 
  // scan strings starting at the strings' heads
  int pos = 0;
  while (str1.Length() > pos && str2.Length() > pos && str1[pos] == str2[pos]) {
    // move current position one character to the right
    pos++;
  }

  return pos;
}


//__________________________________________________________________________________|___________

int TQStringUtils::compareTails(const TString& str1, const TString& str2) {
  // Compares the tails of the two input strings <str1> and <str2> and returns the
  // length of the longest common string sequence both strings end with.

  // scan strings starting at the strings' tails
  int pos = 0;
  while (str1.Length() > pos && str2.Length() > pos
         && str1[str1.Length() - pos - 1] == str2[str2.Length() - pos - 1]) {
    // move current position one character to the left
    pos++;
  }

  return pos;
}


//__________________________________________________________________________________|___________

TString TQStringUtils::getMaxCommonHead(TList * strings) {
  // Returns the longest common sequence of all strings (names of objects obtained
  // using GetName()) present in input list <strings>.
 
  if (!strings || strings->GetEntries() == 0) {
    // invalid input
    return "";
  }
 
  TString str;
  int min = -1;

  // iterate over strings in input list
  TQIterator itr(strings);
  while (itr.hasNext()) {
    TString name = itr.readNext()->GetName();
    if (min == -1) {
      // first string in list
      str = name;
      min = str.Length();
    } else {
      min = TMath::Min(min, compareHeads(str, name));
    }
  }

  // remove everything except maximal common sequence
  str = str(0, min);
  return str;
}


//__________________________________________________________________________________|___________

bool TQStringUtils::isEscaped (const TString& text, int pos, const TString& escChar) {
  // Counts the number of occurences of <escChar> in front of <pos> in <text>.
  // Returns true if the value is odd and false if it is even.
  if (pos > text.Length()) return false;
  int other = findLastNotOf(text,escChar,pos-1);  
  return (pos-other+1)%2; 
}



//__________________________________________________________________________________|___________

TString TQStringUtils::removeEscapes(const TString& text, const TString& escapes) {
  // Returns a string similar to the input string <text> but removing escape
  // characters. The list of characters to escape is given by the input string
  // <escapes>.
 
  // the string to return
  TString output;

  int i = -1;
  while (++i < text.Length()) {
    if (escapes.Index(text[i]) == kNPOS) {
      // non escapes character: keep it
      output.Append(text[i]);
    } else if (i + 1 < text.Length() && escapes.Index(text[i + 1]) != kNPOS) {
      // escaped escape character: keep the second
      output.Append(text[++i]);
    }
  }

  return output;
}


//__________________________________________________________________________________|___________

TString TQStringUtils::insertEscapes(const TString& text, const TString& escapes) {
  // Returns a string similar to the input string <text> but inserting escape
  // characters where necessary. The list of characters to escape is given by the
  // input string <escapes> where the first character is used as the escape
  // character to be inserted in the text.
 
  // the string to return
  TString output;

  // iterate over every character in the input string
  int i = -1;
  while (++i < text.Length()) {
    // for escape characters an additional escape character is inserted
    if (escapes.Length() != 0 && escapes.Index(text[i]) != kNPOS) {
      output.Append(escapes[0]);
    }
    output.Append(text[i]);
  }

  return output;
}

//__________________________________________________________________________________|___________

int TQStringUtils::reduceToCommonPrefix(TString& prefix, const TString& other){
  // reduce the first of two strings to the their common prefix
  // return the length of this prefix
  size_t i=0; 
  size_t max = std::min(prefix.Length(),other.Length());
  while(i < max){
    if(prefix[i] == other[i]) i++;
    else break;
  }
  prefix.Remove(i);
  return i;
}

//__________________________________________________________________________________|___________

TString TQStringUtils::concat(TString first, const TString& second) {
  // concatenate two strings
  first.Append(second);
  return first;
}



//__________________________________________________________________________________|___________


//__________________________________________________________________________________|___________

TString TQStringUtils::concat(TCollection * items, const TString& sep, const TString& quote) {
  // Returns a string being the concatenation of strings in input list <items>
  // (names of objects obtained using GetName()) separated by string <sep>.
 
  bool first = true;
  TString text;

  // iterate over strings in list
  TQIterator itr(items);
  while (itr.hasNext()) {
    if (!first) {
      // insert separator string
      text.Append(sep);
    } else {
      first = false;
    }
    TObject* obj = itr.readNext();
    if(!obj) text.Append("NULL");
    else {
      if(quote.Length() == 2){
        text.Append(quote[0]);
      }
      text.Append(obj->GetName());
      if(quote.Length() == 2){
        text.Append(quote[1]);
      }
    }
  }

  return text;
}

//__________________________________________________________________________________|___________

TString TQStringUtils::getFirstToken(TString text, const TString& sep, bool trim, const TString& blocks, const TString& quotes) {
  // read and return first token in <text>
  TString token;
  TQStringUtils::readUpTo(text, token, sep, blocks, quotes);
  return token; 
}


//__________________________________________________________________________________|___________

TList * TQStringUtils::tokenize(
                                TString text, const TString& sep, bool trim, const TString& blocks, const TString& quotes) {
  // tokenize a string, return the result as a TList

  // the list to return
  TList * result = NULL;

  // split tokens
  TString token;
  bool stop = false;
  while (!stop) {
    // read up to next separator character
    int nRead = TQStringUtils::readUpTo(text, token, sep, blocks, quotes);
    // read separator character
    bool leadingSep = (TQStringUtils::removeLeading(text, sep, 1) > 0);
 
    if (nRead == 0 && !leadingSep) {
      // => thing has been read
      if (!text.IsNull() && result) {
        // invalid string input
        delete result;
        result = NULL;
      }
      stop = true;
      continue;
    } 
 
    if (!result) {
      // this is the first entry: create the list
      result = new TList();
      result->SetOwner(true);
    }
    if (trim) {
      // remove leading and trailing blanks
      token = TQStringUtils::trim(token);
    }
    result->Add(new TObjString(token.Data()));
    token.Clear();
 
    if (leadingSep && text.IsNull()) {
      result->Add(new TObjString());
    }
  }

  return result;
}


//__________________________________________________________________________________|___________

std::vector<TString> TQStringUtils::tokenizeVector(TString text, const TString& sep, bool trim, const TString& blocks, const TString& quotes) {
 
  // the list to return
  std::vector<TString> result;
 
  /* split tokens */
  TString token;
  while (TQStringUtils::readUpTo(text, token, sep, blocks, quotes) || !text.IsNull()) {//FIXME: This can easily lead to an infinite loop if readUpTo fails, e.g. due to an unexpected closing bracket/parenthesis!
    if (trim) {
      // remove leading and trailing blanks
      token = TQStringUtils::trim(token);
    }
    result.push_back(token);
    token.Clear();

    // remove separator and add another empty entry to the
    // list if the input text ends with a separator
    if (TQStringUtils::removeLeading(text, sep) && text.IsNull()) {
      result.push_back("");
    }
  }

  return result;
}


//__________________________________________________________________________________|___________

bool TQStringUtils::append(TString &text, const TString &appendix, const TString& sep) {
  // Appends input string <appendix> to string <text> adding the separator string
  // <sep> in case <text> is not an empty string before the operation.
 
  bool first = true;

  if (!text.IsNull()) {
    // add separator string
    text.Append(sep);
    first = false;
  }

  // add appendix
  text.Append(appendix);

  // return true if a separator string has been added and false otherwise
  return !first;
}


//__________________________________________________________________________________|___________

TString TQStringUtils::trim(const TString& text, const TString& blanks) {
  // Returns <text> without leading and trailing blanks.

  /* find end of leading blanks */
  int start = 0;
  while (start < text.Length() && blanks.Index(text[start]) != kNPOS)
    start++;

  /* find start of trailing blanks */
  int end = text.Length() - 1;
  while (end >= 0 && blanks.Index(text[end]) != kNPOS)
    end--;

  /* return the "core" of the string (ex-
   * cluding leading and trailing blanks) */
  return text(start, end - start + 1);
}


//__________________________________________________________________________________|___________

TString TQStringUtils::repeat(const TString& text, int n, const TString& sep) {
  // Returns the concatenation of <n> times <text> separated by <sep>.
 
  // the string to return
  TString result;
 
  // concatenate the text n times
  for (int i = 0; i < n; i++) {
    result.Append(text);
    if (i < n - 1 && !sep.IsNull()) {
      // add the separator string between elements
      result.Append(sep);
    }
  }

  // return the result
  return result;
}


//__________________________________________________________________________________|___________

TString TQStringUtils::repeatSpaces(int n) {
  // Returns a string with <n> spaces.
 
  return repeat(" ", n);
}


//__________________________________________________________________________________|___________

TString TQStringUtils::repeatTabs(int n) {
  // Returns a string with <n> tabs.
 
  return repeat("\t", n);
}


//__________________________________________________________________________________|___________

int TQStringUtils::getWidth(const TString& text) {
  // retrieve the with of some string in characters
  // NOTE: this width is not identical to the the string length, 
  // since some (unicode) characters and control sequences
  // require different amounts of characters in string length and on screen
  // NOTE: this function is still under developement 
  // and will not work for all symbols

  /* return characters (terminating control sequences) */
  TString returnChars = "mAC";

  /* the width of the text */
  int width = 0;

  /* scan the text and count characters excluding control sequences */
  bool escape = false;
  for (int pos = 0; pos < text.Length(); pos++) {
    if (escape) {
      if (returnChars.Index(text[pos]) != kNPOS)
        escape = false;
    } else {
      if (text[pos] == '\033')
        escape = true;
      else if(text[pos] > 0 || text[pos] == -61 || text[pos] == -62 || text[pos] == -30 || text[pos] == -50 || text[pos] == -54 || text[pos] == -49 || text[pos] == -31 || text[pos] == -53){
        width++;
      } 
    }
  }
 
  /* return the width */
  return width;
}

//__________________________________________________________________________________|___________

int TQStringUtils::getCharIndex(const TString& text, int index) {
  // return the index of a given character in a string
  // this takes into account control sequences and unicode characters
  // please see TQStringUtils::getWidth for details

  /* return characters (terminating control sequences) */
  TString returnChars = "mAC";
 
  /* the width of the text */
  int width = 0;
 
  /* scan the text and count characters excluding control sequences */
  bool escape = false;
  for (int pos = 0; pos < text.Length(); pos++) {
    if(width == index) return pos;
    if (escape) {
      if (returnChars.Index(text[pos]) != kNPOS)
        escape = false;
    } else {
      if (text[pos] == '\033')
        escape = true;
      else if(text[pos] > 0 || text[pos] == -61 || text[pos] == -62 || text[pos] == -30 || text[pos] == -50 || text[pos] == -54 || text[pos] == -49 || text[pos] == -31 || text[pos] == -53){
        width++;
      } 
    }
  }
 
  /* return the width */
  return kNPOS;
}


//__________________________________________________________________________________|___________

TString TQStringUtils::fixedWidth(const TString& text, int width, const char* options) {
  // expand or shrink the given string to a given width, returning the result
  // alignment may be specified as "r", "c" or "l" as the last argument
  TString opts(options);
  return TQStringUtils::fixedWidth(text,width,opts);
}

//__________________________________________________________________________________|___________

TString TQStringUtils::fixedWidth(const char* text, int width, const char* options){
  // expand or shrink the given string to a given width, returning the result
  // alignment may be specified as "r", "c" or "l" as the last argument
  TString opts(options);
  TString newText(text);
  return TQStringUtils::fixedWidth(newText,width,opts);
}

//__________________________________________________________________________________|___________

TString TQStringUtils::fixedWidth(const char* text, int width, const TString& options){
  // expand or shrink the given string to a given width, returning the result
  // alignment may be specified as "r", "c" or "l" as the last argument
  TString newText(text);
  return TQStringUtils::fixedWidth(newText,width,options);
}

//__________________________________________________________________________________|___________


bool TQStringUtils::readCharacter(const TString& text, int& index, int& count){
  /* scan the text and count characters excluding control sequences */
  if (text[index] == '\033'){
    while(index < text.Length()){
      index++;
      if(TQStringUtils::controlReturnCharacters.Index(text[index]) != kNPOS){
        index++;
        return true;
      }
    }
    return false;
  }
  index++;
  count++;
  while(index < text.Length()){
    if(text[index] > 0 
       || ((-61 >= text[index]) && (text[index] >= -62)) 
       || ((-49 >= text[index]) && (text[index] >= -54)) 
       || ((-30 >= text[index]) && (text[index] >= -32)) ){
      return true;
    }
    index++;
  }
  return true;
}

//__________________________________________________________________________________|___________

TString TQStringUtils::fixedWidth(const TString& text, int width, const TString& options) {
  // expand or shrink the given string to a given width, returning the result
  // alignment may be specified as "r", "c" or "l" as the last argument
  int lastidx = TQStringUtils::getCharIndex(text,width+1);
  if((lastidx < 1) || (lastidx >= text.Length())) lastidx = TQStringUtils::getCharIndex(text,width);
  else lastidx--;
 
  /* cut text if it is too long, TODO: handle control characters correctly */
  if ((lastidx >= width) && (lastidx < text.Length())) {
    std::stringstream s;
    bool showDots = options.Contains(".") && (width > 6);
    int reducedWidth = showDots ? (width-3) : width;
    int len = 0;
    int index = 0;
    int lastIndex = 0;
    int lastLen = 0;
    while(index < text.Length()){
      if(!TQStringUtils::readCharacter(text,index,len)) break;;
      if((len == lastLen) || (len < reducedWidth+1)) s << text(lastIndex,index-lastIndex);
      // std::cout << "len = " << len << ", index=" << index << ", lastLen= " << lastLen << ", lastIndex = " << lastIndex << ", substr='" << text(lastIndex,index-lastIndex) << "'" << std::endl;
      lastLen = len;
      lastIndex = index;
    }
    TString retval(s.str().c_str());
    if(showDots) retval.Append("...");
    return retval;
  } else {
    int actualWidth = TQStringUtils::getWidth(text);
    int nSpaces = std::max(width-actualWidth,0);
 
    if(options.Contains("r")){
      return TString::Format("%*s%s", nSpaces, "", text.Data());
    } else if(options.Contains("c")){
      return TString::Format("%*s%s%*s", (int)(0.5*(nSpaces+1)), "", text.Data(), (int)(0.5*(nSpaces)), "");
    } else {
      return TString::Format("%s%*s", text.Data(), nSpaces, "");
    }
  }
}

//__________________________________________________________________________________|___________

TString TQStringUtils::fixedWidth(const TString& text, int width, bool rightJustified) {
  // expand or shrink the given string to a given width, returning the result
  // returns right aligned text if rightJustified is true, otherwise left aligned
  return TQStringUtils::fixedWidth(text,width,rightJustified ? "r" : "l");
}

//__________________________________________________________________________________|___________

TString TQStringUtils::fixedWidth(const char* text, int width, bool rightJustified) {
  // expand or shrink the given string to a given width, returning the result
  // returns right aligned text if rightJustified is true, otherwise left aligned
  TString newText(text);
  return TQStringUtils::fixedWidth(newText,width,rightJustified ? "r" : "l");
}

//__________________________________________________________________________________|___________

TString TQStringUtils::maxLength(const TString& text, int maxLength, const TString& appendix) {
  // trim a string to a maximum length, including the appendix if shortened
  if (text.Length() > maxLength) {
    TString str = text(0, maxLength - appendix.Length());
    str.Append(appendix);
    return str;
  } else {
    return text;
  }
}


//__________________________________________________________________________________|___________

TString TQStringUtils::getThousandsSeparators(int value, const TString& sep) {
  /* convert integer to string */
  return TQStringUtils::getThousandsSeparators(TString::Format("%d", value), sep);
}

//__________________________________________________________________________________|___________

TString TQStringUtils::getThousandsSeparators(TString text, const TString& sep) {
  /* compile result including thousands separators */
  TString result;
  while (text.Length() > 3) {
    result.Prepend(text(text.Length() - 3, 3));
    result.Prepend(sep);
    text.Remove((size_t)(text.Length() - 3), (size_t)3);
  }

  /* prepend remaining text */
  result.Prepend(text);

  /* return the result */
  return result;
}


//__________________________________________________________________________________|___________

TString TQStringUtils::getThousandsSeparators(Long64_t value, const TString& sep) {
  /* convert long to string */
  return TQStringUtils::getThousandsSeparators(TString::LLtoa(value,10), sep);
}


//__________________________________________________________________________________|___________

TString TQStringUtils::getIndentationLines(int indent) {
  // creates indentation lines ("||||-") for tree-style console printing

  /* the indentation string */
  TString lines;

  /* deepest indentation */
  if (indent > 0)
    lines = "|-";

  /* other identation levels */
  lines.Prepend(repeat("| ", indent - 1));

  /* return the string */
  return lines;
}


//__________________________________________________________________________________|___________

TString TQStringUtils::formatSignificantDigits(double val, int nDigits) {
  // format a number to a string with the given number of significant digits

  TString retval;
  if (nDigits < 0) {
    retval = TString::Format("%.*f", -nDigits, val);
  } else if (nDigits > 0) {
    /* keep track of sign */
    double sign = 1.;
    if (val < 0.) {
      val = TMath::Abs(val);
      sign = -1.;
    }
 
    /* number of digits before decimal separator */
    int n = TMath::FloorNint(TMath::Log10(val)) + 1;

    /* if the n = 1 set n = 0*/
    if (n == 1)
      n = 0;
 
    /* cut off the number of digits after decimal separator */
    retval =  TString::Format("%.*f", TMath::Max(nDigits - n, 0), sign * val);

  } else {
    return retval = "";
  }
  if(retval.Contains(".")){
    if(TQStringUtils::removeTrailing(retval,"0") > 0){
      TQStringUtils::removeTrailing(retval,".");
    }
  }
  return retval;
}

//__________________________________________________________________________________|___________

TString TQStringUtils::formatValueErrorPDG(double val, double err, int expval, const TString& format){
  // implementing the PDG rounding guidelines
  //   http://pdg.lbl.gov/2010/reviews/rpp2010-rev-rpp-intro.pdf

  if(err == 0){
    return TString::Format("%g",val);
  }
  
  int exponent = 0;
  double val_shifted = val;
  double err_shifted = err;
  while(err_shifted <= 100){
    val_shifted *= 10;
    err_shifted *= 10;
    exponent -=1;
  }
  while(err_shifted > 1000){
    val_shifted /= 10;
    err_shifted /= 10;
    exponent +=1;
  }
  int firstdigits(err_shifted);
  int ndigits = 0;
  double int_val,int_err;
  
  if(firstdigits <= 354){
    int_val = floor(0.5+0.1*val_shifted);
    int_err = floor(0.5+0.1*err_shifted);
    ndigits = 1;
    exponent += 1;
  } else if(firstdigits <= 949){
    int_val = floor(0.5+0.01*val_shifted);
    int_err = floor(0.5+0.01*err_shifted);
    ndigits = 2;
    exponent += 2;
  } else {
    int_val = floor(0.5+0.01*val_shifted);
    int_err = 1;
    ndigits = 1;
    exponent += 2;
  }
  
  double shift = pow(10,exponent-expval);
  ndigits = std::max(0,ndigits+expval-exponent-1);

  //  std::cout << int_val << " +/- " << int_err << " * 10e" << exponent << " " << ndigits << std::endl;
  
  double val_final = shift*int_val;
  double err_final = shift*int_err;
  
  std::stringstream ss;
  if(format == "latex") ss << "\\ensuremath{";
  if(expval != 0) ss << "(";
  ss << "%." << ndigits << "f";
  if(format == "latex") ss << " \\pm ";
  else if(format == "html") ss << "&pm;";
  else ss << " +/- ";
  ss << "%." << ndigits << "f";
  if(expval != 0){
    ss << ")";
    if(format == "latex") ss << " \\times 10^{" << expval << "}";
    else if(format == "html") ss << "&times; <superscript>" << expval << "</superscript>";
    else ss << "*10^" << expval;
  }
  if(format == "latex") ss << "}";
  return TString::Format(ss.str().c_str(),val_final,err_final);
}


//__________________________________________________________________________________|___________

TString TQStringUtils::formatValueError(double val, double err, const TString& format){
  // format a number and uncertainty to a string in the given format
  if(!TQUtils::isNum(err) || !TQUtils::isNum(val)){
    return TString::Format(format,val,err);
  }
  int nSigDigitsErr = std::min(ceil(-log10(err)),0.);
  int firstdigit = floor(err * pow(10,ceil(-log10(err))));
  if(firstdigit == 1){
    nSigDigitsErr++;
  }
  double nDiff = ceil(log10(val/err));
  double valRounded = TQUtils::roundAuto(val,nDiff+nSigDigitsErr);
  double errRounded = TQUtils::roundAuto(err,nSigDigitsErr);
  return TString::Format(format,valRounded,errRounded);
}


//__________________________________________________________________________________|___________

bool TQStringUtils::printDiffOfLists(TList * l1, TList * l2, bool ignoreMatches) {
  // Performs a 'diff' of the two input lists (instances of TList) <l1> and <l2>
  // based on the result of the Compare(...) methods on contained objects, prints
  // the result on std::cout, and returns true if both lists fully match or false
  // otherwise. The print-out is a list of all elements present in either of the two
  // lists with matching elements (elements with the same name and present in both
  // lists) being printed side-by-side. If <ignoreMatches> == true (default is false)
  // only elements present in one list are shown.

  // expect valid input lists
  if (!l1 || !l2) {
    return false;
  }

  // the number of entries in each list
  int n1 = 0;
  int n2 = 0;

  // sort the list
  if (l1) {
    n1 = l1->GetEntries();
    l1->Sort();
  }
  if (l2) {
    n2 = l2->GetEntries();
    l2->Sort();
  }

  // determine the maximum string length to print a pretty list
  int max1 = 0;
  int max2 = 0;
  TQIterator itr1(l1);
  while (itr1.hasNext()) {
    max1 = TMath::Max(max1, TString(itr1.readNext()->GetName()).Length());
  }
  TQIterator itr2(l2);
  while (itr2.hasNext()) {
    max2 = TMath::Max(max2, TString(itr2.readNext()->GetName()).Length());
  }

  // do both lists fully match?
  bool match = true;

  // print the headline
  TString headline = TString::Format("%-*s %-*s", max1, "List 1", max2, "List 2");
  std::cout << TQStringUtils::makeBoldWhite(headline).Data() << std::endl;
  std::cout << TQStringUtils::makeBoldWhite(
                                       TQStringUtils::repeat("=", headline.Length())).Data() << std::endl;

  // list indices
  int i1 = 0;
  int i2 = 0;

  // process the two lists by incrementing list indices in a synchronized way
  while (i1 < n1 || i2 < n2) {

    // get the two objects at the current list indices
    // (use NULL pointer in case a list has reached its end)
    TObject * obj1 = (i1 < n1) ? l1->At(i1) : NULL;
    TObject * obj2 = (i2 < n2) ? l2->At(i2) : NULL;

    // compare the two current objects
    int compare = -1;
    if (obj1 && obj2) {
      // Compare objects based on their Compare(...) method (usually compares names)
      compare = obj1->Compare(obj2);
      // go to next element in list 1 if list 2 is ahead or both are at same level
      i1 += (compare <= 0) ? 1 : 0;
      // go to next element in list 2 if list 1 is ahead or both are at same level
      i2 += (compare >= 0) ? 1 : 0;
    } else if (obj1) {
      // list 2 has reached its end
      compare = -1;
      i1++;
    } else if (obj2) {
      // list 1 has reached its end
      compare = 1;
      i2++;
    } else {
      // something went totally wrong since both objects are invalid
      return false;
    }

    if (compare < 0) {
      // element in list 1 not present in list 2
      match = false;
      std::cout << TString::Format("%-*s %-*s",
                              max1, obj1->GetName(), max2, "--").Data() << std::endl;
    } else if (compare > 0) {
      // element in list 2 not present in list 1
      match = false;
      std::cout << TString::Format("%-*s %-*s",
                              max1, "--", max2, obj2->GetName()).Data() << std::endl;
    } else if (!ignoreMatches) {
      // matching elements
      std::cout << TString::Format("%-*s %-*s",
                              max1, obj1->GetName(), max2, obj2->GetName()).Data() << std::endl;
    }
  }

  // return true if no mismatch was found and false otherwise
  return match;
}


//__________________________________________________________________________________|___________

int TQStringUtils::removeLeading(TString &text, TString characters, int nMax) {
  // Removes from the head of string <text> all characters listed in <characters>
  // (but not more than <nMax>) and returns the number of characters that have been
  // removed.

  // determine number of leading characters at head of <text> listed in <characters>
  int pos = countLeading(text, characters, nMax);
  if(pos < 1) return 0;

  // remove these characters
  text.Remove(0, pos);

  // return the number of characters removed
  return pos;
}


//__________________________________________________________________________________|___________

int TQStringUtils::removeTrailing(TString &text, TString characters, int nMax) {
  // Removes from the tail of string <text> all characters listed in <characters>
  // (but not more than <nMax>) and returns the number of characters that have been
  // removed.

  // determine number of leading characters at head of <text> listed in <characters>
  int nChar = countTrailing(text, characters, nMax);
  if(nChar < 1) return 0;

  // remove trailing characters
  text.Remove(text.Length() - nChar, text.Length());

  // return the number of characters removed
  return nChar;
}


//__________________________________________________________________________________|___________

bool TQStringUtils::removeLeadingText(TString &text, TString prefix) {
  // Removes one occurence of string sequence <prefix> from head of string <text>
  // if present and returns true in this case and false otherwise.
 
  if (text.BeginsWith(prefix)) {
    // remove sequence from head of string
    text.Remove(0, prefix.Length());
    return true;
  } else {
    // sequence not present at head of string
    return false;
  }
}


//__________________________________________________________________________________|___________

bool TQStringUtils::removeTrailingText(TString &text, TString appendix) {
  // Removes one occurence of string sequence <appendix> from tail of string <text>
  // if present and returns true in this case and false otherwise.

  if (text.EndsWith(appendix)) {
    // remove sequence from tail of string
    text.Remove(text.Length() - appendix.Length(), appendix.Length());
    return true;
  } else {
    // sequence not present at tail of string
    return false;
  }
}

//__________________________________________________________________________________|___________

int TQStringUtils::removeAll(TString &text, const TString& chars, TString::ECaseCompare comp, int max) {
  // Removes up to 'max' occurences of characters contained in chars from text
  // starting at the beginning of text. 'max' is ignored if it is negative.
  int pos = 0;
  int nMatches = 0;
  while (pos<text.Length() && (max<0 || nMatches<max) ) {
    if ( chars.Contains(text(pos), comp) ) { //the character in question is contained in the list of characters to be removed
      text.Remove(pos,1); //do not increment position as all subsequent characters are shifted due to the removal!
      ++nMatches;
    } else {
      ++pos;
    }
    
  }
  return nMatches;
}

//__________________________________________________________________________________|___________

int TQStringUtils::ensureTrailingText(TString &text, const TString& appendix) {
  // ensure that a string ends with a given character sequence
  int tmax = text.Length() -1;
  int amax = appendix.Length() -1;
  int pos = 0;
  while(pos <= amax && pos <= tmax){
    if(appendix[amax-pos] == text[tmax-pos])
      pos++;
    else {
      amax--;
      pos = 0;
    }
  }
  text.Append(appendix(pos,appendix.Length()));
  return appendix.Length()-pos;
}

//__________________________________________________________________________________|___________

int TQStringUtils::ensureLeadingText(TString &text, const TString& prefix) {
  // ensure that a string starts with a given character sequence
  int offset = 0;
  if(prefix.Length() > text.Length()) offset = prefix.Length() - text.Length();
  int pos = prefix.Length() -1;
  while(pos-offset >= 0 && offset < prefix.Length()){
    if(text[pos-offset] == prefix[pos]){
      pos--;
    } else {
      offset++;
      pos=prefix.Length() -1 ;
    }
  }
  text.Prepend(prefix(0,offset));
  return offset;
}


//__________________________________________________________________________________|___________

int TQStringUtils::removeLeadingBlanks(TString &text, int nMax) {
  // Removes leading blanks from head of string <text> (but not more than <nMax>)
  // and returns the number of blanks that have been removed. Characters are
  // recognized as blanks if listed in TQStringUtils::getBlanks().
 
  return TQStringUtils::removeLeading(text, TQStringUtils::getBlanks(), nMax);
}


//__________________________________________________________________________________|___________

int TQStringUtils::removeTrailingBlanks(TString &text, int nMax) {
  // Removes trailing blanks from tail of string <text> (but not more than <nMax>)
  // and returns the number of blanks that have been removed. Characters are
  // recognized as blanks if listed in TQStringUtils::getBlanks().

  return TQStringUtils::removeTrailing(text, TQStringUtils::getBlanks(), nMax);
}


//__________________________________________________________________________________|___________

int TQStringUtils::countLeading(const TString& text, const TString& characters, int nMax) {
  // Count and return the number of characters listed in <characters> at head of
  // string <text>.
 
  // scan head of string <text> for all occurences of characters listed in
  // <characters> and find position of first character not listed in <characters>
  int pos = 0;
  while (pos < text.Length() && characters.Index(text[pos]) != kNPOS
         && (nMax < 0 || pos < nMax)) {
    pos++;
  }

  return pos;
}

//__________________________________________________________________________________|___________

int TQStringUtils::countTrailing(const TString& text, const TString& characters, int nMax) {
  // Count and return the number of characters listed in <characters> at tail of
  // string <text>.

  // scan tail of string <text> for all occurences of characters listed in
  // <characters> and find position of last character not listed in <characters>
  int pos = text.Length() - 1;
  while (pos >= 0 && characters.Index(text[pos]) != kNPOS
         && (nMax < 0 || pos >= (text.Length() - nMax))) {
    pos--;
  }

  return text.Length() - 1 - pos;
}


//__________________________________________________________________________________|___________

int TQStringUtils::countText(const TString& haystack, const TString& needle) {
  // count the occurences of needle in haystack
  if(needle.Length() > haystack.Length() || needle.IsNull() || haystack.IsNull()) return 0;
  int num = 0;
  int hpos = 0;
  int npos = 0;
  while (hpos < haystack.Length()){
    if(needle[npos] == haystack[hpos]){
      if(npos+1 == needle.Length()){
        npos = 0;
        num++;
      } else {
        npos++;
      }
    } else {
      npos = 0; //reset needle position if the current character did not match
    }
    hpos++;
  }

  return num;
}

//__________________________________________________________________________________|___________

TString TQStringUtils::cutUnit(TString &label) {
  // remove the unit "[...]" from some label
  // returns the unit

  /* use a std string to extract position of unit */
  std::string input = label.Data();
  std::size_t start = input.find_last_of("[");
  std::size_t end = input.find_last_of("]");
 
  if (start != std::string::npos && end != std::string::npos && end > start) {

    /* extract and remove the unit and it's appendix */
    TString unit = label(start + 1, end - start - 1);
    TString appendix = label(end + 1, label.Length());
    label.Remove(start, label.Length());

    /* compile label without unit */
    TQStringUtils::removeTrailingBlanks(label);
    label.Append(appendix);

    /* return the unit */
    return unit;
  } else {
    /* couldn't find a unit */
    return "";
  }
}


//__________________________________________________________________________________|___________

TString TQStringUtils::getUnit(TString label) {
  // retrieve the unit "[...]" from some label
  return TQStringUtils::cutUnit(label);
}


//__________________________________________________________________________________|___________

TString TQStringUtils::getWithoutUnit(TString label) {
  // remove the unit "[...]" from some label
  // returns the result
  TQStringUtils::cutUnit(label);
  return label;
}


//__________________________________________________________________________________|___________

int TQStringUtils::readBlanks(TString & in) {
  // removes leading blanks from a string
  TString dummy;

  return readToken(in, dummy, getBlanks());
}


//__________________________________________________________________________________|___________

int TQStringUtils::readBlanksAndNewlines(TString & in) {
  // removes leading blanks and newlines from a string
  int dummy;
  return readBlanksAndNewlines(in, dummy);
}


//__________________________________________________________________________________|___________

int TQStringUtils::readBlanksAndNewlines(TString & in, int &nNewlines) {
  // removes leading blanks from a string, counting removed newlines
  TString dummy;
  int n = 0;
  bool done = false;
  while (!done) {
    dummy.Clear();
    n += readToken(in, dummy, getBlanks());
    int n2 = readToken(in, dummy, "\n");
    nNewlines += n2;
    n += n2;
    done = dummy.IsNull();
  }
  return n;
}


//__________________________________________________________________________________|___________

int TQStringUtils::readToken(TString & in, TString & out, const TString& characters, int nMax) {
  // Counts the number of leading characters at head of string <in> listed in
  // <characters>, moves the corresponding sequence from head of string <in> to
  // tail of string <out>, and returns the number of characters that have been
  // moved. If <nMax> is non-negative at most <nMax> characters are moved. 
 
  // determine number of leading characters at head of <in> listed in <characters>
  int pos = countLeading(in, characters, nMax);

  // move characters from input to output string
  out.Append(in(0, pos));
  in.Remove(0, pos);

  // return the number of characters read
  return pos;
}


//__________________________________________________________________________________|___________

int TQStringUtils::readTokenAndBlock(TString & in,
                                     TString & out, const TString& characters, const TString& blocks) {
  // read a token consisting of the given characters and a subsequent block

  // the number of characters read from string <in>
  int n = 0;

  bool stop = false;
  while (!stop) {
    // read token
    int nToken = readToken(in, out, characters);
    // read block
    int nBlock = readBlock(in, out, blocks, "", true);
    // keep track of the number of characters read
    n += nToken + nBlock;
    // stop if nothing to read is left
    stop = (nToken == 0 && nBlock == 0);
  }

  // return the number of characters read
  return n;
}


//__________________________________________________________________________________|___________

int TQStringUtils::readBlock(
                             TString & in,
                             TString & out,
                             const TString& blocks,
                             const TString& quotes,
                             bool keepEnclosed,
                             int ignoreUnexpectedClosingQuotes) {
  // read a parenthesis enclosed block from a string

  /* stop if no valid block or quote definition was given */
  if (blocks.Length() < 2 || (blocks.Length() % 2) != 0 || (quotes.Length() % 2) != 0)
    return 0;

  /* stop if input string is empty */
  if (in.Length() == 0)
    return 0;

  /* the stacks keeping track of inner blocks and qutoes */
  std::vector<int> * blockStack = new std::vector<int>();
  std::vector<int> * quoteStack = new std::vector<int>();

  /* the number of defined inner block types and quote types */
  int nSubBlocks = blocks.Length() / 2;
  int nQuoteTypes = quotes.Length() / 2;

  if (ignoreUnexpectedClosingQuotes < 0) {
    ignoreUnexpectedClosingQuotes = nQuoteTypes;
  }

  /* true if an error occured (closing
   * inner block with wasn't opened) */
  bool error = false;
  bool found = false;

  /* the position in <in> string */
  int pos = 0;

  do {

    /* >> closing quote */
    if (quoteStack->size() > 0) {
      if (in[pos] == quotes[quoteStack->back()]) {
        quoteStack->pop_back();
      }
      /* go to next character */
      continue;
    }
 
    /* >> opening quote? */
    found = false;
    for (int i = 0; i < (pos > 0 ? nQuoteTypes : 0) && !found && !error; i++) {
      /* block start character? */
      if (in[pos] == quotes[2*i]) {
        quoteStack->push_back(2*i + 1);
        /* go to next character */
        found = true;
        /* block end character? */
      } else if (i < ignoreUnexpectedClosingQuotes && in[pos] == quotes[2*i + 1]) {
        error = true;
      }
    }
    if (found || error)
      continue;

    /* >> closing inner block? */
    if (blockStack->size() > 0 && in[pos] == blocks[blockStack->back()]) {
      blockStack->pop_back();
      /* go to next character */
      continue;
    }

    /* >> opening inner block? */
    found = false;
    for (int i = 0; i < (pos > 0 ? nSubBlocks : 1) && !found && !error; i++) {
      /* block start character? */
      if (in[pos] == blocks[2*i]) {
        blockStack->push_back(2*i + 1);
        /* go to next character */
        found = true;
        /* block end character? */
      } else if (in[pos] == blocks[2*i + 1]) {
        error = true;
      }
    }

  } while (blockStack->size() > 0 && ++pos < in.Length() && !error);
 
  if (blockStack->size() > 0 || quoteStack->size() > 0 || error)
    pos = 0;

  /* delete the stacks */
  delete blockStack;
  delete quoteStack;

  if (pos > 0) {
    /* move the block contents from in
     * to out (excluding the brackets) */
    out.Append(in(keepEnclosed ? 0 : 1, pos - (keepEnclosed ? -1 : 1)));
    in.Remove(0, pos + 1);
    /* return the number of characters read */
    return pos + 1;
  } else {
    return 0;
  }
}


//__________________________________________________________________________________|___________

int TQStringUtils::readUpToText(TString & in, TString & out,
                                const TString& upTo, const TString& blocks, const TString& quotes, int ignoreUnexpectedClosingQuotes) {
  // read a string up to occurence of some other string

  if (upTo.IsNull()) {
    return readUpTo(in, out, upTo, blocks, quotes, ignoreUnexpectedClosingQuotes);
  }
 
  int N = 0;
  bool stop = false;
  while (!stop && !in.BeginsWith(upTo)) {
    int n = readToken(in, out, upTo[0], 1);
    n += readUpTo(in, out, upTo[0], blocks, quotes, ignoreUnexpectedClosingQuotes);
    N += n;
    stop = (n == 0);
  }
 
  return N;
}


//__________________________________________________________________________________|___________

int TQStringUtils::readUpTo(TString & in, TString & out,
                            const TString& upTo, const TString& blocks, const TString& quotes, int ignoreUnexpectedClosingQuotes) {
  // read a string up to occurence of the first of a set of characters

  /* stop if no valid block or quote definition was given */
  if ((blocks.Length() % 2) != 0 || (quotes.Length() % 2) != 0)
    return 0;

  /* stop if input string is empty */
  if (in.Length() == 0)
    return 0;

  /* the stacks keeping track of inner blocks and qutoes */
  std::vector<int> * blockStack = new std::vector<int>();
  std::vector<int> * quoteStack = new std::vector<int>();

  /* the number of defined inner block types and quote types */
  int nSubBlocks = blocks.Length() / 2;
  int nQuoteTypes = quotes.Length() / 2;

  if (ignoreUnexpectedClosingQuotes < 0) {
    ignoreUnexpectedClosingQuotes = nQuoteTypes;
  }

  /* true if an error occured (closing
   * inner block with wasn't opened) */
  bool error = false;
  bool found = false;

  /* the position in <in> string */
  int pos = 0;

  do {

    /* >> closing quote */
    if (quoteStack->size() > 0) {
      if (in[pos] == quotes[quoteStack->back()]) {
        quoteStack->pop_back();
      }
      /* go to next character */
      continue;
    }

    /* >> opening quote? */
    found = false;
    for (int i = 0; i < nQuoteTypes && !found && !error; i++) {
      /* opening quote character? */
      if (in[pos] == quotes[2*i]) {
        quoteStack->push_back(2*i + 1);
        /* go to next character */
        found = true;
        /* unexpected closing quote character? */
      } else if (i < ignoreUnexpectedClosingQuotes && in[pos] == quotes[2*i + 1]) {
        error = true;
      }
    }
    if (found || error)
      continue;

    /* >> closing inner block? */
    if (blockStack->size() > 0 && in[pos] == blocks[blockStack->back()]) {
      blockStack->pop_back();
      /* go to next character */
      continue;
    }

    /* >> opening inner block? */
    found = false;
    for (int i = 0; i < nSubBlocks && !found && !error; i++) {
      /* block start character? */
      if (in[pos] == blocks[2*i]) {
        blockStack->push_back(2*i + 1);
        /* go to next character */
        found = true;
        /* block end character? */
      } else if (in[pos] == blocks[2*i + 1]) {
        error = true;
      }
    }

  } while (!(blockStack->size() == 0 && quoteStack->size() == 0
             && upTo.Index(in[pos]) != kNPOS) && ++pos < in.Length() && !error);
 
  if (blockStack->size() > 0 || quoteStack->size() > 0 || error)
    pos = 0;

  /* delete the stacks */
  delete blockStack;
  delete quoteStack;

  if (pos > 0) {
    /* move the block contents from in
     * to out (excluding the brackets) */
    out.Append(in(0, pos));
    in.Remove(0, pos);
    /* return the number of characters read */
    return pos;
  } else {
    return 0;
  }
}


//__________________________________________________________________________________|___________

TString TQStringUtils::expand(TString in, const TString& characters, const TString& blocks, bool embrace) {

  /* the final expansion */
  TString out;

  while (!in.IsNull()) {

    /* extract one expansion token */
    TString token;
    readTokenAndBlock(in, token, characters, blocks);

    /* read the expansion prefix */
    TString prefix;
    readToken(token, prefix, characters);

    /* read the first expansion block */
    TString block;
    readBlock(token, block, blocks, "");

    /* read expansion suffix */
    TString suffix;
    readTokenAndBlock(token, suffix, characters, blocks);

    if (embrace)
      out.Append("(");

    if (block.Length() > 0) {
      /* expand the (sub)tokens */
      TString expBlock = expand(block, characters, blocks, embrace);

      /* loop over (sub)tokens in the expansion block */
      while (!expBlock.IsNull()) {
 
        /* extract (sub)token */
        TString subToken;
        readTokenAndBlock(expBlock, subToken, characters, blocks);
 
        /* the (sub)expansion suffix */
        TString subSuffix;
        readUpTo(expBlock, subSuffix, characters + blocks);
 
        out.Append(prefix);
        out.Append(subToken);
        out.Append(suffix);
        out.Append(subSuffix);
      }
    } else {
      out.Append(prefix);
      out.Append(suffix);
    }

    if (embrace)
      out.Append(")");

    /* the expansion suffix */
    suffix.Clear();
    readUpTo(in, suffix, characters + blocks);
    out.Append(suffix);
  }

  return out;
}

//__________________________________________________________________________________|___________

TString TQStringUtils::replace(TString str, const TString& needle, const TString& newNeedle){
  str.ReplaceAll(needle,newNeedle);
  return str;
}

//__________________________________________________________________________________|___________

TString TQStringUtils::replace(TString str, const char* needle, const TString& newNeedle){
  str.ReplaceAll(needle,newNeedle);
  return str;
}

//__________________________________________________________________________________|___________

TString TQStringUtils::replace(TString str, const TString& needle, const char* newNeedle){
  str.ReplaceAll(needle,newNeedle);
  return str;
}

//__________________________________________________________________________________|___________

TString TQStringUtils::replace(TString str, const char* needle, const char* newNeedle){
  str.ReplaceAll(needle,newNeedle);
  return str;
}

//__________________________________________________________________________________|___________

TString TQStringUtils::readPrefix(TString &in, const TString& delim, const TString& defaultPrefix) {
  // read a string up to the occurence of some delimiter

  /* find the delimiter (TODO: use TQStringUtils::readUpTo(...) instead */
  Ssiz_t pos = in.Index(delim);

  if (pos == kNPOS) {
    return defaultPrefix;
  } else {
    TString prefix = in(0, pos);
    in.Remove(0, pos + delim.Length());
    return prefix;
  }
}

//__________________________________________________________________________________|___________
 
bool TQStringUtils::writeTextToFile(TList * text, const TString& filename) {
  // Writes the lines of strings in list <text> (names of objects obtained using
  // GetName()) to external file <filename> or to std::cout in case <filename> is
  // an empty string and return true in case of success and false otherwise.
 
  // stop if text is invalid
  if (!text) {
    return false;
  }

  // open file
  std::ofstream * file = NULL;
  if (!filename.IsNull()) {
    file = new std::ofstream(filename.Data());
    if (file->fail()) {
      // failed to open/create file
      delete file;
      return false;
    }
  }

  // iterate over lines in text and write to file or std output
  TQIterator itr(text);
  while (itr.hasNext()) {
    TString str = itr.readNext()->GetName();
    if (file) {
      // write to external file
      *file << str.Data() << std::endl;
    } else {
      // write to std::cout
      std::cout << str.Data() << std::endl;
    }
  }

  if (file) {
    // close file
    file->close();
    delete file;
  }
 
  // apparently no error occured
  return true;
}

//__________________________________________________________________________________|___________
TString TQStringUtils::readFile(const TString& filename, const TString& blacklist, const TString& replace){
  // Read a text file to a single string, ignoring all blacklisted characters
  // if replacements are given, characters with odd indices are replaced by their successor
  std::ifstream in(filename.Data(), std::ios::in | std::ios::binary);
  std::stringstream s;
  if(replace.Length() % 2 == 1) return "";
  char c;
  bool lastWasSpace = true;
  while(in.get(c) && in.good()){
    size_t irep = replace.Index(c);
    if(irep < (size_t)replace.Length() && (irep % 2 == 0)){
      c = replace[irep+1];
    } 
    if(blacklist.Contains(c) || (c==' ' && lastWasSpace)) continue;
    lastWasSpace = (c == ' ');
    s << c;
  }
  return TString(s.str().c_str());
}

//__________________________________________________________________________________|___________
TString TQStringUtils::readSVGtoDataURI(const TString& filename){
  // read an SVG file, returning a data URI string
  TString content = TQStringUtils::trim(TQStringUtils::readFile(filename, "", "\"'\n \t "));
  content.Prepend("data:image/svg+xml;utf8,");
  return content;
}

//__________________________________________________________________________________|___________
std::vector<TString>* TQStringUtils::readFileLines(const TString& filename, size_t len, bool allowComments){
  // Read a text file, line by line, and return the contents as a std::vector
  std::vector<TString>* lines = new std::vector<TString>();
  if(readFileLines(lines,filename,len,allowComments) > 0){
    return lines;
  }
  delete lines;
  return NULL;
}

//__________________________________________________________________________________|___________
size_t TQStringUtils::readFileLines(std::vector<TString>* lines, const TString& filename, size_t len, bool allowComments){
  // Read a text file, line by line, and push the contents into a std::vector
  std::ifstream infilestream(filename.Data(), std::ios::in);
  if(!infilestream.is_open())
    return 0;
  char* tmp = (char*)malloc(len*sizeof(char));
  size_t linecount = 0;
  while (!infilestream.eof()){
    infilestream.getline(tmp, len);
    TString str(tmp);
    while(!infilestream.eof() && (!infilestream.good() || TQStringUtils::removeTrailing(str,"\\") == 1)){
      infilestream.clear();
      infilestream.getline(tmp, len);
      str.Append(tmp);
    }
    DEBUGfunc("reading line '%s'",str.Data());
    TString resstr;
    if(allowComments){
      TQStringUtils::readUpTo(str,resstr,'#');
      while(resstr.EndsWith("\\")){
        TQStringUtils::removeTrailing(resstr,"\\");
        TQStringUtils::readToken(str,resstr,"#");
        TQStringUtils::readUpTo(str,resstr,'#');
      }
      resstr = TQStringUtils::trim(resstr,TQStringUtils::getAllBlanks());
    } else {
      resstr = TQStringUtils::trim(str,TQStringUtils::getAllBlanks());
    }
    if(resstr.Length() > 0){
      lines->push_back(resstr);
      linecount++;
    }
  }
  free(tmp);
  infilestream.close();
  return linecount;
}

//__________________________________________________________________________________|___________
 
TList* TQStringUtils::readDefinitionFile(const TString& filename){
  // read a histogram definition file, returning a TList of TObjStrings with histogram definitions
  std::ifstream file(filename.Data());
  TList* list = new TList();
  list->SetOwner(true);
  if (file.is_open()) {
    /* read line by line */
    std::string stdline;
 
    TString definition;
    while (getline(file, stdline)) {
      TString line = TQStringUtils::trim(TString(stdline.c_str()));
      if (!line.BeginsWith("#") && !line.IsNull()) {
        definition.Append(line);
        if (TQStringUtils::removeTrailing(definition, ";") > 0) {
          /* add cut definition */
          list->Add(new TObjString(definition));
          definition.Clear();
        }
      }
    }
    file.close();
 
    if (!definition.IsNull()) ERRORfunc("missing terminating ';' in line '%d'!",definition.Data());
  } else {
    ERRORfunc("failed to load cut definitions from file '%s'!",filename.Data());
  }
  return list;
}

//______________________________________________________________________________________________

TString TQStringUtils::unquote(TString text, const TString& quotes) {
  // remove enclosing quotes from a string, returning the result
  TQStringUtils::unquoteInPlace(text,quotes);
  return text;
}

//______________________________________________________________________________________________

TString TQStringUtils::quote (const TString& str, char q){
  // enclose a string in quotes, returning the result
  return q + str + q;
}

//______________________________________________________________________________________________

void TQStringUtils::unquoteInPlace(TString& text, const TString& quotes) {
  // remove enclosing quotes from a string, returning the result
  for(int i=0; i < quotes.Length(); ++i){
    const TString s(quotes(i,1));
    DEBUGfunc("stripping '%s'",s.Data());
    if(text.BeginsWith(s) && text.EndsWith(s)){
      text.Remove(0,1);
      text.Remove(text.Length() - 1, 1);
    }
  }
}

//______________________________________________________________________________________________

TString TQStringUtils::unblock(TString text, const TString& blocks) {
  // remove enclosing parenthesis from a string, returning the result
  for(int i=0; i < blocks.Length(); i+=2){
    if(text.BeginsWith(blocks(i,1)) && text.EndsWith(blocks(i+1,1).Data())){
      text.Remove(0, 1);
      text.Remove(text.Length() - 1, 1);
    }
  }
  return text;

}


//______________________________________________________________________________________________

TString TQStringUtils::minimize(TString text) {
  // compactify a string, removing all spaces
  text.ReplaceAll(" ","");
  text.ReplaceAll("\n","");
  text.ReplaceAll("\t","");
  text.ReplaceAll("\r","");
  return text;
}

//______________________________________________________________________________________________

TString TQStringUtils::compactify(const TString& text) {
  // compactify a string, removing all double whitespaces
  TString retval = "";
  bool isSpace = true;
  for(size_t i=0; i<(size_t)text.Length(); i++){
    if(text[i] == ' ' || text[i] == '\t' || text[i] == '\n' || text[i] == '\r'){
      if(isSpace) continue;
      else retval += ' ';
      isSpace=true;
    } else {
      retval += text[i];
      isSpace=false;
    }
  }
  TQStringUtils::removeLeadingBlanks(retval);
  TQStringUtils::removeTrailingBlanks(retval);
  return retval;

}

//______________________________________________________________________________________________

bool TQStringUtils::hasUnquotedStrings(const TString& text, const TString& quotes){
  // returns true if the string contains unquoted string literals, false otherwise
  bool quoted = false;
  for(size_t i=0; i<(size_t)text.Length(); i++){
    if(quotes.Contains(text[i])){
      quoted = !quoted;
      continue;
    }
    if(!quoted && TQStringUtils::letters.Contains(text[i]))
      return true;
  }
  return false;
}

//______________________________________________________________________________________________

bool TQStringUtils::hasTFormulaParameters(const TString& text){
  DEBUGfunc("checking text %s",text.Data());
  if(!text.Contains("[")) return false;
  if(!text.Contains("]")) return false;
  size_t pos = TQStringUtils::find(text,"[");
  size_t close = TQStringUtils::findParenthesisMatch(text,pos,"[","]");
  if (close<0 || close > (size_t)(text.Length())) return false;
  for (size_t i = pos+1; i<close; ++i){
    if (!numerals.Contains(text[i])){
      DEBUG("found that '%s' qualifies as having TFormula parameters",text.Data());
      return true;
    }
  }
  return false;
}

//______________________________________________________________________________________________

char TQStringUtils::hasStringSwitch(const TString& input){
  size_t pos = -1;
  while(true){
    pos = TQStringUtils::findFirstOf(input,"'\"",pos+1);
    if(pos >= 0 && pos < (size_t)(input.Length())){
      size_t nextpos = TQStringUtils::findParenthesisMatch(input,pos,input[pos],input[pos]);
      if(nextpos < 0 || nextpos > (size_t)(input.Length())){
        return input[pos];
      } else {
        pos = nextpos;
      }
    } else {
      return 0;
    }
  }
  return 0;
}

//______________________________________________________________________________________________

TString TQStringUtils::readTextFromFile(std::istream* input, const char* commentLine, const char* commentBlockOpen, const char* commentBlockClose){
  // read a data from the given stream, purging c-style comments
  input->seekg(0, std::ios::end);
  size_t length = (size_t) input->tellg();
  input->seekg(0, std::ios::beg);
  char* buffer = (char *)malloc((length+1)*sizeof(char));
  input->read(buffer, length);
  buffer[length]='\0';
  std::string text = std::string(buffer);
  TString result = "";
  size_t pos=0;
  char isstr = 0;
  while(pos < length){
    size_t nextposLine = text.find(commentLine, pos);
    size_t nextposBlock = text.find(commentBlockOpen, pos);
    size_t nextpos = std::min(nextposLine,nextposBlock);
    if(nextpos > length){
      result += text.substr(pos);
      break;
    } else {
      const std::string s(text.substr(pos, nextpos-pos));
      char stringswitch = hasStringSwitch(s.c_str());
      if(isstr==0) isstr = stringswitch;
      else if(stringswitch==isstr) isstr=0;
      result += s;
    }
    if(isstr!=0 || text[nextpos-1]=='\\'){
      result += text.substr(nextpos,1);
      pos = nextpos+1;
    } else {
      if(nextpos == nextposLine) pos = TQStringUtils::findFirstOf(text,"\n", nextpos);
      else if(nextpos == nextposBlock){
        result += TQStringUtils::repeat("\n",std::count(text.begin()+pos,text.begin()+nextpos,'\n'));
        pos = text.find(commentBlockClose, nextpos)+strlen(commentBlockClose);
      }
    } 
  }
  free(buffer);
  return result;
}

//______________________________________________________________________________________________

size_t TQStringUtils::findParenthesisMatch(const TString& str, size_t nextpos, const TString& paropen, const TString& parclose){
  // finds the nearest mathing closing parenthesis in a string from a given position assuming that the opening parenthesis is at the given position 
  size_t openbrace = 0;
  size_t closebrace = 0;
  size_t bracestack = 1;
  while((bracestack > 0) && (nextpos < (size_t)str.Length())){
    openbrace = TQStringUtils::find(str,paropen, nextpos+1);
    closebrace = TQStringUtils::find(str,parclose, nextpos+1);
    nextpos++;
    if(openbrace < closebrace){
      bracestack++;
      nextpos = openbrace;
    } else {
      bracestack--;
      nextpos = closebrace;
    }
  }
  return nextpos;
}

//______________________________________________________________________________________________

size_t TQStringUtils::rfindParenthesisMatch(const TString& str, size_t nextpos, const TString& paropen, const TString& parclose){
  // finds the nearest mathing (opening) parenthesis in a string from a given position assuming that the closing parenthesis is at the given position (backwards search)
  size_t openbrace = 0;
  size_t closebrace = 0;
  size_t bracestack = 1;
  while((bracestack > 0) && (nextpos < (size_t)str.Length())){
    openbrace = TQStringUtils::rfind(str,paropen, nextpos-1);
    closebrace = TQStringUtils::rfind(str,parclose, nextpos-1);
    // this line is correct and important!
    closebrace = std::min(closebrace, closebrace+1);
    // it helps to avoid overflows of 'closebrace' that lead to wrong return values!
    nextpos--;
    if(openbrace < closebrace){
      bracestack++;
      nextpos = closebrace;
    } else {
      bracestack--;
      nextpos = openbrace;
    }
  }
  return nextpos;
}

//______________________________________________________________________________________________

size_t TQStringUtils::findParenthesisMatch(const TString& str, size_t nextpos, char paropen, char parclose){
  // finds the nearest mathing closing parenthesis in a string from a given position assuming that the opening parenthesis is at the given position 
  return findParenthesisMatch(str, nextpos,chartostr(paropen),chartostr(parclose));
}

//______________________________________________________________________________________________

size_t TQStringUtils::rfindParenthesisMatch(const TString& str, size_t nextpos, char paropen, char parclose){
  // finds the nearest mathing (opening) parenthesis in a string from a given position assuming that the closing parenthesis is at the given position (backwards search)
  return rfindParenthesisMatch(str, nextpos,chartostr(paropen),chartostr(parclose));
}

//______________________________________________________________________________________________

size_t TQStringUtils::findFree(const TString& haystack, const TString& needle, const TString& paropen, const TString& parclose, size_t startpos){
  // finds the next "free" occurrence of needle in haystack. Note that this version checks for an exact match of the entire string paropen and parclose
  size_t needlepos = TQStringUtils::find(haystack,needle, startpos);
  size_t nextparopen = TQStringUtils::find(haystack,paropen, startpos);
  while(needlepos > nextparopen){
    startpos = findParenthesisMatch(haystack, nextparopen, paropen, parclose)+1;
    if (startpos == (size_t)(kNPOS+1)) return kNPOS;
    needlepos = TQStringUtils::find(haystack,needle, startpos);
    nextparopen = TQStringUtils::find(haystack,paropen, startpos);
  }
  return needlepos;
} 

//______________________________________________________________________________________________

size_t TQStringUtils::findFree(const TString& haystack, const TString& needle, const TString& parentheses, size_t startpos){
  // finds the next "free" occurrence of needle in haystack. This version assumes parentheses to be provides as "()[]{}", i.e. opening followed by closing parenthesis. Needle must be matched exactly.
  if (parentheses.Length()%2 != 0) {
    ERRORfunc("Number of parentheses is odd, returning kNPOS");
    return kNPOS;
    }
  TString paropen = "";
  TString parclose = "";
  for (int i=0;i<parentheses.Length(); ++i) {
    if (i%2 == 0) paropen += parentheses(i);
    else parclose += parentheses(i);
  }
  size_t needlepos = TQStringUtils::find(haystack,needle, startpos);
  size_t nextparopen = TQStringUtils::findFirstOf(haystack,paropen, startpos);
  while(needlepos > nextparopen){
    int whichPar = TQStringUtils::findFirstOf(paropen,haystack(nextparopen));
    startpos = findParenthesisMatch(haystack, nextparopen, paropen(whichPar), parclose(whichPar) )+1;
    if (startpos == (size_t)(kNPOS+1)) return kNPOS;
    needlepos = TQStringUtils::find(haystack,needle, startpos);
    nextparopen = TQStringUtils::findFirstOf(haystack,paropen, startpos);
  }
  
  return needlepos;
} 

//______________________________________________________________________________________________

size_t TQStringUtils::rfindFree(const TString& haystack, const TString& needle, const TString& parentheses, size_t startpos){
  // reverse-finds the next "free" occurrence of needle in haystack. This version assumes parentheses to be provides as "()[]{}", i.e. opening followed by closing parenthesis. Needle must be matched exactly.
  if (parentheses.Length()%2 != 0) {
    ERRORfunc("Number of parentheses is odd, returning kNPOS");
    return kNPOS;
    }
  TString paropen = "";
  TString parclose = "";
  for (int i=0;i<parentheses.Length(); ++i) {
    if (i%2 == 0) paropen += parentheses(i);
    else parclose += parentheses(i);
  }
  size_t needlepos = TQStringUtils::rfind(haystack,needle, startpos);
  size_t nextparopen = TQStringUtils::findLastOf(haystack,paropen, startpos);
  while(needlepos < nextparopen){
    int whichPar = TQStringUtils::findLastOf(paropen,haystack(nextparopen));
    startpos = rfindParenthesisMatch(haystack, nextparopen, paropen(whichPar), parclose(whichPar) )-1;
    if (startpos == (size_t)(kNPOS-1)) return kNPOS;
    needlepos = TQStringUtils::rfind(haystack,needle, startpos);
    nextparopen = TQStringUtils::findLastOf(haystack,paropen, startpos);
  }
  
  return needlepos;
}

//______________________________________________________________________________________________

size_t TQStringUtils::rfindFree(const TString& haystack, const TString& needle, const TString& paropen, const TString& parclose, size_t startpos){
  // reverse-finds the next "free" occurrence of needle in haystack. Note that this version checks for an exact match of the entire string paropen and parclose
  size_t needlepos = TQStringUtils::rfind(haystack,needle, startpos);
  size_t nextparclose = TQStringUtils::rfind(haystack,parclose, startpos);
  while(needlepos < nextparclose){
    startpos = rfindParenthesisMatch(haystack, nextparclose, paropen, parclose)-1;
    // this line is correct and important! 
    startpos = std::min(startpos+1, startpos-1); 
    // it helps to avoid overflows of 'startpos' that result in non-terminating function calls!
    needlepos = TQStringUtils::rfind(haystack,needle, startpos);
    nextparclose = TQStringUtils::rfind(haystack,parclose, startpos);
  }
  return needlepos;
} 

//______________________________________________________________________________________________

size_t TQStringUtils::findFreeOf(const TString& haystack, const TString& needles, const TString& paropen, const TString& parclose, size_t startpos){
  // finds the next "free" occurrence of any needle in haystack. Note that this version checks for an exact match of the entire string paropen and parclose
  size_t needlepos = TQStringUtils::findFirstOf(haystack,needles, startpos);
  size_t nextparopen = TQStringUtils::find(haystack,paropen, startpos);
  while(needlepos > nextparopen){
    startpos = findParenthesisMatch(haystack, nextparopen, paropen, parclose)+1;
    if (startpos == (size_t)(kNPOS+1)) return kNPOS;
    needlepos = TQStringUtils::findFirstOf(haystack,needles, startpos);
    nextparopen = TQStringUtils::find(haystack,paropen, startpos);
  }
  return needlepos;
} 

//______________________________________________________________________________________________

size_t TQStringUtils::rfindFreeOf(const TString& haystack, const TString& needles, const TString& paropen, const TString& parclose, size_t startpos){
  // reverse-finds the next "free" occurrence of any needle in haystack. Note that this version checks for an exact match of the entire string paropen and parclose
  size_t needlepos = TQStringUtils::findLastOf(haystack,needles, startpos);
  size_t nextparclose = TQStringUtils::rfind(haystack,parclose, startpos);
  while(needlepos < nextparclose){
    startpos = rfindParenthesisMatch(haystack, nextparclose, paropen, parclose);
    // this line is correct and important! 
    startpos = std::min(startpos+1, startpos-1); 
    // it helps to avoid overflows of 'startpos' that result in non-terminating function calls!
    needlepos = TQStringUtils::findLastOf(haystack,needles, startpos);
    nextparclose = TQStringUtils::rfind(haystack,parclose, startpos);
    // this line is correct and important! 
    nextparclose = std::min(nextparclose, nextparclose+1);
    // it helps to avoid overflows of 'nextparclose' that result wrong return values
  }
  return needlepos;
} 

//__________________________________________________________________________________|___________

int TQStringUtils::find(const TString& item, const std::vector<TString>& vec){
  // find the index of an item in a list, return -1 if not found
  for(size_t i=0; i<vec.size(); ++i){
    if(TQStringUtils::matches(item,vec[i])) return i;
  }
  return -1;
}

//______________________________________________________________________________________________

int TQStringUtils::find(const TString& haystack, const TString& needle, int pos){
  // finds the next occurrence of needle in haystack, starting at given position
  // if needle is found returns position of the first character of needle
  if(haystack.IsNull() || needle.IsNull() || pos > haystack.Length()) return kNPOS;
  if(pos < 0) pos = 0;
  size_t npos = 0;
  while(pos < haystack.Length()){
    if(haystack[pos] == needle[npos]){
      if(npos+1 == (size_t)needle.Length()){
        return pos+1-needle.Length(); //return the position of the first character of needle
      }
      npos++;
    } else {
      npos = 0;
    }
    pos++;
  }
  return kNPOS;
}

//______________________________________________________________________________________________

int TQStringUtils::rfind(const TString& haystack, const TString& needle, int pos){
  // finds the next occurrence of needle in haystack, starting backwards at given position
  if(haystack.IsNull() || needle.IsNull() || pos < 0) return kNPOS;
  if(pos > haystack.Length()) pos = haystack.Length() -1;
  size_t npos = needle.Length()-1;
  while(pos >= 0){
    if(haystack[pos] == needle[npos]){
      if(npos == 0){
        return pos;
      }
      npos--;
    } else {
      npos = needle.Length() -1;
    }
    pos--;
  }
  return kNPOS;
}

//______________________________________________________________________________________________

int TQStringUtils::findFirstOf(const TString& haystack, const TString& needle, int pos){
  // finds the next occurrence of any needle in haystack, starting at given position
  if(haystack.IsNull() || needle.IsNull() || pos > haystack.Length()) return kNPOS;
  if(pos < 0) pos = 0;
  while(pos < haystack.Length()){
    if(needle.Index(haystack[pos]) != kNPOS) return pos;
    pos++;
  }
  return kNPOS;
}

//______________________________________________________________________________________________

int TQStringUtils::findLastOf(const TString& haystack, const TString& needle, int pos){
  // finds the next occurrence of needle in haystack, starting backwards at given position
  if(haystack.IsNull() || needle.IsNull() || pos < 0) return kNPOS;
  if(pos > haystack.Length()) pos = haystack.Length() -1;
  while(pos >= 0){
    if(needle.Index(haystack[pos]) != kNPOS) return pos;
    pos--;
  }
  return kNPOS;
}

//______________________________________________________________________________________________

int TQStringUtils::rfindFirstOf(const TString& haystack, const TString& needle, int pos){
  // see TQStringUtils::findLastOf(const TString& haystack, const TString& needle, int pos)
  return findLastOf(haystack,needle,pos);
}

//______________________________________________________________________________________________

int TQStringUtils::findFirstNotOf(const TString& haystack, const TString& needle, int pos){
  // finds the next occurrence of anything else than needle in haystack, starting at given position
  if(haystack.IsNull() || needle.IsNull() || pos > haystack.Length()) return kNPOS;
  if(pos < 0) pos = 0;
  while(pos < haystack.Length()){
    if(needle.Index(haystack[pos]) == kNPOS) return pos;
    pos++;
  }
  return kNPOS;
}

//______________________________________________________________________________________________

int TQStringUtils::findLastNotOf(const TString& haystack, const TString& needle, int pos){
  // finds the next occurrence of anything else than needle in haystack, starting backwards at given position
  if(haystack.IsNull() || needle.IsNull() || pos < 0) return kNPOS;
  if(pos > haystack.Length()) pos = haystack.Length() -1;
  while(pos >= 0){
    if(needle.Index(haystack[pos]) == kNPOS) return pos;
    pos--;
  }
  return kNPOS;
}

//______________________________________________________________________________________________

int TQStringUtils::rfindFirstNotOf(const TString& haystack, const TString& needle, int pos){
  // see TQStringUtils::findLastNotOf(const TString& haystack, const TString& needle, int pos)
  return findLastNotOf(haystack,needle,pos);
}
//______________________________________________________________________________________________

std::vector<TString> TQStringUtils::split(const TString& str, const TString& del){
  // split a string at all occurrences of the delimiter and return a vector of results
  std::vector<TString> split;
  if(str.IsNull()) return split;
  TString substr;
  if(del.Length()>0){
    size_t oldpos = 0;
    size_t newpos = 0;
    while(newpos<(size_t)str.Length() && oldpos <= newpos){
      oldpos = newpos;
      newpos = TQStringUtils::find(str,del, oldpos);
      substr=str(oldpos, std::min((size_t)str.Length(),newpos)-oldpos);
      if(substr.Length()>0) split.push_back(substr);
      if(newpos < (size_t)str.Length()) newpos += del.Length();
    }
  }
  if(split.size() < 1) split.push_back(str);
  return split; 
}

//______________________________________________________________________________________________

std::vector<TString> TQStringUtils::split(const TString& str, const TString& del, const TString& paropen, const TString& parclose){
  // split a string into little chunks
  // This version of the stringsplit function accepts one delimiter
  // and respects parenthesis (e.g. does not split in a way that will break parenthesis matches)

  std::vector<TString> split;
  TString substr;
  if(del.Length()>0){
    size_t oldpos = 0;
    size_t newpos = 0;
    size_t nextparopen = str.Length();
    while(newpos<(size_t)str.Length() && oldpos <= (size_t)str.Length()){
      // find the next opening brace
      nextparopen = TQStringUtils::find(str,paropen, oldpos);
      // find the next occurance of the delimiter
      newpos = std::min((size_t)TQStringUtils::find(str,del, oldpos), (size_t)str.Length());
      // as long as there are opening braces before the delimiter
      while(nextparopen < newpos){
        // proceed to the corresponding closing brace
        nextparopen = findParenthesisMatch(str, nextparopen, paropen, parclose);
        // place the pointer behind that one
        newpos = std::max(newpos, nextparopen);
        // find the next breakpoint
        newpos = std::min(TQStringUtils::find(str,del, newpos), str.Length());
        // if there is any parenthesis left in the string, find the next opening one
        if(nextparopen < (size_t)str.Length()) nextparopen = TQStringUtils::find(str,paropen, nextparopen+1);
      }
      // extract the string
      substr=str(oldpos, std::min((size_t)str.Length(),newpos)-oldpos);
      // push it to the vector
      if(substr.Length()>0) split.push_back(substr);
      // and continue
      oldpos = newpos+del.Length();
    }
  }
  if(split.size() < 1) split.push_back(str);
  return split; 
}

//______________________________________________________________________________________________

TString TQStringUtils::findFormat(const TString& content){
  // make an elaborate guess of the formatting of some string
  // may return one of html,latex,roottex,unicode,ascii
  if(content.Contains(html)) return "html";
  if(content.Contains(roottex)) return "roottex";
  if(content.Contains(latexcmd) || content.Contains(latexmath)) return "latex";
  for(size_t i=0; i<(size_t)content.Length(); i++){
    if(content[i] < 0 || content[i] > 128) return "unicode";
  }
  return "ascii";
}

//______________________________________________________________________________________________

bool TQStringUtils::isASCII(TString content){
  // convert a string to plain ascii
  std::stringstream s;
  for(size_t i=0; i<(size_t)content.Length(); i++){
    if(!(content[i]>31 && content[i]<127)) return false;
  }
  return true;
}

//______________________________________________________________________________________________

TString TQStringUtils::makeASCII(const TString& content){
  // convert a string to plain ascii
  std::stringstream s;
  for(size_t i=0; i<(size_t)content.Length(); i++){
    if(content[i]>31 && content[i]<127) s << content[i];
  }
  return TString(s.str().c_str());
}


//______________________________________________________________________________________________

TString TQStringUtils::convertPlain2LaTeX(TString text){
  // convert plain text to LaTeX code
  // TODO: implement this function (currently, this is a DUMMY function)
  text.ReplaceAll("_","\\_");
  return text;
}

//______________________________________________________________________________________________

TString TQStringUtils::convertPlain2HTML(TString text){
  // convert plain text to HTML code
  // TODO: implement this function (currently, this is a DUMMY function)
  text.ReplaceAll("<","&lt;");
  text.ReplaceAll(">","&gt;");
  text.ReplaceAll("\"","&quot;");
  return text;
}

//______________________________________________________________________________________________

TString TQStringUtils::convertLaTeX2Plain(TString text, bool unicode){
  // convert LaTeX symbols to plain text (ASCII or unicode)
  text.ReplaceAll("$","");
  text.ReplaceAll("\\pm",unicode ? "±" : "+/-");
  text.ReplaceAll("\\leq","<=");
  text.ReplaceAll("\\geq",">=");
  text.ReplaceAll("\\cdot","*");
  text.ReplaceAll("\\to ",unicode ? "→" : "->");
  text.ReplaceAll("\\rightarrow ",unicode ? "→" : "->");
  text.ReplaceAll("\\to",unicode ? "→" : "->");
  text.ReplaceAll("\\rightarrow",unicode ? "→" : "->");
  text.ReplaceAll("\\ell","l");
  text.ReplaceAll("\\Alpha", unicode ? "Α" : "A"); text.ReplaceAll("\\alpha", unicode ? "α" : "a"); 
  text.ReplaceAll("\\Beta", unicode ? "Β" : "B"); text.ReplaceAll("\\beta", unicode ? "β" : "b"); 
  text.ReplaceAll("\\Gamma", unicode ? "Γ" : "G"); text.ReplaceAll("\\gamma", unicode ? "γ" : "y"); 
  text.ReplaceAll("\\Delta", unicode ? "Δ" : "D"); text.ReplaceAll("\\delta", unicode ? "δ" : "d"); 
  text.ReplaceAll("\\Epsilon",unicode ? "Ε" : "E"); text.ReplaceAll("\\epsilon",unicode ? "ε" : "e"); 
  text.ReplaceAll("\\Zeta", unicode ? "Ζ" : "Z"); text.ReplaceAll("\\zeta", unicode ? "ζ" : "z");
  text.ReplaceAll("\\Eta", unicode ? "Η" : "H"); text.ReplaceAll("\\eta", unicode ? "η" : "n");
  text.ReplaceAll("\\Theta", unicode ? "Θ" : "0"); text.ReplaceAll("\\theta", unicode ? "θ" : "0"); 
  text.ReplaceAll("\\Iota", unicode ? "I" : "I"); text.ReplaceAll("\\iota", unicode ? "ι" : "i"); 
  text.ReplaceAll("\\Kappa", unicode ? "Κ" : "K"); text.ReplaceAll("\\kappa", unicode ? "κ" : "k");
  text.ReplaceAll("\\Lambda", unicode ? "Λ" : "L"); text.ReplaceAll("\\lambda", unicode ? "λ" : "l"); 
  text.ReplaceAll("\\Mu", unicode ? "M" : "M"); text.ReplaceAll("\\mu", unicode ? "μ" : "m");
  text.ReplaceAll("\\Nu", unicode ? "Ν" : "N"); text.ReplaceAll("\\nu", unicode ? "ν" : "v"); 
  text.ReplaceAll("\\Xi", unicode ? "Ξ" : "Xi"); text.ReplaceAll("\\xi", unicode ? "ξ" : "xi");
  text.ReplaceAll("\\Omicron",unicode ? "Ο" : "O"); text.ReplaceAll("\\omicron",unicode ? "ο" : "o"); 
  text.ReplaceAll("\\Pi", unicode ? "Π" : "Pi"); text.ReplaceAll("\\pi", unicode ? "π" : "pi");
  text.ReplaceAll("\\Rho", unicode ? "Ρ" : "P"); text.ReplaceAll("\\rho", unicode ? "ρ" : "rho");
  text.ReplaceAll("\\Sigma", unicode ? "Σ" : "S"); text.ReplaceAll("\\sigma", unicode ? "σ" : "s");
  text.ReplaceAll("\\Tau", unicode ? "Τ" : "Tau");text.ReplaceAll("\\tau", unicode ? "τ" : "t");
  text.ReplaceAll("\\Upsilon",unicode ? "Υ" : "U"); text.ReplaceAll("\\upsilon",unicode ? "υ" : "u"); 
  text.ReplaceAll("\\Phi", unicode ? "Φ" : "Phi");text.ReplaceAll("\\phi", unicode ? "φ" : "phi");
  text.ReplaceAll("\\Chi", unicode ? "Χ" : "Chi");text.ReplaceAll("\\chi", unicode ? "χ" : "chi");
  text.ReplaceAll("\\Psi", unicode ? "Ψ" : "Psi");text.ReplaceAll("\\psi", unicode ? "ψ" : "psi");
  text.ReplaceAll("\\Omega", unicode ? "Ω" : "O"); text.ReplaceAll("\\omega", unicode ? "ω" : "w"); 
  TRegexp cmd("\\\\[a-zA-Z]+[ ]*");
  while(text.Contains(cmd)){
    TString seq = text(cmd);
    int pos = text.Index(cmd);
    int start = TQStringUtils::find(text,"{",pos+1);
    if(start < seq.Length() || start == kNPOS){
      text.Remove(pos,seq.Length());
    } else {
      int end = TQStringUtils::findParenthesisMatch(text,pos+seq.Length(),"{","}");
      if(end != kNPOS) text.Remove(end,1);
      text.Remove(pos,start-pos+1);
    }
  }
  TRegexp subScr("_{[^}]*}");
  while(text.Contains(subScr)){
    TString seq = text(subScr);
    int start = text.Index(seq);
    TString rep(seq);
    rep.Remove(rep.Length()-1,1);
    rep.Remove(0,2);
    text.Replace(start,seq.Length(), unicode ? TQStringUtils::makeUnicodeSubscript(rep) : ("_"+rep));
  }
  text.ReplaceAll("^","_");
  while(text.Contains(subScr)){
    TString seq = text(subScr);
    int start = text.Index(seq);
    TString rep(seq);
    rep.Remove(rep.Length()-1,1);
    rep.Remove(0,2);
    text.Replace(start,seq.Length(),unicode ? TQStringUtils::makeUnicodeSuperscript(rep) : ("^"+rep));
  }
  text.ReplaceAll("{}","");
  return text;
}

//______________________________________________________________________________________________

TString TQStringUtils::convertLaTeX2HTML(TString text){
  // converts LaTeX formattet text to HTML code

  bool open = true;
  text.ReplaceAll("<","&lt;");
  text.ReplaceAll(">","&gt;");
  while(text.Contains("$")){
    int loc = text.Index("$");
    text.Replace(loc,1,open ? "<i>" : "</i>");
    open = !open;
  }
  text.ReplaceAll("\\Alpha", "&Alpha;" ); text.ReplaceAll("\\alpha" ,"&alpha;" ); 
  text.ReplaceAll("\\Beta", "&Beta;" ); text.ReplaceAll("\\beta" ,"&beta;" ); 
  text.ReplaceAll("\\Gamma", "&Gamma;" ); text.ReplaceAll("\\gamma" ,"&gamma;" ); 
  text.ReplaceAll("\\Delta", "&Delta;" ); text.ReplaceAll("\\delta" ,"&delta;" ); 
  text.ReplaceAll("\\Epsilon","&Epsilon;"); text.ReplaceAll("\\epsilon","&epsilon;" ); 
  text.ReplaceAll("\\Zeta", "&Zeta;" ); text.ReplaceAll("\\zeta" ,"&zeta;" );
  text.ReplaceAll("\\Eta", "&Eta;" ); text.ReplaceAll("\\eta" ,"&eta;" );
  text.ReplaceAll("\\Theta", "&Theta;" ); text.ReplaceAll("\\theta" ,"&theta;" ); 
  text.ReplaceAll("\\Iota", "&Iota;" ); text.ReplaceAll("\\iota" ,"&iota;" ); 
  text.ReplaceAll("\\Kappa", "&Kappa;" ); text.ReplaceAll("\\kappa" ,"&kappa;" );
  text.ReplaceAll("\\Lambda", "&Lambda;" ); text.ReplaceAll("\\lambda" ,"&lambda;" ); 
  text.ReplaceAll("\\Mu", "&Mu;" ); text.ReplaceAll("\\mu" ,"&mu;" );
  text.ReplaceAll("\\Nu", "&Nu;" ); text.ReplaceAll("\\nu" ,"&nu;" ); 
  text.ReplaceAll("\\Xi", "&Xi;" ); text.ReplaceAll("\\xi" ,"&xi;" );
  text.ReplaceAll("\\Omicron","&OmicronM"); text.ReplaceAll("\\omicron","&omicron;" ); 
  text.ReplaceAll("\\Pi", "&Pi;" ); text.ReplaceAll("\\pi" ,"&pi;" );
  text.ReplaceAll("\\Rho", "&Rho;" ); text.ReplaceAll("\\rho" ,"&rho;" );
  text.ReplaceAll("\\Sigma", "&Sigma;" ); text.ReplaceAll("\\sigma" ,"&sigma;" );
  text.ReplaceAll("\\Tau", "&Tau;" ); text.ReplaceAll("\\tau" ,"&tau;" );
  text.ReplaceAll("\\Upsilon","&Upsilon;"); text.ReplaceAll("\\upsilon","&upsilon;" ); 
  text.ReplaceAll("\\Phi", "&Phi;" ); text.ReplaceAll("\\phi" ,"&phi;" );
  text.ReplaceAll("\\Chi", "&Chi;" ); text.ReplaceAll("\\chi" ,"&chi;" );
  text.ReplaceAll("\\Psi", "&Psi;" ); text.ReplaceAll("\\psi" ,"&psi;" );
  text.ReplaceAll("\\Omega", "&Omega;" ); text.ReplaceAll("\\omega" ,"&omega;" ); 
  TRegexp cmd("\\\\[a-zA-Z]+[ \n\t]*");
  while(text.Contains(cmd)){
    TString seq = text(cmd);
    int seqlen = seq.Length();
    TString sequence = TQStringUtils::trim(seq);
    TQStringUtils::removeLeading(sequence,"\\");
    int pos = text.Index(cmd);
    if(text[pos+seqlen] != '{'){
      text.Remove(pos,seqlen);
      if(sequence == "itshape"){
        text.Insert(pos,"<i>");
        text.Append("</i>");
      }
      if(sequence == "bfseries"){
        text.Insert(pos,"<b>");
        text.Append("</b>");
      }
      if(sequence == "to" || sequence == "rightarrow") text.Insert(pos,"&rarr;");
      if(sequence == "leftarrow") text.Insert(pos,"&larr;");
      if(sequence == "geq" ) text.Insert(pos,"&ge;");
      if(sequence == "leq" ) text.Insert(pos,"&le;");
      if(sequence == "cdot") text.Insert(pos,"&sdot;");
      if(sequence == "pm" ) text.Insert(pos,"&plumn;");
      if(sequence == "ell" ) text.Insert(pos,"&ell;");
    } else {
      seqlen++;
      int end = TQStringUtils::findParenthesisMatch(text,pos+seq.Length(),"{","}");
      if(sequence == "textit" || sequence == "ensuremath"){
        if(end != kNPOS) text.Replace(end,1,"</i>");
        else text.Append("</i>");
        text.Replace(pos,seqlen,"<i>");
      } else if(sequence == "textbf"){
        if(end != kNPOS) text.Replace(end,1,"</b>");
        else text.Append("</b>");
        text.Replace(pos,seqlen,"<b>");
      } else if(sequence == "mathrm"){
        if(end != kNPOS) text.Replace(end,1,"</span>");
        else text.Append("</span>");
        text.Replace(pos,seqlen,"<span style=\"font-style:normal;\">");
      } else if(sequence == "bar"){
        if(end != kNPOS) text.Replace(end,1,"</span>");
        else text.Append("</span>");
        text.Replace(pos,seqlen,"<span style=\"text-decoration: overline;\">");
      } else {
        if(end != kNPOS) text.Remove(end,1);
        text.Remove(pos,seqlen);
      }
    }
  }
  TRegexp subScr("_{[^}]*}");
  while(text.Contains(subScr)){
    TString seq = text(subScr);
    int start = text.Index(subScr);
    text.Replace(start+seq.Length()-1,1,"</sub>");
    text.Replace(start,2,"<sub>");
  }
  text.ReplaceAll("^","_");
  while(text.Contains(subScr)){
    TString seq = text(subScr);
    int start = text.Index(subScr);
    text.Replace(start+seq.Length()-1,1,"</sup>");
    text.Replace(start,2,"<sup>");
  }
  text.ReplaceAll("{}","");
  return text;
}

//______________________________________________________________________________________________

TString TQStringUtils::convertHTML2Plain(TString text,bool unicode){
  // convert HTML code to plain text (ASCII or unicode)
  // TODO: implement this function (currently, this is a DUMMY function)
  return text;
}

//______________________________________________________________________________________________

TString TQStringUtils::convertHTML2LaTeX(TString text){
  // convert HTML code to LaTeX
  // TODO: implement this function (currently, this is a DUMMY function)
  return text;
}

//______________________________________________________________________________________________

TString TQStringUtils::convertROOTTeX2Plain(TString text, bool unicode){
  // convert ROOTTeX to plain text (ASCII or unicode)
  // TODO: implement this function (currently, this is a DUMMY function)
  return text;
}

//______________________________________________________________________________________________

TString TQStringUtils::convertLaTeX2ROOTTeX(TString text){
  // convert ROOTTeX to plain text (ASCII or unicode)
  bool open = false;
  while(text.Contains("$")){
    int pos = text.First("$");
    if(!open) text.Replace(pos,1,"#it{");
    else text.Replace(pos,1,"}");
    open = !open;
  }
  text.ReplaceAll("^{*}","#kern[-0.2]{#lower[-0.2]{*}}");
  text.ReplaceAll("\\","#");
  text.ReplaceAll("#ell","l");
  text.ReplaceAll("{}","");
  return text;
}

//______________________________________________________________________________________________

TString TQStringUtils::convertROOTTeX2LaTeX(TString text){
  // convert ROOTTeX to (real) LaTeX
  text.ReplaceAll("#","\\");
  return text;
}

//______________________________________________________________________________________________

TString TQStringUtils::convertROOTTeX2HTML(TString text){
  // convert ROOTTeX to HTML code
  // TODO: implement this function (currently, this is a DUMMY function)
  text.ReplaceAll("#","\\");
  return convertLaTeX2HTML(text);
}

//______________________________________________________________________________________________

TString TQStringUtils::makeUppercase(TString s){
  // convert a string to uppercase
  s.ToUpper();
  return s;
}

//______________________________________________________________________________________________

TString TQStringUtils::makeLowercase(TString s){
  // convert a string to lowercase
  s.ToLower();
  return s;
}

//______________________________________________________________________________________________

TString TQStringUtils::makeUnicodeSuperscript(TString s){
  // replace all characters in the given string by their unicode superscript variants if possible
  s.ReplaceAll("0","⁰");s.ReplaceAll("1","¹");s.ReplaceAll("2","²");s.ReplaceAll("3","³");s.ReplaceAll("4","⁴");s.ReplaceAll("5","⁵");s.ReplaceAll("6","⁶");s.ReplaceAll("7","⁷");s.ReplaceAll("8","⁸");s.ReplaceAll("9","⁹");s.ReplaceAll("+","⁺");s.ReplaceAll("-","⁻");s.ReplaceAll("=","⁼");s.ReplaceAll("(","⁽");s.ReplaceAll(")","⁾");s.ReplaceAll("a","ᵃ");s.ReplaceAll("b","ᵇ");s.ReplaceAll("c","ᶜ");s.ReplaceAll("d","ᵈ");s.ReplaceAll("e","ᵉ");s.ReplaceAll("f","ᶠ");s.ReplaceAll("g","ᵍ");s.ReplaceAll("h","ʰ");s.ReplaceAll("i","ⁱ");s.ReplaceAll("j","ʲ");s.ReplaceAll("k","ᵏ");s.ReplaceAll("l","ˡ");s.ReplaceAll("m","ᵐ");s.ReplaceAll("n","ⁿ");s.ReplaceAll("o","ᵒ");s.ReplaceAll("p","ᵖ");s.ReplaceAll("r","ʳ");s.ReplaceAll("s","ˢ");s.ReplaceAll("t","ᵗ");s.ReplaceAll("u","ᵘ");s.ReplaceAll("v","ᵛ");s.ReplaceAll("w","ʷ");s.ReplaceAll("x","ˣ");s.ReplaceAll("y","ʸ");s.ReplaceAll("z","ᶻ");s.ReplaceAll("A","ᴬ");s.ReplaceAll("B","ᴮ");s.ReplaceAll("D","ᴰ");s.ReplaceAll("E","ᴱ");s.ReplaceAll("G","ᴳ");s.ReplaceAll("H","ᴴ");s.ReplaceAll("I","ᴵ");s.ReplaceAll("J","ᴶ");s.ReplaceAll("K","ᴷ");s.ReplaceAll("L","ᴸ");s.ReplaceAll("M","ᴹ");s.ReplaceAll("N","ᴺ");s.ReplaceAll("O","ᴼ");s.ReplaceAll("P","ᴾ");s.ReplaceAll("R","ᴿ");s.ReplaceAll("T","ᵀ");s.ReplaceAll("U","ᵁ");s.ReplaceAll("V","ⱽ");s.ReplaceAll("W","ᵂ");s.ReplaceAll("α","ᵅ");s.ReplaceAll("β","ᵝ");s.ReplaceAll("γ","ᵞ");s.ReplaceAll("δ","ᵟ");s.ReplaceAll("ε","ᵋ");s.ReplaceAll("θ","ᶿ");s.ReplaceAll("ι","ᶥ");s.ReplaceAll("Φ","ᶲ");s.ReplaceAll("φ","ᵠ");s.ReplaceAll("χ","ᵡ");return s;} 

//______________________________________________________________________________________________

TString TQStringUtils::makeUnicodeSubscript(TString s){
  // replace all characters in the given string by their unicode superscript variants if possible
  s.ReplaceAll("0","₀");s.ReplaceAll("1","₁");s.ReplaceAll("2","₂");s.ReplaceAll("3","₃");s.ReplaceAll("4","₄");s.ReplaceAll("5","₅");s.ReplaceAll("6","₆");s.ReplaceAll("7","₇");s.ReplaceAll("8","₈");s.ReplaceAll("9","₉");s.ReplaceAll("+","₊");s.ReplaceAll("-","₋");s.ReplaceAll("=","₌");s.ReplaceAll("(","₍");s.ReplaceAll(")","₎");s.ReplaceAll("a","ₐ");s.ReplaceAll("e","ₑ");s.ReplaceAll("h","ₕ");s.ReplaceAll("i","ᵢ");s.ReplaceAll("j","ⱼ");s.ReplaceAll("k","ₖ");s.ReplaceAll("l","ₗ");s.ReplaceAll("m","ₘ");s.ReplaceAll("n","ₙ");s.ReplaceAll("o","ₒ");s.ReplaceAll("p","ₚ");s.ReplaceAll("r","ᵣ");s.ReplaceAll("s","ₛ");s.ReplaceAll("t","ₜ");s.ReplaceAll("u","ᵤ");s.ReplaceAll("v","ᵥ");s.ReplaceAll("x","ₓ");s.ReplaceAll("β","ᵦ");s.ReplaceAll("γ","ᵧ");s.ReplaceAll("ρ","ᵨ");s.ReplaceAll("φ","ᵩ");s.ReplaceAll("χ","ᵪ");return s;}

//______________________________________________________________________________________________

TString TQStringUtils::format(const char *va_(fmt), ...){
  // format a string - analog to TString::Format
  va_list ap;
  va_start(ap, va_(fmt));
  TString str(vaFormat(va_(fmt), ap));
  va_end(ap);
  return str;
}


//__________________________________________________________________________________|___________

TString TQStringUtils::concatenate(int n, ...){
  // concatenate arguments
  bool first = true;
  TString text;

  va_list ap;
  va_start(ap, va_(n));
 
  for (int i=1;i<n;i++){
    const char* val = va_arg(ap,const char*);
    if(!first) text.Append(",");
    text.Append(val);
    first=false;
  }
  va_end(ap);
 
  return text;
}

//______________________________________________________________________________________________

char* TQStringUtils::vaFormat(const char *fmt, va_list ap){
  // format a string - variadic variant of TString::Format
  // analog to (private) function TString::FormatImp
 
  Ssiz_t buflen = 20 + strlen(fmt); // pick a number, any strictly positive number
  char* buffer = (char*)malloc(buflen*sizeof(char));
 
  va_list sap;
  R__VA_COPY(sap, ap);
  bool done = false;

  int n, vc = 0;
  do {
    n = vsnprintf(buffer, buflen, fmt, ap);
    // old vsnprintf's return -1 if string is truncated new ones return
    // total number of characters that would have been written
    if (n == -1 || n >= buflen) {
      if (n == -1)
        buflen *= 2;
      else
        buflen = n+1;
      buffer = (char*)realloc(buffer,buflen*sizeof(char));
      va_end(ap);
      R__VA_COPY(ap, sap);
      vc = 1;
    } else {
      done = true;
    }
  } while(!done);
 
  va_end(sap);
  if (vc) va_end(ap);
 
  return buffer;
}


TString TQStringUtils::replaceEnclosed(TString haystack,TString needle,TString newNeedle, const TString& symbols){
  // replace all occurences of needle in haystack, as long as they are
  // enclosed in characters from the symbols
  int pos = 0;
  while(true){
    size_t newpos = TQStringUtils::find(haystack,needle,pos);
    if(newpos < (size_t)haystack.Length()){
      if( ( ( newpos == 0 || symbols.Contains(haystack(newpos-1,1))) )
         && ( (newpos==(size_t)haystack.Length()-1) ||
         symbols.Contains(haystack(newpos+needle.Length(),1)) ) ){ //4 ->needle.Length() ?
        haystack.Replace(newpos,needle.Length(),newNeedle);
        pos = newpos+newNeedle.Length(); //pos = newpos+newNeedle.Length() ? (otherwise infinite recursion possible)
      } else {
        pos = newpos+needle.Length();
      }
    } else {
      break;
    }
  }
  return haystack;
}
  
TString TQStringUtils::getColorDefStringLaTeX(const TString& name, int color){
  // retrieve a color definition string for LaTeX
  // of the form "\definecolor{<name>}{rgb}{<r>,<g>,<b>}"
  // for some predefined color enum
  TColor* c = gROOT->GetColor(color);
  return getColorDefStringLaTeX(name,c);
}


TString TQStringUtils::getColorDefStringLaTeX(const TString& name, TColor* color){
  // retrieve a color definition string for LaTeX
  // of the form "\definecolor{<name>}{rgb}{<r>,<g>,<b>}"
  // for some ROOT TColor 
  if(!color) return "";
  float r,g,b;
  color->GetRGB(r,g,b);
  return TString::Format("\\definecolor{%s}{rgb}{%f,%f,%f}",name.Data(),r,g,b);
}

TString TQStringUtils::padNumber(int num, int length){
  // pad an integer with leading zeros
  TString retval(TString::Format("%d",num));
  while(retval.Length() < length){
    retval.Prepend("0");
  }
  return retval;
}

#include <iostream>
#include <fstream>
#include <string>
#include <stdlib.h>

// #define _DEBUG_
#include "QFramework/TQLibrary.h"

#include "QFramework/TQTable.h"
#include "QFramework/TQUtils.h"

#include <math.h>

////////////////////////////////////////////////////////////////////////////////////////////////
//
// The TQTable class provides a general modular interface for creating, modifying, reading
// and writing tables (aka two-dimensional arrays of text), including, but not limited to
// - reading and writing CSV files
// - reading and writing HTML files
// - reading and writing LaTeX-formatted text files
// - reading and writing unicode-ascii-art formatted text files
// 
// Some of these features are still under developement, but general class operations should
// be stable already. This class was designed as an enhancement to the TQCutflowPrinter,
// but can be used for other purposes as well.
//
////////////////////////////////////////////////////////////////////////////////////////////////

ClassImp(TQTable)

using namespace TQStringUtils;

TQTable::TQTable() : 
  TQNamedTaggable("TQTable"),
  nfields(0),
  ncols(0),
  nrows(0),
  data(NULL),
  autoExpand(true),
  manualAllocation(false)
{
  /// default constructor
  DEBUGclass("default constructor called");
  this->vlines.push_back(0);
  this->vlines.push_back(1);
  this->hlines.push_back(0);
  this->hlines.push_back(1);
  this->setup();
}

TQTable::TQTable(const TString& name) : 
  TQNamedTaggable(name),
  nfields(0),
  ncols(0),
  nrows(0),
  data(NULL),
  autoExpand(true),
  manualAllocation(false)
{
  /// default constructor (with name)
  DEBUGclass("named constructor called");
  this->vlines.push_back(0);
  this->vlines.push_back(1);
  this->hlines.push_back(0);
  this->hlines.push_back(1);
  this->setup();
}


TQTable::TQTable(const TQTable* other) :
  TQNamedTaggable(other ? other->GetName() : "TQTable"),
  nfields(0),
  ncols(0),
  nrows(0),
  data(NULL),
  autoExpand(true),
  manualAllocation(false)
{
  // copy constructor for TQTable class
  DEBUGclass("pointer copy constructor called");
  if(other){
    this->appendLines(*other,0);
  }
  this->importTags(other);
}

TQTable::TQTable(const TQTable& other) :
  TQNamedTaggable(other.GetName()),
  nfields(0),
  ncols(0),
  nrows(0),
  data(NULL),
  autoExpand(true),
  manualAllocation(false)
{
  // copy constructor for TQTable class
  DEBUGclass("reference copy constructor called");
  this->appendLines(other,0);
  this->importTags(other);
}

TQTable::TQTable(TList* l) : 
  TQNamedTaggable(l ? l->GetName() : "TList"),
  nfields(0),
  ncols(0),
  nrows(0),
  data(NULL),
  autoExpand(false),
  manualAllocation(false)
{
  // converting constructor for TQTable class from TList
  DEBUGclass("TList copy constructor called");
  this->setFromTList(l);
}

void TQTable::merge(TQTable* other){
  // append the extra lines from another instance to this one (merge)
  if (!other) return;
  int oldrows = this->nrows;
  this->expand(this->nrows + (other->nrows - 1),std::max(this->ncols,other->ncols));
  int offset = oldrows * this->ncols;
  
  int nElements = 0;
  for(unsigned int i=0; i<other->nrows - 1; i++){
    if(this->findRow(other->getEntryPlain(1+i,0,false),0) < 0){
      for(unsigned int j=0; j<other->ncols; j++){
        TQTaggable* entry = other->getEntryInternal(1+i,j);
        if(entry){
          this->data[offset + nElements*this->ncols + j] = new TQTaggable(entry);
          
        } else {
          this->data[offset + nElements*this->ncols + j] = NULL;
        }
      }
      ++nElements; 
    }
  }
  std::cout<<"done!"<<std::endl;
}

int TQTable::appendLines(const TQTable& other, int startAt, bool ignoreHlines){
  // append the lines from another instance to this one (deep copy)
  DEBUGclass("appendLines called");
  if(!ignoreHlines){
    if(other.hlines.size() > 0){
      if(this->nrows == 0){
        this->hlines.push_back(other.hlines[0]);
      } else {
        for(unsigned int i=this->hlines.size(); i<=this->nrows; i++){
          this->hlines.push_back(0);
        }
      }
    }
  }
  int oldrows = this->nrows;
  this->expand(this->nrows + (other.nrows - startAt),std::max(this->ncols,other.ncols));
  int offset = oldrows * this->ncols;
  int nElements = 0;
  for(unsigned int i=0; i<other.nrows - startAt; i++){
    for(unsigned int j=0; j<other.ncols; j++){
      TQTaggable* entry = other.getEntryInternal(startAt+i,j);
      if(entry){
        this->data[offset + i*this->ncols + j] = new TQTaggable(entry);
        nElements++;
      } else {
        this->data[offset + i*this->ncols + j] = NULL;
      }
    } 
  }
  if(!ignoreHlines){
    for(unsigned int i=startAt+1; i< other.hlines.size(); i++){
      this->hlines.push_back(other.hlines[i]);
    }
  }
  return nElements;
}

int TQTable::appendLines(const TQTable* other, int startAt, bool ignoreHlines){
  // append the lines from another instance to this one (deep copy)
  DEBUGclass("appendLines called");
  if(other) return this->appendLines(*other,startAt,ignoreHlines);
  return -1;
}

 
TQTable::~TQTable(){
  /// default destructor
  DEBUGclass("destructor called");
  for(unsigned int i=0; i<(this->nfields); i++){
    if(this->data[i]) delete this->data[i];
  }
  if(this->data){
    if(manualAllocation) free(this->data);
    else delete[] this->data;
  }
} 

int TQTable::getNcols(){
  // return the number of columns
  return this->ncols;
}
int TQTable::getNrows(){
  // return the number of rows
  return this->nrows; 
}

bool TQTable::setVline(int col, int type){
  // define the vertical line between columns
  // col-1 and col to be of the given type
  // supported types are:
  // 0: no line
  // 1: single solid line (black)
  // 2: double solid line (black)
  if(col < 0) col = this->ncols - col;
  while((unsigned int)(col+1) > this->vlines.size()) this->vlines.push_back(0);
  this->vlines[col] = type;
  return true;
}

void TQTable::clearColAlign(){
  // clear all predefined column horizontal alignments
  this->colAlign.clear();
}

void TQTable::setColAlign(unsigned int col, TString align){
  // set the horizontal alignment of the given column
  // known alignments are
  // l: left-aligned
  // r: right-aligned
  // c: centered
  if(col < 0) col = this->ncols - col;
  while(col+1 > this->colAlign.size()) this->colAlign.push_back(this->getTagStringDefault("colAlign","c"));
  this->colAlign[col] = align;
}

TString TQTable::getColAlign(unsigned int col){
  // retrieve the horizontal alignment of the given column
  //@tag: colAlign Sets the default alignment of columns. Default "c", other values "l","r".
  if(col >= this->colAlign.size()) return this->getTagStringDefault("colAlign","c");
  return this->colAlign[col];
}

TString TQTable::getColAlignHTML(unsigned int col){
  // retrieve the horizontal alignment of the given column
  // this function returns the html-aliases for the alignments, i.e.
  // "center" for centered
  // "right" for right-aligned
  // "left" for left-aligned
  TString align = (col >= this->colAlign.size() ? this->getTagStringDefault("colAlign","c") : this->colAlign[col]);
  if(align == "c") return "center";
  if(align == "r") return "right";
  if(align == "l") return "left" ;
  return "";
}

bool TQTable::setHline(int row, int type){
  // define the vertical line between rows
  // row-1 and row to be of the given type
  // supported types are:
  // 0: no line
  // 1: single solid line (black)
  // 2: double solid line (black)
  if(row < 0) row = this->nrows - row;
  while((unsigned int)(row+1) > this->hlines.size()) this->hlines.push_back(0);
  this->hlines[row] = type;
  return true;
}

bool TQTable::clearVlines(){
  // clear all vertical lines
  this->vlines.clear();
  return true;
}

bool TQTable::clearHlines(){
  // clear all horizontal lines
  this->hlines.clear();
  return true;
}

bool TQTable::readCSVfile(const TString& fname, const TString& sep, const TString& leftquote, const TString& rightquote){
  // aquire table data from the given CSV-formatted file
  DEBUGclass("readCSVfile called");
  std::ifstream in(fname.Data());
  if(!in.is_open()) return false;
  this->readCSV(&in,sep,leftquote,rightquote);
  return true;
}

bool TQTable::readTSVfile(const TString& fname, const TString& seps, int ignoreHeadLines, int nsepmin){
  // aquire table data from the given TSV-formatted file
  DEBUGclass("readTSVfile called");
  std::ifstream in(fname.Data());
  if(!in.is_open()) return false;
  std::string str;
  int i=0;
  while(i<ignoreHeadLines && in.good()){
    std::getline(in,str);
    i++;
  }
  this->readTSV(&in,seps,nsepmin);
  return true;
}

bool TQTable::readLaTeXfile(const TString& fname){
  // aquire table data from the given LaTeX-formatted file
  DEBUGclass("readLaTeXfile called");
  std::ifstream in(fname.Data());
  if(!in.is_open()) return false;
  this->readLaTeX(&in);
  return true;
}

bool TQTable::readHTMLfile(const TString& fname){
  // aquire table data from the given HTML-formatted file
  DEBUGclass("readHTMLfile called");
  std::ifstream in(fname.Data());
  if(!in.is_open()) return false;
  this->readHTML(&in);
  return true;
}

void TQTable::readCSV(std::istream* input, const TString& sep, const TString& leftquote, const TString& rightquote){
  // aquire table data from the given CSV-formatted stream
  DEBUGclass("readCSV called");
  this->shrink();
  std::string linebuffer;
  getline(*input, linebuffer);
  int estCols = this->ncols;
  if(estCols == 0){
    estCols = TQStringUtils::countText(linebuffer, sep)+1;
  } else getline(*input, linebuffer);
  unsigned int i = this->nrows;
  this->expand(this->nrows+256,estCols);
  unsigned int pos=0;
  unsigned int nextpos=0;
  bool eof = false;
  bool empty;
  //@tag: [readFormatPrior] Defines the encoding/format assumed when reading from an external source. If this is set to
  //@tag "" or "verbatim", the format is automatically determined via TQStringUtils::findFormat. Unless the prior is empty, only the sepcified/automatically determined variant is overwritten. If variants different from prior are empty, an attempt for an automatic conversion is made. Possible values: "", "verbatim", "unicode", "ascii", "latex", "roottex" and "html". 
  TString formatPrior = this->getTagStringDefault("readFormatPrior","");
  while(!eof){
    eof = !input->good();
    pos =0;
    nextpos = 0;
    if(i == this->nrows){
      this->expand(2*this->nrows,this->ncols);
    }
    empty = true;
    if((linebuffer.size() > 0) && (linebuffer[0] != '#')){
      for(unsigned int j=0; j<this->ncols; j++){
        if(pos < linebuffer.size()){
          nextpos = findFree(linebuffer, sep, leftquote, rightquote, pos);
          if(nextpos < 0 || nextpos > linebuffer.size()) nextpos = linebuffer.size();
          TQTaggable* newEntry = new TQTaggable();
          TString content = TQStringUtils::trim(linebuffer.substr(pos, nextpos-pos), leftquote+rightquote+" ");
          this->setContent(newEntry,content,formatPrior);
          this->data[i*this->ncols + j] = newEntry;
          pos = nextpos+1;
          empty = false;
        } else this->data[i*this->ncols + j] = NULL;
      }
    } 
    if(!empty) i++;
    if(!eof) getline(*input, linebuffer);
  }
  this->shrink();
  this->autoExpand = false;
}

void TQTable::readTSV(std::istream* input, const TString& seps, int nsepmin){
  // aquire table data from the given TSV-formatted stream
  DEBUGclass("readTSV called");
  this->shrink();
  std::string linebuffer;
  unsigned int i = this->nrows;
  this->expand(this->nrows+256,this->ncols == 0 ? 256 : this->ncols);
  unsigned int pos=0;
  bool eof = false;
  bool empty;
  //tag documentation see readCSV(...)
  TString formatPrior = this->getTagStringDefault("readFormatPrior","");
  while(!eof){
    eof = !input->good();
    if(eof) break;
    getline(*input, linebuffer);
    pos = findFirstNotOf(linebuffer, " \t");
    if(i == this->nrows){
      this->expand(2*this->nrows,this->ncols);
    }
    empty = true;
    if((linebuffer.size() > 0) && (linebuffer[0] != '#')){
      for(unsigned int j=0; j<this->ncols; j++){
        if(pos < linebuffer.size()){
          unsigned int tmppos = pos;
          unsigned int nextpos = pos;
          while(tmppos < nextpos+nsepmin && linebuffer[nextpos] != '\t'){
            // std::cout << nextpos << "/" << tmppos << std::endl;
            nextpos = findFirstOf(linebuffer, seps, tmppos);
            if(nextpos < 0 || nextpos > linebuffer.size()){
              nextpos = linebuffer.size();
              break;
            }
            tmppos = findFirstNotOf(linebuffer, seps, nextpos);
            if(tmppos < 0 || tmppos > linebuffer.size()){
              break;
            }
          }
          // std::cout << linebuffer.substr(0,pos) << "|" << pos <<"|" << linebuffer.substr(pos,nextpos-pos) << "|" <<nextpos << "|" << linebuffer.substr(nextpos) << std::endl;
          TQTaggable* newEntry = new TQTaggable();
          TString content = TQStringUtils::trim(linebuffer.substr(pos, nextpos-pos));
          if(!content.IsNull()){
            this->setContent(newEntry,content,formatPrior);
            this->data[i*this->ncols + j] = newEntry;
            empty = false;
          }
          pos = findFirstNotOf(linebuffer,seps,nextpos);
        } else this->data[i*this->ncols + j] = NULL;
      }
    }
    if(!empty){
      i++;
    }
  }
  this->autoExpand = false;
  this->shrink();
}


void TQTable::readHTML(std::istream* input){
  // aquire table data from the given HTML-formatted stream
  // TODO: parse style tags
  DEBUGclass("readHTML called");
  this->shrink();
  TString content = readTextFromFile(input);
  unsigned int pos = TQStringUtils::find(content,"<tr>");
  unsigned int nextpos= TQStringUtils::find(content,"</tr>", pos);
  std::string linebuffer(content(pos, nextpos-pos).Data());
  if(this->ncols == 0) this->ncols = TQStringUtils::countText(linebuffer, "<th>");
  unsigned int i = this->nrows;
  this->expand(this->nrows+256,this->ncols == 0 ? 256 : this->ncols);
  unsigned int lineend = 1;
  while(pos < (unsigned int)content.Length()){
    if(i == this->nrows){
      this->expand(this->nrows*2,this->ncols);
    }
    lineend = std::min(TQStringUtils::find(content,"</tr>", lineend+1), content.Length());
    for(unsigned int j=0; j<this->ncols; j++){
      pos = std::min(TQStringUtils::find(content,"<td>", pos), TQStringUtils::find(content,"<th>", pos));
      if(pos<lineend){
        nextpos = std::min(TQStringUtils::find(content,"</td>", pos), TQStringUtils::find(content,"</th>", pos));
        TQTaggable* newEntry = new TQTaggable();
        TString newcontent = TQStringUtils::trim(content(pos+4, nextpos-pos-4), "<> ");
        this->setContent(newEntry,newcontent,"html");
        this->data[i*this->ncols + j] = newEntry;
        pos = nextpos;
      } else this->data[i*this->ncols + j] = NULL;
    }
    i++;
    if(lineend < (unsigned int)content.Length()) pos = TQStringUtils::find(content,"<tr>", lineend);
    else break;
  }
  this->autoExpand = false;
  this->shrink();
}

void TQTable::readLaTeX(std::istream* input){
  // aquire table data from the given LaTeX-formatted stream
  this->shrink();
  std::string tabularnewline = "\\tabularnewline";
  TString content = readTextFromFile(input);
  unsigned int pos = content.Length();
  std::vector<std::string> tableHeads;
  tableHeads.push_back("tabular");
  tableHeads.push_back("supertabular");
  tableHeads.push_back("table");
  tableHeads.push_back("longtable");
  for(unsigned int i=0; i<tableHeads.size(); i++){
    pos = std::min(pos, (unsigned int)TQStringUtils::find(content,"\\begin{"+tableHeads[i]+"}"));
  }
  pos = TQStringUtils::find(content,"}", pos);
  pos = TQStringUtils::find(content,"{", pos)+1;
  unsigned int nextpos = findParenthesisMatch(content, pos, "{", "}");
  std::vector<TString> buffer = TQStringUtils::split(content(pos, nextpos-pos), " |\t", "{", "}");
  std::string bufstr;
  unsigned int i = this->nrows;
  this->expand(this->nrows+256,buffer.size());
  pos = nextpos;
  nextpos = TQStringUtils::find(content,tabularnewline, pos);
  while(pos < (unsigned int)content.Length()){
    if(i == this->nrows){
      this->expand(this->nrows*2,this->ncols);
    }
    nextpos = std::min(TQStringUtils::find(content,tabularnewline, pos), content.Length());
    bufstr = TQStringUtils::compactify(content(pos, nextpos-pos));
    buffer = TQStringUtils::split(bufstr, "&");
    for(unsigned int j=0; j<std::min((size_t)(this->ncols), buffer.size()); j++){
      TQTaggable* newEntry = new TQTaggable();
      TString content = TQStringUtils::trim(buffer[j], "\t\n ");
      this->setContent(newEntry,content,"latex");
      this->data[i*this->ncols + j] = newEntry;
    }
    i++;
    pos = nextpos+tabularnewline.size();
  }
  this->autoExpand = false;
  this->shrink();
}


bool TQTable::print(std::ostream* output, TQTaggable tags){
  // print table data to the given stream. format can specified by style tags.
  TString format = tags.getTagStringDefault("format","plain");
  format.ToLower();
  return this->print(output,format,tags);
}

bool TQTable::print(std::ostream* output, const TString& format, TQTaggable tags){
  // print table data to the given stream. format is specifiec explicitly,
  tags.setGlobalOverwrite(false);
  tags.importTags(this);
  if(format == "plain" || format == "unicode" || format=="txt") return this->printPlain(output,tags);
  if(format == "latex" || format=="tex") return this->printLaTeX(output,tags);
  if(format == "html") return this->printHTML(output,tags);
  if(format == "csv") return this->printCSV(output,tags);
  return false;
}

bool TQTable::printCSV(std::ostream* output, TQTaggable tags){
  // print table data to the given stream. format is CSV and can specified by style tags.
  tags.setGlobalOverwrite(false);
  tags.importTags(this);
  TString format = tags.getTagStringDefault("format","csv");
  TString sep = tags.getTagStringDefault("sep",",");
  TString quote = tags.getTagStringDefault("quote","\"");
  if(!output || !this->data) return false;
  for(unsigned int i=0; i<this->nrows; i++){
    for(unsigned int j=0; j<this->ncols-1; j++){
      if(quote.IsNull()) *output << this->getEntry(i,j,format);
      else *output << quote << this->getEntry(i,j,format) << quote << sep;
    }
    if(quote.IsNull()) *output << this->getEntry(i,this->ncols-1,format);
    else *output << quote << this->getEntry(i,this->ncols-1,format) << quote << std::endl;
  }
  return true;
}

bool TQTable::printHTML(std::ostream* output, TQTaggable tags){
  // print table data to the given stream. format is HTML and can specified by style tags.
  tags.setGlobalOverwrite(false);
  tags.importTags(this);
  TString title = tags.getTagStringDefault("title",this->GetName());
  if(!output || !this->data) return false;
  //@tag:[standalone] If set to true a standalone document is created. Argument tag overrides object tag.
  bool standalone = tags.getTagBoolDefault("standalone",false);
  if(standalone){
    *output << "<html>" << std::endl;
    *output << "<head>" << std::endl;
    *output << "<title>" << title << "</title>" << std::endl;
    *output << "<meta http-equiv=\"Content-Type\" content=\"text/html; charset=utf-8\">" << std::endl;
    *output << "</head>" << std::endl;
    *output << "<body>" << std::endl;
  }
  //@tag: [comment] The value of this tag is inserted at the beginning of the output document, enclosed between commentary characters corresponding to the output format. Argument tag overrides object tag.
  TString comment; if(tags.getTagString("comment",comment)) *output << "<!-- " << comment << " -->" << std::endl;
  //@tag: [preamble.html] The value of this tag is inserted before the beginning of the actual table. Argument tag overrides object tag.
  TString preamble = ""; if(tags.getTagString("preamble.html",preamble)) *output << preamble << std::endl;
  //@tag: [tableStyle.html] The value of this tag specifies the html table style, default is "border-collapse: collapse". Argument tag overrides object tag.
  *output << "<table style=\""+this->getTagStringDefault("tableStyle.html","border-collapse: collapse")+"\">" << std::endl;
  for(unsigned int i=0; i<this->nrows; i++){
    *output << "<tr>";
    for(unsigned int j=0; j<this->ncols; j++){
      *output << this->getEntryHTML(i,j) << " ";
    }
    *output << "</tr>" << std::endl;
  }
  *output << "</table>" << std::endl;
  if(standalone){
    *output << "</body>" << std::endl;
    *output << "</html>" << std::endl;
  }
  return true;
}

bool TQTable::printLaTeX(std::ostream* output, TQTaggable tags){
  // print table data to the given stream. format is LaTeX and can specified by style tags.
  tags.setGlobalOverwrite(false);
  tags.importTags(this);
  //@tag:[env] Specifies the table environment used for LaTex output, default is "tabular". Other options include "longtable", for example. Arugment tag overrides object tag.
  TString env = tags.getTagStringDefault("env","tabular");
  if(!output || !this->data) return false;
  // for tag documentation see printHTML
  TString comment; if(tags.getTagString("comment",comment)) *output << "%%% " << comment << std::endl;
  // for tag documentation see printHTML
  bool standalone = tags.getTagBoolDefault("standalone",false);
  //@tag:[layout] Sets the layout of the LaTeX document. The default "standalone" will create a document of clas standalone, other values lead to document class article. If the layout is set to "adjustbox", an adjustbox environment is created around the table. Arugment tag overrides object tag.
  TString layout = tags.getTagStringDefault("layout","standalone");
  layout.ToLower();
  if(standalone){
    *output << "\\documentclass{";
    if(layout == "standalone") *output << "standalone";
    else *output << "article";
    *output << "}" << std::endl;
    *output << "\\usepackage{luatex85}" << std::endl;
    *output << "\\usepackage[table]{xcolor}" << std::endl;
    *output << "\\usepackage{multirow}" << std::endl;
    *output << "\\usepackage{rotating}" << std::endl;
    *output << "\\usepackage{pifont}" << std::endl;
    if(env == "longtable"){
      *output << "\\usepackage{longtable}" << std::endl;
    }
    if(layout == "adjustbox") *output << "\\usepackage{adjustbox}" << std::endl;
    *output << "\\newcommand{\\xmark}{\\ding{55}}" << std::endl;
    *output << "\\providecommand\\rotatecell[2]{\\rotatebox[origin=c]{#1}{#2}}" << std::endl;
  } else {
    *output << "\\providecommand{\\xmark}{{\\sffamily \\bfseries X}}" << std::endl;;
    *output << "\\providecommand\\rotatecell[2]{\\rotatebox[origin=c]{#1}{#2}}" << std::endl;
  }
  //@tag:[preamble.latex] Defines additions to the LaTeX preamble (inserted before \begin{document}). Arugment tag overrides object tag.
  TString preamble = ""; if(tags.getTagString("preamble.latex",preamble)) *output << preamble << std::endl;
  if(standalone){
    *output << "\\begin{document}" << std::endl;
    if(layout == "adjustbox") *output << "\\begin{adjustbox}";
  }
  *output << "\\begin{" << env << "}{";
  for(unsigned int j=0; j<this->ncols; j++){
    if(j < this->vlines.size() && this->vlines[j] > 0) *output << TQStringUtils::repeat("|",this->vlines[j]);
    *output << " "+this->getColAlign(j)+" ";
  }
  if(this->ncols < this->vlines.size() && this->vlines[this->ncols] > 0) *output << TQStringUtils::repeat("|",this->vlines[this->ncols]);
  *output << "}" << std::endl;
  for(unsigned int i=0; i<this->nrows; i++){
    if(i < this->hlines.size() && this->hlines[i] > 0){
      for(int j=0; j<this->hlines[i]; j++){
        *output << "\\hline";
      }
      *output << std::endl;
    } 

    bool first = true;
    unsigned int multicol = 0;
    for(unsigned int j=0; j<this->ncols; j++){
      TQTaggable* entry = this->getEntryInternal(i,j,false);
      if(multicol == 0){
        if(!first) *output << " & ";
        first = false;
        if(entry){
          multicol = entry->getTagIntegerDefault("multicol",0);
          int multirow;
          bool doMultirow = entry->getTagInteger("multirow",multirow);
          if(doMultirow) *output << "\\multirow{" << multirow << "}{*}{";
          if(multicol < 0) multicol = this->ncols-j;
          if(multicol > 1){
            *output << "\\multicolumn{" << multicol << "}{" << 
              TQStringUtils::repeat("|",j<this->vlines.size()?this->vlines[j]:0) << 
              this->getColAlign(j) <<
              TQStringUtils::repeat("|",j+multicol<this->vlines.size()?this->vlines[j+multicol]:0) << 
              "}{" << this->getEntryLaTeX(i,j) << "}";
          } else {
            *output << this->getEntryLaTeX(i,j);
          }
          if(doMultirow) *output << "}";
        }
      }
      if(multicol > 0) multicol--;
    }
    if(i+1 < this->nrows)
      *output << "\\tabularnewline" << std::endl;
    else 
      *output << std::endl;
  }
  if(this->nrows < this->hlines.size() && this->hlines[this->nrows] > 0){
    *output << "\\tabularnewline" << std::endl;
    for(int j=0; j<this->hlines[this->nrows]; j++){
      *output << "\\hline";
    }
    *output << std::endl;
  } 
  *output << "\\end{" << env << "}" << std::endl;
  if(standalone){
    if(layout == "adjustbox") *output << "\\end{adjustbox}";
    *output << "\\end{document}" << std::endl;
  }
  return true;
}

bool TQTable::printPlain(std::ostream* output, TQTaggable tags){
  // print table data to the given stream. format is plain ascii/unicode and can specified by style tags.
  tags.setGlobalOverwrite(false);
  tags.importTags(this);
  //@tag: [sep] Defines the seperator between entries for plain and CSV printing. Defaults are " " (plain) and "," (CSV). Arugment tag overrides object tag.
  TString sep = tags.getTagStringDefault("sep"," ");
  //@tag: [cellWidth] Defines the width of cells for plain printing in characters for plain printing, default: 10. If this tag is seet and adjustColWidth is set to true, the value of this tag is used as the maximum column/cell width. Arugment tag overrides object tag.
  int cellwidth = tags.getTagIntegerDefault("cellWidth",10);
  //@tag: [adjustColWidth] If set to true the cell width is automatically determined from the content (default: false). The maximum cell width can be set via the tag cellWidth. Arugment tag overrides object tag.
  bool adjustColWidth = tags.getTagBoolDefault("adjustColWidth",false);
  if(!output || !this->data) return false;
  //for tag documentation see printHTML
  TString comment; if(tags.getTagString("comment",comment)) *output << "# " << comment << std::endl;
  //@tag: [preamble.plain] Preamble text inserted before the table in plain text printing. Arugment tag overrides object tag.
  TString preamble = ""; if(tags.getTagString("preamble.plain",preamble)) *output << preamble << std::endl;
  std::vector<int> cellwidths;
  int linewidth = 0;
  //@tag: [allowUnicode] Selects if unicode characters are allowed in plain printing not (default: true). Arugment tag overrides object tag.
  bool unicode = tags.getTagBoolDefault("allowUnicode",true);
  for(unsigned int j=0; j<this->ncols; j++){
    if(adjustColWidth){
      int width = 0;
      for(unsigned int i=0; i<this->nrows; i++){
        TString entry = this->getEntryPlain(i,j);
        width = std::max(TQStringUtils::getWidth(entry),width);
      }
      //for tag documentation see above
      cellwidth = std::min(width,tags.getTagIntegerDefault("cellWidth",width));
    }
    cellwidths.push_back(cellwidth);
    linewidth += cellwidth;
  }
  for(unsigned int i=0; i<this->hlines.size(); i++){
    if(hlines[i] > 0) linewidth += hlines[i]+sep.Length();
  }
  for(unsigned int i=0; i<this->nrows; i++){
    if(i < this->hlines.size() && this->hlines[i] > 0){
      for(int j=0; j<this->hlines[i]; j++){
        *output << TQStringUtils::repeat("-",linewidth) << std::endl;
      }
    }
    for(unsigned int j=0; j+1<this->ncols; j++){
      if(j < this->vlines.size() && this->vlines[j] > 0) *output << TQStringUtils::repeat("|",this->vlines[j]) << sep;
      *output << TQStringUtils::fixedWidth(this->getEntryPlain(i,j,unicode),
                                           cellwidths[j],this->getColAlign(j)) << sep;
    }
    if(this->ncols-1 < this->vlines.size() && this->vlines[this->ncols-1] > 0) *output << TQStringUtils::repeat("|",this->vlines[this->ncols-1]) << sep;
    *output << TQStringUtils::fixedWidth(this->getEntryPlain(i,this->ncols-1,unicode),cellwidths[this->ncols-1],this->getColAlign(this->ncols-1)) << std::endl;
    if(this->ncols < this->vlines.size() && this->vlines[this->ncols] > 0) *output << TQStringUtils::repeat("|",this->vlines[this->ncols]) << sep;
  }
  return true;
}


TQTaggable* TQTable::getEntryInternal(unsigned int i, unsigned int j, bool create){
  // retrieve the object representing the table entry at row i and column j
  // if create is true, missing entries will be created
  DEBUGclass("trying to access entry %d/%d (create=%d)",i,j,(int)create);
  if(i < 0 || j < 0) return NULL;
  if(i>=this->nrows || j>=this->ncols){
    if(create && this->autoExpand) {
      if(!this->expand(std::max(this->nrows,i+1),std::max(this->ncols,j+1))) return NULL;
    } 
  } 
  if(i<this->nrows && j<this->ncols){
    if(this->data[i*this->ncols + j]){
      return this->data[i*this->ncols + j];
    } else if(create){
      TQTaggable* newEntry = new TQTaggable();
      this->data[i*this->ncols + j] = newEntry;
      return newEntry;
    }
  }
  return NULL;
}


TQTaggable* TQTable::getEntryInternal(unsigned int i, unsigned int j) const {
  // retrieve the object representing the table entry at row i and column j
  if(i<this->nrows && j<this->ncols){
    if(this->data[i*this->ncols + j]){
      return this->data[i*this->ncols + j];
    } 
  }
  return NULL;
}
 
int TQTable::getEntryInteger(unsigned int i, unsigned int j, bool sanitizeString){
  // retrieve the table entry at row i and column j
  // as an integer. If sanitizeString is true all blanks (whitespaces) are removed
  // from the string representation before the conversion to int.
  int retval = std::numeric_limits<int>::quiet_NaN();
  TQTaggable* entry = this->getEntryInternal(i,j);
  if(!entry) return retval;
  //@tag: [content.value] This tag contains the numerical content of the corresponding cell.
  if(entry->getTagInteger("content.value",retval)) return retval;
  TString content = this->formatEntryContents(entry,"ascii");
  if (sanitizeString) TQStringUtils::removeAll(content, TQStringUtils::getAllBlanks(), TString::ECaseCompare::kIgnoreCase,-1);
  return content.Atoi();
}

double TQTable::getEntryDouble(unsigned int i, unsigned int j, bool sanitizeString){
  // retrieve the table entry at row i and column j
  // as a double. If sanitizeString is true all blanks (whitespaces) are removed
  // from the string representation before the conversion to double.
  double retval = std::numeric_limits<double>::quiet_NaN();
  TQTaggable* entry = this->getEntryInternal(i,j);
  if(!entry) return retval;
  //@tag: [content.value] This tag contains the numerical content of the corresponding cell.
  if(entry->getTagDouble("content.value",retval)) return retval;
  TString content = this->formatEntryContents(entry,"ascii");
  if (sanitizeString) TQStringUtils::removeAll(content, TQStringUtils::getAllBlanks(), TString::ECaseCompare::kIgnoreCase,-1);
  return content.Atof();
}

TString TQTable::getEntry(unsigned int i, unsigned int j, TString format){
  // retrieve the table entry at row i and column j
  // in the given format
  format.ToLower();
  if(format=="latex") return this->getEntryLaTeX(i,j);
  if(format=="html") return this->getEntryHTML(i,j);
  if(format=="unicode") return this->getEntryUnicode(i,j);
  if(format=="plain") return this->getEntryUnicode(i,j);
  if(format=="ascii") return this->getEntryASCII(i,j);
  if(format=="verbatim") return this->getEntryVerbatim(i,j);
  if(format=="csv") return this->getEntryASCII(i,j);
  return this->getEntryASCII(i,j);
}

TString TQTable::getEntryPlain(unsigned int i, unsigned int j, bool allowUnicode){
  // retrieve the table entry at row i and column j
  // in plain ascii/unicode format
  TQTaggable* entry = this->getEntryInternal(i,j);
  TString content = entry ? entry->getTagStringDefault( (allowUnicode? "content.unicode":"content.ascii"), this->formatEntryContents(entry,allowUnicode ? "unicode" : "ascii")) : this->formatEntryContents(entry,allowUnicode ? "unicode" : "ascii");
  if(content.IsNull()) return content;
  if(allowUnicode){
    TString textcolor = ""; 
    //@tag: [textcolor] Defines the text color for LaTex, HTML and plain (unicode) printing of the cell entry this tag is set on. In case of HTML/LaTex the value needs to be a valid color for the desired output type. For plain (unicode) printing "blue", "red", "yellow" and "pink" are available.
    //@tag: [cellcolor] Defines the cell's background color for LaTex and HTML printing of the cell entry this tag is set on. The value needs to be a valid color for the desired output type.
    if(entry->getTagString("textcolor",textcolor)){
      if(textcolor == "blue") content = TQStringUtils::makeBoldBlue(content);
      if(textcolor == "red") content = TQStringUtils::makeBoldRed(content);
      if(textcolor == "yellow") content = TQStringUtils::makeBoldYellow(content);
      if(textcolor == "pink") content = TQStringUtils::makeBoldPink(content);
    //@tag: [bold] The entry this tag is set on is printed in bold font if this object tag is set to true. Supported for HTML,LaTeX. For plain printing, the "textcolor" object tag may not be set.
    } else if(entry->getTagBoolDefault("bold",false)){
      content = TQStringUtils::makeBoldWhite(content);
    }
  }
  if(entry){
    //@tag: [prefixText.plain,suffixText.plain] This entry tag determines the text to be prepended/appended to the cell entry for plain text output
    TString prefixText; if(entry->getTagString("prefixText.plain",prefixText)) content.Prepend(prefixText);
    TString suffixText; if(entry->getTagString("suffixText.plain",suffixText)) content.Append(suffixText);
  }
  return content;
}

TString TQTable::getEntryVerbatim(unsigned int i, unsigned int j){
  // retrieve the table entry at row i and column j
  // as verbatim text
  TQTaggable* entry = this->getEntryInternal(i,j);
  //@tag: [content.verbatim] This tag contains the verbatim content of the corresponding cell.
  if(entry) return entry->getTagStringDefault("content.verbatim","");
  return "";
}

TString TQTable::getEntryUnicode(unsigned int i, unsigned int j){
  // retrieve the table entry at row i and column j
  // in Unicode-Format
  return this->getEntryPlain(i,j,true);
}

TString TQTable::getEntryASCII(unsigned int i, unsigned int j){
  // retrieve the table entry at row i and column j
  // in ASCII-Format
  return this->getEntryPlain(i,j,false);
}

TString TQTable::getEntryHTML(unsigned int i, unsigned int j){
  // retrieve the table entry at row i and column j
  // in HTML format
  TQTaggable* entry = this->getEntryInternal(i,j);
  TString tag = (i==0 ? "th" : "td");
  TString content = entry ? entry->getTagStringDefault("content.html", TQTable::formatEntryContents(entry, "html")) : TQTable::formatEntryContents(entry, "html");
  TString style = "";
  style.Append("text-align:"); style.Append(this->getColAlignHTML(j)); style.Append("; ");
  if(i < this->hlines.size() && this->hlines[i] > 0){
    if(this->hlines[i] == 1) style.Append(" border-top: 1px black solid; ");
    else style.Append(" border-top: 3px black double; ");
  }
  if(j < this->vlines.size() && this->vlines[j] > 0){
    if(this->vlines[j] == 1) style.Append(" border-left: 1px black solid; ");
    else style.Append(" border-left: 3px black double; ");
  }
  TString tooltip = ""; 
  if(entry){
    TString textcolor = ""; if(entry->getTagString("textcolor",textcolor)) style.Append(" color:"+textcolor+"; ");
    TString cellcolor = ""; if (entry->getTagString("cellcolor",cellcolor)) style.Append(" background-color: "+cellcolor+"; ");
    //@tag: [italic,bold,smallcaps] These entry tags determine if the cell content is displayed in italics/bold/smallcaps in HTML and LaTeX output.
    if(entry->getTagBoolDefault("italic",false)) style.Append(" font-shape:italic; ");
    if(entry->getTagBoolDefault("bold",false)) style.Append(" font-weight:bold; ");
    if(entry->getTagBoolDefault("smallcaps",false)) style.Append(" font-variant:small-caps; ");
    //@tag: [allowlinewrap] This entry/object tag controlls if cell contents may be wrapped over multiple lines in HTML output, default: false. Entry tag overrides object tag.
    bool allowlinewrap = entry->getTagBoolDefault("allowlinewrap",this->getTagBoolDefault("allowlinewrap",false));
    //@tag: [style.padding.vertical,~.horizontal] These object tags determine the left and right / top and bottom paddings of the table in HTML output, default: vertical: "0px", horizontal: "5px".
    TString vpadding = this->getTagStringDefault("style.padding.vertical", "0px");
    TString hpadding = this->getTagStringDefault("style.padding.horizontal", "5px");
    style.Append(" padding-left:"+hpadding +"; padding-right: "+hpadding+ "; padding-top:"+vpadding+"; padding-bottom:"+vpadding+";");
    if(!allowlinewrap) style.Append(" white-space: nowrap; ");
    //@tag: [tooltip] This entry tag is added to the cell in HTML output as a tool tip.
    if(entry->getTagString("tooltip",tooltip)){ tooltip.Prepend("title=\""); tooltip.Append("\" "); };
  }
  if(entry){
    //@tag: [prefixText.html,suffixText.html] This entry tag determines the text to be prepended/appended to the cell entry for HTML output
    TString prefixText; if(entry->getTagString("prefixText.html",prefixText)) content.Prepend(prefixText);
    TString suffixText; if(entry->getTagString("suffixText.html",suffixText)) content.Append(suffixText);
  }
  content.Append("</"+tag+">");
  content.Prepend("<"+tag+" "+tooltip+"style=\""+style+"\">"); 
  return content;
}


TString TQTable::makeExpSuffix(int exponent, const TString& format, bool useSIsuffix){
  // create the exponential suffix associated with the given exponent
  // in the given format ('ascii', 'unicode', 'latex', 'html', ...)
  // if useSIsuffix is true, SI suffices (... m, k, M, G, ...) will be used instead
  if(useSIsuffix) return TQStringUtils::getSIsuffix(exponent,format);
  if(format == "unicode"){
    return "*10"+TQStringUtils::makeUnicodeSuperscript(TString::Format("%d",exponent));
  }
  //@tag: [format.expsuffix.<format>] This object tag determines how exponential notation is represented for format <format>. The default pattern is "×10^%d".
  TString pattern = this->getTagStringDefault("format.expsuffix."+format,"×10^%d");
  return TString::Format(pattern.Data(),exponent);
}

void TQTable::setup(){
  // initialize some sensible formatting defaults
  // 
  // the formatting is saved as tags on the TQTable object and is accessed on writing the data
  // the values can be accessed and changed easily 
  //
  // the tag naming convention is as follows:
  // formatting is controled by
  // format.$OBJECT.$FORMAT
  // where $OBJECT refers to the object to be formatted, e.g.
  // expsuffix  : exponential suffix, e.g. '*10^7'
  // integer  : integer numbers
  // double  : floating point numbers
  // integerWithSuffix : integer numbers with suffix, e.g. 3*10^7
  // doubleWithSuffix : floating point numbers with suffix, e.g. 3.5*10^7
  // doubleAndUncertainty : floating point numbers with uncertainty, e.g. 3.5 +/- 0.2
  // doubleAndUncertaintyWithSuffix : floating point numbers with uncertainty and suffix, e.g. (3.5 +/- 0.2)*10^7
  // doubleWithSuffixAndUncertaintyWithSuffix : floating point numbers with uncertainty and suffix, e.g. 3.5*10^7 +/- 0.2*10^7
  // doubleAndRelativeUncertainty : floating point numbers with relative uncertainty, e.g. 3.5 +/- 5%
  // doubleWithSuffixAndRelativeUncertainty : floating point numbers with suffix and relative uncertainty, e.g. 3.5*10^7 +/- 5%
  // symbols are controled by
  // symbols.$SYMBOL.$FORMAT
  // where $SYMBOL refers to the symbol to be displayed, e.g.,
  // posInf : symbol used for positively infinite numbers
  // negInf : symbol used for negatively infinite numbers
  // NaN : symbol used for invalid numerical expressions (NaN)
  // in each case, $FORMAT refers to any of the supported output formats, that is
  // latex, html, ascii, unicode
  DEBUGclass("setup called");
  //@tag: [symbols.<symbol>.<format>] Symbol to be used for format <format>, when <symbol> is encountered. Defaults are "Nan", "inf" and "-inf". TQTable::setup() sets format specific symbols, which are (LaTeX/HTML/unicode/ASCII defaults): NaN ("\\xmark" / "&#x2717;" / U+2717 / "NaN"), posInf ("\\ensuremath{\\infty}" / "&infin;" / U+221E / "inf"), negInf ("-\\ensuremath{\\infty}" / "-&infin;" / "-"+ U+221E / "-inf")
  this->setTagString("format.expsuffix.latex", "\\times 10^{%d}");
  this->setTagString("format.integer.latex",  "\\ensuremath{%d}");
  this->setTagString("format.double.latex",  "\\ensuremath{%.$(ndigits)f}");
  //@tag: [format.<type>WithSuffix.<format>, format.doubleWithSuffixAndRelativeUncertainty.<format>, format.doubleWithSuffixAndUncertaintyWithSuffix.<format>] This object tag controls the format how numbers with suffixes are printed. <format> is any of "latex","html","ascii","unicode". <type> is either "double" or "integer". TQTable::setup() sets sensible format settings, if it is not called, defaults may vary.
  this->setTagString("format.integerWithSuffix.latex", "\\ensuremath{%d%s}");
  this->setTagString("format.doubleWithSuffix.latex", "\\ensuremath{%.$(ndigits)f%s}");
  this->setTagString("format.integerAndUncertainty.latex", "\\ensuremath{%d\\pm %d}");
  this->setTagString("format.doubleAndUncertainty.latex", "\\ensuremath{%.$(ndigits)f\\pm %.$(ndigits)f}");
  this->setTagString("format.doubleAndUncertaintyWithSuffix.latex", "\\ensuremath{(%.$(ndigits)f\\pm %.$(ndigits)f)%s}");
  this->setTagString("format.doubleWithSuffixAndUncertaintyWithSuffix.latex","\\ensuremath{%.$(ndigits)f%s\\pm %.$(ndigits)f%s}");
  this->setTagString("format.doubleAndRelativeUncertainty.latex", "\\ensuremath{%.$(ndigits)f\\pm %.$(ndigits)f%%}");
  this->setTagString("format.doubleWithSuffixAndRelativeUncertainty.latex", "\\ensuremath{%.$(ndigits)f%s\\pm %.$(ndigits)f%%}");

  this->setTagString("symbols.posInf.latex","\\ensuremath{\\infty}");
  this->setTagString("symbols.negInf.latex","-\\ensuremath{\\infty}");
  this->setTagString("symbols.NaN.latex","\\xmark");

  this->setTagString("format.integer.unicode",  "%d");
  this->setTagString("format.double.unicode",  "%.$(ndigits)f");
  this->setTagString("format.integerWithSuffix.unicode", "%d%s");
  this->setTagString("format.doubleWithSuffix.unicode", "%.$(ndigits)f%s");
  this->setTagString("format.integerAndUncertainty.unicode", "%d ± %d");
  this->setTagString("format.doubleAndUncertainty.unicode", "%.$(ndigits)f ± %.$(ndigits)f");
  this->setTagString("format.doubleAndUncertaintyWithSuffix.unicode", "(%.$(ndigits)f ± %.$(ndigits)f)%s");
  this->setTagString("format.doubleWithSuffixAndUncertaintyWithSuffix.unicode","%.$(ndigits)f%s ± %.$(ndigits)f%s");
  this->setTagString("format.doubleAndRelativeUncertainty.unicode", "%.$(ndigits)f ± %.$(ndigits)f%%");
  this->setTagString("format.doubleWithSuffixAndRelativeUncertainty.unicode", "%.$(ndigits)f%s ± %.$(ndigits)f%%");
 
  this->setTagString("symbols.posInf.unicode","∞");
  this->setTagString("symbols.negInf.unicode","-∞");
  this->setTagString("symbols.NaN.unicode","✘");

  this->setTagString("format.integer.ascii",  "%d");
  this->setTagString("format.double.ascii",  "%.$(ndigits)f");
  this->setTagString("format.integerWithSuffix.ascii", "%d%s");
  this->setTagString("format.doubleWithSuffix.ascii", "%.$(ndigits)f%s");
  this->setTagString("format.integerAndUncertainty.ascii", "%d +/- %d");
  this->setTagString("format.doubleAndUncertainty.ascii", "%.$(ndigits)f +/- %.$(ndigits)f");
  this->setTagString("format.doubleAndUncertaintyWithSuffix.ascii", "(%.$(ndigits)f +/- %.$(ndigits)f)%s");
  this->setTagString("format.doubleWithSuffixAndUncertaintyWithSuffix.ascii","%.$(ndigits)f%s +/- %.$(ndigits)f%s");
  this->setTagString("format.doubleAndRelativeUncertainty.ascii", "%.$(ndigits)f +/- %.$(ndigits)f%%");
  this->setTagString("format.doubleWithSuffixAndRelativeUncertainty.ascii", "%.$(ndigits)f%s +/- %.$(ndigits)f%%");
 
  this->setTagString("symbols.posInf.ascii","inf");
  this->setTagString("symbols.negInf.ascii","-inf");
  this->setTagString("symbols.NaN.ascii","NaN");

  this->setTagString("format.expsuffix.html", "&times;10<sup>%d</sup>");
  this->setTagString("format.integer.html",  "%d");
  this->setTagString("format.double.html",  "%.$(ndigits)f");
  this->setTagString("format.integerWithSuffix.html", "%d%s");
  this->setTagString("format.doubleWithSuffix.html", "%.$(ndigits)f%s");
  this->setTagString("format.integerAndUncertainty.html", "%d &plusmn; %d");
  this->setTagString("format.doubleAndUncertainty.html", "%.$(ndigits)f &plusmn; %.$(ndigits)f");
  this->setTagString("format.doubleAndUncertaintyWithSuffix.html", "(%.$(ndigits)f &plusmn; %.$(ndigits)f)%s");
  this->setTagString("format.doubleWithSuffixAndUncertaintyWithSuffix.html","%.$(ndigits)f%s &plusmn; %.$(ndigits)f%s");
  this->setTagString("format.doubleAndRelativeUncertainty.html", "%.$(ndigits)f &plusmn; %.$(ndigits)f%%");
  this->setTagString("format.doubleWithSuffixAndRelativeUncertainty.html", "%.$(ndigits)f%s &plusmn; %.$(ndigits)f%%");
 
  this->setTagString("symbols.posInf.html","&infin;");
  this->setTagString("symbols.negInf.html","-&infin;");
  this->setTagString("symbols.NaN.html","&#x2717;");
}


TString TQTable::formatEntryContents(TQTaggable* entry, const TString& format){
  // format the entry contents in the given object according to the predefined formatting 
  // this function will make use of the formatting tags set previously
  if(!entry) return "";
  if(entry->hasTag("content.value")){
    if(entry->tagIsOfTypeInteger("content.value")){
      int val = entry->getTagIntegerDefault("content.value",0);
      int exponent = val == 0 ? 0 : floor(log10((double)val)/3)*3;
      //@tag: [format.useExponentialNotation] This object tag determines if exponential notation should be used, default: false.
      if((exponent != 0) && (this->getTagBoolDefault("format.useExponentialNotation",false))){
        DEBUGclass("using exponential notation (int): exponent=%d", exponent);
        //@tag: [format.nSignificantDigits] This object tag determines the number of significant digits shown in the table, default: 2
        int nDigits = entry->getTagIntegerDefault("format.nSignificantDigits",this->getTagIntegerDefault("format.nSignificantDigits",5));
        double roundVal = TQUtils::round(double(val)/pow(10,exponent),nDigits);
        //@tag: [format.useSIsuffix] If this object tag is set to true, SI prefixes (m,k,M,...) are used as suffixes for numbers.
        TString suffix = this->makeExpSuffix(exponent,format,this->getTagBoolDefault("useSIsuffix",false));
        TString s = this->getTagStringDefault("format.doubleWithSuffix."+format,"%f%s");
        return TString::Format(s.Data(),roundVal,suffix.Data());
      } else {
        DEBUGclass("using standard notation (int): exponent=%d",exponent);
        //@tag: [format.integer] This object tag determines the standard format of integers, default: "%d"
        TString s = this->getTagStringDefault("format.integer."+format,"%d");
        return TString::Format(s.Data(),val);
      } 
    } else {
      double val = entry->getTagDoubleDefault("content.value",0);
      int exponent = floor(log10(val)/3)*3;
      //for tag documentation see above
      int nDigits = entry->getTagIntegerDefault("format.nSignificantDigits",this->getTagIntegerDefault("format.nSignificantDigits",5));
      TString ndig = TString::Format("%d",nDigits);
      double roundVal = TQUtils::round(val/pow(10,exponent),nDigits);
      //@tag: [content.uncertainty] This entry tag contains the uncertainty of the numerical content of the corresponding cell.
      if(entry->hasTag("content.uncertainty")){
        double unc = entry->getTagDoubleDefault("content.uncertainty",0);
        if(val != val) return this->getTagStringDefault("symbols.NaN."+format,"NaN");
        else if(val >= std::numeric_limits<double>::infinity()) return this->getTagStringDefault("symbols.posInf."+format,"inf");
        else if(val <= -std::numeric_limits<double>::infinity()) return this->getTagStringDefault("symbols.negInf."+format,"-inf");
        else if(val == 0) return "0";
        else if(this->getTagBoolDefault("format.pdg",false)){
          if((exponent != 0) && (this->getTagBoolDefault("format.useExponentialNotation",false))){
            return TQStringUtils::formatValueErrorPDG(val,unc,exponent,format);
          } else {
            return TQStringUtils::formatValueErrorPDG(val,unc,0,format);
          }
        } else { 
          if((exponent != 0) && (this->getTagBoolDefault("format.useExponentialNotation",false))){
            DEBUGclass("using exponential notation (float): exponent=%d",exponent);
            double roundUnc = TQUtils::round(unc/pow(10,exponent),nDigits);
            TString suffix = this->makeExpSuffix(exponent,format,this->getTagBoolDefault("useSIsuffix",false));
            //@tag: [format.useRelativeUncertainties] If this object tag is set to true, relative uncertainties are shown (entry value needs to be non-integer!).
            if(this->getTagBoolDefault("format.useRelativeUncertainties",false)){
              TString s = this->getTagStringDefault("format.doubleWithSuffixAndRelativeUncertainty."+format,"%.$(ndigits)f%s +/-%.$(ndigits)f%%");
              s.ReplaceAll("$(ndigits)",ndig);
              return TString::Format(s.Data(),roundVal,suffix.Data(),unc/val*100);
              //@tag: [format.useCommonSuffix] If this object tag is set to true, value and uncertainty are shown with a single suffix (e.g. (a +/- b) fb^{-1}). The format is set via format.doubleAndUncertaintyWithSuffix.<format> .
            } else if(this->getTagBoolDefault("format.useCommonSuffix",true)){
              TString s = this->getTagStringDefault("format.doubleAndUncertaintyWithSuffix."+format,"(%.$(ndigits)f+/-%.$(ndigits)f)%s");
              s.ReplaceAll("$(ndigits)",ndig);
              return TString::Format(s.Data(),roundVal,roundUnc,suffix.Data());
            } else {
              TString s = this->getTagStringDefault("format.doubleWithSuffixAndUncertaintyWithSuffix."+format,"%.$(ndigits)f%s+/-%.$(ndigits)f%s");
              s.ReplaceAll("$(ndigits)",ndig);
              return TString::Format(s.Data(),roundVal,suffix.Data(),roundUnc,suffix.Data());
            }
          } else {
            DEBUGclass("using standard notation (float): exponent=%d",exponent);
            if(this->getTagBoolDefault("format.useRelativeUncertainties",false)){
              TString s = this->getTagStringDefault("format.doubleAndRelativeUncertainty."+format,"%.$(ndigits)f+/-%.$(ndigits)f%%");
              s.ReplaceAll("$(ndigits)",ndig);
              return TString::Format(s.Data(),val,unc/val*100);
            } else {
              TString s = this->getTagStringDefault("format.doubleAndUncertainty."+format,"%.$(ndigits)f+/-%.$(ndigits)f");
              s.ReplaceAll("$(ndigits)",ndig);
              return TString::Format(s.Data(),val,unc);
            }
          }
        }
      } else {
        if((exponent != 0) && (this->getTagBoolDefault("format.useExponentialNotation",false))){
          TString suffix = this->makeExpSuffix(exponent,format,this->getTagBoolDefault("useSIsuffix",false));
          TString s = this->getTagStringDefault("format.doubleWithSuffix."+format,"%.$(ndigits)f%s");
          s.ReplaceAll("$(ndigits)",ndig);
          return TString::Format(s.Data(),roundVal,suffix.Data());
        } else {
          TString s = this->getTagStringDefault("format.double."+format,"%.$(ndigits)f");
          s.ReplaceAll("$(ndigits)",ndig);
          return TString::Format(s.Data(),val);
        } 
      }
    }
  } 
  return entry->getTagStringDefault("content."+format,"");
}



TString TQTable::getEntryLaTeX(unsigned int i, unsigned int j){
  // retrieve the table entry at row i and column j
  // in LaTeX format
  TQTaggable* entry = this->getEntryInternal(i,j);
  TString content = entry ? entry->getTagStringDefault("content.latex", TQTable::formatEntryContents(entry, "latex")) : TQTable::formatEntryContents(entry, "latex");
  if(content.IsNull()) return "";
  if(entry){
    //@tag: [prefixText.latex,suffixText.latex] This entry tag determines the text to be prepended/appended to the cell entry for LaTeX output
    TString prefixText; if(entry->getTagString("prefixText.latex",prefixText)) content.Prepend(prefixText);
    TString suffixText; if(entry->getTagString("suffixText.latex",suffixText)) content.Append(suffixText);
  }
  if(entry->hasTag("textcolor")){
    content.Append("}");
    content.Prepend("}{");
    content.Prepend(entry->getTagStringDefault("textcolor","black"));
    content.Prepend("\\textcolor{");
  }
  if (entry->hasTag("cellcolor")) {
    content.Prepend("}");
    content.Prepend(entry->getTagStringDefault("cellcolor","white"));
    content.Prepend("\\cellcolor{");
  }
  if(entry->getTagBoolDefault("italic",false)){
    content.Prepend("\\textit{");
    content.Append("}");
  }
  if(entry->getTagBoolDefault("bold",false)){
    content.Prepend("\\textbb{");
    content.Append("}");
  }
  if(entry->getTagBoolDefault("smallcaps",false)){
    content.Prepend("\\textsc{");
    content.Append("}");
  }
  if(entry->hasTag("rotate")){
    content.Prepend("}{");
    content.Prepend(entry->getTagStringDefault("rotate","0"));
    content.Prepend("\\rotatecell{");
    content.Append("}");
  }
  return content;
}


int TQTable::setAllEntriesWhere(const TString& searchCol, const TString& searchVal, const TString& setCol, const TString& setVal, const TString& searchFormat, const TString& setFormat){
  // database function
  // find all rows where searchCol takes a value matching searchVal in searchFormat 
  // in these rows, set setCol to setVal in setFormat
  unsigned int col1 = findColumn(searchCol);
  if(col1 < 0 || col1 > this->ncols){
    ERROR("unable to find column '%s'",searchCol.Data());
  }
  unsigned int col2 = findColumn(setCol);
  if(col2 < 0 || col2 > this->ncols){
    ERROR("unable to find column '%s'",setCol.Data());
  }
  int count = 0;
  for(unsigned int i=1; i<this->nrows; i++){
    TString val = this->getEntry(i,col1,searchFormat);
    if(TQStringUtils::matches(val,searchVal)){
      this->setEntry(i,col2,setVal,setFormat);
      count++;
    }
  }
  return count;
}
 


bool TQTable::setEntry(unsigned int i, unsigned int j, const TString& content, const TString& format){
  // set the table entry at row i and column j to the given content
  // input format can be specified with last argument and will be guessed automatically if left empty
  // contents will automatically be converted into all supported output formats
  // if you do not want this or want to edit individual representations
  // without afflicting other formats, please use
  // TQTable::setProperty( ... )
  DEBUGclass("setting content of cell %d/%d to '%s' (format=%s)",(int)i,(int)j,content.Data(),format.Data());
  TQTaggable* entry = this->getEntryInternal(i,j,true);
  if(!entry) return false;
  this->setContent(entry,content,format);
  return true;
}

bool TQTable::setEntry(unsigned int i, unsigned int j, const char* content, const TString& format){
  // set the table entry at row i and column j to the given content
  // input format can be specified with last argument and will be guessed automatically if left empty
  // contents will automatically be converted into all supported output formats
  // if you do not want this or want to edit individual representations
  // without afflicting other formats, please use
  // TQTable::setProperty( ... )
  DEBUGclass("setting content of cell %d/%d to '%s' (format=%s)",(int)i,(int)j,content,format.Data());
  TQTaggable* entry = this->getEntryInternal(i,j,true);
  if(!entry) return false;
  TString tmp(content);
  this->setContent(entry,tmp,format);
  return true;
}


bool TQTable::setEntryValue(unsigned int i, unsigned int j, double content){
  // set the table entry at row i and column j to the given numerical content
  TQTaggable* entry = this->getEntryInternal(i,j,true);
  if(!entry) return false;
  entry->setTagDouble("content.value",content);
  return true;
}

double TQTable::getEntryValue(unsigned int i, unsigned int j, double defaultval){
  // get the table entry at row i and column j
  TQTaggable* entry = this->getEntryInternal(i,j,true);
  if(!entry) return std::numeric_limits<double>::quiet_NaN();
  return entry->getTagDoubleDefault("content.value",defaultval);
}

bool TQTable::setEntryValueAndUncertainty(unsigned int i, unsigned int j, double value, double Uncertainty){
  // set the table entry at row i and column j to the given numerical content
  TQTaggable* entry = this->getEntryInternal(i,j,true);
  if(!entry) return false;
  entry->setTagDouble("content.value",value);
  entry->setTagDouble("content.uncertainty",Uncertainty);
  return true;
}


bool TQTable::setEntryValue(unsigned int i, unsigned int j, int content){
  // set the table entry at row i and column j to the given numerical content
  TQTaggable* entry = this->getEntryInternal(i,j,true);
  if(!entry) return false;
  entry->setTagInteger("content.value",content);
  return true;
}


void TQTable::setAutoExpand(bool val){
  // (de)activate automatic table expansion
  this->autoExpand = val;
}
bool TQTable::getAutoExpand(){
  // retrieve activation status of automatic table expansion
  return this->autoExpand;
}

namespace {
  inline bool ge(unsigned int a, unsigned int b){
    return (a>=b) && (a!=(unsigned int)(-1));
  }
}

bool TQTable::expand(unsigned int i, unsigned int j){
  // expand the table to have at least i rows and j columns
  DEBUGclass("attempting to expand table from %i/%i to %i/%i",this->nrows,this->ncols,i,j);
  unsigned int newrows = std::max(i,(unsigned int)this->nrows);
  unsigned int newcols = std::max(j,(unsigned int)this->ncols);
  if(newcols > this->ncols  || newrows > this->nrows){
    DEBUGclass("preparing to move memory");
    TQTaggable** old = this->data;
    DEBUGclass("attempting to reallocate...");
    this->data = (TQTaggable**)malloc(newrows*newcols*sizeof(TQTaggable*));
    if(!this->data){
      throw std::bad_alloc();
    }
    DEBUGclass("initializing empty cells");
    for(unsigned int i=newrows-1; ge(i,this->nrows); --i){
      for(unsigned int j=newcols-1; ge(j,0); j--){
        this->data[i*newcols + j] = NULL;
      }
    }
    DEBUGclass("moving non-empty cells");
    for(unsigned int i=this->nrows-1; ge(i,0); --i){
      for(unsigned int j=newcols-1; ge(j,this->ncols); --j){
        this->data[i*newcols + j] = NULL;
      }
      for(unsigned int j=this->ncols-1; ge(j,0); --j){
        this->data[i*newcols + j] = old[i*this->ncols + j];
      }
    }
    if(old){
      if(manualAllocation) free(old);
      else delete[] old;
    }
    manualAllocation = true;
    this->nrows = newrows;
    this->ncols = newcols;
    this->nfields = this->nrows * this->ncols;
    DEBUGclass("expansion complete");
  }
  return true;
}

int TQTable::clear(){
  int val = 0;
  for(unsigned int i=0; i<this->nfields; i++){
    if(!this->data[i]) continue;
    delete this->data[i];
    val++;
  }
  if(manualAllocation) free(this->data);
  else delete[] this->data;
  this->nrows = 0;
  this->ncols = 0;
  this->nfields = 0;
  this->vlines.clear();
  this->hlines.clear();
  this->data = NULL;
  return val;
}

int TQTable::cleanup(){
  // clean the table, removing empty entries
  int n=0;
  for(unsigned int i=0; i<this->nfields; i++){
    if(!this->data[i]) continue;
    if(this->data[i]->hasMatchingTag("content.*")) continue;
    delete this->data[i];
    this->data[i] = NULL;
    n++;
  }
  return n;
}

bool TQTable::shrink(){
  // shrink the table, removing empty rows and columns
  DEBUGclass("shrink called");
  this->cleanup();
  std::vector<int> rowEmpty;
  unsigned int rows = 0;
  std::vector<int> colEmpty;
  unsigned int cols = 0;
  for(unsigned int i=0; i<this->nrows; i++){
    bool empty = true;
    for(unsigned int j=0; j<this->ncols; j++){
      if(this->data[i*this->ncols + j]) empty = false;
    }
    if(i < this->hlines.size()){
      // shift the line from [i] to [row]
      this->hlines[rows] = std::max(this->hlines[rows],this->hlines[i]);
      // avoid double-counting
      if(i>rows) this->hlines[i] = 0;
    } 
    if(empty){
      rowEmpty.push_back(-1);
    } else {
      rowEmpty.push_back(rows);
      rows++;
    }
  }
  for(unsigned int j=0; j<this->ncols; j++){
    bool empty = true;
    for(unsigned int i=0; i<this->nrows; i++){
      if(this->data[i*this->ncols + j]) empty = false;
    }
    if(j < this->colAlign.size()) this->colAlign[cols] = this->colAlign[j];
    else if(cols < this->colAlign.size()) this->colAlign[cols] = this->getTagIntegerDefault("colAlign",10);
    if(j < this->vlines.size()){
      this->vlines[cols] = std::max(this->vlines[j],this->vlines[cols]);
      if(j!=cols) this->vlines[j] = 0;
    } else if(cols < this->vlines.size()) this->vlines[cols] = 0;
    if(empty){
      colEmpty.push_back(-1);
    } else {
      colEmpty.push_back(cols);
      cols++;
    }
  }
  if(rows == this->nrows && cols == this->ncols) return true;
  TQTaggable** old = this->data;
  this->data = (TQTaggable**)calloc(rows*cols,sizeof(TQTaggable*));
  if(!data) return false;
  for(unsigned int i=0; i<this->nrows; i++){
    for(unsigned int j=0; j<this->ncols; j++){
      if(rowEmpty[i] >= 0 && colEmpty[j] >= 0){
        this->data[rowEmpty[i]*cols + colEmpty[j]] = old[i*this->ncols + j];
      }
    }
  }
  if(manualAllocation) free(old);
  else delete[] old;
  manualAllocation = true;
  this->nrows = rows;
  this->ncols = cols;
  this->nfields = this->nrows * this->ncols;
  DEBUGclass("shrink done");
  return true;
}

void TQTable::clearRow(unsigned int row){
  // clear/delete the row with given index
  for(unsigned int j=0; j<this->ncols; j++){
    this->removeEntry(row,j);
  }
}

void TQTable::clearCol(unsigned int col){
  // clear/delete the column with given index
  for(unsigned int i=0; i<this->nrows; i++){
    this->removeEntry(i,col);
  }
}

bool TQTable::hasEntry(unsigned int i, unsigned int j){
  // check if entry at row i and column j exists
  TQTaggable* entry = this->getEntryInternal(i,j);
  if(!entry) return false;
  return true;
}

void TQTable::removeEntry(unsigned int i,unsigned int j){
  // remove entry at row i and column j 
  if(i>= this->nrows || j>= this->ncols) return;
  if(this->data[i*this->ncols + j]){
    delete this->data[i*this->ncols + j];
    this->data[i*this->ncols + j] = NULL;
  }
}


bool TQTable::setProperty(unsigned int i, unsigned int j, const TString& key, const TString& value){
  // set a property/style tag for entry at row i and column j 
  TQTaggable* entry = this->getEntryInternal(i,j,true);
  if(entry) return entry->setTagString(key,value);
  return false;
}
bool TQTable::setProperty(unsigned int i, unsigned int j, const TString& key, const char* value){
  // set a property/style tag for entry at row i and column j 
  TQTaggable* entry = this->getEntryInternal(i,j,true);
  if(entry) return entry->setTagString(key,value);
  return false;
}
bool TQTable::setProperty(unsigned int i, unsigned int j, const TString& key, double value){
  // set a property/style tag for entry at row i and column j 
  TQTaggable* entry = this->getEntryInternal(i,j,true);
  if(entry) return entry->setTagDouble(key,value);
  return false;
}
bool TQTable::setProperty(unsigned int i, unsigned int j, const TString& key, int value){
  TQTaggable* entry = this->getEntryInternal(i,j,true);
  if(entry) return entry->setTagInteger(key,value);
  return false;
}
void TQTable::setColProperty(unsigned int j, const TString& key, const TString& value){
  // set a property/style tag for all entries in column j 
  for(unsigned int i=0; i<this->nrows; i++){
    this->setProperty(i,j,key,value);
  }
}
void TQTable::setColProperty(unsigned int j, const TString& key, const char* value){
  // set a property/style tag for all entries in column j 
  for(unsigned int i=0; i<this->nrows; i++){
    this->setProperty(i,j,key,value);
  }
}
void TQTable::setColProperty(unsigned int j, const TString& key, double value){
  // set a property/style tag for all entries in column j 
  for(unsigned int i=0; i<this->nrows; i++){
    this->setProperty(i,j,key,value);
  }
}
void TQTable::setColProperty(unsigned int j, const TString& key, int value){
  // set a property/style tag for all entries in column j 
  for(unsigned int i=0; i<this->nrows; i++){
    this->setProperty(i,j,key,value);
  }
}
void TQTable::setRowProperty(unsigned int i, const TString& key, const TString& value){
  // set a property/style tag for all entries in row i 
  for(unsigned int j=0; j<this->ncols; j++){
    this->setProperty(i,j,key,value);
  }
}
void TQTable::setRowProperty(unsigned int i, const TString& key, const char* value){
  // set a property/style tag for all entries in row i 
  for(unsigned int j=0; j<this->ncols; j++){
    this->setProperty(i,j,key,value);
  }
}
void TQTable::setRowProperty(unsigned int i, const TString& key, double value){
  // set a property/style tag for all entries in row i 
  for(unsigned int j=0; j<this->ncols; j++){
    this->setProperty(i,j,key,value);
  }
}
void TQTable::setRowProperty(unsigned int i, const TString& key, int value){
  // set a property/style tag for all entries in row i 
  for(unsigned int j=0; j<this->ncols; j++){
    this->setProperty(i,j,key,value);
  }
}


bool TQTable::write (const TString& fname, const TString& format, TQTaggable& tags){
  // write table data to the given file
  // format is specified explicitly
  // style is given via tags
  
  //@tag: [ensureDirectory] If this argument tag is set to true, directories on the file system are be created when required to write to the specified output file(path). Default: false.
  if(tags.getTagBoolDefault("ensureDirectory",false)) TQUtils::ensureDirectoryForFile(fname);
  std::ofstream out(fname);
  if(!out.is_open()) return false;
  return this->print(&out,format,tags);
}

bool TQTable::write (const TString& fname, TQTaggable& tags){
  // write table data to the given file
  // format and style can be specified with tags
  if(tags.getTagBoolDefault("ensureDirectory",false)) TQUtils::ensureDirectoryForFile(fname);
  std::ofstream out(fname);
  if(!out.is_open()) return false;
  return this->print(&out,tags);
}
bool TQTable::writeCSV (const TString& fname, TQTaggable& tags){
  // write table data to the given file in CSV format
  // format and style can be further specified with tags
  if(tags.getTagBoolDefault("ensureDirectory",false)) TQUtils::ensureDirectoryForFile(fname);
  std::ofstream out(fname);
  if(!out.is_open()) return false;
  return this->printCSV(&out,tags);
}
bool TQTable::writeHTML (const TString& fname, TQTaggable& tags){
  // write table data to the given file in HTML format
  // format and style can be further specified with tags
  if(tags.getTagBoolDefault("ensureDirectory",false)) TQUtils::ensureDirectoryForFile(fname);
  std::ofstream out(fname);
  if(!out.is_open()) return false;
  return this->printHTML(&out,tags);
}
bool TQTable::writeLaTeX(const TString& fname, TQTaggable& tags){
  // write table data to the given file in LaTeX format
  // format and style can be further specified with tags
  if(tags.getTagBoolDefault("ensureDirectory",false)) TQUtils::ensureDirectoryForFile(fname);
  std::ofstream out(fname);
  if(!out.is_open()) return false;
  return this->printLaTeX(&out,tags);
}
bool TQTable::writePlain(const TString& fname, TQTaggable& tags){
  // write table data to the given file in plain ascii/unicode art format
  // format and style can be further specified with tags
  if(tags.getTagBoolDefault("ensureDirectory",false)) TQUtils::ensureDirectoryForFile(fname);
  std::ofstream out(fname);
  if(!out.is_open()) return false;
  return this->printPlain(&out,tags);
}



bool TQTable::print (std::ostream* out, TQTaggable* tags){ if(tags) return this->print(out,*tags); return this->print(out,"format=unicode"); }
bool TQTable::printCSV (std::ostream* out, TQTaggable* tags){ if(tags) return this->printCSV(out,*tags); return this->printCSV(out,"format=csv"); }
bool TQTable::printHTML (std::ostream* out, TQTaggable* tags){ if(tags) return this->printHTML(out,*tags); return this->printHTML(out,"format=html"); } 
bool TQTable::printLaTeX(std::ostream* out, TQTaggable* tags){ if(tags) return this->printLaTeX(out,*tags); return this->printLaTeX(out,"format=latex"); }
bool TQTable::printPlain(std::ostream* out, TQTaggable* tags){ if(tags) return this->printPlain(out,*tags); return this->printPlain(out,"format=unicode"); }
bool TQTable::print (std::ostream* out, const TString& tags){ TQTaggable tmp(tags); return this->print (out,tmp); }
bool TQTable::printCSV (std::ostream* out, const TString& tags){ TQTaggable tmp(tags); return this->printCSV (out,tmp); }
bool TQTable::printHTML (std::ostream* out, const TString& tags){ TQTaggable tmp(tags); return this->printHTML (out,tmp); }
bool TQTable::printLaTeX(std::ostream* out, const TString& tags){ TQTaggable tmp(tags); return this->printLaTeX(out,tmp); }
bool TQTable::printPlain(std::ostream* out, const TString& tags){ TQTaggable tmp(tags); return this->printPlain(out,tmp); }
bool TQTable::print (std::ostream* out, const char* tags){ TQTaggable tmp(tags); return this->print (out,tmp); }
bool TQTable::printCSV (std::ostream* out, const char* tags){ TQTaggable tmp(tags); return this->printCSV (out,tmp); }
bool TQTable::printHTML (std::ostream* out, const char* tags){ TQTaggable tmp(tags); return this->printHTML (out,tmp); }
bool TQTable::printLaTeX(std::ostream* out, const char* tags){ TQTaggable tmp(tags); return this->printLaTeX(out,tmp); }
bool TQTable::printPlain(std::ostream* out, const char* tags){ TQTaggable tmp(tags); return this->printPlain(out,tmp); }

bool TQTable::print (TQTaggable& tags) { return this->print (&std::cout,tags); }
bool TQTable::printCSV (TQTaggable& tags) { return this->printCSV (&std::cout,tags); }
bool TQTable::printHTML (TQTaggable& tags) { return this->printHTML (&std::cout,tags); }
bool TQTable::printLaTeX(TQTaggable& tags) { return this->printLaTeX(&std::cout,tags); }
bool TQTable::printPlain(TQTaggable& tags) { return this->printPlain(&std::cout,tags); }
bool TQTable::print (TQTaggable* tags) { return this->print (&std::cout,tags); }
bool TQTable::printCSV (TQTaggable* tags) { return this->printCSV (&std::cout,tags); }
bool TQTable::printHTML (TQTaggable* tags) { return this->printHTML (&std::cout,tags); }
bool TQTable::printLaTeX(TQTaggable* tags) { return this->printLaTeX(&std::cout,tags); }
bool TQTable::printPlain(TQTaggable* tags) { return this->printPlain(&std::cout,tags); }
bool TQTable::print (const TString& tags) { return this->print (&std::cout,tags); }
bool TQTable::printCSV (const TString& tags) { return this->printCSV (&std::cout,tags); }
bool TQTable::printHTML (const TString& tags) { return this->printHTML (&std::cout,tags); }
bool TQTable::printLaTeX(const TString& tags) { return this->printLaTeX(&std::cout,tags); }
bool TQTable::printPlain(const TString& tags) { return this->printPlain(&std::cout,tags); }
bool TQTable::print (const char* tags) { return this->print (&std::cout,tags); }
bool TQTable::printCSV (const char* tags) { return this->printCSV (&std::cout,tags); }
bool TQTable::printHTML (const char* tags) { return this->printHTML (&std::cout,tags); }
bool TQTable::printLaTeX(const char* tags) { return this->printLaTeX(&std::cout,tags); }
bool TQTable::printPlain(const char* tags) { return this->printPlain(&std::cout,tags); }

bool TQTable::write (const TString& fname, const TString& tags){ TQTaggable tmp(tags); return this->write (fname,tmp); }
bool TQTable::writeCSV (const TString& fname, const TString& tags){ TQTaggable tmp(tags); return this->writeCSV (fname,tmp); }
bool TQTable::writeHTML (const TString& fname, const TString& tags){ TQTaggable tmp(tags); return this->writeHTML (fname,tmp); }
bool TQTable::writeLaTeX(const TString& fname, const TString& tags){ TQTaggable tmp(tags); return this->writeLaTeX(fname,tmp); }
bool TQTable::writePlain(const TString& fname, const TString& tags){ TQTaggable tmp(tags); return this->writePlain(fname,tmp); }
bool TQTable::write (const TString& fname, const char* tags){ TQTaggable tmp(tags); return this->write (fname,tmp); }
bool TQTable::writeCSV (const TString& fname, const char* tags){ TQTaggable tmp(tags); return this->writeCSV (fname,tmp); }
bool TQTable::writeHTML (const TString& fname, const char* tags){ TQTaggable tmp(tags); return this->writeHTML (fname,tmp); }
bool TQTable::writeLaTeX(const TString& fname, const char* tags){ TQTaggable tmp(tags); return this->writeLaTeX(fname,tmp); }
bool TQTable::writePlain(const TString& fname, const char* tags){ TQTaggable tmp(tags); return this->writePlain(fname,tmp); }
bool TQTable::write (const TString& fname, TQTaggable* tags){ if(tags) return this->write(fname,*tags); return this->write(fname,""); } 
bool TQTable::writeCSV (const TString& fname, TQTaggable* tags){ if(tags) return this->writeCSV(fname,*tags); return this->writeCSV(fname,"format=csv"); } 
bool TQTable::writeHTML (const TString& fname, TQTaggable* tags){ if(tags) return this->writeHTML(fname,*tags); return this->write(fname,"format=html"); } 
bool TQTable::writeLaTeX(const TString& fname, TQTaggable* tags){ if(tags) return this->writeLaTeX(fname,*tags); return this->write(fname,"format=latex"); }
bool TQTable::writePlain(const TString& fname, TQTaggable* tags){ if(tags) return this->writePlain(fname,*tags); return this->write(fname,"format=unicode"); }



bool TQTable::setContent(TQTaggable* entry,const TString& content, TString prior){
  // set the content tags on a given object
  // this function will automatically convert the text into all supported formats
  // this is an internal function accessed by TQTable::setEntry( ... )
  if(!entry) return false;
  entry->setGlobalOverwrite(prior.IsNull());
  if(prior.IsNull() || TQStringUtils::equal(prior,"verbatim")) prior = TQStringUtils::findFormat(content);
  prior.ToLower();
  //tag documentation at first usage.
  entry->setTagString("content.verbatim",content);
  if(prior == "unicode"){
    //@tag: [content.ascii] This entry tag contains the ascii content of the corresponding cell.
    entry->setTagString("content.ascii",TQStringUtils::makeASCII(content));
    //@tag: [content.latex] This entry tag contains the LaTeX content of the corresponding cell.
    entry->setTagString("content.latex",TQStringUtils::convertPlain2LaTeX(content));
    //@tag: [content.html] This entry tag contains the HTML content of the corresponding cell.
    entry->setTagString("content.html",TQStringUtils::convertPlain2HTML(content)); 
    entry->setGlobalOverwrite(true);
    //@tag: [content.unicode] This entry tag contains the unicode content of the corresponding cell.
    entry->setTagString("content.unicode",content);
    return true;
  }
  if(prior == "ascii"){
    entry->setTagString("content.unicode",content);
    entry->setTagString("content.latex",TQStringUtils::convertPlain2LaTeX(content));
    entry->setTagString("content.html",TQStringUtils::convertPlain2HTML(content)); 
    entry->setGlobalOverwrite(true);
    entry->setTagString("content.ascii",content);
    return true;
  }
  if(prior == "latex"){
    entry->setTagString("content.unicode",TQStringUtils::convertLaTeX2Plain(content,true));
    entry->setTagString("content.ascii",TQStringUtils::convertLaTeX2Plain(content,false));
    entry->setTagString("content.html",TQStringUtils::convertLaTeX2HTML(content)); 
    entry->setGlobalOverwrite(true);
    entry->setTagString("content.latex",content);
    return true;
  }
  if(prior == "html"){
    entry->setTagString("content.unicode",TQStringUtils::convertHTML2Plain(content,true));
    entry->setTagString("content.ascii",TQStringUtils::convertHTML2Plain(content,false));
    entry->setTagString("content.latex",TQStringUtils::convertHTML2LaTeX(content));
    entry->setGlobalOverwrite(true);
    entry->setTagString("content.html",content);
    return true;
  }
  if(prior == "roottex"){
    entry->setTagString("content.unicode",TQStringUtils::convertROOTTeX2Plain(content,true));
    entry->setTagString("content.ascii",TQStringUtils::convertROOTTeX2Plain(content,false));
    entry->setTagString("content.latex",TQStringUtils::convertROOTTeX2LaTeX(content));
    entry->setTagString("content.html", TQStringUtils::convertROOTTeX2HTML(content));
    entry->setGlobalOverwrite(true);
    return true;
  }
  entry->setGlobalOverwrite(true);
  return false;
}


TString TQTable::getRowAsCSV(int row, const TString& sep){
  // retrieve the given row as a CSV formatted string
  TString retval;
  if((unsigned int)row > this->nrows) return retval;
  for(unsigned int i=0; i<this->ncols; i++){
    retval.Append(this->getEntryASCII(row,i));
    if(i+1 < this->ncols) retval.Append(sep);
  }
  return retval;
}
 
 
TString TQTable::getDetails(){
  // retrieve table details as a string
  return TString::Format("%d rows, %d columns",(int)(this->nrows), (int)(this->ncols));
}
 
void TQTable::dump(){
  // dump all essential data members to the console
  std::cout << this->GetName() << ": " << this->getDetails() << std::endl;;
  if(this->data){
    std::cout << "table data is:" << std::endl;
    for(unsigned int i=0; i<this->nfields; i++){
      std::cout << TQStringUtils::fixedWidth(TString::Format("%d",(int)i),10,"r") << ": ";
      if(this->data[i]) std::cout << this->data[i]->exportTagsAsString();
      else std::cout << "(empty)";
      std::cout << std::endl;
    }
  } else {
    std::cout << "table data is NULL" << std::endl;
  }
}


const TList& TQTable::makeTList(const TString& sep){ // TODO: not implemented yet
  // convert to a TList
  TList* l = this->makeTListPtr();
  return *l;
}


void TQTable::addToListContents(TList* l, const TString& sep){
  // append contents to a TList
  for(unsigned int i=1; i<this->nrows; i++){
    l->Add(new TObjString(this->getRowAsCSV(i,sep)));
  }
}

void TQTable::setListContents(TList* l, const TString& sep){
  // set the contents of a TList to the contents of this table
  l->Clear();
  l->SetName(this->GetName());
  this->addToListContents(l,sep);
}

TList* TQTable::makeTListPtr(const TString& sep){
  // convert to a TList
  TList* l = new TList();
  this->setListContents(l,sep);
  return l;
}

void TQTable::setFromTList(TList& l){
  // import data from a TList
  this->clear();
  this->expand(l.GetEntries(),1);
  for(unsigned int i=0; i<(unsigned int)l.GetEntries(); i++){
    this->setEntry(i,0,l.At(i)->GetName());
  }
}

void TQTable::setFromTList(TList* l){
  // import data from a TList
  this->clear();
  if(l){
    this->expand(l->GetEntries(),1);
    for(unsigned int i=0; i<(unsigned int)l->GetEntries(); i++){
      this->setEntry(i,0,l->At(i)->GetName());
 
    }
  }
}

std::map<TString,TString> TQTable::getMap(const TString& key, const TString& value, const TString& keyformat, const TString& valformat){
  // retrieve a std::map, mapping entries of column labeled 'key' to entries of column labeled 'value'
  // formatting can be controlled with 'keyformat' and 'valformat'
  int keyidx = this->findColumn(key);
  int validx = this->findColumn(value);
  if(keyidx < 0){
    ERRORclass("unable to generate map: cannot find column '%s'!",key.Data());
  }
  if(validx < 0){
    ERRORclass("unable to generate map: cannot find column '%s'!",value.Data());
  }
  return this->getMap(keyidx,validx,keyformat,valformat,true);
}

std::map<TString,TString> TQTable::getMap(unsigned int keyidx, unsigned int validx, const TString& keyformat, const TString& valformat, bool skipfirstline){
  // retrieve a std::map, mapping entries of column keyidx to entries of column validx
  // formatting can be controlled with 'keyformat' and 'valformat'
  // if skipfirstline is true, the first row will be skipped
  std::map<TString,TString> map;
  for(unsigned int i=skipfirstline; i<this->nrows; i++){
    TString k = this->getEntry(i,keyidx,keyformat);
    TString val = this->getEntry(i,validx,valformat);
    map[k] = val;
  }
  return map;
}

int TQTable::findColumn(TString colname, bool caseSensitive) {
  return this->findColumn(colname, 0, caseSensitive);
}

int TQTable::findColumn(TString colname, int row, bool caseSensitive){
  // retrieve the index of a column with the given name
  // return -1 if none found
  if(!caseSensitive) colname.ToLower();
  for(unsigned int j=0; j<this->ncols; j++){
    TString entry = this->getEntryPlain(row,j,false);
    if(!caseSensitive) entry.ToLower();
    if(TQStringUtils::matches(entry,colname)){
      return j;
    } 
  }
  return -1;
}


int TQTable::findRow(TString content, int column, bool caseSensitive){
  // retrieve the index of a row where the given column has the given content 
  // return -1 if none found
  if(!caseSensitive) content.ToLower();
  for(unsigned int i=0; i<this->nrows; i++){
    TString entry = this->getEntryPlain(i,column,false);
    if(!caseSensitive) entry.ToLower();
    DEBUGclass("comparing '%s' and '%s'",entry.Data(),content.Data());
    if(TQStringUtils::matches(entry,content)){
      DEBUGclass("match found!");
      return i;
    } else {
      DEBUGclass("not matching");
    }
  }
  DEBUGclass("didnt find a match for '%s' in column %d",content.Data(),column);
  return -1;
}

int TQTable::readColumn(TQTable* other, const TString& colname, const TString& matchcolname){
  // read a column from another table
  return this->readColumn(other,other->findColumn(colname),this->findColumn(matchcolname),other->findColumn(matchcolname));
}

int TQTable::readColumn(TQTable* other, const TString& colname, int thismatchcol, int othermatchcol){
  // read a column from another table
  return this->readColumn(other,other->findColumn(colname),thismatchcol,othermatchcol);
}

int TQTable::readColumn(TQTable* other, int col, int thismatchcol,int othermatchcol){
  // read a column from another table
  unsigned int newcol = this->getNcols();
  int set = 0;
  for(unsigned int row=1; row<(unsigned int)(other->getNrows()); ++row){
    const TString current(other->getEntryPlain(row,othermatchcol,false));
    int thisRow = this->findRow(current,thismatchcol);
    if(thisRow < 0){
      DEBUGclass("for '%s', didn't find a matching row in column %d",current.Data(),thismatchcol);
      continue;
    }
    TQTaggable* oldItem = other->getEntryInternal(row,col,false);
    if(oldItem){
      TQTaggable* newItem = this->getEntryInternal(thisRow,newcol,true);       
      newItem->importTags(oldItem);
      DEBUGclass("copying '%s': %s",current.Data(),newItem->exportTagsAsString().Data());
      set++;
    }
  }
  return set;
}       

int TQTable::markDifferences(TQTable* other, const TString& color, int colID, int rowID, const TString& format) {
  // Set the background color of this table's cells to 'color' if 
  // the cell content does not match the content of the corresponding
  // cell in the other table. The equivalence of cell contents is 
  // evaluated based on the content for the specified format.
  // 
  // If the table dimensions do not match no changes are performed and 
  // false is returned unless for the mismatched dimension a column/row
  // number is given which should be used to identify the row/column to 
  // be compared to.
  
  if (!other || (this->getNcols() != other->getNcols() && rowID<0) || (this->getNrows() != other->getNrows() && colID<0) ) return -1;
  int nMarks = 0;
  int ii = 0;
  int jj = 0;
  for (int i=0; i<this->getNrows(); ++i) {
    ii = colID<0 ? i : other->findRow(this->getEntryPlain(i,colID),colID); //find row in other table corresponding to the current row in this table
    for (int j=0; j<this->getNcols(); ++j) {
      jj = rowID<0 ? j : other->findColumn(this->getEntryPlain(rowID,j),rowID); //find column in other table corresponding to the current column in this table
      if (!TQStringUtils::equal(this->getEntry(i,j,format), other->getEntry(ii,jj,format))) {
        this->setProperty(i,j,"cellcolor",color);
        nMarks++;
      }
    }
  }
  return nMarks;
}









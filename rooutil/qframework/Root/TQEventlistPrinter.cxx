#include "QFramework/TQEventlistPrinter.h"
#include "QFramework/TQIterator.h"

// #define _DEBUG_

#include "QFramework/TQLibrary.h"

////////////////////////////////////////////////////////////////////////////////////////////////
//
// TQEventlistPrinter:
//
// The TQEventlistPrinter class uses an TQSampleDataReader to obtain
// an event list and to print it in ordinary text style, in latex
// style or in HTML style.
//
////////////////////////////////////////////////////////////////////////////////////////////////

ClassImp(TQEventlistPrinter)


//______________________________________________________________________________________________

TQEventlistPrinter::TQEventlistPrinter() : 
TQPresenter()
{
  // Default constructor
}

//______________________________________________________________________________________________

TQEventlistPrinter::TQEventlistPrinter(TQSampleFolder * samples) : 
  TQPresenter(samples)
{
  // Constructor 
}

//__________________________________________________________________________________|___________

TQEventlistPrinter::TQEventlistPrinter(TQSampleDataReader * reader) : 
  TQPresenter(reader)
{
  // Constructor 
}

//__________________________________________________________________________________|___________

int TQEventlistPrinter::writeEventlists(const TString& jobname, const TString& outputPath, const TString& tags){
  // write all scheduled event lists
  TQTaggable taggable(tags);
  return this->writeEventlists(jobname, outputPath,taggable);
}

//__________________________________________________________________________________|___________

int TQEventlistPrinter::writeEventlists(const TString& jobname, const TString& outputPath, TQTaggable tags){
  // write all scheduled event lists
  TQTaggableIterator pitr(this->fProcesses);
  TQTaggableIterator citr(this->fCuts);
  //@tag: [verbose] If this argument tag is set to true, additional verbose printouts are enabled. Default: false.
  bool verbose = tags.getTagBoolDefault("verbose",false);
  int nlists = 0;
  //@tag: [formats,format] These tags determine the formats in which output is generated. "formats" is vector valued and takes precendence over "format" which (if neither "format" nor "formats" is provided) defaults to "txt". 
  std::vector<TString> formats = tags.getTagVString("formats");
  if(formats.size() < 1) formats.push_back(tags.getTagStringDefault("format","txt"));
  if(verbose) VERBOSEclass("writing event lists to '%s'",outputPath.Data());
  tags.setGlobalOverwrite(false);
  tags.setTagString("layout","article");
  while(pitr.hasNext()){
    TQNamedTaggable* process = pitr.readNext();
    TString path = tags.replaceInText(process->getTagStringDefault(".path",process->GetName()));
    //@tag: [.name] This process/cut tag determines the name of the process/cut which is used in titles and filenames of the output. Defaults are obtained via process/cut->GetName()
    TString name = tags.replaceInText(TQFolder::makeValidIdentifier(process->getTagStringDefault(".name",process->GetName())));
    if(TQStringUtils::findFirstNotOf(path,"| ") == -1) continue;
    while(citr.hasNext()){
      TQNamedTaggable* cut = citr.readNext();
      TString cutname = cut->getTagStringDefault(".name",cut->GetName());
      TString evtlistname = TQFolder::concatPaths(cutname,jobname);
      TQTable* evtlist = this->fReader->getEventlist(path, evtlistname, &tags);
      if(!evtlist){
        if(verbose) WARNclass("unable to retrieve event list '%s' from path '%s'",evtlistname.Data(),path.Data());
        continue;
      }
      evtlist->setTagString("colAlign","r");
      evtlist->setTagString("title",name + " @ " + cutname);
      TString filename = jobname+"-"+cutname+"-"+name;
      for(size_t i=0; i<formats.size(); i++){
        TString filepath = TQFolder::concatPaths(outputPath,filename)+"."+formats[i];
        bool ok = evtlist->write(filepath,formats[i],tags);
        if(!ok){
          if(verbose) WARNclass("unable to write event list '%s' to '%s'",evtlistname.Data(),filepath.Data());
          continue;
        } else if(verbose){
          VERBOSEclass("successfully written event list '%s' to '%s'",evtlistname.Data(),filepath.Data());
        }
      }
      delete evtlist;
      nlists++;
    }
    citr.reset();
  }
  return nlists;
}

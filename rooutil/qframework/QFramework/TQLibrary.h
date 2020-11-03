//this file looks like plain C, but it's actually -*- c++ -*-
#ifndef __TQ_LIBRARY__
#define __TQ_LIBRARY__

#include "TString.h"
#include "TMultiGraph.h"
#include "TGraph.h"
#include <iostream>
#include <stdarg.h>
#include "QFramework/TQStringUtils.h"
#include "QFramework/TQHistogramUtils.h"
#include "QFramework/TQMessageStream.h"
#include <cstdlib>

#ifndef __CINT__
#ifdef _DEBUG_
#define DEBUG(...) TQLibrary::msgStream.sendMessage (TQMessageStream::DEBUG, __VA_ARGS__)
#define DEBUGfile(...) TQLibrary::msgStream.sendFileLineMessage (TQMessageStream::DEBUG,__FILE__,__LINE__, __VA_ARGS__)
#define DEBUGfunc(...) TQLibrary::msgStream.sendFunctionMessage (TQMessageStream::DEBUG,__FUNCTION__, __VA_ARGS__)
#define DEBUGclass(...) TQLibrary::msgStream.sendClassFunctionMessage (TQMessageStream::DEBUG,Class(),__FUNCTION__,__VA_ARGS__)
#define DEBUGfuncargs(...) TQLibrary::msgStream.sendFunctionArgsMessage (TQMessageStream::DEBUG,__FUNCTION__, __VA_ARGS__)
#define DEBUGclassargs(...) TQLibrary::msgStream.sendClassFunctionArgsMessage (TQMessageStream::DEBUG,Class(),__FUNCTION__,__VA_ARGS__)
#else
#define DEBUG(...) 
#define DEBUGfile(...) 
#define DEBUGfunc(...) 
#define DEBUGclass(...) 
#define DEBUGfuncargs(...) 
#define DEBUGclassargs(...) 
#endif

#define VERBOSE(...) TQLibrary::msgStream.sendMessage (TQMessageStream::VERBOSE, __VA_ARGS__)
#define VERBOSEfile(...) TQLibrary::msgStream.sendFileLineMessage (TQMessageStream::VERBOSE,__FILE__,__LINE__, __VA_ARGS__)
#define VERBOSEfunc(...) TQLibrary::msgStream.sendFunctionMessage (TQMessageStream::VERBOSE,__FUNCTION__, __VA_ARGS__)
#define VERBOSEclass(...) TQLibrary::msgStream.sendClassFunctionMessage (TQMessageStream::VERBOSE,Class(),__FUNCTION__,__VA_ARGS__)
#define VERBOSEfuncargs(...) TQLibrary::msgStream.sendFunctionArgsMessage (TQMessageStream::VERBOSE,__FUNCTION__, __VA_ARGS__)
#define VERBOSEclassargs(...) TQLibrary::msgStream.sendClassFunctionArgsMessage(TQMessageStream::VERBOSE,Class(),__FUNCTION__,__VA_ARGS__)
#define INFO(...) TQLibrary::msgStream.sendMessage (TQMessageStream::INFO, __VA_ARGS__)
#define INFOfile(...) TQLibrary::msgStream.sendFileLineMessage (TQMessageStream::INFO,__FILE__,__LINE__, __VA_ARGS__)
#define INFOfunc(...) TQLibrary::msgStream.sendFunctionMessage (TQMessageStream::INFO,__FUNCTION__, __VA_ARGS__)
#define INFOclass(...) TQLibrary::msgStream.sendClassFunctionMessage (TQMessageStream::INFO,Class(),__FUNCTION__,__VA_ARGS__)
#define INFOfuncargs(...) TQLibrary::msgStream.sendFunctionArgsMessage (TQMessageStream::INFO,__FUNCTION__, __VA_ARGS__)
#define INFOclassargs(...) TQLibrary::msgStream.sendClassFunctionArgsMessage(TQMessageStream::INFO,Class(),__FUNCTION__,__VA_ARGS__)
#define WARN(...) TQLibrary::msgStream.sendMessage (TQMessageStream::WARNING, __VA_ARGS__)
#define WARNfile(...) TQLibrary::msgStream.sendFileLineMessage (TQMessageStream::WARNING,__FILE__,__LINE__, __VA_ARGS__)
#define WARNfunc(...) TQLibrary::msgStream.sendFunctionMessage (TQMessageStream::WARNING,__FUNCTION__, __VA_ARGS__)
#define WARNclass(...) TQLibrary::msgStream.sendClassFunctionMessage (TQMessageStream::WARNING,Class(),__FUNCTION__,__VA_ARGS__)
#define WARNfuncargs(...) TQLibrary::msgStream.sendFunctionArgsMessage (TQMessageStream::WARNING,__FUNCTION__, __VA_ARGS__)
#define WARNclassargs(...) TQLibrary::msgStream.sendClassFunctionArgsMessage(TQMessageStream::WARNING,Class(),__FUNCTION__,__VA_ARGS__)
#define ERROR(...) TQLibrary::msgStream.sendMessage (TQMessageStream::ERROR, __VA_ARGS__)
#define ERRORfile(...) TQLibrary::msgStream.sendFileLineMessage (TQMessageStream::ERROR,__FILE__,__LINE__, __VA_ARGS__)
#define ERRORfunc(...) TQLibrary::msgStream.sendFunctionMessage (TQMessageStream::ERROR,__FUNCTION__, __VA_ARGS__)
#define ERRORclass(...) TQLibrary::msgStream.sendClassFunctionMessage (TQMessageStream::ERROR,Class(),__FUNCTION__,__VA_ARGS__)
#define ERRORfuncargs(...) TQLibrary::msgStream.sendFunctionArgsMessage (TQMessageStream::ERROR,__FUNCTION__, __VA_ARGS__)
#define ERRORclassargs(...) TQLibrary::msgStream.sendClassFunctionArgsMessage(TQMessageStream::ERROR,Class(),__FUNCTION__,__VA_ARGS__)
#define CRITERROR(...) TQLibrary::msgStream.sendMessage (TQMessageStream::CRITICAL, __VA_ARGS__)
#define CRITERRORfile(...) TQLibrary::msgStream.sendFileLineMessage (TQMessageStream::CRITICAL,__FILE__,__LINE__, __VA_ARGS__)
#define CRITERRORfunc(...) TQLibrary::msgStream.sendFunctionMessage (TQMessageStream::CRITICAL,__FUNCTION__, __VA_ARGS__)
#define CRITERRORclass(...) TQLibrary::msgStream.sendClassFunctionMessage (TQMessageStream::CRITICAL,Class(),__FUNCTION__,__VA_ARGS__)
#define CRITERRORfuncargs(...) TQLibrary::msgStream.sendFunctionArgsMessage (TQMessageStream::CRITICAL,__FUNCTION__, __VA_ARGS__)
#define CRITERRORclassargs(...) TQLibrary::msgStream.sendClassFunctionArgsMessage(TQMessageStream::CRITICAL,Class(),__FUNCTION__,__VA_ARGS__)
#define BREAK(...) TQLibrary::msgStream.sendMessage (TQMessageStream::BREAK, __VA_ARGS__)
#define BREAKfile(...) TQLibrary::msgStream.sendFileLineMessage (TQMessageStream::BREAK,__FILE__,__LINE__, __VA_ARGS__)
#define BREAKfunc(...) TQLibrary::msgStream.sendFunctionMessage (TQMessageStream::BREAK,__FUNCTION__, __VA_ARGS__)
#define BREAKclass(...) TQLibrary::msgStream.sendClassFunctionMessage (TQMessageStream::BREAK,Class(),__FUNCTION__,__VA_ARGS__)
#define BREAKfuncargs(...) TQLibrary::msgStream.sendFunctionArgsMessage (TQMessageStream::BREAK,__FUNCTION__, __VA_ARGS__)
#define BREAKclassargs(...) TQLibrary::msgStream.sendClassFunctionArgsMessage(TQMessageStream::BREAK,Class(),__FUNCTION__,__VA_ARGS__)

//more verbose tracebacks in case runtime_errors are thrown within some funtion call. Will catch the error and re-throw it with additional information (message) prepended to the previously thrown error
#ifdef DEBUGBUILD
#define TRY(code,message) try{ code ; } catch (const std::exception& error) { throw std::runtime_error( (TQStringUtils::makePink(TString::Format("[%s, line %i] " , __FILE__, __LINE__ )) + TString(message) + TString(" Preceeding error: \n -> ")+TString(error.what())).Data() ); }
#else
#define TRY(code,message) code
#endif

#endif

class TQLibrary {
 
protected:
  const TString libsvnversion;
  const TString rootversion;
  const TString gccversion;
  TString tqpath;
  TString eosmgmurl;
  TString eoscmd;
  TString localGroupDiskIdentifier;
  TString dq2PathHead;
  TString dCachePathHead;
  TString dq2cmd;
  TString website;
  TString appName;
  TString procStatPath;
  TString workingDirectory;
  TString pdfFontEmbedCommand;
  bool pdfFontEmbedEnabled;
  TString exiftoolPath;
  bool exiftoolEnabled;
  TString libXMLpath;
  int consoleWidth;
  std::vector<unsigned long> rssUsageTimestamps;
  std::vector<size_t> rssUsageMemory;
  std::vector<short> rssUsageColors;
	std::vector<std::string> packages;
  TQLibrary();
  static bool isInitialized;
  static int stdoutfd;
  static int stderrfd;
  static bool stdoutfd_isRedirected;
  static bool stderrfd_isRedirected;
  static bool stdoutfd_allowRedirection;
  static bool stderrfd_allowRedirection;

  static TQLibrary* const gQFramework; 
 
public:

  static TQMessageStream msgStream;
  static TQLibrary* getQLibrary();
  static TString getEOScmd();
  void setEOScmd(TString neweoscmd);
  void setEOSurl(TString neweoscmd);
  static void printMessage();
  static const TString& getVersion();
  static int getVersionNumber();
  static const TString& getROOTVersion();
  static float getROOTVersionNumber();
  static const TString& getlibXMLpath();
  static const TString& getGCCVersion();
  static float getGCCVersionNumber();
  static int redirect_stdout(const TString& fname, bool append=false);
  static int restore_stdout();
  static int redirect_stderr(const TString& fname, bool append=false);
  static int restore_stderr();
  static int redirect(const TString& fname, bool append=false);
  static bool restore();

  static void allowRedirection_stdout(bool allow=true);
  static void allowRedirection_stderr(bool allow=true);
  static void allowRedirection(bool allow=true);

  static const TString& getApplicationName();
  void setApplicationName(TString newName);

  static const TString& getProcStatPath();
  void setProcStatPath(TString newpath);

  static const TString& getEXIFtoolPath();
  void setEXIFtoolPath(TString newpath);
  static bool enableEXIFsupport(bool val = true);
  static bool hasEXIFsupport();
  static bool setEXIF(const TString& fname, const TString& title, const TString& keywords = "ROOT ATLAS HSG3");

  static const TString& getPDFfontEmbedCommand();
  void setPDFfontEmbedCommand(TString newpath);
  static bool enablePDFfontEmbedding(bool val = true);
  static bool hasPDFfontEmbedding();
  static bool embedFonts(const TString& filename, bool verbose=false);

  static const TString& getTQPATH();
  void setTQPATH(TString newpath);

  static TString getAbsolutePath(const TString& path);
  static const TString& getWorkingDirectory();
  void setWorkingDirectory(TString newpath);
  static const TString& pwd();
  void cd(TString newpath);

  static const TString& getWebsite();

	static Long64_t getVirtualMemory();
  static void printMemory(); //print memory usage as found in ProcStatPath, likely to only work under linux
  static void recordMemory(short color=1); //adds a point (TQUtils::getCurrentTime(),TQUtils::getCurrentRSS()) to an internal registry. TQUtils::getCurrentRSS uses precompiler statements to provide a mostly platform independent retrieval of memory usage.
  static TMultiGraph* getMemoryGraph();
  
  static void setMessageStream(std::ostream& os, bool allowColors = true);
  static void setMessageStream(std::ostream* os = NULL, bool allowColors=true);
  static bool openLogFile(const TString& fname, bool allowColors = false);
  static bool closeLogFile();

  static const TString& getLocalGroupDisk();
  static const TString& getDQ2PathHead();
  static const TString& getdCachePathHead();
  static const TString& getDQ2cmd();
  void setLocalGroupDisk(const TString&);
  void setDQ2PathHead (const TString&);
  void setdCachePathHead(const TString&);
  void setDQ2cmd (const TString&);

  static int getConsoleWidth();
  static void setConsoleWidth(int width);
  void findConsoleWidth();

	static bool hasPackage(const char* pkgname);
	
  static TString findLibrary(const TString& filename);

}; 
#endif

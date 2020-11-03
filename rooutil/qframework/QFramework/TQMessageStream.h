//this file looks like plain C, but it's actually -*- c++ -*-
#ifndef __TQ_MESSAGESTREAM__
#define __TQ_MESSAGESTREAM__

#include <iostream>
#include <stdarg.h>
#include "QFramework/TQStringUtils.h"
#include "TClass.h"
#include <cstdlib>
#include <iostream>
#include <sstream>
 
class TQMessageStream {
protected:
  std::ostream* stream;
  mutable bool owner;
  bool colors;
  TString endl;
  std::stringstream buffer;
  bool useBuffer;

  bool overhang;
  TString tmpString;
  TString filename;
 
public:
  inline std::ostream& activeStream(){
    if(this->useBuffer || !this->stream) return this->buffer;
    return *this->stream;
  }

  TString getStatusString();

  const TString& getFilename();
  TString getMessages();
  void clearMessages();
  void close();
  bool open(const TString& fname, bool append = false);
  void reopen(bool append = true);
  bool isGood();
  bool isFile();
  bool isCOUT();
  bool isCERR();
  void setLineEnd(const TString& s);
  TString getLineEnd();
  void absorb(const TQMessageStream& other);

  void newline();
  void newlines(size_t n);

  TQMessageStream();
  TQMessageStream(std::ostream& _stream, bool _owner = false, bool _colors=true);
  TQMessageStream(std::ostream* _stream, bool _owner = false, bool _colors=true);
  TQMessageStream(const TString& filename, bool _colors=false); 
  ~TQMessageStream();
 
  //#ifndef __CINT__
  TQMessageStream& operator=(const TQMessageStream& other);

  enum MessageType {
    DEBUG,VERBOSE,INFO,WARNING,ERROR,CRITICAL,BREAK
  };
  enum InfoType {
    OK,FAIL,WARN,PENDING
  };
 
  void startProcessInfo (TQMessageStream::MessageType type, int lineWidth, const TString& align, TString fmt, ...);
  void endProcessInfo (TQMessageStream::InfoType result);
  void startBuffer();
  void flushBuffer();

  void sendMessage (TQMessageStream::MessageType type, TString, ...);
  void sendFileLineMessage (TQMessageStream::MessageType type, const TString& file, int line, TString fmt, ...);
  void sendClassFunctionMessage (TQMessageStream::MessageType type, TClass* tclass, const TString& fname, TString fmt, ...);
  void sendClassFunctionArgsMessage (TQMessageStream::MessageType type, TClass* tclass, const TString& fname, const TString& fargs, TString fmt, ...);
  void sendFunctionArgsMessage (TQMessageStream::MessageType type, const TString& fname, const TString& fargs, TString fmt, ...);
  void sendFunctionMessage (TQMessageStream::MessageType type, const TString& fname, TString fmt, ...);

  TString formatMessage (TQMessageStream::MessageType type, TString fmt, ...);
  TString formatFileLineMessage (TQMessageStream::MessageType type, const TString& file, int line, TString fmt, ...);
  TString formatClassFunctionMessage (TQMessageStream::MessageType type, TClass* tclass, const TString& fname, TString fmt, ...);
  TString formatClassFunctionArgsMessage (TQMessageStream::MessageType type, TClass* tclass, const TString& fname, const TString& fargs, TString fmt, ...);
  TString formatFunctionArgsMessage (TQMessageStream::MessageType type, const TString& fname, const TString& fargs, TString fmt, ...);
  TString formatFunctionMessage (TQMessageStream::MessageType type, const TString& fname, TString fmt, ...);
  //#endif
 
protected:

  //#ifndef __CINT__
  inline TString getInfoTypeString(InfoType type){
    switch(type){
    case TQMessageStream::OK:
      return colors ? TQStringUtils::makeBoldWhite("[") + TQStringUtils::makeBoldGreen(" OK ") + TQStringUtils::makeBoldWhite("]") : "[ OK ]";
    case TQMessageStream::WARN:
      return colors ? TQStringUtils::makeBoldWhite("[") + TQStringUtils::makeBoldYellow("WARN") + TQStringUtils::makeBoldWhite("]") : "[WARN]";
    case TQMessageStream::FAIL:
      return colors ? TQStringUtils::makeBoldWhite("[") + TQStringUtils::makeBoldRed("FAIL") + TQStringUtils::makeBoldWhite("]") : "[FAIL]";
    case TQMessageStream::PENDING:
      return colors ? TQStringUtils::makeBoldWhite("[") + TQStringUtils::makeBoldWhite(" .. ") + TQStringUtils::makeBoldWhite("]") : "[ .. ]";
    }
    return colors ? TQStringUtils::makeBoldWhite("[") + TQStringUtils::makeBoldPink(" ?? ") + TQStringUtils::makeBoldWhite("]") : "[ ?? ]";
  }
  inline TString getMessageTypeString(MessageType type){
    switch(type){
    case TQMessageStream::DEBUG:
      return colors ? TQStringUtils::makeBoldPink("DEBUG") : "DEBUG";
    case TQMessageStream::VERBOSE:
      return colors ? TQStringUtils::makeBoldPink("INFO") : "INFO";
    case TQMessageStream::INFO:
      return colors ? TQStringUtils::makeBoldWhite("INFO") : "INFO";
    case TQMessageStream::WARNING:
      return colors ? TQStringUtils::makeBoldYellow("WARNING") : "WARNING";
    case TQMessageStream::ERROR:
      return colors ? TQStringUtils::makeBoldRed("ERROR") : "ERROR";
    case TQMessageStream::CRITICAL:
      return colors ? TQStringUtils::makeBoldRed("CRITICAL") : "CRITICAL";
    case TQMessageStream::BREAK:
      return colors ? TQStringUtils::makeBoldRed("BREAK") : "BREAK";
    }
    return "UNKNOWN MESSAGE TYPE";
  }
  inline TString formatMessageContentText(MessageType type, const TString& text){
    switch(type){
    case TQMessageStream::DEBUG:
      return text;
    case TQMessageStream::VERBOSE:
      return text;
    case TQMessageStream::INFO:
      return text;
    case TQMessageStream::WARNING:
      return colors ? TQStringUtils::makeBoldWhite(text) : text;
    case TQMessageStream::ERROR:
      return colors ? TQStringUtils::makeBoldWhite(text) : text;
    case TQMessageStream::CRITICAL:
      return colors ? TQStringUtils::makeBoldYellow(text) : TQStringUtils::makeUppercase(text);
    case TQMessageStream::BREAK:
      return colors ? TQStringUtils::makeBoldRed(text) : TQStringUtils::makeUppercase(text);
    }
    return text;
  }
  inline TString formatMessageMetaText(MessageType type, const TString& text){
    switch(type){
    case TQMessageStream::DEBUG:
      return text;
    case TQMessageStream::VERBOSE:
      return text;
    case TQMessageStream::INFO:
      return text;
    case TQMessageStream::WARNING:
      return colors ? TQStringUtils::makeBoldWhite(text) : text;
    case TQMessageStream::ERROR:
      return colors ? TQStringUtils::makeBoldWhite(text) : text;
    case TQMessageStream::CRITICAL:
      return colors ? TQStringUtils::makeBoldWhite(text) : TQStringUtils::makeUppercase(text);
    case TQMessageStream::BREAK:
      return colors ? TQStringUtils::makeBoldRed(text) : TQStringUtils::makeUppercase(text);
    }
    return text;
  }
  inline void triggerMessageAction(MessageType type){
    // this line takes care that "refreshing lines" will never overlap the last message
    if(overhang && this->isCOUT()){
      this->overhang = false;
    }
    switch(type){
    case TQMessageStream::DEBUG:
      this->activeStream().flush();
      break;
    case TQMessageStream::ERROR:
      this->activeStream().flush();
      break;
    case TQMessageStream::BREAK:
      *(this->stream) << (colors ? TQStringUtils::makeBoldRed("EXIT") : "EXIT") << std::endl;
      this->flushBuffer();
      exit(0);
      break;
    default:
      return;
    }
  }
  //#endif
};

#endif

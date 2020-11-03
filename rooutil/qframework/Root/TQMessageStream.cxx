#include "QFramework/TQMessageStream.h"
#include <fstream>
#include <sstream>
#include "QFramework/TQUtils.h"

// #define _DEBUG_

#include "QFramework/TQLibrary.h"

////////////////////////////////////////////////////////////////////////////////////////////////
//
// TQMessageStream
//
// TQMessageStream is a helper class that provides an additional
// abstraction layer between I/O tasks and the I/O stream objects of
// the standard library. It can be helpful to control verbosity levels
// and redirect text output into log files.
//
////////////////////////////////////////////////////////////////////////////////////////////////

TQMessageStream& TQMessageStream::operator=(const TQMessageStream& other){
  // assign another stream to this one
  // this stream will become an exact copy of the other
  // stream ownership will stay with the other
  // if ownership shall be transferred, please use absorb(...)
  this->owner = false;
  this->colors = other.colors;
  this->close();
  this->stream = other.stream;
  this->endl = other.endl;
  return *this;
}

TString TQMessageStream::getMessages(){
  // retrieve the messages
  // if the underlying stream is I/O (stringstream),
  // this will return a string of all messages sent
  // since the last call of clearMessages
  // of the underlying stream is O (ofstream,cout),
  // this will return the empty string
  std::stringstream* ss = dynamic_cast<std::stringstream*>(this->stream);
  if(ss) return TString(ss->str().c_str());
  else return TString("");
}
 
void TQMessageStream::close(){
  // close the underlying stream
  // calling this function will only have an effect if
  // - the stream is owned by this instance, and
  // - the stream is an ofstream
  if(this->owner){
    std::ofstream* of = dynamic_cast<std::ofstream*>(this->stream);
    if(of){
#ifdef _DEBUG_
      std::cout << "closing output stream at " << &(this->activeStream()) << std::endl;
#endif
      of->flush();
      of->close();
    }
    delete this->stream;
    this->owner = false;
  }
  this->colors = true;
}

bool TQMessageStream::isFile(){
  // return true if this message stream streams to a file, false otherwise
  return (bool)(dynamic_cast<std::ofstream*>(this->stream));
}

bool TQMessageStream::isCOUT(){
  // return true if this message stream streams to std::cout, false otherwise
  return (bool)(&std::cout == this->stream);
}

bool TQMessageStream::isCERR(){
  // return true if this message stream streams to std::cerr, false otherwise
  return (bool)(&std::cerr == this->stream);
}

void TQMessageStream::clearMessages(){
  // clear all messages since the beginning of time
  // will only take effect if the underlying stream
  // is I/O (stringstream)
  std::stringstream* ss = dynamic_cast<std::stringstream*>(this->stream);
  if(ss) ss->str("");
}

TQMessageStream::TQMessageStream():
  stream(&std::cout),
  owner(false),
  colors(true),
  endl("\n"),
  useBuffer(false),
  overhang(false)
{
  // default constructor (will use colored ASCII std::cout)
}

TQMessageStream::TQMessageStream(std::ostream& _stream, bool _owner, bool _colors) :
  stream(&_stream),
  owner(_owner),
  colors(_colors),
  endl("\n"),
  useBuffer(false),
  overhang(false)
{
  // custom constructor with stream reference
}
 
TQMessageStream::TQMessageStream(std::ostream* _stream, bool _owner, bool _colors) :
  stream(_stream),
  owner(_owner),
  colors(_colors),
  endl("\n"),
  useBuffer(false),
  overhang(false)
{
  // custom constructor with stream pointer
}
 
TQMessageStream::TQMessageStream(const TString& fname, bool _colors) :
  stream(NULL),
  owner(true),
  colors(_colors),
  endl("\n"),
  useBuffer(false),
  overhang(false)
{
  // custom constructor with filename
  // will open a log file at the given location
  this->open(fname);
}

void TQMessageStream::reopen(bool append){
  // re-open the log file
  this->open(this->filename,append);
}

bool TQMessageStream::open(const TString& fname, bool append){
  // open a log file at the given location
  if(!TQUtils::ensureDirectoryForFile(fname)){
    ERRORfunc("unable to open file '%s' for logging, continue to use previous stream",fname.Data());
    return false;
  }
  std::ofstream* os = new std::ofstream(fname, append ? std::ofstream::out | std::ofstream::app : std::ofstream::out);
#ifdef _DEBUG_
  std::cerr << "opening output stream '" << fname << "' at " << os << std::endl;
#endif
  if(!os->is_open()){
    delete os;
    ERRORfunc("unable to open file '%s' for logging, continue to use previous stream",fname.Data());
    return false;
  } else {
    this->close();
    this->owner = true;
    this->filename = fname;
    this->stream = os;
    this->colors = false;
  }
  return true;
}
 
TQMessageStream::~TQMessageStream(){
  // destructor - will call close()
  this->close();
}

void TQMessageStream::absorb(const TQMessageStream& other){
  // absorb another stream
  // this stream will become an exact copy of the other
  // and henceforth own the other's stream
  // the ownership of the other stream will be removed
  this->stream = other.stream;
  this->colors = other.colors;
  this->owner = other.owner;
  other.owner = false;
}

bool TQMessageStream::isGood(){
  // return true if this instance is good for streaming output
  // return false otherwise
  return this->stream;
}

const TString& TQMessageStream::getFilename(){
  // return the filename associated to this stream
  return this->filename;
}

void TQMessageStream::startProcessInfo (TQMessageStream::MessageType type, int lineWidth, const TString& align, TString fmt, ...){
  // start a "process info"
  if(!this->isGood()) return;
  va_list ap;
  va_start(ap, va_(fmt));
  TString text = TQStringUtils::vaFormat(va_(fmt), ap);
  va_end(ap);
  this->startBuffer();
  TString msgHead = TQMessageStream::getMessageTypeString(type) + TQMessageStream::formatMessageMetaText(type,": ");
  int headLength = TQStringUtils::getWidth(msgHead) + 8; // the additinal characters come from spaces & the tail, i.e. "[ OK ]"
  this->tmpString = msgHead + TQStringUtils::fixedWidth(TQMessageStream::formatMessageContentText(type,text),lineWidth-headLength,align);
  this->overhang = true;
  std::cout << this->tmpString << " " << TQMessageStream::getInfoTypeString(TQMessageStream::PENDING) << "\r";
  std::cout.flush();
}
void TQMessageStream::endProcessInfo (TQMessageStream::InfoType result){
  // end a "process info"
  if(!this->isGood()) return;
  this->overhang = false;
  std::cout << this->tmpString << " " << TQMessageStream::getInfoTypeString(result) << std::endl;
  this->tmpString.Clear();
  this->flushBuffer();
}

void TQMessageStream::newline(){
  // submit a newline
  this->activeStream() << this->endl;
}

void TQMessageStream::newlines(size_t n){
  // submit n newlines
  for(size_t i=0; i<n; i++){
    this->activeStream() << this->endl;
  }
}

void TQMessageStream::sendMessage (TQMessageStream::MessageType type, TString fmt, ...){
  // send a message
  if(!this->isGood()) return;
  va_list ap;
  va_start(ap, va_(fmt));
  TString text = TQStringUtils::vaFormat(va_(fmt), ap);
  va_end(ap);
#ifdef _DEBUG_
  std::cout << "writing to " << &(this->activeStream()) << std::endl;
#endif
  this->activeStream() << TQMessageStream::getMessageTypeString(type) << TQMessageStream::formatMessageMetaText(type,": ") << TQMessageStream::formatMessageContentText(type,text) << this->endl;
  TQMessageStream::triggerMessageAction(type);
}

void TQMessageStream::sendFileLineMessage (TQMessageStream::MessageType type, const TString& file, int line, TString fmt, ...){
  // send a message message with file and line information
  if(!this->isGood()) return;
  va_list ap;
  va_start(ap, va_(fmt));
  TString text = TQStringUtils::vaFormat(va_(fmt), ap);
  va_end(ap);
  this->activeStream() << TQMessageStream::getMessageTypeString(type) << TQMessageStream::formatMessageMetaText(type,TQStringUtils::format(" in %s [l.%d]: ",file.Data(),line)) << TQMessageStream::formatMessageContentText(type,text) << this->endl;
  TQMessageStream::triggerMessageAction(type);
}

void TQMessageStream::sendClassFunctionMessage (TQMessageStream::MessageType type, TClass* tclass, const TString& fname, TString fmt, ...){
  // send a message with class and function information
  if(!this->isGood()) return;
  va_list ap;
  va_start(ap, va_(fmt));
  TString text = TQStringUtils::vaFormat(va_(fmt), ap);
  va_end(ap);
  this->activeStream() << TQMessageStream::getMessageTypeString(type) << TQMessageStream::formatMessageMetaText(type,TQStringUtils::format(" in %s::%s(...) : ",tclass->GetName(),fname.Data())) << TQMessageStream::formatMessageContentText(type,text) << this->endl;
  TQMessageStream::triggerMessageAction(type);
}

void TQMessageStream::sendFunctionMessage (TQMessageStream::MessageType type, const TString& fname, TString fmt, ...){
  // send a message with function information
  if(!this->isGood()) return;
  va_list ap;
  va_start(ap, va_(fmt));
  TString text = TQStringUtils::vaFormat(va_(fmt), ap);
  va_end(ap);
  this->activeStream() << TQMessageStream::getMessageTypeString(type) << TQMessageStream::formatMessageMetaText(type,TQStringUtils::format(" in %s(...) : ",fname.Data())) << TQMessageStream::formatMessageContentText(type,text) << this->endl;
  TQMessageStream::triggerMessageAction(type);
}

void TQMessageStream::sendClassFunctionArgsMessage (TQMessageStream::MessageType type, TClass* tclass, const TString& fname, const TString& fargs, TString fmt, ...){
  // send a message with class, function and argument information
  if(!this->isGood()) return;
  va_list ap;
  va_start(ap, va_(fmt));
  TString text = TQStringUtils::vaFormat(va_(fmt), ap);
  va_end(ap);
  this->activeStream() << TQMessageStream::getMessageTypeString(type) << TQMessageStream::formatMessageMetaText(type,TQStringUtils::format(" in %s::%s(%s) : ",tclass->GetName(),fname.Data(),fargs.Data())) << TQMessageStream::formatMessageContentText(type,text) << this->endl;
  TQMessageStream::triggerMessageAction(type);
}

void TQMessageStream::sendFunctionArgsMessage (TQMessageStream::MessageType type, const TString& fname, const TString& fargs, TString fmt, ...){
  // send a message with function and argument information
  if(!this->isGood()) return;
  va_list ap;
  va_start(ap, va_(fmt));
  TString text = TQStringUtils::vaFormat(va_(fmt), ap);
  va_end(ap);
  this->activeStream() << TQMessageStream::getMessageTypeString(type) << TQMessageStream::formatMessageMetaText(type,TQStringUtils::format(" in %s(%s) : ",fname.Data(),fargs.Data())) << TQMessageStream::formatMessageContentText(type,text) << this->endl;
  TQMessageStream::triggerMessageAction(type);
}


TString TQMessageStream::formatMessage (TQMessageStream::MessageType type, TString fmt, ...){
  // format a message
  va_list ap;
  va_start(ap, va_(fmt));
  TString text = TQStringUtils::vaFormat(va_(fmt), ap);
  va_end(ap);
  TQMessageStream::triggerMessageAction(type);
  return TQMessageStream::getMessageTypeString(type) + TQMessageStream::formatMessageMetaText(type,": ") + TQMessageStream::formatMessageContentText(type,text) + this->endl;
}

TString TQMessageStream::formatFileLineMessage (TQMessageStream::MessageType type, const TString& file, int line, TString fmt, ...){
  // format a message message with file and line information
  va_list ap;
  va_start(ap, va_(fmt));
  TString text = TQStringUtils::vaFormat(va_(fmt), ap);
  va_end(ap);
  return TQMessageStream::getMessageTypeString(type) + TQMessageStream::formatMessageMetaText(type,TQStringUtils::format(" in %s [l.%d]: ",file.Data(),line)) + TQMessageStream::formatMessageContentText(type,text) + this->endl;
}

TString TQMessageStream::formatClassFunctionMessage (TQMessageStream::MessageType type, TClass* tclass, const TString& fname, TString fmt, ...){
  // format a message with class and function information
  va_list ap;
  va_start(ap, va_(fmt));
  TString text = TQStringUtils::vaFormat(va_(fmt), ap);
  va_end(ap);
  return TQMessageStream::getMessageTypeString(type) + TQMessageStream::formatMessageMetaText(type,TQStringUtils::format(" in %s::%s(...) : ",tclass->GetName(),fname.Data())) + TQMessageStream::formatMessageContentText(type,text) + this->endl;
}

TString TQMessageStream::formatFunctionMessage (TQMessageStream::MessageType type, const TString& fname, TString fmt, ...){
  // format a message with function information
  va_list ap;
  va_start(ap, va_(fmt));
  TString text = TQStringUtils::vaFormat(va_(fmt), ap);
  va_end(ap);
  return TQMessageStream::getMessageTypeString(type) + TQMessageStream::formatMessageMetaText(type,TQStringUtils::format(" in %s(...) : ",fname.Data())) + TQMessageStream::formatMessageContentText(type,text) + this->endl;
}

TString TQMessageStream::formatClassFunctionArgsMessage (TQMessageStream::MessageType type, TClass* tclass, const TString& fname, const TString& fargs, TString fmt, ...){
  // format a message with class, function and argument information
  va_list ap;
  va_start(ap, va_(fmt));
  TString text = TQStringUtils::vaFormat(va_(fmt), ap);
  va_end(ap);
  return TQMessageStream::getMessageTypeString(type) + TQMessageStream::formatMessageMetaText(type,TQStringUtils::format(" in %s::%s(%s) : ",tclass->GetName(),fname.Data(),fargs.Data())) + TQMessageStream::formatMessageContentText(type,text) + this->endl;
}

TString TQMessageStream::formatFunctionArgsMessage (TQMessageStream::MessageType type, const TString& fname, const TString& fargs, TString fmt, ...){
  // format a message with function and argument information
  va_list ap;
  va_start(ap, va_(fmt));
  TString text = TQStringUtils::vaFormat(va_(fmt), ap);
  va_end(ap);
  return TQMessageStream::getMessageTypeString(type) + TQMessageStream::formatMessageMetaText(type,TQStringUtils::format(" in %s(%s) : ",fname.Data(),fargs.Data())) + TQMessageStream::formatMessageContentText(type,text) + this->endl;
}



void TQMessageStream::setLineEnd(const TString& s){
  // set the line endings to be used
  this->endl = s;
}
TString TQMessageStream::getLineEnd(){
  // retrieve the currently used line endings
  return this->endl;
}


void TQMessageStream::startBuffer(){
  // start using the internal buffer
  this->useBuffer = true;
  this->buffer.str("");
}
void TQMessageStream::flushBuffer(){
  // flush the internal buffer to the stream and close it
  if(!this->stream) return;
  this->useBuffer = false;
  this->activeStream() << this->buffer.str();
  this->buffer.str("");
  this->activeStream().flush();
}

TString TQMessageStream::getStatusString(){
  // retrieve a human-readable string identfying the current status of this stream
  if(!this->isGood()){
    return useBuffer ? "BAD (buffered)" : "BAD";
  }
  if(this->isCOUT()){
    return useBuffer ? "std::cout" : "std::cout (buffered)";
  }
  if(this->isCERR()){
    return useBuffer ? "std::cerr" : "std::cerr (buffered)";
  }
  if(this->isFile()){
    return useBuffer ? TString::Format("file:%s (buffered)",this->filename.Data()) : TString::Format("file:%s",this->filename.Data());
  }
  return useBuffer ? TString::Format("stringstream:%p (buffered)",(void*)(this->stream)) : TString::Format("stringstream:%p",(void*)(this->stream));
}

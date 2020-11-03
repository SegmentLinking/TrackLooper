#ifndef PrintUtil_h
#define PrintUtil_h

#include <iostream>
#include <string>

namespace SDL
{

    class prefixbuf : public std::streambuf
    {
        private:
            std::string     prefix;
            std::streambuf* sbuf;
            bool            need_prefix;

            int sync();
            int overflow(int c);

        public:
            prefixbuf(std::string const& prefix, std::streambuf* sbuf);
    };

    class oprefixstream : private virtual prefixbuf, public std::ostream
    {
        public:
            oprefixstream(std::string const& prefix, std::ostream& out);
    };

    // The following modified ostream will prefix "SDL::  " for every line
    extern oprefixstream cout;

    enum LogLevel
    {
        Log_Nothing  = 0,
        Log_Info     = 1,
        Log_Warning  = 2,
        Log_Error    = 3,
        Log_Critical = 4,
        Log_Debug    = 5,
        Log_Debug2   = 6,
        Log_Debug3   = 7,
    };

}

class IndentingOStreambuf : public std::streambuf
{
    std::streambuf*     myDest;
    bool                myIsAtStartOfLine;
    std::string         myIndent;
    std::ostream*       myOwner;
protected:
    virtual int overflow( int ch );
public:
    explicit IndentingOStreambuf(std::streambuf* dest, int indent = 4 );
    explicit IndentingOStreambuf(std::ostream& dest, int indent = 4 );
    virtual  ~IndentingOStreambuf();
};

#endif

#include "PrintUtil.h"

namespace SDL
{
    // The following modified ostream will prefix "SDL::  " for every line
    oprefixstream cout("SDL::  ", std::cout);

    namespace CPU
    {
        // The following modified ostream will prefix "SDL::CPU::  " for every line
        oprefixstream cout("SDL::CPU::  ", std::cout);
    }

}

int SDL::prefixbuf::sync()
{
    return this->sbuf->pubsync();
}

int SDL::prefixbuf::overflow(int c)
{
    if (c != std::char_traits<char>::eof())
    {
        if (this->need_prefix
                && !this->prefix.empty()
                && this->prefix.size() != (unsigned) this->sbuf->sputn(&this->prefix[0], this->prefix.size()))
            return std::char_traits<char>::eof();
        this->need_prefix = c == '\n';
    }
    return this->sbuf->sputc(c);
}

SDL::prefixbuf::prefixbuf(std::string const& prefix1, std::streambuf* sbuf1) : prefix(prefix1) , sbuf(sbuf1) , need_prefix(true)
{
}

SDL::oprefixstream::oprefixstream(std::string const& prefix1, std::ostream& out1):
    prefixbuf(prefix1, out1.rdbuf()),
    std::ios(static_cast<std::streambuf*>(this)),
    std::ostream(static_cast<std::streambuf*>(this))
{
}

int IndentingOStreambuf::overflow( int ch )
{
    if ( myIsAtStartOfLine && ch != '\n' ) {
        myDest->sputn( myIndent.data(), myIndent.size() );
    }
    myIsAtStartOfLine = ch == '\n';
    return myDest->sputc( ch );
}
IndentingOStreambuf::IndentingOStreambuf( 
        std::streambuf* dest, int indent)
    : myDest( dest )
      , myIsAtStartOfLine( true )
    , myIndent( indent, ' ' )
      , myOwner( NULL )
{
}
IndentingOStreambuf::IndentingOStreambuf(
        std::ostream& dest, int indent)
    : myDest( dest.rdbuf() )
      , myIsAtStartOfLine( true )
    , myIndent( indent, ' ' )
      , myOwner( &dest )
{
    myOwner->rdbuf( this );
}
IndentingOStreambuf::~IndentingOStreambuf()
{
    if ( myOwner != NULL ) {
        myOwner->rdbuf( myDest );
    }
}

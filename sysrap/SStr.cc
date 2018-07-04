#include <cassert>
#include <cstring>
#include <iostream>
#include <algorithm>
#include "SStr.hh"

/**

In [15]: s = "hello"

In [18]: encode_ = lambda s:sum(map(lambda ic:ord(ic[1]) << 8*ic[0], enumerate(s[:8]) ))

In [19]: encode_(s)
Out[19]: 478560413032

In [40]: decode_ = lambda v:"".join(map( lambda c:str(unichr(c)), filter(None,map(lambda i:(v >> i*8) & 0xff, range(8))) ))

In [41]: decode_(478560413032)
Out[41]: 'hello'

Hmm presumably base64 code might do this at a higher level ?

**/


void SStr::FillFromULL( char* dest, unsigned long long value, char unprintable)
{
    dest[8] = '\0' ; 
    for( ULL w=0 ; w < 8 ; w++)
    {   
        ULL ullc = (value & (0xffull << w*8)) >> w*8 ;
        char c = static_cast<char>(ullc) ; 
        bool printable = c >= ' ' && c <= '~' ;
        dest[w] = printable ? c : unprintable ; 
    }       
}

const char* SStr::FromULL( unsigned long long value, char unprintable)
{
    assert( sizeof(ULL) == 8 );
    char* s = new char[8+1] ; 
    FillFromULL(s, value, unprintable) ; 
    return s ; 
}


unsigned long long SStr::ToULL( const char* s )
{
    assert( sizeof(ULL) == 8 );

    unsigned len = s ? strlen(s) : 0 ; 
    ULL mxw = len < 8 ? len : 8 ; 

    ULL v = 0ull ;   
    for(ULL w=0 ; w < mxw ; w++)
    {
        ULL c = s[w] ; 
        v |= ( c << 8ull*w ) ; 
    }
    return v ; 
}





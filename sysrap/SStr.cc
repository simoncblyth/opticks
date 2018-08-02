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


template<size_t SIZE>
const char* SStr::Format1( const char* fmt, const char* value )
{
    char buf[SIZE]; 
    size_t cx = snprintf( buf, SIZE, fmt, value );   
    assert( cx < SIZE && "snprintf truncation detected" );  
    return strdup(buf);
}

template<size_t SIZE>
const char* SStr::Format2( const char* fmt, const char* value1, const char* value2 )
{
    char buf[SIZE]; 
    size_t cx = snprintf( buf, SIZE, fmt, value1, value2 );   
    assert( cx < SIZE && "snprintf truncation detected" );  
    return strdup(buf);
}

template<size_t SIZE>
const char* SStr::Format3( const char* fmt, const char* value1, const char* value2, const char* value3 )
{
    char buf[SIZE]; 
    size_t cx = snprintf( buf, SIZE, fmt, value1, value2, value3 );   
    assert( cx < SIZE && "snprintf truncation detected" );  
    return strdup(buf);
}


template const char* SStr::Format1<256>( const char* , const char* );
template const char* SStr::Format2<256>( const char* , const char*, const char* );
template const char* SStr::Format3<256>( const char* , const char*, const char* , const char* );

template const char* SStr::Format1<16>( const char* , const char* );


bool SStr::Contains( const char* s_ , const char* q_ )
{
    std::string s(s_); 
    std::string q(q_); 
    return s.find(q) != std::string::npos ;
}






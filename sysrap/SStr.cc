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

bool SStr::EndsWith( const char* s, const char* q)
{
    int pos = strlen(s) - strlen(q) ;
    return pos > 0 && strncmp(s + pos, q, strlen(q)) == 0 ;
}

/**

SStr::HasPointerSuffix
-----------------------

Typically see 12 hexdigit pointers, as even though have 64 bits it is normal to only use 48 bits in an address space
    0x7ff46e500520 

But with G4 are seeing only 9 hexdigits ??


**/

bool SStr::HasPointerSuffix( const char* name, unsigned hexdigits )
{
   // eg Det0x110d9a820      why 9 hex digits vs
   //       0x7ff46e500520    
   //
    std::string s(name); 
    unsigned l = s.size() ; 
    if(l < hexdigits+2 ) return false ;
 
    for(unsigned i=0 ; i < hexdigits+2 ; i++)
    {
        char c = s[l-11+i] ; 
        bool ok = false ; 
        switch(i)
        {
            case 0: ok = c == '0' ; break ; 
            case 1: ok = c == 'x' ; break ; 
            default:  ok = ( c >= '0' && c <= '9' ) || ( c >= 'a' && c <= 'f' ) ; break ;   
        }
        if(!ok) return false ; 
    }
    return true  ; 
}


/**
SStr::GetPointerSuffixDigits
------------------------------

Check for hexdigits backwards, until reach first non-hexdigit the 'x'::

   World0x7fc10641cbb0  -> 12 
     Det0x110fa38b0     ->  9
     Hello              -> -1

**/

int SStr::GetPointerSuffixDigits( const char* name )
{
    if( name == NULL ) return -1 ; 
    int l = strlen(name) ; 
    int num = 0 ; 
    for(int i=0 ; i < l ; i++ )  
    {
         char c = *(name + l - 1 - i) ; 
         //std::cout << c << " " ; 
         bool hexdigit = ( c >= '0' && c <= '9' ) || ( c >= 'a' && c <= 'f' ) ; 
         if(!hexdigit) break ;  
         num += 1 ; 
    } 
    //std::cout << std::endl ; 
    if(l - num - 1 < 0 )  return -1 ; 
    if(l - num - 2 < 0 )  return -1 ; 

    char c1 = *(name + l - num - 1);
    char c2 = *(name + l - num - 2);

    return  c1 == 'x' && c2 == '0'  ?  num : -1 ; 
}


bool SStr::HasPointerSuffix( const char* name, unsigned min_hexdigits, unsigned max_hexdigits )
{    
    int num_hexdigits = GetPointerSuffixDigits( name ); 
    //std::cout << " num_hexdigits " << num_hexdigits << std::endl ; 
    return  num_hexdigits > -1 && num_hexdigits >= min_hexdigits && num_hexdigits <= max_hexdigits ; 
}








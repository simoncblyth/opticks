/*
 * Copyright (c) 2019 Opticks Team. All Rights Reserved.
 *
 * This file is part of Opticks
 * (see https://bitbucket.org/simoncblyth/opticks).
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 */

#include "SBit.hh"

#include <iostream>
#include <cstring>
#include <cassert>
#include <sstream>
#include <bitset>
#include <vector>
#include <algorithm>

#if defined(_MSC_VER)

#include <intrin.h>

int SBit::ffs(int i)
{
    // https://msdn.microsoft.com/en-us/library/wfd9z0bb.aspx
    unsigned long mask = i ;
    unsigned long index ;
    unsigned char masknonzero = _BitScanForward( &index, mask );
    return masknonzero ? index + 1 : 0 ;
}

#elif defined(__MINGW32__)

int SBit::ffs(int i)
{
   return __builtin_ffs(i);
}

#else

int SBit::ffs(int i)     
{
   return ::ffs(i);
}

long long SBit::ffsll(long long i)   
{
   return ::ffsll(i);
}


#endif



/**
ana/nibble.py:: 

    def count_nibbles(x):
        """
        NB: x can be an np.array

        https://stackoverflow.com/questions/38225571/count-number-of-zero-nibbles-in-an-unsigned-64-bit-integer
        """

        ## gather the zeroness (the OR of all 4 bits)
        x |= x >> 1               # or-with-1bit-right-shift-self is or-of-each-consequtive-2-bits 
        x |= x >> 2               # or-with-2bit-right-shift-self is or-of-each-consequtive-4-bits in the lowest slot 
        x &= 0x1111111111111111   # pluck the zeroth bit of each of the 16 nibbles

        x = (x + (x >> 4)) & 0xF0F0F0F0F0F0F0F    # sum occupied counts of consequtive nibbles, and pluck them 
        count = (x * 0x101010101010101) >> 56     #  add up byte totals into top byte,  and shift that down to pole 64-8 = 56 

        return count
**/

unsigned long long SBit::count_nibbles(unsigned long long x)
{
    x |= x >> 1 ;
    x |= x >> 2 ;
    x &= 0x1111111111111111ull ; 

    x = (x + (x >> 4)) & 0xF0F0F0F0F0F0F0Full ; 

    unsigned long long count = (x * 0x101010101010101ull) >> 56 ; 
    return count ; 
}


bool SBit::HasOneSetBit(int msk0)
{
    int idx0 = SBit::ffs(msk0) - 1 ;  // 0-based index of lsb least-significant-bit set  
    int msk1 = ( 0x1 << idx0 );  
    return msk0 == msk1 ; 
}



struct LengthOrder
{
    bool operator() (const std::string& s1, const std::string& s2)
    {
        size_t n1 = std::count(s1.begin(), s1.end(), ',');
        size_t n2 = std::count(s2.begin(), s2.end(), ',');

        // if( n1 > 0 ) n1 += 1 ;     // move favor for PosString
        // if( n2 > 0 ) n2 += 1 ; 

        size_t l1 = s1.length() - n1 ;    // favor PosString by not counting commas
        size_t l2 = s2.length() - n2 ; 

        return l1 < l2 ;   
    }
};


template <typename T>
std::string SBit::String(T v)  // static
{
    std::vector<std::string> str ; 
    str.push_back( BinString(v) );   // use the default annotation option  
    str.push_back( HexString(v) ); 
    str.push_back( DecString(v) ); 
    str.push_back( PosString(v) ); 

    LengthOrder length_order ; 
    std::sort( str.begin(), str.end(),  length_order );   
 
    return str[0] ; 
}


template <typename T>
std::string SBit::BinString(T v, bool anno)  // static
{
    std::bitset<sizeof(T)*8> bs(v) ; 
    bool express_flipped = 2*bs.count() > bs.size() ; // more than half the bits are set 
    if( express_flipped ) bs.flip(); 

    std::stringstream ss ; 
    ss 
        << ( express_flipped ? "~" : " " )
        << ( anno ? "0b" : "" )
        << bs.to_string() ;
        ;

    std::string s = ss.str(); 
    return s ; 
}

template <typename T>
std::string SBit::HexString(T v, bool anno)  // static
{
    std::bitset<64> bs(v) ;  assert( bs.size() == 64 ) ;
    bool express_flipped = 2*bs.count() > bs.size() ; // more than half the bits are set 
    if( express_flipped ) bs.flip(); 
    unsigned long long ull = bs.to_ullong() ;

    std::stringstream ss ; 
    ss 
        << ( express_flipped ? "~" : "" )
        << ( anno ? "0x" : "" )
        << std::hex << ull << std::dec 
        ;

    std::string s = ss.str(); 
    return s ; 
}

template <typename T>
std::string SBit::DecString(T v, bool anno)  // static
{
    std::bitset<64> bs(v) ;  assert( bs.size() == 64 ) ;
    bool express_flipped = 2*bs.count() > bs.size() ; // more than half the bits are set 
    if( express_flipped ) bs.flip(); 
    unsigned long long ull = bs.to_ullong() ;

    std::stringstream ss ; 
    ss 
       << ( express_flipped ? "~" : "" )
       << ( anno ? "0d" : "" )
       << std::dec << ull 
       ;

    std::string s = ss.str(); 
    return s ; 
}

template <typename T>
std::string SBit::PosString(T v, char delim, bool anno)  // static
{
    std::bitset<64> bs(v) ;  assert( bs.size() == 64 ) ;

    int hi = -1 ;  // find the highest set bit 
    for(int i=0 ; i < int(bs.size()) ; i++ ) if(bs[i]) hi=i ; 

    bool express_flipped = 2*bs.count() > bs.size() ; // more than half the bits are set 
    if( express_flipped ) bs.flip(); 

    std::stringstream ss ; 
    if(express_flipped) ss << "~" ; 
    if(anno) ss << "0p" ; 

    for(int i=0 ; i < int(bs.size()) ; i++ ) 
    {
        if(bs[i]) 
        {
            ss << i ; 
            if(bs.count() == 1 || i < hi ) ss << delim ;   
            // when 1 bit set always include the delim 
            // otherwise skip the delim for the last set bit 
        }
    }
    
    if(bs.count() == 0) ss << delim ;  // no-bits set : still output delim as blank is too difficult to recognize as a value  
    std::string s = ss.str() ; 
    return s ; 
}


/**
SBit::ParseAnnotation
----------------------

Parses unsigned long long (64 bit) integer strings such as : "0x001" "0b001" "0p001" "~0x001" "t0x001" 
and returns the integer string without the annotation prefix and sets the output arguments:

complement 
    bool is true when the string starts with ~ or t 
    (t is used as "~" can be problematic as it has special meaning to the shell)
anno
    char is set to one of 'x','b','d','p' OR '_' if there is no such prefix

    x : hex
    b : binary
    d : decimal 
    p : non-standard posString prefix which specifies positions of bits that are set  

All the below correspond to the same number::

     2
     0d2
     0x2
    ~0b1101   
     0b0010
     1,
     0p1,

    t0b1101

    t1,
    t0p1

**/

const char* SBit::ANNO = "xbdp" ; 

const char* SBit::ParseAnnotation(bool& complement, char& anno, const char* str_ )
{
    complement = strlen(str_) > 0 && ( str_[0] == '~' || str_[0] == 't' ) ;     // str_ starts with ~ or t 
    int postcomp =  complement ? 1 : 0 ;                                        // offset to skip the complement first character                     
    anno = strlen(str_+postcomp) > 2 && str_[postcomp] == '0' && strchr(ANNO, str_[postcomp+1]) != nullptr ?  str_[postcomp+1] : '_' ;  
    return str_ + postcomp + ( anno == '_' ? 0 : 2 ) ; 
}

/**
SBit::FromBinString
----------------------


**/

unsigned long long SBit::FromBinString(const char* str_ )
{
    bool complement ; 
    char anno ; 
    const char* str = ParseAnnotation(complement, anno, str_ );   
    assert( anno == 'b' || anno == '_' ); 

    assert( sizeof(unsigned long long)*8 == 64 ) ; 
    unsigned long long ull = std::bitset<64>(str).to_ullong() ;
    return complement ? ~ull : ull  ; 
}

unsigned long long SBit::FromHexString(const char* str_ )
{
    bool complement ; 
    char anno ; 
    const char* str = ParseAnnotation(complement, anno, str_ );   
    assert( anno == 'x' || anno == '_' ); 

    unsigned long long ull ;   
    std::stringstream ss;
    ss << std::hex << str  ;
    ss >> ull ;
    return complement ? ~ull : ull  ; 
}

unsigned long long SBit::FromDecString(const char* str_ )
{
    bool complement ; 
    char anno ; 
    const char* str = ParseAnnotation(complement, anno, str_ );   
    assert( anno == 'd' || anno == '_' ); 

    unsigned long long ull ;   
    std::stringstream ss;
    ss << std::dec << str ;
    ss >> ull ;
    return complement ? ~ull : ull  ; 
}

unsigned long long SBit::FromPosString(const char* str_, char delim)
{
    bool complement ; 
    char anno ; 
    const char* str = ParseAnnotation(complement, anno, str_ );   
    assert( anno == 'p' || anno == '_' ); 

    std::stringstream ss; 
    ss.str(str)  ;

    assert( sizeof(unsigned long long)*8 == 64 );

    std::bitset<64> bs ; // all bits start zero 
    std::string s;
    while (std::getline(ss, s, delim)) 
    {
        if(s.empty()) continue ;   // "," should give zero 
        int ipos = std::atoi(s.c_str()) ;
        bs.set(ipos, true); 
    }
    unsigned long long ull = bs.to_ullong() ;

#ifdef DEBUG    
    std::cout  
        << "SBit::FromPosString"
        << " str_[" << str_ << "]" 
        << " str[" << str << "]"
        << " anno " << anno  
        << " ull " << ull  
        << std::endl
        ; 
#endif

    return complement ? ~ull : ull  ; 
}


/**
SBit::FromString
---------------------

A PosString is a comma delimited list of ints that indicate that 
those bit positions are set. For example::


   "0,"       -> 1     # must include a delim to distinguish from normal int 
   "0,1"      -> 3
   "0,1,2"    -> 7
   "0,1,2,3"  -> 15
 
   "~0,"      -> all bits set other than first


**/

unsigned long long SBit::FromString(const char* str )
{
    bool complement ; 
    char anno ; 
    ParseAnnotation(complement, anno, str );   

    bool anno_expect = strchr(ANNO, anno ) != nullptr  || anno == '_' ; 
    if(!anno_expect) std::cout << "SBit::FromString unexpected anno " << anno << " from str " << str  << std::endl ; 
    assert(anno_expect);  

    unsigned long long ull = 0ull ; 
    if( strchr(str, ',') != nullptr )
    {
        ull = FromPosString(str, ',') ;   
    }
    else 
    {
        switch( anno )
        {
            case 'x': ull = FromHexString(str) ; break ; 
            case 'b': ull = FromBinString(str) ; break ; 
            case 'd': ull = FromDecString(str) ; break ; 
            case 'p': ull = FromPosString(str) ; break ; 
            case '_': ull = FromDecString(str) ; break ; 
            default : assert(0)                ; break ; 
        }
    }
    return ull ; 
}




template std::string SBit::BinString(char,bool); 
template std::string SBit::BinString(int,bool); 
template std::string SBit::BinString(long,bool); 
template std::string SBit::BinString(long long,bool); 

template std::string SBit::BinString(unsigned char,bool); 
template std::string SBit::BinString(unsigned int,bool); 
template std::string SBit::BinString(unsigned long,bool); 
template std::string SBit::BinString(unsigned long long,bool); 


template std::string SBit::HexString(char,bool); 
template std::string SBit::HexString(int,bool); 
template std::string SBit::HexString(long,bool); 
template std::string SBit::HexString(long long,bool); 

template std::string SBit::HexString(unsigned char,bool); 
template std::string SBit::HexString(unsigned int,bool); 
template std::string SBit::HexString(unsigned long,bool); 
template std::string SBit::HexString(unsigned long long,bool); 


template std::string SBit::DecString(char,bool); 
template std::string SBit::DecString(int,bool); 
template std::string SBit::DecString(long,bool); 
template std::string SBit::DecString(long long,bool); 

template std::string SBit::DecString(unsigned char,bool); 
template std::string SBit::DecString(unsigned int,bool); 
template std::string SBit::DecString(unsigned long,bool); 
template std::string SBit::DecString(unsigned long long,bool); 



template std::string SBit::PosString(char,char,bool); 
template std::string SBit::PosString(int,char,bool); 
template std::string SBit::PosString(long,char,bool); 
template std::string SBit::PosString(long long,char,bool); 

template std::string SBit::PosString(unsigned char,char,bool); 
template std::string SBit::PosString(unsigned int,char,bool); 
template std::string SBit::PosString(unsigned long,char,bool); 
template std::string SBit::PosString(unsigned long long,char,bool); 



template std::string SBit::String(char); 
template std::string SBit::String(int); 
template std::string SBit::String(long); 
template std::string SBit::String(long long); 

template std::string SBit::String(unsigned char); 
template std::string SBit::String(unsigned int); 
template std::string SBit::String(unsigned long); 
template std::string SBit::String(unsigned long long); 



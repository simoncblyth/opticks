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
#include <sstream>
#include <bitset>

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


template <typename T>
std::string SBit::BinString(T v)  // static
{
    std::string s = std::bitset<sizeof(T)*8>(v).to_string() ;
    return s ; 
}

template <typename T>
std::string SBit::HexString(T v)  // static
{
    std::stringstream ss ; 
    ss << std::hex << v << std::dec ; 
    std::string s = ss.str(); 
    return s ; 
}

unsigned long long SBit::FromBinString(const char* binstr )
{
    unsigned long long ull = std::bitset<sizeof(unsigned long long)*8>(binstr).to_ullong() ;
    return ull ; 
}
unsigned long long SBit::FromHexString(const char* hexstr )
{
    unsigned long long ull ;   
    std::stringstream ss;
    ss << std::hex << hexstr  ;
    ss >> ull ;
    return ull ; 
}
unsigned long long SBit::FromDecString(const char* decstr )
{
    unsigned long long ull ;   
    std::stringstream ss;
    ss << std::dec << decstr ;
    ss >> ull ;
    return ull ; 
}

unsigned long long SBit::FromString(const char* str )
{
    unsigned long long ull = 0ull ; 
    if(      strlen(str) > 2 && str[0] == '0' && str[1] == 'x' ) 
    {
        ull = FromHexString(str+2) ;
    }
    else if( strlen(str) > 2 && str[0] == '0' && str[1] == 'b' ) 
    {
        ull = FromBinString(str+2) ; 
    }
    else 
    {
        ull = FromDecString(str) ;
    }
    return ull ; 
}



template std::string SBit::BinString(char); 
template std::string SBit::BinString(int); 
template std::string SBit::BinString(long); 
template std::string SBit::BinString(long long); 

template std::string SBit::BinString(unsigned char); 
template std::string SBit::BinString(unsigned int); 
template std::string SBit::BinString(unsigned long); 
template std::string SBit::BinString(unsigned long long); 


template std::string SBit::HexString(char); 
template std::string SBit::HexString(int); 
template std::string SBit::HexString(long); 
template std::string SBit::HexString(long long); 

template std::string SBit::HexString(unsigned char); 
template std::string SBit::HexString(unsigned int); 
template std::string SBit::HexString(unsigned long); 
template std::string SBit::HexString(unsigned long long); 



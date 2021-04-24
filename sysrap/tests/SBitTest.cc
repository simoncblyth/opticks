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

// om-;TEST=SBitTest om-t

#include "OPTICKS_LOG.hh"
#include "SBit.hh"
#include <cassert>
#include <iostream>
#include <iomanip>

void test_ffs()
{
    LOG(info);
    for(int i=0 ; i < 64 ; i++ )
    {
        int msk = 0x1 << i ; 
        int chk = SBit::ffs(msk) - 1;   
        std::cout 
                  << " msk ( 0x1 << " << std::setw(2) << std::dec << i << ") = "  << std::hex << std::setw(16) << msk  
                  << " SBit::ffs(msk) - 1 =  " << std::dec << std::setw(4) << chk
                  << std::endl ;   

         if(i < 32 ) assert(  chk == i );
    } 
}

void test_ffsll()
{
    LOG(info);
    typedef long long LL ; 

    for(LL i=0 ; i < 64 ; i++ )
    {
        LL msk = 0x1ll << i ; 
        LL chk = SBit::ffsll(msk) - 1ll ;   
        std::cout 
                  << " msk ( 0x1 << " << std::setw(2) << std::dec << i << ") = "  << std::hex << std::setw(16) << msk  
                  << " SBit::ffsll(msk) - 1 =  " << std::dec << std::setw(4) << chk
                  << std::endl ;   

        assert(  chk == i );
    } 
}


void test_count_nibbles()
{
    LOG(info);
    typedef unsigned long long ULL ; 

    ULL msk = 0ull ;  
    for(ULL i=0 ; i < 64ull ; i++ )
    {
        msk |= 0x1ull << i ;  
        ULL nn = SBit::count_nibbles(msk); 
        std::cout 
             << " msk 0x " << std::hex << std::setw(16) << msk  
             << " nibbles " << std::dec << std::setw(4) << nn
             << std::endl 
             ;
    }
}


void test_HasOneSetBit()
{
    LOG(info);

    for(int i=0 ; i < 32 ; i++ )
    {
        unsigned msk0 = 0x1 << i ; 
        unsigned msk1 = 0x1 << (32 - i - 1) ; 
        unsigned msk01 = msk0 | msk1 ; 

        bool onebit0 = SBit::HasOneSetBit(msk0); 
        bool onebit1 = SBit::HasOneSetBit(msk1); 
        bool onebit01 = SBit::HasOneSetBit(msk01); 

        std::cout 
             << " i " << std::setw(3) << i         
             << " msk0 " << std::setw(10) << std::hex << msk0         
             << " msk1 " << std::setw(10) << std::hex << msk1         
             << " msk01 " << std::setw(10) << std::hex << msk01         
             << " onebit0 " << std::setw(2) << std::dec << onebit0
             << " onebit1 " << std::setw(2) << std::dec << onebit1
             << " onebit01 " << std::setw(2) << std::dec << onebit01
             << std::endl 
             ;

        assert( onebit0 == true );
        assert( onebit1 == true );
        assert( onebit01 == false );
    }
}


#define DUMP(l, s) \
    { \
    std::string binstr = SBit::BinString((s)) ; \
    std::string hexstr = SBit::HexString((s)) ; \
    unsigned long long ull = SBit::FromBinString(binstr.c_str()) ; \
    unsigned long long v((s)) ; \
    bool match = ull == v ;  \
    std::cout \
        << std::setw(5) << (l) \
        << std::setw(5) << sizeof((s))   \
        << std::setw(5) << sizeof((s))*8 \
        << std::setw(30) << (s) \
        << " : " \
        << std::setw(64) \
        << binstr \
        << std::setw(32) \
        << hexstr \
        << " : " \
        << std::setw(32) << ull \
        << ( match ? " Y" : " N" ) \
        << std::endl \
        ; \
    } 
  

void test_BinString()
{
    unsigned char uc = ~0 ; 
    unsigned int  ui = ~0 ; 
    unsigned long ul = ~0 ; 
    unsigned long long ull = ~0 ; 

    char c = ~0 ; 
    int  i = ~0 ; 
    long l = ~0 ; 
    long long ll = ~0 ; 

    DUMP("uc",uc); 
    DUMP("ui",ui); 
    DUMP("ul",ul); 
    DUMP("ull",ull); 

    DUMP("c",c); 
    DUMP("i",i); 
    DUMP("l",l); 
    DUMP("ll",ll); 
}


void test_FromBinString()
{
    {
        unsigned long long ull = ~0 ; 
        std::string binstr = SBit::BinString(ull); 
        std::cout 
             << "ull " << ull 
             << "0b " << binstr 
             << std::endl
             ; 
        unsigned long long ull2 = SBit::FromBinString(binstr.c_str()) ; 
        assert( ull == ull2 ); 
    }
    {
        int i = ~0 ; 
        std::string binstr = SBit::BinString(i); 
        unsigned long long ull = SBit::FromBinString(binstr.c_str()) ; 
        std::cout 
             << "i " << std::setw(10) << i 
             << "0b " << std::setw(64) << binstr 
             << "ull " << std::setw(32) << ull 
             << std::endl
             ;
    }
}

void test_FromString_0()
{
    unsigned long long ull_0 = SBit::FromString("0b11111111") ; 
    LOG(info) << " ull_0 " << ull_0 ; 
    assert( ull_0 == 255 ); 
    unsigned long long ull_1 = SBit::FromString("0xff") ; 
    assert( ull_1 == 255 ); 
    unsigned long long ull_2 = SBit::FromString("255") ; 
    assert( ull_2 == 255 ); 
    unsigned long long ull_3 = SBit::FromString("0,1,2,3,4,5,6,7") ; 
    LOG(info) << " ull_3 " << ull_3 ; 
    assert( ull_3 == 255 ); 
}



const char* EXAMPLES  = R"LITERAL(

0p
,
0
0d0
0x0
0x000000000
0b0
0b00000000

0,
0p0
1
0x1
0b1
0b000000001

1,
0p1
2
0x2
0b10
0b0010

0,1
0p0,1
3
0d3
0x3
0b11
0b0000011

2,
0p2
4
0d4
0x4
0b100
0b0000100


0,2
0p0,2
5
0d5
0x5
0b101
0b00000101


# Complemented zero : ie all 64 bits set  

~0


# PosString setting only the comma delimited bitpos ranging from 0 to 15

0,
1,
2,
3,
4,
5,
6,
7,
8,
9,
10,
11,
12,
13,
14,
15,

# PosString all bits set other than the comma delimited bitpos ranging from 0 to 15

~0,
~1,
~2,
~3,
~4,
~5,
~6,
~7,
~8,
~9,
~10,
~11,
~12,
~13,
~14,
~15,


# alt tilde avoiding shell escaping

t0,
t1,

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

~8,8


)LITERAL";


void test_FromString()
{
    LOG(info); 
    std::stringstream ss ; 
    ss.str(EXAMPLES); 
    std::string s;
    while (std::getline(ss, s, '\n')) 
    {
        if(s.empty())
        {
            std::cout << std::endl ; 
        }
        else if(s.c_str()[0] == '#')
        {
            std::cout << s << std::endl ; 
        }
        else
        {
            const char* str = s.c_str(); 
            unsigned long long ull = SBit::FromString(str) ; 

            std::cout
                 << std::setw(15) << str 
                 << " : "
                 << " "      << std::setw(20) << SBit::String(ull)
                 << " (0p) " << std::setw(10) << SBit::PosString(ull)
                 << " (0x) " << std::setw(16) << SBit::HexString(ull) 
                 << " (0d) " << std::setw(20) << SBit::DecString(ull)
                 << " (0b) " << std::setw(20) << SBit::BinString(ull)
                 << std::endl  
                 ;
        }

    }
}



void test_FromPosString()
{
    const char* posstr_0 = "0,1,2,3,4,5,6,7" ; 
    unsigned long long ull_0 = SBit::FromPosString(posstr_0) ; 
    LOG(info) << " ull_0 " << ull_0 ; 
    assert( ull_0 == 255 ); 

    std::string ps = SBit::PosString(ull_0) ; 
    LOG(info) << " PosString ps:" << ps << " posstr_0:" << posstr_0 ; 
    assert( strcmp( ps.c_str(), posstr_0  ) == 0 ); 

    const char* posstr_1 = "63," ; 
    unsigned long long ull_1 = SBit::FromPosString(posstr_1) ; 
    unsigned long long ull_1x = 1ull << 63 ;  
    assert( ull_1 == ull_1x ); 

    std::string ps1 = SBit::PosString(ull_1) ; 
    LOG(info) << " ps1 " << ps1 ; 
    assert( strcmp( ps1.c_str(), posstr_1  ) == 0 ); 

}





int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

/*
    test_ffs(); 
    test_ffsll(); 
    test_count_nibbles(); 
    test_HasOneSetBit(); 
    test_BinString(); 
    test_FromBinString(); 
*/

    test_FromString_0(); 
    test_FromString(); 
    //test_FromPosString(); 

    return 0 ; 
}

// om-;TEST=SBitTest om-t


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



int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    test_ffs(); 
    test_ffsll(); 
    test_count_nibbles(); 
    test_HasOneSetBit(); 

    return 0 ; 
}

// om-;TEST=SBitTest om-t


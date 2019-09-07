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

// TEST=BBitTest om-t

#include "BBit.hh"
#include <cassert>
#include <iostream>
#include <iomanip>

void test_ffs()
{
    for(int i=0 ; i < 64 ; i++ )
    {
        int msk = 0x1 << i ; 
        int chk = BBit::ffs(msk) - 1;   
        std::cout 
                  << " msk ( 0x1 << " << std::setw(2) << std::dec << i << ") = "  << std::hex << std::setw(16) << msk  
                  << " BBit::ffs(msk) - 1 =  " << std::dec << std::setw(4) << chk
                  << std::endl ;   

         if(i < 32 ) assert(  chk == i );
    } 
}

void test_ffsll()
{
    typedef long long LL ; 

    for(LL i=0 ; i < 64 ; i++ )
    {
        LL msk = 0x1ll << i ; 
        LL chk = BBit::ffsll(msk) - 1ll ;   
        std::cout 
                  << " msk ( 0x1 << " << std::setw(2) << std::dec << i << ") = "  << std::hex << std::setw(16) << msk  
                  << " BBit::ffsll(msk) - 1 =  " << std::dec << std::setw(4) << chk
                  << std::endl ;   

        assert(  chk == i );
    } 
}


void test_count_nibbles()
{

    typedef unsigned long long ULL ; 

    ULL msk = 0ull ;  
    for(ULL i=0 ; i < 64ull ; i++ )
    {
        msk |= 0x1ull << i ;  
        ULL nn = BBit::count_nibbles(msk); 
        std::cout 
             << " msk 0x " << std::hex << std::setw(16) << msk  
             << " nibbles " << std::dec << std::setw(4) << nn
             << std::endl 
             ;
    }
}



int main()
{
    test_ffs(); 
    test_ffsll(); 
    //test_count_nibbles(); 

    return 0 ; 
}

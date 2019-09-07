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

#include <iostream>
#include <iomanip>
#include "BBit.hh"


#ifdef OPTICKS_OPTIX

#define DUMP(name) \
{ \
    std::cout << std::setw(18) << #name  \
              << std::dec << std::setw(5) << (name)  \
              << std::hex << std::setw(5) << (name) ; \
    unsigned long long x = BBit::ffs(name) ; \
    for(unsigned int bounce=1 ; bounce < 18 ; bounce++) \
    {  \
        unsigned int shift = (bounce-1)*4 ; \
        unsigned long long y = x << shift ; \
        std::cout << std::hex << std::setw(17) << y ; \
    } \
    std::cout << std::endl ; \
} \


#include "cu/photon.h"

void dumpflags()
{
    printf("sizeof(unsigned long long)*8 = %lu \n", sizeof(unsigned long long)*8 );

    DUMP(CERENKOV);
    DUMP(SCINTILLATION);
    DUMP(MISS);
    DUMP(BULK_ABSORB);
    DUMP(BULK_REEMIT);
    DUMP(BULK_SCATTER);
    DUMP(SURFACE_DETECT);
    DUMP(SURFACE_ABSORB);
    DUMP(SURFACE_DREFLECT);
    DUMP(SURFACE_SREFLECT);
    DUMP(BOUNDARY_REFLECT);
    DUMP(BOUNDARY_TRANSMIT);
    DUMP(NAN_ABORT);


    for(unsigned int bounce=1 ; bounce < 20 ; bounce++ )
    {
       unsigned int shift = (bounce-1)*4 ; 
       unsigned long long x = 1 << shift ;
       printf(" %2u : %2u : %20llu     %llx \n",  bounce,shift,  x, x );
    }

    unsigned long long mask = 0 ; 
    for(unsigned int bounce=1 ; bounce < 20 ; bounce++ )
    {
       unsigned int shift = (bounce-1)*4 ; 
       unsigned long long x = BBit::ffs(BOUNDARY_TRANSMIT) ; 
       unsigned long long y = x << shift ;
       mask = mask | y ;
       mask = mask | y ;
       printf(" %2u : %2u : %20llu     %20llx  %20llx  \n",  bounce,shift,  y, y, mask );
    }
}

#else

void dumpflags()
{
    printf("klop\n");
}

#endif



int main()
{
    dumpflags();
}


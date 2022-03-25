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

// TEST=OpticksFlagsTest om-t

#include "BStr.hh"
#include "BMeta.hh"
#include "Opticks.hh"
#include "OpticksPhoton.hh"
#include "OpticksFlags.hh"
#include "Index.hpp"
#include "Types.hpp"

#include "OPTICKS_LOG.hh"


void test_ctor()
{
    OpticksFlags f ; 
    Index* i = f.getIndex();
    i->dump("test_ctor");
}

void test_cfTypes(Opticks* ok)
{
    Types* types = ok->getTypes();

    for(unsigned i=0 ; i < 32 ; i++)
    {
        unsigned bitpos = i ;
        unsigned flag = 0x1 << bitpos ; 
        std::string hs = types ? types->getHistoryString( flag ) : "notyps" ;

        const char* hs2 = OpticksPhoton::Flag(flag) ; 

        std::cout 
            << " i " << std::setw(3) << i 
            << " flag " << std::setw(10) << flag 
            << " hs " << std::setw(20) << hs
            << " hs2 " << std::setw(20) << hs2
            << std::endl 
            ;

    }
}

void test_getAbbrevMeta(Opticks* ok)
{
    OpticksFlags* f = ok->getFlags(); 
    BMeta* m = f->getAbbrevMeta(); 
    m->dump();
}


/**
test_Opticks_getDbgHitMask
----------------------------

::

    OpticksFlagsTest --dbghitmask TO,BT,SD,SC
    OpticksFlagsTest --dbghitmask SD,EC           # EC: EFFICIENCY_COLLECT 


**/

void test_Opticks_getDbgHitMask(Opticks* ok)
{
    unsigned msk = ok->getDbgHitMask();   // using OpticksFlags::AbbrevSequenceToMask

    LOG(info) 
        << " (dec) " << msk 
        << " (hex) " << std::setw(10) << std::hex << msk << std::dec 
        << " flagmask(abbrev) " << OpticksPhoton::FlagMask(msk, true) 
        << " flagmask " << OpticksPhoton::FlagMask(msk, false)
        ; 
}


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    Opticks ok(argc, argv);
    ok.configure();

    /*
    test_ctor();
    test_cfTypes(&ok);
    test_getAbbrevMeta(&ok);
    */


    test_Opticks_getDbgHitMask(&ok);


    return 0 ; 
}


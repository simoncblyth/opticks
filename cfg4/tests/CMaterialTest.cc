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

#include "Opticks.hh"
#include "OpticksHub.hh"
#include "OpticksMode.hh"

#include "CMaterialLib.hh"
#include "CMPT.hh"
#include "CVec.hh"

#include "CFG4_BODY.hh"
#include "OPTICKS_LOG.hh"

#include "PLOG.hh"


void test_CMPT(CMaterialLib* clib)
{
    const char* shortname = "Acrylic" ; 
    const CMPT* mpt = clib->getG4MPT(shortname);
    mpt->dump(shortname);

    const char* lkey = "GROUPVEL" ; 

    CVec* vec = mpt->getCVec(lkey);
    vec->dump(lkey);
}

void test_getG4MPT_and_dump(const CMaterialLib* clib, const char* shortname,  const char* lkey )
{
    const CMPT* mpt = clib->getG4MPT(shortname);
    if(mpt == NULL) 
    {
        LOG(error) << "canot find material " << shortname ; 
        return ;  
    } 

    mpt->dump(shortname);

    CVec* vec = mpt->getCVec(lkey);
    vec->dump(lkey);
}


void test_getG4MPT_and_dump(const CMaterialLib* clib)
{
    const char* shortname = "Pyrex" ;
    const char* lkey = "ABSLENGTH" ; 
    test_getG4MPT_and_dump(clib, shortname, lkey ); 
}
 

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    LOG(info) << argv[0] ; 

    Opticks ok(argc, argv);

    OpticksHub hub(&ok); 

    CMaterialLib* clib = new CMaterialLib(&hub); 

    LOG(info) << argv[0] << " convert " ; 

    clib->convert();

    LOG(info) << argv[0] << " dump " ; 

    //clib->dump();
   
    test_getG4MPT_and_dump(clib);    

    return 0 ; 
}



